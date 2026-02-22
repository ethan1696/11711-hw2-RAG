"""URL fetching with requests/selenium backends and retry/rate-limit logic."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .config import CrawlConfig
from .types import FetchBackend, FetchResult
from .url import host_from_url, normalize_url


@dataclass(frozen=True, slots=True)
class _AttemptConfig:
    attempts: int
    backoff_seconds: float


class Fetcher:
    """Fetch URLs using either `requests` or `selenium` backends.

    Concurrency model:
    - Requests backend is thread-friendly and can run in parallel workers.
    - Selenium backend is explicitly serialized with a lock because one shared
      browser instance is used, which is generally unstable under multithreaded use.
    """

    def __init__(self, config: CrawlConfig) -> None:
        self.config = config

        self._thread_local = threading.local()

        self._rate_lock = threading.Lock()
        self._next_allowed_time_by_host: dict[str, float] = {}

        self._robots_lock = threading.Lock()
        self._robots_cache: dict[str, RobotFileParser | None] = {}

        self._selenium_lock = threading.Lock()
        self._selenium_driver = None

        self._closed = False
        self._closed_lock = threading.Lock()

    def fetch(self, url: str, *, depth: int | None = None) -> FetchResult:
        """Fetch one URL with configured backend, retries, and policies."""

        normalized = normalize_url(url)
        if normalized is None:
            return FetchResult(
                requested_url=url,
                final_url=None,
                status_code=None,
                content_type=None,
                body=None,
                error="Invalid or unsupported URL",
            )

        if self._is_closed():
            return FetchResult(
                requested_url=normalized,
                final_url=None,
                status_code=None,
                content_type=None,
                body=None,
                error="Fetcher is closed",
            )

        if self.config.respect_robots_for(normalized):
            allowed_by_robots = self._is_allowed_by_robots(normalized)
            if not allowed_by_robots:
                return FetchResult(
                    requested_url=normalized,
                    final_url=None,
                    status_code=None,
                    content_type=None,
                    body=None,
                    error="Blocked by robots.txt",
                )

        backend = self.config.backend_for(normalized, depth=depth)
        attempt_cfg = _AttemptConfig(
            attempts=max(1, self.config.retries + 1),
            backoff_seconds=max(0.0, self.config.retry_backoff_seconds),
        )

        if backend == FetchBackend.SELENIUM:
            return self._fetch_with_retries(
                url=normalized,
                backend=backend,
                fetch_once=self._fetch_once_selenium,
                attempt_cfg=attempt_cfg,
            )

        return self._fetch_with_retries(
            url=normalized,
            backend=FetchBackend.REQUESTS,
            fetch_once=self._fetch_once_requests,
            attempt_cfg=attempt_cfg,
        )

    def close(self) -> None:
        """Close fetcher resources (notably selenium browser)."""

        with self._closed_lock:
            self._closed = True

        with self._selenium_lock:
            if self._selenium_driver is None:
                return
            try:
                self._selenium_driver.quit()
            except Exception:
                pass
            finally:
                self._selenium_driver = None

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _is_closed(self) -> bool:
        with self._closed_lock:
            return self._closed

    def _fetch_with_retries(
        self,
        *,
        url: str,
        backend: FetchBackend,
        fetch_once: Callable[[str], FetchResult],
        attempt_cfg: _AttemptConfig,
    ) -> FetchResult:
        last_result: FetchResult | None = None

        for attempt in range(1, attempt_cfg.attempts + 1):
            if self._is_closed():
                return FetchResult(
                    requested_url=url,
                    final_url=None,
                    status_code=None,
                    content_type=None,
                    body=None,
                    backend=backend,
                    error="Fetcher is closed",
                )

            result = fetch_once(url)
            last_result = result

            if self._is_terminal_result(result):
                return result

            if attempt < attempt_cfg.attempts and attempt_cfg.backoff_seconds > 0:
                # Linear backoff keeps behavior simple and predictable.
                time.sleep(attempt_cfg.backoff_seconds * attempt)

        if last_result is None:
            return FetchResult(
                requested_url=url,
                final_url=None,
                status_code=None,
                content_type=None,
                body=None,
                backend=backend,
                error="Unknown fetch failure",
            )

        return last_result

    @staticmethod
    def _is_terminal_result(result: FetchResult) -> bool:
        if result.error is not None:
            return False

        if result.status_code is None:
            return False

        if result.status_code in {408, 429} or result.status_code >= 500:
            return False

        return True

    def _fetch_once_requests(self, url: str) -> FetchResult:
        self._wait_for_rate_limit(url)
        started = time.perf_counter()

        session = self._thread_local_session()
        headers = self.config.headers_for(url)

        try:
            response = session.get(
                url,
                headers=headers,
                timeout=self.config.timeout_seconds,
                allow_redirects=True,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            body = response.content if response.content is not None else b""
            return FetchResult(
                requested_url=url,
                final_url=response.url or url,
                status_code=response.status_code,
                content_type=response.headers.get("Content-Type"),
                body=body,
                backend=FetchBackend.REQUESTS,
                elapsed_ms=elapsed_ms,
                error=None,
            )
        except requests.RequestException as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return FetchResult(
                requested_url=url,
                final_url=None,
                status_code=None,
                content_type=None,
                body=None,
                backend=FetchBackend.REQUESTS,
                elapsed_ms=elapsed_ms,
                error=f"{exc.__class__.__name__}: {exc}",
            )

    def _fetch_once_selenium(self, url: str) -> FetchResult:
        self._wait_for_rate_limit(url)
        started = time.perf_counter()

        with self._selenium_lock:
            try:
                driver = self._get_or_create_selenium_driver()
            except Exception as exc:
                return FetchResult(
                    requested_url=url,
                    final_url=None,
                    status_code=None,
                    content_type=None,
                    body=None,
                    backend=FetchBackend.SELENIUM,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                    error=f"Failed to initialize selenium driver: {exc}",
                )

            try:
                driver.set_page_load_timeout(max(1, int(self.config.timeout_seconds)))
                driver.get(url)

                domain_cfg = self.config.get_domain_config(url)
                if domain_cfg is not None:
                    wait_seconds = domain_cfg.selenium_wait_seconds
                    if wait_seconds is None:
                        wait_seconds = self.config.timeout_seconds

                    # Optional explicit selector gate: wait until the target node exists.
                    if domain_cfg.selenium_wait_selector and WebDriverWait and EC and By:
                        WebDriverWait(driver, wait_seconds).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, domain_cfg.selenium_wait_selector))
                        )

                    # Always allow optional post-load settling time for async rendering.
                    # This is especially useful for pages that hydrate content after
                    # the root selector (e.g., `body`) appears.
                    if domain_cfg.selenium_wait_seconds and domain_cfg.selenium_wait_seconds > 0:
                        time.sleep(domain_cfg.selenium_wait_seconds)

                final_url = driver.current_url or url
                body = (driver.page_source or "").encode("utf-8", errors="replace")
                elapsed_ms = int((time.perf_counter() - started) * 1000)

                return FetchResult(
                    requested_url=url,
                    final_url=final_url,
                    status_code=200,
                    content_type="text/html; charset=utf-8",
                    body=body,
                    backend=FetchBackend.SELENIUM,
                    elapsed_ms=elapsed_ms,
                    error=None,
                )
            except (TimeoutException, WebDriverException, Exception) as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                return FetchResult(
                    requested_url=url,
                    final_url=None,
                    status_code=None,
                    content_type=None,
                    body=None,
                    backend=FetchBackend.SELENIUM,
                    elapsed_ms=elapsed_ms,
                    error=f"{exc.__class__.__name__}: {exc}",
                )

    def _thread_local_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _wait_for_rate_limit(self, url: str) -> None:
        wait_seconds = max(0.0, self.config.rate_limit_for(url))
        if wait_seconds <= 0:
            return

        host = host_from_url(url)

        while True:
            with self._rate_lock:
                now = time.monotonic()
                next_allowed = self._next_allowed_time_by_host.get(host, 0.0)
                if now >= next_allowed:
                    self._next_allowed_time_by_host[host] = now + wait_seconds
                    return
                sleep_for = next_allowed - now

            if sleep_for > 0:
                time.sleep(sleep_for)

    def _is_allowed_by_robots(self, url: str) -> bool:
        parsed = urlsplit(url)
        host_key = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"

        with self._robots_lock:
            parser = self._robots_cache.get(host_key)

        if parser is None and host_key not in self._robots_cache:
            parser = self._load_robots_parser(host_key)
            with self._robots_lock:
                self._robots_cache[host_key] = parser

        # If robots cannot be loaded, fail open to avoid stalling crawling.
        if parser is None:
            return True

        user_agent = self.config.user_agent or "*"
        try:
            return parser.can_fetch(user_agent, url)
        except Exception:
            return True

    def _load_robots_parser(self, host_root: str) -> RobotFileParser | None:
        robots_url = f"{host_root}/robots.txt"

        try:
            response = requests.get(
                robots_url,
                headers={"User-Agent": self.config.user_agent},
                timeout=min(10.0, self.config.timeout_seconds),
            )
        except requests.RequestException:
            return None

        if response.status_code >= 400:
            return None

        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.text.splitlines())
        return parser

    def _get_or_create_selenium_driver(self):
        if self._selenium_driver is not None:
            return self._selenium_driver

        errors: list[str] = []

        # Try Chrome first.
        try:
            chrome_options = ChromeOptions()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"--user-agent={self.config.user_agent}")
            self._selenium_driver = webdriver.Chrome(options=chrome_options)
            return self._selenium_driver
        except Exception as exc:
            errors.append(f"Chrome: {exc}")

        # Fallback to Firefox.
        try:
            firefox_options = FirefoxOptions()
            firefox_options.add_argument("-headless")
            firefox_options.set_preference("general.useragent.override", self.config.user_agent)
            self._selenium_driver = webdriver.Firefox(options=firefox_options)
            return self._selenium_driver
        except Exception as exc:
            errors.append(f"Firefox: {exc}")

        raise RuntimeError("; ".join(errors) or "No usable Selenium driver found")


__all__ = ["Fetcher"]

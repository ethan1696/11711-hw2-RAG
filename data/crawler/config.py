"""Typed crawler configuration with JSON/YAML load/save helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml  # type: ignore

from .constants import (
    DEFAULT_CONCURRENCY,
    DEFAULT_FETCH_BACKEND,
    DEFAULT_FORCE_DOWNLOAD,
    DEFAULT_FORCE_PARSE,
    DEFAULT_HTTP_HEADERS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PAGES,
    DEFAULT_PER_DOMAIN_CAP,
    DEFAULT_PER_SEED_CAP,
    DEFAULT_RATE_LIMIT_SECONDS,
    DEFAULT_RESPECT_ROBOTS,
    DEFAULT_RETRIES,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    JSON_INDENT,
    SUPPORTED_CONFIG_SUFFIXES,
)
from .types import DomainConfig, FetchBackend, JSONDict, JSONValue


def normalize_domain(domain_or_url: str) -> str:
    """Normalize domain strings for matching and dedup."""

    raw = domain_or_url.strip().lower()
    if not raw:
        return ""

    if "://" in raw:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").strip().lower()
    else:
        host = raw

    if host.startswith("www."):
        host = host[4:]

    return host.strip(".")


def host_from_url(url: str) -> str:
    """Extract a normalized host from URL string."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host.strip(".")


def _as_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for '{key}': {value!r}") from exc


def _as_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid int for '{key}': {value!r}") from exc


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Invalid bool for '{key}': {value!r}")


def _to_backend(value: Any) -> FetchBackend:
    if isinstance(value, FetchBackend):
        return value
    if isinstance(value, str):
        return FetchBackend(value.strip().lower())
    raise ValueError(f"Invalid backend value: {value!r}")


def _coerce_domain_config(value: Any) -> DomainConfig:
    if isinstance(value, DomainConfig):
        return value

    if isinstance(value, str):
        domain = normalize_domain(value)
        if not domain:
            raise ValueError("Domain string cannot be empty")
        return DomainConfig(domain=domain)

    if isinstance(value, Mapping):
        domain = normalize_domain(str(value.get("domain", "")))
        if not domain:
            raise ValueError(f"Domain config missing valid 'domain': {value!r}")

        path_prefixes = [str(item) for item in value.get("allowed_path_prefixes", [])]
        selenium_max_depth = _as_int(
            value.get("selenium_max_depth"),
            "selenium_max_depth",
        )
        if selenium_max_depth is not None and selenium_max_depth < 0:
            raise ValueError("selenium_max_depth must be >= 0 when set")
        headers = {
            str(k): str(v)
            for k, v in dict(value.get("headers", {})).items()
        }
        metadata = dict(value.get("metadata", {}))

        return DomainConfig(
            domain=domain,
            backend=_to_backend(value.get("backend", DEFAULT_FETCH_BACKEND)),
            selenium_max_depth=selenium_max_depth,
            allowed_path_prefixes=path_prefixes,
            rate_limit_seconds=_as_float(value.get("rate_limit_seconds"), "rate_limit_seconds"),
            max_pages=_as_int(value.get("max_pages"), "max_pages"),
            selenium_wait_selector=(
                None
                if value.get("selenium_wait_selector") is None
                else str(value.get("selenium_wait_selector"))
            ),
            selenium_wait_seconds=_as_float(
                value.get("selenium_wait_seconds"),
                "selenium_wait_seconds",
            ),
            headers=headers,
            respect_robots=(
                None
                if value.get("respect_robots") is None
                else _as_bool(value.get("respect_robots"), "respect_robots")
            ),
            metadata=metadata,
        )

    raise TypeError(f"Unsupported domain config value: {type(value)!r}")


def _coerce_domain_list(values: list[Any]) -> list[DomainConfig]:
    dedup: dict[str, DomainConfig] = {}
    for item in values:
        domain_cfg = _coerce_domain_config(item)
        dedup[domain_cfg.domain] = domain_cfg
    return list(dedup.values())


@dataclass(slots=True)
class CrawlConfig:
    """Top-level crawler configuration used by pipeline/frontier/fetcher."""

    seeds: list[str]
    domains: list[DomainConfig]

    max_depth: int = DEFAULT_MAX_DEPTH
    max_pages: int = DEFAULT_MAX_PAGES
    per_domain_cap: int | None = DEFAULT_PER_DOMAIN_CAP
    per_seed_cap: int | None = DEFAULT_PER_SEED_CAP
    concurrency: int = DEFAULT_CONCURRENCY

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    retries: int = DEFAULT_RETRIES
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS
    rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS

    user_agent: str = DEFAULT_USER_AGENT
    default_headers: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HTTP_HEADERS))
    respect_robots: bool = DEFAULT_RESPECT_ROBOTS

    force_download: bool = DEFAULT_FORCE_DOWNLOAD
    force_parse: bool = DEFAULT_FORCE_PARSE

    metadata: dict[str, JSONValue] = field(default_factory=dict)

    _domain_index: dict[str, DomainConfig] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.seeds = [seed.strip() for seed in self.seeds if seed and seed.strip()]
        if not self.seeds:
            raise ValueError("CrawlConfig requires at least one seed URL")

        if self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if self.max_pages <= 0:
            raise ValueError("max_pages must be > 0")
        if self.per_domain_cap is not None and self.per_domain_cap <= 0:
            raise ValueError("per_domain_cap must be > 0 when set")
        if self.per_seed_cap is not None and self.per_seed_cap <= 0:
            raise ValueError("per_seed_cap must be > 0 when set")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.retries < 0:
            raise ValueError("retries must be >= 0")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")
        if self.rate_limit_seconds < 0:
            raise ValueError("rate_limit_seconds must be >= 0")

        if not self.domains:
            seed_domains = [host_from_url(seed) for seed in self.seeds]
            self.domains = [DomainConfig(domain=d) for d in seed_domains if d]

        self.domains = _coerce_domain_list(self.domains)
        if not self.domains:
            raise ValueError("No valid domains configured")

        self._domain_index = {domain.domain: domain for domain in self.domains}

    @property
    def allowed_domains(self) -> list[str]:
        """Return normalized allowed domains."""

        return [domain.domain for domain in self.domains]

    def get_domain_config(self, domain_or_url: str) -> DomainConfig | None:
        """Match a URL/host to the most specific configured domain policy."""

        host = domain_or_url
        if "://" in domain_or_url:
            host = host_from_url(domain_or_url)
        host = normalize_domain(host)
        if not host:
            return None

        # Longest suffix wins for nested domain rules.
        candidates: list[tuple[int, DomainConfig]] = []
        for domain, cfg in self._domain_index.items():
            if host == domain or host.endswith("." + domain):
                candidates.append((len(domain), cfg))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def is_url_allowed(self, url: str) -> bool:
        """Check URL against domain and optional path-prefix constraints."""

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False

        domain_cfg = self.get_domain_config(url)
        if domain_cfg is None:
            return False

        if not domain_cfg.allowed_path_prefixes:
            return True

        path = parsed.path or "/"
        return any(path.startswith(prefix) for prefix in domain_cfg.allowed_path_prefixes)

    def backend_for(self, url: str, *, depth: int | None = None) -> FetchBackend:
        """Return effective fetch backend for a URL/depth combination."""

        domain_cfg = self.get_domain_config(url)
        if domain_cfg is None:
            return DEFAULT_FETCH_BACKEND

        if (
            depth is not None
            and domain_cfg.backend == FetchBackend.SELENIUM
            and domain_cfg.selenium_max_depth is not None
            and depth > domain_cfg.selenium_max_depth
        ):
            return FetchBackend.REQUESTS

        return domain_cfg.backend

    def rate_limit_for(self, url: str) -> float:
        """Return effective per-domain/global rate limit for URL."""

        domain_cfg = self.get_domain_config(url)
        if domain_cfg and domain_cfg.rate_limit_seconds is not None:
            return domain_cfg.rate_limit_seconds
        return self.rate_limit_seconds

    def per_domain_cap_for(self, url: str) -> int | None:
        """Return effective per-domain crawl cap for URL."""

        domain_cfg = self.get_domain_config(url)
        if domain_cfg and domain_cfg.max_pages is not None:
            return domain_cfg.max_pages
        return self.per_domain_cap

    def respect_robots_for(self, url: str) -> bool:
        """Return robots policy with per-domain override support."""

        domain_cfg = self.get_domain_config(url)
        if domain_cfg and domain_cfg.respect_robots is not None:
            return domain_cfg.respect_robots
        return self.respect_robots

    def headers_for(self, url: str) -> dict[str, str]:
        """Return request headers merged from global and per-domain settings."""

        merged: dict[str, str] = dict(self.default_headers)
        domain_cfg = self.get_domain_config(url)
        if domain_cfg:
            merged.update(domain_cfg.headers)

        merged.setdefault("User-Agent", self.user_agent)
        return merged

    def to_dict(self) -> JSONDict:
        """Serialize config for manifests and reproducibility."""

        return {
            "seeds": self.seeds,
            "domains": [domain.to_json() for domain in self.domains],
            "max_depth": self.max_depth,
            "max_pages": self.max_pages,
            "per_domain_cap": self.per_domain_cap,
            "per_seed_cap": self.per_seed_cap,
            "concurrency": self.concurrency,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "rate_limit_seconds": self.rate_limit_seconds,
            "user_agent": self.user_agent,
            "default_headers": self.default_headers,
            "respect_robots": self.respect_robots,
            "force_download": self.force_download,
            "force_parse": self.force_parse,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrawlConfig":
        """Build config from a parsed dictionary."""

        if "seeds" not in payload:
            raise ValueError("Config missing required key: 'seeds'")

        raw_domains: list[Any]
        if "domains" in payload:
            raw_domains = list(payload.get("domains") or [])
        else:
            # Backward-compatible format.
            raw_domains = list(payload.get("allowed_domains") or [])

        if not raw_domains:
            raw_domains = [host_from_url(seed) for seed in list(payload["seeds"])]

        return cls(
            seeds=[str(seed) for seed in list(payload["seeds"])],
            domains=_coerce_domain_list(raw_domains),
            max_depth=int(payload.get("max_depth", DEFAULT_MAX_DEPTH)),
            max_pages=int(payload.get("max_pages", DEFAULT_MAX_PAGES)),
            per_domain_cap=_as_int(payload.get("per_domain_cap", DEFAULT_PER_DOMAIN_CAP), "per_domain_cap"),
            per_seed_cap=_as_int(payload.get("per_seed_cap", DEFAULT_PER_SEED_CAP), "per_seed_cap"),
            concurrency=int(payload.get("concurrency", DEFAULT_CONCURRENCY)),
            timeout_seconds=float(payload.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)),
            retries=int(payload.get("retries", DEFAULT_RETRIES)),
            retry_backoff_seconds=float(
                payload.get("retry_backoff_seconds", DEFAULT_RETRY_BACKOFF_SECONDS)
            ),
            rate_limit_seconds=float(
                payload.get("rate_limit_seconds", DEFAULT_RATE_LIMIT_SECONDS)
            ),
            user_agent=str(payload.get("user_agent", DEFAULT_USER_AGENT)),
            default_headers={
                str(k): str(v)
                for k, v in dict(payload.get("default_headers", DEFAULT_HTTP_HEADERS)).items()
            },
            respect_robots=_as_bool(
                payload.get("respect_robots", DEFAULT_RESPECT_ROBOTS),
                "respect_robots",
            ),
            force_download=_as_bool(
                payload.get("force_download", DEFAULT_FORCE_DOWNLOAD),
                "force_download",
            ),
            force_parse=_as_bool(
                payload.get("force_parse", DEFAULT_FORCE_PARSE),
                "force_parse",
            ),
            metadata=dict(payload.get("metadata", {})),
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config at {path} must be a mapping at top level")
    return data


def load_config(path: str | Path) -> CrawlConfig:
    """Load CrawlConfig from JSON/YAML path."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix not in SUPPORTED_CONFIG_SUFFIXES:
        raise ValueError(
            f"Unsupported config suffix '{suffix}'. Supported: {SUPPORTED_CONFIG_SUFFIXES}"
        )

    if suffix == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        payload = _load_yaml(config_path)

    if not isinstance(payload, dict):
        raise ValueError(f"Config at {config_path} must be a mapping")

    return CrawlConfig.from_dict(payload)


def save_config(config: CrawlConfig, path: str | Path) -> None:
    """Save CrawlConfig as JSON or YAML based on file extension."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    payload = config.to_dict()

    if suffix == ".json":
        out_path.write_text(
            json.dumps(payload, indent=JSON_INDENT, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    if suffix in {".yaml", ".yml"}:
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return

    raise ValueError(
        f"Unsupported config suffix '{suffix}'. Supported: {SUPPORTED_CONFIG_SUFFIXES}"
    )


__all__ = [
    "CrawlConfig",
    "host_from_url",
    "load_config",
    "normalize_domain",
    "save_config",
]

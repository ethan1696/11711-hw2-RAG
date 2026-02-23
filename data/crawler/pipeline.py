"""End-to-end crawl pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
import threading
from typing import Any

from .config import CrawlConfig
from .fetcher import Fetcher
from .frontier import EnqueueResult, Frontier
from .parsers import HTMLParser, PDFParser
from .stats import StatsCollector
from .storage import Storage
from .types import (
    ContentKind,
    CrawlStage,
    DocumentRecord,
    ErrorRecord,
    FetchResult,
    ParseResult,
)
from .url import extract_links_from_html, host_from_url, normalize_url


class PipelineMode(str, Enum):
    """Supported pipeline run modes."""

    CRAWL = "crawl"
    FETCH_ONLY = "fetch_only"
    PARSE_ONLY = "parse_only"


class Pipeline:
    """Orchestrates frontier, fetcher, parsers, storage, and stats."""

    def __init__(
        self,
        config: CrawlConfig,
        *,
        output_dir: str | Path,
        storage: Storage | None = None,
        fetcher: Fetcher | None = None,
        html_parser: HTMLParser | None = None,
        pdf_parser: PDFParser | None = None,
        stats: StatsCollector | None = None,
    ) -> None:
        self.config = config

        self.storage = storage or Storage(output_dir, load_existing=True)
        self.fetcher = fetcher or Fetcher(config)
        self.html_parser = html_parser or HTMLParser()
        self.pdf_parser = pdf_parser or PDFParser()
        self.stats = stats or StatsCollector()

        self._owns_fetcher = fetcher is None

        self._allowed_path_prefixes_by_domain = {
            domain.domain: list(domain.allowed_path_prefixes)
            for domain in self.config.domains
            if domain.allowed_path_prefixes
        }

    def run(self, mode: PipelineMode | str = PipelineMode.CRAWL) -> dict[str, Any]:
        """Run pipeline in one of the supported modes."""

        resolved_mode = self._resolve_mode(mode)

        self.storage.save_crawl_config(self.config)

        try:
            if resolved_mode == PipelineMode.PARSE_ONLY:
                self._run_parse_only()
            else:
                self._run_frontier_mode(resolved_mode)
        finally:
            if self._owns_fetcher:
                self.fetcher.close()

        self.stats.finish()
        summary = self.stats.to_json()
        self.storage.save_crawl_stats(summary)

        return {
            "mode": resolved_mode.value,
            "paths": self.storage.paths,
            "stats": summary,
        }

    @staticmethod
    def _resolve_mode(mode: PipelineMode | str) -> PipelineMode:
        if isinstance(mode, PipelineMode):
            return mode
        return PipelineMode(str(mode).strip().lower())

    def _run_frontier_mode(self, mode: PipelineMode) -> None:
        if mode not in {PipelineMode.CRAWL, PipelineMode.FETCH_ONLY}:
            raise ValueError(f"Unsupported frontier mode: {mode}")

        initial_seen = None if self.config.force_download else self.storage.visited_urls()
        frontier = Frontier(self.config, initial_seen_urls=initial_seen)

        seed_results = frontier.seed(self.config.seeds)
        self.stats.record_enqueue_many(seed_results)
        self._persist_accepted_urls(seed_results)

        workers = [
            threading.Thread(
                target=self._frontier_worker,
                args=(frontier, mode),
                name=f"crawler-worker-{idx}",
                daemon=True,
            )
            for idx in range(self.config.concurrency)
        ]

        for worker in workers:
            worker.start()

        frontier.join()
        frontier.close()

        for worker in workers:
            worker.join(timeout=5.0)

        self.stats.record_frontier_snapshot(frontier.snapshot())

    def _frontier_worker(self, frontier: Frontier, mode: PipelineMode) -> None:
        while True:
            item = frontier.pop(block=True, timeout=0.5)
            if item is None:
                if frontier.closed and frontier.empty():
                    return
                continue

            try:
                fetch_result, from_cache = self._get_content(
                    item.url,
                    depth=item.depth,
                    seed_url=item.seed_url,
                )
                if from_cache:
                    self.stats.increment("fetch_cache_hits")

                if not fetch_result.ok:
                    self._record_fetch_error(fetch_result)
                    continue

                if mode == PipelineMode.FETCH_ONLY:
                    self._maybe_expand_links_fetch_only(
                        frontier,
                        item.depth,
                        fetch_result,
                        seed_url=item.seed_url,
                    )
                    continue

                self._handle_parse_and_store(
                    frontier,
                    item.depth,
                    fetch_result,
                    seed_url=item.seed_url,
                )
            except Exception as exc:
                self._record_exception_error(CrawlStage.STORE, item.url, exc)
            finally:
                frontier.task_done()

    def _run_parse_only(self) -> None:
        urls = self._collect_parse_only_urls()
        self.stats.increment("parse_only_candidate_urls", len(urls))

        for url in urls:
            if not self.config.force_parse and self.storage.has_parsed(url):
                self.stats.increment("parse_skipped_cached")
                continue

            fetch_result = self._cached_fetch_result(url, depth=0, seed_url=url)
            if fetch_result is None:
                self.stats.increment("parse_only_missing_raw")
                continue

            self._handle_parse_and_store(
                None,
                depth=0,
                fetch_result=fetch_result,
                seed_url=url,
            )

    def _collect_parse_only_urls(self) -> list[str]:
        urls: set[str] = set()

        if self.storage.url_meta_path.exists():
            with self.storage.url_meta_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if payload.get("error"):
                        continue

                    status_code = payload.get("status_code")
                    if isinstance(status_code, int) and not (200 <= status_code < 300):
                        continue

                    candidate = payload.get("final_url") or payload.get("url")
                    if not isinstance(candidate, str):
                        continue

                    normalized = normalize_url(candidate)
                    if normalized and self.storage.has_raw(normalized):
                        urls.add(normalized)

        if not urls:
            for candidate in self.storage.visited_urls():
                normalized = normalize_url(candidate)
                if normalized and self.storage.has_raw(normalized):
                    urls.add(normalized)

        return sorted(urls)

    def _get_content(
        self,
        url: str,
        *,
        depth: int | None = None,
        seed_url: str | None = None,
    ) -> tuple[FetchResult, bool]:
        if not self.config.force_download:
            cached = self._cached_fetch_result(url, depth=depth, seed_url=seed_url)
            if cached is not None:
                return cached, True

        fetch_result = self.fetcher.fetch(url, depth=depth, seed_url=seed_url)
        self.stats.record_fetch(fetch_result)
        self.storage.save_url_meta(fetch_result)

        if fetch_result.ok and fetch_result.body is not None:
            self.storage.save_raw(
                url=fetch_result.final_url or fetch_result.requested_url,
                body=fetch_result.body,
                content_type=fetch_result.content_type,
                overwrite=self.config.force_download,
            )

        return fetch_result, False

    def _cached_fetch_result(
        self,
        url: str,
        *,
        depth: int | None = None,
        seed_url: str | None = None,
    ) -> FetchResult | None:
        raw_path = self.storage.find_raw_path(url)
        if raw_path is None:
            return None

        body = raw_path.read_bytes()
        suffix = raw_path.suffix.lower()

        if suffix == ".pdf":
            content_type = "application/pdf"
        elif suffix in {".html", ".htm"}:
            content_type = "text/html; charset=utf-8"
        elif suffix == ".txt":
            content_type = "text/plain; charset=utf-8"
        else:
            content_type = None

        return FetchResult(
            requested_url=url,
            final_url=url,
            status_code=200,
            content_type=content_type,
            body=body,
            backend=self.config.backend_for(url, depth=depth, seed_url=seed_url),
            error=None,
        )

    def _handle_parse_and_store(
        self,
        frontier: Frontier | None,
        depth: int,
        fetch_result: FetchResult,
        *,
        seed_url: str,
    ) -> None:
        parse_result = self._parse_fetch_result(fetch_result)
        self.stats.record_parse(parse_result)

        if not parse_result.ok:
            self._record_parse_error(fetch_result, parse_result)
            return

        target_url = fetch_result.final_url or fetch_result.requested_url
        if self.config.force_parse or not self.storage.has_parsed(target_url):
            doc = DocumentRecord.from_fetch_and_parse(
                doc_id=self._make_doc_id(fetch_result, parse_result),
                source_domain=host_from_url(target_url),
                fetch_result=fetch_result,
                parse_result=parse_result,
            )
            try:
                self.storage.save_doc(doc)
                self.stats.record_doc_saved()
            except Exception as exc:
                self._record_exception_error(CrawlStage.STORE, target_url, exc)
                return
        else:
            self.stats.increment("parse_skipped_cached")

        if frontier is None:
            return

        if depth >= self.config.max_depth_for_seed(seed_url):
            return

        if not parse_result.out_links:
            return

        enqueue_results = frontier.push_many(
            parse_result.out_links,
            depth=depth + 1,
            referrer=fetch_result.final_url or fetch_result.requested_url,
            seed_url=seed_url,
        )
        self.stats.record_enqueue_many(enqueue_results)
        self._persist_accepted_urls(enqueue_results)

    def _parse_fetch_result(self, fetch_result: FetchResult) -> ParseResult:
        target_url = fetch_result.final_url or fetch_result.requested_url
        body = fetch_result.body or b""
        content_kind = fetch_result.normalized_content_kind

        if content_kind == ContentKind.HTML:
            return self.html_parser.parse(
                url=fetch_result.requested_url,
                final_url=target_url,
                html=body,
                allowed_domains=self.config.allowed_domains,
                allowed_path_prefixes_by_domain=self._allowed_path_prefixes_by_domain,
            )

        if content_kind == ContentKind.PDF:
            return self.pdf_parser.parse(
                url=fetch_result.requested_url,
                final_url=target_url,
                pdf_bytes=body,
            )

        if content_kind == ContentKind.TEXT:
            decoded = body.decode("utf-8", errors="replace").strip()
            if not decoded:
                return ParseResult(
                    url=fetch_result.requested_url,
                    final_url=target_url,
                    title=None,
                    text="",
                    out_links=[],
                    content_kind=ContentKind.TEXT,
                    parser="text_parser",
                    metadata={"content_type": fetch_result.content_type},
                    error="Empty plain-text body",
                )
            return ParseResult(
                url=fetch_result.requested_url,
                final_url=target_url,
                title=None,
                text=decoded,
                out_links=[],
                content_kind=ContentKind.TEXT,
                parser="text_parser",
                metadata={"content_type": fetch_result.content_type},
                error=None,
            )

        return ParseResult(
            url=fetch_result.requested_url,
            final_url=target_url,
            title=None,
            text="",
            out_links=[],
            content_kind=content_kind,
            parser="unsupported_parser",
            metadata={"content_type": fetch_result.content_type},
            error=f"Unsupported content kind: {content_kind.value}",
        )

    def _maybe_expand_links_fetch_only(
        self,
        frontier: Frontier,
        depth: int,
        fetch_result: FetchResult,
        *,
        seed_url: str,
    ) -> None:
        if depth >= self.config.max_depth_for_seed(seed_url):
            return

        if fetch_result.normalized_content_kind != ContentKind.HTML:
            return

        if not fetch_result.body:
            return

        base_url = fetch_result.final_url or fetch_result.requested_url
        links = extract_links_from_html(
            fetch_result.body,
            base_url=base_url,
            allowed_domains=self.config.allowed_domains,
            allowed_path_prefixes_by_domain=self._allowed_path_prefixes_by_domain,
            include_nofollow=False,
            normalize=True,
        )
        if not links:
            return

        enqueue_results = frontier.push_many(
            links,
            depth=depth + 1,
            referrer=base_url,
            seed_url=seed_url,
        )
        self.stats.record_enqueue_many(enqueue_results)
        self._persist_accepted_urls(enqueue_results)

    def _persist_accepted_urls(self, results: list[EnqueueResult]) -> None:
        urls = [result.normalized_url for result in results if result.accepted and result.normalized_url]
        if not urls:
            return
        self.storage.mark_visited_many(urls)

    @staticmethod
    def _make_doc_id(fetch_result: FetchResult, parse_result: ParseResult) -> str:
        base_url = fetch_result.final_url or fetch_result.requested_url
        body_hash = fetch_result.body_sha256 or ""
        text_preview = parse_result.text[:512]
        payload = f"{base_url}\n{body_hash}\n{text_preview}".encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()

    def _record_fetch_error(self, fetch_result: FetchResult) -> None:
        message = fetch_result.error or (
            f"HTTP status {fetch_result.status_code}" if fetch_result.status_code is not None else "Unknown fetch failure"
        )
        error_type = message.split(":", maxsplit=1)[0].strip() if ":" in message else None

        error = ErrorRecord(
            stage=CrawlStage.FETCH,
            url=fetch_result.requested_url,
            message=message,
            error_type=error_type,
            status_code=fetch_result.status_code,
            metadata={
                "backend": fetch_result.backend.value,
                "final_url": fetch_result.final_url,
            },
        )
        self.storage.save_error(error)
        self.stats.record_error_saved()

    def _record_parse_error(self, fetch_result: FetchResult, parse_result: ParseResult) -> None:
        message = parse_result.error or "Unknown parse failure"
        error_type = message.split(":", maxsplit=1)[0].strip() if ":" in message else None

        error = ErrorRecord(
            stage=CrawlStage.PARSE,
            url=fetch_result.final_url or fetch_result.requested_url,
            message=message,
            error_type=error_type,
            status_code=fetch_result.status_code,
            metadata={
                "content_kind": parse_result.content_kind.value,
                "parser": parse_result.parser,
            },
        )
        self.storage.save_error(error)
        self.stats.record_error_saved()

    def _record_exception_error(self, stage: CrawlStage, url: str, exc: Exception) -> None:
        error = ErrorRecord.from_exception(stage=stage, url=url, exc=exc)
        self.storage.save_error(error)
        self.stats.record_error_saved()


__all__ = [
    "Pipeline",
    "PipelineMode",
]

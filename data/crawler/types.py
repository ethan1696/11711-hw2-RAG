"""Core type definitions for the crawler pipeline.

This module is intentionally dependency-light so other crawler modules can import
shared records without introducing cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping


class ContentKind(str, Enum):
    """Normalized content categories used across fetch/parse/storage."""

    HTML = "html"
    PDF = "pdf"
    TEXT = "text"
    BINARY = "binary"
    UNKNOWN = "unknown"


class CrawlStage(str, Enum):
    """Pipeline stage names for error reporting."""

    FRONTIER = "frontier"
    FETCH = "fetch"
    PARSE = "parse"
    STORE = "store"


class FetchBackend(str, Enum):
    """Backend used to fetch page content."""

    REQUESTS = "requests"
    SELENIUM = "selenium"


JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
JSONDict = dict[str, JSONValue]


def utc_now_iso() -> str:
    """Return an RFC3339-like UTC timestamp string for manifests/JSONL."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def infer_content_kind(content_type: str | None, url: str) -> ContentKind:
    """Infer coarse content kind from HTTP content type and URL."""

    normalized = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
    lower_url = url.lower()

    if "text/html" in normalized:
        return ContentKind.HTML
    if "application/pdf" in normalized or lower_url.endswith(".pdf"):
        return ContentKind.PDF
    if normalized.startswith("text/"):
        return ContentKind.TEXT
    if normalized:
        return ContentKind.BINARY
    return ContentKind.UNKNOWN


@dataclass(frozen=True, slots=True)
class FrontierItem:
    """A crawl candidate tracked by frontier/scheduler."""

    url: str
    depth: int
    seed_url: str
    referrer: str | None = None
    discovered_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True, slots=True)
class DomainConfig:
    """Per-domain crawl/fetch policy used by config/frontier/fetcher."""

    domain: str
    backend: FetchBackend = FetchBackend.REQUESTS
    selenium_max_depth: int | None = None
    allowed_path_prefixes: list[str] = field(default_factory=list)
    rate_limit_seconds: float | None = None
    max_pages: int | None = None
    selenium_wait_selector: str | None = None
    selenium_wait_seconds: float | None = None
    headers: dict[str, str] = field(default_factory=dict)
    respect_robots: bool | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_json(self) -> JSONDict:
        return {
            "domain": self.domain,
            "backend": self.backend.value,
            "selenium_max_depth": self.selenium_max_depth,
            "allowed_path_prefixes": self.allowed_path_prefixes,
            "rate_limit_seconds": self.rate_limit_seconds,
            "max_pages": self.max_pages,
            "selenium_wait_selector": self.selenium_wait_selector,
            "selenium_wait_seconds": self.selenium_wait_seconds,
            "headers": self.headers,
            "respect_robots": self.respect_robots,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class SeedConfig:
    """Per-seed override policy inherited by URLs discovered from that seed."""

    seed_url: str
    max_depth: int | None = None
    max_pages: int | None = None
    backend: FetchBackend | None = None
    selenium_max_depth: int | None = None
    rate_limit_seconds: float | None = None
    selenium_wait_selector: str | None = None
    selenium_wait_seconds: float | None = None
    headers: dict[str, str] = field(default_factory=dict)
    respect_robots: bool | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_json(self) -> JSONDict:
        return {
            "seed_url": self.seed_url,
            "max_depth": self.max_depth,
            "max_pages": self.max_pages,
            "backend": None if self.backend is None else self.backend.value,
            "selenium_max_depth": self.selenium_max_depth,
            "rate_limit_seconds": self.rate_limit_seconds,
            "selenium_wait_selector": self.selenium_wait_selector,
            "selenium_wait_seconds": self.selenium_wait_seconds,
            "headers": self.headers,
            "respect_robots": self.respect_robots,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FetchResult:
    """Result of attempting to download one URL."""

    requested_url: str
    final_url: str | None
    status_code: int | None
    content_type: str | None
    body: bytes | None
    backend: FetchBackend = FetchBackend.REQUESTS
    fetched_at: str = field(default_factory=utc_now_iso)
    elapsed_ms: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.status_code is not None
            and 200 <= self.status_code < 300
            and self.body is not None
        )

    @property
    def content_length(self) -> int | None:
        return None if self.body is None else len(self.body)

    @property
    def body_sha256(self) -> str | None:
        return None if self.body is None else sha256(self.body).hexdigest()

    @property
    def normalized_content_kind(self) -> ContentKind:
        return infer_content_kind(self.content_type, self.final_url or self.requested_url)

    def to_url_meta(self) -> "URLMetaRecord":
        return URLMetaRecord(
            url=self.requested_url,
            final_url=self.final_url,
            status_code=self.status_code,
            content_type=self.content_type,
            content_length=self.content_length,
            body_sha256=self.body_sha256,
            backend=self.backend,
            fetched_at=self.fetched_at,
            error=self.error,
        )


@dataclass(slots=True)
class ParseResult:
    """Result of parsing fetched bytes into text and links."""

    url: str
    final_url: str | None
    title: str | None
    text: str
    out_links: list[str] = field(default_factory=list)
    content_kind: ContentKind = ContentKind.UNKNOWN
    parser: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """One normalized parsed document row written to parsed/docs.jsonl."""

    doc_id: str
    url: str
    final_url: str | None
    title: str | None
    text: str
    content_type: str | None
    content_kind: ContentKind
    crawl_time: str
    source_domain: str
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    @classmethod
    def from_fetch_and_parse(
        cls,
        *,
        doc_id: str,
        source_domain: str,
        fetch_result: FetchResult,
        parse_result: ParseResult,
        crawl_time: str | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> "DocumentRecord":
        merged_metadata: dict[str, JSONValue] = dict(parse_result.metadata)
        if metadata:
            merged_metadata.update(dict(metadata))

        if fetch_result.content_length is not None:
            merged_metadata.setdefault("content_length", fetch_result.content_length)
        if fetch_result.body_sha256 is not None:
            merged_metadata.setdefault("raw_sha256", fetch_result.body_sha256)
        if parse_result.parser:
            merged_metadata.setdefault("parser", parse_result.parser)

        return cls(
            doc_id=doc_id,
            url=fetch_result.requested_url,
            final_url=fetch_result.final_url,
            title=parse_result.title,
            text=parse_result.text,
            content_type=fetch_result.content_type,
            content_kind=parse_result.content_kind,
            crawl_time=crawl_time or fetch_result.fetched_at,
            source_domain=source_domain,
            metadata=merged_metadata,
        )

    def to_json(self) -> JSONDict:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "final_url": self.final_url,
            "title": self.title,
            "text": self.text,
            "content_type": self.content_type,
            "content_kind": self.content_kind.value,
            "crawl_time": self.crawl_time,
            "source_domain": self.source_domain,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """One error row written to parsed/errors.jsonl."""

    stage: CrawlStage
    url: str
    message: str
    error_type: str | None = None
    doc_id: str | None = None
    referrer: str | None = None
    status_code: int | None = None
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        *,
        stage: CrawlStage,
        url: str,
        exc: Exception,
        **kwargs: Any,
    ) -> "ErrorRecord":
        return cls(
            stage=stage,
            url=url,
            message=str(exc),
            error_type=exc.__class__.__name__,
            **kwargs,
        )

    def to_json(self) -> JSONDict:
        return {
            "stage": self.stage.value,
            "url": self.url,
            "message": self.message,
            "error_type": self.error_type,
            "doc_id": self.doc_id,
            "referrer": self.referrer,
            "status_code": self.status_code,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class URLMetaRecord:
    """Per-URL fetch metadata row for manifests/url_meta.jsonl."""

    url: str
    final_url: str | None
    status_code: int | None
    content_type: str | None
    content_length: int | None
    body_sha256: str | None
    backend: FetchBackend
    fetched_at: str
    error: str | None = None

    def to_json(self) -> JSONDict:
        return {
            "url": self.url,
            "final_url": self.final_url,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "content_length": self.content_length,
            "body_sha256": self.body_sha256,
            "backend": self.backend.value,
            "fetched_at": self.fetched_at,
            "error": self.error,
        }


@dataclass(slots=True)
class CrawlStats:
    """Simple mutable counters used for crawl summary reporting."""

    frontier_enqueued: int = 0
    frontier_skipped_visited: int = 0
    frontier_skipped_depth: int = 0
    frontier_skipped_budget: int = 0

    fetched_ok: int = 0
    fetched_error: int = 0
    parsed_ok: int = 0
    parsed_error: int = 0
    stored_docs: int = 0

    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None

    def finish(self) -> None:
        self.finished_at = utc_now_iso()

    def merge(self, other: "CrawlStats") -> None:
        self.frontier_enqueued += other.frontier_enqueued
        self.frontier_skipped_visited += other.frontier_skipped_visited
        self.frontier_skipped_depth += other.frontier_skipped_depth
        self.frontier_skipped_budget += other.frontier_skipped_budget
        self.fetched_ok += other.fetched_ok
        self.fetched_error += other.fetched_error
        self.parsed_ok += other.parsed_ok
        self.parsed_error += other.parsed_error
        self.stored_docs += other.stored_docs

    def to_json(self) -> JSONDict:
        return {
            "frontier_enqueued": self.frontier_enqueued,
            "frontier_skipped_visited": self.frontier_skipped_visited,
            "frontier_skipped_depth": self.frontier_skipped_depth,
            "frontier_skipped_budget": self.frontier_skipped_budget,
            "fetched_ok": self.fetched_ok,
            "fetched_error": self.fetched_error,
            "parsed_ok": self.parsed_ok,
            "parsed_error": self.parsed_error,
            "stored_docs": self.stored_docs,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


__all__ = [
    "ContentKind",
    "CrawlStage",
    "CrawlStats",
    "DomainConfig",
    "DocumentRecord",
    "ErrorRecord",
    "FetchBackend",
    "FetchResult",
    "FrontierItem",
    "JSONDict",
    "JSONPrimitive",
    "JSONValue",
    "ParseResult",
    "SeedConfig",
    "URLMetaRecord",
    "infer_content_kind",
    "utc_now_iso",
]

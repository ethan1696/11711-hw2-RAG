"""Filesystem-backed storage for crawler artifacts and manifests.

Storage owns the on-disk layout. Other modules should use this API instead of
building paths manually.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .config import CrawlConfig
from .types import (
    ContentKind,
    CrawlStats,
    DocumentRecord,
    ErrorRecord,
    FetchResult,
    JSONDict,
    URLMetaRecord,
    infer_content_kind,
)


RAW_SUBDIR_BY_KIND: dict[ContentKind, str] = {
    ContentKind.HTML: "html",
    ContentKind.PDF: "pdf",
    ContentKind.TEXT: "text",
    ContentKind.BINARY: "binary",
    ContentKind.UNKNOWN: "unknown",
}

RAW_EXTENSION_BY_KIND: dict[ContentKind, str] = {
    ContentKind.HTML: ".html",
    ContentKind.PDF: ".pdf",
    ContentKind.TEXT: ".txt",
    ContentKind.BINARY: ".bin",
    ContentKind.UNKNOWN: ".bin",
}


class Storage:
    """Persist crawl outputs under a single `output_dir` root."""

    def __init__(self, output_dir: str | Path, *, load_existing: bool = True) -> None:
        self.output_dir = Path(output_dir)

        self.raw_dir = self.output_dir / "raw"
        self.parsed_dir = self.output_dir / "parsed"
        self.manifests_dir = self.output_dir / "manifests"
        self.logs_dir = self.output_dir / "logs"

        self.docs_path = self.parsed_dir / "docs.jsonl"
        self.errors_path = self.parsed_dir / "errors.jsonl"
        self.url_meta_path = self.manifests_dir / "url_meta.jsonl"
        self.visited_urls_path = self.manifests_dir / "visited_urls.txt"
        self.crawl_config_path = self.manifests_dir / "crawl_config.json"
        self.crawl_stats_path = self.manifests_dir / "crawl_stats.json"

        self._jsonl_lock = threading.Lock()
        self._raw_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._visited_urls: set[str] = set()
        self._parsed_urls: set[str] = set()

        self._ensure_layout()
        if load_existing:
            self._load_state()

    @property
    def paths(self) -> JSONDict:
        """Return important output paths for logging/CLI status messages."""

        return {
            "output_dir": str(self.output_dir),
            "raw_dir": str(self.raw_dir),
            "parsed_docs": str(self.docs_path),
            "parsed_errors": str(self.errors_path),
            "visited_urls": str(self.visited_urls_path),
            "url_meta": str(self.url_meta_path),
            "crawl_config": str(self.crawl_config_path),
            "crawl_stats": str(self.crawl_stats_path),
            "log_dir": str(self.logs_dir),
        }

    def _ensure_layout(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        for kind_subdir in RAW_SUBDIR_BY_KIND.values():
            (self.raw_dir / kind_subdir).mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> None:
        self._load_visited_urls()
        self._load_parsed_urls()

    def _load_visited_urls(self) -> None:
        if not self.visited_urls_path.exists():
            return
        with self.visited_urls_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                url = line.strip()
                if url:
                    self._visited_urls.add(url)

    def _load_parsed_urls(self) -> None:
        if not self.docs_path.exists():
            return

        with self.docs_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                url = payload.get("url")
                final_url = payload.get("final_url")
                if isinstance(url, str) and url:
                    self._parsed_urls.add(url)
                if isinstance(final_url, str) and final_url:
                    self._parsed_urls.add(final_url)

    @staticmethod
    def _url_host_key(url: str) -> str:
        host = (urlparse(url).hostname or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return "unknown"
        return "".join(char if (char.isalnum() or char in {".", "-", "_"}) else "_" for char in host)

    @staticmethod
    def _url_digest(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def raw_path_for(
        self,
        url: str,
        *,
        content_type: str | None = None,
        content_kind: ContentKind | None = None,
    ) -> Path:
        """Build deterministic raw-file path for a URL + content kind."""

        resolved_kind = content_kind or infer_content_kind(content_type, url)
        subdir = RAW_SUBDIR_BY_KIND[resolved_kind]
        extension = RAW_EXTENSION_BY_KIND[resolved_kind]
        host = self._url_host_key(url)
        digest = self._url_digest(url)
        return self.raw_dir / subdir / host / f"{digest}{extension}"

    def find_raw_path(self, url: str) -> Path | None:
        """Find existing raw path for URL across known content-kind locations."""

        for content_kind in ContentKind:
            candidate = self.raw_path_for(url, content_kind=content_kind)
            if candidate.exists():
                return candidate
        return None

    def has_raw(self, url: str) -> bool:
        """Return True if raw bytes for URL already exist on disk."""

        return self.find_raw_path(url) is not None

    def read_raw(self, url: str) -> bytes | None:
        """Return raw bytes for URL if present."""

        path = self.find_raw_path(url)
        if path is None:
            return None
        return path.read_bytes()

    def save_raw(
        self,
        *,
        url: str,
        body: bytes,
        content_type: str | None = None,
        content_kind: ContentKind | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Persist raw response bytes atomically."""

        if not isinstance(body, (bytes, bytearray)):
            raise TypeError("save_raw expects `body` as bytes")

        path = self.raw_path_for(
            url,
            content_type=content_type,
            content_kind=content_kind,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._raw_lock:
            if path.exists() and not overwrite:
                return path
            self._atomic_write_bytes(path, bytes(body))
        return path

    def has_parsed(self, url: str) -> bool:
        """Return True if URL has a persisted parsed document row."""

        with self._state_lock:
            return url in self._parsed_urls

    def save_doc(self, record: DocumentRecord) -> None:
        """Append parsed document record to `parsed/docs.jsonl`."""

        payload = record.to_json()
        self._append_jsonl(self.docs_path, payload)
        with self._state_lock:
            self._parsed_urls.add(record.url)
            if record.final_url:
                self._parsed_urls.add(record.final_url)

    def save_error(self, record: ErrorRecord) -> None:
        """Append error record to `parsed/errors.jsonl`."""

        self._append_jsonl(self.errors_path, record.to_json())

    def save_url_meta(self, record: URLMetaRecord | FetchResult) -> None:
        """Append URL metadata row to `manifests/url_meta.jsonl`."""

        if isinstance(record, FetchResult):
            payload = record.to_url_meta().to_json()
        else:
            payload = record.to_json()
        self._append_jsonl(self.url_meta_path, payload)

    def mark_visited(self, url: str) -> bool:
        """Persist URL in visited manifest.

        Returns True when newly added, False when it already existed.
        """

        with self._state_lock:
            if url in self._visited_urls:
                return False
            self._visited_urls.add(url)

        with self._jsonl_lock:
            with self.visited_urls_path.open("a", encoding="utf-8") as handle:
                handle.write(url + "\n")
        return True

    def mark_visited_many(self, urls: list[str] | set[str] | tuple[str, ...]) -> int:
        """Persist many visited URLs and return how many were newly added."""

        new_urls: list[str] = []
        with self._state_lock:
            for url in urls:
                if url in self._visited_urls:
                    continue
                self._visited_urls.add(url)
                new_urls.append(url)

        if not new_urls:
            return 0

        with self._jsonl_lock:
            with self.visited_urls_path.open("a", encoding="utf-8") as handle:
                for url in new_urls:
                    handle.write(url + "\n")
        return len(new_urls)

    def visited_urls(self) -> set[str]:
        """Return snapshot copy of known visited URLs."""

        with self._state_lock:
            return set(self._visited_urls)

    def save_crawl_config(self, config: CrawlConfig | Mapping[str, Any]) -> None:
        """Write crawl config manifest atomically as JSON."""

        payload: Mapping[str, Any]
        if isinstance(config, CrawlConfig):
            payload = config.to_dict()
        else:
            payload = config
        self._atomic_write_json(self.crawl_config_path, dict(payload))

    def save_crawl_stats(self, stats: CrawlStats | Mapping[str, Any]) -> None:
        """Write crawl stats manifest atomically as JSON."""

        payload: Mapping[str, Any]
        if isinstance(stats, CrawlStats):
            payload = stats.to_json()
        else:
            payload = stats
        self._atomic_write_json(self.crawl_stats_path, dict(payload))

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._jsonl_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise


__all__ = ["Storage"]

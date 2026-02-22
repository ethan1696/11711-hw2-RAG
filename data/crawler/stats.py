"""Thread-safe crawl statistics aggregation utilities."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import threading
from typing import Any, Mapping

from .frontier import EnqueueResult, EnqueueStatus
from .types import CrawlStats, FetchResult, ParseResult


class StatsCollector:
    """Collect and summarize crawler runtime statistics.

    The collector is thread-safe and intended for use across concurrent
    fetch/parse workers.
    """

    def __init__(self, base: CrawlStats | None = None) -> None:
        self._lock = threading.Lock()
        self._core = base or CrawlStats()

        self._frontier_extra: dict[str, int] = defaultdict(int)
        self._frontier_snapshot: dict[str, int | bool] = {}

        self._fetch_backend_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"ok": 0, "error": 0}
        )
        self._fetch_status_code_counts: dict[str, int] = defaultdict(int)
        self._fetch_error_type_counts: dict[str, int] = defaultdict(int)
        self._fetch_content_kind_counts: dict[str, int] = defaultdict(int)
        self._fetch_elapsed_ms_total = 0
        self._fetch_elapsed_samples = 0
        self._fetch_bytes_total = 0

        self._parse_kind_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"ok": 0, "error": 0}
        )
        self._parse_parser_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"ok": 0, "error": 0}
        )
        self._parse_error_type_counts: dict[str, int] = defaultdict(int)
        self._parse_text_chars_total = 0
        self._parse_text_words_total = 0
        self._parse_links_total = 0

        self._storage_error_rows = 0
        self._custom_counters: dict[str, int] = defaultdict(int)

    def record_enqueue(self, result_or_status: EnqueueResult | EnqueueStatus) -> None:
        """Record one frontier enqueue outcome."""

        if isinstance(result_or_status, EnqueueResult):
            status = result_or_status.status
        else:
            status = result_or_status

        with self._lock:
            if status == EnqueueStatus.ENQUEUED:
                self._core.frontier_enqueued += 1
                return
            if status == EnqueueStatus.SKIPPED_SEEN:
                self._core.frontier_skipped_visited += 1
                return
            if status == EnqueueStatus.SKIPPED_DEPTH:
                self._core.frontier_skipped_depth += 1
                return
            if status in {
                EnqueueStatus.SKIPPED_GLOBAL_BUDGET,
                EnqueueStatus.SKIPPED_DOMAIN_BUDGET,
            }:
                self._core.frontier_skipped_budget += 1
                return

            # Extra skip reasons not present in CrawlStats core fields.
            self._frontier_extra[status.value] += 1

    def record_enqueue_many(
        self, results: list[EnqueueResult] | tuple[EnqueueResult, ...]
    ) -> None:
        """Record many enqueue outcomes."""

        for result in results:
            self.record_enqueue(result)

    def record_frontier_snapshot(self, snapshot: Mapping[str, int | bool]) -> None:
        """Attach latest frontier snapshot for diagnostics."""

        with self._lock:
            self._frontier_snapshot = dict(snapshot)

    def record_fetch(self, result: FetchResult) -> None:
        """Record one fetch result."""

        with self._lock:
            state = "ok" if result.ok else "error"
            self._fetch_backend_counts[result.backend.value][state] += 1

            if result.ok:
                self._core.fetched_ok += 1
            else:
                self._core.fetched_error += 1

            if result.status_code is not None:
                self._fetch_status_code_counts[str(result.status_code)] += 1

            if result.error:
                err_type = result.error.split(":", maxsplit=1)[0].strip() or "Unknown"
                self._fetch_error_type_counts[err_type] += 1

            kind = result.normalized_content_kind.value
            self._fetch_content_kind_counts[kind] += 1

            if result.elapsed_ms is not None:
                self._fetch_elapsed_ms_total += int(result.elapsed_ms)
                self._fetch_elapsed_samples += 1

            if result.content_length is not None:
                self._fetch_bytes_total += int(result.content_length)

    def record_parse(self, result: ParseResult) -> None:
        """Record one parse result."""

        parse_state = "ok" if result.ok else "error"
        parser_name = result.parser or "unknown"
        kind = result.content_kind.value

        text = result.text or ""
        words = len(text.split())
        links = len(result.out_links)

        with self._lock:
            self._parse_kind_counts[kind][parse_state] += 1
            self._parse_parser_counts[parser_name][parse_state] += 1

            if result.ok:
                self._core.parsed_ok += 1
            else:
                self._core.parsed_error += 1

            if result.error:
                err_type = result.error.split(":", maxsplit=1)[0].strip() or "Unknown"
                self._parse_error_type_counts[err_type] += 1

            self._parse_text_chars_total += len(text)
            self._parse_text_words_total += words
            self._parse_links_total += links

    def record_doc_saved(self, count: int = 1) -> None:
        """Record successful persisted parsed document rows."""

        if count <= 0:
            return
        with self._lock:
            self._core.stored_docs += count

    def record_error_saved(self, count: int = 1) -> None:
        """Record persisted error rows in storage."""

        if count <= 0:
            return
        with self._lock:
            self._storage_error_rows += count

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a custom counter for ad-hoc instrumentation."""

        if not name or value == 0:
            return
        with self._lock:
            self._custom_counters[name] += value

    def finish(self) -> None:
        """Mark crawl as finished."""

        with self._lock:
            self._core.finish()

    def merge(self, other: "StatsCollector") -> None:
        """Merge another collector into this one."""

        payload = other.to_json()
        with self._lock:
            incoming_core = CrawlStats(
                frontier_enqueued=int(payload.get("frontier_enqueued", 0)),
                frontier_skipped_visited=int(payload.get("frontier_skipped_visited", 0)),
                frontier_skipped_depth=int(payload.get("frontier_skipped_depth", 0)),
                frontier_skipped_budget=int(payload.get("frontier_skipped_budget", 0)),
                fetched_ok=int(payload.get("fetched_ok", 0)),
                fetched_error=int(payload.get("fetched_error", 0)),
                parsed_ok=int(payload.get("parsed_ok", 0)),
                parsed_error=int(payload.get("parsed_error", 0)),
                stored_docs=int(payload.get("stored_docs", 0)),
                started_at=self._core.started_at,
                finished_at=self._core.finished_at,
            )
            self._core.merge(incoming_core)

            self._merge_count_dict(
                self._frontier_extra,
                payload.get("frontier", {}).get("extra_status_counts", {}),
            )
            if payload.get("frontier", {}).get("snapshot"):
                self._frontier_snapshot = dict(payload["frontier"]["snapshot"])

            self._merge_nested_count_dict(
                self._fetch_backend_counts,
                payload.get("fetch", {}).get("by_backend", {}),
            )
            self._merge_count_dict(
                self._fetch_status_code_counts,
                payload.get("fetch", {}).get("status_code_counts", {}),
            )
            self._merge_count_dict(
                self._fetch_error_type_counts,
                payload.get("fetch", {}).get("error_type_counts", {}),
            )
            self._merge_count_dict(
                self._fetch_content_kind_counts,
                payload.get("fetch", {}).get("content_kind_counts", {}),
            )
            self._fetch_elapsed_ms_total += int(
                payload.get("fetch", {}).get("elapsed_ms_total", 0)
            )
            self._fetch_elapsed_samples += int(
                payload.get("fetch", {}).get("elapsed_ms_samples", 0)
            )
            self._fetch_bytes_total += int(payload.get("fetch", {}).get("bytes_total", 0))

            self._merge_nested_count_dict(
                self._parse_kind_counts,
                payload.get("parse", {}).get("by_content_kind", {}),
            )
            self._merge_nested_count_dict(
                self._parse_parser_counts,
                payload.get("parse", {}).get("by_parser", {}),
            )
            self._merge_count_dict(
                self._parse_error_type_counts,
                payload.get("parse", {}).get("error_type_counts", {}),
            )
            self._parse_text_chars_total += int(payload.get("parse", {}).get("text_chars_total", 0))
            self._parse_text_words_total += int(payload.get("parse", {}).get("text_words_total", 0))
            self._parse_links_total += int(payload.get("parse", {}).get("links_total", 0))

            self._storage_error_rows += int(payload.get("storage", {}).get("error_rows", 0))
            self._merge_count_dict(self._custom_counters, payload.get("custom_counters", {}))

    def core(self) -> CrawlStats:
        """Return a copy of the core `CrawlStats` record."""

        with self._lock:
            return CrawlStats(
                frontier_enqueued=self._core.frontier_enqueued,
                frontier_skipped_visited=self._core.frontier_skipped_visited,
                frontier_skipped_depth=self._core.frontier_skipped_depth,
                frontier_skipped_budget=self._core.frontier_skipped_budget,
                fetched_ok=self._core.fetched_ok,
                fetched_error=self._core.fetched_error,
                parsed_ok=self._core.parsed_ok,
                parsed_error=self._core.parsed_error,
                stored_docs=self._core.stored_docs,
                started_at=self._core.started_at,
                finished_at=self._core.finished_at,
            )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable summary payload."""

        with self._lock:
            core = self._core.to_json()

            start = _parse_iso_utc(self._core.started_at)
            end = _parse_iso_utc(self._core.finished_at) if self._core.finished_at else datetime.now(
                timezone.utc
            )
            duration_seconds = max(0.0, (end - start).total_seconds())

            fetch_elapsed_avg = (
                self._fetch_elapsed_ms_total / self._fetch_elapsed_samples
                if self._fetch_elapsed_samples > 0
                else 0.0
            )

            fetched_total = self._core.fetched_ok + self._core.fetched_error
            parsed_total = self._core.parsed_ok + self._core.parsed_error

            return {
                **core,
                "duration_seconds": duration_seconds,
                "throughput": {
                    "fetched_per_second": (
                        fetched_total / duration_seconds if duration_seconds > 0 else 0.0
                    ),
                    "parsed_per_second": (
                        parsed_total / duration_seconds if duration_seconds > 0 else 0.0
                    ),
                    "stored_docs_per_second": (
                        self._core.stored_docs / duration_seconds if duration_seconds > 0 else 0.0
                    ),
                },
                "frontier": {
                    "extra_status_counts": dict(self._frontier_extra),
                    "snapshot": dict(self._frontier_snapshot),
                },
                "fetch": {
                    "by_backend": self._as_plain_nested_count_dict(
                        self._fetch_backend_counts
                    ),
                    "status_code_counts": dict(self._fetch_status_code_counts),
                    "error_type_counts": dict(self._fetch_error_type_counts),
                    "content_kind_counts": dict(self._fetch_content_kind_counts),
                    "elapsed_ms_total": self._fetch_elapsed_ms_total,
                    "elapsed_ms_samples": self._fetch_elapsed_samples,
                    "elapsed_ms_avg": fetch_elapsed_avg,
                    "bytes_total": self._fetch_bytes_total,
                },
                "parse": {
                    "by_content_kind": self._as_plain_nested_count_dict(
                        self._parse_kind_counts
                    ),
                    "by_parser": self._as_plain_nested_count_dict(
                        self._parse_parser_counts
                    ),
                    "error_type_counts": dict(self._parse_error_type_counts),
                    "text_chars_total": self._parse_text_chars_total,
                    "text_words_total": self._parse_text_words_total,
                    "links_total": self._parse_links_total,
                },
                "storage": {
                    "error_rows": self._storage_error_rows,
                },
                "custom_counters": dict(self._custom_counters),
            }

    @staticmethod
    def _merge_count_dict(target: dict[str, int], incoming: Mapping[str, Any]) -> None:
        for key, value in incoming.items():
            target[str(key)] += int(value)

    @staticmethod
    def _merge_nested_count_dict(
        target: dict[str, dict[str, int]],
        incoming: Mapping[str, Any],
    ) -> None:
        for key, value in incoming.items():
            key = str(key)
            bucket = target.setdefault(key, {})
            if isinstance(value, Mapping):
                for subkey, subvalue in value.items():
                    bucket[str(subkey)] = int(bucket.get(str(subkey), 0)) + int(subvalue)

    @staticmethod
    def _as_plain_nested_count_dict(
        value: Mapping[str, Mapping[str, int]],
    ) -> dict[str, dict[str, int]]:
        return {
            str(key): {str(subkey): int(subvalue) for subkey, subvalue in bucket.items()}
            for key, bucket in value.items()
        }


def _parse_iso_utc(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


__all__ = ["StatsCollector"]

"""Chunker statistics aggregation utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Any, Mapping

from .types import ChunkRecord, ChunkerStats, RejectedRecord, RejectStage, SourceDocument, TextBlock


DOC_WORD_BUCKETS = (100, 250, 500, 1000, 2000, 4000, 8000)
BLOCK_WORD_BUCKETS = (20, 40, 80, 120, 150, 200, 300)
CHUNK_WORD_BUCKETS = (50, 100, 150, 200, 300, 400, 600)


@dataclass(slots=True)
class _LengthAccumulator:
    count: int = 0
    total: int = 0
    min_value: int | None = None
    max_value: int | None = None

    def add(self, value: int) -> None:
        value = int(value)
        self.count += 1
        self.total += value
        if self.min_value is None or value < self.min_value:
            self.min_value = value
        if self.max_value is None or value > self.max_value:
            self.max_value = value

    def to_json(self) -> dict[str, int | float | None]:
        avg = self.total / self.count if self.count > 0 else 0.0
        return {
            "count": self.count,
            "total": self.total,
            "avg": avg,
            "min": self.min_value,
            "max": self.max_value,
        }


def _histogram_key(value: int, buckets: tuple[int, ...]) -> str:
    for upper in buckets:
        if value <= upper:
            return f"<= {upper}"
    return f"> {buckets[-1]}"


def _parse_iso_utc(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class StatsCollector:
    """Thread-safe chunker stats collector."""

    def __init__(self, base: ChunkerStats | None = None) -> None:
        self._lock = threading.Lock()
        self._core = base or ChunkerStats()

        self._doc_words_input = _LengthAccumulator()
        self._doc_chars_input = _LengthAccumulator()
        self._doc_words_kept = _LengthAccumulator()
        self._doc_chars_kept = _LengthAccumulator()

        self._block_words_input = _LengthAccumulator()
        self._block_chars_input = _LengthAccumulator()
        self._block_words_kept = _LengthAccumulator()
        self._block_chars_kept = _LengthAccumulator()

        self._chunk_words_output = _LengthAccumulator()
        self._chunk_chars_output = _LengthAccumulator()

        self._doc_word_hist: dict[str, int] = defaultdict(int)
        self._block_word_hist: dict[str, int] = defaultdict(int)
        self._chunk_word_hist: dict[str, int] = defaultdict(int)

        self._rejections_by_domain: dict[str, int] = defaultdict(int)
        self._rejections_by_stage_domain: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._stage_counts: dict[str, int] = defaultdict(int)

    def record_doc_input(self, doc: SourceDocument) -> None:
        with self._lock:
            self._core.docs_input += 1
            self._core.record_domain_doc(doc.source_domain)
            self._doc_words_input.add(doc.word_count)
            self._doc_chars_input.add(doc.char_count)
            self._doc_word_hist[_histogram_key(doc.word_count, DOC_WORD_BUCKETS)] += 1

    def record_doc_kept(self, doc: SourceDocument) -> None:
        with self._lock:
            self._core.docs_kept += 1
            self._doc_words_kept.add(doc.word_count)
            self._doc_chars_kept.add(doc.char_count)

    def record_doc_rejected(self, *, reason: str, domain: str | None) -> None:
        with self._lock:
            self._core.docs_rejected += 1
            self._core.record_rejection(reason)
            self._record_rejection_domain(RejectStage.DOC, domain)

    def record_block_input(self, block: TextBlock) -> None:
        with self._lock:
            self._core.blocks_input += 1
            self._block_words_input.add(block.word_count)
            self._block_chars_input.add(block.char_count)
            self._block_word_hist[_histogram_key(block.word_count, BLOCK_WORD_BUCKETS)] += 1

    def record_block_kept(self, block: TextBlock) -> None:
        with self._lock:
            self._core.blocks_kept += 1
            self._block_words_kept.add(block.word_count)
            self._block_chars_kept.add(block.char_count)

    def record_block_rejected(self, *, reason: str, domain: str | None) -> None:
        with self._lock:
            self._core.blocks_rejected += 1
            self._core.record_rejection(reason)
            self._record_rejection_domain(RejectStage.BLOCK, domain)

    def record_chunk_output(self, chunk: ChunkRecord) -> None:
        with self._lock:
            self._core.chunks_output += 1
            self._core.record_domain_chunk(chunk.source_domain)
            self._chunk_words_output.add(chunk.word_count)
            self._chunk_chars_output.add(chunk.char_count)
            self._chunk_word_hist[_histogram_key(chunk.word_count, CHUNK_WORD_BUCKETS)] += 1

    def record_chunk_rejected(self, *, reason: str, domain: str | None) -> None:
        with self._lock:
            self._core.chunks_rejected += 1
            self._core.record_rejection(reason)
            self._record_rejection_domain(RejectStage.CHUNK, domain)

    def record_chunk_deduped(self, count: int = 1) -> None:
        if count <= 0:
            return
        with self._lock:
            self._core.chunks_deduped += int(count)

    def record_rejected_row(self, row: RejectedRecord) -> None:
        stage = row.stage
        reason = row.reason
        domain = row.source_domain

        if stage == RejectStage.DOC:
            self.record_doc_rejected(reason=reason, domain=domain)
            return
        if stage == RejectStage.BLOCK:
            self.record_block_rejected(reason=reason, domain=domain)
            return
        if stage == RejectStage.CHUNK:
            self.record_chunk_rejected(reason=reason, domain=domain)
            return

    def record_rejected_rows(self, rows: list[RejectedRecord]) -> None:
        for row in rows:
            self.record_rejected_row(row)

    def finish(self) -> None:
        with self._lock:
            self._core.finish()

    def core(self) -> ChunkerStats:
        with self._lock:
            copy = ChunkerStats(
                docs_input=self._core.docs_input,
                docs_kept=self._core.docs_kept,
                docs_rejected=self._core.docs_rejected,
                blocks_input=self._core.blocks_input,
                blocks_kept=self._core.blocks_kept,
                blocks_rejected=self._core.blocks_rejected,
                chunks_output=self._core.chunks_output,
                chunks_rejected=self._core.chunks_rejected,
                chunks_deduped=self._core.chunks_deduped,
                started_at=self._core.started_at,
                finished_at=self._core.finished_at,
                rejections_by_reason=dict(self._core.rejections_by_reason),
                docs_by_domain=dict(self._core.docs_by_domain),
                chunks_by_domain=dict(self._core.chunks_by_domain),
            )
            return copy

    def to_json(self) -> dict[str, Any]:
        with self._lock:
            core = self._core.to_json()

            start = _parse_iso_utc(self._core.started_at)
            end = _parse_iso_utc(self._core.finished_at) if self._core.finished_at else datetime.now(
                timezone.utc
            )
            duration_seconds = max(0.0, (end - start).total_seconds())

            docs_input = self._core.docs_input
            blocks_input = self._core.blocks_input
            docs_kept = self._core.docs_kept
            blocks_kept = self._core.blocks_kept
            chunks_output = self._core.chunks_output

            return {
                **core,
                "duration_seconds": duration_seconds,
                "rates": {
                    "docs_keep_rate": (docs_kept / docs_input) if docs_input > 0 else 0.0,
                    "blocks_keep_rate": (blocks_kept / blocks_input) if blocks_input > 0 else 0.0,
                    "chunks_per_doc_kept": (chunks_output / docs_kept) if docs_kept > 0 else 0.0,
                    "chunks_per_block_kept": (chunks_output / blocks_kept) if blocks_kept > 0 else 0.0,
                },
                "lengths": {
                    "docs_input_words": self._doc_words_input.to_json(),
                    "docs_input_chars": self._doc_chars_input.to_json(),
                    "docs_kept_words": self._doc_words_kept.to_json(),
                    "docs_kept_chars": self._doc_chars_kept.to_json(),
                    "blocks_input_words": self._block_words_input.to_json(),
                    "blocks_input_chars": self._block_chars_input.to_json(),
                    "blocks_kept_words": self._block_words_kept.to_json(),
                    "blocks_kept_chars": self._block_chars_kept.to_json(),
                    "chunks_output_words": self._chunk_words_output.to_json(),
                    "chunks_output_chars": self._chunk_chars_output.to_json(),
                },
                "histograms": {
                    "doc_words_input": dict(self._doc_word_hist),
                    "block_words_input": dict(self._block_word_hist),
                    "chunk_words_output": dict(self._chunk_word_hist),
                },
                "rejections": {
                    "by_reason": dict(self._core.rejections_by_reason),
                    "by_stage": dict(self._stage_counts),
                    "by_domain": dict(self._rejections_by_domain),
                    "by_stage_domain": {
                        stage: dict(domain_counts)
                        for stage, domain_counts in self._rejections_by_stage_domain.items()
                    },
                    "top_domains": self._top_domains(self._rejections_by_domain, limit=10),
                },
                "domains": {
                    "docs_by_domain": dict(self._core.docs_by_domain),
                    "chunks_by_domain": dict(self._core.chunks_by_domain),
                },
            }

    def _record_rejection_domain(self, stage: RejectStage, domain: str | None) -> None:
        key = _normalize_domain(domain)
        stage_key = stage.value
        self._stage_counts[stage_key] += 1
        self._rejections_by_domain[key] += 1
        self._rejections_by_stage_domain[stage_key][key] += 1

    @staticmethod
    def _top_domains(domain_counts: Mapping[str, int], *, limit: int) -> list[dict[str, int | str]]:
        items = sorted(domain_counts.items(), key=lambda item: item[1], reverse=True)
        return [{"domain": domain, "count": count} for domain, count in items[:limit]]


def _normalize_domain(domain: str | None) -> str:
    key = (domain or "unknown").strip().lower()
    return key or "unknown"


def build_stats_from_rows(
    *,
    docs_input: list[SourceDocument] | None = None,
    docs_kept: list[SourceDocument] | None = None,
    blocks_input: list[TextBlock] | None = None,
    blocks_kept: list[TextBlock] | None = None,
    chunks_output: list[ChunkRecord] | None = None,
    rejected_rows: list[RejectedRecord] | None = None,
) -> dict[str, Any]:
    """Build a summary payload from in-memory pipeline outputs."""

    collector = StatsCollector()

    for doc in docs_input or []:
        collector.record_doc_input(doc)
    for doc in docs_kept or []:
        collector.record_doc_kept(doc)

    for block in blocks_input or []:
        collector.record_block_input(block)
    for block in blocks_kept or []:
        collector.record_block_kept(block)

    for chunk in chunks_output or []:
        collector.record_chunk_output(chunk)

    for row in rejected_rows or []:
        collector.record_rejected_row(row)

    collector.finish()
    return collector.to_json()


__all__ = [
    "StatsCollector",
    "build_stats_from_rows",
    "BLOCK_WORD_BUCKETS",
    "CHUNK_WORD_BUCKETS",
    "DOC_WORD_BUCKETS",
]

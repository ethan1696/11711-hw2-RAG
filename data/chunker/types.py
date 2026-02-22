"""Shared type definitions for the chunker pipeline.

This module is intentionally dependency-light so all chunker components can
import common records without introducing import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
JSONDict = dict[str, JSONValue]


def utc_now_iso() -> str:
    """Return an RFC3339-like UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RejectStage(str, Enum):
    """Stage where an item was rejected."""

    DOC = "doc"
    BLOCK = "block"
    CHUNK = "chunk"


class FilterDecision(str, Enum):
    """Decision produced by quality filtering."""

    KEEP = "keep"
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Normalized source document read from upstream extraction output."""

    doc_id: str
    text: str
    url: str | None = None
    final_url: str | None = None
    title: str | None = None
    source_domain: str | None = None
    content_kind: str | None = None
    content_type: str | None = None
    crawl_time: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    @property
    def effective_url(self) -> str | None:
        return self.final_url or self.url

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SourceDocument":
        """Build document from a JSON-like payload with light coercion."""

        doc_id_raw = payload.get("doc_id", payload.get("id"))
        if doc_id_raw is None:
            raise ValueError("Missing required document id field: 'doc_id' or 'id'")
        doc_id = str(doc_id_raw).strip()
        if not doc_id:
            raise ValueError("Document id cannot be empty")

        text_raw = payload.get("text")
        if text_raw is None:
            raise ValueError("Missing required document field: 'text'")
        text = str(text_raw)

        metadata_raw = payload.get("metadata", {})
        metadata: dict[str, JSONValue]
        if isinstance(metadata_raw, Mapping):
            metadata = dict(metadata_raw)  # type: ignore[arg-type]
        else:
            metadata = {}

        return cls(
            doc_id=doc_id,
            text=text,
            url=_as_optional_str(payload.get("url")),
            final_url=_as_optional_str(payload.get("final_url")),
            title=_as_optional_str(payload.get("title")),
            source_domain=_as_optional_str(payload.get("source_domain")),
            content_kind=_as_optional_str(payload.get("content_kind")),
            content_type=_as_optional_str(payload.get("content_type")),
            crawl_time=_as_optional_str(payload.get("crawl_time")),
            metadata=metadata,
        )

    def to_json(self) -> JSONDict:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "final_url": self.final_url,
            "title": self.title,
            "text": self.text,
            "source_domain": self.source_domain,
            "content_kind": self.content_kind,
            "content_type": self.content_type,
            "crawl_time": self.crawl_time,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One block candidate from a source document prior to chunk packing."""

    doc_id: str
    block_id: int
    text: str
    heading: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_json(self) -> JSONDict:
        return {
            "doc_id": self.doc_id,
            "block_id": self.block_id,
            "text": self.text,
            "heading": self.heading,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class BlockQuality:
    """Quality-filter decision and scores for one block."""

    doc_id: str
    block_id: int
    decision: FilterDecision
    reason: str | None = None
    lang_label: str | None = None
    lang_score: float | None = None
    quality_label: str | None = None
    quality_score: float | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_json(self) -> JSONDict:
        return {
            "doc_id": self.doc_id,
            "block_id": self.block_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "lang_label": self.lang_label,
            "lang_score": self.lang_score,
            "quality_label": self.quality_label,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One output chunk row written to `chunks.jsonl`."""

    chunk_id: str
    doc_id: str
    text: str
    block_ids: list[int]
    url: str | None = None
    title: str | None = None
    source_domain: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_json(self) -> JSONDict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "source_domain": self.source_domain,
            "text": self.text,
            "block_ids": self.block_ids,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """One rejected item row written to `rejected.jsonl`."""

    stage: RejectStage
    reason: str
    doc_id: str | None = None
    block_id: int | None = None
    chunk_id: str | None = None
    url: str | None = None
    source_domain: str | None = None
    text: str | None = None
    text_preview: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_json(self) -> JSONDict:
        return {
            "stage": self.stage.value,
            "reason": self.reason,
            "doc_id": self.doc_id,
            "block_id": self.block_id,
            "chunk_id": self.chunk_id,
            "url": self.url,
            "source_domain": self.source_domain,
            "text": self.text,
            "text_preview": self.text_preview,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ChunkerStats:
    """Mutable counters and summaries for one chunking run."""

    docs_input: int = 0
    docs_kept: int = 0
    docs_rejected: int = 0

    blocks_input: int = 0
    blocks_kept: int = 0
    blocks_rejected: int = 0

    chunks_output: int = 0
    chunks_rejected: int = 0
    chunks_deduped: int = 0

    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None

    rejections_by_reason: dict[str, int] = field(default_factory=dict)
    docs_by_domain: dict[str, int] = field(default_factory=dict)
    chunks_by_domain: dict[str, int] = field(default_factory=dict)

    def increment(self, field_name: str, value: int = 1) -> None:
        if not hasattr(self, field_name):
            raise AttributeError(f"Unknown stats field: {field_name}")
        setattr(self, field_name, int(getattr(self, field_name)) + value)

    def record_rejection(self, reason: str) -> None:
        self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1

    def record_domain_doc(self, domain: str | None) -> None:
        key = (domain or "unknown").strip().lower() or "unknown"
        self.docs_by_domain[key] = self.docs_by_domain.get(key, 0) + 1

    def record_domain_chunk(self, domain: str | None) -> None:
        key = (domain or "unknown").strip().lower() or "unknown"
        self.chunks_by_domain[key] = self.chunks_by_domain.get(key, 0) + 1

    def finish(self) -> None:
        self.finished_at = utc_now_iso()

    def to_json(self) -> JSONDict:
        return {
            "docs_input": self.docs_input,
            "docs_kept": self.docs_kept,
            "docs_rejected": self.docs_rejected,
            "blocks_input": self.blocks_input,
            "blocks_kept": self.blocks_kept,
            "blocks_rejected": self.blocks_rejected,
            "chunks_output": self.chunks_output,
            "chunks_rejected": self.chunks_rejected,
            "chunks_deduped": self.chunks_deduped,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "rejections_by_reason": self.rejections_by_reason,
            "docs_by_domain": self.docs_by_domain,
            "chunks_by_domain": self.chunks_by_domain,
        }


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "BlockQuality",
    "ChunkRecord",
    "ChunkerStats",
    "FilterDecision",
    "JSONDict",
    "JSONPrimitive",
    "JSONValue",
    "RejectedRecord",
    "RejectStage",
    "SourceDocument",
    "TextBlock",
    "utc_now_iso",
]

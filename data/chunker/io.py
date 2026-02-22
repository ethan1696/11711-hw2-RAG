"""I/O helpers for chunker inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .types import ChunkRecord, ChunkerStats, RejectedRecord, SourceDocument


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkerOutputPaths:
    """Resolved output paths under one chunker output root."""

    output_dir: Path
    processed_dir: Path
    chunks_path: Path
    rejected_path: Path
    stats_path: Path


def ensure_output_paths(output_dir: str | Path) -> ChunkerOutputPaths:
    """Create and return output directory layout for chunker artifacts."""

    root = Path(output_dir)
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    return ChunkerOutputPaths(
        output_dir=root,
        processed_dir=processed,
        chunks_path=processed / "chunks.jsonl",
        rejected_path=processed / "rejected.jsonl",
        stats_path=processed / "stats.json",
    )


def iter_source_documents(
    input_path: str | Path,
    *,
    skip_invalid: bool = True,
) -> Iterator[SourceDocument]:
    """Yield normalized source documents from JSON/JSONL input."""

    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        yield from _iter_source_documents_from_jsonl(path, skip_invalid=skip_invalid)
        return

    if suffix == ".json":
        yield from _iter_source_documents_from_json(path, skip_invalid=skip_invalid)
        return

    # Fallback: treat unknown extension as JSONL first.
    yield from _iter_source_documents_from_jsonl(path, skip_invalid=skip_invalid)


def load_source_documents(
    input_path: str | Path,
    *,
    skip_invalid: bool = True,
) -> list[SourceDocument]:
    """Load all source documents into a list."""

    return list(iter_source_documents(input_path, skip_invalid=skip_invalid))


def write_chunks_jsonl(
    chunks: Iterable[ChunkRecord],
    *,
    output_dir: str | Path | None = None,
    paths: ChunkerOutputPaths | None = None,
) -> tuple[Path, int]:
    """Write chunk rows to `processed/chunks.jsonl`."""

    resolved = _resolve_paths(output_dir=output_dir, paths=paths)
    count = _write_jsonl(resolved.chunks_path, (chunk.to_json() for chunk in chunks))
    return resolved.chunks_path, count


def write_rejected_jsonl(
    rejected: Iterable[RejectedRecord],
    *,
    output_dir: str | Path | None = None,
    paths: ChunkerOutputPaths | None = None,
) -> tuple[Path, int]:
    """Write rejection rows to `processed/rejected.jsonl`."""

    resolved = _resolve_paths(output_dir=output_dir, paths=paths)
    count = _write_jsonl(resolved.rejected_path, (row.to_json() for row in rejected))
    return resolved.rejected_path, count


def write_stats_json(
    stats: ChunkerStats | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    paths: ChunkerOutputPaths | None = None,
) -> Path:
    """Write summary stats to `processed/stats.json`."""

    resolved = _resolve_paths(output_dir=output_dir, paths=paths)
    payload: Mapping[str, Any]
    if isinstance(stats, ChunkerStats):
        payload = stats.to_json()
    else:
        payload = stats

    resolved.stats_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved.stats_path


def _resolve_paths(
    *,
    output_dir: str | Path | None,
    paths: ChunkerOutputPaths | None,
) -> ChunkerOutputPaths:
    if paths is not None:
        return paths
    if output_dir is None:
        raise ValueError("Either 'output_dir' or 'paths' must be provided")
    return ensure_output_paths(output_dir)


def _iter_source_documents_from_jsonl(
    path: Path,
    *,
    skip_invalid: bool,
) -> Iterator[SourceDocument]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                if skip_invalid:
                    LOGGER.warning("Skipping invalid JSONL line %d in %s: %s", line_no, path, exc)
                    continue
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc

            if not isinstance(payload, Mapping):
                if skip_invalid:
                    LOGGER.warning(
                        "Skipping non-object JSONL line %d in %s (type=%s)",
                        line_no,
                        path,
                        type(payload).__name__,
                    )
                    continue
                raise ValueError(
                    f"Expected object on line {line_no} in {path}, got {type(payload).__name__}"
                )

            doc = _coerce_source_document(payload, skip_invalid=skip_invalid, context=f"{path}:{line_no}")
            if doc is not None:
                yield doc


def _iter_source_documents_from_json(
    path: Path,
    *,
    skip_invalid: bool,
) -> Iterator[SourceDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for idx, item in enumerate(_iter_doc_payloads_from_json_root(payload), start=1):
        doc = _coerce_source_document(item, skip_invalid=skip_invalid, context=f"{path}#item{idx}")
        if doc is not None:
            yield doc


def _iter_doc_payloads_from_json_root(payload: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item
        return

    if isinstance(payload, Mapping):
        # Common wrappers.
        for key in ("docs", "documents", "items"):
            nested = payload.get(key)
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, Mapping):
                        yield item
                return
            if isinstance(nested, Mapping):
                for nested_key, item in nested.items():
                    if not isinstance(item, Mapping):
                        continue
                    enriched = dict(item)
                    enriched.setdefault("doc_id", str(nested_key))
                    yield enriched
                return

        # Single document object.
        if "text" in payload and ("doc_id" in payload or "id" in payload):
            yield payload
            return

        # Dict keyed by id -> document object.
        for key, value in payload.items():
            if not isinstance(value, Mapping):
                continue
            enriched = dict(value)
            enriched.setdefault("doc_id", str(key))
            yield enriched
        return

    raise ValueError(f"Unsupported JSON root type: {type(payload).__name__}")


def _coerce_source_document(
    payload: Mapping[str, Any],
    *,
    skip_invalid: bool,
    context: str,
) -> SourceDocument | None:
    try:
        return SourceDocument.from_json(payload)
    except Exception as exc:
        if skip_invalid:
            LOGGER.warning("Skipping invalid source document at %s: %s", context, exc)
            return None
        raise ValueError(f"Invalid source document at {context}: {exc}") from exc


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


__all__ = [
    "ChunkerOutputPaths",
    "ensure_output_paths",
    "iter_source_documents",
    "load_source_documents",
    "write_chunks_jsonl",
    "write_rejected_jsonl",
    "write_stats_json",
]

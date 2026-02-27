from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

import numpy as np


WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"


T = TypeVar("T")


def normalize_text(
    text: str,
    *,
    dedup_repeated_lines: bool = False,
) -> str:
    """Normalize chunk text for embedding.

    Steps:
    - remove null bytes
    - normalize line endings
    - optional de-dup of repeated non-empty lines (order-preserving)
    - collapse whitespace and trim
    """

    value = str(text or "")
    value = value.replace("\x00", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")

    if dedup_repeated_lines:
        lines = value.split("\n")
        deduped_lines: list[str] = []
        seen: set[str] = set()
        for line in lines:
            normalized_line = WHITESPACE_RE.sub(" ", line).strip()
            if not normalized_line:
                continue
            if normalized_line in seen:
                continue
            seen.add(normalized_line)
            deduped_lines.append(normalized_line)
        value = "\n".join(deduped_lines)

    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON object rows from JSONL file."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object at {input_path}:{line_no}, got {type(payload).__name__}"
                )
            yield payload


def batch_iter(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Group iterable items into batches of size `batch_size`."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def format_passage(text: str) -> str:
    """Format retrieval corpus text for embedding."""

    return str(text or "")


def format_query(
    query: str,
    *,
    task_description: str = DEFAULT_QUERY_TASK,
) -> str:
    """Format retrieval query text following instruct-style prompt."""

    return f"Instruct: {task_description}\nQuery: {query}"


def make_chunk_uid(record: Mapping[str, Any]) -> str:
    """Return stable chunk UID from record fields.

    Priority:
    1) `chunk_uid`
    2) `chunk_id`
    3) SHA1 hash of canonical fields (`doc_id`, `url`, `title`, `text`).
    """

    for key in ("chunk_uid", "chunk_id"):
        value = record.get(key)
        if value is not None:
            uid = str(value).strip()
            if uid:
                return uid

    canonical = {
        "doc_id": str(record.get("doc_id", "")).strip(),
        "url": str(record.get("url", "")).strip(),
        "title": str(record.get("title", "")).strip(),
        "text": str(record.get("text", "")).strip(),
    }
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def l2_normalize(matrix: Any, *, eps: float = 1e-12) -> Any:
    """L2-normalize rows of an embedding matrix.

    Supports numpy arrays and torch tensors.
    """

    if hasattr(matrix, "norm") and hasattr(matrix, "clamp"):
        # torch.Tensor path
        norms = matrix.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
        return matrix / norms

    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {arr.shape}")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms


def build_chunk_payload(record: Mapping[str, Any], *, chunk_uid: str) -> dict[str, Any]:
    """Build payload row for `chunk_store.jsonl`."""

    payload: dict[str, Any] = {
        "chunk_uid": chunk_uid,
        "title": record.get("title"),
        "text": record.get("text"),
        "url": record.get("url"),
    }

    for optional_key in ("doc_id", "source_domain", "metadata"):
        if optional_key in record:
            payload[optional_key] = record.get(optional_key)

    return payload


__all__ = [
    "DEFAULT_QUERY_TASK",
    "batch_iter",
    "build_chunk_payload",
    "format_passage",
    "format_query",
    "iter_jsonl",
    "l2_normalize",
    "make_chunk_uid",
    "normalize_text",
]

"""Embedding helpers for retrieval pipeline."""

from .storage import OutputPaths, Writer
from .utils import (
    DEFAULT_QUERY_TASK,
    batch_iter,
    build_chunk_payload,
    format_passage,
    format_query,
    iter_jsonl,
    l2_normalize,
    make_chunk_uid,
    normalize_text,
)

__all__ = [
    "DEFAULT_QUERY_TASK",
    "OutputPaths",
    "Writer",
    "batch_iter",
    "build_chunk_payload",
    "format_passage",
    "format_query",
    "iter_jsonl",
    "l2_normalize",
    "make_chunk_uid",
    "normalize_text",
]

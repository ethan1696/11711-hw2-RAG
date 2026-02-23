"""Minimal embedding storage for dense retrieval experiments.

This module stores three artifacts under one output directory:
- ``embeddings.npy``: embedding matrix where row ``i`` is one chunk embedding
- ``embedding_to_uid.jsonl``: mapping from embedding row id to ``chunk_uid``
- ``chunk_store.jsonl``: mapping from ``chunk_uid`` to chunk payload data
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class OutputPaths:
    """Resolved output artifact paths."""

    output_dir: Path
    embeddings_path: Path
    embedding_to_uid_path: Path
    chunk_store_path: Path
    stats_path: Path


class Writer:
    """Simple non-sharded writer for embeddings and chunk mappings."""

    def __init__(
        self,
        output_dir: str | Path,
        dim: int,
        *,
        dtype: str | np.dtype = np.float32,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")

        self.dim = int(dim)
        self.dtype = np.dtype(dtype)
        self.paths = self._ensure_paths(output_dir)

        self._embedding_batches: list[np.ndarray] = []
        self._seen_chunk_uids: set[str] = set()
        self._num_rows = 0
        self._closed = False

        self._embedding_to_uid_handle = self.paths.embedding_to_uid_path.open("w", encoding="utf-8")
        self._chunk_store_handle = self.paths.chunk_store_path.open("w", encoding="utf-8")

    @staticmethod
    def _ensure_paths(output_dir: str | Path) -> OutputPaths:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return OutputPaths(
            output_dir=root,
            embeddings_path=root / "embeddings.npy",
            embedding_to_uid_path=root / "embedding_to_uid.jsonl",
            chunk_store_path=root / "chunk_store.jsonl",
            stats_path=root / "stats.json",
        )

    def append(
        self,
        chunk_uids: Sequence[str],
        embeddings: Any,
        chunk_payloads: Sequence[Mapping[str, Any] | None],
    ) -> int:
        """Append one batch of embeddings and mappings.

        Args:
            chunk_uids: Stable IDs, one per embedding row.
            embeddings: 2D matrix-like object with shape ``[N, dim]``.
            chunk_payloads: Per-chunk data (title/text/url/etc.), one per row.

        Returns:
            Number of rows appended.
        """

        if self._closed:
            raise RuntimeError("Writer is already closed")

        matrix = self._to_numpy_2d(embeddings)
        num_rows = matrix.shape[0]

        if len(chunk_uids) != num_rows:
            raise ValueError(
                f"chunk_uids length ({len(chunk_uids)}) does not match embeddings rows ({num_rows})"
            )
        if len(chunk_payloads) != num_rows:
            raise ValueError(
                f"chunk_payloads length ({len(chunk_payloads)}) does not match embeddings rows ({num_rows})"
            )

        self._embedding_batches.append(matrix)

        for row_offset, (chunk_uid_raw, payload_raw) in enumerate(zip(chunk_uids, chunk_payloads)):
            chunk_uid = str(chunk_uid_raw).strip()
            if not chunk_uid:
                raise ValueError(f"chunk_uid cannot be empty at batch row {row_offset}")

            global_row_id = self._num_rows + row_offset
            self._write_jsonl(
                self._embedding_to_uid_handle,
                {
                    "row_id": global_row_id,
                    "chunk_uid": chunk_uid,
                },
            )

            if chunk_uid not in self._seen_chunk_uids:
                payload = self._coerce_payload(payload_raw)
                payload["chunk_uid"] = chunk_uid
                self._write_jsonl(self._chunk_store_handle, payload)
                self._seen_chunk_uids.add(chunk_uid)

        self._num_rows += num_rows
        return num_rows

    def close(self) -> dict[str, Any]:
        """Finalize files and write summary stats."""

        if self._closed:
            return self._build_summary()

        try:
            matrix = self._final_matrix()
            np.save(self.paths.embeddings_path, matrix)

            summary = self._build_summary()
            self.paths.stats_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return summary
        finally:
            self._embedding_to_uid_handle.close()
            self._chunk_store_handle.close()
            self._closed = True

    def _final_matrix(self) -> np.ndarray:
        if not self._embedding_batches:
            return np.empty((0, self.dim), dtype=self.dtype)
        return np.concatenate(self._embedding_batches, axis=0)

    def _build_summary(self) -> dict[str, Any]:
        return {
            "num_rows": self._num_rows,
            "dim": self.dim,
            "num_unique_chunks": len(self._seen_chunk_uids),
            "paths": {
                "embeddings": str(self.paths.embeddings_path),
                "embedding_to_uid": str(self.paths.embedding_to_uid_path),
                "chunk_store": str(self.paths.chunk_store_path),
                "stats": str(self.paths.stats_path),
            },
        }

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _write_jsonl(handle, payload: Mapping[str, Any]) -> None:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
        handle.write("\n")

    @staticmethod
    def _coerce_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if payload is None:
            return {}
        return dict(payload)

    def _to_numpy_2d(self, embeddings: Any) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            matrix = embeddings
        elif hasattr(embeddings, "detach") and hasattr(embeddings, "cpu") and hasattr(embeddings, "numpy"):
            tensor = embeddings.detach().cpu()
            try:
                matrix = tensor.numpy()
            except TypeError as exc:
                # NumPy does not support torch bfloat16 directly.
                if "BFloat16" not in str(exc):
                    raise
                matrix = tensor.float().numpy()
        else:
            matrix = np.asarray(embeddings)

        if matrix.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {matrix.shape}")
        if matrix.shape[1] != self.dim:
            raise ValueError(f"embeddings dim mismatch: got {matrix.shape[1]}, expected {self.dim}")

        return matrix.astype(self.dtype, copy=False)


__all__ = [
    "OutputPaths",
    "Writer",
]

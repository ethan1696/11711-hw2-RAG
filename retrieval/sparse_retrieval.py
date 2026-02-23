"""Sparse retrieval over cached BM25 index artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pickle
import re
import sys
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np

from retrieval.embed.utils import iter_jsonl, normalize_text


LOGGER = logging.getLogger(__name__)
DEFAULT_INDEX_NAME = "bm25_index.pkl"
DEFAULT_CHUNK_STORE_NAME = "chunk_store.jsonl"
FALLBACK_CHUNK_STORE_NAME = "bm25_chunk_store.jsonl"


@dataclass(frozen=True, slots=True)
class SparsePaths:
    """Resolved sparse retrieval artifact paths."""

    index_dir: Path
    index_path: Path
    chunk_store_path: Path


def _setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


class SparseRetriever:
    """In-memory BM25 retriever that loads index on initialization."""

    def __init__(
        self,
        *,
        index_dir: str | Path,
        index_name: str = DEFAULT_INDEX_NAME,
        chunk_store_name: str | None = None,
    ) -> None:
        self.paths = self._resolve_paths(
            index_dir=index_dir,
            index_name=index_name,
            chunk_store_name=chunk_store_name,
        )
        payload = self._load_index(self.paths.index_path)

        self.bm25 = payload["bm25"]
        self.row_to_uid = [str(value) for value in payload["chunk_uids"]]
        self.tokenizer_config = dict(payload.get("tokenizer", {}))
        self.chunk_store = self._load_chunk_store(self.paths.chunk_store_path)

        self._token_re = self._compile_token_pattern(
            str(self.tokenizer_config.get("pattern", r"[A-Za-z0-9_]+"))
        )
        self._lowercase = bool(self.tokenizer_config.get("lowercase", True))
        self._min_token_len = int(self.tokenizer_config.get("min_token_len", 1))

    @property
    def size(self) -> int:
        return len(self.row_to_uid)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k BM25 results for ``query``."""

        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.size == 0:
            return []

        normalized_query = normalize_text(query)
        if not normalized_query:
            return []

        query_tokens = self._tokenize_query(normalized_query)
        if not query_tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        if scores.ndim != 1:
            raise ValueError(f"Expected BM25 scores to be 1D, got shape {scores.shape}")
        if scores.shape[0] != self.size:
            raise ValueError(
                f"BM25 score length ({scores.shape[0]}) does not match row map length ({self.size})"
            )

        k = min(int(top_k), self.size)
        top_indices = self._topk_desc(scores, k)
        return self._build_results(top_indices=top_indices, scores=scores)

    @staticmethod
    def _topk_desc(values: np.ndarray, k: int) -> np.ndarray:
        if k >= values.shape[0]:
            return np.argsort(-values)
        selected = np.argpartition(-values, kth=k - 1)[:k]
        return selected[np.argsort(-values[selected])]

    def _build_results(self, *, top_indices: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for rank, row_id in enumerate(top_indices.tolist(), start=1):
            row_idx = int(row_id)
            chunk_uid = self.row_to_uid[row_idx]
            payload = dict(self.chunk_store.get(chunk_uid, {}))
            score = float(scores[row_idx])

            result: dict[str, Any] = {
                "rank": rank,
                "row_id": row_idx,
                "chunk_uid": chunk_uid,
                "score": score,
                "metric": "bm25",
            }
            if payload:
                result.update(payload)
            results.append(result)

        return results

    def _tokenize_query(self, text: str) -> list[str]:
        value = text.lower() if self._lowercase else text
        tokens = self._token_re.findall(value)
        if self._min_token_len > 1:
            return [token for token in tokens if len(token) >= self._min_token_len]
        return tokens

    @staticmethod
    def _compile_token_pattern(pattern: str) -> re.Pattern[str]:
        try:
            return re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid token pattern in index: {pattern!r}") from exc

    @staticmethod
    def _load_index(index_path: Path) -> dict[str, Any]:
        if not index_path.exists():
            raise FileNotFoundError(f"Missing BM25 index file: {index_path}")

        with index_path.open("rb") as handle:
            payload = pickle.load(handle)

        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in index file: {index_path}")
        if "bm25" not in payload:
            raise ValueError(f"Missing 'bm25' object in index file: {index_path}")
        if "chunk_uids" not in payload:
            raise ValueError(f"Missing 'chunk_uids' in index file: {index_path}")

        chunk_uids = payload["chunk_uids"]
        if not isinstance(chunk_uids, list):
            raise ValueError(f"'chunk_uids' must be a list in index file: {index_path}")

        return payload

    @staticmethod
    def _load_chunk_store(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Missing chunk store file: {path}")

        mapping: dict[str, dict[str, Any]] = {}
        for payload in iter_jsonl(path):
            chunk_uid = str(payload.get("chunk_uid", "")).strip()
            if not chunk_uid:
                continue
            mapping[chunk_uid] = dict(payload)
        return mapping

    @staticmethod
    def _resolve_paths(
        *,
        index_dir: str | Path,
        index_name: str,
        chunk_store_name: str | None,
    ) -> SparsePaths:
        root = Path(index_dir)
        index_filename = str(index_name).strip()
        if not index_filename:
            raise ValueError("index_name cannot be empty")
        index_path = root / index_filename

        chunk_store_path: Path
        if chunk_store_name is not None:
            raw = Path(str(chunk_store_name).strip())
            if not str(raw):
                raise ValueError("chunk_store_name cannot be empty when set")
            chunk_store_path = raw if raw.is_absolute() else root / raw
        else:
            default_path = root / DEFAULT_CHUNK_STORE_NAME
            fallback_path = root / FALLBACK_CHUNK_STORE_NAME
            if default_path.exists():
                chunk_store_path = default_path
            elif fallback_path.exists():
                chunk_store_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Could not find chunk store in {root}. "
                    f"Tried {DEFAULT_CHUNK_STORE_NAME} and {FALLBACK_CHUNK_STORE_NAME}."
                )

        return SparsePaths(
            index_dir=root,
            index_path=index_path,
            chunk_store_path=chunk_store_path,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse retrieval over cached BM25 index.")
    parser.add_argument("--index_dir", type=Path, required=True, help="Directory containing BM25 artifacts.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--index_name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="BM25 index filename under index_dir.",
    )
    parser.add_argument(
        "--chunk_store_name",
        type=str,
        default=None,
        help=(
            "Chunk store filename or absolute path. "
            "If omitted, tries chunk_store.jsonl then bm25_chunk_store.jsonl."
        ),
    )
    parser.add_argument(
        "--include_text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include chunk text in printed results.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    retriever = SparseRetriever(
        index_dir=args.index_dir,
        index_name=args.index_name,
        chunk_store_name=args.chunk_store_name,
    )
    results = retriever.search(args.query, top_k=args.top_k)

    if not args.include_text:
        for row in results:
            row.pop("text", None)

    return {
        "query": args.query,
        "top_k": args.top_k,
        "num_results": len(results),
        "index_dir": str(args.index_dir),
        "index_name": args.index_name,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


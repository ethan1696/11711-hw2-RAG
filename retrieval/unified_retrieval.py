"""Unified retrieval interface for dense, sparse, and hybrid (RRF) ranking."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Literal

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from retrieval.dense_retrieval import DEFAULT_MODEL, DenseRetriever
from retrieval.sparse_retrieval import DEFAULT_INDEX_NAME, SparseRetriever


LOGGER = logging.getLogger(__name__)
RetrievalMode = Literal["dense", "sparse", "hybrid"]


def _setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


class UnifiedRetriever:
    """One retriever interface with dense/sparse/hybrid modes."""

    def __init__(
        self,
        *,
        dense_embed_dir: str | Path | None = None,
        sparse_index_dir: str | Path | None = None,
        dense_model: str = DEFAULT_MODEL,
        dense_device: str = "auto",
        dense_dtype: str = "auto",
        dense_metric: str = "cosine",
        dense_max_length: int = 8192,
        dense_normalize_query: bool = True,
        dense_trust_remote_code: bool = True,
        sparse_index_name: str = DEFAULT_INDEX_NAME,
        sparse_chunk_store_name: str | None = None,
    ) -> None:
        self.dense_embed_dir = Path(dense_embed_dir) if dense_embed_dir is not None else None
        self.sparse_index_dir = Path(sparse_index_dir) if sparse_index_dir is not None else None

        self.dense_model = dense_model
        self.dense_device = dense_device
        self.dense_dtype = dense_dtype
        self.dense_metric = dense_metric
        self.dense_max_length = dense_max_length
        self.dense_normalize_query = dense_normalize_query
        self.dense_trust_remote_code = dense_trust_remote_code

        self.sparse_index_name = sparse_index_name
        self.sparse_chunk_store_name = sparse_chunk_store_name

        self._dense: DenseRetriever | None = None
        self._sparse: SparseRetriever | None = None

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = "hybrid",
        fusion_top_k: int | None = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError("mode must be one of: dense, sparse, hybrid")

        if resolved_mode == "dense":
            return self._get_dense().search(query, top_k=top_k)

        if resolved_mode == "sparse":
            return self._get_sparse().search(query, top_k=top_k)

        # Hybrid mode (RRF)
        return self._search_hybrid(
            query=query,
            top_k=top_k,
            fusion_top_k=fusion_top_k,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    def _search_hybrid(
        self,
        *,
        query: str,
        top_k: int,
        fusion_top_k: int | None,
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[dict[str, Any]]:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("dense_weight and sparse_weight must be >= 0")
        if dense_weight == 0 and sparse_weight == 0:
            raise ValueError("At least one of dense_weight/sparse_weight must be > 0")

        candidate_k = (
            fusion_top_k
            if fusion_top_k is not None
            else max(top_k * 5, 50)
        )
        if candidate_k <= 0:
            raise ValueError("fusion_top_k must be > 0 when set")

        dense_results = self._get_dense().search(query, top_k=candidate_k)
        sparse_results = self._get_sparse().search(query, top_k=candidate_k)

        fused = self._rrf_fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        return fused[:top_k]

    @staticmethod
    def _rrf_fuse(
        *,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[dict[str, Any]]:
        # chunk_uid -> fused record
        fused: dict[str, dict[str, Any]] = {}

        def ensure(uid: str) -> dict[str, Any]:
            if uid not in fused:
                fused[uid] = {
                    "chunk_uid": uid,
                    "metric": "rrf",
                    "score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "dense_score": None,
                    "sparse_score": None,
                }
            return fused[uid]

        for rank, row in enumerate(dense_results, start=1):
            uid = str(row.get("chunk_uid", "")).strip()
            if not uid:
                continue
            item = ensure(uid)
            item["score"] = float(item["score"]) + (dense_weight * (1.0 / (rrf_k + rank)))
            item["dense_rank"] = rank
            item["dense_score"] = row.get("score")

            # Keep useful payload fields from dense path.
            for key in ("row_id", "doc_id", "title", "text", "url", "source_domain", "metadata"):
                if key in row and key not in item:
                    item[key] = row[key]

        for rank, row in enumerate(sparse_results, start=1):
            uid = str(row.get("chunk_uid", "")).strip()
            if not uid:
                continue
            item = ensure(uid)
            item["score"] = float(item["score"]) + (sparse_weight * (1.0 / (rrf_k + rank)))
            item["sparse_rank"] = rank
            item["sparse_score"] = row.get("score")

            # Fill payload fields if missing.
            for key in ("row_id", "doc_id", "title", "text", "url", "source_domain", "metadata"):
                if key in row and key not in item:
                    item[key] = row[key]

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -float(item["score"]),
                item["dense_rank"] if item["dense_rank"] is not None else 10**9,
                item["sparse_rank"] if item["sparse_rank"] is not None else 10**9,
                str(item["chunk_uid"]),
            ),
        )

        for idx, row in enumerate(ranked, start=1):
            row["rank"] = idx

        return ranked

    def _get_dense(self) -> DenseRetriever:
        if self._dense is not None:
            return self._dense
        if self.dense_embed_dir is None:
            raise ValueError("dense_embed_dir is required for dense/hybrid mode")

        LOGGER.info("Loading dense retriever from %s", self.dense_embed_dir)
        self._dense = DenseRetriever(
            embed_dir=self.dense_embed_dir,
            model=self.dense_model,
            device=self.dense_device,
            dtype=self.dense_dtype,
            metric=self.dense_metric,
            max_length=self.dense_max_length,
            normalize_query=self.dense_normalize_query,
            trust_remote_code=self.dense_trust_remote_code,
        )
        return self._dense

    def _get_sparse(self) -> SparseRetriever:
        if self._sparse is not None:
            return self._sparse
        if self.sparse_index_dir is None:
            raise ValueError("sparse_index_dir is required for sparse/hybrid mode")

        LOGGER.info("Loading sparse retriever from %s", self.sparse_index_dir)
        self._sparse = SparseRetriever(
            index_dir=self.sparse_index_dir,
            index_name=self.sparse_index_name,
            chunk_store_name=self.sparse_chunk_store_name,
        )
        return self._sparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified dense/sparse/hybrid retriever.")

    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--mode", type=str, default="hybrid", choices=("dense", "sparse", "hybrid"))

    parser.add_argument("--dense_embed_dir", type=Path, default=Path("retrieval/output_embed"))
    parser.add_argument("--dense_model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dense_device", type=str, default="auto")
    parser.add_argument(
        "--dense_dtype",
        type=str,
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument("--dense_metric", type=str, default="cosine", choices=("cosine", "ip", "l2"))
    parser.add_argument("--dense_max_length", type=int, default=8192)
    parser.add_argument(
        "--dense_normalize_query",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dense_trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--sparse_index_dir", type=Path, default=Path("retrieval/output_embed"))
    parser.add_argument("--sparse_index_name", type=str, default=DEFAULT_INDEX_NAME)
    parser.add_argument("--sparse_chunk_store_name", type=str, default=None)

    parser.add_argument(
        "--fusion_top_k",
        type=int,
        default=None,
        help="Candidate depth per retriever before RRF (default: max(top_k*5, 50)).",
    )
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--dense_weight", type=float, default=1.0)
    parser.add_argument("--sparse_weight", type=float, default=1.0)

    parser.add_argument(
        "--include_text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include chunk text in output rows.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    retriever = UnifiedRetriever(
        dense_embed_dir=args.dense_embed_dir,
        sparse_index_dir=args.sparse_index_dir,
        dense_model=args.dense_model,
        dense_device=args.dense_device,
        dense_dtype=args.dense_dtype,
        dense_metric=args.dense_metric,
        dense_max_length=args.dense_max_length,
        dense_normalize_query=args.dense_normalize_query,
        dense_trust_remote_code=args.dense_trust_remote_code,
        sparse_index_name=args.sparse_index_name,
        sparse_chunk_store_name=args.sparse_chunk_store_name,
    )

    results = retriever.search(
        args.query,
        top_k=args.top_k,
        mode=args.mode,
        fusion_top_k=args.fusion_top_k,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )

    if not args.include_text:
        for row in results:
            row.pop("text", None)

    return {
        "query": args.query,
        "mode": args.mode,
        "top_k": args.top_k,
        "num_results": len(results),
        "dense_embed_dir": str(args.dense_embed_dir),
        "sparse_index_dir": str(args.sparse_index_dir),
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


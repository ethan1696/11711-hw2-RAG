from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from tqdm import tqdm

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from rag.config import RAGDebugConfig, load_rag_config
from rag.system import RAGSystem


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG over leaderboard queries and write answer JSON."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to RAG YAML/JSON config.",
    )
    parser.add_argument(
        "--queries_json",
        type=Path,
        default=Path("rag/leaderboard_queries.json"),
        help="Path to leaderboard queries JSON file.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("rag/leaderboard_answers.json"),
        help="Where to write the output answers JSON.",
    )
    parser.add_argument(
        "--andrewid",
        type=str,
        default="ethanwan",
        help="Andrew ID to store in the output JSON.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Optional cap for quick sanity checks.",
    )
    parser.add_argument(
        "--closed_book",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run without retrieval context (LLM only).",
    )
    return parser.parse_args(argv)


def _build_system(config: RAGDebugConfig) -> RAGSystem:
    return RAGSystem(
        dense_embed_dir=config.dense_embed_dir,
        sparse_index_dir=config.sparse_index_dir,
        llm_model_name=config.llm_model_name,
        retrieval_mode=config.retrieval_mode,
        retrieval_top_k=config.retrieval_top_k,
        dense_device=config.dense_device,
        dense_dtype=config.dense_dtype,
        llm_torch_dtype=config.llm_torch_dtype,
        llm_device=config.llm_device,
        llm_device_map=config.llm_device_map,
        dense_model_name=config.dense_model_name,
        sparse_index_name=config.sparse_index_name,
        sparse_chunk_store_name=config.sparse_chunk_store_name,
        trust_remote_code=config.trust_remote_code,
        system_prompt=config.system_prompt,
        user_prompt_template=config.user_prompt_template,
        context_entry_template=config.context_entry_template,
        empty_context_text=config.empty_context_text,
        fusion_top_k=config.fusion_top_k,
        rrf_k=config.rrf_k,
        dense_weight=config.dense_weight,
        sparse_weight=config.sparse_weight,
    )


def _load_queries(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a JSON list of query objects")

    rows: list[dict[str, str]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Query row {idx} must be a JSON object")
        query_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        if not query_id:
            raise ValueError(f"Query row {idx} missing non-empty 'id'")
        if not question:
            raise ValueError(f"Query row {idx} missing non-empty 'question'")
        rows.append({"id": query_id, "question": question})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_rag_config(args.config)
    if bool(args.closed_book):
        # Force closed-book at construction time so retriever models are not loaded.
        config.retrieval_mode = "closed_book"
    rag = _build_system(config)
    run_mode = config.retrieval_mode

    queries = _load_queries(args.queries_json)
    if args.max_queries is not None:
        if args.max_queries <= 0:
            raise ValueError("--max_queries must be > 0 when set")
        queries = queries[: args.max_queries]

    results: dict[str, Any] = {"andrewid": str(args.andrewid).strip()}
    for row in tqdm(queries, desc="Answering leaderboard queries", unit="query"):
        result = rag.answer_question(
            row["question"],
            top_k=config.retrieval_top_k,
            mode=run_mode,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            return_context=False,
            return_prompt=False,
        )
        results[row["id"]] = str(result.get("answer", "")).strip()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "output_json": str(args.output_json),
        "query_count": len(queries),
        "mode": run_mode,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

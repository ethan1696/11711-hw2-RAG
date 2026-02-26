"""Interactive debug/testing interface for config-driven RAG runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from rag.config import RAGDebugConfig, load_rag_config
from rag.system import RAGSystem


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive RAG debug interface.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to RAG YAML/JSON config.",
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


def run_interactive(config: RAGDebugConfig) -> int:
    rag = _build_system(config)
    exits = set(config.exit_commands)

    print("RAG debug interface ready.")
    print(f"Type a question. Exit commands: {', '.join(sorted(exits))}")

    while True:
        try:
            query = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return 0

        if not query:
            continue
        if query.lower() in exits:
            print("Exiting.")
            return 0

        result = rag.answer_question(
            query,
            top_k=config.retrieval_top_k,
            mode=config.retrieval_mode,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            return_context=config.show_contexts,
            return_prompt=config.show_model_prompt,
        )

        if config.show_model_prompt:
            print("\n=== Prompt Sent To Model ===")
            print(result.get("model_prompt", ""))

        print("\n=== Model Answer ===")
        print(result.get("answer", ""))

        if config.show_contexts:
            contexts = result.get("contexts", [])
            print(f"\n=== Retrieved Contexts ({len(contexts)}) ===")
            for row in contexts:
                rank = row.get("rank")
                title = row.get("title") or "Untitled"
                url = row.get("url") or "unknown"
                print(f"[{rank}] {title} | {url}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_rag_config(args.config)
    return run_interactive(config)


if __name__ == "__main__":
    raise SystemExit(main())

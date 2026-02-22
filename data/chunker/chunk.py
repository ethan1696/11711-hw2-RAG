"""Chunker CLI entrypoint.

Pipeline:
1) load source docs
2) split to blocks
3) classify blocks
4) write each passing block as one chunk
5) write chunks/rejections/stats
"""

from __future__ import annotations

import hashlib
import os
import sys

if __package__ in {None, ""}:
    # Avoid shadowing stdlib `types` by local `data/chunker/types.py`.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filtered_sys_path: list[str] = []
    for entry in sys.path:
        normalized = os.path.abspath(entry) if entry else os.getcwd()
        if normalized == script_dir:
            continue
        filtered_sys_path.append(entry)
    sys.path = filtered_sys_path

    repo_root = os.path.dirname(os.path.dirname(script_dir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from data.chunker.block_split import BlockSplitConfig, split_document
from data.chunker.config import ChunkerConfig, resolve_config
from data.chunker.io import (
    ensure_output_paths,
    iter_source_documents,
    write_chunks_jsonl,
    write_rejected_jsonl,
    write_stats_json,
)
from data.chunker.quality_filter import BlockFilterResult, QualityFilter
from data.chunker.stats import StatsCollector
from data.chunker.types import ChunkRecord, RejectedRecord, RejectStage, SourceDocument, TextBlock


LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chunking pipeline end-to-end.")
    parser.add_argument("--config", type=Path, default=None, help="Path to chunker JSON/YAML config.")
    parser.add_argument("--input_json", type=Path, default=None, help="Override input JSON/JSONL path.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--strict", action="store_true", help="Fail on invalid input rows.")
    parser.add_argument("--max_docs", type=int, default=None, help="Optional cap for number of docs.")
    parser.add_argument(
        "--split_max_block_words",
        type=int,
        default=None,
        help="Override maximum words per sentence-packed block.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def _setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _preview_text(text: str, max_chars: int = 240) -> str:
    value = " ".join((text or "").split()).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _normalize_for_dedup(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _build_chunk_record(
    *,
    doc: SourceDocument,
    chunk_index: int,
    text: str,
    block_ids: list[int],
    token_count_estimate: int,
) -> ChunkRecord:
    digest = hashlib.sha1(f"{doc.doc_id}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
    return ChunkRecord(
        chunk_id=digest,
        doc_id=doc.doc_id,
        text=text,
        block_ids=block_ids,
        url=doc.effective_url,
        title=doc.title,
        source_domain=doc.source_domain,
        metadata={
            "chunk_index": chunk_index,
            "token_count_estimate": token_count_estimate,
            "block_count": len(block_ids),
        },
    )


def _build_block_rejection(
    *,
    doc: SourceDocument,
    block: TextBlock,
    result: BlockFilterResult,
) -> RejectedRecord:
    return RejectedRecord(
        stage=RejectStage.BLOCK,
        reason=result.reason or "block_filtered",
        doc_id=doc.doc_id,
        block_id=block.block_id,
        url=doc.effective_url,
        source_domain=doc.source_domain,
        text=block.text,
        text_preview=_preview_text(block.text),
        metadata={
            "quality_label": result.quality_label,
            "quality_score": result.quality_score,
            **result.metadata,
        },
    )


def _build_chunk_rejection(
    *,
    doc: SourceDocument,
    reason: str,
    chunk_text: str,
    chunk_index: int,
    block_ids: Iterable[int],
) -> RejectedRecord:
    return RejectedRecord(
        stage=RejectStage.CHUNK,
        reason=reason,
        doc_id=doc.doc_id,
        chunk_id=f"{doc.doc_id}:{chunk_index}",
        url=doc.effective_url,
        source_domain=doc.source_domain,
        text_preview=_preview_text(chunk_text),
        metadata={
            "chunk_index": chunk_index,
            "block_ids": list(block_ids),
        },
    )


def run_pipeline(
    config: ChunkerConfig,
    *,
    skip_invalid: bool = True,
    max_docs: int | None = None,
    split_max_block_words: int | None = None,
) -> dict[str, object]:
    """Run chunking pipeline and write outputs."""

    output_paths = ensure_output_paths(config.output_path)
    effective_max_block_words = (
        split_max_block_words if split_max_block_words is not None else config.max_block_words
    )
    if effective_max_block_words <= 0:
        raise ValueError("max_block_words must be > 0")

    split_cfg = BlockSplitConfig(
        min_block_words=config.min_block_words,
        max_block_words=effective_max_block_words,
    )
    quality_filter = QualityFilter.from_chunker_config(config)
    stats = StatsCollector()

    chunks: list[ChunkRecord] = []
    rejected: list[RejectedRecord] = []
    seen_chunk_hashes: set[str] = set()

    for index, doc in enumerate(iter_source_documents(config.input_path, skip_invalid=skip_invalid)):
        if max_docs is not None and index >= max_docs:
            break

        stats.record_doc_input(doc)
        blocks = split_document(doc, config=split_cfg)
        for block in blocks:
            stats.record_block_input(block)

        emitted_chunks = 0
        for block in blocks:
            result = quality_filter.evaluate_block(block)
            if not result.passed:
                row = _build_block_rejection(doc=doc, block=block, result=result)
                rejected.append(row)
                stats.record_rejected_row(row)
                continue

            stats.record_block_kept(block)
            chunk_text = block.text.strip()
            if not chunk_text:
                continue

            if config.dedup:
                key = hashlib.sha1(_normalize_for_dedup(chunk_text).encode("utf-8")).hexdigest()
                if key in seen_chunk_hashes:
                    row = _build_chunk_rejection(
                        doc=doc,
                        reason="duplicate_chunk_text",
                        chunk_text=chunk_text,
                        chunk_index=emitted_chunks,
                        block_ids=[block.block_id],
                    )
                    rejected.append(row)
                    stats.record_rejected_row(row)
                    stats.record_chunk_deduped()
                    continue
                seen_chunk_hashes.add(key)

            chunk = _build_chunk_record(
                doc=doc,
                chunk_index=emitted_chunks,
                text=chunk_text,
                block_ids=[block.block_id],
                token_count_estimate=block.word_count,
            )
            chunks.append(chunk)
            stats.record_chunk_output(chunk)
            emitted_chunks += 1

        if emitted_chunks > 0:
            stats.record_doc_kept(doc)

    stats.finish()
    stats_payload = stats.to_json()
    stats_payload["config"] = config.to_dict()
    stats_payload["paths"] = {
        "chunks": str(output_paths.chunks_path),
        "rejected": str(output_paths.rejected_path),
        "stats": str(output_paths.stats_path),
    }

    chunks_path, chunks_count = write_chunks_jsonl(chunks, paths=output_paths)
    rejected_path, rejected_count = write_rejected_jsonl(rejected, paths=output_paths)
    stats_path = write_stats_json(stats_payload, paths=output_paths)

    summary: dict[str, object] = {
        "chunks_count": chunks_count,
        "rejected_count": rejected_count,
        "docs_input": stats_payload.get("docs_input", 0),
        "docs_kept": stats_payload.get("docs_kept", 0),
        "docs_rejected": stats_payload.get("docs_rejected", 0),
        "blocks_input": stats_payload.get("blocks_input", 0),
        "blocks_kept": stats_payload.get("blocks_kept", 0),
        "blocks_rejected": stats_payload.get("blocks_rejected", 0),
        "paths": {
            "chunks": str(chunks_path),
            "rejected": str(rejected_path),
            "stats": str(stats_path),
        },
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.log_level)

    config = resolve_config(
        config_path=args.config,
        input_json=str(args.input_json) if args.input_json else None,
        output_dir=str(args.output_dir) if args.output_dir else None,
    )

    summary = run_pipeline(
        config,
        skip_invalid=not args.strict,
        max_docs=args.max_docs,
        split_max_block_words=args.split_max_block_words,
    )

    LOGGER.info("Chunking complete:\n%s", json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

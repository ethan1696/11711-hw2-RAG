"""Sanity-check utility for per-block quality filtering.

Example:
  python data/chunker/sanity_check_quality_filter.py \
    --input_json data/crawled_output_assignment/parsed/docs.jsonl \
    --max_docs 120 \
    --seed 42
"""

from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    # Avoid shadowing stdlib `types` by local `data/chunker/types.py` when this
    # file is executed as a script.
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
from dataclasses import dataclass, replace
from pathlib import Path
import random
from typing import Any

from data.chunker.block_split import BlockSplitConfig, split_document
from data.chunker.config import load_config
from data.chunker.io import iter_source_documents
from data.chunker.quality_filter import QualityFilter, QualityFilterConfig
from data.chunker.types import FilterDecision, JSONValue, TextBlock


@dataclass(frozen=True, slots=True)
class BlockSample:
    decision: FilterDecision
    reason: str | None
    quality_label: str | None
    quality_score: float | None
    doc_id: str
    block_id: int
    source_domain: str | None
    url: str | None
    word_count: int
    char_count: int
    text_preview: str


class _DummyFastTextModel:
    """Optional local-only mock model for quick script validation."""

    def get_labels(self) -> list[str]:
        return ["__label__0", "__label__1"]

    def predict(self, text: str, k: int = 1) -> tuple[list[str], list[float]]:
        lowered = text.lower()
        bad_terms = ("cookie", "subscribe", "newsletter", "captcha", "sign up")
        if any(term in lowered for term in bad_terms):
            return ["__label__0"], [0.98]
        return ["__label__1"], [0.84]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample pass/fail blocks from quality filter output.",
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="Path to source docs (.jsonl or .json).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional chunker config file (.json/.yaml/.yml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible samples.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=120,
        help="Maximum number of documents to scan.",
    )
    parser.add_argument(
        "--pass_samples",
        type=int,
        default=5,
        help="Number of passing samples to print.",
    )
    parser.add_argument(
        "--fail_samples",
        type=int,
        default=5,
        help="Number of failing samples to print.",
    )
    parser.add_argument(
        "--preview_chars",
        type=int,
        default=220,
        help="Max text preview chars in each sample.",
    )

    # Splitter controls (for generating candidate blocks)
    parser.add_argument(
        "--split_min_block_words",
        type=int,
        default=8,
        help="Splitter min words before tiny-block merge.",
    )
    parser.add_argument(
        "--split_min_block_chars",
        type=int,
        default=40,
        help="Splitter min chars before tiny-block merge.",
    )

    # Quality filter overrides
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=None,
        help="Quality score threshold override (0..1).",
    )
    parser.add_argument(
        "--min_block_words",
        type=int,
        default=None,
        help="Quality filter minimum block words override.",
    )
    parser.add_argument(
        "--min_block_chars",
        type=int,
        default=None,
        help="Quality filter minimum block chars override.",
    )
    parser.add_argument(
        "--max_list_lines",
        type=int,
        default=None,
        help="Quality filter max list lines override.",
    )
    parser.add_argument(
        "--keep_list_blocks",
        choices=("true", "false"),
        default=None,
        help="Override list block filtering behavior.",
    )
    parser.add_argument(
        "--require_pass_label",
        choices=("true", "false"),
        default=None,
        help="Whether predicted label must match pass labels.",
    )
    parser.add_argument(
        "--pass_labels",
        type=str,
        default=None,
        help="Comma-separated pass labels, e.g. '__label__1'.",
    )
    parser.add_argument(
        "--quality_model_path",
        type=Path,
        default=None,
        help="Optional local fastText model path override.",
    )
    parser.add_argument(
        "--fasttext_repo_id",
        type=str,
        default=None,
        help="HF repo id override for model download.",
    )
    parser.add_argument(
        "--fasttext_filename",
        type=str,
        default=None,
        help="HF model filename override for model download.",
    )
    parser.add_argument(
        "--use_dummy_model",
        action="store_true",
        help="Use local dummy model (no HF download) for quick script validation.",
    )
    return parser.parse_args(argv)


def _maybe_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def _parse_pass_labels(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    labels = tuple(item.strip() for item in raw.split(",") if item.strip())
    return labels or None


def _merge_quality_config(args: argparse.Namespace) -> QualityFilterConfig:
    if args.config is not None:
        chunker_cfg = load_config(args.config)
        cfg = QualityFilterConfig(
            quality_model_path=chunker_cfg.quality_model_path,
            quality_threshold=chunker_cfg.quality_threshold or 0.0,
            min_block_words=chunker_cfg.min_block_words,
            keep_list_blocks=chunker_cfg.keep_list_blocks,
            max_list_lines=chunker_cfg.max_list_lines,
        )
    else:
        cfg = QualityFilterConfig()

    if args.quality_threshold is not None:
        cfg = replace(cfg, quality_threshold=args.quality_threshold)
    if args.min_block_words is not None:
        cfg = replace(cfg, min_block_words=args.min_block_words)
    if args.min_block_chars is not None:
        cfg = replace(cfg, min_block_chars=args.min_block_chars)
    if args.max_list_lines is not None:
        cfg = replace(cfg, max_list_lines=args.max_list_lines)

    keep_list_blocks = _maybe_bool(args.keep_list_blocks)
    if keep_list_blocks is not None:
        cfg = replace(cfg, keep_list_blocks=keep_list_blocks)

    require_pass_label = _maybe_bool(args.require_pass_label)
    if require_pass_label is not None:
        cfg = replace(cfg, require_pass_label=require_pass_label)

    pass_labels = _parse_pass_labels(args.pass_labels)
    if pass_labels is not None:
        cfg = replace(cfg, pass_labels=pass_labels)

    if args.quality_model_path is not None:
        cfg = replace(cfg, quality_model_path=str(args.quality_model_path))
    if args.fasttext_repo_id is not None:
        cfg = replace(cfg, fasttext_repo_id=args.fasttext_repo_id)
    if args.fasttext_filename is not None:
        cfg = replace(cfg, fasttext_filename=args.fasttext_filename)

    return cfg


def _preview(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _add_reservoir_sample(
    reservoir: list[BlockSample],
    sample: BlockSample,
    *,
    sample_size: int,
    seen_count: int,
    rng: random.Random,
) -> None:
    if sample_size <= 0:
        return
    if len(reservoir) < sample_size:
        reservoir.append(sample)
        return
    replace_idx = rng.randrange(seen_count)
    if replace_idx < sample_size:
        reservoir[replace_idx] = sample


def _build_sample(
    block: TextBlock,
    result: Any,
    *,
    url: str | None,
    source_domain: str | None,
    preview_chars: int,
) -> BlockSample:
    return BlockSample(
        decision=result.decision,
        reason=result.reason,
        quality_label=result.quality_label,
        quality_score=result.quality_score,
        doc_id=block.doc_id,
        block_id=block.block_id,
        source_domain=source_domain,
        url=url,
        word_count=result.word_count,
        char_count=result.char_count,
        text_preview=_preview(block.text, preview_chars),
    )


def _print_samples(title: str, samples: list[BlockSample]) -> None:
    print(title)
    if not samples:
        print("  (none)")
        print("")
        return

    for idx, sample in enumerate(samples, start=1):
        score_text = f"{sample.quality_score:.4f}" if sample.quality_score is not None else "None"
        print(
            f"  [{idx}] decision={sample.decision.value} reason={sample.reason} "
            f"label={sample.quality_label} score={score_text}"
        )
        print(
            f"      doc_id={sample.doc_id} block_id={sample.block_id} "
            f"words={sample.word_count} chars={sample.char_count}"
        )
        print(f"      domain={sample.source_domain} url={sample.url}")
        print(f"      text={sample.text_preview}")
        print("")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.max_docs <= 0:
        raise ValueError("--max_docs must be > 0")
    if args.preview_chars <= 0:
        raise ValueError("--preview_chars must be > 0")
    if args.pass_samples < 0:
        raise ValueError("--pass_samples must be >= 0")
    if args.fail_samples < 0:
        raise ValueError("--fail_samples must be >= 0")

    rng = random.Random(args.seed)
    qcfg = _merge_quality_config(args)
    model = _DummyFastTextModel() if args.use_dummy_model else None
    quality_filter = QualityFilter(config=qcfg, model=model)

    split_cfg = BlockSplitConfig(
        min_block_words=args.split_min_block_words,
        min_block_chars=args.split_min_block_chars,
    )

    docs_seen = 0
    blocks_seen = 0
    pass_seen = 0
    fail_seen = 0
    reason_counts: dict[str, int] = {}

    pass_samples: list[BlockSample] = []
    fail_samples: list[BlockSample] = []

    for doc in iter_source_documents(args.input_json, skip_invalid=True):
        docs_seen += 1
        if docs_seen > args.max_docs:
            break

        blocks = split_document(doc, config=split_cfg)
        for block in blocks:
            blocks_seen += 1
            result = quality_filter.evaluate_block(block)
            sample = _build_sample(
                block,
                result,
                url=doc.effective_url,
                source_domain=doc.source_domain,
                preview_chars=args.preview_chars,
            )

            if result.passed:
                pass_seen += 1
                _add_reservoir_sample(
                    pass_samples,
                    sample,
                    sample_size=args.pass_samples,
                    seen_count=pass_seen,
                    rng=rng,
                )
            else:
                fail_seen += 1
                reason = result.reason or "unknown"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                _add_reservoir_sample(
                    fail_samples,
                    sample,
                    sample_size=args.fail_samples,
                    seen_count=fail_seen,
                    rng=rng,
                )

    keep_ratio = (pass_seen / blocks_seen) if blocks_seen else 0.0
    fail_ratio = (fail_seen / blocks_seen) if blocks_seen else 0.0

    print("=== Quality Filter Sanity Check ===")
    print(f"input_json: {args.input_json}")
    print(f"docs_scanned: {docs_seen}")
    print(f"blocks_scanned: {blocks_seen}")
    print(f"passed: {pass_seen} ({keep_ratio:.2%})")
    print(f"failed: {fail_seen} ({fail_ratio:.2%})")
    print(f"use_dummy_model: {args.use_dummy_model}")
    print(f"quality_threshold: {qcfg.quality_threshold}")
    print(f"pass_labels: {qcfg.pass_labels}")
    print(f"require_pass_label: {qcfg.require_pass_label}")
    print("")

    if reason_counts:
        print("Top fail reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
            print(f"  - {reason}: {count}")
        print("")

    _print_samples("Pass samples:", pass_samples)
    _print_samples("Fail samples:", fail_samples)

    if not pass_samples:
        print("No pass samples found. Try lowering --quality_threshold or relaxing rules.")
    if not fail_samples:
        print("No fail samples found. Try increasing --quality_threshold or tightening rules.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

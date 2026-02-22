"""Simple sanity check for block classification.

Loads random docs, splits into blocks, runs fastText language classification.
"""

from __future__ import annotations

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
from dataclasses import replace
from pathlib import Path
import random

from data.chunker.block_split import BlockSplitConfig, split_document
from data.chunker.config import load_config
from data.chunker.io import iter_source_documents
from data.chunker.quality_filter import QualityFilter, QualityFilterConfig
from data.chunker.types import SourceDocument


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample random docs and inspect block classification.",
    )
    parser.add_argument("--input_json", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_docs", type=int, default=3, help="Number of random docs to inspect.")
    parser.add_argument(
        "--show_blocks_per_doc",
        type=int,
        default=6,
        help="How many random blocks to display per sampled doc.",
    )
    parser.add_argument(
        "--pass_samples",
        type=int,
        default=5,
        help="Number of passing blocks to keep and print.",
    )
    parser.add_argument(
        "--fail_samples",
        type=int,
        default=5,
        help="Number of failing blocks to keep and print.",
    )
    parser.add_argument("--preview_chars", type=int, default=180)

    # split options
    parser.add_argument("--split_min_block_words", type=int, default=0)
    parser.add_argument("--split_max_block_words", type=int, default=150)
    parser.add_argument("--split_min_block_chars", type=int, default=0)

    # classifier overrides
    parser.add_argument("--quality_model_path", type=Path, default=None)
    parser.add_argument("--model_url", type=str, default=None)
    parser.add_argument("--model_filename", type=str, default=None)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--keep_labels", type=str, default=None, help="Comma-separated labels, e.g. en,fr")
    parser.add_argument("--min_score", type=float, default=None)
    return parser.parse_args(argv)


def _sample_random_docs(
    input_json: Path,
    *,
    num_docs: int,
    rng: random.Random,
) -> tuple[list[SourceDocument], int]:
    reservoir: list[SourceDocument] = []
    seen = 0
    for doc in iter_source_documents(input_json, skip_invalid=True):
        seen += 1
        if len(reservoir) < num_docs:
            reservoir.append(doc)
            continue
        idx = rng.randrange(seen)
        if idx < num_docs:
            reservoir[idx] = doc
    return reservoir, seen


def _parse_keep_labels(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    labels = tuple(part.strip() for part in raw.split(",") if part.strip())
    return labels or None


def _build_filter_config(args: argparse.Namespace) -> QualityFilterConfig:
    if args.config:
        chunker_cfg = load_config(args.config)
        min_score = (
            chunker_cfg.quality_threshold
            if chunker_cfg.quality_threshold is not None
            else (chunker_cfg.lang_threshold if chunker_cfg.lang_threshold is not None else 0.5)
        )
        cfg = QualityFilterConfig(
            quality_model_path=chunker_cfg.quality_model_path or chunker_cfg.lang_model_path,
            keep_labels=(chunker_cfg.lang or "en",),
            min_score=min_score,
        )
    else:
        cfg = QualityFilterConfig()

    if args.quality_model_path is not None:
        cfg = replace(cfg, quality_model_path=str(args.quality_model_path))
    if args.model_url is not None:
        cfg = replace(cfg, model_url=args.model_url)
    if args.model_filename is not None:
        cfg = replace(cfg, model_filename=args.model_filename)
    if args.cache_dir is not None:
        cfg = replace(cfg, cache_dir=str(args.cache_dir))
    keep_labels = _parse_keep_labels(args.keep_labels)
    if keep_labels is not None:
        cfg = replace(cfg, keep_labels=keep_labels)
    if args.min_score is not None:
        cfg = replace(cfg, min_score=args.min_score)
    return cfg


def _full_block_text(text: str) -> str:
    return str(text).strip()


def _add_reservoir_sample(
    reservoir: list[dict[str, object]],
    sample: dict[str, object],
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


def _print_samples(title: str, samples: list[dict[str, object]]) -> None:
    print(title)
    if not samples:
        print("  (none)")
        print("")
        return
    for i, sample in enumerate(samples, start=1):
        score = sample["score"]
        score_str = f"{score:.4f}" if isinstance(score, float) else "None"
        print(
            f"  [{i}] doc_id={sample['doc_id']} block_id={sample['block_id']} "
            f"label={sample['label']} score={score_str} reason={sample['reason']}"
        )
        print(f"      url={sample['url']}")
        print(f"      source_domain={sample['source_domain']}")
        print("      text:")
        print(sample["text"])
        print("")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_docs <= 0:
        raise ValueError("--num_docs must be > 0")
    if args.show_blocks_per_doc < 0:
        raise ValueError("--show_blocks_per_doc must be >= 0")
    if args.pass_samples < 0:
        raise ValueError("--pass_samples must be >= 0")
    if args.fail_samples < 0:
        raise ValueError("--fail_samples must be >= 0")
    if args.preview_chars <= 0:
        raise ValueError("--preview_chars must be > 0")

    rng = random.Random(args.seed)
    docs, total_docs_seen = _sample_random_docs(args.input_json, num_docs=args.num_docs, rng=rng)
    if not docs:
        print(f"No valid documents found in {args.input_json}")
        return 1

    split_cfg = BlockSplitConfig(
        min_block_words=args.split_min_block_words,
        max_block_words=args.split_max_block_words,
        min_block_chars=args.split_min_block_chars,
    )
    filter_cfg = _build_filter_config(args)
    quality_filter = QualityFilter(filter_cfg)

    print("=== Quality Filter Sanity Check ===")
    print(f"input_json: {args.input_json}")
    print(f"total_docs_seen: {total_docs_seen}")
    print(f"sampled_docs: {len(docs)}")
    print(f"keep_labels: {filter_cfg.keep_labels}")
    print(f"min_score: {filter_cfg.min_score}")
    print("")

    total_blocks = 0
    total_pass = 0
    total_fail = 0
    seen_pass_for_sampling = 0
    seen_fail_for_sampling = 0
    pass_samples: list[dict[str, object]] = []
    fail_samples: list[dict[str, object]] = []

    for doc_index, doc in enumerate(docs, start=1):
        blocks = split_document(doc, config=split_cfg)
        results = [quality_filter.evaluate_block(block) for block in blocks]
        passed = sum(1 for result in results if result.passed)
        failed = len(results) - passed

        total_blocks += len(results)
        total_pass += passed
        total_fail += failed

        print(f"--- Document {doc_index} ---")
        print(f"doc_id: {doc.doc_id}")
        print(f"url: {doc.effective_url}")
        print(f"title: {doc.title}")
        print(f"source_domain: {doc.source_domain}")
        print(f"blocks: {len(results)} (pass={passed}, fail={failed})")

        show_n = min(args.show_blocks_per_doc, len(blocks))
        block_indices = list(range(len(blocks)))
        rng.shuffle(block_indices)
        chosen = block_indices[:show_n]

        for idx in chosen:
            block = blocks[idx]
            result = results[idx]
            score_str = f"{result.quality_score:.4f}" if result.quality_score is not None else "None"
            print(
                f"  - block_id={block.block_id} pass={result.passed} "
                f"label={result.quality_label} score={score_str} reason={result.reason}"
            )
            print("    text:")
            print(_full_block_text(block.text))

        for block, result in zip(blocks, results):
            sample = {
                "doc_id": doc.doc_id,
                "block_id": block.block_id,
                "label": result.quality_label,
                "score": result.quality_score,
                "reason": result.reason,
                "url": doc.effective_url,
                "source_domain": doc.source_domain,
                "text": _full_block_text(block.text),
            }
            if result.passed:
                seen_pass_for_sampling += 1
                _add_reservoir_sample(
                    pass_samples,
                    sample,
                    sample_size=args.pass_samples,
                    seen_count=seen_pass_for_sampling,
                    rng=rng,
                )
            else:
                seen_fail_for_sampling += 1
                _add_reservoir_sample(
                    fail_samples,
                    sample,
                    sample_size=args.fail_samples,
                    seen_count=seen_fail_for_sampling,
                    rng=rng,
                )
        print("")

    keep_ratio = (total_pass / total_blocks) if total_blocks else 0.0
    print("=== Summary ===")
    print(f"blocks_total: {total_blocks}")
    print(f"blocks_pass: {total_pass}")
    print(f"blocks_fail: {total_fail}")
    print(f"pass_ratio: {keep_ratio:.2%}")
    print("")

    _print_samples("=== Pass Block Samples ===", pass_samples)
    _print_samples("=== Fail Block Samples ===", fail_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

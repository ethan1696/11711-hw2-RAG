"""Sanity-check utility for visually inspecting block splitting.

Example:
  python data/chunker/sanity_check_split.py \
    --input_json data/crawled_output_assignment/parsed/docs.jsonl \
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
from pathlib import Path
import random

from data.chunker.block_split import BlockSplitConfig, split_text
from data.chunker.io import iter_source_documents
from data.chunker.types import SourceDocument


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample text and inspect block splitting behavior.",
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="Path to source docs (.jsonl or .json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling.",
    )
    parser.add_argument(
        "--max_excerpt_chars",
        type=int,
        default=2400,
        help="Maximum characters in sampled excerpt.",
    )
    parser.add_argument(
        "--min_excerpt_chars",
        type=int,
        default=700,
        help="Minimum characters in sampled excerpt when possible.",
    )
    parser.add_argument(
        "--min_block_words",
        type=int,
        default=0,
        help="Optional minimum words per block (0 disables this threshold).",
    )
    parser.add_argument(
        "--max_block_words",
        type=int,
        default=150,
        help="Splitter config: maximum words per block.",
    )
    parser.add_argument(
        "--min_block_chars",
        type=int,
        default=0,
        help="Optional minimum chars per block (0 disables this threshold).",
    )
    return parser.parse_args(argv)


def reservoir_sample_one_doc(input_path: Path, rng: random.Random) -> SourceDocument | None:
    selected: SourceDocument | None = None
    seen = 0

    for doc in iter_source_documents(input_path, skip_invalid=True):
        seen += 1
        if rng.randrange(seen) == 0:
            selected = doc

    return selected


def sample_excerpt(
    text: str,
    *,
    rng: random.Random,
    min_chars: int,
    max_chars: int,
) -> tuple[str, int, int]:
    value = text.strip()
    if not value:
        return "", 0, 0

    min_chars = max(1, min_chars)
    max_chars = max(min_chars, max_chars)

    total = len(value)
    if total <= max_chars:
        return value, 0, total

    target = rng.randint(min_chars, max_chars)
    target = min(target, total)

    start = rng.randint(0, total - target)
    end = start + target

    # Light boundary alignment for readability.
    left = value.rfind("\n\n", max(0, start - 120), start + 1)
    if left != -1:
        start = left + 2

    right = value.find("\n\n", end, min(total, end + 120))
    if right != -1:
        end = right

    excerpt = value[start:end].strip()
    if not excerpt:
        excerpt = value[start : min(total, start + target)].strip()
        end = start + len(excerpt)

    return excerpt, start, end


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.min_excerpt_chars <= 0:
        raise ValueError("--min_excerpt_chars must be > 0")
    if args.max_excerpt_chars <= 0:
        raise ValueError("--max_excerpt_chars must be > 0")
    if args.max_excerpt_chars < args.min_excerpt_chars:
        raise ValueError("--max_excerpt_chars must be >= --min_excerpt_chars")

    rng = random.Random(args.seed)
    doc = reservoir_sample_one_doc(args.input_json, rng)
    if doc is None:
        print(f"No valid documents found in {args.input_json}")
        return 1

    excerpt, start, end = sample_excerpt(
        doc.text,
        rng=rng,
        min_chars=args.min_excerpt_chars,
        max_chars=args.max_excerpt_chars,
    )

    split_cfg = BlockSplitConfig(
        min_block_words=args.min_block_words,
        max_block_words=args.max_block_words,
        min_block_chars=args.min_block_chars,
    )
    blocks = split_text(excerpt, config=split_cfg)

    print("=== Sampled Document ===")
    print(f"doc_id: {doc.doc_id}")
    print(f"url: {doc.effective_url}")
    print(f"title: {doc.title}")
    print(f"source_domain: {doc.source_domain}")
    print(f"doc_chars: {doc.char_count}")
    print(f"doc_words: {doc.word_count}")
    print(f"excerpt_range: [{start}:{end}]")
    print(f"excerpt_chars: {len(excerpt)}")
    print(f"excerpt_words: {len(excerpt.split())}")
    print("")

    print("=== Original Excerpt ===")
    print(excerpt)
    print("")

    print(f"=== Split Blocks ({len(blocks)}) ===")
    for idx, block in enumerate(blocks):
        words = len(block.split())
        chars = len(block)
        print(f"[Block {idx}] words={words} chars={chars}")
        print(block)
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build and cache a BM25 index from chunk JSONL.

Outputs under ``output_dir``:
- ``bm25_index.pkl``: pickled BM25 object + row-to-chunk mapping + config
- ``bm25_stats.json``: run summary
- optional ``bm25_chunk_store.jsonl`` copy of payload rows

Notes:
- If input rows already contain ``chunk_uid`` (for example embedding ``chunk_store.jsonl``),
  those IDs are reused exactly.
- This script avoids overwriting embedding artifacts like ``chunk_store.jsonl`` and ``stats.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
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

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from retrieval.embed.utils import build_chunk_payload, iter_jsonl, make_chunk_uid, normalize_text


LOGGER = logging.getLogger(__name__)
DEFAULT_TOKEN_PATTERN = r"[A-Za-z0-9_]+"


@dataclass(frozen=True, slots=True)
class BuildStats:
    input_rows: int
    indexed_rows: int
    skipped_empty_text: int
    skipped_empty_tokens: int
    max_chunks: int | None

    def to_json(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "indexed_rows": self.indexed_rows,
            "skipped_empty_text": self.skipped_empty_text,
            "skipped_empty_tokens": self.skipped_empty_tokens,
            "max_chunks": self.max_chunks,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and store BM25 index from chunk JSONL.")
    parser.add_argument("--input_chunks", type=Path, required=True, help="Chunk JSONL input.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Optional cap on number of chunks to index.",
    )
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="BM25 epsilon parameter.")
    parser.add_argument(
        "--token_pattern",
        type=str,
        default=DEFAULT_TOKEN_PATTERN,
        help="Regex used to tokenize text.",
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lowercase text before tokenization.",
    )
    parser.add_argument(
        "--min_token_len",
        type=int,
        default=1,
        help="Drop tokens shorter than this length.",
    )
    parser.add_argument(
        "--dedup_repeated_lines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply repeated-line dedup during text normalization.",
    )
    parser.add_argument(
        "--output_index_name",
        type=str,
        default="bm25_index.pkl",
        help="Output filename for serialized BM25 index.",
    )
    parser.add_argument(
        "--output_stats_name",
        type=str,
        default="bm25_stats.json",
        help="Output filename for BM25 build stats.",
    )
    parser.add_argument(
        "--write_chunk_store_copy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write a BM25-specific chunk store copy under output_dir.",
    )
    parser.add_argument(
        "--chunk_store_copy_name",
        type=str,
        default="bm25_chunk_store.jsonl",
        help="Filename for BM25 chunk store copy when enabled.",
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


def _estimate_total_rows(path: Path, *, max_chunks: int | None) -> int | None:
    if max_chunks is not None:
        return max_chunks
    if path.suffix.lower() != ".jsonl":
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                count += 1
    return count


def _compile_token_pattern(token_pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(token_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid --token_pattern regex: {token_pattern!r}") from exc


def _tokenize(
    text: str,
    *,
    token_re: re.Pattern[str],
    lowercase: bool,
    min_token_len: int,
) -> list[str]:
    value = text.lower() if lowercase else text
    tokens = token_re.findall(value)
    if min_token_len > 1:
        return [tok for tok in tokens if len(tok) >= min_token_len]
    return tokens


def _write_chunk_store(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _resolve_chunk_uid(raw_row: dict[str, Any], normalized_row: dict[str, Any]) -> str:
    existing = raw_row.get("chunk_uid")
    if existing is not None:
        value = str(existing).strip()
        if value:
            return value
    return make_chunk_uid(normalized_row)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("--max_chunks must be > 0 when set")
    if args.k1 <= 0:
        raise ValueError("--k1 must be > 0")
    if not (0.0 <= args.b <= 1.0):
        raise ValueError("--b must be between 0 and 1")
    if args.epsilon < 0:
        raise ValueError("--epsilon must be >= 0")
    if args.min_token_len <= 0:
        raise ValueError("--min_token_len must be > 0")

    token_re = _compile_token_pattern(args.token_pattern)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_index_name = str(args.output_index_name).strip()
    output_stats_name = str(args.output_stats_name).strip()
    chunk_store_copy_name = str(args.chunk_store_copy_name).strip()
    if not output_index_name:
        raise ValueError("--output_index_name cannot be empty")
    if not output_stats_name:
        raise ValueError("--output_stats_name cannot be empty")
    if args.write_chunk_store_copy and not chunk_store_copy_name:
        raise ValueError("--chunk_store_copy_name cannot be empty when --write_chunk_store_copy is set")

    index_path = output_dir / output_index_name
    stats_path = output_dir / output_stats_name
    chunk_store_copy_path = output_dir / chunk_store_copy_name if args.write_chunk_store_copy else None

    input_rows = 0
    indexed_rows = 0
    skipped_empty_text = 0
    skipped_empty_tokens = 0

    chunk_uids: list[str] = []
    tokenized_corpus: list[list[str]] = []
    chunk_payloads: list[dict[str, Any]] = []

    total_rows = _estimate_total_rows(args.input_chunks, max_chunks=args.max_chunks)
    progress = tqdm(total=total_rows, desc="Building BM25", unit="chunk")
    try:
        for row in iter_jsonl(args.input_chunks):
            if args.max_chunks is not None and input_rows >= args.max_chunks:
                break

            input_rows += 1
            progress.update(1)

            normalized_text = normalize_text(
                str(row.get("text", "")),
                dedup_repeated_lines=args.dedup_repeated_lines,
            )
            if not normalized_text:
                skipped_empty_text += 1
                continue

            tokens = _tokenize(
                normalized_text,
                token_re=token_re,
                lowercase=bool(args.lowercase),
                min_token_len=int(args.min_token_len),
            )
            if not tokens:
                skipped_empty_tokens += 1
                continue

            normalized_row = dict(row)
            normalized_row["text"] = normalized_text

            chunk_uid = _resolve_chunk_uid(row, normalized_row)
            chunk_uids.append(chunk_uid)
            tokenized_corpus.append(tokens)
            chunk_payloads.append(build_chunk_payload(normalized_row, chunk_uid=chunk_uid))
            indexed_rows += 1

            progress.set_postfix(indexed=indexed_rows, refresh=False)
    finally:
        progress.close()

    if indexed_rows == 0:
        raise ValueError("No chunks were indexed. Check input and tokenization settings.")

    LOGGER.info(
        "Fitting BM25: rows=%d k1=%.3f b=%.3f epsilon=%.3f",
        indexed_rows,
        args.k1,
        args.b,
        args.epsilon,
    )
    bm25 = BM25Okapi(tokenized_corpus, k1=args.k1, b=args.b, epsilon=args.epsilon)

    serialized = {
        "format": "bm25_okapi_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_chunks": str(args.input_chunks),
        "bm25_params": {
            "k1": float(args.k1),
            "b": float(args.b),
            "epsilon": float(args.epsilon),
        },
        "tokenizer": {
            "pattern": args.token_pattern,
            "lowercase": bool(args.lowercase),
            "min_token_len": int(args.min_token_len),
            "dedup_repeated_lines": bool(args.dedup_repeated_lines),
        },
        "chunk_uids": chunk_uids,
        "bm25": bm25,
    }
    with index_path.open("wb") as handle:
        pickle.dump(serialized, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if chunk_store_copy_path is not None:
        _write_chunk_store(chunk_store_copy_path, chunk_payloads)

    stats = BuildStats(
        input_rows=input_rows,
        indexed_rows=indexed_rows,
        skipped_empty_text=skipped_empty_text,
        skipped_empty_tokens=skipped_empty_tokens,
        max_chunks=args.max_chunks,
    )
    summary = {
        **stats.to_json(),
        "bm25_params": {
            "k1": float(args.k1),
            "b": float(args.b),
            "epsilon": float(args.epsilon),
        },
        "tokenizer": {
            "pattern": args.token_pattern,
            "lowercase": bool(args.lowercase),
            "min_token_len": int(args.min_token_len),
            "dedup_repeated_lines": bool(args.dedup_repeated_lines),
        },
        "paths": {
            "index": str(index_path),
            "stats": str(stats_path),
            "chunk_store_copy": str(chunk_store_copy_path) if chunk_store_copy_path else None,
        },
    }
    stats_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

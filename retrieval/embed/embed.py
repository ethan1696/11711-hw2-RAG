from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
from tqdm import tqdm

from retrieval.embed.storage import Writer
from retrieval.embed.utils import (
    batch_iter,
    build_chunk_payload,
    format_passage,
    iter_jsonl,
    make_chunk_uid,
    normalize_text,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"


@dataclass(frozen=True, slots=True)
class EmbedStats:
    input_rows: int = 0
    embedded_rows: int = 0
    skipped_empty_text: int = 0
    failed_rows: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "embedded_rows": self.embedded_rows,
            "skipped_empty_text": self.skipped_empty_text,
            "failed_rows": self.failed_rows,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunk JSONL for dense retrieval.")

    parser.add_argument("--input_chunks", type=Path, required=True, help="Path to chunk JSONL.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory.")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Optional cap on number of input chunks to embed.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string, e.g. auto, cpu, cuda:0, cuda:1.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Model compute dtype.",
    )

    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize output embeddings.",
    )
    parser.add_argument(
        "--dedup_repeated_lines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deduplicate repeated non-empty lines during text normalization.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to tokenizer/model loading.",
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


def _resolve_device(device_arg: str) -> str:
    import torch

    raw = str(device_arg).strip().lower()
    if not raw or raw == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(raw)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but no CUDA device is available")
        if device.index is not None and (device.index < 0 or device.index >= torch.cuda.device_count()):
            raise ValueError(
                f"Requested CUDA device index {device.index} is out of range "
                f"(available: 0..{torch.cuda.device_count() - 1})"
            )

    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS requested but not available")

    return str(device)


def _resolve_torch_dtype(dtype_arg: str, *, resolved_device: str) -> torch.dtype:
    import torch

    key = str(dtype_arg).strip().lower()
    if key == "auto":
        if resolved_device.startswith("cuda"):
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[key]


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool final-token embeddings for both left/right padding tokenization."""

    import torch

    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _encode_passages(
    *,
    model: Any,
    tokenizer: Any,
    passages: list[str],
    max_length: int,
    device: str,
    normalize: bool,
) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(
        passages,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def _write_empty_outputs(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.npy"
    id_map_path = output_dir / "embedding_to_uid.jsonl"
    chunk_store_path = output_dir / "chunk_store.jsonl"
    stats_path = output_dir / "stats.json"

    np.save(embeddings_path, np.empty((0, 0), dtype=np.float32))
    id_map_path.write_text("", encoding="utf-8")
    chunk_store_path.write_text("", encoding="utf-8")

    summary = {
        "num_rows": 0,
        "dim": 0,
        "num_unique_chunks": 0,
        "paths": {
            "embeddings": str(embeddings_path),
            "embedding_to_uid": str(id_map_path),
            "chunk_store": str(chunk_store_path),
            "stats": str(stats_path),
        },
    }
    stats_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from itertools import islice

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.max_length <= 0:
        raise ValueError("--max_length must be > 0")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("--max_chunks must be > 0 when set")

    resolved_device = _resolve_device(args.device)
    torch_dtype = _resolve_torch_dtype(args.dtype, resolved_device=resolved_device)

    LOGGER.info(
        "Loading model/tokenizer: model=%s device=%s dtype=%s",
        args.model,
        resolved_device,
        str(torch_dtype).replace("torch.", ""),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model.to(resolved_device)
    model.eval()

    writer: Writer | None = None
    stats = EmbedStats()

    input_rows = 0
    embedded_rows = 0
    skipped_empty_text = 0
    failed_rows = 0

    chunk_iter = iter_jsonl(args.input_chunks)
    if args.max_chunks is not None:
        chunk_iter = islice(chunk_iter, args.max_chunks)

    progress = tqdm(
        total=args.max_chunks,
        desc="Embedding chunks",
        unit="chunk",
    )
    try:
        for batch in batch_iter(chunk_iter, args.batch_size):
            input_rows += len(batch)
            progress.update(len(batch))

            chunk_uids: list[str] = []
            passage_texts: list[str] = []
            payloads: list[dict[str, Any]] = []

            for record in batch:
                normalized_text = normalize_text(
                    str(record.get("text", "")),
                    dedup_repeated_lines=args.dedup_repeated_lines,
                )
                if not normalized_text:
                    skipped_empty_text += 1
                    continue

                normalized_record = dict(record)
                normalized_record["text"] = normalized_text

                chunk_uid = make_chunk_uid(normalized_record)
                chunk_uids.append(chunk_uid)
                passage_texts.append(format_passage(normalized_text))
                payloads.append(build_chunk_payload(normalized_record, chunk_uid=chunk_uid))

            if not passage_texts:
                progress.set_postfix(
                    embedded=embedded_rows,
                    skipped=skipped_empty_text,
                    failed=failed_rows,
                    refresh=False,
                )
                continue

            try:
                embeddings = _encode_passages(
                    model=model,
                    tokenizer=tokenizer,
                    passages=passage_texts,
                    max_length=args.max_length,
                    device=resolved_device,
                    normalize=args.normalize,
                )

                if writer is None:
                    writer = Writer(output_dir=args.output_dir, dim=int(embeddings.shape[1]))

                writer.append(chunk_uids=chunk_uids, embeddings=embeddings, chunk_payloads=payloads)
                embedded_rows += len(chunk_uids)
            except Exception:
                failed_rows += len(chunk_uids)
                LOGGER.exception("Failed to embed batch with %d rows", len(chunk_uids))
            finally:
                progress.set_postfix(
                    embedded=embedded_rows,
                    skipped=skipped_empty_text,
                    failed=failed_rows,
                    refresh=False,
                )
    finally:
        progress.close()

    stats = EmbedStats(
        input_rows=input_rows,
        embedded_rows=embedded_rows,
        skipped_empty_text=skipped_empty_text,
        failed_rows=failed_rows,
    )

    storage_summary = writer.close() if writer is not None else _write_empty_outputs(args.output_dir)

    run_summary = {
        **stats.to_json(),
        "model": args.model,
        "device": resolved_device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_chunks": args.max_chunks,
        "normalize": bool(args.normalize),
        "storage": storage_summary,
    }

    embed_stats_path = Path(args.output_dir) / "embed_stats.json"
    embed_stats_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    LOGGER.info("Embedding complete: %s", json.dumps(run_summary, sort_keys=True))
    return run_summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.log_level)

    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

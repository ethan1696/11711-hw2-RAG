"""Simple dense retriever over precomputed chunk embeddings.

Expected artifacts in ``embed_dir`` (produced by ``retrieval/embed/embed.py``):
- ``embeddings.npy``
- ``embedding_to_uid.jsonl``
- ``chunk_store.jsonl``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np

from retrieval.embed.utils import DEFAULT_QUERY_TASK, format_query, iter_jsonl, normalize_text


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"


@dataclass(frozen=True, slots=True)
class RetrieverPaths:
    """Resolved retrieval artifact paths."""

    embed_dir: Path
    embeddings_path: Path
    embedding_to_uid_path: Path
    chunk_store_path: Path


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


def _resolve_torch_dtype(dtype_arg: str, *, resolved_device: str):
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


def _tensor_to_numpy_2d(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach") and hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
        value = tensor.detach().cpu()
        try:
            array = value.numpy()
        except TypeError as exc:
            if "BFloat16" not in str(exc):
                raise
            array = value.float().numpy()
    else:
        array = np.asarray(tensor)

    if array.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {array.shape}")
    return array


def last_token_pool(last_hidden_states, attention_mask):
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


class DenseRetriever:
    """In-memory dense retriever using exact nearest-neighbor search."""

    def __init__(
        self,
        *,
        embed_dir: str | Path,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: str = "auto",
        metric: str = "cosine",
        max_length: int = 8192,
        normalize_query: bool = True,
        trust_remote_code: bool = True,
        task_description: str = DEFAULT_QUERY_TASK,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        metric_value = str(metric).strip().lower()
        if metric_value not in {"cosine", "ip", "l2"}:
            raise ValueError("metric must be one of: cosine, ip, l2")
        if max_length <= 0:
            raise ValueError("max_length must be > 0")

        self.paths = self._resolve_paths(embed_dir)
        self.embeddings = self._load_embeddings(self.paths.embeddings_path)
        self.row_to_uid = self._load_row_to_uid(
            self.paths.embedding_to_uid_path,
            expected_rows=self.embeddings.shape[0],
        )
        self.chunk_store = self._load_chunk_store(self.paths.chunk_store_path)

        self.metric = metric_value
        self.max_length = int(max_length)
        self.normalize_query = bool(normalize_query)
        self.task_description = str(task_description)

        if self.metric == "cosine":
            self.embeddings = self._l2_normalize(self.embeddings)

        if self.metric == "l2":
            self._embedding_sq_norms = np.sum(self.embeddings * self.embeddings, axis=1)
        else:
            self._embedding_sq_norms = None

        self.device = _resolve_device(device)
        self.torch_dtype = _resolve_torch_dtype(dtype, resolved_device=self.device)

        LOGGER.info(
            "Loading retriever model/tokenizer: model=%s device=%s dtype=%s",
            model,
            self.device,
            str(self.torch_dtype).replace("torch.", ""),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        if self.embeddings.ndim != 2:
            return 0
        return int(self.embeddings.shape[1])

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return ranked retrieval results for ``query``."""

        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.size == 0:
            return []

        normalized_query = normalize_text(query)
        if not normalized_query:
            return []

        query_text = format_query(normalized_query, task_description=self.task_description)
        query_vector = self._encode_query_vector(query_text)

        if self.metric == "cosine":
            query_vector = self._l2_normalize(query_vector[np.newaxis, :])[0]
        elif self.metric in {"ip", "l2"} and self.normalize_query:
            query_vector = self._l2_normalize(query_vector[np.newaxis, :])[0]

        k = min(int(top_k), self.size)

        if self.metric in {"cosine", "ip"}:
            scores = self.embeddings @ query_vector
            top_indices = self._topk_desc(scores, k)
            return self._build_results(top_indices=top_indices, scores=scores)

        # L2 distance mode (smaller is better).
        query_sq_norm = float(np.dot(query_vector, query_vector))
        distances = self._embedding_sq_norms + query_sq_norm - (2.0 * (self.embeddings @ query_vector))
        top_indices = self._topk_asc(distances, k)
        return self._build_results(top_indices=top_indices, distances=distances)

    @staticmethod
    def _topk_desc(values: np.ndarray, k: int) -> np.ndarray:
        if k >= values.shape[0]:
            return np.argsort(-values)
        selected = np.argpartition(-values, kth=k - 1)[:k]
        return selected[np.argsort(-values[selected])]

    @staticmethod
    def _topk_asc(values: np.ndarray, k: int) -> np.ndarray:
        if k >= values.shape[0]:
            return np.argsort(values)
        selected = np.argpartition(values, kth=k - 1)[:k]
        return selected[np.argsort(values[selected])]

    def _build_results(
        self,
        *,
        top_indices: np.ndarray,
        scores: np.ndarray | None = None,
        distances: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for rank, row_id in enumerate(top_indices.tolist(), start=1):
            row_idx = int(row_id)
            chunk_uid = self.row_to_uid[row_idx]
            payload = dict(self.chunk_store.get(chunk_uid, {}))

            if scores is not None:
                metric_value = float(scores[row_idx])
                score = metric_value
            else:
                metric_value = float(distances[row_idx])
                score = -metric_value

            result: dict[str, Any] = {
                "rank": rank,
                "row_id": row_idx,
                "chunk_uid": chunk_uid,
                "score": score,
                "metric": self.metric,
            }
            if distances is not None:
                result["distance"] = metric_value

            if payload:
                result.update(payload)

            results.append(result)

        return results

    def _encode_query_vector(self, query_text: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(
            [query_text],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            if self.normalize_query and self.metric in {"cosine", "ip"}:
                query_embedding = F.normalize(query_embedding, p=2, dim=1)

        matrix = _tensor_to_numpy_2d(query_embedding).astype(np.float32, copy=False)
        return matrix[0]

    @staticmethod
    def _l2_normalize(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return matrix / norms

    @staticmethod
    def _resolve_paths(embed_dir: str | Path) -> RetrieverPaths:
        root = Path(embed_dir)
        return RetrieverPaths(
            embed_dir=root,
            embeddings_path=root / "embeddings.npy",
            embedding_to_uid_path=root / "embedding_to_uid.jsonl",
            chunk_store_path=root / "chunk_store.jsonl",
        )

    @staticmethod
    def _load_embeddings(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {path}")

        matrix = np.load(path)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D embeddings matrix, got shape {matrix.shape}")
        return matrix.astype(np.float32, copy=False)

    @staticmethod
    def _load_row_to_uid(path: Path, *, expected_rows: int) -> list[str]:
        if not path.exists():
            raise FileNotFoundError(f"Missing id map file: {path}")

        row_to_uid: list[str | None] = [None] * expected_rows
        for payload in iter_jsonl(path):
            row_id = int(payload["row_id"])
            if row_id < 0 or row_id >= expected_rows:
                raise ValueError(
                    f"row_id out of range in {path}: row_id={row_id}, expected 0..{expected_rows - 1}"
                )
            chunk_uid = str(payload.get("chunk_uid", "")).strip()
            if not chunk_uid:
                raise ValueError(f"Missing chunk_uid for row_id={row_id} in {path}")
            row_to_uid[row_id] = chunk_uid

        missing = [idx for idx, value in enumerate(row_to_uid) if value is None]
        if missing:
            first_missing = missing[0]
            raise ValueError(f"Missing row mapping for row_id={first_missing} in {path}")

        return [value for value in row_to_uid if value is not None]

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple dense retrieval over embedded chunks.")

    parser.add_argument("--embed_dir", type=Path, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--metric", type=str, default="cosine", choices=("cosine", "ip", "l2"))

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
    )
    parser.add_argument(
        "--normalize_query",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--task_description",
        type=str,
        default=DEFAULT_QUERY_TASK,
        help="Instruction prefix used when formatting retrieval queries.",
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
    retriever = DenseRetriever(
        embed_dir=args.embed_dir,
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        metric=args.metric,
        max_length=args.max_length,
        normalize_query=args.normalize_query,
        trust_remote_code=args.trust_remote_code,
        task_description=args.task_description,
    )

    results = retriever.search(args.query, top_k=args.top_k)

    if not args.include_text:
        for row in results:
            row.pop("text", None)

    summary = {
        "query": args.query,
        "top_k": args.top_k,
        "num_results": len(results),
        "embed_dir": str(args.embed_dir),
        "metric": args.metric,
        "results": results,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.log_level)

    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

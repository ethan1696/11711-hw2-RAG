"""Config loading for interactive RAG debug/testing."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore

from .system import (
    DEFAULT_CONTEXT_ENTRY_TEMPLATE,
    DEFAULT_EMPTY_CONTEXT_TEXT,
    DEFAULT_QWEN_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
    RetrievalMode,
)


SUPPORTED_CONFIG_SUFFIXES = (".yaml", ".yml", ".json")


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Invalid bool for '{key}': {value!r}")


def _as_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid int for '{key}': {value!r}") from exc


def _as_float(value: Any, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for '{key}': {value!r}") from exc


def _as_str(value: Any, key: str) -> str:
    if value is None:
        raise ValueError(f"Missing required string for '{key}'")
    text = str(value)
    if not text.strip():
        raise ValueError(f"'{key}' cannot be empty")
    return text


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


@dataclass(slots=True)
class RAGDebugConfig:
    """Flat config for constructing RAGSystem and interactive debug loop."""

    dense_embed_dir: str = "retrieval/output_embed"
    sparse_index_dir: str = "retrieval/output_embed"

    dense_model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    sparse_index_name: str = "bm25_index.pkl"
    sparse_chunk_store_name: str | None = None
    dense_device: str = "auto"
    dense_dtype: str = "auto"

    llm_model_name: str = DEFAULT_QWEN_MODEL
    llm_torch_dtype: str | None = "auto"
    llm_device: str | None = None
    llm_device_map: str | None = "auto"
    trust_remote_code: bool = True

    retrieval_mode: RetrievalMode = "hybrid"
    retrieval_top_k: int = 6
    fusion_top_k: int | None = None
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    max_new_tokens: int = 512
    temperature: float = 0.0

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE
    context_entry_template: str = DEFAULT_CONTEXT_ENTRY_TEMPLATE
    empty_context_text: str = DEFAULT_EMPTY_CONTEXT_TEXT

    show_model_prompt: bool = True
    show_contexts: bool = False
    exit_commands: list[str] = field(default_factory=lambda: ["exit", "quit", ":q"])

    def __post_init__(self) -> None:
        self.dense_embed_dir = _as_str(self.dense_embed_dir, "dense_embed_dir")
        self.sparse_index_dir = _as_str(self.sparse_index_dir, "sparse_index_dir")
        self.dense_model_name = _as_str(self.dense_model_name, "dense_model_name")
        self.sparse_index_name = _as_str(self.sparse_index_name, "sparse_index_name")
        self.sparse_chunk_store_name = _as_optional_str(self.sparse_chunk_store_name)
        self.dense_device = _as_str(self.dense_device, "dense_device")
        self.dense_dtype = _as_str(self.dense_dtype, "dense_dtype")

        self.llm_model_name = _as_str(self.llm_model_name, "llm_model_name")
        self.llm_torch_dtype = _as_optional_str(self.llm_torch_dtype)
        self.llm_device = _as_optional_str(self.llm_device)
        self.llm_device_map = _as_optional_str(self.llm_device_map)

        self.retrieval_mode = str(self.retrieval_mode).strip().lower()  # type: ignore[assignment]
        if self.retrieval_mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError("retrieval_mode must be one of: dense, sparse, hybrid")
        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be > 0")
        if self.fusion_top_k is not None and self.fusion_top_k <= 0:
            raise ValueError("fusion_top_k must be > 0 when set")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("dense_weight and sparse_weight must be >= 0")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")

        if not self.exit_commands:
            self.exit_commands = ["exit", "quit", ":q"]
        self.exit_commands = [cmd.strip().lower() for cmd in self.exit_commands if cmd.strip()]
        if not self.exit_commands:
            self.exit_commands = ["exit", "quit", ":q"]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RAGDebugConfig":
        return cls(
            dense_embed_dir=str(payload.get("dense_embed_dir", "retrieval/output_embed")),
            sparse_index_dir=str(payload.get("sparse_index_dir", "retrieval/output_embed")),
            dense_model_name=str(payload.get("dense_model_name", "Alibaba-NLP/gte-Qwen2-7B-instruct")),
            sparse_index_name=str(payload.get("sparse_index_name", "bm25_index.pkl")),
            sparse_chunk_store_name=payload.get("sparse_chunk_store_name"),
            dense_device=str(payload.get("dense_device", "auto")),
            dense_dtype=str(payload.get("dense_dtype", "auto")),
            llm_model_name=str(payload.get("llm_model_name", DEFAULT_QWEN_MODEL)),
            llm_torch_dtype=payload.get("llm_torch_dtype", "auto"),
            llm_device=payload.get("llm_device"),
            llm_device_map=payload.get("llm_device_map", "auto"),
            trust_remote_code=_as_bool(payload.get("trust_remote_code", True), "trust_remote_code"),
            retrieval_mode=str(payload.get("retrieval_mode", "hybrid")),
            retrieval_top_k=_as_int(payload.get("retrieval_top_k", 6), "retrieval_top_k"),
            fusion_top_k=(
                None
                if payload.get("fusion_top_k") is None
                else _as_int(payload.get("fusion_top_k"), "fusion_top_k")
            ),
            rrf_k=_as_int(payload.get("rrf_k", 60), "rrf_k"),
            dense_weight=_as_float(payload.get("dense_weight", 1.0), "dense_weight"),
            sparse_weight=_as_float(payload.get("sparse_weight", 1.0), "sparse_weight"),
            max_new_tokens=_as_int(payload.get("max_new_tokens", 512), "max_new_tokens"),
            temperature=_as_float(payload.get("temperature", 0.0), "temperature"),
            system_prompt=str(payload.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
            user_prompt_template=str(payload.get("user_prompt_template", DEFAULT_USER_PROMPT_TEMPLATE)),
            context_entry_template=str(
                payload.get("context_entry_template", DEFAULT_CONTEXT_ENTRY_TEMPLATE)
            ),
            empty_context_text=str(payload.get("empty_context_text", DEFAULT_EMPTY_CONTEXT_TEXT)),
            show_model_prompt=_as_bool(payload.get("show_model_prompt", True), "show_model_prompt"),
            show_contexts=_as_bool(payload.get("show_contexts", False), "show_contexts"),
            exit_commands=list(payload.get("exit_commands", ["exit", "quit", ":q"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dense_embed_dir": self.dense_embed_dir,
            "sparse_index_dir": self.sparse_index_dir,
            "dense_model_name": self.dense_model_name,
            "sparse_index_name": self.sparse_index_name,
            "sparse_chunk_store_name": self.sparse_chunk_store_name,
            "dense_device": self.dense_device,
            "dense_dtype": self.dense_dtype,
            "llm_model_name": self.llm_model_name,
            "llm_torch_dtype": self.llm_torch_dtype,
            "llm_device": self.llm_device,
            "llm_device_map": self.llm_device_map,
            "trust_remote_code": self.trust_remote_code,
            "retrieval_mode": self.retrieval_mode,
            "retrieval_top_k": self.retrieval_top_k,
            "fusion_top_k": self.fusion_top_k,
            "rrf_k": self.rrf_k,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "context_entry_template": self.context_entry_template,
            "empty_context_text": self.empty_context_text,
            "show_model_prompt": self.show_model_prompt,
            "show_contexts": self.show_contexts,
            "exit_commands": self.exit_commands,
        }


def load_rag_config(path: str | Path) -> RAGDebugConfig:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix not in SUPPORTED_CONFIG_SUFFIXES:
        raise ValueError(
            f"Unsupported config suffix '{suffix}'. Supported: {SUPPORTED_CONFIG_SUFFIXES}"
        )

    if suffix == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {config_path} must be a mapping")

    return RAGDebugConfig.from_dict(payload)


__all__ = [
    "RAGDebugConfig",
    "SUPPORTED_CONFIG_SUFFIXES",
    "load_rag_config",
]

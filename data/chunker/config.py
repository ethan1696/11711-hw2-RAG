"""Chunker configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore

from .types import JSONDict, JSONValue


SUPPORTED_CONFIG_SUFFIXES = (".json", ".yaml", ".yml")
JSON_INDENT = 2


def _as_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for '{key}': {value!r}") from exc


def _as_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid int for '{key}': {value!r}") from exc


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Invalid bool for '{key}': {value!r}")


def _as_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(slots=True)
class ChunkerConfig:
    """Top-level chunker configuration."""

    input_json: str | None = None
    output_dir: str | None = None

    lang_model_path: str | None = None
    lang: str | None = "en"
    lang_threshold: float | None = 0.5

    quality_model_path: str | None = None
    quality_threshold: float | None = 0.5

    min_block_words: int = 8
    min_doc_words_after_filter: int = 30

    chunk_max_tokens: int = 220
    chunk_overlap_tokens: int = 40

    keep_list_blocks: bool = True
    max_list_lines: int = 20
    dedup: bool = True

    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.input_json = _as_str_or_none(self.input_json)
        self.output_dir = _as_str_or_none(self.output_dir)
        self.lang_model_path = _as_str_or_none(self.lang_model_path)
        self.lang = _as_str_or_none(self.lang)
        self.quality_model_path = _as_str_or_none(self.quality_model_path)

        if self.lang_threshold is not None and not (0.0 <= self.lang_threshold <= 1.0):
            raise ValueError("lang_threshold must be between 0 and 1 when set")
        if self.quality_threshold is not None and not (0.0 <= self.quality_threshold <= 1.0):
            raise ValueError("quality_threshold must be between 0 and 1 when set")

        if self.min_block_words < 0:
            raise ValueError("min_block_words must be >= 0")
        if self.min_doc_words_after_filter < 0:
            raise ValueError("min_doc_words_after_filter must be >= 0")

        if self.chunk_max_tokens <= 0:
            raise ValueError("chunk_max_tokens must be > 0")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be >= 0")
        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise ValueError("chunk_overlap_tokens must be < chunk_max_tokens")

        if self.max_list_lines <= 0:
            raise ValueError("max_list_lines must be > 0")

    @property
    def input_path(self) -> Path:
        if not self.input_json:
            raise ValueError("input_json is not set")
        return Path(self.input_json)

    @property
    def output_path(self) -> Path:
        if not self.output_dir:
            raise ValueError("output_dir is not set")
        return Path(self.output_dir)

    def validate_required_io(self) -> None:
        if not self.input_json:
            raise ValueError("ChunkerConfig requires 'input_json'")
        if not self.output_dir:
            raise ValueError("ChunkerConfig requires 'output_dir'")

    def with_overrides(
        self,
        *,
        input_json: str | None = None,
        output_dir: str | None = None,
    ) -> "ChunkerConfig":
        payload = self.to_dict()
        if input_json is not None:
            payload["input_json"] = input_json
        if output_dir is not None:
            payload["output_dir"] = output_dir
        return ChunkerConfig.from_dict(payload)

    def to_dict(self) -> JSONDict:
        return {
            "input_json": self.input_json,
            "output_dir": self.output_dir,
            "lang_model_path": self.lang_model_path,
            "lang": self.lang,
            "lang_threshold": self.lang_threshold,
            "quality_model_path": self.quality_model_path,
            "quality_threshold": self.quality_threshold,
            "min_block_words": self.min_block_words,
            "min_doc_words_after_filter": self.min_doc_words_after_filter,
            "chunk_max_tokens": self.chunk_max_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "keep_list_blocks": self.keep_list_blocks,
            "max_list_lines": self.max_list_lines,
            "dedup": self.dedup,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChunkerConfig":
        return cls(
            input_json=_as_str_or_none(payload.get("input_json")),
            output_dir=_as_str_or_none(payload.get("output_dir")),
            lang_model_path=_as_str_or_none(payload.get("lang_model_path")),
            lang=_as_str_or_none(payload.get("lang")),
            lang_threshold=_as_float(payload.get("lang_threshold", 0.5), "lang_threshold"),
            quality_model_path=_as_str_or_none(payload.get("quality_model_path")),
            quality_threshold=_as_float(
                payload.get("quality_threshold", 0.5),
                "quality_threshold",
            ),
            min_block_words=int(payload.get("min_block_words", 8)),
            min_doc_words_after_filter=int(payload.get("min_doc_words_after_filter", 30)),
            chunk_max_tokens=int(payload.get("chunk_max_tokens", 220)),
            chunk_overlap_tokens=int(payload.get("chunk_overlap_tokens", 40)),
            keep_list_blocks=_as_bool(payload.get("keep_list_blocks", True), "keep_list_blocks"),
            max_list_lines=int(payload.get("max_list_lines", 20)),
            dedup=_as_bool(payload.get("dedup", True), "dedup"),
            metadata=dict(payload.get("metadata", {})),
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config at {path} must be a mapping at top level")
    return data


def load_config(path: str | Path) -> ChunkerConfig:
    """Load chunker config from JSON/YAML file."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix not in SUPPORTED_CONFIG_SUFFIXES:
        raise ValueError(
            f"Unsupported config suffix '{suffix}'. Supported: {SUPPORTED_CONFIG_SUFFIXES}"
        )

    if suffix == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        payload = _load_yaml(config_path)

    if not isinstance(payload, dict):
        raise ValueError(f"Config at {config_path} must be a mapping")

    return ChunkerConfig.from_dict(payload)


def save_config(config: ChunkerConfig, path: str | Path) -> None:
    """Save chunker config as JSON or YAML based on output file extension."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = config.to_dict()
    suffix = out_path.suffix.lower()

    if suffix == ".json":
        out_path.write_text(
            json.dumps(payload, indent=JSON_INDENT, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    if suffix in {".yaml", ".yml"}:
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return

    raise ValueError(
        f"Unsupported config suffix '{suffix}'. Supported: {SUPPORTED_CONFIG_SUFFIXES}"
    )


def resolve_config(
    *,
    config_path: str | Path | None = None,
    input_json: str | None = None,
    output_dir: str | None = None,
) -> ChunkerConfig:
    """Resolve config with optional file base + CLI overrides."""

    base = load_config(config_path) if config_path else ChunkerConfig()
    resolved = base.with_overrides(input_json=input_json, output_dir=output_dir)
    resolved.validate_required_io()
    return resolved


__all__ = [
    "ChunkerConfig",
    "JSON_INDENT",
    "SUPPORTED_CONFIG_SUFFIXES",
    "load_config",
    "resolve_config",
    "save_config",
]

"""Per-block quality filter with fastText scoring.

This module intentionally focuses on *single block* evaluation:
given one block, return pass/fail with score + reason.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import fasttext  # type: ignore
from huggingface_hub import hf_hub_download

from .config import ChunkerConfig
from .types import FilterDecision, JSONValue, TextBlock


DEFAULT_FASTTEXT_REPO_ID = "kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2"
DEFAULT_FASTTEXT_FILENAME = "model_quantized.bin"

WORD_RE = re.compile(r"\b\w+\b")
LIST_LINE_RE = re.compile(r"^\s*(?:[-*+•]\s+|(?:\d+|[A-Za-z])[.)]\s+)")


@dataclass(frozen=True, slots=True)
class QualityFilterConfig:
    """Configuration for per-block quality evaluation."""

    # fastText model source
    quality_model_path: str | None = None
    fasttext_repo_id: str = DEFAULT_FASTTEXT_REPO_ID
    fasttext_filename: str = DEFAULT_FASTTEXT_FILENAME

    # classifier decision
    quality_threshold: float = 0.5
    pass_labels: tuple[str, ...] | None = None
    require_pass_label: bool = True

    # lightweight heuristic gates
    min_block_words: int = 8
    min_block_chars: int = 40
    keep_list_blocks: bool = True
    max_list_lines: int = 20

    hard_drop_patterns: tuple[str, ...] = (
        r"\bcaptcha\b",
        r"\bcookie(?:\s+policy)?\b",
        r"\baccept all\b",
        r"\bprivacy preferences\b",
        r"\bnewsletter\b",
        r"\bsubscribe\b",
        r"\bsign up\b",
        r"\benable javascript\b",
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.quality_threshold <= 1.0):
            raise ValueError("quality_threshold must be between 0 and 1")
        if self.min_block_words < 0:
            raise ValueError("min_block_words must be >= 0")
        if self.min_block_chars < 0:
            raise ValueError("min_block_chars must be >= 0")
        if self.max_list_lines <= 0:
            raise ValueError("max_list_lines must be > 0")
        if not self.fasttext_repo_id.strip():
            raise ValueError("fasttext_repo_id cannot be empty")
        if not self.fasttext_filename.strip():
            raise ValueError("fasttext_filename cannot be empty")
        if self.pass_labels is not None and len(self.pass_labels) == 0:
            raise ValueError("pass_labels cannot be empty when provided")


@dataclass(frozen=True, slots=True)
class BlockFilterResult:
    """Pass/fail output for a single block."""

    passed: bool
    reason: str | None
    quality_label: str | None
    quality_score: float | None
    word_count: int
    char_count: int
    list_like: bool
    list_line_count: int
    metadata: dict[str, JSONValue]

    @property
    def decision(self) -> FilterDecision:
        return FilterDecision.KEEP if self.passed else FilterDecision.DROP

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "passed": self.passed,
            "decision": self.decision.value,
            "reason": self.reason,
            "quality_label": self.quality_label,
            "quality_score": self.quality_score,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "list_like": self.list_like,
            "list_line_count": self.list_line_count,
            "metadata": self.metadata,
        }


class QualityFilter:
    """Evaluate one text block at a time using heuristics + fastText score."""

    def __init__(
        self,
        config: QualityFilterConfig | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self.config = config or QualityFilterConfig()
        self._hard_drop_regexes = [
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in self.config.hard_drop_patterns
        ]

        # Required by user: load from HF Hub unless a local model path is provided.
        self.model = model or self._load_model()
        self.model_labels = tuple(str(label) for label in self.model.get_labels())
        self.pass_labels = self._resolve_pass_labels(self.config.pass_labels, self.model_labels)

    @classmethod
    def from_chunker_config(
        cls,
        chunker_config: ChunkerConfig,
        *,
        model: Any | None = None,
    ) -> "QualityFilter":
        config = QualityFilterConfig(
            quality_model_path=chunker_config.quality_model_path,
            quality_threshold=chunker_config.quality_threshold or 0.0,
            min_block_words=chunker_config.min_block_words,
            keep_list_blocks=chunker_config.keep_list_blocks,
            max_list_lines=chunker_config.max_list_lines,
        )
        return cls(config=config, model=model)

    def evaluate_block(
        self,
        block: TextBlock | str,
        *,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> BlockFilterResult:
        text, block_meta = self._extract_inputs(block, metadata=metadata)
        normalized = _normalize_text(text)

        if not normalized:
            return self._reject(
                reason="empty_block",
                quality_label=None,
                quality_score=None,
                word_count=0,
                char_count=0,
                list_like=False,
                list_line_count=0,
                metadata={"source": "quality_filter"},
            )

        word_count = count_words(normalized)
        char_count = len(normalized)
        list_like = _is_list_like(normalized, block_meta)
        list_line_count = _line_count(normalized)

        base_meta: dict[str, JSONValue] = {
            "source": "quality_filter",
            "word_count": word_count,
            "char_count": char_count,
            "list_like": list_like,
            "list_line_count": list_line_count,
        }

        if self.config.min_block_words > 0 and word_count < self.config.min_block_words:
            return self._reject(
                reason=f"min_block_words:{word_count}<{self.config.min_block_words}",
                quality_label=None,
                quality_score=None,
                word_count=word_count,
                char_count=char_count,
                list_like=list_like,
                list_line_count=list_line_count,
                metadata=base_meta,
            )

        if self.config.min_block_chars > 0 and char_count < self.config.min_block_chars:
            return self._reject(
                reason=f"min_block_chars:{char_count}<{self.config.min_block_chars}",
                quality_label=None,
                quality_score=None,
                word_count=word_count,
                char_count=char_count,
                list_like=list_like,
                list_line_count=list_line_count,
                metadata=base_meta,
            )

        if list_like:
            if not self.config.keep_list_blocks:
                return self._reject(
                    reason="list_blocks_disabled",
                    quality_label=None,
                    quality_score=None,
                    word_count=word_count,
                    char_count=char_count,
                    list_like=list_like,
                    list_line_count=list_line_count,
                    metadata=base_meta,
                )
            if list_line_count > self.config.max_list_lines:
                return self._reject(
                    reason=f"list_too_long:{list_line_count}>{self.config.max_list_lines}",
                    quality_label=None,
                    quality_score=None,
                    word_count=word_count,
                    char_count=char_count,
                    list_like=list_like,
                    list_line_count=list_line_count,
                    metadata=base_meta,
                )

        hard_reason = self._hard_drop_reason(normalized)
        if hard_reason is not None:
            return self._reject(
                reason=hard_reason,
                quality_label=None,
                quality_score=None,
                word_count=word_count,
                char_count=char_count,
                list_like=list_like,
                list_line_count=list_line_count,
                metadata=base_meta,
            )

        quality_label, quality_score = self._predict_quality(normalized)

        if (
            self.config.require_pass_label
            and len(self.pass_labels) > 0
            and quality_label not in self.pass_labels
        ):
            base_meta["pass_labels"] = list(self.pass_labels)
            return self._reject(
                reason=f"label_not_allowed:{quality_label}",
                quality_label=quality_label,
                quality_score=quality_score,
                word_count=word_count,
                char_count=char_count,
                list_like=list_like,
                list_line_count=list_line_count,
                metadata=base_meta,
            )

        if quality_score is None or quality_score < self.config.quality_threshold:
            return self._reject(
                reason=(
                    "quality_score_below_threshold:"
                    f"{quality_score if quality_score is not None else 'None'}"
                    f"<{self.config.quality_threshold:.4f}"
                ),
                quality_label=quality_label,
                quality_score=quality_score,
                word_count=word_count,
                char_count=char_count,
                list_like=list_like,
                list_line_count=list_line_count,
                metadata=base_meta,
            )

        return BlockFilterResult(
            passed=True,
            reason=None,
            quality_label=quality_label,
            quality_score=quality_score,
            word_count=word_count,
            char_count=char_count,
            list_like=list_like,
            list_line_count=list_line_count,
            metadata=base_meta,
        )

    def passes_block(
        self,
        block: TextBlock | str,
        *,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> bool:
        return self.evaluate_block(block, metadata=metadata).passed

    def _load_model(self) -> Any:
        if self.config.quality_model_path:
            return fasttext.load_model(self.config.quality_model_path)

        model_path = hf_hub_download(
            repo_id=self.config.fasttext_repo_id,
            filename=self.config.fasttext_filename,
        )
        return fasttext.load_model(model_path)

    def _predict_quality(self, text: str) -> tuple[str | None, float | None]:
        # fastText `predict` expects one line; collapse internal newlines/whitespace.
        line = re.sub(r"\s+", " ", text).strip()
        if not line:
            return None, None

        labels, scores = self.model.predict(line, k=1)
        if not labels:
            return None, None
        label = str(labels[0])
        score = float(scores[0]) if scores else None
        return label, score

    def _hard_drop_reason(self, text: str) -> str | None:
        for idx, regex in enumerate(self._hard_drop_regexes):
            if regex.search(text):
                return f"hard_drop_pattern:{idx}"
        return None

    @staticmethod
    def _extract_inputs(
        block: TextBlock | str,
        *,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> tuple[str, Mapping[str, JSONValue]]:
        if isinstance(block, TextBlock):
            merged_meta: dict[str, JSONValue] = dict(block.metadata)
            if metadata:
                merged_meta.update(dict(metadata))
            return block.text, merged_meta

        return str(block), (metadata or {})

    @staticmethod
    def _resolve_pass_labels(
        configured_pass_labels: tuple[str, ...] | None,
        model_labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        if configured_pass_labels is not None:
            return tuple(configured_pass_labels)

        if "__label__1" in model_labels:
            return ("__label__1",)

        keyword_hits = []
        keywords = ("high", "good", "positive", "quality", "pass", "clean", "textbook")
        for label in model_labels:
            lowered = label.lower()
            if any(keyword in lowered for keyword in keywords):
                keyword_hits.append(label)

        if keyword_hits:
            return tuple(keyword_hits)

        return tuple(model_labels)

    @staticmethod
    def _reject(
        *,
        reason: str,
        quality_label: str | None,
        quality_score: float | None,
        word_count: int,
        char_count: int,
        list_like: bool,
        list_line_count: int,
        metadata: dict[str, JSONValue],
    ) -> BlockFilterResult:
        return BlockFilterResult(
            passed=False,
            reason=reason,
            quality_label=quality_label,
            quality_score=quality_score,
            word_count=word_count,
            char_count=char_count,
            list_like=list_like,
            list_line_count=list_line_count,
            metadata=metadata,
        )


def _normalize_text(text: str) -> str:
    value = str(text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = "\n".join(line.strip() for line in value.split("\n"))
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value


def _line_count(text: str) -> int:
    return sum(1 for line in text.split("\n") if line.strip())


def _is_list_like(text: str, metadata: Mapping[str, JSONValue]) -> bool:
    from_meta = metadata.get("list_like")
    if isinstance(from_meta, bool):
        return from_meta

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return False

    list_marked = sum(1 for line in lines if LIST_LINE_RE.match(line))
    if list_marked >= 2:
        return True
    return any(line.startswith("|") and line.endswith("|") for line in lines)


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


__all__ = [
    "BlockFilterResult",
    "DEFAULT_FASTTEXT_FILENAME",
    "DEFAULT_FASTTEXT_REPO_ID",
    "QualityFilter",
    "QualityFilterConfig",
    "count_words",
]

"""Per-block fastText language filter using `lid.176.bin`."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.request import urlretrieve

import fasttext

from .config import ChunkerConfig
from .types import FilterDecision, JSONValue, TextBlock


LOGGER = logging.getLogger(__name__)

DEFAULT_FASTTEXT_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)
DEFAULT_FASTTEXT_MODEL_FILENAME = "lid.176.bin"
DEFAULT_FASTTEXT_CACHE_DIR = Path.home() / ".cache" / "fasttext"
WHITESPACE_RE = re.compile(r"\s+")
HYPHEN_ONLY_RE = re.compile(r"^-+$")


@dataclass(frozen=True, slots=True)
class QualityFilterConfig:
    """Config for fastText language classification."""

    quality_model_path: str | None = None
    model_url: str = DEFAULT_FASTTEXT_MODEL_URL
    model_filename: str = DEFAULT_FASTTEXT_MODEL_FILENAME
    cache_dir: str | None = None
    keep_labels: tuple[str, ...] = ("en",)
    min_score: float = 0.5
    max_pipe_token_fraction: float | None = None
    max_dash_token_fraction: float | None = None
    top_k: int = 4
    newline_replacement: str = " "

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_score <= 1.0):
            raise ValueError("min_score must be between 0 and 1")
        if self.max_pipe_token_fraction is not None and not (
            0.0 <= self.max_pipe_token_fraction <= 1.0
        ):
            raise ValueError("max_pipe_token_fraction must be between 0 and 1 when set")
        if self.max_dash_token_fraction is not None and not (
            0.0 <= self.max_dash_token_fraction <= 1.0
        ):
            raise ValueError("max_dash_token_fraction must be between 0 and 1 when set")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not self.model_url.strip():
            raise ValueError("model_url cannot be empty")
        if not self.model_filename.strip():
            raise ValueError("model_filename cannot be empty")
        if len(self.keep_labels) == 0:
            raise ValueError("keep_labels cannot be empty")


@dataclass(frozen=True, slots=True)
class BlockFilterResult:
    """Pass/fail output for one block."""

    passed: bool
    reason: str | None
    quality_label: str | None
    quality_score: float | None
    metadata: dict[str, JSONValue]

    @property
    def decision(self) -> FilterDecision:
        return FilterDecision.KEEP if self.passed else FilterDecision.DROP


class QualityFilter:
    """Classify blocks with fastText language identification only."""

    def __init__(
        self,
        config: QualityFilterConfig | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self.config = config or QualityFilterConfig()
        model_path = self._resolve_model_path()
        self.model = model if model is not None else fasttext.load_model(model_path)
        self.model_labels = self._extract_model_labels(self.model)
        self.keep_labels_full = {self._full_label(label) for label in self.config.keep_labels}

    @classmethod
    def from_chunker_config(
        cls,
        chunker_config: ChunkerConfig,
        *,
        model: Any | None = None,
    ) -> "QualityFilter":
        threshold = chunker_config.quality_threshold if chunker_config.quality_threshold is not None else 0.5
        target_lang = chunker_config.lang or "en"
        model_path = chunker_config.quality_model_path
        config = QualityFilterConfig(
            quality_model_path=model_path,
            keep_labels=(target_lang,),
            min_score=threshold,
            max_pipe_token_fraction=chunker_config.max_pipe_token_fraction,
            max_dash_token_fraction=chunker_config.max_dash_token_fraction,
        )
        return cls(config=config, model=model)

    def evaluate_block(
        self,
        block: TextBlock | str,
        *,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> BlockFilterResult:
        text, _ = self._extract_inputs(block, metadata=metadata)
        normalized = self._normalize_text(text)
        if not normalized:
            return BlockFilterResult(
                passed=False,
                reason="empty_block",
                quality_label=None,
                quality_score=None,
                metadata={"label_scores": {}},
            )

        total_words, pipe_words, pipe_fraction, dash_words, dash_fraction = self._token_stats(normalized)
        table_heuristics = {
            "word_count_total": total_words,
            "pipe_token_count": pipe_words,
            "pipe_token_fraction": pipe_fraction,
            "max_pipe_token_fraction": self.config.max_pipe_token_fraction,
            "dash_token_count": dash_words,
            "dash_token_fraction": dash_fraction,
            "max_dash_token_fraction": self.config.max_dash_token_fraction,
        }
        if (
            self.config.max_pipe_token_fraction is not None
            and total_words > 0
            and pipe_fraction > self.config.max_pipe_token_fraction
        ):
            return BlockFilterResult(
                passed=False,
                reason=(
                    "pipe_token_fraction_above_threshold:"
                    f"{pipe_fraction:.4f}>{self.config.max_pipe_token_fraction:.4f}"
                ),
                quality_label=None,
                quality_score=None,
                metadata={"label_scores": {}, **table_heuristics},
            )
        if (
            self.config.max_dash_token_fraction is not None
            and total_words > 0
            and dash_fraction > self.config.max_dash_token_fraction
        ):
            return BlockFilterResult(
                passed=False,
                reason=(
                    "dash_token_fraction_above_threshold:"
                    f"{dash_fraction:.4f}>{self.config.max_dash_token_fraction:.4f}"
                ),
                quality_label=None,
                quality_score=None,
                metadata={"label_scores": {}, **table_heuristics},
            )

        labels, scores = self.model.predict(normalized, k=self.config.top_k)
        if not labels:
            return BlockFilterResult(
                passed=False,
                reason="classifier_no_prediction",
                quality_label=None,
                quality_score=None,
                metadata={"label_scores": {}, **table_heuristics},
            )

        label_scores = self._build_label_scores(labels, scores)
        top_label = self._full_label(str(labels[0]))
        top_score = float(scores[0])
        passed = top_label in self.keep_labels_full and top_score >= self.config.min_score

        if passed:
            return BlockFilterResult(
                passed=True,
                reason=None,
                quality_label=top_label,
                quality_score=top_score,
                metadata={
                    "label_scores": label_scores,
                    "keep_labels": sorted(self.keep_labels_full),
                    **table_heuristics,
                },
            )

        return BlockFilterResult(
            passed=False,
            reason=self._reject_reason(top_label=top_label, top_score=top_score),
            quality_label=top_label,
            quality_score=top_score,
            metadata={
                "label_scores": label_scores,
                "keep_labels": sorted(self.keep_labels_full),
                **table_heuristics,
            },
        )

    def passes_block(
        self,
        block: TextBlock | str,
        *,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> bool:
        return self.evaluate_block(block, metadata=metadata).passed

    def _resolve_model_path(self) -> str:
        if self.config.quality_model_path:
            return self.config.quality_model_path

        cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else DEFAULT_FASTTEXT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / self.config.model_filename
        if not model_path.exists():
            LOGGER.info("Downloading fastText model from %s to %s", self.config.model_url, model_path)
            urlretrieve(self.config.model_url, model_path)
        return str(model_path)

    def _reject_reason(self, *, top_label: str, top_score: float) -> str:
        if top_label not in self.keep_labels_full:
            return f"classifier_label_not_allowed:{top_label}"
        return (
            "classifier_score_below_threshold:"
            f"{top_label}={top_score:.4f}<{self.config.min_score:.4f}"
        )

    def _normalize_text(self, text: str) -> str:
        value = str(text or "")
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = value.replace("\n", self.config.newline_replacement)
        value = WHITESPACE_RE.sub(" ", value)
        return value.strip()

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
    def _full_label(label: str) -> str:
        value = str(label).strip()
        if value.startswith("__label__"):
            return value
        return f"__label__{value}"

    @staticmethod
    def _build_label_scores(labels: list[str], scores: list[float]) -> dict[str, float]:
        return {
            QualityFilter._full_label(str(label)): float(score)
            for label, score in zip(labels, scores)
        }

    @staticmethod
    def _token_stats(text: str) -> tuple[int, int, float, int, float]:
        words = text.split()
        total_words = len(words)
        pipe_words = sum(1 for word in words if word == "|")
        dash_words = sum(1 for word in words if HYPHEN_ONLY_RE.fullmatch(word) is not None)
        if total_words == 0:
            return 0, 0, 0.0, 0, 0.0
        return (
            total_words,
            pipe_words,
            pipe_words / total_words,
            dash_words,
            dash_words / total_words,
        )

    @staticmethod
    def _extract_model_labels(model: Any) -> tuple[str, ...]:
        labels = model.get_labels()
        return tuple(QualityFilter._full_label(str(label)) for label in labels)


__all__ = [
    "BlockFilterResult",
    "DEFAULT_FASTTEXT_CACHE_DIR",
    "DEFAULT_FASTTEXT_MODEL_FILENAME",
    "DEFAULT_FASTTEXT_MODEL_URL",
    "QualityFilter",
    "QualityFilterConfig",
]

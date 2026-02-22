"""Split source document text into cleaned block candidates."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Iterator

from .types import SourceDocument, TextBlock


BLANK_LINE_SPLIT_RE = re.compile(r"\n\s*\n+")
INLINE_SPACE_RE = re.compile(r"[ \t\f\v]+")
LIST_LINE_RE = re.compile(r"^\s*(?:[-*+•]\s+|(?:\d+|[A-Za-z])[.)]\s+)")


@dataclass(frozen=True, slots=True)
class BlockSplitConfig:
    """Configuration for basic text-to-block splitting."""

    min_block_words: int = 8
    min_block_chars: int = 40
    merge_separator: str = "\n\n"
    preserve_list_newlines: bool = True

    def __post_init__(self) -> None:
        if self.min_block_words < 0:
            raise ValueError("min_block_words must be >= 0")
        if self.min_block_chars < 0:
            raise ValueError("min_block_chars must be >= 0")
        if not self.merge_separator:
            raise ValueError("merge_separator cannot be empty")


@dataclass(slots=True)
class _CandidateBlock:
    text: str
    raw_indices: list[int]

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def split_document(
    doc: SourceDocument,
    *,
    config: BlockSplitConfig | None = None,
) -> list[TextBlock]:
    """Split one source document into normalized blocks."""

    cfg = config or BlockSplitConfig()
    candidates = _split_text_to_candidates(
        doc.text,
        preserve_list_newlines=cfg.preserve_list_newlines,
    )
    merged = _merge_tiny_candidates(
        candidates,
        min_block_words=cfg.min_block_words,
        min_block_chars=cfg.min_block_chars,
        separator=cfg.merge_separator,
    )

    blocks: list[TextBlock] = []
    for block_id, candidate in enumerate(merged):
        text = candidate.text.strip()
        if not text:
            continue
        blocks.append(
            TextBlock(
                doc_id=doc.doc_id,
                block_id=block_id,
                text=text,
                metadata={
                    "raw_block_indices": candidate.raw_indices,
                    "raw_block_count": len(candidate.raw_indices),
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "list_like": _looks_like_list_block(text),
                },
            )
        )

    return blocks


def split_documents(
    docs: Iterable[SourceDocument],
    *,
    config: BlockSplitConfig | None = None,
) -> Iterator[tuple[SourceDocument, list[TextBlock]]]:
    """Yield `(doc, blocks)` for each source document."""

    for doc in docs:
        yield doc, split_document(doc, config=config)


def split_text(
    text: str,
    *,
    config: BlockSplitConfig | None = None,
) -> list[str]:
    """Split raw text and return cleaned block strings (without metadata)."""

    cfg = config or BlockSplitConfig()
    candidates = _split_text_to_candidates(
        text,
        preserve_list_newlines=cfg.preserve_list_newlines,
    )
    merged = _merge_tiny_candidates(
        candidates,
        min_block_words=cfg.min_block_words,
        min_block_chars=cfg.min_block_chars,
        separator=cfg.merge_separator,
    )
    return [candidate.text for candidate in merged if candidate.text.strip()]


def _split_text_to_candidates(
    text: str,
    *,
    preserve_list_newlines: bool,
) -> list[_CandidateBlock]:
    normalized = _normalize_doc_text(text)
    if not normalized:
        return []

    raw_parts = BLANK_LINE_SPLIT_RE.split(normalized)
    candidates: list[_CandidateBlock] = []

    for raw_index, raw_part in enumerate(raw_parts):
        cleaned = _clean_raw_block(raw_part, preserve_list_newlines=preserve_list_newlines)
        if not cleaned:
            continue
        candidates.append(_CandidateBlock(text=cleaned, raw_indices=[raw_index]))

    return candidates


def _normalize_doc_text(text: str) -> str:
    value = str(text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\xa0", " ")
    value = value.strip()
    return value


def _clean_raw_block(raw_block: str, *, preserve_list_newlines: bool) -> str:
    lines = [INLINE_SPACE_RE.sub(" ", line).strip() for line in raw_block.split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    if preserve_list_newlines and _looks_like_list_lines(lines):
        return "\n".join(lines).strip()

    return " ".join(lines).strip()


def _looks_like_list_lines(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False

    list_marked = sum(1 for line in lines if LIST_LINE_RE.match(line))
    if list_marked >= 2:
        return True

    if any(line.startswith("|") and line.endswith("|") for line in lines):
        return True

    return list_marked >= 1 and len(lines) <= 4


def _looks_like_list_block(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return _looks_like_list_lines(lines)


def _merge_tiny_candidates(
    candidates: list[_CandidateBlock],
    *,
    min_block_words: int,
    min_block_chars: int,
    separator: str,
) -> list[_CandidateBlock]:
    if not candidates:
        return []

    merged: list[_CandidateBlock] = []
    carry: _CandidateBlock | None = None

    for candidate in candidates:
        if _is_tiny(candidate, min_block_words=min_block_words, min_block_chars=min_block_chars):
            carry = _merge_two(carry, candidate, separator=separator) if carry else candidate
            continue

        if carry is not None:
            candidate = _merge_two(carry, candidate, separator=separator)
            carry = None

        merged.append(candidate)

    if carry is not None:
        if merged:
            merged[-1] = _merge_two(merged[-1], carry, separator=separator)
        else:
            merged.append(carry)

    return merged


def _is_tiny(candidate: _CandidateBlock, *, min_block_words: int, min_block_chars: int) -> bool:
    words_too_small = min_block_words > 0 and candidate.word_count < min_block_words
    chars_too_small = min_block_chars > 0 and candidate.char_count < min_block_chars
    return words_too_small or chars_too_small


def _merge_two(left: _CandidateBlock, right: _CandidateBlock, *, separator: str) -> _CandidateBlock:
    merged_text = left.text.strip()
    right_text = right.text.strip()

    if merged_text and right_text:
        merged_text = f"{merged_text}{separator}{right_text}"
    elif right_text:
        merged_text = right_text

    return _CandidateBlock(
        text=merged_text,
        raw_indices=[*left.raw_indices, *right.raw_indices],
    )


__all__ = [
    "BlockSplitConfig",
    "split_document",
    "split_documents",
    "split_text",
]

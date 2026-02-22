"""Split source document text into sentence-packed blocks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Iterator

from .types import SourceDocument, TextBlock


WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class BlockSplitConfig:
    """Configuration for sentence-based block splitting."""

    min_block_words: int = 0
    max_block_words: int = 150
    min_block_chars: int = 0
    sentence_boundary_pattern: str = r"(?<=[.!?])\s+"

    def __post_init__(self) -> None:
        if self.min_block_words < 0:
            raise ValueError("min_block_words must be >= 0")
        if self.max_block_words <= 0:
            raise ValueError("max_block_words must be > 0")
        if self.min_block_chars < 0:
            raise ValueError("min_block_chars must be >= 0")
        if not self.sentence_boundary_pattern:
            raise ValueError("sentence_boundary_pattern cannot be empty")


def split_document(
    doc: SourceDocument,
    *,
    config: BlockSplitConfig | None = None,
) -> list[TextBlock]:
    """Split one source document into sentence-packed blocks."""

    cfg = config or BlockSplitConfig()
    block_texts = split_text(doc.text, config=cfg)

    blocks: list[TextBlock] = []
    for block_id, text in enumerate(block_texts):
        text = text.strip()
        if not text:
            continue
        word_count = len(text.split())
        char_count = len(text)
        blocks.append(
            TextBlock(
                doc_id=doc.doc_id,
                block_id=block_id,
                text=text,
                metadata={
                    "word_count": word_count,
                    "char_count": char_count,
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
    """Split raw text into sentence-packed block strings."""

    cfg = config or BlockSplitConfig()
    normalized = _normalize_doc_text(text)
    if not normalized:
        return []

    sentence_boundary_re = re.compile(cfg.sentence_boundary_pattern)
    sentences = _split_sentences(normalized, boundary_re=sentence_boundary_re)
    packed = _pack_sentences(sentences, max_block_words=cfg.max_block_words)
    return _apply_minimums(
        packed,
        min_block_words=cfg.min_block_words,
        min_block_chars=cfg.min_block_chars,
    )


def _normalize_doc_text(text: str) -> str:
    value = str(text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\xa0", " ")
    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def _split_sentences(text: str, *, boundary_re: re.Pattern[str]) -> list[str]:
    parts = [part.strip() for part in boundary_re.split(text) if part.strip()]
    return parts


def _pack_sentences(sentences: list[str], *, max_block_words: int) -> list[str]:
    blocks: list[str] = []
    current_sentences: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        if sentence_word_count == 0:
            continue

        if sentence_word_count > max_block_words:
            if current_sentences:
                blocks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_words = 0
            blocks.extend(_split_long_sentence(sentence_words, max_block_words=max_block_words))
            continue

        if current_sentences and (current_words + sentence_word_count > max_block_words):
            blocks.append(" ".join(current_sentences).strip())
            current_sentences = [sentence]
            current_words = sentence_word_count
            continue

        current_sentences.append(sentence)
        current_words += sentence_word_count

    if current_sentences:
        blocks.append(" ".join(current_sentences).strip())

    return [block for block in blocks if block]


def _split_long_sentence(words: list[str], *, max_block_words: int) -> list[str]:
    chunks: list[str] = []
    for start in range(0, len(words), max_block_words):
        piece = " ".join(words[start : start + max_block_words]).strip()
        if piece:
            chunks.append(piece)
    return chunks


def _apply_minimums(
    blocks: list[str],
    *,
    min_block_words: int,
    min_block_chars: int,
) -> list[str]:
    if min_block_words <= 0 and min_block_chars <= 0:
        return blocks

    filtered: list[str] = []
    for block in blocks:
        if min_block_words > 0 and len(block.split()) < min_block_words:
            continue
        if min_block_chars > 0 and len(block) < min_block_chars:
            continue
        filtered.append(block)
    return filtered


__all__ = [
    "BlockSplitConfig",
    "split_document",
    "split_documents",
    "split_text",
]

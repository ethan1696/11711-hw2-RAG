"""PDF parser with robust extraction and aggressive text-quality filtering."""

from __future__ import annotations

import io
import re
import string
from dataclasses import dataclass, field

import pdfplumber
from pypdf import PdfReader

from ..types import ContentKind, ParseResult


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

BOILERPLATE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*page\s+\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"\ball rights reserved\b", re.IGNORECASE),
    re.compile(r"\bconfidential\b", re.IGNORECASE),
    re.compile(r"\bdo not distribute\b", re.IGNORECASE),
    re.compile(r"\bprinted on\b", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class PDFQualityProfile:
    """Thresholds for filtering noisy PDF chunks."""

    min_chars_per_chunk: int = 70
    min_words_per_chunk: int = 12
    min_alpha_ratio: float = 0.55
    max_digit_ratio: float = 0.35
    max_punct_ratio: float = 0.40
    min_unique_token_ratio: float = 0.25


@dataclass(slots=True)
class PDFParserConfig:
    """Config for PDF extraction and filtering."""

    parser_name: str = "pdf_parser_v1"
    prefer_pypdf: bool = True
    use_pdfplumber_fallback: bool = True
    max_pages: int | None = None

    min_document_chars: int = 180
    min_document_words: int = 30
    repeated_line_threshold_ratio: float = 0.6

    quality_profile: PDFQualityProfile = field(default_factory=PDFQualityProfile)
    relaxed_quality_profile: PDFQualityProfile = field(
        default_factory=lambda: PDFQualityProfile(
            min_chars_per_chunk=45,
            min_words_per_chunk=8,
            min_alpha_ratio=0.48,
            max_digit_ratio=0.45,
            max_punct_ratio=0.50,
            min_unique_token_ratio=0.18,
        )
    )


@dataclass(frozen=True, slots=True)
class _ChunkStats:
    words: int
    chars: int
    alpha_ratio: float
    digit_ratio: float
    punct_ratio: float
    unique_token_ratio: float


class PDFParser:
    """Parse a PDF into cleaned text for retrieval."""

    def __init__(self, config: PDFParserConfig | None = None) -> None:
        self.config = config or PDFParserConfig()

    def parse(
        self,
        *,
        url: str,
        pdf_bytes: bytes,
        final_url: str | None = None,
    ) -> ParseResult:
        """Parse one PDF payload into filtered text."""

        if not isinstance(pdf_bytes, (bytes, bytearray)):
            return ParseResult(
                url=url,
                final_url=final_url,
                title=None,
                text="",
                out_links=[],
                content_kind=ContentKind.PDF,
                parser=self.config.parser_name,
                metadata={},
                error="pdf_bytes must be bytes",
            )

        pypdf_pages: list[str] = []
        pypdf_error: str | None = None

        if self.config.prefer_pypdf:
            pypdf_pages, pypdf_error = self._extract_pages_with_pypdf(pdf_bytes)

        chosen_pages = pypdf_pages
        extractor = "pypdf"
        fallback_used = False

        if self.config.use_pdfplumber_fallback:
            if self._is_low_signal_pages(chosen_pages):
                plumber_pages, plumber_error = self._extract_pages_with_pdfplumber(pdf_bytes)
                if self._page_word_count(plumber_pages) > self._page_word_count(chosen_pages):
                    chosen_pages = plumber_pages
                    extractor = "pdfplumber"
                    fallback_used = True
                elif plumber_error and not chosen_pages:
                    pypdf_error = plumber_error if pypdf_error is None else pypdf_error

        if not chosen_pages:
            return ParseResult(
                url=url,
                final_url=final_url,
                title=None,
                text="",
                out_links=[],
                content_kind=ContentKind.PDF,
                parser=self.config.parser_name,
                metadata={
                    "extractor": extractor,
                    "fallback_used": fallback_used,
                },
                error=pypdf_error or "No extractable text from PDF",
            )

        cleaned_text, filter_meta = self._filter_pages(chosen_pages, self.config.quality_profile)
        quality_mode = "strict"

        if not self._passes_document_quality(cleaned_text):
            relaxed_text, relaxed_meta = self._filter_pages(
                chosen_pages,
                self.config.relaxed_quality_profile,
            )
            if self._passes_document_quality(relaxed_text):
                cleaned_text = relaxed_text
                filter_meta = relaxed_meta
                quality_mode = "relaxed"
            elif len(relaxed_text) > len(cleaned_text):
                cleaned_text = relaxed_text
                filter_meta = relaxed_meta
                quality_mode = "relaxed_partial"

        if not self._passes_document_quality(cleaned_text):
            return ParseResult(
                url=url,
                final_url=final_url,
                title=None,
                text="",
                out_links=[],
                content_kind=ContentKind.PDF,
                parser=self.config.parser_name,
                metadata={
                    "extractor": extractor,
                    "fallback_used": fallback_used,
                    "quality_mode": quality_mode,
                    **filter_meta,
                },
                error="Filtered PDF text did not meet quality thresholds",
            )

        return ParseResult(
            url=url,
            final_url=final_url,
            title=None,
            text=cleaned_text,
            out_links=[],
            content_kind=ContentKind.PDF,
            parser=self.config.parser_name,
            metadata={
                "extractor": extractor,
                "fallback_used": fallback_used,
                "quality_mode": quality_mode,
                **filter_meta,
            },
            error=None,
        )

    def _extract_pages_with_pypdf(self, pdf_bytes: bytes) -> tuple[list[str], str | None]:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    return [], "PDF is encrypted and could not be decrypted"

            pages = []
            limit = self.config.max_pages or len(reader.pages)
            for idx, page in enumerate(reader.pages):
                if idx >= limit:
                    break
                text = page.extract_text() or ""
                pages.append(text)

            return pages, None
        except Exception as exc:
            return [], f"pypdf extraction failed: {exc.__class__.__name__}: {exc}"

    def _extract_pages_with_pdfplumber(self, pdf_bytes: bytes) -> tuple[list[str], str | None]:
        try:
            pages = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                limit = self.config.max_pages or len(pdf.pages)
                for idx, page in enumerate(pdf.pages):
                    if idx >= limit:
                        break
                    text = page.extract_text() or ""
                    pages.append(text)
            return pages, None
        except Exception as exc:
            return [], f"pdfplumber extraction failed: {exc.__class__.__name__}: {exc}"

    def _filter_pages(
        self,
        pages: list[str],
        profile: PDFQualityProfile,
    ) -> tuple[str, dict[str, int | float]]:
        page_lines = [self._normalize_lines(page_text) for page_text in pages]

        line_freq: dict[str, int] = {}
        for lines in page_lines:
            unique = {self._line_dedup_key(line) for line in lines if line}
            for key in unique:
                line_freq[key] = line_freq.get(key, 0) + 1

        page_count = max(1, len(page_lines))
        repeated_line_min_count = max(2, int(page_count * self.config.repeated_line_threshold_ratio))

        chunks: list[str] = []
        dropped_boilerplate = 0
        dropped_repeated = 0

        for lines in page_lines:
            kept_lines: list[str] = []
            for line in lines:
                if self._looks_like_boilerplate_line(line):
                    dropped_boilerplate += 1
                    continue

                key = self._line_dedup_key(line)
                if line_freq.get(key, 0) >= repeated_line_min_count and len(line) <= 120:
                    dropped_repeated += 1
                    continue

                kept_lines.append(line)

            if kept_lines:
                chunks.extend(self._lines_to_chunks(kept_lines))

        clean_chunks: list[str] = []
        seen_chunks: set[str] = set()
        dropped_quality = 0
        dropped_duplicate = 0

        for chunk in chunks:
            chunk = self._normalize_whitespace(chunk)
            if not chunk:
                continue

            stats = self._chunk_stats(chunk)
            if not self._passes_chunk_quality(stats, profile):
                dropped_quality += 1
                continue

            key = self._line_dedup_key(chunk)
            if key in seen_chunks:
                dropped_duplicate += 1
                continue

            seen_chunks.add(key)
            clean_chunks.append(chunk)

        text = "\n\n".join(clean_chunks)

        meta: dict[str, int | float] = {
            "pages_total": len(pages),
            "chunks_input": len(chunks),
            "chunks_kept": len(clean_chunks),
            "dropped_boilerplate": dropped_boilerplate,
            "dropped_repeated": dropped_repeated,
            "dropped_quality": dropped_quality,
            "dropped_duplicate": dropped_duplicate,
            "clean_chars": len(text),
            "clean_words": len(TOKEN_RE.findall(text)),
        }
        return text, meta

    @staticmethod
    def _normalize_lines(page_text: str) -> list[str]:
        if not page_text:
            return []

        text = page_text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\xa0", " ")

        lines = []
        for raw in text.split("\n"):
            line = re.sub(r"\s+", " ", raw).strip()
            if line:
                lines.append(line)
        return lines

    @staticmethod
    def _lines_to_chunks(lines: list[str]) -> list[str]:
        if not lines:
            return []

        chunks: list[str] = []
        current: list[str] = []

        for line in lines:
            current.append(line)
            # break on likely sentence end if chunk already has decent length
            if len(" ".join(current)) >= 420 or line.endswith((".", "!", "?", ":")):
                chunks.append(" ".join(current))
                current = []

        if current:
            chunks.append(" ".join(current))

        return chunks

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _line_dedup_key(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\W+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _looks_like_boilerplate_line(line: str) -> bool:
        lowered = line.lower()
        if len(lowered) <= 180:
            for pattern in BOILERPLATE_LINE_PATTERNS:
                if pattern.search(lowered):
                    return True

        if len(lowered) < 120 and lowered.count("|") >= 3:
            return True

        return False

    @staticmethod
    def _chunk_stats(chunk: str) -> _ChunkStats:
        tokens = TOKEN_RE.findall(chunk)
        token_count = len(tokens)
        chars = len(chunk)

        non_space = max(1, sum(1 for c in chunk if not c.isspace()))
        alpha_chars = sum(1 for c in chunk if c.isalpha())
        digit_chars = sum(1 for c in chunk if c.isdigit())
        punct_chars = sum(1 for c in chunk if c in string.punctuation)

        unique_token_ratio = 0.0
        if token_count > 0:
            unique_token_ratio = len({t.lower() for t in tokens}) / token_count

        return _ChunkStats(
            words=token_count,
            chars=chars,
            alpha_ratio=alpha_chars / non_space,
            digit_ratio=digit_chars / non_space,
            punct_ratio=punct_chars / non_space,
            unique_token_ratio=unique_token_ratio,
        )

    def _passes_chunk_quality(self, stats: _ChunkStats, p: PDFQualityProfile) -> bool:
        if stats.chars < p.min_chars_per_chunk:
            return False
        if stats.words < p.min_words_per_chunk:
            return False
        if stats.alpha_ratio < p.min_alpha_ratio:
            return False
        if stats.digit_ratio > p.max_digit_ratio:
            return False
        if stats.punct_ratio > p.max_punct_ratio:
            return False
        if stats.words >= 18 and stats.unique_token_ratio < p.min_unique_token_ratio:
            return False
        return True

    def _passes_document_quality(self, text: str) -> bool:
        if not text:
            return False

        if len(text) < self.config.min_document_chars:
            return False

        tokens = TOKEN_RE.findall(text)
        if len(tokens) < self.config.min_document_words:
            return False

        return True

    @staticmethod
    def _page_word_count(pages: list[str]) -> int:
        return sum(len(TOKEN_RE.findall(page)) for page in pages)

    def _is_low_signal_pages(self, pages: list[str]) -> bool:
        if not pages:
            return True

        words = self._page_word_count(pages)
        chars = sum(len(page) for page in pages)

        return words < self.config.min_document_words or chars < self.config.min_document_chars


__all__ = [
    "PDFParser",
    "PDFParserConfig",
    "PDFQualityProfile",
]

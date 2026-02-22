"""Simple HTML parser: Trafilatura extraction + link discovery."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument
import trafilatura

from ..types import ContentKind, ParseResult
from ..url import extract_links_from_html


@dataclass(slots=True)
class HTMLParserConfig:
    """Config for HTML extraction."""

    parser_name: str = "html_parser_v4_trafilatura_readability_filtered"
    include_nofollow_links: bool = False
    use_trafilatura: bool = True
    use_readability: bool = True
    merge_separator: str = "\n\n"
    min_text_chars: int | None = 60
    min_text_words: int | None = 20


class HTMLParser:
    """Parse HTML using Trafilatura + Readability and merge the outputs."""

    def __init__(self, config: HTMLParserConfig | None = None) -> None:
        self.config = config or HTMLParserConfig()

    def parse(
        self,
        *,
        url: str,
        html: str | bytes,
        final_url: str | None = None,
        allowed_domains: Iterable[str] | None = None,
        allowed_path_prefixes_by_domain: dict[str, Sequence[str]] | None = None,
    ) -> ParseResult:
        base_url = final_url or url
        html_text = self._coerce_html_text(html)

        soup = BeautifulSoup(html_text, "lxml")
        title = self._extract_title(soup)

        out_links = extract_links_from_html(
            html_text,
            base_url=base_url,
            allowed_domains=allowed_domains,
            allowed_path_prefixes_by_domain=allowed_path_prefixes_by_domain,
            include_nofollow=self.config.include_nofollow_links,
            normalize=True,
        )

        trafilatura_text, trafilatura_error = self._extract_with_trafilatura(html_text)
        readability_text, readability_title, readability_error = self._extract_with_readability(html_text)

        if not title and readability_title:
            title = readability_title

        text = self._merge_texts(trafilatura_text, readability_text)
        if not text:
            details = "; ".join(
                message
                for message in (trafilatura_error, readability_error)
                if message
            )
            error_msg = "No extractable text from HTML"
            if details:
                error_msg = f"{error_msg} ({details})"
            return ParseResult(
                url=url,
                final_url=final_url,
                title=title,
                text="",
                out_links=out_links,
                content_kind=ContentKind.HTML,
                parser=self.config.parser_name,
                metadata={
                    "extractor": "trafilatura+readability",
                    "raw_chars": len(html_text),
                    "trafilatura_chars": len(trafilatura_text),
                    "readability_chars": len(readability_text),
                    "links_found": len(out_links),
                    "trafilatura_error": trafilatura_error,
                    "readability_error": readability_error,
                },
                error=error_msg,
            )

        char_count = len(text)
        word_count = self._word_count(text)
        quality_error = self._quality_error(char_count=char_count, word_count=word_count)
        if quality_error:
            return ParseResult(
                url=url,
                final_url=final_url,
                title=title,
                text="",
                out_links=out_links,
                content_kind=ContentKind.HTML,
                parser=self.config.parser_name,
                metadata={
                    "extractor": "trafilatura+readability",
                    "raw_chars": len(html_text),
                    "trafilatura_chars": len(trafilatura_text),
                    "readability_chars": len(readability_text),
                    "clean_chars": char_count,
                    "word_count": word_count,
                    "min_text_chars": self.config.min_text_chars,
                    "min_text_words": self.config.min_text_words,
                    "links_found": len(out_links),
                    "trafilatura_error": trafilatura_error,
                    "readability_error": readability_error,
                },
                error=quality_error,
            )

        return ParseResult(
            url=url,
            final_url=final_url,
            title=title,
            text=text,
            out_links=out_links,
            content_kind=ContentKind.HTML,
            parser=self.config.parser_name,
            metadata={
                "extractor": "trafilatura+readability",
                "raw_chars": len(html_text),
                "trafilatura_chars": len(trafilatura_text),
                "readability_chars": len(readability_text),
                "clean_chars": char_count,
                "word_count": word_count,
                "min_text_chars": self.config.min_text_chars,
                "min_text_words": self.config.min_text_words,
                "links_found": len(out_links),
                "trafilatura_error": trafilatura_error,
                "readability_error": readability_error,
            },
            error=None,
        )

    def _extract_with_trafilatura(self, html_text: str) -> tuple[str, str | None]:
        if not self.config.use_trafilatura:
            return "", "Trafilatura disabled by config"

        try:
            extracted = trafilatura.extract(
                html_text,
                output_format="txt",
                include_comments=False,
                include_tables=True,
                include_images=False,
                deduplicate=True,
                favor_precision=True,
            )
            return (extracted or "").strip(), None
        except Exception as exc:
            return "", f"Trafilatura extraction failed: {exc.__class__.__name__}: {exc}"

    def _extract_with_readability(self, html_text: str) -> tuple[str, str | None, str | None]:
        if not self.config.use_readability:
            return "", None, "Readability disabled by config"

        try:
            doc = ReadabilityDocument(html_text)
            title = (doc.short_title() or "").strip() or None
            summary_html = doc.summary()
            if isinstance(summary_html, bytes):
                summary_html = summary_html.decode("utf-8", errors="replace")

            if not summary_html:
                return "", title, None

            soup = BeautifulSoup(summary_html, "lxml")
            text = soup.get_text("\n", strip=True).strip()
            return text, title, None
        except Exception as exc:
            return "", None, f"Readability extraction failed: {exc.__class__.__name__}: {exc}"

    def _merge_texts(self, trafilatura_text: str, readability_text: str) -> str:
        ordered_paragraphs: list[str] = []
        seen: set[str] = set()

        for text in (trafilatura_text, readability_text):
            for paragraph in self._split_paragraphs(text):
                key = self._dedupe_key(paragraph)
                if not key or key in seen:
                    continue
                seen.add(key)
                ordered_paragraphs.append(paragraph)

        return self.config.merge_separator.join(ordered_paragraphs).strip()

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        chunks = re.split(r"\n\s*\n+", normalized)
        paragraphs: list[str] = []
        for chunk in chunks:
            compact = re.sub(r"[ \t]+", " ", chunk).strip()
            if compact:
                paragraphs.append(compact)
        return paragraphs

    @staticmethod
    def _dedupe_key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    def _quality_error(self, *, char_count: int, word_count: int) -> str | None:
        min_chars = self.config.min_text_chars
        min_words = self.config.min_text_words

        failed: list[str] = []
        if min_chars is not None and char_count < min_chars:
            failed.append(f"chars={char_count} < min_text_chars={min_chars}")
        if min_words is not None and word_count < min_words:
            failed.append(f"words={word_count} < min_text_words={min_words}")

        if failed:
            return "Filtered text did not meet quality thresholds: " + "; ".join(failed)
        return None

    @staticmethod
    def _coerce_html_text(html: str | bytes) -> str:
        if isinstance(html, bytes):
            return html.decode("utf-8", errors="replace")
        return html

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str | None:
        if soup.title and soup.title.get_text(strip=True):
            return soup.title.get_text(" ", strip=True)
        heading = soup.find(["h1", "h2"])
        if heading:
            text = heading.get_text(" ", strip=True)
            if text:
                return text
        return None


__all__ = [
    "HTMLParser",
    "HTMLParserConfig",
]

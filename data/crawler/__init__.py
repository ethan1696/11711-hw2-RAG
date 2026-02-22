"""Crawler package: config, shared types, and pipeline components."""

from .config import CrawlConfig, host_from_url, load_config, normalize_domain, save_config
from .fetcher import Fetcher
from .frontier import EnqueueResult, EnqueueStatus, Frontier
from .pipeline import Pipeline, PipelineMode
from .parsers import (
    HTMLParser,
    HTMLParserConfig,
    PDFParser,
    PDFParserConfig,
    PDFQualityProfile,
)
from .stats import StatsCollector
from .storage import Storage
from .types import (
    ContentKind,
    CrawlStage,
    CrawlStats,
    DomainConfig,
    DocumentRecord,
    ErrorRecord,
    FetchBackend,
    FetchResult,
    FrontierItem,
    ParseResult,
    URLMetaRecord,
    infer_content_kind,
    utc_now_iso,
)
from .url import extract_links_from_html, filter_in_scope_urls, is_url_in_scope, normalize_url, resolve_url

__all__ = [
    "ContentKind",
    "CrawlConfig",
    "CrawlStage",
    "CrawlStats",
    "DomainConfig",
    "DocumentRecord",
    "EnqueueResult",
    "EnqueueStatus",
    "ErrorRecord",
    "FetchBackend",
    "FetchResult",
    "Fetcher",
    "FrontierItem",
    "HTMLParser",
    "HTMLParserConfig",
    "PDFParser",
    "PDFParserConfig",
    "PDFQualityProfile",
    "ParseResult",
    "Pipeline",
    "PipelineMode",
    "URLMetaRecord",
    "host_from_url",
    "infer_content_kind",
    "extract_links_from_html",
    "filter_in_scope_urls",
    "Frontier",
    "is_url_in_scope",
    "load_config",
    "normalize_domain",
    "normalize_url",
    "resolve_url",
    "save_config",
    "StatsCollector",
    "Storage",
    "utc_now_iso",
]

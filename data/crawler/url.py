"""URL normalization, scope filtering, and link extraction helpers."""

from __future__ import annotations

import posixpath
import re
from typing import Iterable, Mapping, Sequence
from urllib.parse import (
    parse_qsl,
    quote,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

from bs4 import BeautifulSoup


DEFAULT_ALLOWED_SCHEMES = ("http", "https")
SKIP_HREF_PREFIXES = ("javascript:", "mailto:", "tel:", "data:")
TRACKING_QUERY_PARAM_PREFIXES = ("utm_",)
TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "mkt_tok",
    "spm",
    "igshid",
    "ref_src",
}


def normalize_domain(domain_or_url: str) -> str:
    """Normalize a domain (or URL containing one) for matching.

    This strips `www.` and leading/trailing dots and lowercases the host.
    """

    raw = (domain_or_url or "").strip().lower()
    if not raw:
        return ""

    parsed = urlsplit(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host.strip(".")


def host_from_url(url: str) -> str:
    """Extract normalized host from URL."""

    parsed = urlsplit(url)
    host = (parsed.hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host.strip(".")


def is_http_url(url: str, allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES) -> bool:
    """Return True if URL is absolute and has an allowed HTTP-like scheme."""

    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    return parsed.scheme.lower() in {scheme.lower() for scheme in allowed_schemes}


def _has_default_port(scheme: str, port: int | None) -> bool:
    if port is None:
        return False
    return (scheme == "http" and port == 80) or (scheme == "https" and port == 443)


def _normalize_netloc(
    parsed_url,  # urllib.parse.SplitResult
    *,
    strip_default_port: bool,
) -> str:
    host = (parsed_url.hostname or "").lower()
    if not host:
        return parsed_url.netloc.lower()

    userinfo = ""
    if parsed_url.username:
        userinfo = quote(parsed_url.username, safe="")
        if parsed_url.password:
            userinfo += ":" + quote(parsed_url.password, safe="")
        userinfo += "@"

    port: int | None
    try:
        port = parsed_url.port
    except ValueError:
        port = None

    include_port = port is not None and (not strip_default_port or not _has_default_port(parsed_url.scheme.lower(), port))

    if include_port:
        return f"{userinfo}{host}:{port}"
    return f"{userinfo}{host}"


def _normalize_path(path: str, *, remove_trailing_slash: bool) -> str:
    if not path:
        return "/"

    collapsed = re.sub(r"/{2,}", "/", path)
    normalized = posixpath.normpath(collapsed)

    if collapsed.startswith("/") and not normalized.startswith("/"):
        normalized = "/" + normalized

    if normalized in {"", "."}:
        normalized = "/"

    if remove_trailing_slash and normalized != "/":
        normalized = normalized.rstrip("/")

    return normalized or "/"


def _is_tracking_query_key(key: str) -> bool:
    normalized = key.strip().lower()
    if not normalized:
        return False

    if normalized in TRACKING_QUERY_PARAMS:
        return True

    return any(normalized.startswith(prefix) for prefix in TRACKING_QUERY_PARAM_PREFIXES)


def _normalize_query(
    query: str,
    *,
    strip_tracking_params: bool,
    sort_query_params: bool,
) -> str:
    if not query:
        return ""

    pairs = parse_qsl(query, keep_blank_values=True)
    if strip_tracking_params:
        pairs = [(key, value) for key, value in pairs if not _is_tracking_query_key(key)]

    if sort_query_params:
        pairs = sorted(pairs, key=lambda item: (item[0], item[1]))

    if not pairs:
        return ""

    return urlencode(pairs, doseq=True)


def normalize_url(
    url: str,
    *,
    strip_fragment: bool = True,
    strip_default_port: bool = True,
    strip_tracking_params: bool = True,
    sort_query_params: bool = True,
    remove_trailing_slash: bool = True,
    allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES,
) -> str | None:
    """Canonicalize absolute URL for dedup and frontier consistency.

    Returns `None` for URLs that are invalid or outside allowed schemes.
    """

    if not url:
        return None

    raw = url.strip()
    if not raw:
        return None

    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in {item.lower() for item in allowed_schemes}:
        return None

    netloc = _normalize_netloc(parsed, strip_default_port=strip_default_port)
    if not netloc:
        return None

    path = _normalize_path(parsed.path, remove_trailing_slash=remove_trailing_slash)
    query = _normalize_query(
        parsed.query,
        strip_tracking_params=strip_tracking_params,
        sort_query_params=sort_query_params,
    )
    fragment = "" if strip_fragment else parsed.fragment

    return urlunsplit((scheme, netloc, path, query, fragment))


def resolve_url(
    base_url: str,
    href: str | None,
    *,
    normalize: bool = True,
    allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES,
) -> str | None:
    """Resolve possibly relative link against base URL and validate scheme."""

    if href is None:
        return None

    candidate = href.strip()
    if not candidate or candidate.startswith("#"):
        return None

    lowered = candidate.lower()
    if any(lowered.startswith(prefix) for prefix in SKIP_HREF_PREFIXES):
        return None

    absolute = urljoin(base_url, candidate)
    if normalize:
        return normalize_url(absolute, allowed_schemes=allowed_schemes)

    if is_http_url(absolute, allowed_schemes=allowed_schemes):
        return absolute
    return None


def matching_allowed_domain(url_or_host: str, allowed_domains: Iterable[str]) -> str | None:
    """Return best matching allowed domain for a URL/host, or None."""

    if "://" in url_or_host:
        host = host_from_url(url_or_host)
    else:
        host = normalize_domain(url_or_host)

    if not host:
        return None

    normalized_allowed = [normalize_domain(domain) for domain in allowed_domains]
    normalized_allowed = [domain for domain in normalized_allowed if domain]

    matches = [
        domain
        for domain in normalized_allowed
        if host == domain or host.endswith("." + domain)
    ]
    if not matches:
        return None

    return max(matches, key=len)


def is_allowed_domain(url_or_host: str, allowed_domains: Iterable[str]) -> bool:
    """Return True if URL/host matches the allowlist domain suffix rules."""

    return matching_allowed_domain(url_or_host, allowed_domains) is not None


def is_url_in_scope(
    url: str,
    *,
    allowed_domains: Iterable[str],
    allowed_path_prefixes_by_domain: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    """Return True when URL is in allowed domain/path scope."""

    normalized = normalize_url(url)
    if normalized is None:
        return False

    matched_domain = matching_allowed_domain(normalized, allowed_domains)
    if matched_domain is None:
        return False

    if not allowed_path_prefixes_by_domain:
        return True

    prefixes = allowed_path_prefixes_by_domain.get(matched_domain, ())
    if not prefixes:
        return True

    path = urlsplit(normalized).path or "/"
    return any(path.startswith(prefix) for prefix in prefixes)


def filter_in_scope_urls(
    urls: Iterable[str],
    *,
    allowed_domains: Iterable[str],
    allowed_path_prefixes_by_domain: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Filter URL collection to in-scope URLs while preserving order."""

    output: list[str] = []
    seen: set[str] = set()

    for url in urls:
        normalized = normalize_url(url)
        if normalized is None:
            continue
        if normalized in seen:
            continue
        if not is_url_in_scope(
            normalized,
            allowed_domains=allowed_domains,
            allowed_path_prefixes_by_domain=allowed_path_prefixes_by_domain,
        ):
            continue
        seen.add(normalized)
        output.append(normalized)

    return output


def extract_links_from_html(
    html: str | bytes,
    *,
    base_url: str,
    allowed_domains: Iterable[str] | None = None,
    allowed_path_prefixes_by_domain: Mapping[str, Sequence[str]] | None = None,
    include_nofollow: bool = False,
    normalize: bool = True,
) -> list[str]:
    """Extract resolved links from HTML anchor/area tags.

    Returns links in document order with duplicates removed.
    """

    soup = BeautifulSoup(html, "lxml")

    out: list[str] = []
    seen: set[str] = set()

    for element in soup.find_all(["a", "area"]):
        href = element.get("href")
        if not href:
            continue

        rel_values = {value.lower() for value in (element.get("rel") or [])}
        if not include_nofollow and "nofollow" in rel_values:
            continue

        resolved = resolve_url(base_url, href, normalize=normalize)
        if not resolved:
            continue

        if allowed_domains is not None:
            if not is_url_in_scope(
                resolved,
                allowed_domains=allowed_domains,
                allowed_path_prefixes_by_domain=allowed_path_prefixes_by_domain,
            ):
                continue

        if resolved in seen:
            continue

        seen.add(resolved)
        out.append(resolved)

    return out


__all__ = [
    "DEFAULT_ALLOWED_SCHEMES",
    "SKIP_HREF_PREFIXES",
    "TRACKING_QUERY_PARAM_PREFIXES",
    "TRACKING_QUERY_PARAMS",
    "extract_links_from_html",
    "filter_in_scope_urls",
    "host_from_url",
    "is_allowed_domain",
    "is_http_url",
    "is_url_in_scope",
    "matching_allowed_domain",
    "normalize_domain",
    "normalize_url",
    "resolve_url",
]

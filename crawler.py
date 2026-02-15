"""Web crawler module for extracting text and links from web pages."""

from __future__ import annotations

import logging
from typing import List, Tuple
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

from utils import domain_of, ensure_url, normalize_url

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "InfinityKnowledgeMesh/1.0 (+https://github.com/Infinity-Knowledge-Mesh)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
}

_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


def crawl(
    url: str,
    *,
    same_domain_only: bool = False,
    max_content_bytes: int = 1_500_000,
) -> Tuple[str, List[str]]:
    """
    Crawl a web page and extract its text content and links.
    
    Args:
        url: The URL of the web page to crawl.
        same_domain_only: Whether to keep only links on the same domain.
        max_content_bytes: Upper bound for response size to avoid huge pages.
    
    Returns:
        A tuple containing:
            - text: The extracted text content of the page.
            - links: A list of cleaned URLs found on the page.
    
    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If the URL is invalid.
    """
    target_url = ensure_url(url)
    logger.info("Crawling URL: %s", target_url)
    session = _build_session()

    try:
        response = session.get(target_url, headers=DEFAULT_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed HTTP request for %s", target_url)
        raise

    if not _is_html(response.headers.get("Content-Type", "")):
        raise ValueError(f"Non-HTML content at {target_url}")

    if max_content_bytes:
        declared_length = response.headers.get("Content-Length")
        if declared_length:
            try:
                length_int = int(declared_length)
            except (TypeError, ValueError):
                length_int = 0
            else:
                if length_int > max_content_bytes:
                    raise ValueError(
                        f"Content-Length {declared_length} exceeds limit at {target_url}"
                    )

    content = response.content
    if max_content_bytes and len(content) > max_content_bytes:
        raise ValueError(f"Content too large ({len(content)} bytes) at {target_url}")

    soup = BeautifulSoup(content, "html.parser")
    _strip_unwanted_tags(soup)
    text = _extract_text(soup)
    links = _extract_links(soup, target_url, same_domain_only)

    logger.info("Extracted %d links from %s", len(links), target_url)
    return text, links


def _strip_unwanted_tags(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()


def _extract_text(soup: BeautifulSoup) -> str:
    return " ".join(chunk for chunk in soup.stripped_strings if chunk)


def _extract_links(soup: BeautifulSoup, base_url: str, same_domain_only: bool) -> List[str]:
    links: List[str] = []
    seen = set()
    base_domain = domain_of(base_url)
    for anchor in soup.find_all("a", href=True):
        joined = urljoin(base_url, anchor["href"])
        normalized = normalize_url(joined)
        if not normalized or normalized in seen:
            continue
        if same_domain_only and domain_of(normalized) != base_domain:
            continue
        seen.add(normalized)
        links.append(normalized)
    return links


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _is_html(content_type: str) -> bool:
    lowered = content_type.lower()
    return any(ct in lowered for ct in _HTML_CONTENT_TYPES)

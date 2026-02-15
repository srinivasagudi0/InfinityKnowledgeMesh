"""Utility helpers for working with URLs."""

from __future__ import annotations

import logging
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")


def normalize_url(raw_url: str) -> str:
    """
    Normalize a URL so downstream components can treat nodes consistently.

    Args:
        raw_url: The input URL which may be partial or missing a scheme.

    Returns:
        A normalized URL string (lower-cased host, trimmed path, no fragments)
        or an empty string if the URL is not a supported HTTP(S) address.
    """
    if not raw_url:
        return ""

    candidate = raw_url.strip()
    if not candidate:
        return ""

    parsed = urlparse(candidate)
    if not parsed.scheme:
        parsed = urlparse(f"https://{candidate}")

    if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.netloc:
        return ""

    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=_normalize_path(parsed.path),
        fragment="",
    )
    return urlunparse(normalized)


def ensure_url(raw_url: str) -> str:
    """
    Validate that a URL points to an HTTP(S) resource and normalize it.

    Raises:
        ValueError: If the provided string cannot be coerced into a URL.
    """
    normalized = normalize_url(raw_url)
    if not normalized:
        raise ValueError(f"Unsupported URL '{raw_url}'")
    return normalized


def domain_of(raw_url: str) -> str:
    """
    Return the normalized domain for a URL or an empty string if invalid.
    """
    normalized = normalize_url(raw_url)
    return urlparse(normalized).netloc if normalized else ""


def _normalize_path(path: str) -> str:
    if not path or path == "/":
        return "/"

    cleaned = path.rstrip("/")
    return cleaned or "/"

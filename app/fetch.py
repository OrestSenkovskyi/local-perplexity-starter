"""Full-page content extraction for the top-K search results.

Snippets from DuckDuckGo are ~150 characters - too thin for good grounded
answers. This module fetches each URL and pulls out the main article text
using `trafilatura`. Runs fully in parallel with bounded timeouts so one
slow site doesn't block the pipeline.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

try:
    import trafilatura  # type: ignore
    _HAS_TRAFILATURA = True
except Exception:  # noqa: BLE001
    trafilatura = None  # type: ignore
    _HAS_TRAFILATURA = False

from app.config import settings

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


async def _fetch_one(client: httpx.AsyncClient, url: str) -> str:
    """Fetch one URL and extract main content; return empty string on failure."""
    if not _HAS_TRAFILATURA:
        return ""
    try:
        r = await client.get(url, headers={"User-Agent": _USER_AGENT})
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.debug("fetch failed for %s: %s", url, exc)
        return ""
    try:
        text = trafilatura.extract(
            r.text,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        ) or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("extract failed for %s: %s", url, exc)
        return ""
    return text.strip()


async def enrich(sources: list[dict], top_k: int | None = None) -> list[dict]:
    """Fetch full-page content for the first K sources in-place.

    Each enriched source gets a new `content` key containing cleaned body
    text. The original `snippet` is left untouched as a fallback.
    """
    if not sources or not _HAS_TRAFILATURA:
        return sources

    k = top_k if top_k is not None else settings.page_fetch_top_k
    targets = sources[:k]

    timeout = httpx.Timeout(settings.page_fetch_timeout)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        results = await asyncio.gather(
            *(_fetch_one(client, s["url"]) for s in targets),
            return_exceptions=True,
        )

    for src, content in zip(targets, results):
        if isinstance(content, str) and content:
            src["content"] = content
    return sources

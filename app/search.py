import logging

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from app.config import settings

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(0.5),
    retry=retry_if_exception_type(DuckDuckGoSearchException),
    reraise=True,
)
def _raw_search(query: str) -> list[dict]:
    return DDGS().text(
        keywords=query,
        region=settings.search_region,
        safesearch=settings.search_safesearch,
        backend=settings.search_backend,
        max_results=settings.search_max_results,
    ) or []


def run_search(query: str) -> list[dict]:
    """Run a DuckDuckGo text search and return normalized results.

    Returns a list of dicts: [{title, url, snippet}]
    Returns an empty list on any search failure.
    """
    try:
        raw = _raw_search(query)
    except DuckDuckGoSearchException as exc:
        logger.warning("DuckDuckGo search failed after retry: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected search error: %s", exc)
        return []

    # Normalize and deduplicate by URL.
    seen: set[str] = set()
    results: list[dict] = []
    for item in raw:
        url = item.get("href", "")
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("body", ""),
            }
        )
    return results

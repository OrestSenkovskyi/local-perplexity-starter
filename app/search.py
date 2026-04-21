import logging

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from app.config import settings

logger = logging.getLogger(__name__)


def run_search(query: str) -> list[dict]:
    """Run a DuckDuckGo text search and return normalized results.

    Returns a list of dicts: [{title, url, snippet}]
    Returns an empty list on any search failure.
    """
    try:
        raw = DDGS().text(
            keywords=query,
            region=settings.search_region,
            safesearch=settings.search_safesearch,
            backend=settings.search_backend,
            max_results=settings.search_max_results,
        )
    except DuckDuckGoSearchException as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected search error: %s", exc)
        return []

    # Normalize and deduplicate by URL
    seen: set[str] = set()
    results: list[dict] = []
    for item in raw or []:
        url = item.get("href", "")
        if url in seen:
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

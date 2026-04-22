import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app import fetch as fetch_mod


def test_enrich_empty_returns_empty():
    out = asyncio.run(fetch_mod.enrich([]))
    assert out == []


def test_enrich_without_trafilatura_is_noop():
    sources = [{"url": "https://a", "title": "T", "snippet": "s"}]
    with patch.object(fetch_mod, "_HAS_TRAFILATURA", False):
        out = asyncio.run(fetch_mod.enrich(sources))
    assert out[0].get("content") is None


def test_enrich_populates_content_on_success():
    sources = [
        {"url": "https://a", "title": "T1", "snippet": "s1"},
        {"url": "https://b", "title": "T2", "snippet": "s2"},
    ]

    async def fake_fetch_one(_client, url):
        return f"body for {url}"

    with patch.object(fetch_mod, "_HAS_TRAFILATURA", True), \
         patch.object(fetch_mod, "_fetch_one", side_effect=fake_fetch_one):
        out = asyncio.run(fetch_mod.enrich(sources, top_k=2))

    assert out[0]["content"] == "body for https://a"
    assert out[1]["content"] == "body for https://b"


def test_enrich_skips_empty_extractions():
    sources = [
        {"url": "https://a", "title": "T1", "snippet": "s1"},
        {"url": "https://b", "title": "T2", "snippet": "s2"},
    ]

    async def fake_fetch_one(_client, url):
        return "" if url.endswith("a") else "good"

    with patch.object(fetch_mod, "_HAS_TRAFILATURA", True), \
         patch.object(fetch_mod, "_fetch_one", side_effect=fake_fetch_one):
        out = asyncio.run(fetch_mod.enrich(sources, top_k=2))

    assert "content" not in out[0]
    assert out[1]["content"] == "good"

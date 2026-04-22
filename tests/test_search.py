from unittest.mock import patch

from app import search as search_mod
from duckduckgo_search.exceptions import DuckDuckGoSearchException


def test_run_search_normalizes_and_dedups():
    raw = [
        {"title": "A", "href": "https://a", "body": "snippet a"},
        {"title": "B", "href": "https://b", "body": "snippet b"},
        {"title": "A-dup", "href": "https://a", "body": "dup"},
        {"title": "C", "href": "", "body": "no url"},
    ]
    with patch.object(search_mod, "_raw_search", return_value=raw):
        results = search_mod.run_search("q")
    assert [r["url"] for r in results] == ["https://a", "https://b"]
    assert results[0]["snippet"] == "snippet a"
    assert results[0]["title"] == "A"


def test_run_search_swallows_exceptions():
    with patch.object(
        search_mod, "_raw_search", side_effect=DuckDuckGoSearchException("boom")
    ):
        assert search_mod.run_search("q") == []


def test_run_search_swallows_unexpected_exceptions():
    with patch.object(search_mod, "_raw_search", side_effect=RuntimeError("weird")):
        assert search_mod.run_search("q") == []


def test_raw_search_retries_on_ddg_exception():
    calls = {"n": 0}

    def fake_text(self, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise DuckDuckGoSearchException("rate limited")
        return [{"title": "T", "href": "u", "body": "b"}]

    with patch.object(search_mod.DDGS, "text", fake_text):
        out = search_mod._raw_search("q")
    assert calls["n"] == 2
    assert out == [{"title": "T", "href": "u", "body": "b"}]

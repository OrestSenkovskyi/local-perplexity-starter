"""End-to-end tests for the /chat and /health routes.

We stub the OpenAI client and the DuckDuckGo search so no network is touched.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import main as main_mod


class StubLLM:
    """Minimal stand-in for openai.OpenAI.

    `script` is a list of callables or strings, one per chat.completions.create
    invocation. Strings are returned verbatim as the assistant message content;
    callables receive kwargs and return the content string.
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

        stub = self

        class _Completions:
            def create(self, **kwargs):
                stub.calls.append(kwargs)
                if not stub._script:
                    raise AssertionError("StubLLM ran out of scripted responses")
                item = stub._script.pop(0)
                content = item(kwargs) if callable(item) else item
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content=content))
                    ]
                )

        class _Chat:
            completions = _Completions()

        class _Models:
            def list(self_inner):
                return SimpleNamespace(data=[SimpleNamespace(id="stub-model")])

        self.chat = _Chat()
        self.models = _Models()


@pytest.fixture
def client_with_stub(monkeypatch):
    """Boot the FastAPI app with a controllable LLM and no real search/fetch."""

    # Disable page fetch so we don't hit the network
    monkeypatch.setattr(main_mod.settings, "enable_page_fetch", False)
    # Disable factsheet so simple tests only need one answer call
    monkeypatch.setattr(main_mod.settings, "enable_factsheet", False)
    monkeypatch.setattr(main_mod.settings, "enable_query_rewrite", False)

    script_holder: dict = {"script": []}

    def make_client(**_kwargs):
        return StubLLM(script_holder["script"])

    monkeypatch.setattr(main_mod.openai, "OpenAI", make_client)

    def _run(script, search_results=None):
        script_holder["script"] = script
        with patch.object(main_mod, "run_search", return_value=(search_results or [])):
            with TestClient(main_mod.app) as tc:
                yield tc

    return _run


def test_health_endpoint(client_with_stub):
    for tc in client_with_stub(script=["{\"need_search\": false, \"query\": \"\"}"]):
        r = tc.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"] == "stub-model"
        assert body["lm_studio_reachable"] is True


def test_chat_empty_message_rejected(client_with_stub):
    for tc in client_with_stub(script=[]):
        r = tc.post("/chat", json={"message": "   "})
        assert r.status_code == 400


def test_chat_trivial_fastpath_skips_router(client_with_stub):
    """A greeting should go straight to the offline answer (1 LLM call)."""
    for tc in client_with_stub(script=["Hello! How can I help?"]):
        r = tc.post("/chat", json={"message": "hi"})
        assert r.status_code == 200
        body = r.json()
        assert body["searched"] is False
        assert body["query"] is None
        assert body["answer"] == "Hello! How can I help?"
        # Fast-path means router_ms should be 0.
        assert body["timings"]["router_ms"] == 0
        assert body["timings"]["answer_ms"] >= 0
        assert body["cached"] is False


def test_chat_router_no_search(client_with_stub):
    """Router says no search -> 2 LLM calls (router + answer)."""
    script = [
        '{"need_search": false, "query": ""}',
        "2 plus 2 equals 4.",
    ]
    for tc in client_with_stub(script=script):
        r = tc.post("/chat", json={"message": "What is 2+2?"})
        assert r.status_code == 200
        body = r.json()
        assert body["searched"] is False
        assert body["answer"].startswith("2 plus 2")
        assert body["timings"]["router_ms"] >= 0
        assert body["timings"]["search_ms"] == 0


def test_chat_router_triggers_search(client_with_stub):
    """Router says search -> search stage runs and sources come back."""
    script = [
        '{"need_search": true, "query": "latest python version"}',
        "Python 3.14 is latest [1].",
    ]
    sources = [{"title": "PyOrg", "url": "https://python.org", "snippet": "Python 3.14..."}]
    for tc in client_with_stub(script=script, search_results=sources):
        r = tc.post("/chat", json={"message": "What's the newest Python?"})
        assert r.status_code == 200
        body = r.json()
        assert body["searched"] is True
        assert body["query"] == "latest python version"
        assert len(body["sources"]) == 1
        assert body["sources"][0]["url"] == "https://python.org"
        assert body["timings"]["total_ms"] >= 0


def test_chat_force_search_skips_router(client_with_stub):
    """force_search=true -> no router call, straight to search + answer."""
    script = ["Here is the price [1]."]
    sources = [{"title": "BTC", "url": "https://btc", "snippet": "price..."}]
    for tc in client_with_stub(script=script, search_results=sources):
        r = tc.post(
            "/chat",
            json={"message": "Bitcoin price", "force_search": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["searched"] is True
        assert body["query"] == "Bitcoin price"
        # Router was bypassed.
        assert body["timings"]["router_ms"] == 0


def test_chat_cache_hit_marks_cached(client_with_stub):
    """Repeat the same request -> second response has cached=true and no new LLM calls."""
    script = [
        '{"need_search": false, "query": ""}',
        "First answer.",
    ]
    for tc in client_with_stub(script=script):
        first = tc.post("/chat", json={"message": "What is 17*23?"}).json()
        second = tc.post("/chat", json={"message": "What is 17*23?"}).json()
        assert first["cached"] is False
        assert second["cached"] is True
        assert first["answer"] == second["answer"] == "First answer."

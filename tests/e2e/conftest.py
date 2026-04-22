"""Shared fixtures for the e2e test suite.

Architecture
────────────
The `e2e` fixture:
  1. Patches openai.OpenAI with StubLLM *before* the FastAPI lifespan runs.
  2. Patches run_search with a side_effect that reads from a shared holder.
  3. Boots the ASGI app via httpx.AsyncClient + httpx.ASGITransport.
  4. Disables all optional pipeline stages (page_fetch, factsheet,
     query_rewrite) by default so every test controls exactly how many
     LLM calls are expected.

Tests call `e2e.configure(llm=[...], search=[...])` before each request.
Because StubLLM holds a *reference* to the shared holder dict (not a copy),
configure() updates are seen immediately by the already-running client.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio

from app import main as main_mod


# ---------------------------------------------------------------------------
# StubLLM — holds a *reference* to the shared holder so configure() works
# ---------------------------------------------------------------------------
class StubLLM:
    def __init__(self, holder: dict):
        self._holder = holder
        stub = self

        class _Completions:
            def create(self_inner, **kwargs):
                responses = stub._holder["responses"]
                if not responses:
                    raise AssertionError(
                        "StubLLM exhausted — unexpected extra LLM call.\n"
                        f"messages={kwargs.get('messages', [{}])[0].get('content', '')[:80]!r}"
                    )
                item = responses.pop(0)
                content = item(kwargs) if callable(item) else item
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        class _Chat:
            completions = _Completions()

        class _Models:
            def list(self_inner):
                return SimpleNamespace(data=[SimpleNamespace(id="e2e-stub-model")])

        self.chat = _Chat()
        self.models = _Models()


# ---------------------------------------------------------------------------
# e2e fixture
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def e2e(monkeypatch):
    """Yield a helper that owns both the HTTP client and the script holder."""

    # Disable all optional stages; individual tests opt-in as needed.
    monkeypatch.setattr(main_mod.settings, "enable_page_fetch", False)
    monkeypatch.setattr(main_mod.settings, "enable_factsheet", False)
    monkeypatch.setattr(main_mod.settings, "enable_query_rewrite", False)

    holder: dict = {"responses": [], "search": []}

    # Replace openai.OpenAI so the lifespan gets our StubLLM.
    monkeypatch.setattr(main_mod.openai, "OpenAI", lambda **_kw: StubLLM(holder))

    class _E2EClient:
        def configure(self, *, llm: list, search: list | None = None):
            # Mutate the *same* dict the already-running StubLLM references.
            holder["responses"] = list(llm)
            holder["search"] = list(search or [])

    wrapper = _E2EClient()

    with patch.object(main_mod, "run_search", side_effect=lambda _q: list(holder["search"])):
        transport = httpx.ASGITransport(app=main_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with main_mod.app.router.lifespan_context(main_mod.app):
                wrapper.http = client
                yield wrapper

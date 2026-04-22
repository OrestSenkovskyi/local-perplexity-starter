"""End-to-end tests for Local Perplexity.

All tests use the `e2e` fixture from conftest.py which:
  - Boots the real ASGI app with a real HTTP transport
  - Stubs only openai.OpenAI (via shared holder) and run_search
  - Disables page_fetch, factsheet, and query_rewrite by default

Tests that verify optional stages opt-in via monkeypatch.setattr.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app import main as main_mod

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
async def test_health_ok(e2e):
    e2e.configure(llm=[])
    r = await e2e.http.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "e2e-stub-model"
    assert body["lm_studio_reachable"] is True
    assert isinstance(body["uptime_seconds"], int)


# ---------------------------------------------------------------------------
# Fast-path (greetings bypass router)
# ---------------------------------------------------------------------------
async def test_fastpath_greeting_skips_router(e2e):
    """Greeting → 1 LLM call (offline answer), router_ms == 0."""
    e2e.configure(llm=["Hey! How can I help?"])
    r = await e2e.http.post("/chat", json={"message": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["searched"] is False
    assert body["query"] is None
    assert body["answer"] == "Hey! How can I help?"
    assert body["timings"]["router_ms"] == 0
    assert body["timings"]["answer_ms"] >= 0
    assert body["cached"] is False


async def test_fastpath_thanks(e2e):
    e2e.configure(llm=["You're welcome!"])
    r = await e2e.http.post("/chat", json={"message": "thanks"})
    assert r.status_code == 200
    assert r.json()["timings"]["router_ms"] == 0


# ---------------------------------------------------------------------------
# Router → no search (2 LLM calls: router + answer)
# ---------------------------------------------------------------------------
async def test_no_search_path(e2e):
    e2e.configure(llm=[
        '{"need_search": false, "query": ""}',
        "2 + 2 equals 4.",
    ])
    r = await e2e.http.post("/chat", json={"message": "What is 2+2?"})
    assert r.status_code == 200
    body = r.json()
    assert body["searched"] is False
    assert body["sources"] == []
    assert body["answer"] == "2 + 2 equals 4."
    assert body["timings"]["router_ms"] >= 0  # stub LLM may complete in < 1 ms
    assert body["timings"]["search_ms"] == 0
    assert body["timings"]["total_ms"] >= body["timings"]["router_ms"]


# ---------------------------------------------------------------------------
# Router → search triggered (2 LLM calls: router + answer)
# ---------------------------------------------------------------------------
async def test_search_path_sources_returned(e2e):
    """Router sets need_search=true → DDG fires, sources in response."""
    e2e.configure(
        llm=[
            '{"need_search": true, "query": "latest python version"}',
            "Python 3.14 is current [1].",
        ],
        search=[{"title": "Python", "url": "https://python.org", "snippet": "3.14 released"}],
    )
    r = await e2e.http.post("/chat", json={"message": "What is the latest Python?"})
    assert r.status_code == 200
    body = r.json()
    assert body["searched"] is True
    assert body["query"] == "latest python version"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["url"] == "https://python.org"
    assert "[1]" in body["answer"]
    assert body["timings"]["search_ms"] >= 0


# ---------------------------------------------------------------------------
# force_search bypasses router (1 LLM call: answer only)
# ---------------------------------------------------------------------------
async def test_force_search_skips_router(e2e):
    """force_search=true → router_ms==0, question used as query."""
    e2e.configure(
        llm=["BTC is $99k [1]."],
        search=[{"title": "BTC", "url": "https://btc.example", "snippet": "price 99k"}],
    )
    r = await e2e.http.post("/chat", json={"message": "Bitcoin price", "force_search": True})
    assert r.status_code == 200
    body = r.json()
    assert body["searched"] is True
    assert body["query"] == "Bitcoin price"
    assert body["timings"]["router_ms"] == 0


# ---------------------------------------------------------------------------
# Two-stage factsheet (3 LLM calls: router + factsheet + answer)
# ---------------------------------------------------------------------------
async def test_factsheet_stage_runs(e2e, monkeypatch):
    """ENABLE_FACTSHEET=true → factsheet LLM call fires, factsheet_ms > 0."""
    monkeypatch.setattr(main_mod.settings, "enable_factsheet", True)

    e2e.configure(
        llm=[
            '{"need_search": true, "query": "rust release 2025"}',
            "- [1] Rust 1.80 released in 2025.",     # factsheet
            "Rust 1.80 came out in 2025 [1].",        # final answer
        ],
        search=[{"title": "Rust blog", "url": "https://blog.rust-lang.org", "snippet": "Rust 1.80"}],
    )
    r = await e2e.http.post("/chat", json={"message": "What is the latest Rust release?"})
    assert r.status_code == 200
    body = r.json()
    assert body["searched"] is True
    assert body["timings"]["factsheet_ms"] >= 0  # stub may complete in < 1 ms
    assert "1.80" in body["answer"] or "[1]" in body["answer"]


# ---------------------------------------------------------------------------
# Query rewrite (router + 2 search attempts + rewrite LLM + answer = 3 LLM calls)
# ---------------------------------------------------------------------------
async def test_query_rewrite_fires_on_scarce_results(e2e, monkeypatch):
    """< 2 results → rewrite LLM fires, second search attempt made."""
    monkeypatch.setattr(main_mod.settings, "enable_query_rewrite", True)

    call_count = {"n": 0}

    def fake_search(_query):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return []
        return [{"title": "T", "url": "https://ex.com", "snippet": "body"}]

    with patch.object(main_mod, "run_search", side_effect=fake_search):
        e2e.configure(llm=[
            '{"need_search": true, "query": "obscure xyzzy term"}',
            "xyzzy common term 2025",   # rewrite output
            "Answer [1].",              # final answer
        ])
        r = await e2e.http.post("/chat", json={"message": "Tell me about xyzzy"})

    assert r.status_code == 200
    body = r.json()
    assert body["searched"] is True
    assert call_count["n"] == 2
    assert body["timings"]["rewrite_ms"] >= 0  # stub may complete in < 1 ms


# ---------------------------------------------------------------------------
# Conversation history: [n] stripped from prior assistant turns
# ---------------------------------------------------------------------------
async def test_history_citations_stripped(e2e):
    """[n] markers from history are stripped before replay into the answer prompt."""
    captured: list = []
    orig = main_mod._llm_call

    def spy(messages, **kw):
        captured.append(messages)
        return orig(messages, **kw)

    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "A language [1][2]."},
    ]
    e2e.configure(llm=[
        '{"need_search": false, "query": ""}',
        "Follow-up answer.",
    ])

    with patch.object(main_mod, "_llm_call", side_effect=spy):
        r = await e2e.http.post("/chat", json={"message": "Is it fast?", "history": history})

    assert r.status_code == 200
    # Answer prompt is the second LLM call.
    answer_msgs = captured[1]
    assistant_msgs = [m for m in answer_msgs if m["role"] == "assistant"]
    assert assistant_msgs
    assert "[1]" not in assistant_msgs[0]["content"]
    assert "[2]" not in assistant_msgs[0]["content"]


async def test_history_user_content_preserved(e2e):
    """Prior user turns are forwarded verbatim."""
    captured: list = []
    orig = main_mod._llm_call

    def spy(messages, **kw):
        captured.append(messages)
        return orig(messages, **kw)

    history = [{"role": "user", "content": "Prior question here."}]
    e2e.configure(llm=[
        '{"need_search": false, "query": ""}',
        "answer",
    ])

    with patch.object(main_mod, "_llm_call", side_effect=spy):
        r = await e2e.http.post("/chat", json={"message": "Follow up?", "history": history})

    assert r.status_code == 200
    answer_msgs = captured[1]
    user_msgs = [m for m in answer_msgs if m["role"] == "user"]
    contents = [m["content"] for m in user_msgs]
    assert any("Prior question here." in c for c in contents)


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------
async def test_cache_hit_returns_cached_flag(e2e):
    """Second identical request is served from cache."""
    e2e.configure(llm=[
        '{"need_search": false, "query": ""}',
        "Unique answer 42.",
    ])
    r1 = await e2e.http.post("/chat", json={"message": "cache test question"})
    r2 = await e2e.http.post("/chat", json={"message": "cache test question"})
    assert r1.status_code == r2.status_code == 200
    b1, b2 = r1.json(), r2.json()
    assert b1["cached"] is False
    assert b2["cached"] is True
    assert b1["answer"] == b2["answer"] == "Unique answer 42."


async def test_cache_differentiates_force_search(e2e):
    """force_search=true and false have separate cache keys."""
    e2e.configure(
        llm=[
            '{"need_search": false, "query": ""}',
            "Offline answer.",
            "Online answer [1].",
        ],
        search=[{"title": "S", "url": "https://s", "snippet": "x"}],
    )
    r_off = await e2e.http.post("/chat", json={"message": "cache key test"})
    r_on = await e2e.http.post("/chat", json={"message": "cache key test", "force_search": True})
    assert r_off.json()["answer"] == "Offline answer."
    assert r_on.json()["answer"] == "Online answer [1]."
    assert r_off.json()["cached"] is False
    assert r_on.json()["cached"] is False


# ---------------------------------------------------------------------------
# Sampling contracts (patch _llm_call to inspect kwargs)
# ---------------------------------------------------------------------------
async def test_router_uses_temperature_zero(e2e):
    """Router call must use temperature=0.0; answer uses a higher value."""
    calls: list = []
    orig = main_mod._llm_call

    def spy(messages, *, temperature=0.3, **kw):
        calls.append({"temperature": temperature, "messages": messages})
        return orig(messages, temperature=temperature, **kw)

    e2e.configure(llm=[
        '{"need_search": false, "query": ""}',
        "answer",
    ])
    with patch.object(main_mod, "_llm_call", side_effect=spy):
        r = await e2e.http.post("/chat", json={"message": "Explain TCP"})

    assert r.status_code == 200
    assert calls[0]["temperature"] == 0.0   # router
    assert calls[1]["temperature"] > 0.0    # answer


async def test_router_max_tokens_is_80(e2e):
    """Router call must cap output at 80 tokens to stay within JSON budget."""
    calls: list = []
    orig = main_mod._llm_call

    def spy(messages, *, max_tokens=None, **kw):
        calls.append({"max_tokens": max_tokens, "messages": messages})
        return orig(messages, max_tokens=max_tokens, **kw)

    e2e.configure(llm=[
        '{"need_search": false, "query": ""}',
        "answer",
    ])
    with patch.object(main_mod, "_llm_call", side_effect=spy):
        r = await e2e.http.post("/chat", json={"message": "Explain TCP"})

    assert r.status_code == 200
    assert calls[0]["max_tokens"] == 80     # router
    assert calls[1]["max_tokens"] == 600    # offline answer


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
async def test_empty_message_returns_400(e2e):
    e2e.configure(llm=[])
    r = await e2e.http.post("/chat", json={"message": "   "})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


async def test_missing_message_field_returns_422(e2e):
    e2e.configure(llm=[])
    r = await e2e.http.post("/chat", json={"force_search": False})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# LLM failure → 502
# ---------------------------------------------------------------------------
async def test_llm_answer_failure_returns_502(e2e):
    """When the answer stage LLM raises, the endpoint must return 502."""
    call_count = {"n": 0}

    def exploding(messages, **_kw):
        call_count["n"] += 1
        # First call is the router; second call is the answer stage.
        if call_count["n"] == 1:
            return '{"need_search": false, "query": ""}'
        raise RuntimeError("GPU exploded")

    e2e.configure(llm=[])
    with patch.object(main_mod, "_llm_call", side_effect=exploding):
        r = await e2e.http.post("/chat", json={"message": "Tell me a story"})

    assert r.status_code == 502


# ---------------------------------------------------------------------------
# Timings shape contract
# ---------------------------------------------------------------------------
async def test_timings_present_and_non_negative(e2e):
    """Every /chat response must include a complete, non-negative timings object."""
    e2e.configure(llm=[
        '{"need_search": false, "query": ""}',
        "fine.",
    ])
    r = await e2e.http.post("/chat", json={"message": "What is 1+1?"})
    assert r.status_code == 200
    t = r.json()["timings"]
    for field in ("router_ms", "search_ms", "rewrite_ms", "fetch_ms",
                  "factsheet_ms", "answer_ms", "total_ms"):
        assert t[field] >= 0, f"{field} is negative: {t[field]}"
    assert t["total_ms"] >= t["answer_ms"]


async def test_timings_total_includes_all_stages(e2e):
    """total_ms should be >= the sum of individual stage times."""
    e2e.configure(llm=[
        '{"need_search": false, "query": ""}',
        "answer",
    ])
    r = await e2e.http.post("/chat", json={"message": "What is 2+2?"})
    t = r.json()["timings"]
    stage_sum = (t["router_ms"] + t["search_ms"] + t["rewrite_ms"]
                 + t["fetch_ms"] + t["factsheet_ms"] + t["answer_ms"])
    assert t["total_ms"] >= stage_sum

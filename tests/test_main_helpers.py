from app.main import (
    ChatRequest,
    HistoryMessage,
    _cache_key,
    _is_trivial,
    _parse_router_response,
)


def test_parse_router_response_plain_json():
    out = _parse_router_response('{"need_search": true, "query": "foo"}')
    assert out == {"need_search": True, "query": "foo"}


def test_parse_router_response_markdown_fenced():
    raw = '```json\n{"need_search": false, "query": ""}\n```'
    out = _parse_router_response(raw)
    assert out == {"need_search": False, "query": ""}


def test_parse_router_response_noise_around_json():
    raw = "Sure thing! {\"need_search\": true, \"query\": \"bar\"} hope that helps"
    out = _parse_router_response(raw)
    assert out["need_search"] is True
    assert out["query"] == "bar"


def test_parse_router_response_invalid_falls_back():
    out = _parse_router_response("not json at all")
    assert out == {"need_search": False, "query": ""}


def test_is_trivial_greetings():
    for q in ["hi", "Hello!", "thanks", "thx", "ok", "yes", "bye"]:
        assert _is_trivial(q), q


def test_is_trivial_short():
    assert _is_trivial("ab")
    assert _is_trivial("   ")


def test_is_trivial_false_for_real_questions():
    for q in [
        "What is the latest Python version?",
        "Explain TCP handshake",
        "who won F1 today",
    ]:
        assert not _is_trivial(q), q


def test_cache_key_deterministic_and_differentiating():
    r1 = ChatRequest(message="same", force_search=False, history=[])
    r2 = ChatRequest(message="same", force_search=False, history=[])
    r3 = ChatRequest(message="same", force_search=True, history=[])
    r4 = ChatRequest(
        message="same",
        force_search=False,
        history=[HistoryMessage(role="user", content="prior")],
    )
    assert _cache_key(r1) == _cache_key(r2)
    assert _cache_key(r1) != _cache_key(r3)
    assert _cache_key(r1) != _cache_key(r4)

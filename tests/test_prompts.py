from app.prompts import (
    ANSWER_SYSTEM_GROUNDED,
    ANSWER_SYSTEM_OFFLINE,
    FACTSHEET_SYSTEM,
    PAGE_MAX_CHARS,
    ROUTER_SYSTEM,
    SNIPPET_MAX_CHARS,
    _format_sources,
    _strip_citations,
    _trim_history,
    answer_prompt,
    factsheet_prompt,
    rewrite_prompt,
    router_prompt,
)


def test_router_prompt_structure():
    msgs = router_prompt("What is 2+2?")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == ROUTER_SYSTEM
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "What is 2+2?"


def test_rewrite_prompt_contains_originals():
    msgs = rewrite_prompt("python ver", "What is the newest Python?")
    assert "python ver" in msgs[1]["content"]
    assert "What is the newest Python?" in msgs[1]["content"]


def test_format_sources_uses_content_when_present():
    results = [
        {"title": "T1", "url": "https://a", "snippet": "short snip", "content": "X" * 3000},
        {"title": "T2", "url": "https://b", "snippet": "Y" * 1000},
    ]
    formatted = _format_sources(results)
    # [1] uses content, trimmed to PAGE_MAX_CHARS
    assert formatted.count("X") == PAGE_MAX_CHARS
    # [2] uses snippet, trimmed to SNIPPET_MAX_CHARS
    assert formatted.count("Y") == SNIPPET_MAX_CHARS
    assert "[1] T1" in formatted
    assert "[2] T2" in formatted


def test_strip_citations():
    assert _strip_citations("Foo [1] bar [2][3].") == "Foo bar."
    assert _strip_citations("plain text") == "plain text"


def test_trim_history_respects_max_turns():
    history = [
        {"role": "user", "content": str(i)} for i in range(20)
    ]
    trimmed = _trim_history(history, max_turns=3)
    # 3 turns = 6 messages
    assert len(trimmed) == 6
    assert trimmed[0]["content"] == "14"


def test_answer_prompt_grounded_uses_factsheet():
    results = [{"title": "T", "url": "u", "snippet": "s"}]
    msgs = answer_prompt(
        "q", results, searched=True, search_query="q",
        history=None, factsheet="- [1] fact",
    )
    assert msgs[0]["content"] == ANSWER_SYSTEM_GROUNDED.format(
        lang=__import__("app.prompts", fromlist=["_lang"])._lang()
    )
    last = msgs[-1]["content"]
    assert "Factsheet" in last
    assert "- [1] fact" in last


def test_answer_prompt_grounded_without_factsheet_uses_sources():
    results = [{"title": "T", "url": "u", "snippet": "body"}]
    msgs = answer_prompt("q", results, searched=True, search_query="q")
    last = msgs[-1]["content"]
    assert "Sources:" in last
    assert "[1] T" in last


def test_answer_prompt_offline_when_no_results():
    msgs = answer_prompt("q", [], searched=False, search_query=None)
    from app.prompts import _lang
    assert msgs[0]["content"] == ANSWER_SYSTEM_OFFLINE.format(lang=_lang())


def test_answer_prompt_strips_citations_from_history():
    class H:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    history = [
        H("user", "prior"),
        H("assistant", "answer with [1] and [2]"),
    ]
    msgs = answer_prompt("q2", [], searched=False, search_query=None, history=history)
    # Find the assistant turn in the constructed messages.
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    assert assistant_msgs
    assert "[1]" not in assistant_msgs[0]["content"]
    assert "[2]" not in assistant_msgs[0]["content"]


def test_factsheet_prompt_system_is_factsheet():
    msgs = factsheet_prompt("q", [{"title": "T", "url": "u", "snippet": "s"}])
    assert msgs[0]["content"] == FACTSHEET_SYSTEM

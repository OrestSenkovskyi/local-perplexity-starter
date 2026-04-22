"""Prompt templates tuned for small local LLMs (3-8B parameters).

These prompts are intentionally mechanical, example-rich, and constraint-heavy,
which is the style small instruction-tuned models handle best.
"""
from __future__ import annotations

import re

from app.config import settings

LANGUAGE_NAMES = {
    "uk": "Ukrainian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "pl": "Polish",
    "ru": "Russian",
}

# Maximum characters per raw snippet before it is included in a prompt.
SNIPPET_MAX_CHARS = 500
# Maximum characters of extracted page content before it is included in a prompt.
PAGE_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = """You are a strict binary search router. Output ONE JSON object, nothing else.

Schema:
{"need_search": <true|false>, "query": "<string>"}

Set need_search=true for:
- current events, news, prices, scores, weather, schedules
- software versions, release notes, API/library docs
- specific people, companies, or products that change over time
- any mention of: today, now, latest, current, this week, 2024, 2025, 2026

Set need_search=false for:
- arithmetic, logic, pure reasoning
- stable definitions (what is TCP, what is photosynthesis)
- greetings, small talk, opinions, creative writing
- code explanation that does not reference a specific library version

Query rules:
- Under 10 words.
- Strip first-person words, fillers, question marks.
- Prefer nouns and proper names over verbs.
- If need_search is false, set "query" to "".

Examples:
Q: What is 17 * 23?
A: {"need_search": false, "query": ""}

Q: Who won the last F1 race?
A: {"need_search": true, "query": "latest F1 race winner"}

Q: Explain bubble sort.
A: {"need_search": false, "query": ""}

Q: Newest Python 3.13 feature?
A: {"need_search": true, "query": "Python 3.13 new features"}

Q: Thanks!
A: {"need_search": false, "query": ""}

Output ONLY the JSON. No markdown. No prose. No code fences."""


# ---------------------------------------------------------------------------
# Query rewriter (used when retrieval yields too few results)
# ---------------------------------------------------------------------------
REWRITE_SYSTEM = """The previous search query returned too few results. Rewrite it.

Output ONLY the new query on a single line. No quotes, no prose.

Guidelines:
- Remove rare jargon; replace with common terms.
- Add a year if the topic is time-bound.
- Convert question phrasing into keyword phrasing.
- Keep it under 10 words."""


# ---------------------------------------------------------------------------
# Factsheet extractor (first stage of two-stage answer)
# ---------------------------------------------------------------------------
FACTSHEET_SYSTEM = """Extract facts from the search results below. Do not answer the user's question yet.

Output format (markdown bullet list):
- [n] <atomic fact copied or tightly paraphrased from source n>
- [n] <...>

Rules:
- One fact per bullet.
- Every bullet MUST start with [n] where n is the source number.
- Only include facts directly relevant to the question.
- Copy numbers, dates, and versions verbatim.
- If a source is irrelevant, skip it silently.
- No introduction, no conclusion, no prose."""


# ---------------------------------------------------------------------------
# Grounded answer (search context present)
# ---------------------------------------------------------------------------
ANSWER_SYSTEM_GROUNDED = """You are a precise search assistant. Answer using ONLY the provided context.

Output language: {lang}.

Rules:
1. Ground every factual claim in a source. Cite with [n] immediately after the claim.
2. Combine sources when appropriate: "Adopted widely in 2025 [1][3]."
3. If the context does not contain the answer, say so explicitly and stop. Never invent facts.
4. Never cite a number that is not in the provided list.
5. Copy dates, versions, and numbers verbatim from the context.
6. Short paragraphs or bullet lists. No walls of text.
7. Do not repeat the question. Do not apologise. No "As an AI" disclaimers.
8. If sources conflict, note the disagreement and cite both sides.
9. Do NOT add a "Sources:" section - the UI renders sources separately."""


# ---------------------------------------------------------------------------
# Offline answer (no search)
# ---------------------------------------------------------------------------
ANSWER_SYSTEM_OFFLINE = """You are a helpful assistant. Answer from general knowledge.

Output language: {lang}.

Rules:
1. If the question requires information that changes over time (prices, versions, news, schedules), respond exactly:
   "I don't have up-to-date information on that. Enable 'Always search the web' for a live answer."
   Then stop.
2. For stable knowledge, answer concisely (1-3 short paragraphs).
3. Use markdown: **bold** key terms, lists for enumerations, fenced code blocks for code.
4. State uncertainty ("I'm not sure, but...") rather than guessing.
5. Do not fabricate citations, URLs, or statistics."""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(r"\s*\[\d+\]")


def _lang() -> str:
    return LANGUAGE_NAMES.get(settings.system_language, settings.system_language)


def router_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": question},
    ]


def rewrite_prompt(previous_query: str, question: str) -> list[dict]:
    user = (
        f"Original question: {question}\n"
        f"Previous query (returned too few results): {previous_query}\n"
        f"Rewrite:"
    )
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _format_sources(results: list[dict]) -> str:
    """Format search results as numbered blocks.

    Uses `content` (extracted page text) when present, otherwise falls back
    to the DDG `snippet`. Each body is trimmed to keep the context window
    under control.
    """
    blocks = []
    for i, r in enumerate(results, start=1):
        body = r.get("content") or r.get("snippet") or ""
        max_len = PAGE_MAX_CHARS if r.get("content") else SNIPPET_MAX_CHARS
        body = body[:max_len]
        blocks.append(f"[{i}] {r.get('title', '')}\n{r.get('url', '')}\n{body}")
    return "\n\n".join(blocks)


def factsheet_prompt(question: str, results: list[dict]) -> list[dict]:
    sources = _format_sources(results)
    user = f"Question: {question}\n\nSearch results:\n{sources}"
    return [
        {"role": "system", "content": FACTSHEET_SYSTEM},
        {"role": "user", "content": user},
    ]


def _trim_history(history: list | None, max_turns: int) -> list:
    if not history:
        return []
    return history[-max_turns * 2:]


def _strip_citations(text: str) -> str:
    """Remove [n] markers so stale numbers don't leak into new answers."""
    return _CITATION_RE.sub("", text)


def answer_prompt(
    question: str,
    results: list[dict],
    searched: bool,
    search_query: str | None,
    history: list | None = None,
    factsheet: str | None = None,
) -> list[dict]:
    """Build the final answer prompt.

    If `factsheet` is provided (two-stage pipeline), the model sees the
    factsheet as primary context. Otherwise it sees the raw search results.
    """
    lang = _lang()
    trimmed = _trim_history(history, settings.max_history_turns)

    if searched and results:
        system = ANSWER_SYSTEM_GROUNDED.format(lang=lang)
    else:
        system = ANSWER_SYSTEM_OFFLINE.format(lang=lang)

    messages: list[dict] = [{"role": "system", "content": system}]

    # Include prior conversation turns (with citation markers stripped so old
    # source numbers don't leak into the new answer).
    for h in trimmed:
        role = getattr(h, "role", None)
        content = getattr(h, "content", None)
        if role is None and isinstance(h, dict):
            role = h.get("role")
            content = h.get("content")
        if not role or not content:
            continue
        if role == "assistant":
            content = _strip_citations(content)
        messages.append({"role": role, "content": content})

    if searched and results:
        if factsheet:
            user_content = (
                f"Question: {question}\n\n"
                f"Search query: {search_query}\n\n"
                f"Factsheet (pre-extracted; cite with the [n] numbers shown):\n{factsheet}\n\n"
                "Answer the question using ONLY the factsheet above. Cite with [n]."
            )
        else:
            sources = _format_sources(results)
            user_content = (
                f"Question: {question}\n\n"
                f"Search query: {search_query}\n\n"
                f"Sources:\n{sources}\n\n"
                "Answer the question. Cite with [n]."
            )
    else:
        user_content = f"Question: {question}"

    messages.append({"role": "user", "content": user_content})
    return messages

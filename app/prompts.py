from app.config import settings

LANGUAGE_NAMES = {
    "uk": "Ukrainian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "pl": "Polish",
    "ru": "Russian",
}

ROUTER_SYSTEM = """\
You are a search router. Given a user question, decide whether a web search is needed.
Rules:
- If the question is about recent events, news, prices, versions, documentation, or anything \
that requires up-to-date information — set need_search to true.
- If the question is conversational, mathematical, or can be answered from general knowledge — \
set need_search to false.
- Respond ONLY with valid JSON, no markdown, no explanation.
Output format: {"need_search": true, "query": "concise search query"}\
"""


def router_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": question},
    ]


def answer_prompt(
    question: str,
    results: list[dict],
    searched: bool,
    search_query: str | None,
    history: list | None = None,
) -> list[dict]:
    lang = LANGUAGE_NAMES.get(settings.system_language, settings.system_language)

    system = (
        f"You are a helpful search assistant similar to Perplexity AI. "
        f"Always respond in {lang}. "
        "Be concise, accurate, and cite sources by their number [1], [2], etc. "
        "when you use information from them."
    )

    messages: list[dict] = [{"role": "system", "content": system}]

    # Insert prior conversation turns so the model has context.
    for h in (history or []):
        messages.append({"role": h.role, "content": h.content})

    if searched and results:
        snippets = "\n\n".join(
            f"[{i + 1}] {r['title']}\n{r['url']}\n{r['snippet']}"
            for i, r in enumerate(results)
        )
        user_content = (
            f"Question: {question}\n\n"
            f"Search query used: {search_query}\n\n"
            f"Search results:\n{snippets}\n\n"
            "Based on the search results above, answer the question. "
            "Cite sources using [number] notation."
        )
    else:
        user_content = f"Question: {question}\n\nAnswer from your general knowledge."

    messages.append({"role": "user", "content": user_content})
    return messages

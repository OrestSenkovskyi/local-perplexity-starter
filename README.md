# Local Perplexity Starter

A self-hosted Perplexity-style search assistant built on top of **LM Studio** (local LLM) and **DuckDuckGo** (free web search). No cloud APIs or paid subscriptions required.

## Features

- Runs entirely locally — no OpenAI, no paid APIs.
- Connects to any LM Studio model via the OpenAI-compatible API.
- **Multi-stage pipeline** with automatic search routing, query rewriting, full-page extraction, and grounded two-stage answering.
- **Prompts tuned for small local models** (3–8B parameters): mechanical, example-rich, and constraint-heavy.
- Returns answers with numbered citations `[1]`, `[2]`, etc.
- Per-stage timings in every response for easy profiling.
- In-memory TTL cache for repeat questions.
- `/health` endpoint for readiness probes.
- Simple dark-themed web UI and a plain JSON REST API.

## Pipeline architecture

Every `/chat` request flows through a configurable pipeline:

```text
user question
    │
    ▼
┌─────────────┐ trivial? (greeting, "ok", < 4 chars)
│ Fast-path   │──────────────► offline answer
└─────────────┘
    │ otherwise
    ▼
┌─────────────┐ temperature=0.0, max_tokens=80
│ Router LLM  │ returns {"need_search": bool, "query": "..."}
└─────────────┘
    │ need_search=false ──────► offline answer
    │ need_search=true
    ▼
┌─────────────┐ DuckDuckGo + tenacity retry (2 attempts, 0.5s backoff)
│ Search      │ k = SEARCH_MAX_RESULTS
└─────────────┘
    │
    ▼
┌─────────────┐ if len(results) < 2 and ENABLE_QUERY_REWRITE
│ Rewrite LLM │ rewrite query with REWRITE_SYSTEM prompt, re-search
└─────────────┘
    │
    ▼
┌─────────────┐ if ENABLE_PAGE_FETCH
│ Enrich      │ fetch top PAGE_FETCH_TOP_K URLs in parallel,
│             │ extract article text with trafilatura (5s timeout)
└─────────────┘
    │
    ▼
┌─────────────┐ if ENABLE_FACTSHEET (two-stage answer)
│ Factsheet   │ LLM extracts "- [n] fact" bullets with temperature=0.1
│ LLM         │
└─────────────┘
    │
    ▼
┌─────────────┐ temperature=0.2, top_p=0.9, max_tokens=800
│ Answer LLM  │ uses ANSWER_SYSTEM_GROUNDED + factsheet OR sources
└─────────────┘
    │
    ▼
response: {answer, sources, searched, query, cached, timings}
```

### Stage details

| Stage | When it runs | Prompt | Sampling |
|---|---|---|---|
| **Fast-path** | Greetings, short messages | — | — |
| **Router** | Non-trivial questions, `force_search=false` | `ROUTER_SYSTEM` | `T=0.0`, `max_tokens=80` |
| **Search** | `need_search=true` | — | — |
| **Query rewrite** | `<2` results and `ENABLE_QUERY_REWRITE` | `REWRITE_SYSTEM` | `T=0.2`, `max_tokens=30` |
| **Enrich** | `ENABLE_PAGE_FETCH` and sources present | — | — |
| **Factsheet** | `ENABLE_FACTSHEET` and sources present | `FACTSHEET_SYSTEM` | `T=0.1`, `top_p=0.9`, `max_tokens=600` |
| **Answer (grounded)** | Sources present | `ANSWER_SYSTEM_GROUNDED` | `T=0.2`, `top_p=0.9`, `max_tokens=800` |
| **Answer (offline)** | No sources / router said no | `ANSWER_SYSTEM_OFFLINE` | `T=0.4`, `top_p=0.9`, `max_tokens=600` |

### Prompts

All prompts live in `app/prompts.py`:

- **`ROUTER_SYSTEM`** — strict binary router; returns JSON only; 5 worked in/out examples.
- **`REWRITE_SYSTEM`** — rewrites a failing query into keyword-phrased, time-bounded form.
- **`FACTSHEET_SYSTEM`** — extracts one `[n]` bullet per fact from search results. Used as first stage of two-stage answer.
- **`ANSWER_SYSTEM_GROUNDED`** — 9 explicit rules: cite immediately after each claim, never cite missing numbers, copy numbers verbatim, refuse when context is missing.
- **`ANSWER_SYSTEM_OFFLINE`** — answers from general knowledge; explicit refusal phrase for time-sensitive questions.

The assistant history is trimmed to the last `MAX_HISTORY_TURNS` exchanges, and `[n]` citation markers are stripped from prior assistant turns so old numbers don't leak into new answers.

## Project structure

```text
local-perplexity-starter/
├── .env.example
├── requirements.txt
├── app/
│   ├── config.py       # Settings loaded from .env
│   ├── main.py         # FastAPI app + pipeline orchestration
│   ├── prompts.py      # Prompt templates + message builders
│   ├── search.py       # DuckDuckGo wrapper with retry
│   └── fetch.py        # Full-page extraction (httpx + trafilatura)
├── templates/
│   └── index.html      # Single-page web UI
└── tests/
    ├── test_prompts.py
    ├── test_main_helpers.py
    ├── test_search.py
    ├── test_fetch.py
    └── test_chat_integration.py
```

## Prerequisites

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) with a model downloaded and Local Server running
  (tested with **Gemma 4 e2b**; any instruction-following model works)

## Setup

### 1. Start LM Studio

1. Open LM Studio and download a model.
2. Go to the **Developer** (or **Local Server**) tab.
3. Click **Start Server**.
4. Confirm the server is available at `http://localhost:1234/v1`.

### 2. Install dependencies

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` to taste. See the **Configuration reference** table below for every knob.

### 4. Run the server

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

## Usage

### Web UI

Type a question and press **Ask** or Enter. Check **Always search the web** to bypass the router and force a web search.

### REST API

#### `POST /chat`

Request body:

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | The user's question |
| `force_search` | boolean | no | Skip the router and always run a web search (default: `false`) |
| `history` | array | no | Prior `{role, content}` turns to include as conversation context |

Response body:

| Field | Type | Description |
|---|---|---|
| `answer` | string | LLM-generated answer, may contain `[n]` citation markers |
| `sources` | array | `[{title, url, snippet}]` used as context |
| `searched` | boolean | Whether a web search was performed |
| `query` | string \| null | The search query actually used, or `null` |
| `cached` | boolean | `true` if this response was served from the in-memory cache |
| `timings` | object | `{router_ms, search_ms, rewrite_ms, fetch_ms, factsheet_ms, answer_ms, total_ms}` |

#### `GET /health`

Returns `{status, model, lm_studio_reachable, uptime_seconds}`.
Useful for container liveness probes.

#### Example — automatic web search

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest stable Python version?"}'
```

```json
{
  "answer": "The latest stable Python release is **Python 3.14** [1].",
  "sources": [{"title": "Python downloads", "url": "https://python.org/downloads", "snippet": "..."}],
  "searched": true,
  "query": "latest Python stable version",
  "cached": false,
  "timings": {
    "router_ms": 410,
    "search_ms": 820,
    "rewrite_ms": 0,
    "fetch_ms": 1340,
    "factsheet_ms": 980,
    "answer_ms": 2150,
    "total_ms": 5700
  }
}
```

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | Base URL of the LM Studio local server |
| `LM_STUDIO_API_KEY` | `lm-studio` | API key sent in requests (LM Studio ignores it) |
| `LM_STUDIO_MODEL` | _(empty)_ | Model ID; auto-detects the first available model when blank |
| `APP_HOST` | `127.0.0.1` | FastAPI bind host |
| `APP_PORT` | `8000` | FastAPI bind port |
| `SEARCH_REGION` | `wt-wt` | DuckDuckGo region (`wt-wt` = worldwide) |
| `SEARCH_SAFESEARCH` | `moderate` | `on`, `moderate`, or `off` |
| `SEARCH_MAX_RESULTS` | `6` | Max results fetched per search |
| `SEARCH_BACKEND` | `html` | DuckDuckGo backend: `html` (stable) or `lite` |
| `SYSTEM_LANGUAGE` | `en` | Response language: `en`, `uk`, `de`, `fr`, `pl`, `ru` |
| `LLM_TIMEOUT` | `60` | Seconds before an LLM call times out |
| `LLM_RETRIES` | `1` | Extra attempts on LLM failure (total = retries + 1) |
| `ENABLE_FACTSHEET` | `true` | Enable two-stage answer (factsheet then answer) |
| `ENABLE_QUERY_REWRITE` | `true` | Retry with a rewritten query if <2 results |
| `ENABLE_PAGE_FETCH` | `true` | Enrich top-K sources with full-page extraction |
| `PAGE_FETCH_TOP_K` | `3` | Number of top sources to enrich |
| `PAGE_FETCH_TIMEOUT` | `5` | Seconds per page fetch |
| `MAX_HISTORY_TURNS` | `6` | Max prior exchanges included in answer prompt |
| `CACHE_TTL` | `300` | TTL (seconds) for the answer cache |
| `CACHE_SIZE` | `128` | Max entries in the answer cache |

## Testing

```bash
pip install pytest
pytest tests/ -v
```

The suite (33 tests, ~1.5 s) covers:

- Prompt builders and helper functions
- Router JSON parsing (plain, fenced, noisy, invalid)
- Fast-path detection and cache key derivation
- Search normalization, deduplication, and retry behaviour
- Page-fetch fallback when `trafilatura` is unavailable
- End-to-end `/chat` flow for all pipeline branches (fast-path, router-no-search, router-triggers-search, force_search, cache hit) with a stubbed LLM

No real LLM or network is hit during tests.

## Troubleshooting

**`503 LLM client not initialised` or `No models found`**
LM Studio is not running or no model is loaded.

**DuckDuckGo returns few or no results**
Leave `ENABLE_QUERY_REWRITE=true` — the pipeline will rewrite and retry automatically. If that still fails, try `SEARCH_BACKEND=lite` or raise `SEARCH_MAX_RESULTS`.

**Router always (or never) searches**
Adjust `ROUTER_SYSTEM` in `app/prompts.py` or send `force_search=true` in the request.

**Answer quality is poor on a small model**
Lower `temperature` in `app/main.py` (currently `0.2` for grounded answers). Ensure `ENABLE_FACTSHEET=true` — the two-stage split is specifically designed to keep small models faithful to sources.

**Answers are too slow**
Inspect the per-stage `timings` in the response. The usual suspects are:
- `fetch_ms` high → reduce `PAGE_FETCH_TOP_K` or disable `ENABLE_PAGE_FETCH`.
- `factsheet_ms` high → disable `ENABLE_FACTSHEET`.
- `answer_ms` high → lower `max_tokens` in `_llm_call` or use a smaller model.

## Extending the project

1. **Streaming** — stream the answer via SSE or WebSocket.
2. **Reranking** — cross-encoder (e.g. `bge-reranker-base`) between search and enrich.
3. **News mode** — separate `DDGS().news(...)` path for time-sensitive queries.
4. **Dockerfile** — containerize the app (LM Studio stays on the host).
5. **Persistent chat history** — SQLite + session IDs.

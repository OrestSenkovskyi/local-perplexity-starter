# Local Perplexity Starter

A self-hosted [Perplexity AI](https://www.perplexity.ai)-style search assistant that runs entirely on your machine.
It wires together **LM Studio** (any local instruction-following model) and **DuckDuckGo** (free web search) into a
multi-stage pipeline that routes questions, fetches and enriches sources, and returns grounded answers with numbered
citations — no cloud APIs, no paid subscriptions.

---

## Features

- **Fully local** — LM Studio runs the model; DuckDuckGo provides free search results.
- **Multi-stage pipeline** — fast-path → router → search → query-rewrite → page-enrich → factsheet → answer.
- **Prompts tuned for small models** (3–8 B parameters): mechanical, example-rich, constraint-heavy.
- **Grounded citations** — every factual claim is cited `[1]`, `[2]`, etc., and sources are rendered in the UI.
- **Two-stage answer generation** — a factsheet extraction pass followed by a grounded answer pass; keeps small
  models faithful to sources.
- **Full-page extraction** — fetches article text from the top-K URLs via `trafilatura` for richer context.
- **Query-rewrite retry** — if the first search returns < 2 results, the model rewrites the query and retries.
- **Conversation history** — prior turns are included in the answer prompt (capped at `MAX_HISTORY_TURNS`).
- **In-memory TTL cache** — repeat questions are served instantly.
- **Per-stage timings** — every response includes `{router_ms, search_ms, rewrite_ms, fetch_ms, factsheet_ms, answer_ms, total_ms}`.
- **`/health` endpoint** — reports LM Studio reachability and uptime.
- **52 automated tests** — unit, integration, and async e2e tests; no real LLM or network needed.

---

## How it works

Every `/chat` request runs through a configurable pipeline:

```text
user question
    │
    ▼
┌─────────────┐  greeting / len < 4?
│  Fast-path  │──────────────────────► offline answer  (1 LLM call)
└─────────────┘
    │ real question
    ▼
┌─────────────┐  T=0.0, max_tokens=80
│ Router LLM  │  → {"need_search": bool, "query": "..."}
└─────────────┘
    │ need_search=false ────────────► offline answer   (1 LLM call)
    │ need_search=true
    ▼
┌─────────────┐  DuckDuckGo, tenacity retry (×2, 0.5 s backoff)
│   Search    │  k = SEARCH_MAX_RESULTS
└─────────────┘
    │
    ▼
┌─────────────┐  only when len(results) < 2 and ENABLE_QUERY_REWRITE
│ Rewrite LLM │  produces a better keyword query → re-runs search
└─────────────┘
    │
    ▼
┌─────────────┐  only when ENABLE_PAGE_FETCH
│   Enrich    │  fetches top PAGE_FETCH_TOP_K URLs in parallel (trafilatura)
└─────────────┘
    │
    ▼
┌─────────────┐  only when ENABLE_FACTSHEET
│ Factsheet   │  T=0.1 → "- [n] <atomic fact>" bullet list
│    LLM      │
└─────────────┘
    │
    ▼
┌─────────────┐  T=0.2, top_p=0.9, max_tokens=800
│ Answer LLM  │  uses ANSWER_SYSTEM_GROUNDED + factsheet (or raw sources)
└─────────────┘
    │
    ▼
{answer, sources, searched, query, cached, timings}
```

### Pipeline stage reference

| Stage | Runs when | Prompt | Sampling |
|---|---|---|---|
| Fast-path | Greeting / message < 4 chars | — | — |
| Router | Non-trivial, `force_search=false` | `ROUTER_SYSTEM` | `T=0.0`, `max_tokens=80` |
| Search | `need_search=true` | — | — |
| Query rewrite | < 2 results + `ENABLE_QUERY_REWRITE` | `REWRITE_SYSTEM` | `T=0.2`, `max_tokens=30` |
| Enrich | `ENABLE_PAGE_FETCH` + sources present | — | — |
| Factsheet | `ENABLE_FACTSHEET` + sources present | `FACTSHEET_SYSTEM` | `T=0.1`, `top_p=0.9`, `max_tokens=600` |
| Answer (grounded) | Sources present | `ANSWER_SYSTEM_GROUNDED` | `T=0.2`, `top_p=0.9`, `max_tokens=800` |
| Answer (offline) | No sources | `ANSWER_SYSTEM_OFFLINE` | `T=0.4`, `top_p=0.9`, `max_tokens=600` |

All prompts are in `app/prompts.py` and can be edited without changing any application logic.

---

## Project structure

```text
local-perplexity-starter/
├── .env.example                   # All configuration knobs with defaults
├── pytest.ini                     # asyncio_mode=auto, e2e marker
├── requirements.txt
├── app/
│   ├── config.py                  # Settings loaded from .env
│   ├── main.py                    # FastAPI app + pipeline orchestration
│   ├── prompts.py                 # System prompts + message builders
│   ├── search.py                  # DuckDuckGo wrapper with tenacity retry
│   └── fetch.py                   # Parallel page extraction (httpx + trafilatura)
├── templates/
│   └── index.html                 # Single-page dark-themed web UI
└── tests/
    ├── test_prompts.py            # Unit: prompt builders, source formatting, history trimming
    ├── test_main_helpers.py       # Unit: router JSON parsing, fast-path, cache key
    ├── test_search.py             # Unit: dedup, error handling, tenacity retry
    ├── test_fetch.py              # Unit: trafilatura fallback, empty-body handling
    ├── test_chat_integration.py   # Integration: full /chat pipeline via TestClient
    └── e2e/
        ├── conftest.py            # Async ASGI fixture with StubLLM + shared holder
        └── test_e2e.py            # E2E: 19 async tests over real HTTP stack
```

---

## Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (recommended) — or plain `pip`
- [LM Studio](https://lmstudio.ai/) with a model downloaded and the **Local Server** running
  (tested with **Gemma 4 e2b**; any instruction-following model works)

---

## Setup

### 1. Start LM Studio

1. Open LM Studio and download a model (e.g. Gemma 4 e2b, Llama-3.2-3B, Phi-3-mini).
2. Go to the **Developer** → **Local Server** tab and click **Start Server**.
3. Confirm the server responds at `http://localhost:1234/v1`. If your port differs, set `LM_STUDIO_BASE_URL` in `.env`.

### 2. Create a virtual environment and install dependencies

**Using uv (recommended):**
```bash
uv venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell
uv pip install -r requirements.txt
```

**Using plain pip:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

The defaults work out of the box for a standard LM Studio setup. Edit `.env` if your port, model, or language differs.
See the [Configuration reference](#configuration-reference) table for every available knob.

### 4. Run the server

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

---

## Usage

### Web UI

Type a question and press **Ask** (or Enter). Tick **Always search the web** to bypass the router and force a live
DuckDuckGo search regardless of the question type.

### REST API

#### `POST /chat`

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | The user's question |
| `force_search` | boolean | no | Skip the router, always search (default: `false`) |
| `history` | array | no | Prior `{role, content}` turns for multi-turn context |

**Response body:**

| Field | Type | Description |
|---|---|---|
| `answer` | string | LLM-generated answer; may contain `[n]` citation markers |
| `sources` | array | `[{title, url, snippet}]` objects used as context |
| `searched` | boolean | Whether a web search was performed |
| `query` | string \| null | The query sent to DuckDuckGo, or `null` |
| `cached` | boolean | `true` when served from the in-memory cache |
| `timings` | object | Per-stage milliseconds (see example below) |

**Example — question that triggers a web search:**

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest stable Python version?"}'
```

```json
{
  "answer": "The latest stable Python release is **Python 3.14** [1].",
  "sources": [
    {"title": "Python downloads", "url": "https://python.org/downloads", "snippet": "..."}
  ],
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

**Example — multi-turn conversation:**

```bash
# Second turn includes history from the first
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Is it faster than Python 3.12?",
    "history": [
      {"role": "user",      "content": "What is the latest Python version?"},
      {"role": "assistant", "content": "Python 3.14 [1]."}
    ]
  }'
```

#### `GET /health`

```json
{
  "status": "ok",
  "model": "gemma-4-e2b",
  "lm_studio_reachable": true,
  "uptime_seconds": 142
}
```

Returns `"status": "degraded"` when LM Studio is unreachable. Suitable for container liveness probes.

---

## Testing

The project has **52 tests** in three layers, none of which make real network calls or require LM Studio to be running.

### Install test dependencies

```bash
uv pip install pytest pytest-asyncio
```

### Run all tests

```bash
pytest tests/ -v
```

Expected output: **52 passed in ~2 s**.

---

### Layer 1 — Unit tests (`tests/test_*.py`)

Pure function tests with no I/O.

```bash
pytest tests/ --ignore=tests/e2e -v
```

| File | What is tested |
|---|---|
| `test_prompts.py` | Prompt builders, source formatting (content vs snippet, trimming), history trimming, citation stripping |
| `test_main_helpers.py` | Router JSON parsing (plain, markdown-fenced, noisy, invalid), `_is_trivial`, `_cache_key` |
| `test_search.py` | Result normalisation, URL deduplication, exception swallowing, tenacity retry path |
| `test_fetch.py` | Parallel page extraction, `trafilatura`-unavailable fallback, empty-body skip |

---

### Layer 2 — Integration tests (`tests/test_chat_integration.py`)

Full `/chat` and `/health` routes via FastAPI's `TestClient`. LLM and search are replaced with
a script-driven `StubLLM` and a mocked `run_search`.

```bash
pytest tests/test_chat_integration.py -v
```

Covers: fast-path, router→no-search, router→search, `force_search`, cache hit, empty input (400).

---

### Layer 3 — End-to-end tests (`tests/e2e/`)

Async tests that send real HTTP requests to the running ASGI app through
`httpx.AsyncClient(transport=ASGITransport(app))`. The only stubs are at the true external
boundaries: `openai.OpenAI` and `run_search`. Every middleware, route, Pydantic model,
prompt builder, cache, and timing path runs as-is.

```bash
# Run only e2e tests
pytest tests/e2e/ -v

# Run e2e tests by marker
pytest -m e2e -v
```

#### How the fixture works

`tests/e2e/conftest.py` provides a single `e2e` async fixture that:

1. **Patches `openai.OpenAI`** before the FastAPI lifespan starts, so `app.state.client` is a `StubLLM`.
2. `StubLLM` holds a **reference** to a shared `holder` dict — not a copy — so calling `e2e.configure(llm=[...])` in a test updates the responses seen by the already-running client.
3. **Patches `run_search`** with a lambda that reads from the same holder.
4. **Disables all optional stages** (`page_fetch`, `factsheet`, `query_rewrite`) by default — individual tests opt-in via `monkeypatch.setattr(settings, ...)`.
5. Boots the app via `httpx.ASGITransport` and the FastAPI lifespan context manager.

Each test calls `e2e.configure(llm=[...], search=[...])` before making requests to set the scripted responses for that scenario.

#### E2E test coverage

| Test | Scenario |
|---|---|
| `test_health_ok` | `/health` returns model name and `lm_studio_reachable=true` |
| `test_fastpath_greeting_skips_router` | "hi" → 1 LLM call, `router_ms==0` |
| `test_fastpath_thanks` | "thanks" → fast-path triggered |
| `test_no_search_path` | Router returns `need_search=false` → 2 LLM calls, no sources |
| `test_search_path_sources_returned` | Router returns `need_search=true` → sources in response |
| `test_force_search_skips_router` | `force_search=true` → `router_ms==0`, message used as query |
| `test_factsheet_stage_runs` | `ENABLE_FACTSHEET=true` → 3 LLM calls, `factsheet_ms` recorded |
| `test_query_rewrite_fires_on_scarce_results` | < 2 results → rewrite LLM fires, second search attempt |
| `test_history_citations_stripped` | `[n]` markers removed from prior assistant turns |
| `test_history_user_content_preserved` | Prior user turns forwarded verbatim |
| `test_cache_hit_returns_cached_flag` | Repeat request → `cached=true`, no new LLM calls |
| `test_cache_differentiates_force_search` | `force_search=true/false` have separate cache keys |
| `test_router_uses_temperature_zero` | Router call: `T=0.0`; answer call: `T>0.0` |
| `test_router_max_tokens_is_80` | Router: `max_tokens=80`; offline answer: `max_tokens=600` |
| `test_empty_message_returns_400` | Blank input → HTTP 400 |
| `test_missing_message_field_returns_422` | Missing `message` field → HTTP 422 |
| `test_llm_answer_failure_returns_502` | Answer stage exception → HTTP 502 |
| `test_timings_present_and_non_negative` | All 7 timing fields present and ≥ 0 |
| `test_timings_total_includes_all_stages` | `total_ms ≥ sum(stage_ms)` |

#### Spy-patching `_llm_call` for contract tests

Some tests verify _how_ the LLM is called, not just _what_ it returns. They patch `_llm_call`
directly while still routing through the StubLLM for the actual response:

```python
orig = main_mod._llm_call

def spy(messages, *, temperature=0.3, **kw):
    calls.append({"temperature": temperature})
    return orig(messages, temperature=temperature, **kw)

with patch.object(main_mod, "_llm_call", side_effect=spy):
    r = await e2e.http.post("/chat", json={"message": "Explain TCP"})

assert calls[0]["temperature"] == 0.0   # router must be deterministic
assert calls[1]["temperature"] > 0.0    # answer may be creative
```

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio local server base URL |
| `LM_STUDIO_API_KEY` | `lm-studio` | Sent with every request; LM Studio ignores it |
| `LM_STUDIO_MODEL` | _(empty)_ | Model ID; auto-detects the first loaded model when blank |
| `APP_HOST` | `127.0.0.1` | FastAPI bind host |
| `APP_PORT` | `8000` | FastAPI bind port |
| `SEARCH_REGION` | `wt-wt` | DuckDuckGo region (`wt-wt` = worldwide) |
| `SEARCH_SAFESEARCH` | `moderate` | `on`, `moderate`, or `off` |
| `SEARCH_MAX_RESULTS` | `6` | Max results fetched per search |
| `SEARCH_BACKEND` | `html` | DuckDuckGo backend: `html` (stable) or `lite` |
| `SYSTEM_LANGUAGE` | `en` | Response language: `en`, `uk`, `de`, `fr`, `pl`, `ru` |
| `LLM_TIMEOUT` | `60` | Seconds before an LLM call times out |
| `LLM_RETRIES` | `1` | Extra LLM attempts on failure (total = retries + 1) |
| `ENABLE_FACTSHEET` | `true` | Two-stage answer (factsheet extraction then final answer) |
| `ENABLE_QUERY_REWRITE` | `true` | Rewrite and re-search when < 2 results returned |
| `ENABLE_PAGE_FETCH` | `true` | Enrich top-K sources with full article text |
| `PAGE_FETCH_TOP_K` | `3` | Number of URLs to enrich |
| `PAGE_FETCH_TIMEOUT` | `5` | Per-URL fetch timeout in seconds |
| `MAX_HISTORY_TURNS` | `6` | Prior exchanges included in the answer prompt |
| `CACHE_TTL` | `300` | Answer cache TTL in seconds |
| `CACHE_SIZE` | `128` | Max entries in the answer cache |

---

## Troubleshooting

**`503 LLM client not initialised` / `No models found`**
LM Studio is not running or no model is loaded. Start the server and load a model first.

**DuckDuckGo returns few or no results**
The pipeline retries automatically when `ENABLE_QUERY_REWRITE=true`. If still scarce, try
`SEARCH_BACKEND=lite` or increase `SEARCH_MAX_RESULTS`.

**Router always (or never) triggers a search**
Edit `ROUTER_SYSTEM` in `app/prompts.py`, or bypass the router entirely with `force_search=true`.

**Answer quality is poor**
Ensure `ENABLE_FACTSHEET=true` — the two-stage split significantly improves faithfulness on small
models. Also try reducing `temperature` in `app/main.py` (current: `0.2` for grounded answers).

**Answers are slow**
Inspect the `timings` object in the response to find the bottleneck:
- High `fetch_ms` → lower `PAGE_FETCH_TOP_K` or set `ENABLE_PAGE_FETCH=false`.
- High `factsheet_ms` → set `ENABLE_FACTSHEET=false`.
- High `answer_ms` → use a smaller model or lower `max_tokens` in `app/main.py`.

**Tests fail after editing pipeline code**
Run `pytest tests/ -v` to confirm all 52 tests pass before committing. The e2e suite
(`pytest -m e2e`) gives the most complete coverage of the full HTTP stack.

---

## Extending the project

1. **Streaming** — stream answer tokens via SSE using `stream=True` from the OpenAI client.
2. **Reranking** — insert a cross-encoder (e.g. `bge-reranker-base`) between Search and Enrich.
3. **News mode** — add a `DDGS().news(...)` path for time-sensitive queries.
4. **Dockerfile** — containerise the app; LM Studio stays on the host via `host.docker.internal`.
5. **Persistent history** — SQLite + session IDs for stateful multi-turn conversations.

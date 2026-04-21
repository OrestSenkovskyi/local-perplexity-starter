# Local Perplexity Starter

A self-hosted Perplexity-style search assistant built on top of **LM Studio** (local LLM) and **DuckDuckGo** (free web search). No cloud APIs or paid subscriptions required.

## Features

- Runs entirely locally — no OpenAI, no paid APIs.
- Connects to any LM Studio model via the OpenAI-compatible API.
- Automatically decides whether a question needs a web search (router stage).
- Searches the web for free using the `duckduckgo-search` library.
- Returns answers with numbered citations `[1]`, `[2]`, etc.
- Simple dark-themed web UI and a plain JSON REST API.
- Good starting point for extending with RAG, streaming, chat history, or reranking.

## How it works

Every request goes through a three-stage pipeline:

1. **Router** — a cheap LLM call decides whether a web search is needed and, if so, what query to use. Returns `{"need_search": true/false, "query": "..."}`. Skipped entirely when `force_search` is set.
2. **Search** — calls DuckDuckGo and returns up to `SEARCH_MAX_RESULTS` deduplicated results as `{title, url, snippet}` objects.
3. **Answer** — the LLM generates a final response using the search snippets as context, citing sources by number.

## Project structure

```text
local-perplexity-starter/
├── .env.example        # Configuration template
├── requirements.txt
├── app/
│   ├── config.py       # Settings loaded from .env
│   ├── main.py         # FastAPI app and pipeline logic
│   ├── prompts.py      # Router and answer prompt builders
│   └── search.py       # DuckDuckGo search wrapper
└── templates/
    └── index.html      # Single-page web UI
```

## Prerequisites

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) with a model downloaded and Local Server running
  (tested with **Gemma 4 e2b**; any instruction-following model works)

## Setup

### 1. Start LM Studio

1. Open LM Studio and download a model (e.g. Gemma 4 e2b).
2. Go to the **Developer** (or **Local Server**) tab.
3. Click **Start Server**.
4. Confirm the server is available at `http://localhost:1234/v1`. If your port differs, update `LM_STUDIO_BASE_URL` in `.env`.

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

Edit `.env` as needed:

```env
# LM Studio
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio      # dummy value, LM Studio does not validate it
LM_STUDIO_MODEL=                  # leave blank to auto-detect the first loaded model

# Server
APP_HOST=127.0.0.1
APP_PORT=8000

# Search
SEARCH_REGION=wt-wt              # neutral global region; use e.g. us-en for US results
SEARCH_SAFESEARCH=moderate
SEARCH_MAX_RESULTS=6
SEARCH_BACKEND=html              # most stable backend; try "lite" if results are scarce

# Response language (en, uk, de, fr, pl, ru)
SYSTEM_LANGUAGE=en
```

### 4. Run the server

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

## Usage

### Web UI

Type a question in the input field and press **Ask** or hit Enter. Check **Always search the web** to bypass the router and always fetch fresh results.

### REST API

**Endpoint:** `POST /chat`

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | The user's question |
| `force_search` | boolean | no | Skip the router and always run a web search (default: `false`) |

**Response body:**

| Field | Type | Description |
|---|---|---|
| `answer` | string | LLM-generated answer, may contain `[n]` citation markers |
| `sources` | array | List of `{title, url, snippet}` objects used as context |
| `searched` | boolean | Whether a web search was performed |
| `query` | string \| null | The search query sent to DuckDuckGo, or `null` if no search |

#### Example 1 — general knowledge question (no web search)

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 2 + 2?"}'
```

```json
{
  "answer": "2 + 2 equals 4.",
  "sources": [],
  "searched": false,
  "query": null
}
```

#### Example 2 — question that triggers automatic web search

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest stable Python version?"}'
```

```json
{
  "answer": "The latest stable Python release is **Python 3.14**, released on October 7, 2025 [4]. The main development branch is already targeting Python 3.15 [1]. For the full list of supported versions and their end-of-life dates, see the official download page [6].",
  "sources": [
    {"title": "Status of Python versions", "url": "https://devguide.python.org/versions/", "snippet": "..."},
    ...
  ],
  "searched": true,
  "query": "latest stable Python version"
}
```

#### Example 3 — force a web search regardless of the router decision

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Current Bitcoin price", "force_search": true}'
```

`force_search` is useful for news, prices, library versions, and any other rapidly changing information.

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | Base URL of the LM Studio local server |
| `LM_STUDIO_API_KEY` | `lm-studio` | API key sent in requests (LM Studio ignores it) |
| `LM_STUDIO_MODEL` | _(empty)_ | Model ID to use; auto-detects first available model when blank |
| `APP_HOST` | `127.0.0.1` | Host the FastAPI server binds to |
| `APP_PORT` | `8000` | Port the FastAPI server listens on |
| `SEARCH_REGION` | `wt-wt` | DuckDuckGo region code (`wt-wt` = worldwide) |
| `SEARCH_SAFESEARCH` | `moderate` | SafeSearch setting: `on`, `moderate`, or `off` |
| `SEARCH_MAX_RESULTS` | `6` | Maximum number of search results to fetch |
| `SEARCH_BACKEND` | `html` | DuckDuckGo backend: `html` (most stable) or `lite` |
| `SYSTEM_LANGUAGE` | `uk` | Language for LLM responses: `en`, `uk`, `de`, `fr`, `pl`, `ru` |

## Troubleshooting

**`503 LLM client not initialised` or `No models found`**
LM Studio is not running or no model is loaded. Start the server and ensure at least one model is loaded.

**DuckDuckGo returns few or no results**
Try setting `SEARCH_BACKEND=lite` or increasing `SEARCH_MAX_RESULTS`. If the router generates overly long queries, tighten the `ROUTER_SYSTEM` prompt in `app/prompts.py` to produce shorter queries.

**Router always searches (or never searches)**
Adjust the `ROUTER_SYSTEM` prompt in `app/prompts.py`, or use `force_search=true` in the request to bypass the router entirely.

**Answer quality is poor**
Lower the model temperature (currently `0.3` for the answer stage). For Gemma-class models, values between `0.1` and `0.2` tend to reduce hallucinations. You can also make the answer prompt stricter: add "Do not use any knowledge outside the provided search results."

## Extending the project

Suggested next steps:

1. **Streaming** — stream the answer token-by-token via SSE or WebSocket.
2. **Chat history** — pass recent turns to the LLM for multi-turn conversations.
3. **Full-page fetch** — retrieve full page content instead of snippets for higher-quality answers.
4. **Two-stage answer generation** — first summarise search results into a factsheet, then write the final answer from the factsheet (improves consistency on smaller models).
5. **News mode** — add a separate `DDGS().news(...)` search path for time-sensitive queries.
6. **Reranking** — score and reorder search results before passing them to the LLM.
7. **Dockerfile** — containerise the app for easier deployment.
8. **Tests** — add pytest coverage for the router parsing logic and search module.

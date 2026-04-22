"""FastAPI pipeline: fast-path -> router -> search -> (rewrite) -> enrich -> factsheet -> answer."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from contextlib import asynccontextmanager

import openai
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import settings
from app.fetch import enrich as enrich_sources
from app.prompts import (
    answer_prompt,
    factsheet_prompt,
    rewrite_prompt,
    router_prompt,
)
from app.search import run_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = openai.OpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
        timeout=settings.llm_timeout,
    )

    if settings.lm_studio_model:
        app.state.model = settings.lm_studio_model
        logger.info("Using model from config: %s", app.state.model)
    else:
        try:
            models = app.state.client.models.list()
            app.state.model = models.data[0].id
            logger.info("Auto-detected model: %s", app.state.model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not auto-detect model: %s", exc)
            app.state.model = "local-model"

    app.state.answer_cache = TTLCache(maxsize=settings.cache_size, ttl=settings.cache_ttl)
    app.state.started_at = time.time()

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Local Perplexity", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


def get_client() -> openai.OpenAI:
    client = getattr(app.state, "client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialised")
    return client


def get_model() -> str:
    model = getattr(app.state, "model", None)
    if not model:
        raise HTTPException(status_code=503, detail="No LLM model available")
    return model


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class HistoryMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    force_search: bool = False
    history: list[HistoryMessage] = []


class Source(BaseModel):
    title: str
    url: str
    snippet: str


class Timings(BaseModel):
    router_ms: int = 0
    search_ms: int = 0
    rewrite_ms: int = 0
    fetch_ms: int = 0
    factsheet_ms: int = 0
    answer_ms: int = 0
    total_ms: int = 0


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    searched: bool
    query: str | None = None
    cached: bool = False
    timings: Timings | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|hola|yo|thanks|thank you|thx|ty|bye|goodbye|ok|okay|yes|no)[\s!.?,]*$",
    re.I,
)


def _parse_router_response(text: str) -> dict:
    """Extract JSON from router LLM output, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    match = _JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"need_search": False, "query": ""}


def _llm_call(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    top_p: float = 1.0,
    max_tokens: int | None = None,
) -> str:
    """Synchronous LLM call with retry. Wrap in asyncio.to_thread from async code."""
    client = get_client()
    model = get_model()
    attempts = max(1, settings.llm_retries + 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "timeout": settings.llm_timeout,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, attempts, exc)
            if attempt < attempts - 1:
                time.sleep(0.5)
    assert last_exc is not None
    raise last_exc


def _cache_key(req: ChatRequest) -> str:
    # History is part of the key so multi-turn answers don't collide.
    payload = json.dumps(
        {
            "m": req.message.strip(),
            "f": bool(req.force_search),
            "h": [{"r": h.role, "c": h.content} for h in (req.history or [])],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _is_trivial(question: str) -> bool:
    q = question.strip()
    if len(q) < 4:
        return True
    return bool(_GREETING_RE.match(q))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    client = getattr(app.state, "client", None)
    model = getattr(app.state, "model", None)
    reachable = False
    if client is not None:
        try:
            await asyncio.to_thread(client.models.list)
            reachable = True
        except Exception:  # noqa: BLE001
            reachable = False
    uptime = int(time.time() - getattr(app.state, "started_at", time.time()))
    return {
        "status": "ok" if reachable else "degraded",
        "model": model,
        "lm_studio_reachable": reachable,
        "uptime_seconds": uptime,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # --- Cache lookup ---
    cache: TTLCache = app.state.answer_cache
    key = _cache_key(req)
    cached = cache.get(key)
    if cached is not None:
        return cached.model_copy(update={"cached": True})

    t_start = time.perf_counter()
    timings = Timings()

    searched = False
    search_query: str | None = None
    sources: list[dict] = []

    # --- Router stage ---
    if req.force_search:
        need_search = True
        search_query = question
    elif _is_trivial(question):
        # Fast-path: greetings, thanks, etc. never need search.
        need_search = False
    else:
        t0 = time.perf_counter()
        try:
            raw_decision = await asyncio.to_thread(
                _llm_call,
                router_prompt(question),
                temperature=0.0,
                top_p=1.0,
                max_tokens=80,
            )
            decision = _parse_router_response(raw_decision)
            need_search = bool(decision.get("need_search", False))
            search_query = decision.get("query") or question
        except Exception as exc:  # noqa: BLE001
            logger.warning("Router stage failed: %s", exc)
            need_search = False
        timings.router_ms = int((time.perf_counter() - t0) * 1000)

    # --- Search stage ---
    if need_search and search_query:
        t0 = time.perf_counter()
        sources = await asyncio.to_thread(run_search, search_query)
        timings.search_ms = int((time.perf_counter() - t0) * 1000)
        searched = True

        # --- Query rewrite retry ---
        if settings.enable_query_rewrite and len(sources) < 2:
            t0 = time.perf_counter()
            try:
                new_query_raw = await asyncio.to_thread(
                    _llm_call,
                    rewrite_prompt(search_query, question),
                    temperature=0.2,
                    top_p=1.0,
                    max_tokens=30,
                )
                new_query = new_query_raw.strip().splitlines()[0].strip(' "\'')
                if new_query and new_query.lower() != search_query.lower():
                    new_sources = await asyncio.to_thread(run_search, new_query)
                    if len(new_sources) > len(sources):
                        sources = new_sources
                        search_query = new_query
            except Exception as exc:  # noqa: BLE001
                logger.warning("Query rewrite failed: %s", exc)
            timings.rewrite_ms = int((time.perf_counter() - t0) * 1000)

        # --- Enrichment (full-page fetch) ---
        if settings.enable_page_fetch and sources:
            t0 = time.perf_counter()
            try:
                sources = await enrich_sources(sources)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Page enrichment failed: %s", exc)
            timings.fetch_ms = int((time.perf_counter() - t0) * 1000)

    # --- Factsheet stage (two-stage answer) ---
    factsheet: str | None = None
    if settings.enable_factsheet and searched and sources:
        t0 = time.perf_counter()
        try:
            factsheet = await asyncio.to_thread(
                _llm_call,
                factsheet_prompt(question, sources),
                temperature=0.1,
                top_p=0.9,
                max_tokens=600,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Factsheet stage failed: %s", exc)
            factsheet = None
        timings.factsheet_ms = int((time.perf_counter() - t0) * 1000)

    # --- Answer stage ---
    t0 = time.perf_counter()
    try:
        if searched and sources:
            answer = await asyncio.to_thread(
                _llm_call,
                answer_prompt(question, sources, searched, search_query, req.history, factsheet),
                temperature=0.2,
                top_p=0.9,
                max_tokens=800,
            )
        else:
            answer = await asyncio.to_thread(
                _llm_call,
                answer_prompt(question, sources, searched, search_query, req.history),
                temperature=0.4,
                top_p=0.9,
                max_tokens=600,
            )
    except Exception as exc:
        logger.error("Answer stage failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc
    timings.answer_ms = int((time.perf_counter() - t0) * 1000)
    timings.total_ms = int((time.perf_counter() - t_start) * 1000)

    logger.info(
        "chat done searched=%s sources=%d timings=%s",
        searched, len(sources), timings.model_dump(),
    )

    response = ChatResponse(
        answer=answer,
        sources=[Source(**{k: s.get(k, "") for k in ("title", "url", "snippet")}) for s in sources],
        searched=searched,
        query=search_query if searched else None,
        cached=False,
        timings=timings,
    )
    cache[key] = response
    return response

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import settings
from app.prompts import answer_prompt, router_prompt
from app.search import run_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global LLM client state
# ---------------------------------------------------------------------------
_client: openai.OpenAI | None = None
_model: str | None = None


def get_client() -> openai.OpenAI:
    if _client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialised")
    return _client


def get_model() -> str:
    if _model is None:
        raise HTTPException(status_code=503, detail="No LLM model available")
    return _model


# ---------------------------------------------------------------------------
# Lifespan: resolve model name on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _model

    _client = openai.OpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
    )

    if settings.lm_studio_model:
        _model = settings.lm_studio_model
        logger.info("Using model from config: %s", _model)
    else:
        try:
            models = _client.models.list()
            _model = models.data[0].id
            logger.info("Auto-detected model: %s", _model)
        except Exception as exc:
            logger.warning("Could not auto-detect model: %s. LM Studio may not be running.", exc)
            _model = "local-model"  # fallback placeholder

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Local Perplexity", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Pydantic models
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


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    searched: bool
    query: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_router_response(text: str) -> dict:
    """Extract JSON from router LLM output, handling markdown fences."""
    # Strip markdown code fences if present
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


def _llm_call(messages: list[dict], temperature: float = 0.3) -> str:
    client = get_client()
    model = get_model()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    searched = False
    search_query: str | None = None
    sources: list[dict] = []

    # --- Router stage ---
    if req.force_search:
        need_search = True
        search_query = question
    else:
        router_messages = router_prompt(question)
        try:
            raw_decision = await asyncio.to_thread(
                _llm_call, router_messages, 0.0
            )
            decision = _parse_router_response(raw_decision)
            need_search = bool(decision.get("need_search", False))
            search_query = decision.get("query") or question
        except Exception as exc:
            logger.warning("Router stage failed: %s", exc)
            need_search = False

    # --- Search stage ---
    if need_search and search_query:
        sources = await asyncio.to_thread(run_search, search_query)
        searched = True

    # --- Answer stage ---
    answer_messages = answer_prompt(question, sources, searched, search_query, req.history)
    try:
        answer = await asyncio.to_thread(_llm_call, answer_messages, 0.3)
    except Exception as exc:
        logger.error("Answer stage failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    return ChatResponse(
        answer=answer,
        sources=[Source(**s) for s in sources],
        searched=searched,
        query=search_query if searched else None,
    )

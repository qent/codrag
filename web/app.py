from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

import httpx
import markdown2
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Serve static assets (CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

ROOT_PATH = os.environ.get("ROOT_PATH", "")
HYDE_PROMPT_PATH = os.environ.get("HYDE_PROMPT_PATH", "")
QUERY_URL = os.environ.get("QUERY_URL", "http://rag:8080/v1/query")
# Total timeout (seconds) for the upstream RAG call. Defaults to ~5 minutes.
WEB_UPSTREAM_TIMEOUT_SEC = float(os.environ.get("WEB_UPSTREAM_TIMEOUT_SEC", "310"))

# LLM configuration for web post-processing
WEB_LLM_ENDPOINT = os.environ.get("WEB_LLM_ENDPOINT", "")
WEB_LLM_API_KEY = os.environ.get("WEB_LLM_API_KEY", "")
WEB_LLM_MODEL = os.environ.get("WEB_LLM_MODEL", "")
WEB_LLM_IGNORE_HTTPS = os.environ.get("WEB_LLM_IGNORE_HTTPS", "false").lower() in {
    "1",
    "true",
    "yes",
}
WEB_SYSTEM_PROMPT_PATH = os.environ.get("WEB_SYSTEM_PROMPT_PATH", "")

HYDE_PROMPT: str = ""
if HYDE_PROMPT_PATH:
    try:
        HYDE_PROMPT = Path(HYDE_PROMPT_PATH).read_text(encoding="utf-8")
    except OSError:
        HYDE_PROMPT = ""

WEB_SYSTEM_PROMPT: str = ""
if WEB_SYSTEM_PROMPT_PATH:
    try:
        WEB_SYSTEM_PROMPT = Path(WEB_SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    except OSError:
        WEB_SYSTEM_PROMPT = ""

COOKIE_NAME = "uid"


class SessionState:
    """Holds per-user query execution state.

    Attributes:
        task: Background task executing the upstream query.
        query: The text of the current query.
        result: Result payload from upstream when successful.
        error: Error payload when upstream fails or when cancelled.
        started_at: Unix timestamp when the task started.
        answer: Last generated Markdown answer from the LLM.
    """

    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.query: str | None = None
        self.result: dict | None = None
        self.error: dict | None = None
        self.started_at: float | None = None
        self.answer: str | None = None


SESSIONS: dict[str, SessionState] = {}


@dataclass
class LlmConfig:
    """Holds configuration for the LLM used to summarize RAG results."""

    endpoint: str
    api_key: str
    model: str
    ignore_https: bool


def _llm_config() -> LlmConfig:
    """Return the LLM configuration resolved from environment variables."""

    return LlmConfig(
        endpoint=WEB_LLM_ENDPOINT,
        api_key=WEB_LLM_API_KEY,
        model=WEB_LLM_MODEL,
        ignore_https=WEB_LLM_IGNORE_HTTPS,
    )


_LLM: Any | None = None


def _build_llm() -> Any:
    """Build and cache the LangChain chat model instance.

    Returns:
        A LangChain-compatible chat model instance. If LangChain is not
        available, raises HTTPException to surface a clear error to the user.
    """

    global _LLM
    if _LLM is not None:
        return _LLM

    cfg = _llm_config()
    http_client = httpx.Client(verify=not cfg.ignore_https)

    _LLM = ChatOpenAI(
        model=cfg.model,
        base_url=(cfg.endpoint or None),
        api_key=(cfg.api_key or "EMPTY"),
        timeout=60,
        max_retries=1,
        http_client=http_client,
    )

    return _LLM


DEFAULT_WEB_SYSTEM_PROMPT = (
    "You are a precise documentation and code assistant that writes clear, "
    "concise answers in Markdown. You MUST rely only on the provided RAG "
    "context and never use outside knowledge. If the context is insufficient, "
    "say so and propose what additional files or details are needed.\n\n"
    "Guidelines:\n"
    "- Write the final answer in the SAME LANGUAGE as the user's question.\n"
    "- Prefer short sections with headings, bullet lists, and code fences.\n"
    "- Use only facts present in the context; do not speculate.\n"
    "- If the user asks for code, include minimal, correct snippets taken only "
    "from the provided sources.\n"
    "- For each snippet, show a link to the source file near the snippet.\n"
    "- End with a 'Sources' section listing the file paths you used.\n"
)


def _extract_item_text(item: Dict[str, Any]) -> str:
    """Extract textual content from a RAG item, handling nested payloads.

    The upstream may return text under ``item['text']`` or embed a serialized
    node payload under ``item['metadata']['_node_content']`` containing a
    ``text`` field. This function normalizes those cases.

    Args:
        item: A single item from the RAG response.

    Returns:
        The extracted textual content or an empty string if none is present.
    """

    text = str(item.get("text") or "")
    if text:
        return text
    meta = item.get("metadata") or {}
    node_payload = meta.get("_node_content")
    if isinstance(node_payload, str) and node_payload.strip():
        try:
            parsed = json.loads(node_payload)
            # Many nodes place the code under the top-level 'text'
            if isinstance(parsed, dict) and parsed.get("text"):
                return str(parsed["text"])  # type: ignore[index]
        except Exception:
            pass
    return ""


def _format_context(items: List[Dict[str, Any]]) -> str:
    """Format RAG items into a compact, LLM-friendly context block.

    Each source is clearly separated and labeled with its file path and
    language to guide the LLM when selecting snippets.

    Args:
        items: The list of RAG result items.

    Returns:
        A single string with all sources serialized in a deterministic format.
    """

    parts: list[str] = []
    for idx, it in enumerate(items, start=1):
        meta = it.get("metadata") or {}
        path = str(meta.get("file_path") or meta.get("path") or "")
        lang = str(meta.get("lang") or meta.get("language") or "")
        content = _extract_item_text(it).strip()
        if not content:
            continue
        header = f"Source {idx}: {path} ({lang})".strip()
        fenced_lang = lang if lang and lang != "unknown" else ""
        snippet = f"```{fenced_lang}\n{content}\n```" if content else ""
        parts.append(f"{header}\n{snippet}")
    return "\n\n".join(parts)


def _build_chain() -> Any:
    """Build the LangChain prompt + model pipeline for summarization."""

    llm = _build_llm()
    system_prompt = WEB_SYSTEM_PROMPT or DEFAULT_WEB_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Question:\n{question}\n\nContext (use only this):\n{context}\n\n"
                "Write a helpful answer strictly based on the context.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


async def _summarize_to_markdown(question: str, rag_payload: dict[str, Any]) -> str:
    """Use the LLM to produce a Markdown answer from the RAG payload.

    Args:
        question: The original user query.
        rag_payload: The JSON payload returned by the RAG service.

    Returns:
        Markdown-formatted answer.
    """

    items = list(rag_payload.get("items") or [])
    context = _format_context(items)
    if not context.strip():
        # Nothing to summarize — surface a friendly message
        return (
            "I could not find relevant information in the retrieved context.\n\n"
            "Please refine your question or broaden the search scope."
        )
    chain = _build_chain()
    # Use async invoke if available
    if hasattr(chain, "ainvoke"):
        return await chain.ainvoke({"question": question, "context": context})  # type: ignore[no-any-return]
    # Synchronous fallback executed in a thread
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: chain.invoke({"question": question, "context": context}))


async def _summarize_to_markdown_stream(
    question: str, rag_payload: dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Stream the Markdown answer from the LLM.

    Args:
        question: The original user query.
        rag_payload: The JSON payload returned by the RAG service.

    Yields:
        Chunks of the Markdown-formatted answer.
    """

    items = list(rag_payload.get("items") or [])
    context = _format_context(items)
    if not context.strip():
        yield (
            "I could not find relevant information in the retrieved context.\n\n"
            "Please refine your question or broaden the search scope."
        )
        return
    chain = _build_chain()
    if hasattr(chain, "astream"):
        async for chunk in chain.astream({"question": question, "context": context}):
            yield chunk
        return
    yield await _summarize_to_markdown(question, rag_payload)


def _get_or_set_user_id(request: Request, response: Response) -> str:
    """Return user id from cookie; create if missing and set cookie."""

    uid = request.cookies.get(COOKIE_NAME)
    if not uid:
        import secrets

        uid = secrets.token_urlsafe(16)
        # Session cookie (no max-age) scoped to root, httpOnly
        response.set_cookie(COOKIE_NAME, uid, httponly=True, samesite="lax", path="/")
    return uid


def _state_for(uid: str) -> SessionState:
    """Return session state for ``uid`` creating it if needed."""

    if uid not in SESSIONS:
        SESSIONS[uid] = SessionState()
    return SESSIONS[uid]


async def _run_query(uid: str, query: str) -> None:
    """Execute the upstream RAG query and store the result or error.

    This function respects a long timeout (5 minutes) and is designed to be run
    as a background task. It populates module-level state so the UI can poll
    status across page reloads and allows cancellation.
    """

    state = _state_for(uid)
    state.result = None
    state.error = None
    state.answer = None
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                QUERY_URL,
                json={
                    "root_path": ROOT_PATH,
                    "q": query,
                    "return_text_description": True,
                    "hyde_system_prompt": HYDE_PROMPT,
                },
                timeout=httpx.Timeout(WEB_UPSTREAM_TIMEOUT_SEC),
            )
        if response.status_code >= 400:
            # Try to surface structured detail.
            try:
                detail = response.json()
            except Exception:
                detail = {"detail": response.text}
            state.error = {"status_code": response.status_code, "detail": detail}
        else:
            state.result = response.json()
    except asyncio.CancelledError:
        state.error = {"status_code": 499, "detail": "Client cancelled the request"}
        raise
    except httpx.HTTPError as exc:
        state.error = {"status_code": 502, "detail": f"Upstream error: {exc}"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Return the search page."""

    # Ensure the user id cookie is set on initial visit
    response = templates.TemplateResponse("index.html", {"request": request})
    _get_or_set_user_id(request, response)
    return response


@app.post("/search/start")
async def search_start(request: Request, response: Response, payload: dict[str, str]) -> dict:
    """Start a single in-flight query; return 409 if busy."""

    uid = _get_or_set_user_id(request, response)
    state = _state_for(uid)
    if state.task and not state.task.done():
        raise HTTPException(status_code=409, detail="A query is already in progress for this user")

    query = payload.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    # Reset state and launch background task
    state.query = query
    state.started_at = time.time()
    state.answer = None
    state.task = asyncio.create_task(_run_query(uid, query))
    return {"status": "started"}


@app.get("/search/status")
def search_status(request: Request, response: Response) -> dict:
    """Return whether a query is in progress and basic metadata."""

    uid = _get_or_set_user_id(request, response)
    state = _state_for(uid)
    in_progress = bool(state.task and not state.task.done())
    elapsed_ms = int((time.time() - state.started_at) * 1000) if state.started_at else 0
    return {
        "in_progress": in_progress,
        "query": state.query or "",
        "elapsed_ms": elapsed_ms,
    }


@app.get("/search/result/stream")
async def search_result_stream(request: Request, response: Response) -> StreamingResponse:
    """Stream the LLM answer as it is generated."""

    uid = _get_or_set_user_id(request, response)
    state = _state_for(uid)
    if state.task and not state.task.done():
        raise HTTPException(status_code=409, detail="Query is still in progress")
    if state.error is not None:
        status = int(state.error.get("status_code", 500))
        raise HTTPException(status_code=status, detail=state.error.get("detail"))
    if not state.result:
        raise HTTPException(status_code=404, detail="No result available")

    async def iterator() -> AsyncGenerator[str, None]:
        acc = ""
        async for chunk in _summarize_to_markdown_stream(state.query or "", state.result):
            acc += chunk
            yield chunk
        state.answer = acc

    return StreamingResponse(iterator(), media_type="text/plain")


@app.get("/search/result")
async def search_result(request: Request, response: Response) -> dict:
    """Return the result if completed; propagate errors with status codes.

    On success, this endpoint returns both the raw RAG payload and a
    Markdown/HTML rendering generated via the configured LLM. The HTML is a
    simple conversion of Markdown for direct rendering on the page.
    """

    uid = _get_or_set_user_id(request, response)
    state = _state_for(uid)
    if state.task and not state.task.done():
        return {"in_progress": True}
    if state.error is not None:
        status = int(state.error.get("status_code", 500))
        raise HTTPException(status_code=status, detail=state.error.get("detail"))
    if not state.result:
        return {"status": "idle"}

    # Generate a human-readable Markdown answer using only RAG data
    if state.answer is not None:
        markdown = state.answer
    else:
        try:
            markdown = await _summarize_to_markdown(state.query or "", state.result)
            state.answer = markdown
        except HTTPException:
            # Bubble up LangChain import/config issues as-is
            raise
        except Exception as exc:
            # If LLM processing fails, still return the raw payload
            markdown = f"LLM rendering failed: {exc}"

    # Convert Markdown to HTML for in-page rendering
    html: str
    try:
        html = markdown2.markdown(
            markdown,
            extras=[
                "fenced-code-blocks",
                "tables",
                "strike",
                "task_list",
                "toc",
                "code-friendly",
            ],
        )
    except Exception:
        # Graceful fallback to preformatted text
        html = f"<pre>{markdown}</pre>"

    return {
        "in_progress": False,
        "question": state.query,
        "markdown": markdown,
        "html": html,
        "raw": state.result,
    }


@app.post("/search/cancel")
async def search_cancel(request: Request, response: Response) -> dict:
    """Cancel the in-flight query if present."""

    uid = _get_or_set_user_id(request, response)
    state = _state_for(uid)
    if state.task and not state.task.done():
        state.task.cancel()
        try:
            await state.task
        except asyncio.CancelledError:
            pass
        return {"status": "cancelled"}
    return {"status": "idle"}

from __future__ import annotations

import os
import asyncio
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

ROOT_PATH = os.environ.get("ROOT_PATH", "")
HYDE_PROMPT_PATH = os.environ.get("HYDE_PROMPT_PATH", "")
QUERY_URL = os.environ.get("QUERY_URL", "http://rag:8080/v1/query")

HYDE_PROMPT: str = ""
if HYDE_PROMPT_PATH:
    try:
        HYDE_PROMPT = Path(HYDE_PROMPT_PATH).read_text(encoding="utf-8")
    except OSError:
        HYDE_PROMPT = ""

COOKIE_NAME = "uid"


class SessionState:
    """Holds per-user query execution state.

    Attributes:
        task: Background task executing the upstream query.
        query: The text of the current query.
        result: Result payload from upstream when successful.
        error: Error payload when upstream fails or when cancelled.
        started_at: Unix timestamp when the task started.
    """

    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.query: str | None = None
        self.result: dict | None = None
        self.error: dict | None = None
        self.started_at: float | None = None


SESSIONS: dict[str, SessionState] = {}


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
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                QUERY_URL,
                json={
                    "root_path": ROOT_PATH,
                    "q": query,
                    "hyde_system_prompt": HYDE_PROMPT,
                },
                timeout=httpx.Timeout(310.0),
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


@app.get("/search/result")
async def search_result(request: Request, response: Response) -> dict:
    """Return the result if completed; propagate errors with status codes."""

    uid = _get_or_set_user_id(request, response)
    state = _state_for(uid)
    if state.task and not state.task.done():
        return {"in_progress": True}
    if state.error is not None:
        status = int(state.error.get("status_code", 500))
        raise HTTPException(status_code=status, detail=state.error.get("detail"))
    return state.result or {"status": "idle"}


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

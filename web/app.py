from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
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


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Return the search page."""

    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
async def search(payload: dict[str, str]) -> dict:
    """Forward the query to the RAG service."""

    query = payload.get("query", "")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            QUERY_URL,
            json={
                "root_path": ROOT_PATH,
                "q": query,
                "hyde_system_prompt": HYDE_PROMPT,
            },
        )
    return response.json()

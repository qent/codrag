"""Query rewriting utilities for specialized collections."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from httpx import Client
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .config import OpenAIClientConfig


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "query_rewriter.md"
_PROMPT_TMPL = ChatPromptTemplate.from_template(_PROMPT_PATH.read_text())


class _Queries(BaseModel):
    """Structured output for specialized queries."""

    code: str = Field(description="Query for code snippets")
    file: str = Field(description="Query for file descriptions")
    dir: str = Field(description="Query for directory overviews")


def _build_llm(cfg: OpenAIClientConfig | None) -> BaseChatModel | None:
    """Create an LLM from ``cfg`` or return ``None`` if misconfigured."""

    if cfg is None:
        return None
    api_key = cfg.api_key
    if api_key.startswith("env:"):
        api_key = os.environ.get(api_key.split(":", 1)[1], "")
    if not api_key:
        return None
    http_client = Client(verify=cfg.verify_ssl, timeout=cfg.timeout_sec)
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=api_key,
        max_retries=cfg.retries,
        http_client=http_client,
    )


def rewrite_for_collections(
    query: str, cfg: OpenAIClientConfig | None = None
) -> Tuple[str, str, str]:
    """Rewrite ``query`` for code, file and directory collections using ``cfg``.

    Parameters
    ----------
    query:
        Original user query.
    cfg:
        Configuration for the LLM. When ``None`` or misconfigured the original
        ``query`` is returned for all collections.

    Returns
    -------
    tuple[str, str, str]
        Queries for the code, file and directory collections respectively.
    """

    llm = _build_llm(cfg)
    if llm is None:
        return query, query, query
    chain = _PROMPT_TMPL | llm.with_structured_output(_Queries)
    data = chain.invoke({"query": query})
    return data.code, data.file, data.dir


"""Query rewriting utilities for specialized collections."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .config import OpenAIClientConfig
from .openai_utils import build_langchain_llm


# Collection-specific prompt templates
_CODE_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "query_rewriter_code.md"
)
_FILE_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "query_rewriter_file.md"
)
_DIR_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "query_rewriter_dir.md"
)

_CODE_PROMPT_TMPL = ChatPromptTemplate.from_template(_CODE_PROMPT_PATH.read_text())
_FILE_PROMPT_TMPL = ChatPromptTemplate.from_template(_FILE_PROMPT_PATH.read_text())
_DIR_PROMPT_TMPL = ChatPromptTemplate.from_template(_DIR_PROMPT_PATH.read_text())

_EXPAND_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "query_expander.md"
)
_EXPAND_PROMPT_TMPL = ChatPromptTemplate.from_template(
    _EXPAND_PROMPT_PATH.read_text()
)


class _SingleQuery(BaseModel):
    """Structured output containing a single rewritten query."""

    query: str = Field(description="Rewritten query")


class _Expansions(BaseModel):
    """Structured output containing alternative queries."""

    queries: List[str] = Field(default_factory=list, description="Alternative queries")


_HYDE_CODE_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "hyde_code.md"
)
_HYDE_CODE_PROMPT_TEXT = _HYDE_CODE_PROMPT_PATH.read_text()


class _HyDEDrafts(BaseModel):
    """Structured output containing hypothetical documents for HyDE."""

    docs: List[str] = Field(default_factory=list, description="Hypothetical documents")


def _build_llm(cfg: OpenAIClientConfig | None) -> BaseChatModel | None:
    """Thin wrapper delegating to shared LangChain LLM factory."""

    return build_langchain_llm(cfg)


def rewrite_for_collections(
    query: str, cfg: OpenAIClientConfig | None = None
) -> Tuple[str, str, str]:
    """Rewrite ``query`` for code, file and directory collections using ``cfg``.

    The function uses three dedicated prompts (one per collection) to avoid mixing
    guidance. If the LLM configuration is missing or invalid, the original
    ``query`` is returned for all collections.

    Parameters
    ----------
    query:
        Original user query.
    cfg:
        Configuration for the LLM.

    Returns
    -------
    tuple[str, str, str]
        Queries for the code, file and directory collections respectively.
    """

    llm = _build_llm(cfg)
    if llm is None:
        return query, query, query

    code_chain = _CODE_PROMPT_TMPL | llm.with_structured_output(_SingleQuery)
    file_chain = _FILE_PROMPT_TMPL | llm.with_structured_output(_SingleQuery)
    dir_chain = _DIR_PROMPT_TMPL | llm.with_structured_output(_SingleQuery)

    code = code_chain.invoke({"query": query}).query
    file = file_chain.invoke({"query": query}).query
    dir = dir_chain.invoke({"query": query}).query
    return code, file, dir


def expand_queries(
    query: str,
    cfg: OpenAIClientConfig | None = None,
    max_expansions: int = 0,
) -> List[str]:
    """Generate up to ``max_expansions`` alternative phrasings for ``query``.

    Parameters
    ----------
    query:
        Original user query.
    cfg:
        Configuration for the LLM. When ``None`` or misconfigured, an empty list
        is returned.
    max_expansions:
        Maximum number of paraphrases to produce.

    Returns
    -------
    list[str]
        Alternative query phrasings.
    """

    if max_expansions <= 0:
        return []
    llm = _build_llm(cfg)
    if llm is None:
        return []
    chain = _EXPAND_PROMPT_TMPL | llm.with_structured_output(_Expansions)
    data = chain.invoke({"query": query, "n": max_expansions})
    return data.queries[:max_expansions]


def hyde_code_documents(
    query: str,
    cfg: OpenAIClientConfig | None = None,
    n: int = 1,
    system_prompt: str = "",
) -> List[str]:
    """Return up to ``n`` hypothetical code-centric documents for HyDE.

    The function prompts an LLM to create concise, plausible documents that would
    answer ``query`` in a code context. These drafts are intended to be embedded
    and used for doc–doc similarity search against code chunks.

    Parameters
    ----------
    query:
        Original user query or a code-optimized rewrite of it.
    cfg:
        LLM configuration. When ``None`` or invalid, returns an empty list.
    n:
        Maximum number of hypothetical documents to generate.

    Returns
    -------
    list[str]
        List of hypothetical documents (may be empty).
    """

    if n <= 0:
        return []
    llm = _build_llm(cfg)
    if llm is None:
        return []
    messages = [("system", _HYDE_CODE_PROMPT_TEXT)]
    if system_prompt.strip():
        messages.append(("system", system_prompt))
    messages.append(("human", 'User query:\n"""\n{query}\n"""'))
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm.with_structured_output(_HyDEDrafts)
    data = chain.invoke({"query": query})
    return data.docs[:n]

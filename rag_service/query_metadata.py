from __future__ import annotations

"""Utilities for extracting query metadata for file-card retrieval.

This module extracts metadata similar to what we store with file descriptions
and provides lightweight re-scoring helpers to use those signals during search.
"""

from pathlib import Path
from typing import List, Sequence

from langchain_core.prompts import ChatPromptTemplate
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field

from .config import OpenAIClientConfig
from .openai_utils import build_langchain_llm


class FileQueryMetadata(BaseModel):
    """Structured query metadata for searching file descriptions.

    Attributes
    ----------
    languages:
        Programming language hints (e.g., python, java, typescript). Lowercase.
    keywords:
        Searchable tags/terms expected in file-card metadata keywords.
    dir_paths:
        Repository-relative directory hints (e.g., auth/, src/auth, api/v1).
    file_extensions:
        File extensions of interest without leading dots (e.g., py, ts, md).
    filenames:
        Specific filenames or glob-like patterns (e.g., docker-compose.yml, *_config.py).
    """

    languages: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    dir_paths: List[str] = Field(default_factory=list)
    file_extensions: List[str] = Field(default_factory=list)
    filenames: List[str] = Field(default_factory=list)


_FILE_QMETA_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "query_metadata_file.md"
)
_FILE_QMETA_TMPL = ChatPromptTemplate.from_template(
    _FILE_QMETA_PROMPT_PATH.read_text()
)


def extract_file_query_metadata(
    query: str, cfg: OpenAIClientConfig | None
) -> FileQueryMetadata:
    """Extract metadata from a user query for file-card search.

    When the LLM configuration is missing or invalid, returns an instance with
    empty lists for all fields.

    Parameters
    ----------
    query:
        Original user query.
    cfg:
        Configuration for the LLM used to extract metadata.

    Returns
    -------
    FileQueryMetadata
        Parsed metadata suitable for matching against file-card payload.
    """

    llm = build_langchain_llm(cfg)
    if llm is None:
        return FileQueryMetadata()
    chain = _FILE_QMETA_TMPL | llm.with_structured_output(FileQueryMetadata)
    try:
        return chain.invoke({"query": query})
    except Exception:
        # In restricted or offline environments, degrade gracefully.
        return FileQueryMetadata()


def boost_file_nodes_by_metadata(
    nodes: Sequence[NodeWithScore], meta: FileQueryMetadata
) -> List[NodeWithScore]:
    """Return nodes re-scored using query metadata soft matches.

    The function does not filter results; it adds small positive boosts for
    matches and re-sorts. Boosts are applied only when both the query metadata
    and node payload contain the relevant field.

    Scoring heuristics (additive boosts):
    - language match: +0.25
    - directory path hint (substring in dir_path or file_path): +0.20 (first match)
    - keyword overlap: +0.05 per match up to +0.20
    - filename / extension match: +0.20 (first match)

    Parameters
    ----------
    nodes:
        Initial retrieval results.
    meta:
        Extracted query metadata.

    Returns
    -------
    list[NodeWithScore]
        Re-scored results, sorted by descending score.
    """

    if not nodes:
        return []

    # Normalize query metadata for simple comparisons
    langs = {l.strip().lower() for l in meta.languages if l.strip()}
    keywords = {k.strip().lower() for k in meta.keywords if k.strip()}
    dir_hints = [d.strip().lower() for d in meta.dir_paths if d.strip()]
    exts = {e.strip().lower().lstrip(".") for e in meta.file_extensions if e.strip()}
    fnames = [f.strip().lower() for f in meta.filenames if f.strip()]

    boosted: List[NodeWithScore] = []
    for n in nodes:
        payload = getattr(n.node, "metadata", {}) or {}
        score = float(n.score or 0.0)

        # Language boost
        node_lang = str(payload.get("lang", "")).strip().lower()
        if langs and node_lang and node_lang in langs:
            score += 0.25

        # Directory/path boosts
        file_path = str(payload.get("file_path", "")).strip().lower()
        dir_path = str(payload.get("dir_path", "")).strip().lower()
        if dir_hints and (file_path or dir_path):
            target = f"{dir_path} {file_path}".strip()
            if any(h in target for h in dir_hints):
                score += 0.20

        # Keyword overlap
        node_keywords = [str(k).strip().lower() for k in (payload.get("keywords") or [])]
        if keywords and node_keywords:
            matches = len(set(node_keywords) & keywords)
            score += min(0.20, 0.05 * matches)

        # Filename / extension
        if file_path:
            # Extension
            if exts:
                fp_ext = Path(file_path).suffix.lstrip(".").lower()
                if fp_ext and fp_ext in exts:
                    score += 0.20
            # Filename patterns (simple contains)
            if fnames and any(pat in Path(file_path).name.lower() for pat in fnames):
                score += 0.20

        n.score = score
        boosted.append(n)

    return sorted(boosted, key=lambda x: x.score or 0.0, reverse=True)

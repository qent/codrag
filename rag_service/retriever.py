from __future__ import annotations

from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex

from .config import AppConfig
from .llama_facade import LlamaIndexFacade
from .retrievers.simple import build_simple_retriever
from .retrievers.smart import build_smart_retriever
from .retrievers.utils import fuse_results, CrossEncoderReranker
from .query_rewriter import (
    rewrite_for_collections,
    expand_queries,
    hyde_code_documents,
)
from .query_metadata import (
    extract_file_query_metadata,
    FileQueryMetadata,
    boost_file_nodes_by_metadata,
)


def build_query_engine(
    cfg: AppConfig,
    qdrant: QdrantClient,
    llama: LlamaIndexFacade | None = None,
    collection_prefix: str = "",
):
    """Return a retriever instance according to configuration.

    This module also re-exports helper utilities used by tests and callers:
    - `fuse_results`: fusion of retrieval results
    - `VectorStoreIndex`: LlamaIndex vector index class
    - `rewrite_for_collections`, `expand_queries`, `hyde_code_documents`: query rewriting utilities
    - `CrossEncoderReranker`: optional reranker utility
    """

    # Default to "simple" when attribute is missing for compatibility
    name = getattr(getattr(cfg.llamaindex, "retrieval", object()), "retriever", "simple")
    if name == "smart":
        return build_smart_retriever(cfg, qdrant, llama, collection_prefix)
    return build_simple_retriever(cfg, qdrant, llama, collection_prefix)


__all__ = [
    "build_query_engine",
    # Re-exports for tests and convenience
    "fuse_results",
    "VectorStoreIndex",
    "rewrite_for_collections",
    "expand_queries",
    "hyde_code_documents",
    "CrossEncoderReranker",
    "extract_file_query_metadata",
    "FileQueryMetadata",
    "boost_file_nodes_by_metadata",
]

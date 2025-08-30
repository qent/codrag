from __future__ import annotations

from llama_index.core import VectorStoreIndex  # re-exported for tests
from qdrant_client import QdrantClient

from .config import AppConfig
from .llama_facade import LlamaIndexFacade
from .retrievers.simple import build_simple_retriever
from .retrievers.utils import CrossEncoderReranker, fuse_results
from .query_rewriter import expand_queries, rewrite_for_collections, hyde_code_documents
from .query_metadata import extract_file_query_metadata, boost_file_nodes_by_metadata


def build_query_engine(
    cfg: AppConfig,
    qdrant: QdrantClient,
    llama: LlamaIndexFacade | None = None,
    collection_prefix: str = "",
):
    """Return a retriever instance according to configuration."""

    return build_simple_retriever(cfg, qdrant, llama, collection_prefix)


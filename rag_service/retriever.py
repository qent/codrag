from __future__ import annotations

from typing import List

from llama_index.core import VectorStoreIndex
from qdrant_client import QdrantClient

from .config import AppConfig
from .llama_facade import LlamaIndexFacade


def build_query_engine(cfg: AppConfig, qdrant: QdrantClient, llama: LlamaIndexFacade | None = None):
    """Build a simple retriever combining code and file card indexes."""

    llama = llama or LlamaIndexFacade(cfg, qdrant)
    code_vs = llama.code_vs()
    file_vs = llama.file_vs()

    code_ret = VectorStoreIndex.from_vector_store(code_vs).as_retriever(
        similarity_top_k=cfg.llamaindex.retrieval.code_nodes_top_k
    )
    file_ret = VectorStoreIndex.from_vector_store(file_vs).as_retriever(
        similarity_top_k=cfg.llamaindex.retrieval.file_cards_top_k
    )

    class SimpleRetriever:
        def retrieve(self, query: str):
            return code_ret.retrieve(query) + file_ret.retrieve(query)

    return SimpleRetriever()

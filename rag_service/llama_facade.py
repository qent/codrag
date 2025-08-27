from __future__ import annotations

from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .config import AppConfig
from .openai_utils import init_llamaindex_clients


class LlamaIndexFacade:
    """Facade over global LlamaIndex settings and vector stores."""

    def __init__(self, cfg: AppConfig, qdrant: QdrantClient, initialize: bool = True) -> None:
        self.cfg = cfg
        self.qdrant = qdrant
        if initialize:
            init_llamaindex_clients(cfg)

    def llm(self):
        return Settings.llm

    def embed_model(self):
        return Settings.embed_model

    def code_vs(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.qdrant, collection_name=f"{self.cfg.qdrant.collection_prefix}code_nodes"
        )

    def file_vs(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.qdrant, collection_name=f"{self.cfg.qdrant.collection_prefix}file_cards"
        )

    def dir_vs(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.qdrant, collection_name=f"{self.cfg.qdrant.collection_prefix}dir_cards"
        )

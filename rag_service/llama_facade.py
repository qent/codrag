from __future__ import annotations

from llama_index.core import Settings
from llama_index.core.llms import LLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import AppConfig
from .openai_utils import init_llamaindex_clients


DISTANCE_MAP = {
    "Cosine": models.Distance.COSINE,
    "Euclid": models.Distance.EUCLID,
    "Manhattan": models.Distance.MANHATTAN,
}


class LlamaIndexFacade:
    """Facade over global LlamaIndex settings and vector stores."""

    def __init__(self, cfg: AppConfig, qdrant: QdrantClient, initialize: bool = True) -> None:
        self.cfg = cfg
        self.qdrant = qdrant
        if initialize:
            init_llamaindex_clients(cfg)
        self._code_vector_size = len(
            Settings.code_embed_model.get_text_embedding("qdrant_dim_probe")
        )
        self._text_vector_size = len(
            Settings.text_embed_model.get_text_embedding("qdrant_dim_probe")
        )
        self._stores: dict[str, QdrantVectorStore] = {}

    def llm(self) -> LLM:
        """Return the configured language model."""

        return Settings.llm

    def _create_vs(self, name: str, distance_name: str, vector_size: int) -> QdrantVectorStore:
        """Create a Qdrant collection when missing and return its vector store.

        Avoids deprecated ``recreate_collection`` by checking for existence and
        creating the collection if it does not already exist.
        """

        if name not in self._stores:
            if not self.qdrant.collection_exists(name):
                distance = DISTANCE_MAP[distance_name]
                self.qdrant.create_collection(
                    name,
                    vectors_config=models.VectorParams(size=vector_size, distance=distance),
                )
            self._stores[name] = QdrantVectorStore(
                client=self.qdrant, collection_name=name
            )
        return self._stores[name]

    def code_vs(self, collection_prefix: str) -> QdrantVectorStore:
        """Return vector store for code nodes using ``collection_prefix``."""

        name = f"{collection_prefix}code_nodes"
        return self._create_vs(name, self.cfg.qdrant.code_distance, self._code_vector_size)

    def file_vs(self, collection_prefix: str) -> QdrantVectorStore:
        """Return vector store for file cards using ``collection_prefix``."""

        name = f"{collection_prefix}file_cards"
        return self._create_vs(name, self.cfg.qdrant.text_distance, self._text_vector_size)

    def dir_vs(self, collection_prefix: str) -> QdrantVectorStore:
        """Return vector store for directory cards using ``collection_prefix``."""

        name = f"{collection_prefix}dir_cards"
        return self._create_vs(name, self.cfg.qdrant.text_distance, self._text_vector_size)

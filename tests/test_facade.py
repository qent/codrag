from pathlib import Path

from qdrant_client import QdrantClient
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import MockLLM

from rag_service.config import AppConfig
from rag_service.llama_facade import LlamaIndexFacade


class DummyEmbedding(BaseEmbedding):
    def _get_text_embedding(self, text: str):
        return [0.0]

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)

    def _get_query_embedding(self, text: str):
        return [0.0]

    async def _aget_query_embedding(self, text: str):
        return self._get_query_embedding(text)


def test_facade_vector_store_names(tmp_path):
    """Vector stores should use expected collection suffixes."""

    cfg = AppConfig.load(Path("config.json"))
    qdrant = QdrantClient(location=":memory:")
    Settings.embed_model = DummyEmbedding()
    Settings.llm = MockLLM()
    facade = LlamaIndexFacade(cfg, qdrant, initialize=False)
    assert facade.code_vs().collection_name.endswith("code_nodes")
    assert facade.file_vs().collection_name.endswith("file_cards")
    assert facade.dir_vs().collection_name.endswith("dir_cards")

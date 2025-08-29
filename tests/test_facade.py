from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
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
    prefix = "demo_"
    assert facade.code_vs(prefix).collection_name.endswith("code_nodes")
    assert facade.file_vs(prefix).collection_name.endswith("file_cards")
    assert facade.dir_vs(prefix).collection_name.endswith("dir_cards")


def test_facade_respects_distances(tmp_path):
    """Facade should recreate collections with configured distances."""

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        '{'
        '"version":1,'
        '"indexing":{},'
        '"ast":{},'
        '"openai":{"embeddings":{"base_url":"","model":"m","api_key":"k"},'
        '"generator":{"base_url":"","model":"m","api_key":"k"},'
        '"query_rewriter":{"base_url":"","model":"m","api_key":"k"}},'
        '"prompts":{"file_card_md":"a","dir_card_md":"b"},'
        '"qdrant":{"code_distance":"Euclid","text_distance":"Manhattan"},'
        '"llamaindex":{}'
        '}'
    )
    cfg = AppConfig.load(cfg_path)
    qdrant = QdrantClient(location=":memory:")
    Settings.embed_model = DummyEmbedding()
    Settings.llm = MockLLM()
    facade = LlamaIndexFacade(cfg, qdrant, initialize=False)
    prefix = "demo_"
    facade.code_vs(prefix)
    facade.file_vs(prefix)
    code_info = qdrant.get_collection(f"{prefix}code_nodes")
    file_info = qdrant.get_collection(f"{prefix}file_cards")
    assert code_info.config.params.vectors.distance == models.Distance.EUCLID
    assert file_info.config.params.vectors.distance == models.Distance.MANHATTAN

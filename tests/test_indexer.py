from pathlib import Path

from qdrant_client import QdrantClient
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import MockLLM

from rag_service.config import AppConfig
from rag_service.indexer import index_path
from rag_service.llama_facade import LlamaIndexFacade
from rag_service.retriever import build_query_engine


class DummyEmbedding(BaseEmbedding):
    def _get_text_embedding(self, text: str):
        return [0.0, 0.0, 0.0]

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)

    def _get_query_embedding(self, text: str):
        return [0.0, 0.0, 0.0]

    async def _aget_query_embedding(self, text: str):
        return self._get_query_embedding(text)


Settings.embed_model = DummyEmbedding()
Settings.llm = MockLLM()


def test_index_and_query(tmp_path):
    """Index a file and ensure it can be retrieved."""

    src = tmp_path / "src"
    src.mkdir()
    code = "class Foo { fun bar() {} }"
    (src / "Test.kt").write_text(code)

    cfg = AppConfig.load(Path("config.json"))
    qdrant = QdrantClient(location=":memory:")
    llama = LlamaIndexFacade(cfg, qdrant, initialize=False)
    stats = index_path(src, cfg, qdrant, llama)
    assert stats.code_nodes_upserted > 0

    qe = build_query_engine(cfg, qdrant, llama)
    res = qe.retrieve("Foo")
    assert res


def test_scan_respects_blacklist(tmp_path):
    """Ensure blacklisted paths are skipped during scanning."""

    src = tmp_path / "src"
    (src / "skip").mkdir(parents=True)
    (src / "skip" / "Test.kt").write_text("class A{}")
    cfg = AppConfig.load(Path("config.json"))
    cfg.indexing.blacklist.append("skip/*")
    qdrant = QdrantClient(location=":memory:")
    llama = LlamaIndexFacade(cfg, qdrant, initialize=False)
    stats = index_path(src, cfg, qdrant, llama)
    assert stats.files_total == 0

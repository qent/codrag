from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import MockLLM
from pydantic import PrivateAttr

from rag_service.config import AppConfig
from rag_service.indexer import index_path
from rag_service.llama_facade import LlamaIndexFacade
from rag_service.retriever import build_query_engine
from rag_service.collection_utils import collection_prefix_from_path


class DummyEmbedding(BaseEmbedding):
    """Embedding model returning zeros for every request."""

    def _get_text_embedding(self, text: str) -> list[float]:
        """Return a dummy embedding for text."""

        return [0.0, 0.0, 0.0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Asynchronously return a dummy embedding for text."""

        return self._get_text_embedding(text)

    def _get_query_embedding(self, text: str) -> list[float]:
        """Return a dummy embedding for a query."""

        return [0.0, 0.0, 0.0]

    async def _aget_query_embedding(self, text: str) -> list[float]:
        """Asynchronously return a dummy embedding for a query."""

        return self._get_query_embedding(text)


Settings.code_embed_model = DummyEmbedding()
Settings.text_embed_model = DummyEmbedding()
Settings.llm = MockLLM()


def test_index_and_query(tmp_path):
    """Index files and ensure directory cards can be retrieved."""

    src = tmp_path / "src"
    pkg = src / "pkg"
    pkg.mkdir(parents=True)
    code = "class Foo { fun bar() {} }"
    (pkg / "Test.kt").write_text(code)

    cfg = AppConfig.load(Path("config.json"))
    cfg.features.process_directories = True
    qdrant = QdrantClient(location=":memory:")
    llama = LlamaIndexFacade(cfg, qdrant, initialize=False)
    stats = index_path(src, cfg, qdrant, llama)
    assert stats.code_nodes_upserted > 0
    assert stats.dir_cards_upserted > 0

    prefix = collection_prefix_from_path(src)
    qe = build_query_engine(cfg, qdrant, llama, prefix)
    res = qe.retrieve("Foo")
    assert any(n.node.metadata.get("type") == "dir_card" for n in res)


def test_directories_skipped_by_default(tmp_path):
    """Directory cards are not generated when the feature flag is disabled."""

    src = tmp_path / "src"
    src.mkdir()
    code = "class Foo { fun bar() {} }"
    (src / "Test.kt").write_text(code)

    cfg = AppConfig.load(Path("config.json"))
    qdrant = QdrantClient(location=":memory:")
    llama = LlamaIndexFacade(cfg, qdrant, initialize=False)
    stats = index_path(src, cfg, qdrant, llama)
    assert stats.dir_cards_upserted == 0

    prefix = collection_prefix_from_path(src)
    qe = build_query_engine(cfg, qdrant, llama, prefix)
    res = qe.retrieve("Foo")
    assert all(n.node.metadata.get("type") != "dir_card" for n in res)


class CaptureLLM(MockLLM):
    """LLM that stores the last prompt."""

    _last_prompt: str = PrivateAttr("")

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        """Record the prompt and return a mock completion."""

        self._last_prompt = prompt
        return super().complete(prompt, **kwargs)

    @property
    def last_prompt(self) -> str:
        """Return the last recorded prompt."""

        return self._last_prompt


def test_file_text_passed_to_llm(tmp_path):
    """Ensure file text, not path, is sent to the LLM."""

    src = tmp_path / "src"
    src.mkdir()
    code = "class Foo { fun bar() {} }"
    (src / "Test.kt").write_text(code)

    cfg = AppConfig.load(Path("config.json"))
    qdrant = QdrantClient(location=":memory:")
    llama = LlamaIndexFacade(cfg, qdrant, initialize=False)

    original_llm = Settings.llm
    capture_llm = CaptureLLM()
    Settings.llm = capture_llm
    try:
        index_path(src, cfg, qdrant, llama)
    finally:
        Settings.llm = original_llm

    assert code in capture_llm.last_prompt


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


def test_skip_cached_files(tmp_path):
    """Ensure unchanged files are not indexed again."""

    src = tmp_path / "src"
    src.mkdir()
    code = "class Foo { fun bar() {} }"
    (src / "Test.kt").write_text(code)

    cfg = AppConfig.load(Path("config.json"))
    qdrant = QdrantClient(location=":memory:")
    llama = LlamaIndexFacade(cfg, qdrant, initialize=False)

    first = index_path(src, cfg, qdrant, llama)
    assert first.files_processed == 1

    second = index_path(src, cfg, qdrant, llama)
    assert second.files_skipped_cache == 1
    assert second.files_processed == 0
    assert second.code_nodes_upserted == 0

from pathlib import Path
from typing import Any
import pytest

from qdrant_client import QdrantClient
from types import SimpleNamespace
from unittest.mock import patch

from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import MockLLM
from pydantic import PrivateAttr

from rag_service.config import AppConfig
from rag_service.indexer import PathIndexer, index_path
from llama_index.core.schema import TextNode
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


class DummyLLM(MockLLM):
    """LLM stub returning structured output."""

    def with_structured_output(self, model):  # pragma: no cover - simple stub
        def _call(_):
            return SimpleNamespace(
                summary="s", key_points=[], embedding_text="e", keywords=[]
            )

        return _call


Settings.llm = DummyLLM()


class LCStub:
    """LangChain-like LLM stub capturing templated messages."""

    def __init__(self) -> None:
        self.messages = None

    def with_structured_output(self, model):  # pragma: no cover - simple stub
        def _call(chat_prompt_value):
            # ChatPromptTemplate pipes a ChatPromptValue with .messages
            self.messages = (
                chat_prompt_value.messages
                if hasattr(chat_prompt_value, "messages")
                else None
            )
            return SimpleNamespace(
                summary="s", key_points=[], embedding_text="e", keywords=[]
            )

        return _call


@pytest.fixture(autouse=True)
def patch_langchain_llm():
    """Patch indexer to use a local LangChain stub instead of real LLM."""

    import rag_service.indexer as indexer_mod

    stub = LCStub()
    original = indexer_mod.build_langchain_llm
    indexer_mod.build_langchain_llm = lambda _cfg: stub
    try:
        yield stub
    finally:
        indexer_mod.build_langchain_llm = original


def setup_function(_):
    """Reset LLM before each test."""

    Settings.llm = DummyLLM()


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
    # Avoid network download of reranker model
    cfg.llamaindex.retrieval.use_reranker = False
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
    cfg.llamaindex.retrieval.use_reranker = False
    qe = build_query_engine(cfg, qdrant, llama, prefix)
    res = qe.retrieve("Foo")
    assert all(n.node.metadata.get("type") != "dir_card" for n in res)


class CaptureLLM(MockLLM):
    """LLM capturing structured inputs and prompts."""

    _inputs: list[dict] = PrivateAttr(default_factory=list)
    _prompts: list[str] = PrivateAttr(default_factory=list)

    def with_structured_output(self, model):  # pragma: no cover - simple stub
        def _call(data):
            self._inputs.append(data)
            return SimpleNamespace(
                summary="s", key_points=[], embedding_text="e", keywords=[]
            )

        return _call

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        """Record the prompt and return a mock completion."""

        self._prompts.append(prompt)
        return super().complete(prompt, **kwargs)

    @property
    def last_input(self) -> dict:
        """Return the last captured structured input."""

        return self._inputs[-1] if self._inputs else {}

    @property
    def prompts(self) -> list[str]:
        """Return a copy of recorded prompts."""

        return list(self._prompts)

    @property
    def inputs(self) -> list[dict]:
        """Return a copy of structured inputs."""

        return list(self._inputs)


def test_file_text_passed_to_llm(tmp_path, patch_langchain_llm: LCStub):
    """Ensure file text, not path, is sent to the LLM."""

    src = tmp_path / "src"
    src.mkdir()
    code = "class Foo { fun bar() {} }"
    (src / "Test.kt").write_text(code)

    cfg = AppConfig.load(Path("config.json"))
    qdrant = QdrantClient(location=":memory:")
    llama = LlamaIndexFacade(cfg, qdrant, initialize=False)

    # Run indexing and assert the templated prompt includes file content
    index_path(src, cfg, qdrant, llama)
    messages = patch_langchain_llm.messages
    assert messages, "Expected ChatPrompt messages to be captured"
    content = messages[0].content
    assert code in content


def test_repo_prompt_included(tmp_path, patch_langchain_llm: LCStub):
    """Ensure repository prompt is appended to file and directory prompts."""

    src = tmp_path / "src"
    pkg = src / "pkg"
    pkg.mkdir(parents=True)
    code = "class Foo { fun bar() {} }"
    (pkg / "Test.kt").write_text(code)

    cfg = AppConfig.load(Path("config.json"))
    cfg.features.process_directories = True
    qdrant = QdrantClient(location=":memory:")
    llama = LlamaIndexFacade(cfg, qdrant, initialize=False)

    repo_desc = "Custom repository description"
    # Capture directory prompts through Settings.llm
    original_llm = Settings.llm
    capture_llm = CaptureLLM()
    Settings.llm = capture_llm
    try:
        index_path(src, cfg, qdrant, llama, repo_prompt=repo_desc)
    finally:
        Settings.llm = original_llm

    # Directory prompts contain repo description
    assert capture_llm.prompts
    assert all(repo_desc in p for p in capture_llm.prompts)

    # File card prompt (LangChain) also contains repo description
    messages = patch_langchain_llm.messages
    assert messages, "Expected ChatPrompt messages to be captured"
    assert repo_desc in messages[0].content


def test_file_card_metadata_stored(tmp_path):
    """File card should store embedding text and metadata."""

    src = tmp_path / "src"
    src.mkdir()
    code = "class Foo { fun bar() {} }"
    (src / "a.kt").write_text(code)

    cfg = AppConfig.load(Path("config.json"))
    qdrant = QdrantClient(location=":memory:")
    llama = LlamaIndexFacade(cfg, qdrant, initialize=False)

    index_path(src, cfg, qdrant, llama)

    prefix = collection_prefix_from_path(src)
    points, _ = qdrant.scroll(f"{prefix}file_cards", limit=1)
    payload = points[0].payload
    import json

    node = json.loads(payload["_node_content"])
    assert node["text"] == "e"
    assert payload["summary"] == "s"
    assert payload["key_points"] == []
    assert payload["keywords"] == []


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


def test_code_nodes_have_neighbor_metadata(tmp_path):
    """Ensure generated code chunks include neighbor metadata."""

    src = tmp_path / "src"
    src.mkdir()
    file_path = src / "Test.kt"

    cfg = AppConfig.load(Path("config.json"))
    cfg.ast.chunk_lines = 3
    cfg.ast.chunk_overlap = 1
    qdrant = QdrantClient(location=":memory:")
    llama = LlamaIndexFacade(cfg, qdrant, initialize=False)
    indexer = PathIndexer(cfg, qdrant, llama, "test")

    with patch("rag_service.indexer.CodeSplitter") as splitter_cls:
        splitter = splitter_cls.return_value
        splitter.get_nodes_from_documents.return_value = [
            TextNode(text="a"),
            TextNode(text="b"),
            TextNode(text="c"),
        ]
        nodes = indexer._create_nodes("dummy", file_path, "hash")

    for i, node in enumerate(nodes):
        expected_prev = nodes[i - 1].node_id if i > 0 else None
        expected_next = nodes[i + 1].node_id if i < len(nodes) - 1 else None
        assert node.metadata["prev_id"] == expected_prev
        assert node.metadata["next_id"] == expected_next
        assert node.metadata["file_path"] == str(file_path)
        assert node.metadata["lang"] == "kotlin"
        assert node.metadata["symbols_in_chunk"] == len(node.get_content())

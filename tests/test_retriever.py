from types import SimpleNamespace
from typing import Sequence

from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

from rag_service.retriever import build_query_engine, fuse_results


def _node(node_id: str, score: float) -> NodeWithScore:
    """Create a ``NodeWithScore`` for testing."""

    return NodeWithScore(node=TextNode(id_=node_id, text=""), score=score)


class DummyEmbedding(BaseEmbedding):
    def _get_text_embedding(self, text: str):
        return [0.0]

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)

    def _get_query_embedding(self, text: str):
        return [0.0]

    async def _aget_query_embedding(self, text: str):
        return self._get_query_embedding(text)


Settings.embed_model = DummyEmbedding()


def test_relative_score_fusion_sorting() -> None:
    """Relative score fusion should respect weights and sort nodes."""

    code = [_node("a", 0.8), _node("b", 0.4)]
    file = [_node("c", 0.9)]
    fused = fuse_results(
        code, file, mode="relative_score", code_weight=1.0, file_weight=2.0, rrf_k=60
    )
    ids = [n.node.node_id for n in fused]
    assert ids == ["c", "a", "b"]


def test_rrf_fusion_sorting() -> None:
    """Reciprocal rank fusion should rescore using ranks."""

    code = [_node("a", 0.5), _node("b", 0.4)]
    file = [_node("c", 0.3)]
    fused = fuse_results(code, file, mode="rrf", code_weight=1.0, file_weight=2.0, rrf_k=0)
    ids = [n.node.node_id for n in fused]
    assert ids == ["c", "a", "b"]


def test_simple_retriever_uses_query_rewriter(monkeypatch) -> None:
    """SimpleRetriever should pass specialized queries to each retriever."""

    retrieval_cfg = SimpleNamespace(
        code_nodes_top_k=1,
        file_cards_top_k=1,
        dir_cards_top_k=1,
        fusion_mode="relative_score",
        code_weight=1.0,
        file_weight=1.0,
        dir_weight=1.0,
        rrf_k=60,
        max_expansions=2,
    )
    query_rewriter_cfg = SimpleNamespace(
        model="m",
        base_url="http://localhost",
        api_key="k",
        verify_ssl=True,
        timeout_sec=60,
        retries=0,
    )
    cfg = SimpleNamespace(
        llamaindex=SimpleNamespace(retrieval=retrieval_cfg),
        openai=SimpleNamespace(query_rewriter=query_rewriter_cfg),
        features=SimpleNamespace(process_directories=True),
    )

    class DummyRet:
        def __init__(self) -> None:
            self.queries = []

        def retrieve(self, query: str):  # pragma: no cover - simple stub
            self.queries.append(query)
            return []

    code_ret = DummyRet()
    file_ret = DummyRet()
    dir_ret = DummyRet()

    code_vs = object()
    file_vs = object()
    dir_vs = object()
    llama = SimpleNamespace(
        code_vs=lambda prefix: code_vs,
        file_vs=lambda prefix: file_vs,
        dir_vs=lambda prefix: dir_vs,
    )

    def from_vs(vs, embed_model=None):  # type: ignore[override]
        class _Idx:
            def as_retriever(self, similarity_top_k):
                return {code_vs: code_ret, file_vs: file_ret, dir_vs: dir_ret}[vs]

        return _Idx()

    monkeypatch.setattr(
        "rag_service.retriever.VectorStoreIndex.from_vector_store", from_vs
    )
    def _rewrite(q, cfg=None):
        return (f"{q}c", f"{q}f", f"{q}d")

    monkeypatch.setattr("rag_service.retriever.rewrite_for_collections", _rewrite)
    monkeypatch.setattr(
        "rag_service.retriever.expand_queries", lambda q, cfg, n: ["alt1", "alt2"]
    )

    retriever = build_query_engine(cfg, qdrant=None, llama=llama, collection_prefix="test_")
    retriever.retrieve("orig")
    assert code_ret.queries == ["origc", "alt1c", "alt2c"]
    assert file_ret.queries == ["origf", "alt1f", "alt2f"]
    assert dir_ret.queries == ["origd", "alt1d", "alt2d"]


def test_retriever_skips_directories_when_flag_disabled(monkeypatch) -> None:
    """Directory retriever is bypassed when the feature flag is off."""

    retrieval_cfg = SimpleNamespace(
        code_nodes_top_k=1,
        file_cards_top_k=1,
        dir_cards_top_k=1,
        fusion_mode="relative_score",
        code_weight=1.0,
        file_weight=1.0,
        dir_weight=1.0,
        rrf_k=60,
        max_expansions=2,
    )
    query_rewriter_cfg = SimpleNamespace(
        model="m",
        base_url="http://localhost",
        api_key="k",
        verify_ssl=True,
        timeout_sec=60,
        retries=0,
    )
    cfg = SimpleNamespace(
        llamaindex=SimpleNamespace(retrieval=retrieval_cfg),
        openai=SimpleNamespace(query_rewriter=query_rewriter_cfg),
        features=SimpleNamespace(process_directories=False),
    )

    class DummyRet:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def retrieve(self, query: str):  # pragma: no cover - simple stub
            self.queries.append(query)
            return []

    code_ret = DummyRet()
    file_ret = DummyRet()
    dir_ret = DummyRet()

    code_vs = object()
    file_vs = object()
    dir_vs = object()
    llama = SimpleNamespace(
        code_vs=lambda prefix: code_vs,
        file_vs=lambda prefix: file_vs,
        dir_vs=lambda prefix: dir_vs,
    )

    def from_vs(vs, embed_model=None):  # type: ignore[override]
        class _Idx:
            def as_retriever(self, similarity_top_k):
                return {code_vs: code_ret, file_vs: file_ret, dir_vs: dir_ret}[vs]

        return _Idx()

    monkeypatch.setattr(
        "rag_service.retriever.VectorStoreIndex.from_vector_store", from_vs
    )
    def _rewrite(q, cfg=None):
        return (f"{q}c", f"{q}f", f"{q}d")

    monkeypatch.setattr("rag_service.retriever.rewrite_for_collections", _rewrite)
    monkeypatch.setattr(
        "rag_service.retriever.expand_queries", lambda q, cfg, n: ["alt1", "alt2"]
    )

    retriever = build_query_engine(cfg, qdrant=None, llama=llama, collection_prefix="test_")
    retriever.retrieve("orig")
    assert code_ret.queries == ["origc", "alt1c", "alt2c"]
    assert file_ret.queries == ["origf", "alt1f", "alt2f"]
    assert dir_ret.queries == []


def test_simple_retriever_filters_duplicate_nodes(monkeypatch) -> None:
    """SimpleRetriever should remove duplicate nodes across queries and retrievers."""

    retrieval_cfg = SimpleNamespace(
        code_nodes_top_k=1,
        file_cards_top_k=1,
        dir_cards_top_k=1,
        fusion_mode="relative_score",
        code_weight=1.0,
        file_weight=1.0,
        dir_weight=1.0,
        rrf_k=60,
        max_expansions=1,
    )
    query_rewriter_cfg = SimpleNamespace(
        model="m",
        base_url="http://localhost",
        api_key="k",
        verify_ssl=True,
        timeout_sec=60,
        retries=0,
    )
    cfg = SimpleNamespace(
        llamaindex=SimpleNamespace(retrieval=retrieval_cfg),
        openai=SimpleNamespace(query_rewriter=query_rewriter_cfg),
        features=SimpleNamespace(process_directories=False),
    )

    class DummyRet:
        def retrieve(self, query: str):  # pragma: no cover - simple stub
            return [_node("dup", 0.9)]

    code_ret = DummyRet()
    file_ret = DummyRet()

    code_vs = object()
    file_vs = object()
    llama = SimpleNamespace(
        code_vs=lambda prefix: code_vs,
        file_vs=lambda prefix: file_vs,
        dir_vs=lambda prefix: None,
    )

    def from_vs(vs, embed_model=None):  # type: ignore[override]
        class _Idx:
            def as_retriever(self, similarity_top_k):
                return {code_vs: code_ret, file_vs: file_ret}[vs]

        return _Idx()

    monkeypatch.setattr(
        "rag_service.retriever.VectorStoreIndex.from_vector_store", from_vs
    )
    monkeypatch.setattr(
        "rag_service.retriever.rewrite_for_collections", lambda q, cfg=None: (q, q, q)
    )
    monkeypatch.setattr(
        "rag_service.retriever.expand_queries", lambda q, cfg, n: ["alt"]
    )

    retriever = build_query_engine(cfg, qdrant=None, llama=llama, collection_prefix="test_")
    results = retriever.retrieve("orig")
    ids = [n.node.node_id for n in results]
    assert ids == ["dup"]


def test_simple_retriever_applies_reranker(monkeypatch) -> None:
    """Nodes should be reranked when the reranker flag is enabled."""

    retrieval_cfg = SimpleNamespace(
        code_nodes_top_k=1,
        file_cards_top_k=1,
        dir_cards_top_k=0,
        fusion_mode="relative_score",
        code_weight=1.0,
        file_weight=1.0,
        dir_weight=1.0,
        rrf_k=60,
        max_expansions=0,
        use_reranker=True,
    )
    query_rewriter_cfg = SimpleNamespace(
        model="m",
        base_url="http://localhost",
        api_key="k",
        verify_ssl=True,
        timeout_sec=60,
        retries=0,
    )
    cfg = SimpleNamespace(
        llamaindex=SimpleNamespace(retrieval=retrieval_cfg),
        openai=SimpleNamespace(query_rewriter=query_rewriter_cfg),
        features=SimpleNamespace(process_directories=False),
    )

    code_nodes = [_node("a", 0.8), _node("b", 0.4)]

    class DummyRet:
        def retrieve(self, query: str):  # pragma: no cover - simple stub
            return code_nodes

    code_ret = DummyRet()
    code_vs = object()
    llama = SimpleNamespace(code_vs=lambda prefix: code_vs, file_vs=lambda prefix: code_vs)

    def from_vs(vs, embed_model=None):  # type: ignore[override]
        class _Idx:
            def as_retriever(self, similarity_top_k):
                return code_ret

        return _Idx()

    class DummyReranker:
        def __init__(self) -> None:
            self.called = False

        def rerank(self, query: str, nodes: Sequence[NodeWithScore]):
            self.called = True
            return list(reversed(nodes))

    monkeypatch.setattr(
        "rag_service.retriever.VectorStoreIndex.from_vector_store", from_vs
    )
    monkeypatch.setattr(
        "rag_service.retriever.rewrite_for_collections", lambda q, cfg=None: (q, q, q)
    )
    monkeypatch.setattr(
        "rag_service.retriever.expand_queries", lambda q, cfg, n: []
    )
    monkeypatch.setattr(
        "rag_service.retriever.CrossEncoderReranker", DummyReranker
    )

    retriever = build_query_engine(cfg, qdrant=None, llama=llama, collection_prefix="test_")
    results = retriever.retrieve("q")
    assert [n.node.node_id for n in results] == ["b", "a"]

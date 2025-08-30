from __future__ import annotations

from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode

from rag_service.query_metadata import (
    FileQueryMetadata,
    boost_file_nodes_by_metadata,
    extract_file_query_metadata,
)


def test_extract_file_query_metadata_returns_empty_without_llm() -> None:
    """Extractor should return empty fields when no LLM is configured."""

    meta = extract_file_query_metadata("поиск jwt авторизация", cfg=None)
    assert isinstance(meta, FileQueryMetadata)
    assert meta.languages == []
    assert meta.keywords == []
    assert meta.dir_paths == []
    assert meta.file_extensions == []
    assert meta.filenames == []


def _nws(node_id: str, score: float, metadata: dict) -> NodeWithScore:
    node = TextNode(id_=node_id, text="", metadata=metadata)
    return NodeWithScore(node=node, score=score)


def test_boost_file_nodes_by_metadata_reorders() -> None:
    """Boosting should prioritize nodes matching language and keywords."""

    nodes = [
        _nws("A", 0.40, {"lang": "python", "keywords": ["auth", "jwt"], "file_path": "srv/auth/jwt.py"}),
        _nws("B", 0.60, {"lang": "js", "keywords": ["react"], "file_path": "web/ui/app.jsx"}),
        _nws("C", 0.50, {"file_path": "misc/readme.md"}),
    ]
    meta = FileQueryMetadata(
        languages=["python"], keywords=["jwt"], dir_paths=["auth/"], file_extensions=["py"], filenames=[]
    )
    boosted = boost_file_nodes_by_metadata(nodes, meta)
    assert [n.node.node_id for n in boosted][:2] == ["A", "B"]


def test_simple_retriever_applies_file_metadata_boost(monkeypatch) -> None:
    """Integration: SimpleRetriever should boost file-card results using query metadata."""

    from rag_service.retriever import build_query_engine

    # Minimal retrieval config with only file results
    retrieval_cfg = SimpleNamespace(
        code_nodes_top_k=0,
        file_cards_top_k=2,
        dir_cards_top_k=0,
        fusion_mode="relative_score",
        code_weight=1.0,
        file_weight=1.0,
        dir_weight=1.0,
        rrf_k=60,
        max_expansions=0,
        use_reranker=False,
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

    class DummyFileRet:
        def retrieve(self, query: str):  # pragma: no cover - simple stub
            return [
                _nws("A", 0.40, {"lang": "python", "keywords": ["auth", "jwt"], "file_path": "srv/auth/jwt.py"}),
                _nws("B", 0.60, {"lang": "js", "keywords": ["react"], "file_path": "web/ui/app.jsx"}),
            ]

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
                return DummyFileRet() if vs is file_vs else SimpleNamespace(retrieve=lambda q: [])

        return _Idx()

    # No query rewriting / expansions
    monkeypatch.setattr(
        "rag_service.retriever.VectorStoreIndex.from_vector_store", from_vs
    )
    monkeypatch.setattr(
        "rag_service.retriever.rewrite_for_collections", lambda q, cfg=None: (q, q, q)
    )
    monkeypatch.setattr(
        "rag_service.retriever.expand_queries", lambda q, cfg, n: []
    )

    # Force metadata extractor to return our preferences
    monkeypatch.setattr(
        "rag_service.retriever.extract_file_query_metadata",
        lambda q, cfg=None: FileQueryMetadata(languages=["python"], keywords=["jwt"], dir_paths=[], file_extensions=[], filenames=[]),
    )

    retriever = build_query_engine(cfg, qdrant=None, llama=llama, collection_prefix="test_")
    results = retriever.retrieve("jwt auth")
    assert [n.node.node_id for n in results][:2] == ["A", "B"]

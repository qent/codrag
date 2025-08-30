from types import SimpleNamespace

from rag_service.retriever import build_query_engine


def test_hyde_generates_exact_n_docs(monkeypatch) -> None:
    """HyDE should generate and use exactly N drafts when configured."""

    retrieval_cfg = SimpleNamespace(
        code_nodes_top_k=1,
        file_cards_top_k=0,
        dir_cards_top_k=0,
        fusion_mode="relative_score",
        code_weight=1.0,
        file_weight=1.0,
        dir_weight=1.0,
        rrf_k=60,
        max_expansions=0,
        use_hyde_for_code=True,
        hyde_docs=3,
    )
    qc = SimpleNamespace(model="m", base_url="http://localhost", api_key="k", verify_ssl=True, timeout_sec=60, retries=0)
    cfg = SimpleNamespace(
        llamaindex=SimpleNamespace(retrieval=retrieval_cfg),
        openai=SimpleNamespace(query_rewriter=qc, generator=qc),
        features=SimpleNamespace(process_directories=False),
    )

    class DummyRet:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def retrieve(self, query: str):  # pragma: no cover - simple stub
            self.queries.append(query)
            return []

    code_ret = DummyRet()
    code_vs = object()
    llama = SimpleNamespace(code_vs=lambda prefix: code_vs, file_vs=lambda prefix: code_vs)

    def from_vs(vs, embed_model=None):  # type: ignore[override]
        class _Idx:
            def as_retriever(self, similarity_top_k):
                return code_ret

        return _Idx()

    monkeypatch.setattr(
        "rag_service.retriever.VectorStoreIndex.from_vector_store", from_vs
    )
    monkeypatch.setattr(
        "rag_service.retriever.rewrite_for_collections", lambda q, cfg=None: (q, q, q)
    )

    def fake_hyde(query: str, cfg=None, n: int = 1, system_prompt: str = ""):
        return [f"D{i}:{query}" for i in range(n)]

    monkeypatch.setattr("rag_service.retriever.hyde_code_documents", fake_hyde)

    retriever = build_query_engine(cfg, qdrant=None, llama=llama, collection_prefix="t_")
    retriever.retrieve("Q")
    d_queries = [q for q in code_ret.queries if q.startswith("D")]
    assert len(d_queries) == 3


def test_hyde_zero_docs_skips(monkeypatch) -> None:
    """HyDE disabled or zero docs should not generate drafts."""

    retrieval_cfg = SimpleNamespace(
        code_nodes_top_k=1,
        file_cards_top_k=0,
        dir_cards_top_k=0,
        fusion_mode="relative_score",
        code_weight=1.0,
        file_weight=1.0,
        dir_weight=1.0,
        rrf_k=60,
        max_expansions=0,
        use_hyde_for_code=True,
        hyde_docs=0,
    )
    qc = SimpleNamespace(model="m", base_url="http://localhost", api_key="k", verify_ssl=True, timeout_sec=60, retries=0)
    cfg = SimpleNamespace(
        llamaindex=SimpleNamespace(retrieval=retrieval_cfg),
        openai=SimpleNamespace(query_rewriter=qc, generator=qc),
        features=SimpleNamespace(process_directories=False),
    )

    class DummyRet:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def retrieve(self, query: str):  # pragma: no cover - simple stub
            self.queries.append(query)
            return []

    code_ret = DummyRet()
    code_vs = object()
    llama = SimpleNamespace(code_vs=lambda prefix: code_vs, file_vs=lambda prefix: code_vs)

    def from_vs(vs, embed_model=None):  # type: ignore[override]
        class _Idx:
            def as_retriever(self, similarity_top_k):
                return code_ret

        return _Idx()

    monkeypatch.setattr(
        "rag_service.retriever.VectorStoreIndex.from_vector_store", from_vs
    )
    monkeypatch.setattr(
        "rag_service.retriever.rewrite_for_collections", lambda q, cfg=None: (q, q, q)
    )
    # Ensure hyde_code_documents would be called if N>0
    called = {"n": 0}

    def fake_hyde(query: str, cfg=None, n: int = 1, system_prompt: str = ""):
        called["n"] += 1
        return []

    monkeypatch.setattr("rag_service.retriever.hyde_code_documents", fake_hyde)

    retriever = build_query_engine(cfg, qdrant=None, llama=llama, collection_prefix="t_")
    retriever.retrieve("Q")
    assert called["n"] == 0


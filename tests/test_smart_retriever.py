from types import SimpleNamespace
from typing import Sequence

from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

from rag_service.retriever import build_query_engine


class DummyEmbedding(BaseEmbedding):
    """Minimal embedding model used by tests to avoid external calls."""

    def _get_text_embedding(self, text: str):  # type: ignore[override]
        return [0.0]

    async def _aget_text_embedding(self, text: str):  # type: ignore[override]
        return self._get_text_embedding(text)

    def _get_query_embedding(self, text: str):  # type: ignore[override]
        return [0.0]

    async def _aget_query_embedding(self, text: str):  # type: ignore[override]
        return self._get_query_embedding(text)


Settings.code_embed_model = DummyEmbedding()
Settings.text_embed_model = DummyEmbedding()


def _node(node_id: str, score: float, metadata: dict) -> NodeWithScore:
    """Helper to create a ``NodeWithScore`` with metadata."""

    return NodeWithScore(node=TextNode(id_=node_id, text="", metadata=metadata), score=score)


def test_smart_retriever_expands_neighbors(monkeypatch) -> None:
    """SmartRetriever should append prev/next neighbors for code nodes when available."""

    # Retrieval config with no expansions, no reranker, no directories
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
        use_reranker=False,
        use_hyde_for_code=False,
        retriever="smart",
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
        openai=SimpleNamespace(query_rewriter=query_rewriter_cfg, generator=query_rewriter_cfg),
        features=SimpleNamespace(process_directories=False),
    )

    # One code node with neighbors
    code_meta = {
        "file_path": "src/a.py",
        "prev_id": "P",
        "next_id": "N",
    }
    code_nodes = [_node("C", 1.0, code_meta)]

    class DummyRet:
        def __init__(self, nodes: Sequence[NodeWithScore]):
            self._nodes = list(nodes)

        def retrieve(self, _: str):  # pragma: no cover - simple stub
            return self._nodes

    code_ret = DummyRet(code_nodes)
    file_ret = DummyRet([])

    code_vs = object()
    file_vs = object()
    llama = SimpleNamespace(
        code_vs=lambda prefix: code_vs,
        file_vs=lambda prefix: file_vs,
        dir_vs=lambda prefix: None,
    )

    def from_vs(vs, embed_model=None):  # type: ignore[override]
        class _Idx:
            def as_retriever(self, similarity_top_k):  # pragma: no cover - stub
                return {code_vs: code_ret, file_vs: file_ret}[vs]

        return _Idx()

    monkeypatch.setattr(
        "rag_service.retriever.VectorStoreIndex.from_vector_store", from_vs
    )
    # Avoid network LLM calls in rewrite/metadata
    monkeypatch.setattr(
        "rag_service.retriever.rewrite_for_collections", lambda q, cfg=None: (q, q, q)
    )
    monkeypatch.setattr(
        "rag_service.retriever.extract_file_query_metadata",
        lambda q, cfg=None: SimpleNamespace(
            languages=[], keywords=[], dir_paths=[], file_extensions=[], filenames=[]
        ),
    )
    # Stub qdrant scroll to return neighbor payloads
    class ScrollPoint:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

    class QdrantStub:
        def scroll(self, collection: str, limit: int, scroll_filter=None):  # type: ignore[override]
            return [
                ScrollPoint({"node_id": "P", "file_path": "src/a.py", "text": "prev"}),
                ScrollPoint({"node_id": "N", "file_path": "src/a.py", "text": "next"}),
                ScrollPoint({"node_id": "X", "file_path": "src/a.py", "text": "other"}),
            ], None

    retriever = build_query_engine(cfg, qdrant=QdrantStub(), llama=llama, collection_prefix="t_")
    results = retriever.retrieve("q")
    ids = {n.node.node_id for n in results}
    # Contains original and both neighbors
    assert {"C", "P", "N"}.issubset(ids)


def test_smart_retriever_prioritizes_selected_files(monkeypatch) -> None:
    """Code retrieval should prefer nodes from file-card selected files."""

    retrieval_cfg = SimpleNamespace(
        code_nodes_top_k=2,
        file_cards_top_k=2,
        dir_cards_top_k=0,
        fusion_mode="relative_score",
        code_weight=1.0,
        file_weight=1.0,
        dir_weight=1.0,
        rrf_k=60,
        max_expansions=0,
        use_reranker=False,
        use_hyde_for_code=False,
        retriever="smart",
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
        openai=SimpleNamespace(query_rewriter=query_rewriter_cfg, generator=query_rewriter_cfg),
        features=SimpleNamespace(process_directories=False),
    )

    # File cards select A.py and B.py
    file_nodes = [
        _node("FA", 0.9, {"type": "file_card", "file_path": "A.py"}),
        _node("FB", 0.8, {"type": "file_card", "file_path": "B.py"}),
    ]
    # Code nodes include A, B, and C; top_k=2 should prefer A and B
    code_nodes = [
        _node("CA", 0.7, {"file_path": "A.py"}),
        _node("CC", 0.6, {"file_path": "C.py"}),
        _node("CB", 0.5, {"file_path": "B.py"}),
    ]

    class DummyRet:
        def __init__(self, nodes: Sequence[NodeWithScore]):
            self._nodes = list(nodes)

        def retrieve(self, _: str):  # pragma: no cover - simple stub
            return self._nodes

    code_ret = DummyRet(code_nodes)
    file_ret = DummyRet(file_nodes)
    code_vs = object()
    file_vs = object()
    llama = SimpleNamespace(
        code_vs=lambda prefix: code_vs,
        file_vs=lambda prefix: file_vs,
        dir_vs=lambda prefix: None,
    )

    def from_vs(vs, embed_model=None):  # type: ignore[override]
        class _Idx:
            def as_retriever(self, similarity_top_k):  # pragma: no cover - stub
                return {code_vs: code_ret, file_vs: file_ret}[vs]

        return _Idx()

    monkeypatch.setattr(
        "rag_service.retriever.VectorStoreIndex.from_vector_store", from_vs
    )
    # Avoid network LLM calls in rewrite/metadata
    monkeypatch.setattr(
        "rag_service.retriever.rewrite_for_collections", lambda q, cfg=None: (q, q, q)
    )
    monkeypatch.setattr(
        "rag_service.retriever.extract_file_query_metadata",
        lambda q, cfg=None: SimpleNamespace(
            languages=[], keywords=[], dir_paths=[], file_extensions=[], filenames=[]
        ),
    )

    retriever = build_query_engine(cfg, qdrant=SimpleNamespace(), llama=llama, collection_prefix="t_")
    results = retriever.retrieve("q")
    # Collect file paths of code nodes in the fused results
    code_fps = [n.node.metadata.get("file_path") for n in results if n.node.metadata.get("type") != "file_card"]
    assert "A.py" in code_fps and "B.py" in code_fps
    assert "C.py" not in code_fps[:2]  # top-2 code nodes are from selected files

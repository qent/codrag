from llama_index.core.schema import NodeWithScore, TextNode

from rag_service.retrievers.utils import CrossEncoderReranker


def _nodes(ids: list[str]) -> list[NodeWithScore]:
    """Create a list of nodes with ascending default scores."""

    return [NodeWithScore(node=TextNode(id_=i, text=i), score=float(idx)) for idx, i in enumerate(ids)]


def test_reranker_fallback_without_encoder(monkeypatch) -> None:
    """When encoder is missing, rerank returns input order unchanged."""

    nodes = _nodes(["a", "b", "c"])
    reranker = CrossEncoderReranker()

    # If sentence-transformers is available, force fallback for this test
    if getattr(reranker, "_encoder", None) is not None:
        reranker._encoder = None  # type: ignore[attr-defined]

    out = reranker.rerank("q", nodes)
    assert [n.node.node_id for n in out] == ["a", "b", "c"]


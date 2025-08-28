from llama_index.core.schema import NodeWithScore, TextNode

from rag_service.retriever import fuse_results


def _node(node_id: str, score: float) -> NodeWithScore:
    """Create a ``NodeWithScore`` for testing."""

    return NodeWithScore(node=TextNode(id_=node_id, text=""), score=score)


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

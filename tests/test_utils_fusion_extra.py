from llama_index.core.schema import NodeWithScore, TextNode

from rag_service.retrievers.utils import fuse_results


def _node(node_id: str, score: float) -> NodeWithScore:
    """Create a minimal NodeWithScore for fusion tests."""

    return NodeWithScore(node=TextNode(id_=node_id, text=""), score=score)


def test_fuse_results_empty_inputs() -> None:
    """Fusion returns empty list when all inputs are empty."""

    out = fuse_results([], [])
    assert out == []


def test_relative_fusion_stable_order_on_ties() -> None:
    """When rescored values tie, code nodes precede file nodes due to stability."""

    # Both lists top-score normalize to 1.0 with equal weights
    code_nodes = [_node("c1", 0.5), _node("c2", 0.25)]
    file_nodes = [_node("f1", 0.75), _node("f2", 0.25)]
    fused = fuse_results(code_nodes, file_nodes, mode="relative_score", code_weight=1.0, file_weight=1.0)
    # Top elements from each list will score 1.0; expect code top first by stable sort
    ids = [n.node.node_id for n in fused]
    assert ids[0] == "c1"
    assert "f1" in ids[1:]


def test_rrf_handles_single_list() -> None:
    """RRF fusion on a single non-empty list works without errors."""

    file_nodes = [_node("f1", 0.3), _node("f2", 0.2)]
    fused = fuse_results([], file_nodes, mode="rrf", code_weight=1.0, file_weight=2.0, rrf_k=10)
    ids = [n.node.node_id for n in fused]
    assert ids == ["f1", "f2"]


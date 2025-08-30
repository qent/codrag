from __future__ import annotations

from typing import Sequence

from llama_index.core.schema import NodeWithScore, TextNode

from rag_service.query_metadata import FileQueryMetadata, boost_file_nodes_by_metadata


def _n(node_id: str, score: float, file_path: str) -> NodeWithScore:
    node = TextNode(id_=node_id, text="", metadata={"file_path": file_path})
    return NodeWithScore(node=node, score=score)


def test_boost_by_extension_and_filename() -> None:
    """Filename and extension signals add boosts and affect ordering."""

    nodes: list[NodeWithScore] = [
        _n("X", 0.50, "docs/readme.md"),
        _n("Y", 0.55, "srv/auth/jwt.py"),
        _n("Z", 0.60, "misc/other.txt"),
    ]
    meta = FileQueryMetadata(
        languages=[], keywords=[], dir_paths=[], file_extensions=["md"], filenames=["jwt.py"]
    )
    boosted = boost_file_nodes_by_metadata(nodes, meta)
    # Expect Y and X to receive +0.20 and rank over Z
    assert [n.node.node_id for n in boosted][:2] == ["Y", "X"]


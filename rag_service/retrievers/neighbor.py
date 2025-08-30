from __future__ import annotations

from typing import Dict, List, Sequence, Set

from llama_index.core.schema import NodeWithScore, TextNode
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue


def expand_with_neighbors(
    qdrant: QdrantClient,
    collection_prefix: str,
    nodes: Sequence[NodeWithScore],
    neighbor_decay: float = 0.9,
    neighbor_limit: int = 2,
) -> List[NodeWithScore]:
    """Return ``nodes`` extended with prev/next code neighbors when available.

    The function augments code nodes (non-file/dir types) with up to
    ``neighbor_limit`` immediate neighbors based on ``prev_id`` and ``next_id``
    present in the node metadata. Neighbor scores are scaled by
    ``neighbor_decay`` times the source node score. Duplicate node IDs are
    avoided.

    Parameters
    ----------
    qdrant:
        Qdrant client used to fetch code nodes for each file.
    collection_prefix:
        Prefix of the collection names (e.g., per-repository partitioning).
    nodes:
        Initial retrieval results to expand.
    neighbor_decay:
        Multiplier applied to the source node score for neighbor nodes.
    neighbor_limit:
        Maximum number of neighbors to add per source code node.

    Returns
    -------
    list[NodeWithScore]
        The expanded result list.
    """

    if not nodes or qdrant is None:
        return list(nodes)

    # Identify candidate code nodes and gather neighbor IDs grouped by file
    code_nodes: List[NodeWithScore] = []
    needed_by_file: Dict[str, Set[str]] = {}
    for n in nodes:
        ntype = str((n.node.metadata or {}).get("type", "code_node"))
        if ntype in ("file_card", "dir_card"):
            continue
        code_nodes.append(n)
        md = n.node.metadata or {}
        fp = str(md.get("file_path", ""))
        if not fp:
            continue
        ids = needed_by_file.setdefault(fp, set())
        for neigh_id in (md.get("prev_id"), md.get("next_id")):
            if isinstance(neigh_id, str) and neigh_id:
                ids.add(neigh_id)

    if not needed_by_file:
        return list(nodes)

    # Fetch code nodes once per file, build lookup by node_id
    lookup: Dict[str, dict] = {}
    collection = f"{collection_prefix}code_nodes"
    for file_path, ids in needed_by_file.items():
        if not ids:
            continue
        try:
            flt = Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=str(file_path)))
                ]
            )
            points, _ = qdrant.scroll(collection, limit=1000, scroll_filter=flt)
        except Exception:
            continue
        for p in points:
            payload = p.payload or {}
            nid = str(payload.get("node_id") or "")
            if nid:
                lookup[nid] = payload

    # Build neighbor NodeWithScore entries with score decay and per-node limit
    existing_ids = {n.node.node_id for n in nodes}
    expanded: List[NodeWithScore] = list(nodes)
    for n in code_nodes:
        md = n.node.metadata or {}
        added = 0
        for neigh_id in (md.get("prev_id"), md.get("next_id")):
            if added >= max(0, int(neighbor_limit)):
                break
            if not isinstance(neigh_id, str) or not neigh_id or neigh_id in existing_ids:
                continue
            payload = lookup.get(neigh_id)
            if not payload:
                continue
            text = str(payload.get("text", ""))
            meta = {k: v for k, v in payload.items() if k != "text"}
            neigh = TextNode(id_=neigh_id, text=text, metadata=meta)
            expanded.append(
                NodeWithScore(node=neigh, score=(n.score or 0.0) * float(neighbor_decay))
            )
            existing_ids.add(neigh_id)
            added += 1

    return expanded


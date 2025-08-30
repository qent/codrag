from __future__ import annotations

from typing import List, Sequence

from llama_index.core.schema import NodeWithScore
import logging


class CrossEncoderReranker:
    """Rescore nodes for a query using a cross-encoder model."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Create a reranker backed by ``model``.

        The sentence-transformers package is imported lazily. If it is
        unavailable, reranking will be skipped and a warning logged.
        """

        try:  # pragma: no cover - import error pathway
            from sentence_transformers import CrossEncoder
        except Exception:  # pragma: no cover - best effort warning
            logging.getLogger(__name__).warning(
                "sentence-transformers not installed, reranking disabled",
            )
            self._encoder = None
        else:
            try:
                self._encoder = CrossEncoder(model)
            except Exception:  # pragma: no cover - model load errors
                logging.getLogger(__name__).warning(
                    "CrossEncoder model could not be initialized, reranking disabled",
                )
                self._encoder = None

    def rerank(self, query: str, nodes: Sequence[NodeWithScore]) -> List[NodeWithScore]:
        """Return ``nodes`` sorted by cross-encoder relevance to ``query``."""

        if not nodes:
            return []
        if self._encoder is None:
            return list(nodes)
        pairs = [(query, n.node.get_content()) for n in nodes]
        scores = self._encoder.predict(pairs)
        for node, score in zip(nodes, scores):
            node.score = float(score)
        return sorted(nodes, key=lambda n: n.score, reverse=True)


def _relative_rescore(nodes: Sequence[NodeWithScore], weight: float) -> List[NodeWithScore]:
    """Rescore nodes relative to the top score and apply ``weight``."""

    if not nodes:
        return []
    top = nodes[0].score or 1.0
    for node in nodes:
        node.score = weight * (node.score / top)
    return list(nodes)


def _rrf_rescore(
    nodes: Sequence[NodeWithScore], weight: float, k: int,
) -> List[NodeWithScore]:
    """Apply reciprocal rank fusion to ``nodes`` with ``weight``."""

    rescored: List[NodeWithScore] = []
    for idx, node in enumerate(nodes, start=1):
        node.score = weight / (k + idx)
        rescored.append(node)
    return rescored


def fuse_results(
    code_nodes: Sequence[NodeWithScore],
    file_nodes: Sequence[NodeWithScore],
    dir_nodes: Sequence[NodeWithScore] | None = None,
    mode: str = "relative_score",
    code_weight: float = 1.0,
    file_weight: float = 1.0,
    dir_weight: float = 1.0,
    rrf_k: int = 60,
) -> List[NodeWithScore]:
    """Fuse code, file and directory retrieval results according to ``mode``."""

    dir_nodes = dir_nodes or []
    if mode == "rrf":
        rescored = _rrf_rescore(code_nodes, code_weight, rrf_k)
        rescored += _rrf_rescore(file_nodes, file_weight, rrf_k)
        rescored += _rrf_rescore(dir_nodes, dir_weight, rrf_k)
    else:
        rescored = _relative_rescore(code_nodes, code_weight)
        rescored += _relative_rescore(file_nodes, file_weight)
        rescored += _relative_rescore(dir_nodes, dir_weight)
    return sorted(rescored, key=lambda n: n.score, reverse=True)

from __future__ import annotations

from typing import List, Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from qdrant_client import QdrantClient

from .config import AppConfig
from .llama_facade import LlamaIndexFacade


def _relative_rescore(nodes: Sequence[NodeWithScore], weight: float) -> List[NodeWithScore]:
    """Rescore nodes relative to the top score and apply ``weight``."""

    if not nodes:
        return []
    top = nodes[0].score or 1.0
    for node in nodes:
        node.score = weight * (node.score / top)
    return list(nodes)


def _rrf_rescore(
    nodes: Sequence[NodeWithScore], weight: float, k: int
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
    mode: str,
    code_weight: float,
    file_weight: float,
    rrf_k: int,
) -> List[NodeWithScore]:
    """Fuse code and file retrieval results according to ``mode``."""

    if mode == "rrf":
        rescored = _rrf_rescore(code_nodes, code_weight, rrf_k)
        rescored += _rrf_rescore(file_nodes, file_weight, rrf_k)
    else:
        rescored = _relative_rescore(code_nodes, code_weight)
        rescored += _relative_rescore(file_nodes, file_weight)
    return sorted(rescored, key=lambda n: n.score, reverse=True)

def build_query_engine(cfg: AppConfig, qdrant: QdrantClient, llama: LlamaIndexFacade | None = None):
    """Build a simple retriever combining code and file card indexes."""

    llama = llama or LlamaIndexFacade(cfg, qdrant)
    code_vs = llama.code_vs()
    file_vs = llama.file_vs()

    code_ret = VectorStoreIndex.from_vector_store(code_vs).as_retriever(
        similarity_top_k=cfg.llamaindex.retrieval.code_nodes_top_k
    )
    file_ret = VectorStoreIndex.from_vector_store(file_vs).as_retriever(
        similarity_top_k=cfg.llamaindex.retrieval.file_cards_top_k
    )

    fusion_mode = cfg.llamaindex.retrieval.fusion_mode
    code_weight = getattr(cfg.llamaindex.retrieval, "code_weight", 1.0)
    file_weight = getattr(cfg.llamaindex.retrieval, "file_weight", 1.0)
    rrf_k = getattr(cfg.llamaindex.retrieval, "rrf_k", 60)

    class SimpleRetriever:
        def retrieve(self, query: str):
            code_nodes = code_ret.retrieve(query)
            file_nodes = file_ret.retrieve(query)
            return fuse_results(
                code_nodes, file_nodes, fusion_mode, code_weight, file_weight, rrf_k
            )

    return SimpleRetriever()

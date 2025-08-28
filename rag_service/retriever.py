from __future__ import annotations

from typing import List, Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from qdrant_client import QdrantClient

from .config import AppConfig
from .llama_facade import LlamaIndexFacade
from .query_rewriter import expand_queries, rewrite_for_collections


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

def build_query_engine(cfg: AppConfig, qdrant: QdrantClient, llama: LlamaIndexFacade | None = None):
    """Build a simple retriever combining code, file and directory indexes."""

    llama = llama or LlamaIndexFacade(cfg, qdrant)
    code_vs = llama.code_vs()
    file_vs = llama.file_vs()
    dir_vs = llama.dir_vs()

    code_ret = VectorStoreIndex.from_vector_store(code_vs).as_retriever(
        similarity_top_k=cfg.llamaindex.retrieval.code_nodes_top_k
    )
    file_ret = VectorStoreIndex.from_vector_store(file_vs).as_retriever(
        similarity_top_k=cfg.llamaindex.retrieval.file_cards_top_k
    )
    dir_ret = VectorStoreIndex.from_vector_store(dir_vs).as_retriever(
        similarity_top_k=cfg.llamaindex.retrieval.dir_cards_top_k
    )

    fusion_mode = cfg.llamaindex.retrieval.fusion_mode
    code_weight = getattr(cfg.llamaindex.retrieval, "code_weight", 1.0)
    file_weight = getattr(cfg.llamaindex.retrieval, "file_weight", 1.0)
    dir_weight = getattr(cfg.llamaindex.retrieval, "dir_weight", 1.0)
    rrf_k = getattr(cfg.llamaindex.retrieval, "rrf_k", 60)

    max_expansions = getattr(cfg.llamaindex.retrieval, "max_expansions", 0)

    class SimpleRetriever:
        def retrieve(self, query: str):
            queries = [query]
            if max_expansions > 0:
                expansions = expand_queries(
                    query, cfg.openai.query_rewriter, max_expansions
                )
                queries.extend(expansions[:max_expansions])

            code_nodes: List[NodeWithScore] = []
            file_nodes: List[NodeWithScore] = []
            dir_nodes: List[NodeWithScore] = []
            for q in queries:
                code_q, file_q, dir_q = rewrite_for_collections(
                    q, cfg.openai.query_rewriter
                )
                code_nodes.extend(code_ret.retrieve(code_q))
                file_nodes.extend(file_ret.retrieve(file_q))
                dir_nodes.extend(dir_ret.retrieve(dir_q))

            return fuse_results(
                code_nodes,
                file_nodes,
                dir_nodes,
                fusion_mode,
                code_weight,
                file_weight,
                dir_weight,
                rrf_k,
            )

    return SimpleRetriever()

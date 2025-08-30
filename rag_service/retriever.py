from __future__ import annotations

from typing import List, Sequence

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from qdrant_client import QdrantClient

from .config import AppConfig
from .llama_facade import LlamaIndexFacade
from .query_rewriter import (
    expand_queries,
    rewrite_for_collections,
    hyde_code_documents,
)


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
            import logging

            logging.getLogger(__name__).warning(
                "sentence-transformers not installed, reranking disabled"
            )
            self._encoder = None
        else:
            self._encoder = CrossEncoder(model)

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

def build_query_engine(
    cfg: AppConfig,
    qdrant: QdrantClient,
    llama: LlamaIndexFacade | None = None,
    collection_prefix: str = "",
):
    """Build a simple retriever combining code, file and directory indexes."""

    llama = llama or LlamaIndexFacade(cfg, qdrant)
    code_vs = llama.code_vs(collection_prefix)
    file_vs = llama.file_vs(collection_prefix)
    dir_vs = (
        llama.dir_vs(collection_prefix) if cfg.features.process_directories else None
    )

    code_ret = VectorStoreIndex.from_vector_store(
        code_vs, embed_model=Settings.code_embed_model
    ).as_retriever(
        similarity_top_k=cfg.llamaindex.retrieval.code_nodes_top_k
    )
    file_ret = VectorStoreIndex.from_vector_store(
        file_vs, embed_model=Settings.text_embed_model
    ).as_retriever(
        similarity_top_k=cfg.llamaindex.retrieval.file_cards_top_k
    )
    if cfg.features.process_directories and dir_vs is not None:
        dir_ret = VectorStoreIndex.from_vector_store(
            dir_vs, embed_model=Settings.text_embed_model
        ).as_retriever(
            similarity_top_k=cfg.llamaindex.retrieval.dir_cards_top_k
        )
    else:
        dir_ret = None

    fusion_mode = cfg.llamaindex.retrieval.fusion_mode
    code_weight = getattr(cfg.llamaindex.retrieval, "code_weight", 1.0)
    file_weight = getattr(cfg.llamaindex.retrieval, "file_weight", 1.0)
    dir_weight = getattr(cfg.llamaindex.retrieval, "dir_weight", 1.0)
    rrf_k = getattr(cfg.llamaindex.retrieval, "rrf_k", 60)
    max_expansions = getattr(cfg.llamaindex.retrieval, "max_expansions", 0)
    use_reranker = getattr(cfg.llamaindex.retrieval, "use_reranker", False)
    reranker = CrossEncoderReranker() if use_reranker else None

    class SimpleRetriever:
        def __init__(self) -> None:
            self._hyde_system_prompt: str = ""

        def set_runtime_options(self, hyde_system_prompt: str = "") -> None:
            """Set per-request options such as HyDE system prompt."""

            self._hyde_system_prompt = hyde_system_prompt or ""

        def retrieve(self, query: str):
            queries = [query]
            queries += expand_queries(
                query, cfg.openai.query_rewriter, max_expansions
            )
            code_nodes: List[NodeWithScore] = []
            file_nodes: List[NodeWithScore] = []
            dir_nodes: List[NodeWithScore] = []
            seen: set[str] = set()

            def _extend_unique(
                nodes: Sequence[NodeWithScore], dest: List[NodeWithScore]
            ) -> None:
                """Append ``nodes`` to ``dest`` if their IDs are unseen."""

                for node in nodes:
                    node_id = node.node.node_id
                    if node_id in seen:
                        continue
                    seen.add(node_id)
                    dest.append(node)

            use_hyde = getattr(cfg.llamaindex.retrieval, "use_hyde_for_code", False)
            hyde_n = max(0, int(getattr(cfg.llamaindex.retrieval, "hyde_docs", 1)))

            for q in queries:
                code_q, file_q, dir_q = rewrite_for_collections(
                    q, cfg.openai.query_rewriter
                )
                if use_hyde and hyde_n > 0:
                    drafts = hyde_code_documents(
                        code_q, cfg.openai.generator, n=hyde_n, system_prompt=self._hyde_system_prompt
                    )
                    for d in drafts:
                        _extend_unique(code_ret.retrieve(d), code_nodes)
                # Also query with the rewritten code query to keep recall high.
                _extend_unique(code_ret.retrieve(code_q), code_nodes)
                _extend_unique(file_ret.retrieve(file_q), file_nodes)
                if cfg.features.process_directories and dir_ret is not None:
                    _extend_unique(dir_ret.retrieve(dir_q), dir_nodes)
            fused = fuse_results(
                code_nodes,
                file_nodes,
                dir_nodes if cfg.features.process_directories else None,
                fusion_mode,
                code_weight,
                file_weight,
                dir_weight,
                rrf_k,
            )
            if reranker is not None:
                fused = reranker.rerank(query, fused)
            return fused

    return SimpleRetriever()

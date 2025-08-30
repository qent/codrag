from __future__ import annotations

from typing import List, Sequence

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from qdrant_client import QdrantClient

from ..config import AppConfig, RetrievalConfig
from ..llama_facade import LlamaIndexFacade
from .utils import CrossEncoderReranker, fuse_results


class SimpleRetriever:
    """Basic retriever combining code, file and directory indexes."""

    def __init__(
        self,
        cfg: AppConfig,
        qdrant: QdrantClient,
        llama: LlamaIndexFacade | None = None,
        collection_prefix: str = "",
    ) -> None:
        self._cfg = cfg
        self._llama = llama or LlamaIndexFacade(cfg, qdrant)
        self._code_vs = self._llama.code_vs(collection_prefix)
        self._file_vs = self._llama.file_vs(collection_prefix)
        self._dir_vs = (
            self._llama.dir_vs(collection_prefix)
            if cfg.features.process_directories
            else None
        )
        self._hyde_system_prompt: str = ""

        retrieval = cfg.llamaindex.retrieval
        if not isinstance(retrieval, RetrievalConfig):
            retrieval = RetrievalConfig(**retrieval.__dict__)
            cfg.llamaindex.retrieval = retrieval
        self._retrieval = retrieval

        code_model = (
            Settings.code_embed_model if hasattr(Settings, "code_embed_model") else None
        )
        self._code_ret = VectorStoreIndex.from_vector_store(
            self._code_vs, embed_model=code_model
        ).as_retriever(
            similarity_top_k=self._retrieval.code_nodes_top_k,
        )
        text_model = (
            Settings.text_embed_model if hasattr(Settings, "text_embed_model") else None
        )
        self._file_ret = VectorStoreIndex.from_vector_store(
            self._file_vs, embed_model=text_model
        ).as_retriever(
            similarity_top_k=self._retrieval.file_cards_top_k,
        )
        if cfg.features.process_directories and self._dir_vs is not None:
            self._dir_ret = VectorStoreIndex.from_vector_store(
                self._dir_vs, embed_model=text_model
            ).as_retriever(
                similarity_top_k=self._retrieval.dir_cards_top_k,
            )
        else:
            self._dir_ret = None

        self._fusion_mode = self._retrieval.fusion_mode
        self._code_weight = self._retrieval.code_weight
        self._file_weight = self._retrieval.file_weight
        self._dir_weight = self._retrieval.dir_weight
        self._rrf_k = self._retrieval.rrf_k
        self._max_expansions = self._retrieval.max_expansions
        use_reranker = self._retrieval.use_reranker
        self._reranker = CrossEncoderReranker() if use_reranker else None

    def set_runtime_options(self, hyde_system_prompt: str = "") -> None:
        """Set per-request options such as HyDE system prompt."""

        self._hyde_system_prompt = hyde_system_prompt or ""

    def _extend_unique(
        self,
        nodes: Sequence[NodeWithScore],
        dest: List[NodeWithScore],
        seen: set[str],
    ) -> None:
        """Append ``nodes`` to ``dest`` if their IDs are unseen."""

        for node in nodes:
            node_id = node.node.node_id
            if node_id in seen:
                continue
            seen.add(node_id)
            dest.append(node)

    def retrieve(self, query: str) -> List[NodeWithScore]:
        """Retrieve relevant nodes for ``query``."""

        queries = [query]
        from .. import retriever as _r  # late import for monkeypatching

        queries += _r.expand_queries(
            query, self._cfg.openai.query_rewriter, self._max_expansions
        )
        code_nodes: List[NodeWithScore] = []
        file_nodes: List[NodeWithScore] = []
        dir_nodes: List[NodeWithScore] = []
        seen: set[str] = set()

        use_hyde = self._retrieval.use_hyde_for_code
        hyde_n = max(0, int(self._retrieval.hyde_docs))

        file_query_meta = _r.extract_file_query_metadata(
            query, self._cfg.openai.query_rewriter
        )

        for q in queries:
            code_q, file_q, dir_q = _r.rewrite_for_collections(
                q, self._cfg.openai.query_rewriter
            )
            if use_hyde and hyde_n > 0:
                drafts = _r.hyde_code_documents(
                    code_q,
                    self._cfg.openai.generator,
                    n=hyde_n,
                    system_prompt=self._hyde_system_prompt,
                )
                for d in drafts:
                    self._extend_unique(self._code_ret.retrieve(d), code_nodes, seen)
            self._extend_unique(self._code_ret.retrieve(code_q), code_nodes, seen)
            self._extend_unique(self._file_ret.retrieve(file_q), file_nodes, seen)
            if (
                self._cfg.features.process_directories
                and self._dir_ret is not None
            ):
                self._extend_unique(self._dir_ret.retrieve(dir_q), dir_nodes, seen)
        file_nodes = _r.boost_file_nodes_by_metadata(file_nodes, file_query_meta)

        fused = fuse_results(
            code_nodes,
            file_nodes,
            dir_nodes if self._cfg.features.process_directories else None,
            self._fusion_mode,
            self._code_weight,
            self._file_weight,
            self._dir_weight,
            self._rrf_k,
        )
        if self._reranker is not None:
            fused = self._reranker.rerank(query, fused)
        return fused


def build_simple_retriever(
    cfg: AppConfig,
    qdrant: QdrantClient,
    llama: LlamaIndexFacade | None = None,
    collection_prefix: str = "",
) -> SimpleRetriever:
    """Build and return :class:`SimpleRetriever`."""

    return SimpleRetriever(cfg, qdrant, llama, collection_prefix)

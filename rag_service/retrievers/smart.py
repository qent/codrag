from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from .neighbor import expand_with_neighbors
from .utils import fuse_results
from ..config import AppConfig
from ..llama_facade import LlamaIndexFacade
from ..query_metadata import (
    FileQueryMetadata,
    boost_file_nodes_by_metadata,
    extract_file_query_metadata,
)


def _unique_extend(
    src: Sequence[NodeWithScore], dest: List[NodeWithScore], seen: set[str]
) -> None:
    """Append nodes from ``src`` to ``dest`` if their IDs are unseen.

    Parameters
    ----------
    src:
        Nodes to consider.
    dest:
        Destination list to extend in-place.
    seen:
        Set of already seen node IDs to prevent duplicates.
    """

    for n in src:
        nid = n.node.node_id
        if nid in seen:
            continue
        seen.add(nid)
        dest.append(n)


def _looks_like_code_query(query: str) -> bool:
    """Heuristic classifier to detect code-oriented queries.

    The heuristic looks for common programming tokens and patterns. It is kept
    intentionally lightweight and language-agnostic.
    """

    q = query.lower()
    code_markers = [
        "def ",
        "class ",
        "func ",
        "return ",
        "::",
        "->",
        "()",
        "{}",
        "[].",
        ".py",
        ".ts",
        ".java",
        ".kt",
        "interface ",
        "implements ",
        " extends ",
    ]
    return any(m in q for m in code_markers)


def _looks_like_file_query(query: str) -> bool:
    """Heuristic classifier to detect file/overview oriented queries."""

    q = query.lower()
    markers = [
        "readme",
        "docs",
        "documentation",
        "file ",
        "папк",
        "директ",
        "описани",
        "overview",
        "how to",
        "config",
        "settings",
    ]
    return any(m in q for m in markers)


class SmartRetriever:
    """A smarter, file-first retriever with filtered code refinement.

    Strategy
    --------
    - Use a file-card first pass (text embeddings) to identify likely files.
    - Extract query metadata and boost file-card scores using soft matches.
    - Constrain code retrieval to the top file candidates and fill remaining
      slots from global code search.
    - Optionally retrieve directory cards when enabled.
    - Fuse results with configurable mode and weights and optionally rerank.
    """

    def __init__(
        self,
        cfg: AppConfig,
        qdrant: QdrantClient,
        llama: LlamaIndexFacade | None = None,
        collection_prefix: str = "",
    ) -> None:
        """Initialize the smart retriever with configuration and stores."""

        self._cfg = cfg
        self._llama = llama or LlamaIndexFacade(cfg, qdrant)
        self._qdrant = qdrant
        self._prefix = collection_prefix
        self._hyde_system_prompt: str = ""

        # Vector stores and retrievers
        self._code_vs = self._llama.code_vs(collection_prefix)
        self._file_vs = self._llama.file_vs(collection_prefix)
        self._dir_vs = (
            self._llama.dir_vs(collection_prefix)
            if cfg.features.process_directories
            else None
        )

        code_model = (
            Settings.code_embed_model if hasattr(Settings, "code_embed_model") else None
        )
        text_model = (
            Settings.text_embed_model if hasattr(Settings, "text_embed_model") else None
        )

        self._code_ret = VectorStoreIndex.from_vector_store(
            self._code_vs, embed_model=code_model
        ).as_retriever(similarity_top_k=cfg.llamaindex.retrieval.code_nodes_top_k)

        self._file_ret = VectorStoreIndex.from_vector_store(
            self._file_vs, embed_model=text_model
        ).as_retriever(similarity_top_k=cfg.llamaindex.retrieval.file_cards_top_k)

        if cfg.features.process_directories and self._dir_vs is not None:
            self._dir_ret = VectorStoreIndex.from_vector_store(
                self._dir_vs, embed_model=text_model
            ).as_retriever(similarity_top_k=cfg.llamaindex.retrieval.dir_cards_top_k)
        else:
            self._dir_ret = None

        # Retrieval and fusion settings
        r = cfg.llamaindex.retrieval
        self._fusion_mode = r.fusion_mode
        self._base_code_weight = getattr(r, "code_weight", 1.0)
        self._base_file_weight = getattr(r, "file_weight", 1.0)
        self._base_dir_weight = getattr(r, "dir_weight", 1.0)
        self._rrf_k = getattr(r, "rrf_k", 60)
        self._max_expansions = getattr(r, "max_expansions", 0)
        self._use_reranker = getattr(r, "use_reranker", False)
        self._use_hyde = getattr(r, "use_hyde_for_code", False)
        self._hyde_docs = max(0, int(getattr(r, "hyde_docs", 1)))
        self._neighbor_decay = float(getattr(r, "neighbor_decay", 0.9))
        self._neighbor_limit = int(getattr(r, "neighbor_limit", 2))
        self._code_per_file_top_k = int(getattr(r, "code_per_file_top_k", 2))

        if self._use_reranker:
            from .. import retriever as _r  # late import for monkeypatching/tests

            self._reranker = _r.CrossEncoderReranker()
        else:
            self._reranker = None

    def set_runtime_options(self, hyde_system_prompt: str = "") -> None:
        """Set per-request options such as HyDE system prompt."""

        self._hyde_system_prompt = hyde_system_prompt or ""

    def _select_top_files(
        self, queries: Iterable[str], qmeta: FileQueryMetadata
    ) -> List[NodeWithScore]:
        """Retrieve and rescore file cards for the queries and return top items."""

        results: List[NodeWithScore] = []
        seen: set[str] = set()
        from .. import retriever as _r  # late import for monkeypatching/tests

        for q in queries:
            _, file_q, _ = _r.rewrite_for_collections(q, self._cfg.openai.query_rewriter)
            _unique_extend(self._file_ret.retrieve(file_q), results, seen)
        results = boost_file_nodes_by_metadata(results, qmeta)
        k = max(0, int(self._cfg.llamaindex.retrieval.file_cards_top_k))
        return results[:k]

    def _retrieve_code_for_files(
        self, queries: Iterable[str], selected_files: Sequence[str]
    ) -> List[NodeWithScore]:
        """Retrieve code nodes, prioritizing those from ``selected_files``.

        When a Qdrant client with search capability is available, performs
        per-file targeted searches constrained by ``file_path`` and respects a
        per-file quota, then backfills from global results. Otherwise, falls
        back to global retrieval and filtering.
        """

        k = max(0, int(self._cfg.llamaindex.retrieval.code_nodes_top_k))
        if k == 0:
            return []

        # Gather candidates from all queries (including HyDE drafts when enabled)
        candidates: List[NodeWithScore] = []
        seen: set[str] = set()

        from .. import retriever as _r  # late import for monkeypatching/tests

        use_targeted = (
            self._qdrant is not None and hasattr(self._qdrant, "search") and selected_files
        )

        if use_targeted:
            # 1) Per-file targeted search using Qdrant filters
            collection = f"{self._prefix}code_nodes"
            per_file_k = max(0, int(self._code_per_file_top_k))

            for q in queries:
                code_q, _, _ = _r.rewrite_for_collections(
                    q, self._cfg.openai.query_rewriter
                )
                # Generate HyDE drafts when enabled
                q_variants: List[str] = [code_q]
                if self._use_hyde and self._hyde_docs > 0:
                    drafts = _r.hyde_code_documents(
                        code_q,
                        self._cfg.openai.generator,
                        n=self._hyde_docs,
                        system_prompt=self._hyde_system_prompt,
                    )
                    q_variants.extend(drafts)

                for qv in q_variants:
                    try:
                        vec = Settings.code_embed_model.get_query_embedding(qv)
                    except Exception:
                        # Fallback to text embedding if query embedding unavailable
                        vec = Settings.code_embed_model.get_text_embedding(qv)
                    for fp in selected_files:
                        if len(candidates) >= k:
                            break
                        try:
                            flt = Filter(
                                must=[
                                    FieldCondition(
                                        key="file_path", match=MatchValue(value=str(fp))
                                    )
                                ]
                            )
                            results = self._qdrant.search(
                                collection_name=collection,
                                query_vector=vec,
                                limit=per_file_k,
                                with_payload=True,
                                with_vectors=False,
                                query_filter=flt,
                            )
                        except Exception:
                            results = []
                        for p in results:
                            payload = (p.payload or {}).copy()
                            text = str(payload.pop("text", ""))
                            nid = str(payload.get("node_id") or "")
                            node = TextNode(id_=nid or None, text=text, metadata=payload)
                            _unique_extend([NodeWithScore(node=node, score=float(p.score or 0.0))], candidates, seen)

            # 2) Backfill with global retrieval using LlamaIndex retriever if needed
            if len(candidates) < k:
                for q in queries:
                    code_q, _, _ = _r.rewrite_for_collections(
                        q, self._cfg.openai.query_rewriter
                    )
                    if self._use_hyde and self._hyde_docs > 0:
                        drafts = _r.hyde_code_documents(
                            code_q,
                            self._cfg.openai.generator,
                            n=self._hyde_docs,
                            system_prompt=self._hyde_system_prompt,
                        )
                        for d in drafts:
                            _unique_extend(self._code_ret.retrieve(d), candidates, seen)
                    _unique_extend(self._code_ret.retrieve(code_q), candidates, seen)

            return candidates[:k]

        # Fallback: global retrieval and filter by selected files, then backfill
        for q in queries:
            code_q, _, _ = _r.rewrite_for_collections(q, self._cfg.openai.query_rewriter)
            if self._use_hyde and self._hyde_docs > 0:
                drafts = _r.hyde_code_documents(
                    code_q,
                    self._cfg.openai.generator,
                    n=self._hyde_docs,
                    system_prompt=self._hyde_system_prompt,
                )
                for d in drafts:
                    _unique_extend(self._code_ret.retrieve(d), candidates, seen)
            _unique_extend(self._code_ret.retrieve(code_q), candidates, seen)

        sel = {s.lower() for s in selected_files}
        preferred: List[NodeWithScore] = []
        others: List[NodeWithScore] = []
        for n in candidates:
            fp = str(n.node.metadata.get("file_path", "")).lower()
            if fp and fp in sel:
                preferred.append(n)
            else:
                others.append(n)

        out: List[NodeWithScore] = preferred[:k]
        if len(out) < k:
            out.extend(others[: k - len(out)])
        return out

    def _dir_results(self, queries: Iterable[str]) -> List[NodeWithScore]:
        """Retrieve directory card results if enabled and available."""

        if not (self._cfg.features.process_directories and self._dir_ret):
            return []
        results: List[NodeWithScore] = []
        seen: set[str] = set()
        from .. import retriever as _r  # late import for monkeypatching/tests
        for q in queries:
            _, _, dir_q = _r.rewrite_for_collections(q, self._cfg.openai.query_rewriter)
            _unique_extend(self._dir_ret.retrieve(dir_q), results, seen)
        k = max(0, int(self._cfg.llamaindex.retrieval.dir_cards_top_k))
        return results[:k]

    def _dynamic_weights(self, query: str) -> Tuple[float, float, float]:
        """Return adjusted fusion weights based on query intent heuristics."""

        code_w = self._base_code_weight
        file_w = self._base_file_weight
        dir_w = self._base_dir_weight

        codeish = _looks_like_code_query(query)
        fileish = _looks_like_file_query(query)

        if codeish and not fileish:
            code_w *= 1.5
        elif fileish and not codeish:
            file_w *= 1.5
        else:
            # Mixed/ambiguous: keep base weights
            pass

        return code_w, file_w, dir_w

    def retrieve(self, query: str) -> List[NodeWithScore]:
        """Retrieve relevant nodes for ``query`` using a two-stage pipeline."""

        # 1) Generate query variants
        queries = [query]
        from .. import retriever as _r  # late import for monkeypatching/tests
        queries += _r.expand_queries(
            query, self._cfg.openai.query_rewriter, self._max_expansions
        )

        # 2) File-card pass with metadata boosting
        qmeta = extract_file_query_metadata(query, self._cfg.openai.query_rewriter)
        file_nodes = self._select_top_files(queries, qmeta)
        selected_files = [n.node.metadata.get("file_path", "") for n in file_nodes]
        selected_files = [f for f in selected_files if f]

        # 3) Code retrieval, focusing on selected files and backfilling
        code_nodes = self._retrieve_code_for_files(queries, selected_files)

        # 4) Optional directory retrieval
        dir_nodes = self._dir_results(queries)

        # 5) Fuse and optionally rerank
        code_w, file_w, dir_w = self._dynamic_weights(query)
        fused = fuse_results(
            code_nodes,
            file_nodes,
            dir_nodes if self._cfg.features.process_directories else None,
            mode=self._fusion_mode,
            code_weight=code_w,
            file_weight=file_w,
            dir_weight=dir_w,
            rrf_k=self._rrf_k,
        )
        # 6) Neighbor expansion for code nodes using prev_id/next_id
        fused = expand_with_neighbors(
            self._qdrant, self._prefix, fused, self._neighbor_decay, self._neighbor_limit
        )
        if self._reranker is not None:
            fused = self._reranker.rerank(query, fused)
        return fused


def build_smart_retriever(
    cfg: AppConfig,
    qdrant: QdrantClient,
    llama: LlamaIndexFacade | None = None,
    collection_prefix: str = "",
) -> SmartRetriever:
    """Build and return :class:`SmartRetriever`."""

    return SmartRetriever(cfg, qdrant, llama, collection_prefix)

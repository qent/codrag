from rag_service import query_rewriter as qr


def test_rewrite_for_collections_no_llm_returns_originals(monkeypatch) -> None:
    """When LLM is unavailable, rewrites fall back to original query."""

    monkeypatch.setattr(qr, "_build_llm", lambda cfg: None)
    code_q, file_q, dir_q = qr.rewrite_for_collections("search", cfg=None)
    assert (code_q, file_q, dir_q) == ("search", "search", "search")


def test_expand_queries_no_llm_or_zero_max(monkeypatch) -> None:
    """Expansion returns empty list when max_expansions <= 0 or LLM missing."""

    monkeypatch.setattr(qr, "_build_llm", lambda cfg: None)
    assert qr.expand_queries("q", cfg=None, max_expansions=0) == []
    assert qr.expand_queries("q", cfg=None, max_expansions=3) == []


from types import SimpleNamespace

from rag_service.query_rewriter import expand_queries, rewrite_for_collections


class DummyLLM:
    """LLM stub returning deterministic single-field outputs across calls."""

    def __init__(self):
        self._vals = iter(["c", "f", "d"])  # three sequential calls -> code, file, dir

    def with_structured_output(self, model):  # pragma: no cover - simple stub
        def _call(_):
            try:
                val = next(self._vals)
            except StopIteration:
                val = "x"
            return SimpleNamespace(query=val)

        return _call


def test_rewrite_for_collections(monkeypatch) -> None:
    """The function should return queries for all collections."""

    monkeypatch.setattr(
        "rag_service.query_rewriter._build_llm", lambda cfg: DummyLLM()
    )
    code_q, file_q, dir_q = rewrite_for_collections("query", cfg=object())
    assert (code_q, file_q, dir_q) == ("c", "f", "d")


def test_expand_queries(monkeypatch) -> None:
    """The function should return alternative queries."""

    class DummyLLM2:
        def with_structured_output(self, model):  # pragma: no cover - simple stub
            def _call(_):
                return SimpleNamespace(queries=["a", "b", "c"])

            return _call

    monkeypatch.setattr(
        "rag_service.query_rewriter._build_llm", lambda cfg: DummyLLM2()
    )
    res = expand_queries("query", cfg=object(), max_expansions=2)
    assert res == ["a", "b"]

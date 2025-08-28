from types import SimpleNamespace

from rag_service.query_rewriter import rewrite_for_collections


class DummyLLM:
    """LLM stub returning fixed structured output."""

    def with_structured_output(self, model):  # pragma: no cover - simple stub
        def _call(_):
            return SimpleNamespace(code="c", file="f", dir="d")

        return _call


def test_rewrite_for_collections(monkeypatch) -> None:
    """The function should return queries for all collections."""

    monkeypatch.setattr(
        "rag_service.query_rewriter._build_llm", lambda cfg: DummyLLM()
    )
    code_q, file_q, dir_q = rewrite_for_collections("query", cfg=object())
    assert (code_q, file_q, dir_q) == ("c", "f", "d")


from pathlib import Path

from rag_service import main


def test_query_endpoint_respects_top_k(monkeypatch, tmp_path: Path) -> None:
    """Query endpoint limits items by top_k when returning text descriptions."""

    class Node:
        def __init__(self, text: str, metadata: dict) -> None:
            self._text = text
            self.metadata = metadata

        def get_content(self) -> str:
            return self._text

    class Result:
        def __init__(self, text: str, score: float, idx: int) -> None:
            self.node = Node(text, {"type": "code_node", "file_path": f"f{idx}.py", "lang": "python"})
            self.score = score

    # Return 5 results, expect only top 3 forwarded when flag enabled
    results = [Result(f"t{i}", score=1.0 - i * 0.1, idx=i) for i in range(5)]

    class Retriever:
        def retrieve(self, q: str):  # type: ignore[override]
            return results

    def fake_build_query_engine(cfg, qdrant, llama, collection_prefix):  # type: ignore[unused-argument]
        return Retriever()

    monkeypatch.setattr(main, "build_query_engine", fake_build_query_engine)
    main.CONFIG = type("Cfg", (), {"features": type("F", (), {})(), "qdrant": object()})()
    main.QDRANT = object()
    main.LLAMA = object()

    req = main.QueryRequest(root_path=str(tmp_path), q="q", top_k=3, return_text_description=True)
    resp = main.query_endpoint(req)
    assert len(resp["items"]) == 3
    assert [it["text"] for it in resp["items"]] == ["t0", "t1", "t2"]


from pathlib import Path

from pytest import MonkeyPatch

from rag_service import main


def test_file_path_prefix_in_results(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Query endpoint applies the configured file path prefix to results."""

    code = "print('hi')\n"
    fp = tmp_path / "test.py"
    fp.write_text(code)

    class Node:
        def __init__(self, text: str, metadata: dict) -> None:
            self._text = text
            self.metadata = metadata

        def get_content(self) -> str:
            return self._text

    class Result:
        def __init__(self, text: str, metadata: dict, score: float) -> None:
            self.node = Node(text, metadata)
            self.score = score

    results = [
        Result("card text", {"type": "file_card", "file_path": str(fp), "lang": "python"}, 1.0),
        Result("code text", {"type": "code_node", "file_path": str(fp), "lang": "python"}, 0.5),
    ]

    class Retriever:
        def retrieve(self, q: str):  # type: ignore[override]
            return results

    def fake_build_query_engine(cfg, qdrant, llama, collection_prefix):  # type: ignore[unused-argument]
        return Retriever()

    class ScrollPoint:
        def __init__(self, text: str, metadata: dict) -> None:
            self.payload = {"text": text, **metadata}

    class FakeQdrant:
        def scroll(self, collection, limit, scroll_filter):  # type: ignore[unused-argument]
            metadata = {"type": "code_node", "file_path": str(fp), "lang": "python"}
            return [ScrollPoint("scroll code", metadata)], None

    monkeypatch.setattr(main, "build_query_engine", fake_build_query_engine)
    main.CONFIG = type(
        "Cfg",
        (),
        {
            "qdrant": object(),
            "features": type("F", (), {"file_path_prefix": "/prefix/"})(),
        },
    )()
    main.QDRANT = FakeQdrant()
    main.LLAMA = object()

    resp = main.query_endpoint(main.QueryRequest(root_path=str(tmp_path), q="test"))
    assert all(
        item["metadata"].get("file_path", "").startswith("/prefix/") for item in resp["items"]
    )

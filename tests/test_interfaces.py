from pathlib import Path

from rag_service.interface_extractor import extract_public_interfaces
from rag_service import main


def test_extract_public_interfaces(tmp_path):
    src = (
        "class Example:\n"
        "    def public(self):\n        pass\n"
        "    def _private(self):\n        pass\n\n"
        "def standalone():\n    pass\n"
        "def _helper():\n    pass\n"
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text(src)
    interfaces = extract_public_interfaces(file_path, "python")
    assert "class Example:" in interfaces
    assert "def public(self):" in interfaces
    assert "def standalone():" in interfaces
    assert all("_private" not in s for s in interfaces)
    assert all("_helper" not in s for s in interfaces)


def test_query_endpoint_interfaces(tmp_path, monkeypatch):
    code = "def foo():\n    pass\n"
    fp = tmp_path / "test.py"
    fp.write_text(code)

    class Node:
        def __init__(self, metadata):
            self.metadata = metadata

        def get_content(self):
            return code

    class Result:
        def __init__(self, metadata):
            self.node = Node(metadata)
            self.score = 1.0

    class Retriever:
        def retrieve(self, q):
            return [Result({"file_path": str(fp), "lang": "python"})]

    def fake_build_query_engine(cfg, qdrant, llama):
        return Retriever()

    monkeypatch.setattr(main, "build_query_engine", fake_build_query_engine)
    main.CONFIG = object()
    main.QDRANT = object()
    main.LLAMA = object()

    resp = main.query_endpoint(main.QueryRequest(q="test", interfaces=True))
    assert resp["items"][0]["interfaces"] == ["def foo():"]

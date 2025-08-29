from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from .config import AppConfig
from .indexer import index_path
from .llama_facade import LlamaIndexFacade
from .retriever import build_query_engine
from .openai_utils import close_llamaindex_clients
from .collection_utils import collection_prefix_from_path

app = FastAPI()
CONFIG: AppConfig | None = None
QDRANT: QdrantClient | None = None
LLAMA: LlamaIndexFacade | None = None
QE: dict[str, Any] = {}
QE_CONFIG_ID: int | None = None


class IndexRequest(BaseModel):
    root_path: str
    clean: bool = False


class QueryRequest(BaseModel):
    root_path: str
    q: str
    top_k: int = 10
    interfaces: bool = False
    return_text_description: bool = False


def _with_file_path_prefix(metadata: dict) -> dict:
    """Return a copy of ``metadata`` with configured file path prefix applied."""

    assert CONFIG
    features = getattr(CONFIG, "features", None)
    prefix = getattr(features, "file_path_prefix", "")
    file_path = metadata.get("file_path")
    if prefix and file_path:
        return {**metadata, "file_path": prefix + file_path}
    return dict(metadata)


def _build_query_engine(prefix: str) -> None:
    """(Re)build the query engine for ``prefix`` using current config."""

    global QE_CONFIG_ID
    assert CONFIG and QDRANT and LLAMA
    QE[prefix] = build_query_engine(CONFIG, QDRANT, LLAMA, prefix)
    QE_CONFIG_ID = id(CONFIG)


def _load_config(config_path: Path) -> None:
    """Load configuration and initialize shared clients."""

    global CONFIG, QDRANT, LLAMA, QE_CONFIG_ID
    CONFIG = AppConfig.load(config_path)
    QDRANT = QdrantClient(url=CONFIG.qdrant.url)
    LLAMA = LlamaIndexFacade(CONFIG, QDRANT)
    QE.clear()
    QE_CONFIG_ID = id(CONFIG)


@app.on_event("startup")
def _startup() -> None:
    """Initialize configuration and shared clients."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args, _ = parser.parse_known_args()
    _load_config(Path(args.config))


@app.on_event("shutdown")
def _shutdown():
    """Close OpenAI clients.

    When ``LlamaIndexFacade`` is used outside FastAPI, invoke this handler
    directly to release HTTP resources.
    """

    close_llamaindex_clients()


@app.post("/v1/index")
def index_endpoint(req: IndexRequest):
    """Index the repository located at ``root_path``."""

    assert CONFIG and QDRANT and LLAMA
    start = time.time()
    stats = index_path(Path(req.root_path), CONFIG, QDRANT, LLAMA)
    took = int((time.time() - start) * 1000)
    return {"status": "ok", "indexed": stats.__dict__, "took_ms": took}


@app.post("/v1/query")
def query_endpoint(req: QueryRequest):
    """Query indexed data and return matching items."""

    assert CONFIG and QDRANT and LLAMA
    prefix = collection_prefix_from_path(req.root_path)
    if QE_CONFIG_ID != id(CONFIG) or prefix not in QE:
        _build_query_engine(prefix)
    result = QE[prefix].retrieve(req.q)

    if req.interfaces:
        from .interface_extractor import extract_public_interfaces

        items = []
        seen: set[str] = set()
        for r in result:
            file_path = r.node.metadata.get("file_path")
            lang = r.node.metadata.get("lang")
            if not file_path or file_path in seen:
                continue
            seen.add(file_path)
            interfaces = extract_public_interfaces(Path(file_path), lang)
            metadata = _with_file_path_prefix({"file_path": file_path, "lang": lang})
            items.append(
                {
                    "type": "file_interface",
                    "score": r.score,
                    "interfaces": interfaces,
                    "metadata": metadata,
                }
            )
    else:
        if req.return_text_description:
            items = [
                {
                    "type": r.node.metadata.get("type", "code_node"),
                    "score": r.score,
                    "text": r.node.get_content(),
                    "metadata": _with_file_path_prefix(r.node.metadata),
                }
                for r in result
            ]
        else:
            items = []
            seen_files: set[str] = set()
            for r in result:
                node_type = r.node.metadata.get("type", "code_node")
                if node_type == "file_card":
                    file_path = r.node.metadata.get("file_path")
                    if not file_path or file_path in seen_files:
                        continue
                    seen_files.add(file_path)
                    items.extend(_fetch_code_nodes(file_path, r.score, prefix))
                else:
                    items.append(
                        {
                            "type": node_type,
                            "score": r.score,
                            "text": r.node.get_content(),
                            "metadata": _with_file_path_prefix(r.node.metadata),
                        }
                    )
    return {"status": "ok", "items": items}


def _fetch_code_nodes(file_path: str, score: float, prefix: str) -> list[dict]:
    """Return code node items for ``file_path``."""

    assert CONFIG and QDRANT
    flt = Filter(
        must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
    )
    points, _ = QDRANT.scroll(f"{prefix}code_nodes", limit=100, scroll_filter=flt)
    items: list[dict] = []
    for p in points:
        payload = p.payload or {}
        items.append(
            {
                "type": payload.get("type", "code_node"),
                "score": score,
                "text": payload.get("text", ""),
                "metadata": _with_file_path_prefix(payload),
            }
        )
    return items


@app.get("/v1/config")
def get_config():
    """Return sanitized runtime configuration."""

    assert CONFIG
    return CONFIG.sanitized_dict()


@app.get("/v1/collections")
def collections(root_path: str):
    """Return summary information about Qdrant collections for ``root_path``."""

    assert QDRANT and CONFIG
    prefix = collection_prefix_from_path(root_path)
    names = [f"{prefix}code_nodes", f"{prefix}file_cards"]
    if CONFIG.features.process_directories:
        names.append(f"{prefix}dir_cards")
    info = {
        name: QDRANT.get_collection(name).dict()
        if QDRANT.collection_exists(name)
        else None
        for name in names
    }
    return info


@app.get("/v1/healthz")
def healthz():
    """Health-check endpoint."""

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    _load_config(Path(args.config))
    uvicorn.run(app, host="0.0.0.0", port=CONFIG.http.port)

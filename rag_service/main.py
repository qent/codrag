from __future__ import annotations

import argparse
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient

from .config import AppConfig
from .indexer import index_path
from .llama_facade import LlamaIndexFacade
from .retriever import build_query_engine

app = FastAPI()
CONFIG: AppConfig | None = None
QDRANT: QdrantClient | None = None
LLAMA: LlamaIndexFacade | None = None


class IndexRequest(BaseModel):
    root_path: str
    clean: bool = False


class QueryRequest(BaseModel):
    q: str
    top_k: int = 10
    interfaces: bool = False


@app.on_event("startup")
def _startup():
    """Initialize configuration and shared clients."""

    global CONFIG, QDRANT, LLAMA
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args, _ = parser.parse_known_args()
    CONFIG = AppConfig.load(Path(args.config))
    QDRANT = QdrantClient(url=CONFIG.qdrant.url)
    LLAMA = LlamaIndexFacade(CONFIG, QDRANT)


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
    qe = build_query_engine(CONFIG, QDRANT, LLAMA)
    result = qe.retrieve(req.q)

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
            items.append(
                {
                    "type": "file_interface",
                    "score": r.score,
                    "interfaces": interfaces,
                    "metadata": {"file_path": file_path, "lang": lang},
                }
            )
    else:
        items = [
            {
                "type": r.node.metadata.get("type", "code_node"),
                "score": r.score,
                "text": r.node.get_content(),
                "metadata": r.node.metadata,
            }
            for r in result
        ]
    return {"status": "ok", "items": items}


@app.get("/v1/config")
def get_config():
    """Return sanitized runtime configuration."""

    assert CONFIG
    return CONFIG.sanitized_dict()


@app.get("/v1/collections")
def collections():
    """Return summary information about Qdrant collections."""

    assert QDRANT and CONFIG
    names = [
        f"{CONFIG.qdrant.collection_prefix}code_nodes",
        f"{CONFIG.qdrant.collection_prefix}file_cards",
    ]
    info = {name: QDRANT.get_collection(name).dict() if QDRANT.collection_exists(name) else None for name in names}
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
    CONFIG = AppConfig.load(Path(args.config))
    QDRANT = QdrantClient(url=CONFIG.qdrant.url)
    LLAMA = LlamaIndexFacade(CONFIG, QDRANT)
    uvicorn.run(app, host="0.0.0.0", port=CONFIG.http.port)

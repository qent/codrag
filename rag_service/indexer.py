from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient

from .config import AppConfig
from .llama_facade import LlamaIndexFacade


@dataclass
class IndexStats:
    """Statistics produced during indexing."""

    files_total: int = 0
    files_processed: int = 0
    files_skipped_cache: int = 0
    code_nodes_upserted: int = 0
    file_cards_upserted: int = 0
    dir_cards_upserted: int = 0


class PathIndexer:
    """Index files under a root directory."""

    def __init__(self, cfg: AppConfig, qdrant: QdrantClient, llama: LlamaIndexFacade) -> None:
        self.cfg = cfg
        self.llama = llama
        self.code_vs = llama.code_vs()
        self.file_vs = llama.file_vs()

    def index_path(self, root: Path) -> IndexStats:
        """Index all files under ``root`` and return statistics."""

        stats = IndexStats()
        files = self._scan_files(root)
        stats.files_total = len(files)
        for file_path in files:
            self._process_file(file_path, stats)
        return stats

    def _scan_files(self, root: Path) -> List[Path]:
        """Return a list of files eligible for indexing."""

        files: List[Path] = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                p = Path(dirpath) / fname
                if not any(fname.endswith(ext) for ext in self.cfg.indexing.include_extensions):
                    continue
                rel = p.relative_to(root)
                if any(fnmatch.fnmatch(str(rel), pattern) for pattern in self.cfg.indexing.blacklist):
                    continue
                if p.stat().st_size > self.cfg.indexing.max_file_size_mb * 1024 * 1024:
                    continue
                files.append(p)
        return files

    def _read_file(self, file_path: Path) -> Tuple[str, str]:
        """Return file text and its SHA256 hash."""

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text, file_hash

    def _create_nodes(self, text: str, file_path: Path, file_hash: str):
        """Create code nodes for a file."""

        lang = self.cfg.ast.languages.get(file_path.suffix, "")
        doc = Document(
            text=text,
            metadata={
                "file_path": str(file_path),
                "file_hash": file_hash,
                "dir_path": str(file_path.parent),
                "lang": lang,
            },
        )
        splitter = SentenceSplitter(
            chunk_size=self.cfg.ast.max_chars,
            chunk_overlap=self.cfg.ast.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents([doc])
        for n in nodes:
            n.metadata.update({"file_hash": file_hash, "dir_path": str(file_path.parent), "lang": lang})
        return nodes

    def _upsert_code_nodes(self, nodes) -> int:
        """Write code nodes to the vector store."""

        VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=self.code_vs))
        return len(nodes)

    def _generate_file_card(self, file_path: Path) -> str:
        """Generate a textual description for the file."""

        prompt = Path(self.cfg.prompts.file_card_md).read_text() + f"\nFile: {file_path.name}\n"
        try:
            return self.llama.llm().complete(prompt).text
        except Exception:
            return f"Description for {file_path.name}"

    def _upsert_file_card(self, file_path: Path, file_hash: str, card_text: str) -> None:
        """Write the file card to the vector store."""

        doc = Document(
            text=card_text,
            metadata={
                "type": "file_card",
                "file_path": str(file_path),
                "file_hash": file_hash,
                "dir_path": str(file_path.parent),
                "lang": self.cfg.ast.languages.get(file_path.suffix, ""),
            },
        )
        VectorStoreIndex([doc], storage_context=StorageContext.from_defaults(vector_store=self.file_vs))

    def _process_file(self, file_path: Path, stats: IndexStats) -> None:
        """Index a single file and update ``stats``."""

        stats.files_processed += 1
        text, file_hash = self._read_file(file_path)
        nodes = self._create_nodes(text, file_path, file_hash)
        stats.code_nodes_upserted += self._upsert_code_nodes(nodes)
        card_text = self._generate_file_card(file_path)
        self._upsert_file_card(file_path, file_hash, card_text)
        stats.file_cards_upserted += 1


def index_path(
    root: Path, cfg: AppConfig, qdrant: QdrantClient, llama: LlamaIndexFacade | None = None
) -> IndexStats:
    """Index a path and return statistics."""

    llama = llama or LlamaIndexFacade(cfg, qdrant)
    return PathIndexer(cfg, qdrant, llama).index_path(root)

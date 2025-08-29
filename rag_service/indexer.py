from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import CodeSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

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

    def __init__(
        self,
        cfg: AppConfig,
        qdrant: QdrantClient,
        llama: LlamaIndexFacade,
        collection_prefix: str,
        repo_prompt: str = "",
    ) -> None:
        """Initialize the indexer with configuration, clients, and a repository prompt."""

        self.cfg = cfg
        self.qdrant = qdrant
        self.llama = llama
        self.collection_prefix = collection_prefix
        self.repo_prompt = repo_prompt
        self.code_vs = llama.code_vs(collection_prefix)
        self.file_vs = llama.file_vs(collection_prefix)
        self.dir_vs = (
            llama.dir_vs(collection_prefix) if cfg.features.process_directories else None
        )

    def index_path(self, root: Path) -> IndexStats:
        """Index all files under ``root`` and return statistics."""

        stats = IndexStats()
        dir_items: Dict[Path, List[str]] = {}
        files = sorted(self._scan_files(root))
        stats.files_total = len(files)
        for file_path in files:
            text, file_hash = self._read_file(file_path)
            if self._is_cached(file_path, file_hash):
                stats.files_skipped_cache += 1
                continue
            card_text = self._process_file(file_path, stats, text, file_hash)
            if self.cfg.features.process_directories:
                dir_items.setdefault(file_path.parent, []).append(card_text)
        if self.cfg.features.process_directories:
            self._generate_dir_cards(root, dir_items, stats)
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

    def _is_cached(self, file_path: Path, file_hash: str) -> bool:
        """Return ``True`` if ``file_path`` with ``file_hash`` exists in Qdrant."""

        collection = f"{self.collection_prefix}file_cards"
        flt = Filter(
            must=[
                FieldCondition(key="file_path", match=MatchValue(value=str(file_path))),
                FieldCondition(key="file_hash", match=MatchValue(value=file_hash)),
            ]
        )
        try:
            points, _ = self.qdrant.scroll(collection, limit=1, scroll_filter=flt)
            return bool(points)
        except Exception:
            return False

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
        splitter = CodeSplitter(
            language=lang,
            chunk_lines=self.cfg.ast.chunk_lines,
            chunk_lines_overlap=self.cfg.ast.chunk_overlap,
            max_chars=self.cfg.ast.max_chars,
        )
        nodes = splitter.get_nodes_from_documents([doc])
        for i, node in enumerate(nodes):
            prev_id = nodes[i - 1].node_id if i > 0 else None
            next_id = nodes[i + 1].node_id if i < len(nodes) - 1 else None
            content = node.get_content() or ""
            node.metadata.update(
                {
                    "file_path": str(file_path),
                    "file_hash": file_hash,
                    "dir_path": str(file_path.parent),
                    "lang": lang,
                    "prev_id": prev_id,
                    "next_id": next_id,
                    "symbols_in_chunk": len(content),
                }
            )
        return nodes

    def _upsert_code_nodes(self, nodes) -> int:
        """Write code nodes to the vector store."""

        VectorStoreIndex(
            nodes,
            storage_context=StorageContext.from_defaults(vector_store=self.code_vs),
            embed_model=Settings.code_embed_model,
        )
        return len(nodes)

    class _FileCard(BaseModel):
        """Structured data describing a source file."""

        summary: str = Field(description="Concise multi-sentence overview of the file")
        key_points: List[str] = Field(
            default_factory=list, description="High-signal bullet points"
        )
        embedding_text: str = Field(description="Dense paragraph for embeddings")
        keywords: List[str] = Field(
            default_factory=list, description="Searchable keyword tags"
        )

    def _generate_file_card(self, file_path: Path, file_text: str) -> _FileCard:
        """Generate a structured description for ``file_path``."""

        prompt = Path(self.cfg.prompts.file_card_md).read_text()
        tmpl = ChatPromptTemplate.from_template(prompt)
        lang = self.cfg.ast.languages.get(file_path.suffix, "")
        chain = tmpl | self.llama.llm().with_structured_output(self._FileCard)
        data = chain.invoke(
            {
                "file_path": str(file_path),
                "file_content": file_text,
                "language_hint": lang,
                "repo_context": self.repo_prompt,
            }
        )
        return data

    def _upsert_file_card(self, file_path: Path, file_hash: str, card: _FileCard) -> None:
        """Write ``card`` for ``file_path`` to the vector store."""

        doc = Document(
            text=card.embedding_text,
            metadata={
                "type": "file_card",
                "file_path": str(file_path),
                "file_hash": file_hash,
                "dir_path": str(file_path.parent),
                "lang": self.cfg.ast.languages.get(file_path.suffix, ""),
                "summary": card.summary,
                "key_points": card.key_points,
                "keywords": card.keywords,
            },
        )
        VectorStoreIndex(
            [doc],
            storage_context=StorageContext.from_defaults(vector_store=self.file_vs),
            embed_model=Settings.text_embed_model,
        )

    def _process_file(
        self, file_path: Path, stats: IndexStats, text: str, file_hash: str
    ) -> str:
        """Index a single file, update ``stats`` and return summary text."""

        stats.files_processed += 1
        nodes = self._create_nodes(text, file_path, file_hash)
        stats.code_nodes_upserted += self._upsert_code_nodes(nodes)
        card = self._generate_file_card(file_path, text)
        self._upsert_file_card(file_path, file_hash, card)
        stats.file_cards_upserted += 1
        return card.summary

    def _generate_dir_card(self, dir_path: Path, items: List[str]) -> str:
        """Generate a textual description for a directory."""

        prompt = Path(self.cfg.prompts.dir_card_md).read_text()
        if self.repo_prompt:
            prompt += f"\nRepository description:\n{self.repo_prompt}\n"
        prompt += f"\nDirectory: {dir_path.name}\n"
        prompt += "\n".join(items) + "\n"
        return self.llama.llm().complete(prompt).text

    def _upsert_dir_card(self, dir_path: Path, card_text: str) -> None:
        """Write a directory card to the vector store."""

        if not self.dir_vs:
            return
        doc = Document(
            text=card_text,
            metadata={
                "type": "dir_card",
                "dir_path": str(dir_path),
                "parent_dir": str(dir_path.parent),
            },
        )
        VectorStoreIndex(
            [doc],
            storage_context=StorageContext.from_defaults(vector_store=self.dir_vs),
            embed_model=Settings.text_embed_model,
        )

    def _generate_dir_cards(
        self, root: Path, dir_items: Dict[Path, List[str]], stats: IndexStats
    ) -> None:
        """Generate and upsert directory cards bottom-up."""
        if not self.cfg.features.process_directories or not self.dir_vs:
            return
        processed: set[Path] = set()
        while True:
            pending = [p for p in dir_items.keys() if p not in processed]
            if not pending:
                break
            for dir_path in sorted(pending, key=lambda p: len(p.parts), reverse=True):
                card_text = self._generate_dir_card(dir_path, dir_items.get(dir_path, []))
                self._upsert_dir_card(dir_path, card_text)
                stats.dir_cards_upserted += 1
                processed.add(dir_path)
                if dir_path == root:
                    continue
                parent = dir_path.parent
                dir_items.setdefault(parent, []).append(card_text)


def index_path(
    root: Path,
    cfg: AppConfig,
    qdrant: QdrantClient,
    llama: LlamaIndexFacade | None = None,
    repo_prompt: str = "",
) -> IndexStats:
    """Index a path and return statistics."""

    from .collection_utils import collection_prefix_from_path

    llama = llama or LlamaIndexFacade(cfg, qdrant)
    prefix = collection_prefix_from_path(root)
    indexer = PathIndexer(cfg, qdrant, llama, prefix, repo_prompt)
    return indexer.index_path(root)

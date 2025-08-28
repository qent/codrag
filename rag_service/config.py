from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IndexingConfig(BaseModel):
    include_extensions: List[str] = Field(default_factory=list)
    blacklist: List[str] = Field(default_factory=list)
    follow_symlinks: bool = False
    max_file_size_mb: int = 2


class ASTConfig(BaseModel):
    """Settings for code chunking."""

    chunk_lines: int = 60
    chunk_overlap: int = 10
    max_chars: int = 2000
    languages: Dict[str, str] = Field(default_factory=dict)


class OpenAIClientConfig(BaseModel):
    base_url: str
    model: str
    api_key: str
    verify_ssl: bool = True
    timeout_sec: int = 60
    retries: int = 3
    rate_limit_rps: int = 1


class OpenAIConfig(BaseModel):
    embeddings: OpenAIClientConfig
    generator: OpenAIClientConfig
    query_rewriter: OpenAIClientConfig


class PromptsConfig(BaseModel):
    file_card_md: str
    dir_card_md: str
    generate_dir_cards: bool = False


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection_prefix: str = ""
    distance: str = "Cosine"
    prefer_grpc: bool = False


class RetrievalConfig(BaseModel):
    code_nodes_top_k: int = 6
    file_cards_top_k: int = 4
    dir_cards_top_k: int = 2
    fusion_mode: str = "relative_score"
    code_weight: float = 1.0
    file_weight: float = 1.0
    rrf_k: int = 60


class LlamaIndexConfig(BaseModel):
    use: bool = True
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


class HTTPConfig(BaseModel):
    port: int = 8080
    max_context_items: int = 12
    max_total_chars: int = 20000


class FeaturesConfig(BaseModel):
    cache_llm_cards: bool = True
    skip_on_llm_error: bool = True
    concurrency: int = 6


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json: bool = True


class AppConfig(BaseModel):
    version: int = 1
    indexing: IndexingConfig
    ast: ASTConfig
    openai: OpenAIConfig
    prompts: PromptsConfig
    qdrant: QdrantConfig
    llamaindex: LlamaIndexConfig
    http: HTTPConfig = Field(default_factory=HTTPConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        """Load configuration from ``path``."""

        import json

        data = json.loads(Path(path).read_text())
        return cls(**data)

    def sanitized_dict(self) -> Dict:
        """Return dict representation without sensitive keys."""

        data = self.dict()
        data["openai"]["embeddings"].pop("api_key", None)
        data["openai"]["generator"].pop("api_key", None)
        data["openai"]["query_rewriter"].pop("api_key", None)
        return data

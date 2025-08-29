from __future__ import annotations

import logging
import os
from httpx import Client
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
import numpy as np

from .config import AppConfig, OpenAIClientConfig

logger = logging.getLogger(__name__)

EMBEDDINGS_CLIENT: Client | None = None
GENERATOR_CLIENT: Client | None = None


class NormalizedEmbedding(OpenAIEmbedding):
    """Embedding model wrapper that L2‑normalizes output vectors."""

    def _normalize(self, embedding: Embedding) -> Embedding:
        """Return the L2‑normalized version of ``embedding``."""

        arr = np.array(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0:
            return embedding
        return (arr / norm).tolist()

    def get_text_embedding(self, text: str) -> Embedding:
        """Embed ``text`` and return a normalized vector."""

        embedding = super().get_text_embedding(text)
        return self._normalize(embedding)

    async def aget_text_embedding(self, text: str) -> Embedding:
        """Asynchronously embed ``text`` and return a normalized vector."""

        embedding = await super().aget_text_embedding(text)
        return self._normalize(embedding)

    def get_query_embedding(self, query: str) -> Embedding:
        """Embed ``query`` and return a normalized vector."""

        embedding = super().get_query_embedding(query)
        return self._normalize(embedding)

    async def aget_query_embedding(self, query: str) -> Embedding:
        """Asynchronously embed ``query`` and return a normalized vector."""

        embedding = await super().aget_query_embedding(query)
        return self._normalize(embedding)


def init_llamaindex_clients(cfg: AppConfig) -> None:
    """Initialize LlamaIndex global settings with OpenAI‑like clients.

    The created HTTP clients are stored globally for graceful shutdown.
    """

    global EMBEDDINGS_CLIENT, GENERATOR_CLIENT

    def get_key(c: OpenAIClientConfig) -> str:
        """Resolve API key, supporting ``env:VAR`` indirection."""

        api_key = c.api_key
        if api_key.startswith("env:"):
            api_key = os.environ.get(api_key.split(":", 1)[1], "")
        return api_key

    def get_http_client(c: OpenAIClientConfig) -> Client:
        """Create an HTTP client for the provided OpenAI‑like settings."""

        http_client = Client(verify=c.verify_ssl, timeout=c.timeout_sec)
        if not c.verify_ssl:
            logger.warning("SSL verification disabled for OpenAI-like client %s", c.model)
        return http_client

    EMBEDDINGS_CLIENT = get_http_client(cfg.openai.embeddings)
    Settings.embed_model = NormalizedEmbedding(
        model=cfg.openai.embeddings.model,
        api_base=cfg.openai.embeddings.base_url,
        api_key=get_key(cfg.openai.embeddings),
        http_client=EMBEDDINGS_CLIENT,
    )

    GENERATOR_CLIENT = get_http_client(cfg.openai.generator)
    Settings.llm = OpenAILike(
        model=cfg.openai.generator.model,
        api_base=cfg.openai.generator.base_url,
        api_key=get_key(cfg.openai.generator),
        http_client=GENERATOR_CLIENT,
        is_chat_model=True,
    )

    if os.getenv("LLM_LOGGING"):
        Settings.callback_manager = CallbackManager([LlamaDebugHandler()])
        logging.getLogger("llama_index").setLevel(logging.DEBUG)


def close_llamaindex_clients() -> None:
    """Close global HTTP clients used by LlamaIndex.

    When ``LlamaIndexFacade`` is used outside FastAPI, this function should
    be called manually to release HTTP resources.
    """

    global EMBEDDINGS_CLIENT, GENERATOR_CLIENT
    for client in (EMBEDDINGS_CLIENT, GENERATOR_CLIENT):
        if client is not None:
            client.close()
    EMBEDDINGS_CLIENT = None
    GENERATOR_CLIENT = None

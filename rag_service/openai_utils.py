from __future__ import annotations

import logging
import os

import numpy as np
from httpx import Client
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from .config import AppConfig, OpenAIClientConfig
from .langchain_logging import LangChainLogHandler

logger = logging.getLogger(__name__)

CODE_EMBEDDINGS_CLIENT: Client | None = None
TEXT_EMBEDDINGS_CLIENT: Client | None = None
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
    """Initialize LlamaIndex global settings with OpenAI-like clients.

    Separate embedding models are created for code and text processing. The
    created HTTP clients are stored globally for graceful shutdown.
    """

    global CODE_EMBEDDINGS_CLIENT, TEXT_EMBEDDINGS_CLIENT, GENERATOR_CLIENT

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

    CODE_EMBEDDINGS_CLIENT = get_http_client(cfg.openai.code_embeddings)
    Settings.code_embed_model = NormalizedEmbedding(
        model=cfg.openai.code_embeddings.model,
        api_base=cfg.openai.code_embeddings.base_url,
        api_key=get_key(cfg.openai.code_embeddings),
        http_client=CODE_EMBEDDINGS_CLIENT,
    )

    TEXT_EMBEDDINGS_CLIENT = get_http_client(cfg.openai.text_embeddings)
    Settings.text_embed_model = NormalizedEmbedding(
        model=cfg.openai.text_embeddings.model,
        api_base=cfg.openai.text_embeddings.base_url,
        api_key=get_key(cfg.openai.text_embeddings),
        http_client=TEXT_EMBEDDINGS_CLIENT,
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

    global CODE_EMBEDDINGS_CLIENT, TEXT_EMBEDDINGS_CLIENT, GENERATOR_CLIENT
    for client in (
        CODE_EMBEDDINGS_CLIENT,
        TEXT_EMBEDDINGS_CLIENT,
        GENERATOR_CLIENT,
    ):
        if client is not None:
            client.close()
    CODE_EMBEDDINGS_CLIENT = None
    TEXT_EMBEDDINGS_CLIENT = None
    GENERATOR_CLIENT = None


def build_langchain_llm(cfg: OpenAIClientConfig | None) -> BaseChatModel | None:
    """Create a LangChain LLM from ``cfg`` or return ``None`` if misconfigured.

    Resolves ``env:VAR`` API keys, respects SSL verification and timeout, and
    enables optional logging via ``LLM_LOGGING`` environment variable.
    """

    if cfg is None:
        return None
    api_key = cfg.api_key
    if api_key.startswith("env:"):
        api_key = os.environ.get(api_key.split(":", 1)[1], "")
    if not api_key:
        return None
    http_client = Client(verify=cfg.verify_ssl, timeout=cfg.timeout_sec)
    callbacks = [LangChainLogHandler()] if os.getenv("LLM_LOGGING") else None
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=api_key,
        max_retries=cfg.retries,
        http_client=http_client,
        callbacks=callbacks,
    )

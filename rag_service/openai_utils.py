from __future__ import annotations

import logging
import os
from httpx import Client
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from .config import AppConfig

logger = logging.getLogger(__name__)


def init_llamaindex_clients(cfg: AppConfig) -> None:
    """Initialize LlamaIndex global settings with OpenAI clients."""

    def get_key(c):
        api_key = c.api_key
        if api_key.startswith("env:"):
            api_key = os.environ.get(api_key.split(":", 1)[1], "")
        return api_key

    def get_http_client(c):
        http_client = Client(verify=c.verify_ssl, timeout=c.timeout_sec)
        if not c.verify_ssl:
            logger.warning("SSL verification disabled for OpenAI client %s", c.model)
        return http_client

    Settings.embed_model = OpenAIEmbedding(
        model=cfg.openai.embeddings.model,
        api_base=cfg.openai.embeddings.base_url,
        api_key=get_key(cfg.openai.embeddings),
        http_client=get_http_client(cfg.openai.embeddings)
    )
    Settings.llm = OpenAI(
        model=cfg.openai.generator.model,
        api_base=cfg.openai.generator.base_url,
        api_key=get_key(cfg.openai.generator),
        http_client=get_http_client(cfg.openai.generator)
    )

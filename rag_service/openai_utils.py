from __future__ import annotations

import logging
import os
from httpx import Client
from openai import OpenAI as OpenAIClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from .config import AppConfig

logger = logging.getLogger(__name__)


def init_llamaindex_clients(cfg: AppConfig) -> None:
    """Initialize LlamaIndex global settings with OpenAI clients."""

    def build_client(c):
        """Create an OpenAI client from configuration."""

        api_key = c.api_key
        if api_key.startswith("env:"):
            api_key = os.environ.get(api_key.split(":", 1)[1], "")
        http_client = Client(verify=c.verify_ssl, timeout=c.timeout_sec)
        if not c.verify_ssl:
            logger.warning("SSL verification disabled for OpenAI client %s", c.model)
        return OpenAIClient(base_url=c.base_url, api_key=api_key, http_client=http_client)

    emb_client = build_client(cfg.openai.embeddings)
    gen_client = build_client(cfg.openai.generator)

    Settings.embed_model = OpenAIEmbedding(model=cfg.openai.embeddings.model, client=emb_client)
    Settings.llm = OpenAI(model=cfg.openai.generator.model, client=gen_client)

from __future__ import annotations

"""LangChain callback handler for logging LLM interactions."""

import logging
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class LangChainLogHandler(BaseCallbackHandler):
    """Log prompts and responses for LangChain LLM calls."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Log prompts sent to the LLM."""
        for prompt in prompts:
            self.logger.info("LLM request: %s", prompt)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log responses returned by the LLM."""
        for generations in response.generations:
            for gen in generations:
                self.logger.info("LLM response: %s", gen.text)

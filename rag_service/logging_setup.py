from __future__ import annotations

"""Logging initialization utilities.

This module configures Python logging according to the application configuration
(`AppConfig.logging`). It supports plain text or JSON logs and ensures that our
module loggers (including LangChain/LlamaIndex hooks) emit at the desired level.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - fallback when optional dep missing
    orjson = None  # type: ignore

from .config import AppConfig


def _level_from_string(level: str) -> int:
    """Return a logging level constant from a case-insensitive name.

    Defaults to ``logging.INFO`` when the provided name is invalid.
    """

    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    return mapping.get(level.upper(), logging.INFO)


class _JSONFormatter(logging.Formatter):
    """Minimal JSON formatter using ``orjson`` when available.

    The payload includes time (UTC ISO 8601), level, logger name and message.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - custom format
        data: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        if orjson is not None:
            return orjson.dumps(data).decode("utf-8")
        import json

        return json.dumps(data, ensure_ascii=False)


def setup_logging(cfg: AppConfig) -> None:
    """Initialize logging based on ``cfg.logging``.

    - Sets the root logger level.
    - Installs a single stdout handler with plain or JSON formatting.
    - Leaves Uvicorn loggers in place but aligns their level to match.
    """

    level = _level_from_string(cfg.logging.level)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing non-uvicorn handlers to avoid duplicate messages
    new_handlers: list[logging.Handler] = []
    for h in list(root.handlers):
        try:
            root.removeHandler(h)
        except Exception:
            pass
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)

    if cfg.logging.json:
        handler.setFormatter(_JSONFormatter())
    else:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        datefmt = "%Y-%m-%dT%H:%M:%S%z"
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(handler)

    # Align common third-party loggers with our level; keep their handlers
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "llama_index"):
        logging.getLogger(name).setLevel(level)

    # Honor environment override if present (e.g., PYTHONLOGLEVEL)
    env_level = os.getenv("PYTHONLOGLEVEL")
    if env_level:
        root.setLevel(_level_from_string(env_level))


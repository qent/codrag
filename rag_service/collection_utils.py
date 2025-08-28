from __future__ import annotations

from pathlib import Path
import re


def collection_prefix_from_path(path: str | Path) -> str:
    """Return a Qdrant collection prefix derived from ``path``.

    The prefix is built by joining sanitized path parts with underscores and
    appending a trailing underscore. For example, ``/home/user`` becomes
    ``home_user_``.
    """
    p = Path(path)
    parts = [re.sub(r"[^0-9A-Za-z]+", "_", part).strip("_") for part in p.parts]
    parts = [part for part in parts if part]
    return ("_".join(parts) + "_").lower() if parts else ""

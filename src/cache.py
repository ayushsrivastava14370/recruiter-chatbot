"""Tiny disk cache for LLM responses and embeddings.

Why: the Gemini free tier is rate-limited, and caching makes demos fast,
repeatable and cheap. Identical prompts return the identical stored answer.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import threading
from pathlib import Path
from typing import Any

from .config import CACHE_DIR

_lock = threading.Lock()


def key_for(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, name: str):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path: Path = CACHE_DIR / f"{name}.pkl"
        self._data: dict[str, Any] = {}
        if self.path.exists():
            try:
                with open(self.path, "rb") as fh:
                    self._data = pickle.load(fh)
            except Exception:
                self._data = {}

    def get(self, key: str) -> Any | None:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with _lock:
            self._data[key] = value
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(self._data, fh)
            tmp.replace(self.path)

    def clear(self) -> None:
        with _lock:
            self._data = {}
            if self.path.exists():
                self.path.unlink()

"""Record real model outputs, and replay them later.

Every live run (MLX or transformers) is recorded to data/cache/outputs.json, keyed by
the exact prompt and the content hash of each image. The replay backend plays those
recorded outputs back, with simulated streaming, so the demo still works if the
laptop is struggling on stage. The UI labels replayed output as such.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Iterator, Sequence

from .base import Backend


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def cache_key(kind: str, prompt: str, images: Sequence[Path] = ()) -> str:
    h = hashlib.sha256()
    h.update(kind.encode())
    h.update(b"\0")
    h.update(prompt.encode())
    for img in images:
        h.update(b"\0")
        h.update(_file_hash(img).encode())
    return h.hexdigest()


class OutputCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def get(self, key: str) -> str | None:
        entry = self._data.get(key)
        return entry["output"] if entry else None

    def put(self, key: str, output: str, meta: dict) -> None:
        with self._lock:
            self._data[key] = {"output": output, **meta}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))


class RecordingBackend(Backend):
    """Wraps a live backend and records everything it says."""

    def __init__(self, inner: Backend, cache: OutputCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.label = inner.label
        self.is_live_model = inner.is_live_model

    def stream_medgemma(self, prompt: str, images: Sequence[Path], max_tokens: int) -> Iterator[str]:
        parts: list[str] = []
        for chunk in self.inner.stream_medgemma(prompt, images, max_tokens):
            parts.append(chunk)
            yield chunk
        meta = {"kind": "medgemma", "backend": self.inner.name, "images": [Path(p).name for p in images]}
        self.cache.put(cache_key("medgemma", prompt, images), "".join(parts).strip(), meta)

    def txgemma(self, prompt: str, max_tokens: int = 8) -> str:
        out = self.inner.txgemma(prompt, max_tokens)
        self.cache.put(cache_key("txgemma", prompt), out, {"kind": "txgemma", "backend": self.inner.name})
        return out


class CacheMiss(RuntimeError):
    pass


class ReplayBackend(Backend):
    name = "replay"
    label = "Replay: recorded live MedGemma/TxGemma output"
    is_live_model = False

    def __init__(self, cache: OutputCache, words_per_second: float = 40.0):
        self.cache = cache
        self.delay = 1.0 / words_per_second if words_per_second else 0.0

    def _lookup(self, key: str) -> str:
        out = self.cache.get(key)
        if out is None:
            raise CacheMiss(
                "No recorded output for this exact input. Replay only works for inputs that were "
                "run live before (same prompt, same images). Run it live once with the mlx backend."
            )
        return out

    def stream_medgemma(self, prompt: str, images: Sequence[Path], max_tokens: int) -> Iterator[str]:
        text = self._lookup(cache_key("medgemma", prompt, images))
        for i, word in enumerate(text.split(" ")):
            if self.delay:
                time.sleep(self.delay)
            yield word if i == 0 else " " + word

    def txgemma(self, prompt: str, max_tokens: int = 8) -> str:
        return self._lookup(cache_key("txgemma", prompt))

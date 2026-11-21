"""Backend factory."""

from __future__ import annotations

import os

from .. import config
from .base import Backend
from .replay_backend import OutputCache, RecordingBackend, ReplayBackend

BACKENDS = ("mlx", "hf", "replay", "mock")


def get_backend(name: str | None = None, record: bool | None = None) -> Backend:
    name = (name or config.default_backend()).lower()
    if record is None:
        record = os.getenv("SECONDLOOK_RECORD", "1") != "0"
    cache = OutputCache(config.CACHE_DIR / "outputs.json")

    if name == "mlx":
        from .mlx_backend import MLXBackend

        backend: Backend = MLXBackend(config.MEDGEMMA_MLX_PATH, config.TXGEMMA_MLX_PATH)
    elif name == "hf":
        from .hf_backend import HFBackend

        backend = HFBackend(config.MEDGEMMA_HF_ID, config.TXGEMMA_HF_ID)
    elif name == "replay":
        return ReplayBackend(cache)
    elif name == "mock":
        from .mock_backend import MockBackend

        return MockBackend(delay=float(os.getenv("SECONDLOOK_MOCK_DELAY", "0.01")))
    else:
        raise ValueError(f"Unknown backend {name!r}. Choose one of: {', '.join(BACKENDS)}")

    return RecordingBackend(backend, cache) if record else backend


__all__ = ["Backend", "get_backend", "BACKENDS"]

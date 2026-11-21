"""Apple Silicon backend using MLX (mlx-vlm for MedGemma, mlx-lm for TxGemma).

Models load lazily on first use so the UI starts instantly. On an 8GB M1 the two
4-bit models together use about 4.5GB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from .base import Backend, clean_chunk


class MLXBackend(Backend):
    name = "mlx"
    label = "Live: MLX on-device"
    is_live_model = True

    def __init__(self, medgemma_path: Path, txgemma_path: Path):
        self.medgemma_path = Path(medgemma_path)
        self.txgemma_path = Path(txgemma_path)
        self._med = None  # (model, processor, config)
        self._tx = None  # (model, tokenizer)

    def _load_medgemma(self):
        if self._med is None:
            if not self.medgemma_path.exists():
                raise FileNotFoundError(
                    f"MedGemma MLX weights not found at {self.medgemma_path}. Run scripts/setup_mac.sh first."
                )
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            model, processor = load(str(self.medgemma_path))
            config = load_config(str(self.medgemma_path))
            self._med = (model, processor, config)
        return self._med

    def _load_txgemma(self):
        if self._tx is None:
            if not self.txgemma_path.exists():
                raise FileNotFoundError(
                    f"TxGemma MLX weights not found at {self.txgemma_path}. Run scripts/setup_mac.sh first."
                )
            from mlx_lm import load

            self._tx = load(str(self.txgemma_path))
        return self._tx

    def stream_medgemma(self, prompt: str, images: Sequence[Path], max_tokens: int) -> Iterator[str]:
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor, config = self._load_medgemma()
        image_args = [str(p) for p in images] or None
        formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
        for chunk in stream_generate(model, processor, formatted, image_args, max_tokens=max_tokens):
            text = clean_chunk(getattr(chunk, "text", chunk) or "")
            if text:
                yield text

    def txgemma(self, prompt: str, max_tokens: int = 8) -> str:
        from mlx_lm import generate

        model, tokenizer = self._load_txgemma()
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        return clean_chunk(getattr(out, "text", out)).strip()

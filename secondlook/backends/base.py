"""Backend interface: one MedGemma (vision + text) model and one TxGemma (text) model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Sequence

STOP_STRINGS = ("<end_of_turn>", "<eos>")


def clean_chunk(text: str) -> str:
    for stop in STOP_STRINGS:
        text = text.replace(stop, "")
    return text


class Backend(ABC):
    """A model runtime.

    `label` is shown in the UI so it is always obvious whether output is from a live
    model, a replay of recorded live output, or a mock used for development.
    """

    name: str = "base"
    label: str = "base"
    is_live_model: bool = False

    @abstractmethod
    def stream_medgemma(self, prompt: str, images: Sequence[Path], max_tokens: int) -> Iterator[str]:
        """Stream MedGemma text for a prompt plus zero or more images."""

    def medgemma(self, prompt: str, images: Sequence[Path], max_tokens: int) -> str:
        return "".join(self.stream_medgemma(prompt, images, max_tokens)).strip()

    @abstractmethod
    def txgemma(self, prompt: str, max_tokens: int = 8) -> str:
        """Run TxGemma (predict variant) on a raw TDC-format prompt."""

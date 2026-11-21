"""Hugging Face transformers backend (Linux / NVIDIA GPU, or CPU if you are patient)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator, Sequence

from .base import Backend, clean_chunk


class HFBackend(Backend):
    name = "hf"
    label = "Live: transformers"
    is_live_model = True

    def __init__(self, medgemma_id: str, txgemma_id: str):
        self.medgemma_id = medgemma_id
        self.txgemma_id = txgemma_id
        self._med = None
        self._tx = None

    @staticmethod
    def _torch():
        import torch

        return torch

    def _load_medgemma(self):
        if self._med is None:
            torch = self._torch()
            from transformers import AutoModelForImageTextToText, AutoProcessor

            processor = AutoProcessor.from_pretrained(self.medgemma_id)
            model = AutoModelForImageTextToText.from_pretrained(
                self.medgemma_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self._med = (model, processor)
        return self._med

    def _load_txgemma(self):
        if self._tx is None:
            torch = self._torch()
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.txgemma_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.txgemma_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self._tx = (model, tokenizer)
        return self._tx

    def stream_medgemma(self, prompt: str, images: Sequence[Path], max_tokens: int) -> Iterator[str]:
        torch = self._torch()
        from PIL import Image
        from transformers import TextIteratorStreamer

        model, processor = self._load_medgemma()
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=False, streamer=streamer)
        thread = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        for text in streamer:
            text = clean_chunk(text)
            if text:
                yield text
        thread.join()

    def txgemma(self, prompt: str, max_tokens: int = 8) -> str:
        model, tokenizer = self._load_txgemma()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return clean_chunk(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()

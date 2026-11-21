"""Image normalization: RGB, capped size, deterministic output path (keeps replay keys stable)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from . import config


def prepare_image(path: str | Path, max_side: int = config.MAX_IMAGE_SIDE) -> Path:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    out = config.PREP_DIR / f"{digest}_{max_side}.png"
    if out.exists():
        return out
    config.PREP_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("I;16", "I"):
            # 16-bit grayscale X-rays: rescale to 8-bit before converting
            img = img.point(lambda v: v * (255.0 / 65535.0)).convert("L")
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        img.save(out, format="PNG")
    return out

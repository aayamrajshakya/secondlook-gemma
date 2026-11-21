"""Central configuration. Everything is overridable with environment variables."""

from __future__ import annotations

import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CASES_DIR = DATA_DIR / "cases"
IMAGES_DIR = DATA_DIR / "images"
PREP_DIR = DATA_DIR / ".prep"
CACHE_DIR = DATA_DIR / "cache"  # recorded real model outputs, used by the replay backend
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"

# Hugging Face model ids (both are gated: accept the HAI-DEF terms on their HF pages first)
MEDGEMMA_HF_ID = os.getenv("SECONDLOOK_MEDGEMMA_HF_ID", "google/medgemma-1.5-4b-it")
TXGEMMA_HF_ID = os.getenv("SECONDLOOK_TXGEMMA_HF_ID", "google/txgemma-2b-predict")

# Local 4-bit MLX conversions produced by scripts/setup_mac.sh
MEDGEMMA_MLX_PATH = Path(os.getenv("SECONDLOOK_MEDGEMMA_MLX", MODELS_DIR / "medgemma-1.5-4b-it-4bit"))
TXGEMMA_MLX_PATH = Path(os.getenv("SECONDLOOK_TXGEMMA_MLX", MODELS_DIR / "txgemma-2b-predict-4bit"))

# TxGemma ships its official TDC prompt templates in the model repo
TDC_PROMPTS_PATH = Path(os.getenv("SECONDLOOK_TDC_PROMPTS", MODELS_DIR / "tdc_prompts.json"))

# Generation limits (kept modest so an 8GB M1 Air stays responsive)
MAX_TOKENS_READ = int(os.getenv("SECONDLOOK_MAX_TOKENS_READ", "400"))
MAX_TOKENS_EXTRACT = int(os.getenv("SECONDLOOK_MAX_TOKENS_EXTRACT", "250"))
MAX_TOKENS_RECONCILE = int(os.getenv("SECONDLOOK_MAX_TOKENS_RECONCILE", "450"))

# Longest image side fed to the model (MedGemma's processor resizes to 896 anyway)
MAX_IMAGE_SIDE = int(os.getenv("SECONDLOOK_MAX_IMAGE_SIDE", "1024"))


def default_backend() -> str:
    """Pick a backend: explicit env var wins, else MLX on Apple Silicon, else transformers."""
    explicit = os.getenv("SECONDLOOK_BACKEND")
    if explicit:
        return explicit.lower()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "hf"

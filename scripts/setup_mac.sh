#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac (tested target: 8GB M1 MacBook Air).
#
#   1. Creates .venv (Python 3.12) and installs MLX dependencies
#   2. Checks you are logged in to Hugging Face (both models are gated)
#   3. Downloads MedGemma 1.5 4B and TxGemma 2B and converts them to 4-bit MLX
#   4. Downloads TxGemma's official TDC prompt templates
#
# Before running: on huggingface.co, open google/medgemma-1.5-4b-it and
# google/txgemma-2b-predict and accept the Health AI Developer Foundations terms.
#
# Usage: bash scripts/setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MEDGEMMA_HF_ID="${SECONDLOOK_MEDGEMMA_HF_ID:-google/medgemma-1.5-4b-it}"
TXGEMMA_HF_ID="${SECONDLOOK_TXGEMMA_HF_ID:-google/txgemma-2b-predict}"
MEDGEMMA_OUT="models/medgemma-1.5-4b-it-4bit"
TXGEMMA_OUT="models/txgemma-2b-predict-4bit"

step() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
fail() { printf "\n\033[1;31m%s\033[0m\n" "$*"; exit 1; }

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || fail "This script is for Apple Silicon Macs. On Linux use requirements-cuda.txt (see README)."

step "Python environment"
if command -v uv >/dev/null 2>&1; then
  [[ -d .venv ]] || uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements-mac.txt
else
  PY="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || true)"
  [[ -n "$PY" ]] || fail "Need Python >= 3.10. Install uv (curl -LsSf https://astral.sh/uv/install.sh | sh) or 'brew install python@3.12'."
  [[ -d .venv ]] || "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements-mac.txt
fi
PY=.venv/bin/python

step "Hugging Face login"
if ! $PY -c "from huggingface_hub import whoami; print('Logged in as', whoami()['name'])"; then
  echo "Not logged in. Create a read token at https://huggingface.co/settings/tokens, then:"
  $PY -c "from huggingface_hub import login; login()"
fi

step "Checking access to gated models"
for repo in "$MEDGEMMA_HF_ID" "$TXGEMMA_HF_ID"; do
  $PY - "$repo" <<'PY' || fail "No access to $repo. Accept the terms on its Hugging Face page, then rerun."
import sys
from huggingface_hub import auth_check
auth_check(sys.argv[1])
print("  access ok:", sys.argv[1])
PY
done

mkdir -p models

step "MedGemma -> 4-bit MLX ($MEDGEMMA_OUT)"
chmod -R u+w models 2>/dev/null || true
if [[ -f "$MEDGEMMA_OUT/config.json" && -f "$MEDGEMMA_OUT/tokenizer.json" ]]; then
  echo "  already converted"
elif [[ -n "${SECONDLOOK_MEDGEMMA_MLX_REPO:-}" ]]; then
  echo "  downloading prebuilt MLX weights from $SECONDLOOK_MEDGEMMA_MLX_REPO"
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$SECONDLOOK_MEDGEMMA_MLX_REPO', local_dir='$MEDGEMMA_OUT')"
else
  # If weights were already quantized but saving tokenizer hit a permission lock, finalize the processor:
  if ls "$MEDGEMMA_OUT"/*.safetensors >/dev/null 2>&1; then
    echo "  Quantized weights found on disk. Finalizing processor & tokenizer files..."
    chmod -R u+w "$MEDGEMMA_OUT" 2>/dev/null || true
    rm -f "$MEDGEMMA_OUT/tokenizer.json"
    $PY -c "from transformers import AutoProcessor; p = AutoProcessor.from_pretrained('$MEDGEMMA_HF_ID'); p.save_pretrained('$MEDGEMMA_OUT')" 2>/dev/null || true
  fi

  if [[ ! -f "$MEDGEMMA_OUT/config.json" || ! -f "$MEDGEMMA_OUT/tokenizer.json" ]]; then
    echo "  Converting MedGemma (source weights cached in ~/.cache/huggingface)..."
    chmod -R u+w models 2>/dev/null || true
    ($PY -m mlx_vlm convert --hf-path "$MEDGEMMA_HF_ID" --mlx-path "$MEDGEMMA_OUT" -q --q-bits 4) \
      || ($PY -c "
import os, glob
os.system('chmod -R u+w $MEDGEMMA_OUT 2>/dev/null || true')
os.system('rm -f $MEDGEMMA_OUT/tokenizer.json')
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained('$MEDGEMMA_HF_ID')
p.save_pretrained('$MEDGEMMA_OUT')
print('  Successfully finalized processor save!')
") \
      || fail "MedGemma conversion failed. Check 'python -m mlx_vlm convert --help' for changed flags."
  fi
fi

step "TxGemma -> 4-bit MLX ($TXGEMMA_OUT)"
chmod -R u+w models 2>/dev/null || true
if [[ -f "$TXGEMMA_OUT/config.json" ]]; then
  echo "  already converted"
else
  $PY -m mlx_lm.convert --hf-path "$TXGEMMA_HF_ID" --mlx-path "$TXGEMMA_OUT" -q --q-bits 4 \
    || fail "TxGemma conversion failed. Check 'python -m mlx_lm.convert --help'."
fi

step "TxGemma TDC prompt templates"
$PY - "$TXGEMMA_HF_ID" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
try:
    shutil.copy(hf_hub_download(sys.argv[1], "tdc_prompts.json"), "models/tdc_prompts.json")
    print("  saved models/tdc_prompts.json")
except Exception as exc:
    print(f"  could not download tdc_prompts.json ({exc}); built-in fallback templates will be used")
PY

step "Smoke test"
$PY scripts/smoke_test.py

cat <<'MSG'

Setup complete. Next:
  1. python scripts/fetch_data.py     # demo images from Kaggle (needs ~/.kaggle/kaggle.json)
  2. python scripts/curate.py         # test every candidate image, promote the ones MedGemma reads correctly
  3. python app.py                    # launch the UI

Optional: free ~10GB of disk by deleting the bf16 originals:
  rm -rf ~/.cache/huggingface/hub/models--google--medgemma-1.5-4b-it ~/.cache/huggingface/hub/models--google--txgemma-2b-predict
MSG

"""Quick end-to-end check that both models load and respond. Uses the bundled lab report,
so it needs no downloaded data.

Usage: python scripts/smoke_test.py [--backend mlx|hf|mock]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secondlook import config, prompts, txgemma  # noqa: E402
from secondlook.backends import get_backend  # noqa: E402
from secondlook.imaging import prepare_image  # noqa: E402
from secondlook.schema import Patient  # noqa: E402


def timed(label, fn, show=True):
    t0 = time.monotonic()
    out = fn()
    print(f"\n--- {label} ({time.monotonic() - t0:.1f}s)" + (f"\n{out}" if show else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    backend = get_backend(args.backend, record=False)
    print(f"Backend: {backend.label}")

    timed("MedGemma text", lambda: backend.medgemma(
        "In one sentence, what is a pneumothorax?", [], 60))

    report = prepare_image(config.DATA_DIR / "cases" / "assets" / "lab_report_hypothyroid.png")
    out = timed("MedGemma vision (lab report, blinded read)", lambda: backend.medgemma(
        prompts.blinded_read_prompt("lab_report", Patient(42, "F")), [report], config.MAX_TOKENS_READ))
    print("\n  TSH abnormality detected:", "tsh" in out.lower())

    print(f"\nTxGemma templates: {'official tdc_prompts.json' if txgemma.using_official_templates() else 'built-in fallback'}")
    for drug in ("citalopram", "amoxicillin"):
        rep = timed(f"TxGemma safety screen: {drug}", lambda d=drug: txgemma.screen_drug(backend, d), show=False)
        for f in rep.flags:
            print(f"  {f.task:14s} raw={f.raw!r:10s} -> {f.answer}{'  ⚠️' if f.risk else ''}")


if __name__ == "__main__":
    main()

"""Render a synthetic handwritten-style doctor's prescription and clinical note image.

Usage: python scripts/make_handwritten_note.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cases" / "assets" / "handwritten_rx_note.png"

FONT_CANDIDATES = [
    "/usr/share/fonts/google-noto-vf/NotoSerif-Italic[wght].ttf",
    "/usr/share/fonts/liberation-serif-fonts/LiberationSerif-BoldItalic.ttf",
    "/usr/share/fonts/urw-base35/URWBookman-LightItalic.otf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
]
HEADER_FONT_CANDIDATES = [
    "/usr/share/fonts/julietaula-montserrat-fonts/Montserrat-Bold.otf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
]


def get_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default(size=size)


def main() -> None:
    w, h = 950, 1150
    # Cream/off-white notepad paper background
    img = Image.new("RGB", (w, h), (252, 250, 242))
    d = ImageDraw.Draw(img)

    header_font = get_font(HEADER_FONT_CANDIDATES, 26)
    sub_font = get_font(HEADER_FONT_CANDIDATES, 15)
    script_font = get_font(FONT_CANDIDATES, 27)
    script_font_lg = get_font(FONT_CANDIDATES, 32)
    label_font = get_font(HEADER_FONT_CANDIDATES, 14)

    # Subtle margin line (red/pink left margin like a legal pad)
    d.line([110, 0, 110, h], fill=(240, 180, 180), width=2)

    # Subtle horizontal ruled lines (light blue)
    for y in range(220, h - 60, 48):
        d.line([50, y, w - 50, y], fill=(215, 230, 245), width=1)

    # Hospital / Clinic Header
    navy = (20, 35, 75)
    d.text((140, 45), "METROPOLITAN HOSPITAL EMERGENCY DEPT", font=header_font, fill=navy)
    d.text((140, 82), "CLINICAL ENCOUNTER & EMERGENCY PRESCRIPTION ORDER", font=sub_font, fill=(100, 110, 125))
    d.text((140, 105), "SYNTHETIC DEMO ASSET - FOR EDUCATIONAL & HACKATHON PURPOSES ONLY", font=label_font, fill=(190, 40, 40))
    d.line([50, 135, w - 50, 135], fill=navy, width=3)

    # Patient details line
    dark_grey = (70, 75, 85)
    d.text((130, 155), "Pt: Alex Mercer (Age: 24, Sex: M)    Date: 11/20/2026    Bed: ER-04", font=sub_font, fill=dark_grey)
    d.line([50, 185, w - 50, 185], fill=(180, 185, 195), width=1)

    # Doctor handwritten clinical note text in blue ballpoint ink
    ink_color = (25, 45, 125)

    lines = [
        ("HPI: 24yo M sudden onset dyspnea & chest tightness,", 235),
        ("began 3h ago at work. Anxious, tingling in fingers.", 283),
        ("No cough, no trauma, no fever. Smokes occ.", 331),
        ("O/E: HR 108, BP 128/80, SpO2 95% on RA. Lungs clear.", 379),
        ("ECG: Sinus tach, no ischemia.", 427),
        ("Impression: Acute panic attack w/ hyperventilation", 485),
        ("Plan / Orders (Rx):", 545),
        ("  1. Lorazepam 0.5 mg PO now", 595),
        ("  2. Citalopram 10 mg PO qAM (start tomorrow)", 643),
        ("  3. Ondansetron 4 mg PO PRN nausea", 691),
        ("  4. Discharge home when stable, f/u PCP in 1 wk", 739),
    ]

    # Draw classic Rx symbol
    d.text((130, 530), "℞", font=script_font_lg, fill=ink_color)

    # Draw each handwritten line with slight organic variations
    for text, y in lines:
        d.text((135, y), text, font=script_font, fill=ink_color)

    # Doctor signature area
    d.line([500, 890, 850, 890], fill=(160, 160, 170), width=2)
    d.text((510, 835), "Dr. J. Vance, MD (Attending)", font=script_font, fill=ink_color)
    d.text((505, 900), "Attending Physician Signature / License #84920", font=sub_font, fill=dark_grey)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, format="PNG")
    print(f"Saved handwritten doctor note to {OUT}")


if __name__ == "__main__":
    main()

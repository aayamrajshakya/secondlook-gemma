"""Render the synthetic lab report image used by the "Depression" case.

The report is clearly labelled synthetic and uses a fictional lab and patient.
Usage: python scripts/make_lab_report.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cases" / "assets" / "lab_report_hypothyroid.png"

FONT_CANDIDATES = [
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]
BOLD_CANDIDATES = [
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for candidate in BOLD_CANDIDATES if bold else FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


# (test, result, unit, reference range, flag)
PANELS = {
    "COMPLETE BLOOD COUNT": [
        ("WBC", "6.1", "x10^3/uL", "4.0 - 11.0", ""),
        ("Hemoglobin", "12.9", "g/dL", "12.0 - 15.5", ""),
        ("Platelets", "245", "x10^3/uL", "150 - 400", ""),
    ],
    "COMPREHENSIVE METABOLIC PANEL": [
        ("Sodium", "137", "mmol/L", "135 - 145", ""),
        ("Potassium", "4.2", "mmol/L", "3.5 - 5.1", ""),
        ("Creatinine", "0.9", "mg/dL", "0.6 - 1.1", ""),
        ("Glucose, fasting", "92", "mg/dL", "70 - 99", ""),
        ("ALT", "28", "U/L", "7 - 35", ""),
        ("AST", "24", "U/L", "10 - 35", ""),
    ],
    "LIPID PANEL": [
        ("Total cholesterol", "248", "mg/dL", "< 200", "H"),
        ("LDL cholesterol", "168", "mg/dL", "< 100", "H"),
        ("HDL cholesterol", "52", "mg/dL", "> 45", ""),
        ("Triglycerides", "140", "mg/dL", "< 150", ""),
    ],
    "THYROID FUNCTION": [
        ("TSH", "18.6", "mIU/L", "0.4 - 4.0", "H"),
        ("Free T4", "0.5", "ng/dL", "0.8 - 1.8", "L"),
    ],
    "VITAMINS": [
        ("Vitamin B12", "450", "pg/mL", "200 - 900", ""),
    ],
}


def main() -> None:
    w, h = 1100, 1180
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    navy, grey, red = (20, 40, 80), (110, 110, 110), (190, 20, 20)

    d.rectangle([0, 0, w, 120], fill=(235, 240, 248))
    d.text((40, 28), "Demo Clinical Laboratory", font=font(34, True), fill=navy)
    d.text((40, 76), "SYNTHETIC REPORT - FOR SOFTWARE DEMONSTRATION ONLY", font=font(18, True), fill=red)

    y = 145
    info = [
        ("Patient:", "DEMO PATIENT (synthetic)"), ("Sex / Age:", "F / 42"),
        ("Collected:", "2026-11-18 08:12"), ("Reported:", "2026-11-18 14:40"),
        ("Ordering clinician:", "Primary care"), ("Specimen:", "Serum, whole blood"),
    ]
    for i, (k, v) in enumerate(info):
        x = 40 if i % 2 == 0 else 580
        d.text((x, y), k, font=font(18, True), fill=grey)
        d.text((x + 190, y), v, font=font(18), fill="black")
        if i % 2 == 1:
            y += 32
    y += 20

    cols = [40, 430, 580, 750, 1000]
    d.line([40, y, w - 40, y], fill=navy, width=2)
    y += 10
    for x, label in zip(cols, ["TEST", "RESULT", "UNITS", "REFERENCE RANGE", "FLAG"]):
        d.text((x, y), label, font=font(17, True), fill=navy)
    y += 32
    d.line([40, y, w - 40, y], fill=navy, width=2)
    y += 14

    for panel, rows in PANELS.items():
        d.text((40, y), panel, font=font(19, True), fill=navy)
        y += 34
        for test, result, unit, ref, flag in rows:
            color = red if flag else "black"
            d.text((cols[0] + 20, y), test, font=font(19), fill="black")
            d.text((cols[1], y), result, font=font(19, bool(flag)), fill=color)
            d.text((cols[2], y), unit, font=font(19), fill="black")
            d.text((cols[3], y), ref, font=font(19), fill="black")
            d.text((cols[4], y), flag, font=font(19, True), fill=color)
            y += 32
        y += 14

    y += 10
    d.line([40, y, w - 40, y], fill=grey, width=1)
    d.text((40, y + 14), "H = above reference range    L = below reference range", font=font(16), fill=grey)
    d.text((40, y + 40), "This report is fictional and was generated for a hackathon demo.", font=font(16), fill=grey)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

"""Prompt templates and robust parsers for MedGemma.

Design rule: blinded reads NEVER see the clinician's note. They get only the image
and basic demographics, the same way an independent second reader would.

Outputs use simple `KEY: value` sections instead of JSON because small models follow
line formats more reliably than strict JSON.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import BlindedRead, NoteSummary, Patient, Reconciliation


@dataclass(frozen=True)
class Modality:
    key: str
    label: str
    icon: str
    reader: str
    instructions: str


MODALITIES: dict[str, Modality] = {
    "chest_xray": Modality(
        key="chest_xray",
        label="Chest X-ray",
        icon="🫁",
        reader="an expert radiologist",
        instructions=(
            "Read this chest X-ray. Systematically assess the lungs, pleura (including any "
            "pneumothorax or effusion), heart and mediastinum, bones, and any lines or devices. "
            "Be specific about each abnormality and its location (side and zone)."
        ),
    ),
    "skin": Modality(
        key="skin",
        label="Skin lesion photo",
        icon="🔬",
        reader="an expert dermatologist",
        instructions=(
            "Assess this image of a skin lesion. Evaluate asymmetry, border, color variegation, "
            "diameter, and dermoscopic structures (pigment network, dots and globules, streaks, "
            "regression, blue-white veil). State the most likely diagnosis and whether the lesion "
            "is suspicious for malignancy such as melanoma."
        ),
    ),
    "fundus": Modality(
        key="fundus",
        label="Retinal fundus photo",
        icon="👁️",
        reader="an expert ophthalmologist",
        instructions=(
            "Assess this retinal fundus photograph. Evaluate the optic disc, macula, vessels, and "
            "retina for microaneurysms, hemorrhages, hard exudates, cotton wool spots, venous "
            "beading, and neovascularization. If diabetic retinopathy is present, grade it as "
            "mild, moderate, or severe non-proliferative, or proliferative."
        ),
    ),
    "lab_report": Modality(
        key="lab_report",
        label="Lab report",
        icon="🧪",
        reader="an expert clinical pathologist",
        instructions=(
            "Read this laboratory report. List every abnormal result with its value, unit, and "
            "reference range. Then state what the pattern of abnormal results most likely "
            "indicates clinically."
        ),
    ),
}

URGENCY_LEVELS = ("ROUTINE", "SOON", "URGENT")


def blinded_read_prompt(modality: str, patient: Patient) -> str:
    m = MODALITIES[modality]
    return f"""You are {m.reader} providing an independent second read.
Patient: {patient.describe()}. No clinical history is provided, by design. Base your read only on what you see.

{m.instructions}

Respond in exactly this format:
FINDINGS:
<your findings>
IMPRESSION:
1. <most clinically important finding>
2. <next finding, if any>
URGENCY: <ROUTINE | SOON | URGENT>"""


def note_extraction_prompt(note: str) -> str:
    return f"""You are a clinical documentation specialist. Read the clinician's note below and extract the clinician's own working diagnosis and plan. Do not add your own opinion.

CLINICIAN NOTE:
<<<
{note.strip()}
>>>

Respond in exactly this format:
WORKING DIAGNOSIS: <the clinician's leading diagnosis>
KEY SYMPTOMS: <symptom>; <symptom>; <symptom>
PLANNED MEDICATIONS: <drug>; <drug>  (write NONE if no medications are planned)
DISPOSITION: <discharge, admit, follow-up interval, etc.>"""


def reconciliation_prompt(note: str, summary: NoteSummary, reads: list[tuple[str, BlindedRead]]) -> str:
    evidence_lines = []
    for i, (label, read) in enumerate(reads, 1):
        impression = read.impression or read.findings or read.raw
        evidence_lines.append(f"[{i}] {label} (urgency: {read.urgency.upper()})\n{impression.strip()}")
    evidence = "\n\n".join(evidence_lines)
    meds = "; ".join(summary.planned_medications) or "none"
    return f"""You are a senior attending physician performing a diagnostic safety review.
A clinician documented the note below. Separately, each piece of evidence was read independently by a specialist who did NOT see the note.

Decide whether the independent evidence supports the clinician's working diagnosis and plan.
- DISCORDANT: the evidence shows a significant finding that contradicts the working diagnosis, or that the plan does not address.
- CONCORDANT: the evidence is consistent with the working diagnosis and plan. Then use SEVERITY: INFO.

CLINICIAN NOTE:
<<<
{note.strip()}
>>>

CLINICIAN'S WORKING DIAGNOSIS: {summary.working_diagnosis or "unclear"}
PLANNED MEDICATIONS: {meds}
DISPOSITION: {summary.disposition or "unclear"}

INDEPENDENT EVIDENCE READS:
{evidence}

Respond in exactly this format:
VERDICT: <CONCORDANT | DISCORDANT>
SEVERITY: <INFO | IMPORTANT | URGENT>
HEADLINE: <one sentence>
FINDING: <the key finding from the evidence>
CONFLICTS WITH: <what in the note it conflicts with, or NONE>
WHY IT MATTERS: <one or two sentences of clinical reasoning>
SUGGESTED ACTION: <one sentence>
ADDENDUM: <a two-sentence addendum the clinician could paste into the chart>"""


# --------------------------------------------------------------------------- parsing

_DECORATION = re.compile(r"^[\s*#>\-_`]+|[*_`]+(?=\s*:)")


def parse_sections(text: str, keys: list[str]) -> dict[str, str]:
    """Split `KEY: value` formatted text into a dict. Tolerates markdown bold, bullets,
    different casing, and values that continue across multiple lines."""
    keys_upper = sorted((k.upper() for k in keys), key=len, reverse=True)
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        cleaned = _DECORATION.sub("", line).strip()
        upper = cleaned.upper()
        matched = None
        for key in keys_upper:
            if upper.startswith(key) and upper[len(key):].lstrip(" *_").startswith(":"):
                matched = key
                break
        if matched:
            current = matched
            value = cleaned[len(matched):].lstrip(" *_")[1:].strip(" *_")
            sections.setdefault(current, [])
            if value:
                sections[current].append(value)
        elif current is not None:
            sections[current].append(line.rstrip())
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _pick(value: str, options: tuple[str, ...], default: str) -> str:
    upper = value.upper()
    # Check longer/more specific options first so "DISCORDANT" is not read as "CONCORDANT".
    for option in sorted(options, key=len, reverse=True):
        if option in upper:
            return option.lower()
    return default


def _split_list(value: str) -> list[str]:
    if not value or value.strip().upper().startswith("NONE"):
        return []
    parts = re.split(r";|\n|,(?![^()]*\))", value)
    return [p.strip(" -*•.") for p in parts if p.strip(" -*•.")]


def parse_blinded_read(modality: str, text: str) -> BlindedRead:
    s = parse_sections(text, ["FINDINGS", "IMPRESSION", "URGENCY"])
    return BlindedRead(
        modality=modality,
        findings=s.get("FINDINGS", ""),
        impression=s.get("IMPRESSION", ""),
        urgency=_pick(s.get("URGENCY", ""), URGENCY_LEVELS, "unknown"),
        raw=text,
    )


def parse_note_summary(text: str) -> NoteSummary:
    s = parse_sections(text, ["WORKING DIAGNOSIS", "KEY SYMPTOMS", "PLANNED MEDICATIONS", "DISPOSITION"])
    return NoteSummary(
        working_diagnosis=s.get("WORKING DIAGNOSIS", ""),
        key_symptoms=_split_list(s.get("KEY SYMPTOMS", "")),
        planned_medications=_split_list(s.get("PLANNED MEDICATIONS", "")),
        disposition=s.get("DISPOSITION", ""),
        raw=text,
    )


def parse_reconciliation(text: str) -> Reconciliation:
    keys = [
        "VERDICT", "SEVERITY", "HEADLINE", "FINDING", "CONFLICTS WITH",
        "WHY IT MATTERS", "SUGGESTED ACTION", "ADDENDUM",
    ]
    s = parse_sections(text, keys)
    verdict = _pick(s.get("VERDICT", ""), ("CONCORDANT", "DISCORDANT"), "uncertain")
    severity = _pick(s.get("SEVERITY", ""), ("INFO", "IMPORTANT", "URGENT"), "important")
    if verdict == "concordant" and "SEVERITY" not in s:
        severity = "info"
    return Reconciliation(
        verdict=verdict,
        severity=severity,
        headline=s.get("HEADLINE", ""),
        finding=s.get("FINDING", ""),
        conflicts_with=s.get("CONFLICTS WITH", ""),
        why_it_matters=s.get("WHY IT MATTERS", ""),
        suggested_action=s.get("SUGGESTED ACTION", ""),
        addendum=s.get("ADDENDUM", ""),
        raw=text,
    )

"""Load demo cases from data/cases/*.json.

Notes contain {age} and {sex_word} placeholders. scripts/fetch_data.py records the real
demographics of each downloaded image in data/images/manifest.json, so the note always
matches the image (a 70-year-old's X-ray never gets a note saying "24-year-old").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .schema import Evidence, Patient

MANIFEST_PATH = config.IMAGES_DIR / "manifest.json"


@dataclass
class Case:
    id: str
    title: str
    specialty: str
    summary: str
    patient: Patient
    note: str
    evidence: list[Evidence]
    expected_verdict: str
    expected_keywords: list[list[str]] = field(default_factory=list)
    order: int = 99

    @property
    def missing(self) -> list[Path]:
        return [ev.path for ev in self.evidence if not Path(ev.path).exists()]

    @property
    def available(self) -> bool:
        return not self.missing


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}


def fill_note(template: str, patient: Patient) -> str:
    sex_word = {"M": "male", "F": "female"}.get((patient.sex or "").upper(), "patient")
    return template.replace("{age}", str(patient.age or "Adult")).replace("{sex_word}", sex_word)


def load_case(path: Path, manifest: dict | None = None) -> Case:
    raw = json.loads(Path(path).read_text())
    manifest = load_manifest() if manifest is None else manifest
    demo = manifest.get(raw["id"], {})
    patient = Patient(
        age=demo.get("age") or raw["patient"].get("age"),
        sex=demo.get("sex") or raw["patient"].get("sex"),
    )
    note = fill_note((config.CASES_DIR / raw["note_file"]).read_text(), patient)
    evidence = [
        Evidence(
            modality=e["modality"],
            path=config.DATA_DIR / e["path"],
            source=e.get("source", ""),
            ground_truth=e.get("ground_truth", ""),
        )
        for e in raw["evidence"]
    ]
    return Case(
        id=raw["id"],
        title=raw["title"],
        specialty=raw["specialty"],
        summary=raw.get("summary", ""),
        patient=patient,
        note=note,
        evidence=evidence,
        expected_verdict=raw["expected_verdict"],
        expected_keywords=[e.get("expected_keywords", []) for e in raw["evidence"]],
        order=raw.get("order", 99),
    )


def load_cases() -> list[Case]:
    manifest = load_manifest()
    cases = [load_case(p, manifest) for p in sorted(config.CASES_DIR.glob("*.json"))]
    return sorted(cases, key=lambda c: c.order)


def detected(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)

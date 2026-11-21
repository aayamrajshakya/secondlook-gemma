"""Plain data containers passed between pipeline stages and the UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Patient:
    age: int | None = None
    sex: str | None = None  # "M" / "F"

    def describe(self) -> str:
        sex = {"M": "male", "F": "female"}.get((self.sex or "").upper(), "patient")
        return f"{self.age}-year-old {sex}" if self.age else f"Adult {sex}"


@dataclass
class Evidence:
    modality: str  # key into prompts.MODALITIES
    path: Path
    source: str = ""  # dataset attribution, e.g. "NIH ChestX-ray14"
    ground_truth: str = ""  # dataset label, used only for curation and scoring, never shown to the model


@dataclass
class NoteSummary:
    working_diagnosis: str = ""
    key_symptoms: list[str] = field(default_factory=list)
    planned_medications: list[str] = field(default_factory=list)
    disposition: str = ""
    raw: str = ""


@dataclass
class BlindedRead:
    modality: str
    findings: str = ""
    impression: str = ""
    urgency: str = "unknown"  # routine | soon | urgent | unknown
    raw: str = ""


@dataclass
class Reconciliation:
    verdict: str = "uncertain"  # concordant | discordant | uncertain
    severity: str = "info"  # info | important | urgent
    headline: str = ""
    finding: str = ""
    conflicts_with: str = ""
    why_it_matters: str = ""
    suggested_action: str = ""
    addendum: str = ""
    raw: str = ""


@dataclass
class DrugFlag:
    task: str  # e.g. "hERG"
    label: str  # human label, e.g. "hERG blockade (QT prolongation)"
    answer: str  # model's option text, e.g. "blocks hERG"
    risk: bool | None  # None when the output could not be parsed
    raw: str = ""


@dataclass
class DrugReport:
    drug: str
    smiles: str = ""
    flags: list[DrugFlag] = field(default_factory=list)
    error: str = ""

    @property
    def risky(self) -> list[DrugFlag]:
        return [f for f in self.flags if f.risk]


def to_dict(obj) -> dict:
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return convert(asdict(obj))

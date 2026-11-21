"""Mock backend for developing the UI and pipeline WITHOUT model weights.

Its output is canned text, not model output. The UI labels it loudly. Never demo it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Sequence

from .base import Backend

_READ = """FINDINGS:
[MOCK OUTPUT - NOT A REAL MODEL READ] Abnormality noted for development purposes.
IMPRESSION:
1. Mock abnormal finding.
URGENCY: URGENT"""

_TRANSCRIBE = """EMERGENCY DEPARTMENT NOTE (Decoded from Doctor's Handwritten Chart)
HPI: 24-year-old male presenting with sudden acute shortness of breath and chest tightness while at desk.
Exam: HR 108, RR 22, SpO2 95% on room air. Lungs: breath sounds present bilaterally.
Impression: Acute panic attack with hyperventilation.
Plan:
- Lorazepam 0.5 mg PO now
- Citalopram 10 mg PO daily
- Ondansetron 4 mg PO PRN nausea
- Discharge home once symptoms improve."""

_EXTRACT = """WORKING DIAGNOSIS: Mock working diagnosis
KEY SYMPTOMS: mock symptom one; mock symptom two
PLANNED MEDICATIONS: citalopram
DISPOSITION: Discharge home"""

_RECONCILE = """VERDICT: DISCORDANT
SEVERITY: URGENT
HEADLINE: [MOCK] Evidence does not support the working diagnosis
FINDING: Mock finding from the blinded read
CONFLICTS WITH: Mock working diagnosis
WHY IT MATTERS: Mock explanation used only for UI development.
SUGGESTED ACTION: Mock action. Review the evidence before discharge.
ADDENDUM: [MOCK] SecondLook review found a discrepancy between the evidence and the working diagnosis."""


class MockBackend(Backend):
    name = "mock"
    label = "MOCK: canned text, not model output"
    is_live_model = False

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def stream_medgemma(self, prompt: str, images: Sequence[Path], max_tokens: int) -> Iterator[str]:
        prompt_lower = prompt.lower()
        if "handwritten" in prompt_lower or "transcribe" in prompt_lower:
            text = _TRANSCRIBE
        elif "VERDICT:" in prompt:
            text = _RECONCILE
        elif "WORKING DIAGNOSIS:" in prompt:
            text = _EXTRACT
        else:
            text = _READ
        for word in text.split(" "):
            if self.delay:
                time.sleep(self.delay)
            yield word + " "

    def txgemma(self, prompt: str, max_tokens: int = 8) -> str:
        return "(B)"

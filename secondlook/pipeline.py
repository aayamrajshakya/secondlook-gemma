"""The SecondLook pipeline.

  1. Blinded reads: MedGemma reads each piece of evidence WITHOUT the note.
  2. Note extraction: MedGemma pulls the clinician's working diagnosis and plan.
  3. Reconciliation: MedGemma compares the two and calls concordant / discordant.
  4. Medication screen: TxGemma checks every drug in the note for safety signals.

`run()` is a generator that yields the full RunState after every meaningful update,
so a UI can stream tokens without knowing anything about the stages.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

from . import config, prompts, txgemma
from .backends.base import Backend
from .imaging import prepare_image
from .schema import BlindedRead, DrugReport, Evidence, NoteSummary, Patient, Reconciliation, to_dict

STAGES = ("reads", "note", "reconcile", "meds", "done")


@dataclass
class RunState:
    backend_label: str = ""
    stage: str = "reads"
    log: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    read_raw: list[str] = field(default_factory=list)
    reads: list[BlindedRead | None] = field(default_factory=list)
    note_raw: str = ""
    summary: NoteSummary | None = None
    recon_raw: str = ""
    recon: Reconciliation | None = None
    drugs_found: list[str] = field(default_factory=list)
    drugs_planned: list[str] = field(default_factory=list)
    drug_reports: list[DrugReport] = field(default_factory=list)
    med_warnings: list[str] = field(default_factory=list)
    addendum: str = ""
    error: str = ""
    timings: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return to_dict(self)


class _Throttle:
    """Limit how often token-level updates are yielded to the UI."""

    def __init__(self, interval: float = 0.08):
        self.interval = interval
        self.last = 0.0

    def ready(self) -> bool:
        now = time.monotonic()
        if now - self.last >= self.interval:
            self.last = now
            return True
        return False


def compose_addendum(recon: Reconciliation | None, med_warnings: list[str]) -> str:
    lines = ["SecondLook review (AI decision support; requires clinician verification):"]
    if recon and recon.addendum:
        lines.append(recon.addendum.strip())
    elif recon and recon.headline:
        lines.append(recon.headline.strip())
    for w in med_warnings:
        lines.append(f"Medication safety (TxGemma prediction): {w}")
    return "\n".join(lines)


def transcribe_note(backend: Backend, image_path: str | Path) -> Iterator[str]:
    """Transcribe a handwritten doctor's note or paper prescription image using MedGemma."""
    image = prepare_image(image_path)
    prompt = prompts.handwritten_transcription_prompt()
    accumulated = ""
    for chunk in backend.stream_medgemma(prompt, [image], config.MAX_TOKENS_EXTRACT):
        accumulated += chunk
        yield accumulated


def run(
    backend: Backend,
    note: str,
    evidence: list[Evidence],
    patient: Patient,
    screen_meds: bool = True,
    save: bool = True,
) -> Iterator[RunState]:
    state = RunState(backend_label=backend.label, evidence=list(evidence))
    state.read_raw = [""] * len(evidence)
    state.reads = [None] * len(evidence)
    throttle = _Throttle()

    try:
        # 1. Blinded reads -------------------------------------------------------
        t0 = time.monotonic()
        state.stage = "reads"
        for i, ev in enumerate(evidence):
            modality = prompts.MODALITIES[ev.modality]
            state.log.append(f"{modality.icon} Blinded read of {modality.label} (note hidden from model)")
            yield state
            image = prepare_image(ev.path)
            prompt = prompts.blinded_read_prompt(ev.modality, patient)
            for chunk in backend.stream_medgemma(prompt, [image], config.MAX_TOKENS_READ):
                state.read_raw[i] += chunk
                if throttle.ready():
                    yield state
            state.reads[i] = prompts.parse_blinded_read(ev.modality, state.read_raw[i])
            yield state
        state.timings["reads"] = time.monotonic() - t0

        # 2. Note extraction -----------------------------------------------------
        t0 = time.monotonic()
        state.stage = "note"
        state.log.append("📝 Extracting the clinician's working diagnosis and plan")
        yield state
        for chunk in backend.stream_medgemma(prompts.note_extraction_prompt(note), [], config.MAX_TOKENS_EXTRACT):
            state.note_raw += chunk
            if throttle.ready():
                yield state
        state.summary = prompts.parse_note_summary(state.note_raw)
        state.timings["note"] = time.monotonic() - t0
        yield state

        # 3. Reconciliation ------------------------------------------------------
        t0 = time.monotonic()
        state.stage = "reconcile"
        state.log.append("⚖️ Reconciling independent reads against the working diagnosis")
        yield state
        labelled = [
            (prompts.MODALITIES[ev.modality].label, read)
            for ev, read in zip(evidence, state.reads)
            if read is not None
        ]
        recon_prompt = prompts.reconciliation_prompt(note, state.summary, labelled)
        for chunk in backend.stream_medgemma(recon_prompt, [], config.MAX_TOKENS_RECONCILE):
            state.recon_raw += chunk
            if throttle.ready():
                yield state
        state.recon = prompts.parse_reconciliation(state.recon_raw)
        state.timings["reconcile"] = time.monotonic() - t0
        state.addendum = compose_addendum(state.recon, [])
        yield state

        # 4. Medication screen ---------------------------------------------------
        if screen_meds:
            t0 = time.monotonic()
            state.stage = "meds"
            planned_text = "; ".join(state.summary.planned_medications)
            state.drugs_planned = txgemma.find_drugs(planned_text)
            state.drugs_found = txgemma.find_drugs(planned_text, note)
            if state.drugs_found:
                state.log.append(f"💊 TxGemma safety screen: {', '.join(state.drugs_found)}")
            else:
                state.log.append("💊 No recognized medications in the note")
            yield state
            for drug in state.drugs_found:
                state.drug_reports.append(txgemma.screen_drug(backend, drug))
                yield state
            state.med_warnings = txgemma.interaction_warnings(state.drug_reports)
            state.timings["meds"] = time.monotonic() - t0

        state.addendum = compose_addendum(state.recon, state.med_warnings)
        state.stage = "done"
        state.log.append("✅ Done")
        yield state

    except Exception as exc:  # noqa: BLE001 - surface any failure in the UI instead of crashing
        state.error = f"{type(exc).__name__}: {exc}"
        state.log.append(f"❌ {state.error}")
        traceback.print_exc()
        yield state

    finally:
        if save:
            config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            (config.RUNS_DIR / f"run-{stamp}.json").write_text(json.dumps(state.to_json(), indent=2))


def run_to_completion(*args, **kwargs) -> RunState:
    state = None
    for state in run(*args, **kwargs):
        pass
    return state

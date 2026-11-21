from PIL import Image

from secondlook.backends.mock_backend import MockBackend
from secondlook.pipeline import run, run_to_completion, transcribe_note
from secondlook.schema import Evidence, Patient

NOTE = """ED note. 24M with acute dyspnea and chest tightness. Impression: panic attack.
Plan: citalopram 10 mg daily, ondansetron PRN. Discharge home."""


def _image(tmp_path):
    path = tmp_path / "cxr.png"
    Image.new("L", (64, 64), 128).save(path)
    return path


def test_pipeline_end_to_end_with_mock(tmp_path):
    ev = [Evidence(modality="chest_xray", path=_image(tmp_path))]
    state = run_to_completion(MockBackend(), NOTE, ev, Patient(24, "M"), save=False)
    assert not state.error, state.error
    assert state.stage == "done"
    assert state.reads[0].urgency == "urgent"
    assert state.summary.working_diagnosis
    assert state.recon.verdict == "discordant"
    assert state.drugs_found == ["citalopram", "ondansetron"]
    assert any("QT stacking" in w for w in state.med_warnings)
    assert state.addendum.startswith("SecondLook review")


def test_reads_happen_before_note_is_parsed(tmp_path):
    ev = [Evidence(modality="chest_xray", path=_image(tmp_path))]
    order = []
    for state in run(MockBackend(), NOTE, ev, Patient(24, "M"), save=False):
        if not order or order[-1] != state.stage:
            order.append(state.stage)
    assert order[:2] == ["reads", "note"]


def test_pipeline_reports_errors(tmp_path):
    ev = [Evidence(modality="chest_xray", path=tmp_path / "missing.png")]
    state = run_to_completion(MockBackend(), NOTE, ev, Patient(24, "M"), save=False)
    assert state.error


def test_transcribe_note_with_mock(tmp_path):
    img_path = _image(tmp_path)
    chunks = list(transcribe_note(MockBackend(), img_path))
    assert len(chunks) > 0
    final = chunks[-1]
    assert "EMERGENCY DEPARTMENT NOTE" in final


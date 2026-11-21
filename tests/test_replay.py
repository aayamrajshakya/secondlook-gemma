import pytest
from PIL import Image

from secondlook.backends.mock_backend import MockBackend
from secondlook.backends.replay_backend import CacheMiss, OutputCache, RecordingBackend, ReplayBackend
from secondlook.pipeline import run_to_completion
from secondlook.schema import Evidence, Patient


def test_record_then_replay_reproduces_run(tmp_path):
    img = tmp_path / "x.png"
    Image.new("L", (32, 32), 50).save(img)
    ev = [Evidence(modality="skin", path=img)]
    note = "Clinic note. Mole, benign. Plan: reassure. Ibuprofen PRN."
    cache = OutputCache(tmp_path / "outputs.json")

    live = run_to_completion(RecordingBackend(MockBackend(), cache), note, ev, Patient(50, "F"), save=False)
    replay = run_to_completion(ReplayBackend(OutputCache(tmp_path / "outputs.json"), 0), note, ev,
                               Patient(50, "F"), save=False)

    assert not replay.error, replay.error
    assert replay.recon.verdict == live.recon.verdict
    assert replay.read_raw[0].strip() == live.read_raw[0].strip()
    assert [r.drug for r in replay.drug_reports] == [r.drug for r in live.drug_reports]


def test_replay_miss_raises(tmp_path):
    backend = ReplayBackend(OutputCache(tmp_path / "empty.json"), 0)
    with pytest.raises(CacheMiss):
        list(backend.stream_medgemma("never seen", [], 10))

import json

from secondlook import cases as cases_mod
from secondlook.prompts import MODALITIES


def test_all_cases_load_with_filled_notes():
    cases = cases_mod.load_cases()
    assert [c.id for c in cases][0] == "er_panic_ptx"
    assert len(cases) == 5
    for c in cases:
        assert "{age}" not in c.note and "{sex_word}" not in c.note
        assert c.expected_verdict in ("concordant", "discordant")
        for ev in c.evidence:
            assert ev.modality in MODALITIES


def test_lab_report_case_is_available_out_of_the_box():
    case = next(c for c in cases_mod.load_cases() if c.id == "labs_fatigue_hypothyroid")
    assert case.available


def test_manifest_demographics_override_defaults(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"er_panic_ptx": {"age": 31, "sex": "F"}}))
    monkeypatch.setattr(cases_mod, "MANIFEST_PATH", manifest)
    case = next(c for c in cases_mod.load_cases() if c.id == "er_panic_ptx")
    assert case.patient.age == 31
    assert "31-year-old female" in case.note


def test_detected_keywords():
    assert cases_mod.detected("Large RIGHT Pneumothorax.", ["pneumothorax"])
    assert not cases_mod.detected("Clear lungs.", ["pneumothorax"])

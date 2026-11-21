from secondlook.prompts import (
    MODALITIES,
    blinded_read_prompt,
    parse_blinded_read,
    parse_note_summary,
    parse_reconciliation,
    parse_sections,
)
from secondlook.schema import Patient


def test_blinded_prompt_never_contains_note_language():
    for key in MODALITIES:
        prompt = blinded_read_prompt(key, Patient(age=24, sex="M"))
        assert "24-year-old male" in prompt
        assert "No clinical history is provided" in prompt
        assert "URGENCY:" in prompt


def test_parse_sections_handles_markdown_and_multiline():
    text = """**FINDINGS:** There is a large right pneumothorax.
The left lung is clear.

**IMPRESSION:**
1. Large right pneumothorax.
**Urgency**: URGENT"""
    s = parse_sections(text, ["FINDINGS", "IMPRESSION", "URGENCY"])
    assert s["FINDINGS"].startswith("There is a large right pneumothorax.")
    assert "left lung is clear" in s["FINDINGS"]
    assert s["IMPRESSION"] == "1. Large right pneumothorax."
    assert s["URGENCY"] == "URGENT"


def test_parse_blinded_read_urgency():
    read = parse_blinded_read("chest_xray", "FINDINGS:\nx\nIMPRESSION:\n1. y\nURGENCY: urgent")
    assert read.urgency == "urgent"
    assert read.impression == "1. y"


def test_parse_note_summary_lists():
    text = """WORKING DIAGNOSIS: Panic attack
KEY SYMPTOMS: dyspnea; chest tightness; palpitations
PLANNED MEDICATIONS: citalopram 10 mg daily; lorazepam 0.5 mg PRN
DISPOSITION: Discharge home"""
    s = parse_note_summary(text)
    assert s.working_diagnosis == "Panic attack"
    assert s.key_symptoms == ["dyspnea", "chest tightness", "palpitations"]
    assert s.planned_medications == ["citalopram 10 mg daily", "lorazepam 0.5 mg PRN"]


def test_parse_note_summary_none_meds():
    s = parse_note_summary("WORKING DIAGNOSIS: x\nPLANNED MEDICATIONS: NONE")
    assert s.planned_medications == []


def test_reconciliation_discordant_not_confused_with_concordant():
    r = parse_reconciliation("VERDICT: DISCORDANT\nSEVERITY: URGENT\nHEADLINE: h\nADDENDUM: a b")
    assert r.verdict == "discordant"
    assert r.severity == "urgent"
    assert r.addendum == "a b"


def test_reconciliation_concordant_defaults_to_info():
    r = parse_reconciliation("VERDICT: Concordant\nHEADLINE: fine")
    assert r.verdict == "concordant"
    assert r.severity == "info"


def test_reconciliation_garbage_is_uncertain():
    r = parse_reconciliation("I am not sure what you mean.")
    assert r.verdict == "uncertain"

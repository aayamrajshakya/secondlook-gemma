from secondlook.backends.mock_backend import MockBackend
from secondlook.txgemma import (
    FALLBACK_TEMPLATES,
    find_drugs,
    interaction_warnings,
    is_risk,
    options_from_template,
    parse_choice,
    screen_drug,
)


def test_find_drugs_generic_and_brand_in_order():
    note = "Given Zofran in ED. Plan: start Celexa 10 mg, lorazepam PRN."
    assert find_drugs(note) == ["ondansetron", "citalopram", "lorazepam"]


def test_find_drugs_ignores_substrings():
    assert find_drugs("Patient is a metformingo enthusiast") == []


def test_options_and_risk_for_every_template():
    for key, template in FALLBACK_TEMPLATES.items():
        a, b = options_from_template(template)
        assert not is_risk(a), key
        assert is_risk(b), key


def test_parse_choice():
    assert parse_choice("(B)") == "B"
    assert parse_choice(" A") == "A"
    assert parse_choice("Answer: (A) does not") == "A"
    assert parse_choice("unclear") is None


def test_screen_drug_with_mock():
    report = screen_drug(MockBackend(), "citalopram")
    assert report.smiles
    assert all(f.risk for f in report.flags)  # mock always answers (B)


def test_qt_stacking_warning():
    backend = MockBackend()
    reports = [screen_drug(backend, d) for d in ("citalopram", "ondansetron")]
    warnings = interaction_warnings(reports)
    assert any("QT stacking" in w for w in warnings)

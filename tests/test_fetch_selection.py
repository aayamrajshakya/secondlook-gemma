import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("fetch_data", Path(__file__).parent.parent / "scripts" / "fetch_data.py")
fetch_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_data)


def nih(idx, labels, age, view="PA", sex="M"):
    return {"Image Index": idx, "Finding Labels": labels, "Patient Age": age, "View Position": view, "Patient Gender": sex}


def test_parse_nih_age_formats():
    assert fetch_data.parse_nih_age("060Y") == 60
    assert fetch_data.parse_nih_age("33") == 33
    assert fetch_data.parse_nih_age("") is None


def test_select_nih_prefers_single_label_pa_in_age_range():
    rows = [
        nih("a.png", "Pneumothorax", "025Y"),
        nih("b.png", "Pneumothorax|Effusion", "030Y"),
        nih("c.png", "Pneumothorax", "070Y"),
        nih("d.png", "Pneumothorax", "028Y", view="AP"),
        nih("e.png", "No Finding", "025Y"),
    ]
    picks = fetch_data.select_nih(rows, {"Pneumothorax"}, (18, 40), k=1)
    assert [p["file"] for p in picks] == ["a.png"]
    assert picks[0]["age"] == 25 and picks[0]["sex"] == "M"


def test_select_nih_relaxes_when_needed_and_never_picks_wrong_label():
    rows = [nih("b.png", "Pneumothorax|Effusion", "030Y"), nih("e.png", "No Finding", "025Y")]
    picks = fetch_data.select_nih(rows, {"Pneumothorax"}, (18, 40), k=3)
    assert [p["file"] for p in picks] == ["b.png"]


def test_select_ham_only_histo_confirmed_melanoma():
    rows = [
        {"image_id": "ISIC_1", "dx": "mel", "dx_type": "histo", "age": "55.0", "sex": "male", "localization": "back"},
        {"image_id": "ISIC_2", "dx": "mel", "dx_type": "consensus", "age": "55.0", "sex": "male", "localization": "back"},
        {"image_id": "ISIC_3", "dx": "nv", "dx_type": "histo", "age": "55.0", "sex": "male", "localization": "back"},
    ]
    picks = fetch_data.select_ham(rows, k=5)
    assert [p["file"] for p in picks] == ["ISIC_1.jpg"]
    assert picks[0]["sex"] == "M" and picks[0]["age"] == 55


def test_select_aptos_prefers_proliferative():
    rows = [{"id_code": "x", "diagnosis": "3"}, {"id_code": "y", "diagnosis": "4"}, {"id_code": "z", "diagnosis": "0"}]
    picks = fetch_data.select_aptos(rows, k=2)
    assert [p["file"] for p in picks] == ["y.png", "x.png"]

"""Download a few candidate demo images per case from Kaggle (not the full datasets).

Sources:
  - NIH ChestX-ray14 (Kaggle 5% sample)  -> chest X-ray cases
  - HAM10000 / ISIC                      -> melanoma case (biopsy-confirmed only)
  - APTOS 2019 Blindness Detection       -> diabetic retinopathy case

Setup:
  1. Create a Kaggle API token (kaggle.com > Settings > API > Create New Token) and save it
     to ~/.kaggle/kaggle.json (chmod 600), or export KAGGLE_USERNAME and KAGGLE_KEY.
  2. Accept the APTOS 2019 competition rules on its Kaggle page (needed for download).

Usage:
  python scripts/fetch_data.py            # all cases, 4 candidates each
  python scripts/fetch_data.py --k 6 --cases er_panic_ptx

Candidates go to data/images/candidates/<case_id>/. The first candidate is promoted to
data/images/<case_id>.<ext> so the app works right away. Then run scripts/curate.py to
test every candidate with MedGemma and promote the one it reads correctly.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "images"
CANDIDATES = IMAGES / "candidates"
MANIFEST = IMAGES / "manifest.json"

NIH = ("dataset", "nih-chest-xrays/sample")
HAM = ("dataset", "kmader/skin-cancer-mnist-ham10000")
APTOS = ("competition", "aptos2019-blindness-detection")

NIH_CSV = ["sample_labels.csv", "sample/sample_labels.csv"]
NIH_IMAGE_DIRS = ["sample/images", "images", "sample/sample/images"]
HAM_CSV = ["HAM10000_metadata.csv"]
HAM_IMAGE_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2", "ham10000_images_part_1", "ham10000_images_part_2"]
APTOS_CSV = ["train.csv"]
APTOS_IMAGE_DIRS = ["train_images"]


# ----------------------------------------------------------------------------- selection (pure)

def parse_nih_age(value: str) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return None
    age = int(digits)
    if str(value).strip().upper().endswith(("M", "D")):  # months/days in some rows
        return 0
    return age


def select_nih(rows: list[dict], labels: set[str], age_range: tuple[int, int], k: int, seed: int = 42) -> list[dict]:
    """Prefer single-label PA films in the age range, then relax step by step."""

    def matches(row, exact: bool, use_age: bool, pa_only: bool) -> bool:
        found = set(row["Finding Labels"].split("|"))
        if exact and not (len(found) == 1 and found <= labels):
            return False
        if not exact and not (found & labels):
            return False
        if pa_only and row.get("View Position", "PA") != "PA":
            return False
        if use_age:
            age = parse_nih_age(row.get("Patient Age", ""))
            if age is None or not (age_range[0] <= age <= age_range[1]):
                return False
        return True

    chosen: list[dict] = []
    for exact, use_age, pa_only in [(True, True, True), (True, True, False), (True, False, True), (False, True, True), (False, False, False)]:
        pool = [r for r in rows if matches(r, exact, use_age, pa_only) and r not in chosen]
        random.Random(seed).shuffle(pool)
        chosen += pool[: k - len(chosen)]
        if len(chosen) >= k:
            break
    return [
        {
            "file": r["Image Index"],
            "age": parse_nih_age(r.get("Patient Age", "")),
            "sex": (r.get("Patient Gender") or "")[:1].upper() or None,
            "label": r["Finding Labels"],
            "source_id": r["Image Index"],
        }
        for r in chosen
    ]


def select_ham(rows: list[dict], k: int, seed: int = 42) -> list[dict]:
    def ok(r, strict: bool) -> bool:
        if r.get("dx") != "mel" or r.get("dx_type") != "histo":
            return False
        if not strict:
            return True
        try:
            age = float(r.get("age") or 0)
        except ValueError:
            return False
        return r.get("localization") in ("back", "trunk", "upper extremity") and 35 <= age <= 70

    pool = [r for r in rows if ok(r, True)] or [r for r in rows if ok(r, False)]
    random.Random(seed).shuffle(pool)
    return [
        {
            "file": f"{r['image_id']}.jpg",
            "age": int(float(r["age"])) if r.get("age") else None,
            "sex": {"male": "M", "female": "F"}.get(r.get("sex", ""), None),
            "label": f"melanoma (histopathology), {r.get('localization', '')}",
            "source_id": r["image_id"],
        }
        for r in pool[:k]
    ]


def select_aptos(rows: list[dict], k: int, seed: int = 42) -> list[dict]:
    chosen: list[dict] = []
    for grade in ("4", "3"):  # proliferative, then severe NPDR
        pool = [r for r in rows if r.get("diagnosis") == grade]
        random.Random(seed).shuffle(pool)
        chosen += pool[: k - len(chosen)]
        if len(chosen) >= k:
            break
    names = {"4": "proliferative DR", "3": "severe non-proliferative DR"}
    return [
        {"file": f"{r['id_code']}.png", "age": None, "sex": None, "label": names[r["diagnosis"]], "source_id": r["id_code"]}
        for r in chosen
    ]


# ----------------------------------------------------------------------------- kaggle I/O

def kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except BaseException as exc:  # the kaggle package can SystemExit on missing credentials
        sys.exit(
            f"Kaggle authentication failed ({exc}).\n"
            "Create a token at kaggle.com > Settings > API, save it to ~/.kaggle/kaggle.json "
            "and chmod 600 it, or export KAGGLE_USERNAME and KAGGLE_KEY."
        )


def download(api, source: tuple[str, str], file_name: str, dest: Path) -> Path:
    kind, ref = source
    dest.mkdir(parents=True, exist_ok=True)
    base = Path(file_name).name
    target = dest / base
    if target.exists():
        return target
    if kind == "dataset":
        api.dataset_download_file(ref, file_name, path=str(dest), force=True, quiet=True)
    else:
        api.competition_download_file(ref, file_name, path=str(dest), force=True, quiet=True)
    zipped = dest / f"{base}.zip"
    if zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            member = next(n for n in zf.namelist() if n.endswith(base))
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
        zipped.unlink()
    if not target.exists():
        raise FileNotFoundError(f"{file_name} not found after download from {ref}")
    return target


def download_first(api, source, names: list[str], dest: Path) -> Path:
    errors = []
    for name in names:
        try:
            return download(api, source, name, dest)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    raise FileNotFoundError(f"Could not download any of {names} from {source[1]}:\n  " + "\n  ".join(errors))


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fetch_images(api, source, dirs: list[str], picks: list[dict], dest: Path) -> list[dict]:
    """Download each pick, trying each known image directory. Remembers the one that works."""
    got = []
    for pick in picks:
        for d in list(dirs):
            try:
                pick["path"] = str(download(api, source, f"{d}/{pick['file']}", dest).relative_to(ROOT))
                dirs.remove(d)
                dirs.insert(0, d)  # try the working directory first next time
                got.append(pick)
                print(f"    ok   {pick['file']}  ({pick['label']}, age {pick['age']}, sex {pick['sex']})")
                break
            except Exception:  # noqa: BLE001
                continue
        else:
            print(f"    FAIL {pick['file']}")
    return got


# ----------------------------------------------------------------------------- main

CASE_SPECS = {
    "er_panic_ptx": dict(source=NIH, kind="nih", labels={"Pneumothorax"}, ages=(18, 40)),
    "er_pneumonia_concordant": dict(source=NIH, kind="nih", labels={"Pneumonia", "Consolidation"}, ages=(18, 65)),
    "derm_mole_melanoma": dict(source=HAM, kind="ham"),
    "eye_dm_retinopathy": dict(source=APTOS, kind="aptos"),
}


def promote(case_id: str, pick: dict, manifest: dict) -> None:
    src = ROOT / pick["path"]
    for old in IMAGES.glob(f"{case_id}.*"):
        old.unlink()
    dst = IMAGES / f"{case_id}{src.suffix.lower()}"
    shutil.copyfile(src, dst)
    manifest[case_id] = {k: pick.get(k) for k in ("file", "age", "sex", "label", "source_id", "path")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=4, help="candidates per case")
    ap.add_argument("--cases", nargs="*", default=list(CASE_SPECS), help="subset of case ids")
    args = ap.parse_args()

    api = kaggle_api()
    meta_dir = IMAGES / "_metadata"
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    candidates_index_path = CANDIDATES / "candidates.json"
    candidates_index = json.loads(candidates_index_path.read_text()) if candidates_index_path.exists() else {}
    tables: dict[str, list[dict]] = {}

    for case_id in args.cases:
        spec = CASE_SPECS[case_id]
        print(f"\n== {case_id}")
        try:
            if spec["kind"] == "nih":
                if "nih" not in tables:
                    tables["nih"] = read_csv(download_first(api, NIH, NIH_CSV, meta_dir))
                picks = select_nih(tables["nih"], spec["labels"], spec["ages"], args.k)
                dirs = NIH_IMAGE_DIRS
            elif spec["kind"] == "ham":
                tables.setdefault("ham", read_csv(download_first(api, HAM, HAM_CSV, meta_dir)))
                picks, dirs = select_ham(tables["ham"], args.k), HAM_IMAGE_DIRS
            else:
                tables.setdefault("aptos", read_csv(download_first(api, APTOS, APTOS_CSV, meta_dir)))
                picks, dirs = select_aptos(tables["aptos"], args.k), APTOS_IMAGE_DIRS
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped: {exc}")
            if spec["kind"] == "aptos":
                print("  (Did you accept the APTOS 2019 competition rules on Kaggle?)")
            continue

        got = fetch_images(api, spec["source"], dirs, picks, CANDIDATES / case_id)
        if not got:
            print("  no images downloaded")
            continue
        candidates_index[case_id] = got
        promote(case_id, got[0], manifest)
        print(f"  promoted {got[0]['file']} -> data/images/{case_id}{Path(got[0]['path']).suffix.lower()}")

    CANDIDATES.mkdir(parents=True, exist_ok=True)
    candidates_index_path.write_text(json.dumps(candidates_index, indent=2))
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print("\nNext: python scripts/curate.py   (tests each candidate with MedGemma and promotes the best)")


if __name__ == "__main__":
    main()

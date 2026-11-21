"""Fetch SMILES strings for demo drugs from PubChem and store them in data/drugs.json.

Run once while online. The app then works fully offline.
Usage: python scripts/fetch_smiles.py [extra drug names...]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "drugs.json"

DEFAULT_DRUGS = [
    # psychiatric / anxiolytic
    "citalopram", "escitalopram", "sertraline", "fluoxetine", "lorazepam", "alprazolam", "haloperidol",
    # antibiotics
    "azithromycin", "clarithromycin", "levofloxacin", "moxifloxacin", "ciprofloxacin", "amoxicillin",
    "doxycycline", "ceftriaxone",
    # antiemetic / GI
    "ondansetron", "omeprazole",
    # cardio / metabolic
    "amiodarone", "metformin", "atorvastatin", "lisinopril", "furosemide", "levothyroxine",
    # analgesic / anti-inflammatory
    "acetaminophen", "ibuprofen", "naproxen", "prednisone", "oxycodone", "tramadol",
    # renal colic
    "tamsulosin", "ketorolac",
]

PROPS = "SMILES,IsomericSMILES,CanonicalSMILES,ConnectivitySMILES"


def fetch(name: str) -> tuple[str, int]:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{urllib.parse.quote(name)}/property/{PROPS}/JSON"
    )
    with urllib.request.urlopen(url, timeout=20) as resp:
        props = json.load(resp)["PropertyTable"]["Properties"][0]
    for key in ("SMILES", "IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES"):
        if props.get(key):
            return props[key], props["CID"]
    raise ValueError(f"No SMILES returned for {name}")


def main() -> None:
    names = DEFAULT_DRUGS + [n.lower() for n in sys.argv[1:]]
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        if name in data:
            continue
        try:
            smiles, cid = fetch(name)
            data[name] = {"smiles": smiles, "pubchem_cid": cid}
            print(f"ok   {name:15s} CID {cid}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {exc}")
        time.sleep(0.25)  # PubChem asks for <= 5 requests/second
    OUT.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")
    print(f"Wrote {len(data)} drugs to {OUT}")


if __name__ == "__main__":
    main()

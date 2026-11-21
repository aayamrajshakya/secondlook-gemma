"""Medication safety screen with TxGemma (predict variant).

TxGemma predicts molecular properties from a SMILES string using Therapeutics Data
Commons (TDC) task prompts. We use a handful of safety-relevant tasks. These are
molecule-level predictions to prompt a pharmacist check, not prescribing advice.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import config
from .backends.base import Backend
from .schema import DrugFlag, DrugReport

DRUGS_PATH = config.DATA_DIR / "drugs.json"

_PREAMBLE = "Instructions: Answer the following question about drug properties.\n"
_QUESTION = "Question: Given a drug SMILES string, predict whether it\n"
_FOOTER = "Drug SMILES: {Drug SMILES}\nAnswer:"

# Fallback templates in TDC format. The official ones from the TxGemma repo
# (models/tdc_prompts.json, downloaded by scripts/setup_mac.sh) take priority.
FALLBACK_TEMPLATES = {
    "hERG": _PREAMBLE
    + "Context: Human ether-à-go-go related gene (hERG) is crucial for the coordination of the heart's "
    "beating. Thus, if a drug blocks the hERG, it could lead to severe adverse effects. Therefore, reliable "
    "prediction of hERG liability in the early stages of drug design is quite important to reduce the risk "
    "of cardiotoxicity-related attritions in the later development stages.\n"
    + _QUESTION
    + "(A) does not block hERG (B) blocks hERG\n"
    + _FOOTER,
    "CYP3A4_Veith": _PREAMBLE
    + "Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various "
    "molecules and chemicals within cells. Specifically, CYP3A4 is an important enzyme in the body, mainly "
    "found in the liver and in the intestine. It oxidizes small foreign organic molecules (xenobiotics), "
    "such as toxins or drugs, so that they can be removed from the body.\n"
    + _QUESTION
    + "(A) does not inhibit CYP3A4 (B) inhibits CYP3A4\n"
    + _FOOTER,
    "DILI": _PREAMBLE
    + "Context: Drug-induced liver injury (DILI) is fatal liver disease caused by drugs and it has been the "
    "single most frequent cause of safety-related drug marketing withdrawals for the past 50 years.\n"
    + _QUESTION
    + "(A) cannot cause DILI (B) can cause DILI\n"
    + _FOOTER,
    "ClinTox": _PREAMBLE
    + "Context: The ClinTox dataset includes drugs that have failed clinical trials for toxicity reasons "
    "and also drugs that are associated with successful trials.\n"
    + _QUESTION
    + "(A) is not toxic (B) is toxic\n"
    + _FOOTER,
}


@dataclass(frozen=True)
class SafetyTask:
    key: str
    label: str


SAFETY_TASKS = (
    SafetyTask("hERG", "hERG blockade (QT prolongation risk)"),
    SafetyTask("CYP3A4_Veith", "CYP3A4 inhibition (drug interaction risk)"),
    SafetyTask("DILI", "Drug-induced liver injury"),
    SafetyTask("ClinTox", "Clinical toxicity"),
)

# Common brand names so notes written either way are recognized.
BRAND_NAMES = {
    "celexa": "citalopram", "lexapro": "escitalopram", "zoloft": "sertraline", "prozac": "fluoxetine",
    "ativan": "lorazepam", "xanax": "alprazolam", "haldol": "haloperidol",
    "zithromax": "azithromycin", "z-pak": "azithromycin", "zpak": "azithromycin", "biaxin": "clarithromycin",
    "levaquin": "levofloxacin", "avelox": "moxifloxacin", "cipro": "ciprofloxacin",
    "amoxil": "amoxicillin", "augmentin": "amoxicillin", "rocephin": "ceftriaxone",
    "zofran": "ondansetron", "prilosec": "omeprazole", "cordarone": "amiodarone",
    "glucophage": "metformin", "lipitor": "atorvastatin", "lasix": "furosemide",
    "synthroid": "levothyroxine", "tylenol": "acetaminophen", "advil": "ibuprofen", "motrin": "ibuprofen",
    "aleve": "naproxen", "ultram": "tramadol", "flomax": "tamsulosin", "toradol": "ketorolac",
}


@lru_cache(maxsize=1)
def load_drugs() -> dict[str, dict]:
    return json.loads(DRUGS_PATH.read_text()) if DRUGS_PATH.exists() else {}


@lru_cache(maxsize=1)
def load_templates() -> dict[str, str]:
    templates = dict(FALLBACK_TEMPLATES)
    path = Path(config.TDC_PROMPTS_PATH)
    if path.exists():
        official = json.loads(path.read_text())
        for key in templates:
            if key in official:
                templates[key] = official[key]
    return templates


def using_official_templates() -> bool:
    return Path(config.TDC_PROMPTS_PATH).exists()


def find_drugs(*texts: str) -> list[str]:
    """Return known generic drug names mentioned in the texts, in order of first mention."""
    known = load_drugs()
    vocab = {name: name for name in known} | {b: g for b, g in BRAND_NAMES.items() if g in known}
    hits: list[tuple[int, str]] = []
    for text in texts:
        lower = text.lower()
        for term, generic in vocab.items():
            m = re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lower)
            if m:
                hits.append((m.start(), generic))
    seen: list[str] = []
    for _, generic in sorted(hits):
        if generic not in seen:
            seen.append(generic)
    return seen


def options_from_template(template: str) -> tuple[str, str]:
    m = re.search(r"\(A\)\s*(.+?)\s*\(B\)\s*(.+?)\s*(?:\n|$)", template)
    if not m:
        return "option A", "option B"
    return m.group(1).strip(), m.group(2).strip()


def parse_choice(output: str) -> str | None:
    m = re.search(r"\(([AB])\)", output) or re.match(r"\s*([AB])\b", output)
    return m.group(1) if m else None


def is_risk(option_text: str) -> bool:
    # Negative options are phrased "does not block", "is not toxic", "cannot cause".
    return not re.search(r"\bnot\b|cannot", option_text.lower())


def screen_drug(backend: Backend, drug: str, tasks=SAFETY_TASKS) -> DrugReport:
    info = load_drugs().get(drug)
    if not info:
        return DrugReport(drug=drug, error="No SMILES on file. Run scripts/fetch_smiles.py.")
    report = DrugReport(drug=drug, smiles=info["smiles"])
    templates = load_templates()
    for task in tasks:
        template = templates[task.key]
        prompt = template.replace("{Drug SMILES}", info["smiles"])
        raw = backend.txgemma(prompt, max_tokens=8)
        choice = parse_choice(raw)
        opt_a, opt_b = options_from_template(template)
        answer = {"A": opt_a, "B": opt_b}.get(choice or "", "unparsed output")
        risk = is_risk(answer) if choice else None
        report.flags.append(DrugFlag(task=task.key, label=task.label, answer=answer, risk=risk, raw=raw))
    return report


def interaction_warnings(reports: list[DrugReport]) -> list[str]:
    """Simple rules layered on top of TxGemma's per-drug predictions."""
    warnings = []
    herg = [r.drug for r in reports if any(f.task == "hERG" and f.risk for f in r.flags)]
    if len(herg) >= 2:
        warnings.append(
            f"QT stacking: {', '.join(herg)} are each predicted to block hERG. Combining QT-prolonging "
            "drugs raises arrhythmia risk. Consider a baseline ECG and pharmacist review."
        )
    cyp = [r.drug for r in reports if any(f.task == "CYP3A4_Veith" and f.risk for f in r.flags)]
    if cyp and len(reports) >= 2:
        others = [r.drug for r in reports if r.drug not in cyp]
        if others:
            warnings.append(
                f"Interaction check: {', '.join(cyp)} predicted to inhibit CYP3A4 and may raise levels of "
                f"co-prescribed drugs ({', '.join(others)}) that are CYP3A4 substrates. Verify with a pharmacist."
            )
    return warnings

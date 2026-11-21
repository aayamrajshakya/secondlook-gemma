# 🩺 SecondLook

**A second read of the patient chart that hasn't already made up its mind.**

Doctors rarely miss findings because they can't see them. They miss them because they've already decided what they're looking for (anchoring bias). SecondLook gives every chart an independent second read: MedGemma reads the evidence *without seeing the clinician's note*, then checks whether that evidence supports the working diagnosis. TxGemma screens every drug in the plan for safety signals. Everything runs on an 8GB MacBook Air, so patient data never leaves the device.

Built for the GDG Cloud Boston x Northeastern University Hackathon (Nov 21, 2026).

## How it works

```
        Clinician note (working diagnosis + plan)
                        │
   ┌────────────────────┼─────────────────────────┐
   ▼                    ▼                         ▼
Evidence image      (note withheld)           Drugs in the note
X-ray / skin /                                     │
fundus / lab report                                ▼
   │                                          TxGemma 2B
   ▼                                          hERG · CYP3A4 · DILI · ClinTox
1. MedGemma blinded read                           │
   (image + age/sex only)                          │
   │                                               │
2. MedGemma extracts the clinician's hypothesis    │
   │                                               │
3. MedGemma reconciles: concordant / discordant    │
   └──────────────────────┬────────────────────────┘
                          ▼
   🟢 Concordant  /  🚨 Discrepancy + draft chart addendum
```

| Step | Model | Sees |
|---|---|---|
| 0. Handwritten transcription *(optional)* | MedGemma 1.5 4B (vision) | Handwritten doctor note / prescription pad image |
| 1. Blinded read | MedGemma 1.5 4B (vision) | Image + age/sex. **Never the note.** |
| 2. Hypothesis extraction | MedGemma 1.5 4B (text) | The note (typed or decoded from handwriting) |
| 3. Reconciliation | MedGemma 1.5 4B (text) | Note + independent reads |
| 4. Medication screen | TxGemma 2B predict | Each drug's SMILES string |

## Demo cases

| # | Note says | Evidence | Expected |
|---|---|---|---|
| 1 | "Panic attack" (discharge on citalopram) | Chest X-ray: pneumothorax | 🚨 + QT stacking (citalopram + ondansetron) |
| 2 | "Benign mole, reassure" | Skin photo: biopsy-proven melanoma | 🚨 |
| 3 | "Diabetes, no complications" | Fundus photo: proliferative retinopathy | 🚨 |
| 4 | "Depression, start sertraline" | Lab report: TSH 18.6, low free T4 | 🚨 |
| 5 | "Pneumonia" | Chest X-ray: consolidation | 🟢 control, shows it doesn't cry wolf |

Notes and the lab report are synthetic. Images come from public de-identified datasets.

## 🍎 Apple Silicon Mac Setup (with `uv`)

Follow these steps on your MacBook (Apple Silicon M1/M2/M3/M4, 8GB+ RAM):

### 1. Prerequisites (Install `uv`)
If you don't have `uv` installed yet on your Mac:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(Or via Homebrew: `brew install uv`)*

### 2. Accept Model Terms & Get HF Token
Both models are gated under Google's Health AI Developer Foundations terms:
1. Log into Hugging Face and click **"Acknowledge license / Accept terms"** on:
   - [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
   - [google/txgemma-2b-predict](https://huggingface.co/google/txgemma-2b-predict)
2. Create a Hugging Face **Read Token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3. Create Virtualenv & Install Dependencies with `uv`
```bash
# Create Python 3.12 environment
uv venv --python 3.12 .venv

# Install Apple Silicon MLX requirements
uv pip install --python .venv/bin/python -r requirements-mac.txt
```

### 4. Authenticate Hugging Face
```bash
uv run --python .venv/bin/python huggingface-cli login
```
*(Paste your Hugging Face read token when prompted.)*

### 5. Download & Convert Models to 4-bit MLX
Run the automated Mac setup script (downloads models and converts to 4-bit MLX, ~9GB download, 10–20 min):
```bash
bash scripts/setup_mac.sh
```
*Note: This script automatically detects `uv` and uses it. It also runs a smoke test on the bundled lab report.*

### 6. Get Demo Data & Assets
1. **Render synthetic assets** (printed lab report & doctor handwritten prescription pad):
   ```bash
   uv run --python .venv/bin/python scripts/make_lab_report.py
   uv run --python .venv/bin/python scripts/make_handwritten_note.py
   ```
2. *(Optional)* **Download Kaggle imaging datasets**:
   Place your Kaggle API token at `~/.kaggle/kaggle.json` (`chmod 600 ~/.kaggle/kaggle.json`), accept the [APTOS 2019 rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules), and run:
   ```bash
   uv run --python .venv/bin/python scripts/fetch_data.py
   uv run --python .venv/bin/python scripts/curate.py --full
   ```

### 7. Run SecondLook
```bash
# Launch local server (http://127.0.0.1:7860) with live MLX on-device acceleration:
uv run --python .venv/bin/python app.py

# Or launch with a public phone link to snap photos of notes:
uv run --python .venv/bin/python app.py --share
```

---

### Linux / NVIDIA

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-cuda.txt
.venv/bin/python app.py --backend hf
```
Needs about 10GB of VRAM at bf16. CPU-only works but is very slow.

## Backends

| Backend | Use | UI badge |
|---|---|---|
| `mlx` | Live on Apple Silicon (default on Mac) | green **Live** |
| `hf` | Live via transformers (default elsewhere) | green **Live** |
| `replay` | Plays back recorded live outputs (stage fallback if the laptop struggles) | blue **Replay** |
| `mock` | Canned text for UI development only | red **MOCK** |

Every live run is recorded to `data/cache/outputs.json`. Commit that file after curating so `replay` works on the day. Replay only works for inputs that ran live before.

## Project layout

```
app.py                     Gradio UI
secondlook/
  pipeline.py              4-stage streaming pipeline
  prompts.py               MedGemma prompts + tolerant KEY: value parsers
  txgemma.py               TDC safety tasks, drug matching, QT-stacking rules
  cases.py                 demo case loader (fills note demographics from the image)
  imaging.py               image normalization
  backends/                mlx, hf, mock, record/replay
scripts/
  setup_mac.sh             venv + 4-bit MLX conversion + smoke test
  fetch_data.py            a few candidate images per case from Kaggle
  curate.py                pick images MedGemma reads correctly; scorecard
  smoke_test.py            check both models respond
  fetch_smiles.py          PubChem SMILES -> data/drugs.json
  make_lab_report.py       renders the synthetic lab report
data/cases/                case definitions, synthetic notes, lab report
tests/                     pytest suite (runs with the mock backend, no weights needed)
```

## Tests

```bash
uv run --python .venv/bin/python pytest -q
```

## Limitations (say these out loud if asked)

- Research prototype, not a medical device. Output needs clinician verification.
- Demo images were chosen from a few candidates per case by `curate.py`, which reports the hit rate. The tool is not perfect, and the demo shouldn't imply it is.
- MedGemma at 4-bit loses some accuracy compared with bf16.
- TxGemma predicts molecular properties from structure. That's a signal to prompt a pharmacist check, not a drug-interaction database.
- Drug recognition matches a curated list of ~30 drugs (`data/drugs.json`). Add more with `scripts/fetch_smiles.py <name>`.

## Data and licenses

- NIH ChestX-ray14: NIH Clinical Center, public release
- HAM10000: Tschandl et al., CC BY-NC 4.0
- APTOS 2019 Blindness Detection: Kaggle competition data, subject to the competition rules (non-commercial)
- MedGemma and TxGemma: Health AI Developer Foundations terms of use

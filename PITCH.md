# SecondLook: pitch and demo script (3 minutes)

## Opening (20s)
> "Most of us will experience at least one diagnostic error in our lifetime. That's from the National Academies. A big reason is anchoring: a doctor decides early, then reads the evidence looking for what they already believe. We built SecondLook, a second read of the chart that hasn't made up its mind."

## Demo (2 min)

**Case 1: "Panic attack"** (the main one)
1. Show the note: a young patient, short of breath, labeled a panic attack, discharged on citalopram.
2. Click **Run SecondLook**. Point at step 1: *"MedGemma is reading this X-ray right now, and it has not seen the note. Here's the exact prompt"* (open the Transparency accordion for a second).
3. Blinded read streams: pneumothorax.
4. Verdict card turns red: **Diagnostic discrepancy, URGENT.**
5. Medication table: *"And TxGemma flags that citalopram and ondansetron both block hERG. That's QT stacking. The wrong diagnosis led to the wrong drugs, and both were caught before discharge."*
6. Point at the draft addendum: *"Copy, paste into the chart. It fits the workflow doctors already have."*

**Case 4 or 2 (quick)**: switch the dropdown to show it isn't X-ray only. A lab report reveals hypothyroidism behind "depression," or a melanoma behind "benign mole."

**Case 5 (10s)**: the pneumonia control comes back green. *"It doesn't cry wolf."*

## Why Gemma (20s)
> "Two models from Google's Health AI Developer Foundations: MedGemma 1.5 for multimodal clinical reasoning across X-rays, skin, retina, and lab reports, and TxGemma for molecular safety. Both run at 4-bit on an 8GB MacBook Air, so no patient data ever leaves the room. A hospital could deploy the same pipeline on Vertex AI or Cloud Run."

## Close (10s)
> "SecondLook: every chart deserves a second look."

## Likely judge questions
- **"How accurate is it?"** Be honest: "We tested several real dataset images per case. curate.py reports the hit rate. It's decision support with a human in the loop, not a diagnosis."
- **"Why blind the model to the note?"** "If the AI reads the note first it can anchor too. Blinding is how independent double-reading works in radiology."
- **"Why not one big prompt?"** "Separate stages make each step inspectable, keep the read unanchored, and fit a 4B model's context and attention better."
- **"Is TxGemma a drug-interaction checker?"** "No. It predicts properties like hERG blockade from molecular structure. We layer a simple QT-stacking rule on top and frame it as a pharmacist prompt."
- **"What's next?"** "EHR integration through FHIR, more modalities (CT slices with MedGemma 1.5), comparing with prior studies over time, and a clinical validation study."

## If something breaks on stage
- Model slow or crashing: switch the Backend dropdown to `replay`. The badge turns blue and says it's replaying recorded live output. Say so.
- Close Chrome tabs before the demo. 8GB is tight.

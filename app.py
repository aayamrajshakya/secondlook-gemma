"""SecondLook web UI (Gradio).

  python app.py                    # live models (MLX on Apple Silicon, transformers elsewhere)
  python app.py --backend replay   # replay recorded live outputs (stage fallback)
  python app.py --backend mock     # canned text for UI development, clearly labelled
"""

from __future__ import annotations

import argparse
import html

import gradio as gr

from secondlook import config, prompts, txgemma
from secondlook.backends import BACKENDS, Backend, get_backend
from secondlook.cases import load_cases
from secondlook.pipeline import RunState, run
from secondlook.schema import Evidence, Patient

CUSTOM = "Custom case: write your own note and upload evidence"
MODALITY_CHOICES = [(f"{m.icon} {m.label}", m.key) for m in prompts.MODALITIES.values()]
STEPS = [
    ("reads", "Blinded read", "MedGemma reads the evidence without the note"),
    ("note", "Clinician's hypothesis", "MedGemma extracts the working diagnosis"),
    ("reconcile", "Reconcile", "Does the evidence support it?"),
    ("meds", "Medication safety", "TxGemma screens every drug"),
]

_backends: dict[str, Backend] = {}


def backend_for(name: str) -> Backend:
    if name not in _backends:
        _backends[name] = get_backend(name)
    return _backends[name]


def esc(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


# ----------------------------------------------------------------------------- renderers

def render_badge(name: str) -> str:
    backend = backend_for(name)
    kind = "live" if backend.is_live_model else ("replay" if name == "replay" else "mock")
    return f'<div class="sl-badge-wrap"><span class="sl-badge sl-badge-{kind}">{esc(backend.label)}</span></div>'


def render_steps(state: RunState | None) -> str:
    order = [s[0] for s in STEPS] + ["done"]
    current = order.index(state.stage) if state else -1
    parts = []
    for i, (_key, title, sub) in enumerate(STEPS):
        if state and state.error and i == current:
            cls = "error"
        elif current > i:
            cls = "done"
        elif current == i:
            cls = "active"
        else:
            cls = "pending"
        mark = "✓" if cls == "done" else ("!" if cls == "error" else str(i + 1))
        parts.append(
            f'<div class="sl-step sl-step-{cls}"><div class="sl-step-num">{mark}</div>'
            f'<div><div class="sl-step-title">{title}</div><div class="sl-step-sub">{sub}</div></div></div>'
        )
    return f'<div class="sl-steps">{"".join(parts)}</div>'


def render_verdict(state: RunState | None) -> str:
    if state is None:
        return '<div class="sl-verdict sl-verdict-idle">Pick a case and press <b>Run SecondLook</b>.</div>'
    if state.error:
        return f'<div class="sl-verdict sl-verdict-uncertain"><div class="sl-verdict-title">Run failed</div>{esc(state.error)}</div>'
    r = state.recon
    if r is None:
        return '<div class="sl-verdict sl-verdict-idle">⏳ Waiting for the reconciliation step…</div>'
    if r.verdict == "discordant":
        cls, icon, title = "discordant", "🚨", f"Diagnostic discrepancy · {r.severity.upper()}"
    elif r.verdict == "concordant":
        cls, icon, title = "concordant", "🟢", "Concordant: the evidence supports the working diagnosis"
    else:
        cls, icon, title = "uncertain", "⚠️", "Uncertain: review the raw output below"
    rows = [
        ("Working diagnosis (note)", state.summary.working_diagnosis if state.summary else ""),
        ("Key finding (evidence)", r.finding),
        ("Conflicts with", r.conflicts_with),
        ("Why it matters", r.why_it_matters),
        ("Suggested action", r.suggested_action),
    ]
    body = "".join(
        f'<div class="sl-row"><div class="sl-row-k">{k}</div><div class="sl-row-v">{esc(v)}</div></div>'
        for k, v in rows
        if v and not (k == "Conflicts with" and v.strip().upper().startswith("NONE"))
    )
    return (
        f'<div class="sl-verdict sl-verdict-{cls}"><div class="sl-verdict-title">{icon} {title}</div>'
        f'<div class="sl-verdict-headline">{esc(r.headline)}</div>{body}</div>'
    )


def _format_sections(raw: str) -> str:
    out = raw.strip()
    for key in ("FINDINGS:", "IMPRESSION:", "URGENCY:"):
        out = out.replace(key, f"\n**{key}**")
    return out.strip().replace("\n", "  \n")


def render_reads(state: RunState | None, patient_desc: str = "") -> str:
    if state is None or not state.evidence:
        return "*The blinded read will stream here. The model sees only the image and demographics, never the note.*"
    blocks = []
    for i, ev in enumerate(state.evidence):
        m = prompts.MODALITIES[ev.modality]
        read = state.reads[i]
        urgency = f" · urgency **{read.urgency.upper()}**" if read and read.urgency != "unknown" else ""
        raw = state.read_raw[i] or "⏳ waiting…"
        blocks.append(
            f"#### {m.icon} {m.label}{urgency}\n"
            f"<sub>Model input: image + “{patient_desc}”. Clinician note withheld.</sub>\n\n{_format_sections(raw)}"
        )
    return "\n\n---\n\n".join(blocks)


def render_hypothesis(state: RunState | None) -> str:
    if state is None or (not state.note_raw and state.summary is None):
        return "*Starts after the blinded read finishes, so the read can't be anchored by the note.*"
    s = state.summary
    if s is None:
        return _format_sections(state.note_raw)
    meds = ", ".join(s.planned_medications) or "none"
    symptoms = ", ".join(s.key_symptoms) or "—"
    return (
        f"**Working diagnosis:** {s.working_diagnosis or '—'}  \n"
        f"**Key symptoms:** {symptoms}  \n"
        f"**Planned medications:** {meds}  \n"
        f"**Disposition:** {s.disposition or '—'}"
    )


def render_meds(state: RunState | None) -> str:
    caption = (
        '<div class="sl-caption">TxGemma 2B predictions from each drug\'s molecular structure (SMILES). '
        "Safety signals for pharmacist review, not prescribing advice.</div>"
    )
    if state is None or state.stage in ("reads", "note", "reconcile"):
        return '<div class="sl-muted">Runs last: every drug named in the note is screened.</div>' + caption
    if not state.drugs_found:
        return '<div class="sl-muted">No recognized medications in the note.</div>' + caption
    head = "".join(f"<th>{esc(t.label)}</th>" for t in txgemma.SAFETY_TASKS)
    rows = []
    for drug in state.drugs_found:
        report = next((r for r in state.drug_reports if r.drug == drug), None)
        tag = '<span class="sl-pill">planned</span>' if drug in state.drugs_planned else '<span class="sl-pill sl-pill-muted">in note</span>'
        if report is None:
            cells = "".join('<td class="sl-cell-wait">…</td>' for _ in txgemma.SAFETY_TASKS)
        elif report.error:
            cells = f'<td colspan="{len(txgemma.SAFETY_TASKS)}">{esc(report.error)}</td>'
        else:
            cells = ""
            for f in report.flags:
                if f.risk is None:
                    cells += f'<td class="sl-cell-wait" title="{esc(f.raw)}">?</td>'
                elif f.risk:
                    cells += f'<td class="sl-cell-risk" title="{esc(f.answer)}">⚠️ {esc(f.answer)}</td>'
                else:
                    cells += f'<td class="sl-cell-ok" title="{esc(f.answer)}">✓</td>'
        rows.append(f"<tr><td><b>{esc(drug)}</b> {tag}</td>{cells}</tr>")
    warnings = "".join(f'<div class="sl-warning">⚠️ {esc(w)}</div>' for w in state.med_warnings)
    return f'<table class="sl-meds"><tr><th>Drug</th>{head}</tr>{"".join(rows)}</table>{warnings}{caption}'


def render_raw(state: RunState | None, blinded_prompts: list[str]) -> str:
    if state is None:
        return "*Nothing yet.*"
    parts = []
    for i, p in enumerate(blinded_prompts):
        parts.append(f"**Exact blinded-read prompt #{i + 1}** (note it contains no clinical history):\n```\n{p}\n```")
    if state.note_raw:
        parts.append(f"**Note extraction output:**\n```\n{state.note_raw.strip()}\n```")
    if state.recon_raw:
        parts.append(f"**Reconciliation output:**\n```\n{state.recon_raw.strip()}\n```")
    if state.timings:
        parts.append("**Timings:** " + ", ".join(f"{k} {v:.1f}s" for k, v in state.timings.items()))
    parts.append("**Log:**  \n" + "  \n".join(state.log))
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- handlers

CASES = {f"{c.order}. {c.title} · {c.specialty}": c for c in load_cases()}


def case_choices() -> list[str]:
    labels = [label + ("" if c.available else "  (image not downloaded)") for label, c in CASES.items()]
    return labels + [CUSTOM]


def default_case() -> str:
    """First demo case whose evidence is on disk, so the opening screen is runnable."""
    for label, c in CASES.items():
        if c.available:
            return label
    return case_choices()[0]


def on_case(choice: str):
    label = choice.replace("  (image not downloaded)", "")
    case = CASES.get(label)
    if case is None:
        return ("", 45, "F", None, "chest_xray", "*Write a note and upload an image.*", None, "lab_report")
    ev = case.evidence[0]
    image = str(ev.path) if ev.path.exists() else None
    info = f"**{case.summary}**  \nEvidence source: {ev.source}."
    if image is None:
        info += "  \n⚠️ Image not downloaded yet: run `python scripts/fetch_data.py`."
    return (case.note, case.patient.age, case.patient.sex, image, ev.modality, info, None, "lab_report")


def on_run(backend_name, note, age, sex, img1, mod1, img2, mod2, screen_meds):
    patient = Patient(int(age) if age else None, sex)
    evidence = [Evidence(modality=m, path=p) for p, m in ((img1, mod1), (img2, mod2)) if p]
    blinded = [prompts.blinded_read_prompt(ev.modality, patient) for ev in evidence]

    def frame(state, addendum=""):
        return (
            render_steps(state),
            render_verdict(state),
            render_reads(state, patient.describe()),
            render_hypothesis(state),
            render_meds(state),
            addendum,
            render_raw(state, blinded),
        )

    if not note.strip() or not evidence:
        err = RunState(error="Add a clinician note and at least one piece of evidence.")
        yield frame(err)
        return

    backend = backend_for(backend_name)
    for state in run(backend, note, evidence, patient, screen_meds=screen_meds):
        yield frame(state, state.addendum if state.stage in ("meds", "done") or state.recon else "")


# ----------------------------------------------------------------------------- layout

CSS = """
.sl-header {display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:4px}
.sl-brand h1 {margin:0; font-size:2rem; letter-spacing:-0.02em}
.sl-brand p {margin:2px 0 0; opacity:.75}
.sl-badge-wrap {text-align:right}
.sl-badge {display:inline-block; padding:6px 12px; border-radius:999px; font-weight:600; font-size:.85rem; border:1px solid}
.sl-badge-live {color:#15803d; border-color:#15803d55; background:#16a34a1a}
.sl-badge-replay {color:#1d4ed8; border-color:#1d4ed855; background:#2563eb1a}
.sl-badge-mock {color:#b91c1c; border-color:#b91c1c; background:#dc26261f}
.dark .sl-badge-live {color:#4ade80} .dark .sl-badge-replay {color:#93c5fd} .dark .sl-badge-mock {color:#fca5a5}
.sl-steps {display:grid; grid-template-columns:repeat(4, 1fr); gap:10px}
@media (max-width: 800px) {.sl-steps {grid-template-columns:repeat(2, 1fr)}}
.sl-step {display:flex; gap:10px; align-items:center; padding:10px 12px; border-radius:12px; border:1px solid var(--border-color-primary); background:var(--background-fill-secondary)}
.sl-step-num {width:28px; height:28px; flex:none; border-radius:50%; display:grid; place-items:center; font-weight:700; background:var(--background-fill-primary); border:1px solid var(--border-color-primary)}
.sl-step-title {font-weight:600; font-size:.92rem}
.sl-step-sub {font-size:.78rem; opacity:.7}
.sl-step-active {border-color:#2563eb; box-shadow:0 0 0 2px #2563eb33}
.sl-step-active .sl-step-num {background:#2563eb; color:white; border-color:#2563eb; animation:sl-pulse 1.2s infinite}
.sl-step-done .sl-step-num {background:#16a34a; color:white; border-color:#16a34a}
.sl-step-error .sl-step-num {background:#dc2626; color:white; border-color:#dc2626}
.sl-step-pending {opacity:.55}
@keyframes sl-pulse {50% {box-shadow:0 0 0 6px #2563eb22}}
.sl-verdict {border-radius:14px; padding:18px 20px; border:1px solid var(--border-color-primary); border-left-width:6px}
.sl-verdict-idle {opacity:.7; border-left-color:var(--border-color-primary)}
.sl-verdict-discordant {border-left-color:#dc2626; background:#dc262612}
.sl-verdict-concordant {border-left-color:#16a34a; background:#16a34a12}
.sl-verdict-uncertain {border-left-color:#d97706; background:#d9770612}
.sl-verdict-title {font-size:1.25rem; font-weight:700; margin-bottom:4px}
.sl-verdict-headline {font-size:1.02rem; margin-bottom:10px}
.sl-row {display:grid; grid-template-columns:190px 1fr; gap:12px; padding:6px 0; border-top:1px solid var(--border-color-primary)}
@media (max-width: 640px) {.sl-row {grid-template-columns:1fr; gap:2px}}
.sl-row-k {font-weight:600; opacity:.75; font-size:.9rem}
.sl-meds {width:100%; border-collapse:collapse; font-size:.9rem}
.sl-meds th, .sl-meds td {padding:8px 10px; border-bottom:1px solid var(--border-color-primary); text-align:left; vertical-align:top}
.sl-meds th {font-size:.78rem; text-transform:uppercase; letter-spacing:.03em; opacity:.7}
.sl-cell-risk {color:#b91c1c; font-weight:600} .dark .sl-cell-risk {color:#fca5a5}
.sl-cell-ok {color:#15803d} .dark .sl-cell-ok {color:#4ade80}
.sl-cell-wait {opacity:.5}
.sl-pill {font-size:.7rem; padding:2px 8px; border-radius:999px; background:#2563eb22; color:#1d4ed8; font-weight:600; margin-left:4px}
.dark .sl-pill {color:#93c5fd}
.sl-pill-muted {background:var(--background-fill-secondary); color:inherit; opacity:.7}
.sl-warning {margin-top:10px; padding:10px 12px; border-radius:10px; background:#d9770618; border:1px solid #d9770655}
.sl-caption, .sl-muted {font-size:.8rem; opacity:.7; margin-top:8px}
.sl-footer {font-size:.8rem; opacity:.65; text-align:center; margin-top:12px}
"""


def build(default_backend: str) -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    with gr.Blocks(title="SecondLook", theme=theme, css=CSS) as demo:
        with gr.Row(elem_classes="sl-header"):
            gr.HTML(
                '<div class="sl-brand"><h1>🩺 SecondLook</h1>'
                "<p>A second read of the patient chart that hasn't already made up its mind · "
                "MedGemma 1.5 + TxGemma · fully on-device</p></div>"
            )
            badge = gr.HTML(render_badge(default_backend))

        with gr.Row():
            case = gr.Dropdown(case_choices(), value=default_case(), label="Demo case", scale=4)
            backend = gr.Dropdown(list(BACKENDS), value=default_backend, label="Backend", scale=1)
        case_info = gr.Markdown()

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.Markdown("### 📝 Clinician note")
                note = gr.Textbox(lines=18, max_lines=30, show_label=False, placeholder="Paste a clinician note…")
                with gr.Row():
                    age = gr.Number(label="Age", precision=0)
                    sex = gr.Dropdown(["F", "M"], label="Sex")
                    screen_meds = gr.Checkbox(value=True, label="TxGemma medication screen")
            with gr.Column(scale=4):
                gr.Markdown("### 🔍 Evidence")
                img1 = gr.Image(type="filepath", show_label=False, height=380)
                mod1 = gr.Dropdown(MODALITY_CHOICES, value="chest_xray", label="Evidence type")
                with gr.Accordion("Add a second piece of evidence", open=False):
                    img2 = gr.Image(type="filepath", show_label=False, height=240)
                    mod2 = gr.Dropdown(MODALITY_CHOICES, value="lab_report", label="Evidence type")

        go = gr.Button("Run SecondLook", variant="primary", size="lg")
        steps = gr.HTML(render_steps(None))
        verdict = gr.HTML(render_verdict(None))

        with gr.Row(equal_height=False):
            with gr.Column():
                gr.Markdown("### 👁️ Independent blinded read")
                reads = gr.Markdown(render_reads(None))
            with gr.Column():
                gr.Markdown("### 🧠 Clinician's working hypothesis")
                hypothesis = gr.Markdown(render_hypothesis(None))

        gr.Markdown("### 💊 Medication safety (TxGemma)")
        meds = gr.HTML(render_meds(None))

        addendum = gr.Textbox(label="Draft chart addendum (copy into the note)", lines=4, show_copy_button=True)
        with gr.Accordion("Transparency: exact prompts and raw model output", open=False):
            raw = gr.Markdown("*Nothing yet.*")

        gr.HTML(
            '<div class="sl-footer">Research prototype for a hackathon. Not a medical device and not for clinical use. '
            "Clinician notes and the lab report are synthetic; images are from public de-identified datasets "
            "(NIH ChestX-ray14, HAM10000, APTOS 2019).</div>"
        )

        case_outputs = [note, age, sex, img1, mod1, case_info, img2, mod2]
        case.change(on_case, case, case_outputs)
        demo.load(on_case, case, case_outputs)
        backend.change(render_badge, backend, badge)
        go.click(
            on_run,
            [backend, note, age, sex, img1, mod1, img2, mod2, screen_meds],
            [steps, verdict, reads, hypothesis, meds, addendum, raw],
        )
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=config.default_backend(), choices=BACKENDS)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = ap.parse_args()
    build(args.backend).queue().launch(
        server_port=args.port, share=args.share, allowed_paths=[str(config.DATA_DIR)]
    )


if __name__ == "__main__":
    main()

import pickle
import textwrap
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "do_fmbs_data"
OUT = ROOT / "DO_FMBS_slide_Presentation.pptx"

INK = RGBColor(16, 42, 67)
TEXT = RGBColor(51, 78, 104)
MUTED = RGBColor(98, 125, 152)
WHITE = RGBColor(255, 255, 255)
BG = RGBColor(246, 248, 251)
GRAY = RGBColor(216, 222, 233)
BLUE = RGBColor(47, 128, 237)
GREEN = RGBColor(0, 168, 150)
ORANGE = RGBColor(245, 158, 11)
RED = RGBColor(214, 69, 80)
PURPLE = RGBColor(124, 58, 237)
BODY_FONT_PT = 14
BODY_FONT_MIN_PT = 14


def load_pickle(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def load_npz(name):
    return np.load(DATA / name, allow_pickle=True)


def pct(value, decimals=2):
    return f"{value * 100:.{decimals}f}%"


def rgb_hex(rgb):
    return f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def fit_font(text, box_w, box_h, base=12, minimum=7):
    chars_per_line = max(18, int(box_w * 13))
    lines = 0
    for part in str(text).split("\n"):
        lines += max(1, len(textwrap.wrap(part, chars_per_line)))
    capacity = max(1, int(box_h * 5.7))
    if lines <= capacity:
        return Pt(base)
    return Pt(max(minimum, base - (lines - capacity) * 0.55))


def add_bg(slide):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BG
    shape.line.fill.background()


def add_header(slide, title, subtitle, number):
    bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.68))
    bar.fill.solid()
    bar.fill.fore_color.rgb = INK
    bar.line.fill.background()
    tx = slide.shapes.add_textbox(Inches(0.45), Inches(0.12), Inches(11.5), Inches(0.30))
    p = tx.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = WHITE
    if subtitle:
        st = slide.shapes.add_textbox(Inches(0.45), Inches(0.43), Inches(11.3), Inches(0.18))
        p = st.text_frame.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(8.8)
        p.font.color.rgb = RGBColor(217, 226, 236)
    sn = slide.shapes.add_textbox(Inches(12.25), Inches(0.16), Inches(0.55), Inches(0.22))
    p = sn.text_frame.paragraphs[0]
    p.text = f"{number:02d}"
    p.alignment = PP_ALIGN.RIGHT
    p.font.size = Pt(9)
    p.font.color.rgb = RGBColor(217, 226, 236)


def add_footer(slide, text=""):
    if not text:
        return
    left = slide.shapes.add_textbox(Inches(0.45), Inches(7.17), Inches(8.2), Inches(0.18))
    p = left.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(7)
    p.font.color.rgb = MUTED
    right = slide.shapes.add_textbox(Inches(10.7), Inches(7.17), Inches(2.1), Inches(0.18))
    p = right.text_frame.paragraphs[0]
    p.text = "Lydiah Moraa Machora"
    p.alignment = PP_ALIGN.RIGHT
    p.font.size = Pt(7)
    p.font.color.rgb = MUTED


def new_slide(prs, title, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_header(slide, title, subtitle, len(prs.slides))
    add_footer(slide)
    return slide


def add_box(slide, x, y, w, h, title, bullets, color=BLUE, font=12):
    box = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = GRAY
    band = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.22))
    band.fill.solid()
    band.fill.fore_color.rgb = color
    band.line.fill.background()
    title_box = slide.shapes.add_textbox(Inches(x + 0.13), Inches(y + 0.30), Inches(w - 0.26), Inches(0.30))
    tf = title_box.text_frame
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    p = tf.paragraphs[0]
    p.text = title
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = INK
    body_text = "\n".join([f"- {b}" for b in bullets])
    body = slide.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.68), Inches(w - 0.30), Inches(h - 0.78))
    tf = body.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = body_text
    body_size = fit_font(body_text, w, h, base=max(BODY_FONT_PT, float(font)), minimum=BODY_FONT_MIN_PT)
    p.font.size = body_size
    p.font.color.rgb = TEXT
    for run in p.runs:
        run.font.size = body_size
        run.font.color.rgb = TEXT


def add_metric(slide, x, y, w, h, label, value, note, color):
    box = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = GRAY
    t = slide.shapes.add_textbox(Inches(x + 0.12), Inches(y + 0.10), Inches(w - 0.24), Inches(0.20))
    p = t.text_frame.paragraphs[0]
    p.text = label
    p.font.size = Pt(7.8)
    p.font.bold = True
    p.font.color.rgb = TEXT
    v = slide.shapes.add_textbox(Inches(x + 0.12), Inches(y + 0.36), Inches(w - 0.24), Inches(0.34))
    p = v.text_frame.paragraphs[0]
    p.text = value
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = color
    n = slide.shapes.add_textbox(Inches(x + 0.12), Inches(y + h - 0.25), Inches(w - 0.24), Inches(0.18))
    p = n.text_frame.paragraphs[0]
    p.text = note
    p.font.size = Pt(6.7)
    p.font.color.rgb = MUTED


def add_research_opening_slides(prs):
    s = new_slide(
        prs,
        "Privacy-Preserving Dynamic Optimization Model for FMBS",
        "Mitigating adversarial privacy attacks in federated multimodal biometric systems",
    )
    title = s.shapes.add_textbox(Inches(0.65), Inches(1.15), Inches(12.0), Inches(0.92))
    p = title.text_frame.paragraphs[0]
    p.text = "A Privacy-Preserving Dynamic Optimization Model for Federated Multimodal Biometric Systems"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = INK
    add_box(s, 0.75, 2.45, 5.85, 2.05, "Candidate", [
        "Lydiah Moraa Machora",
        "I152/P/9515/23",
        "Master of Science research presentation",
    ], BLUE, 16)
    add_box(s, 6.95, 2.45, 5.65, 2.05, "Supervision", [
        "Dr. Richard Omollo, PhD",
        "Dr. Gonza Omoro, PhD",
        "Focus: privacy, robustness, and federated biometric learning",
    ], GREEN, 16)
    add_box(s, 0.75, 4.9, 11.85, 1.05, "Presentation focus", [
        "The study addresses adversarial privacy attacks, federated learning risks, and defense validation for face and fingerprint biometric modalities.",
    ], ORANGE, 16)

    s = new_slide(prs, "Statement of the Problem", "Why the study was necessary")
    add_box(s, 0.45, 1.05, 6.15, 2.35, "Research gap", [
        "FMBS supports privacy-preserving authentication by keeping biometric data decentralized.",
        "Federated learning still depends on semi-trusted participants, creating insider-threat and cross-modal risks.",
    ], RED, 14)
    add_box(s, 6.95, 1.05, 5.9, 2.35, "Adversarial threat", [
        "Attackers can manipulate inputs, infer private information, or corrupt training through poisoned updates.",
        "The study focuses on evasion, membership inference, and model poisoning.",
    ], ORANGE, 14)
    add_box(s, 0.45, 3.85, 12.4, 2.05, "Core problem", [
        "The key challenge was the lack of an adaptive defense model that preserves biometric utility, confidentiality, and robustness under attack.",
        "DO-FMBS addresses this by combining dynamic optimization, privacy preservation, adversarial robustness, and federated multimodal learning.",
    ], BLUE, 14)

    s = new_slide(prs, "Research Objectives", "General and specific objectives guiding the study")
    add_box(s, 0.55, 1.05, 12.2, 1.45, "Main objective", [
        "To develop a privacy-preserving dynamic optimization model for mitigating adversarial attacks in FMBS.",
    ], GREEN, 16)
    add_box(s, 0.55, 2.85, 5.9, 2.95, "Specific objectives I", [
        "Examine the types and characteristics of adversarial attacks on FMBS.",
        "Assess how adversarial inputs affect privacy and model performance.",
        "Review existing privacy-preserving models for FMBS.",
    ], BLUE, 14)
    add_box(s, 6.85, 2.85, 5.9, 2.95, "Specific objectives II", [
        "Develop the proposed privacy-preserving dynamic optimization model.",
        "Validate the model using robustness, privacy, performance, and statistical evidence.",
    ], PURPLE, 14)

    s = new_slide(prs, "Research Questions", "Questions addressed by the implementation and validation")
    add_box(s, 0.45, 1.05, 6.05, 2.35, "Attack and impact questions", [
        "Which adversarial attacks affect FMBS, and what are their characteristics?",
        "How do adversarial inputs affect privacy and recognition performance?",
    ], ORANGE, 14)
    add_box(s, 6.85, 1.05, 6.05, 2.35, "Model development question", [
        "How can a robust privacy-preserving dynamic optimization model be developed for FMBS?",
        "Which defenses can jointly address privacy leakage and adversarial robustness?",
    ], BLUE, 14)
    add_box(s, 0.45, 3.85, 12.45, 2.05, "Validation question", [
        "How can the proposed model be validated against adversarial privacy attacks?",
        "The implementation answers this through attack simulation, defense construction, robustness testing, classification metrics, and statistical validation.",
    ], GREEN, 14)

    s = new_slide(prs, "Scope of the Study", "Boundaries and evaluation focus")
    add_box(s, 0.45, 1.05, 6.1, 2.65, "Technical scope", [
        "The study focused on privacy attacks against FMBS within federated learning environments.",
        "It examined evasion, membership inference, and poisoning because they affect authentication, privacy, and model integrity.",
    ], BLUE, 14)
    add_box(s, 6.9, 1.05, 5.95, 2.65, "Biometric scope", [
        "The multimodal setting used face and fingerprint biometric modalities.",
        "The implementation used publicly available biometric datasets, not field data collection.",
    ], GREEN, 14)
    add_box(s, 0.45, 4.05, 12.4, 1.75, "Validation scope", [
        "Evaluation covered attack impact, privacy-preserving defense behavior, adversarial robustness, classification performance, and statistical significance.",
    ], PURPLE, 14)

    s = new_slide(prs, "Theoretical Framework", "The theories supporting model design and analysis")
    add_box(s, 0.45, 1.05, 6.05, 4.85, "Adversarial Machine Learning Theory", [
        "Explains learning under attack, including evasion, inference, and poisoning behavior.",
        "Supports separation of adversarial inputs and privacy attacks in the model.",
        "Guides attacker-defender simulation, model adaptation, coefficient constraints, and attack-impact measurement.",
    ], RED, 14)
    add_box(s, 6.85, 1.05, 6.05, 4.85, "Federated Learning Theory", [
        "Explains decentralized training, local updates, and global aggregation without centralizing biometric data.",
        "Supports the global server as an efficiency and coordination parameter.",
        "Guides privacy-utility analysis, communication-aware robustness, and defense propagation.",
    ], BLUE, 14)

    s = new_slide(prs, "Comparison of Privacy-Preserving Models", "How the proposed model extends existing approaches")
    add_box(s, 0.45, 1.05, 3.95, 4.95, "Differential Privacy Model", [
        "Adds noise to gradients or outputs to reduce privacy leakage.",
        "Useful against inference risk, but limited against evasion when used alone.",
        "DO-FMBS keeps privacy protection while modeling adversarial pressure.",
    ], BLUE, 13)
    add_box(s, 4.7, 1.05, 3.95, 4.95, "Adversarial Robust Training", [
        "Uses perturbed inputs during training to improve robustness.",
        "Improves adversarial resilience, but does not fully model privacy trade-offs.",
        "DO-FMBS connects robustness training with privacy-preserving federated learning.",
    ], ORANGE, 13)
    add_box(s, 8.95, 1.05, 3.95, 4.95, "Dynamic Optimization Model", [
        "Models adversarial inputs, privacy attacks, defenses, and global server behavior together.",
        "Uses coefficient constraints and feedback logic for defense-attack trade-offs.",
        "Balances accuracy, privacy, and robustness in a federated multimodal setting.",
    ], GREEN, 13)


def add_picture_box(slide, path, x, y, w, h, title=None):
    box = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = GRAY
    top = y + 0.12
    if title:
        tt = slide.shapes.add_textbox(Inches(x + 0.12), Inches(y + 0.08), Inches(w - 0.24), Inches(0.22))
        p = tt.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(9)
        p.font.bold = True
        p.font.color.rgb = INK
        top = y + 0.38
    full = ROOT / path
    if full.exists():
        slide.shapes.add_picture(str(full), Inches(x + 0.12), Inches(top), width=Inches(w - 0.24), height=Inches(h - (top - y) - 0.12))


def add_equation_card(slide, x, y, w, h, phase, title, equation, satisfied, color):
    box = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = GRAY
    band = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.22))
    band.fill.solid()
    band.fill.fore_color.rgb = color
    band.line.fill.background()

    phase_box = slide.shapes.add_textbox(Inches(x + 0.14), Inches(y + 0.33), Inches(w - 0.28), Inches(0.20))
    p = phase_box.text_frame.paragraphs[0]
    p.text = phase
    p.font.size = Pt(7.5)
    p.font.bold = True
    p.font.color.rgb = MUTED

    title_box = slide.shapes.add_textbox(Inches(x + 0.14), Inches(y + 0.57), Inches(w - 0.28), Inches(0.28))
    p = title_box.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = INK

    eq_box = slide.shapes.add_textbox(Inches(x + 0.14), Inches(y + 0.95), Inches(w - 0.28), Inches(0.58))
    tf = eq_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = equation
    p.font.size = fit_font(equation, w, 0.58, base=12, minimum=8)
    p.font.bold = True
    p.font.color.rgb = color
    p.font.name = "Cambria Math"
    for run in p.runs:
        run.font.size = p.font.size
        run.font.bold = True
        run.font.color.rgb = color
        run.font.name = "Cambria Math"

    sat_box = slide.shapes.add_textbox(Inches(x + 0.14), Inches(y + 1.62), Inches(w - 0.28), Inches(h - 1.78))
    tf = sat_box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    p = tf.paragraphs[0]
    p.text = "How it was met: " + satisfied
    note_size = fit_font(p.text, w, max(0.45, h - 1.78), base=BODY_FONT_PT, minimum=BODY_FONT_MIN_PT)
    p.font.size = note_size
    p.font.color.rgb = TEXT
    for run in p.runs:
        run.font.size = note_size
        run.font.color.rgb = TEXT


def add_bar_chart(slide, x, y, w, h, title, categories, series, ylim=None):
    data = CategoryChartData()
    data.categories = categories
    for name, vals, _ in series:
        data.add_series(name, vals)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    if ylim:
        chart.value_axis.maximum_scale = ylim[1]
        chart.value_axis.minimum_scale = ylim[0]
    for idx, (_, _, color) in enumerate(series):
        chart.series[idx].format.fill.solid()
        chart.series[idx].format.fill.fore_color.rgb = color
    return chart


def add_line_chart(slide, x, y, w, h, title, categories, series, ylim=None):
    data = CategoryChartData()
    data.categories = [str(c) for c in categories]
    for name, vals, _ in series:
        data.add_series(name, vals)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    if ylim:
        chart.value_axis.maximum_scale = ylim[1]
        chart.value_axis.minimum_scale = ylim[0]
    for idx, (_, _, color) in enumerate(series):
        chart.series[idx].format.line.color.rgb = color
    return chart


def main():
    baseline = load_pickle("baseline_metrics.pkl")
    attacks = load_pickle("attack_results.pkl")
    final = load_pickle("final_results.pkl")
    validation = load_pickle("validation_results.pkl")
    face = load_npz("face_data.npz")
    finger = load_npz("fingerprint_data.npz")

    comparison = final["comparison"]
    models = ["baseline", "dp", "art", "do_fmbs"]
    labels = ["Baseline", "DP", "ART", "DO-FMBS"]
    face_acc = [comparison[m]["face_accuracy"] * 100 for m in models]
    privacy = [comparison[m]["privacy_score"] * 100 for m in models]
    attack_success = [comparison[m]["attack_success_rate"] * 100 for m in models]
    fgsm_x = [r["epsilon"] for r in attacks["fgsm_results"]]
    fgsm_y = [r["success_rate"] * 100 for r in attacks["fgsm_results"]]
    pgd_y = [r["success_rate"] * 100 for r in attacks["pgd_results"]]
    member = attacks["membership_inference"]
    poison = attacks["model_poisoning"]
    stat = validation["statistical_test"]
    class_metrics = validation["classification_metrics"]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_research_opening_slides(prs)

    s = new_slide(prs, "DO-FMBS: Sequential Model Design and Results", "A phase-by-phase implementation presentation")
    add_box(s, 0.35, 1.15, 12.6, 2.2, "Presentation scope", [
        "This presentation documents the complete DO-FMBS implementation workflow from raw multimodal biometric data preparation to final model validation.",
        "Each phase shows the input data, the process performed, the generated output, and the evidence produced by the implementation.",
        "The attack-analysis section gives separate graphical evidence for evasion attacks, membership inference, and model poisoning so the security evaluation is visible in the slides.",
        "The numerical results are extracted from the saved implementation artifacts: baseline_metrics.pkl, attack_results.pkl, final_results.pkl, and validation_results.pkl.",
    ], GREEN, 14)
    add_metric(s, 0.65, 3.75, 2.55, 1.05, "Final DO-FMBS accuracy", f"{comparison['do_fmbs']['face_accuracy'] * 100:.2f}%", "Phase 4 comparison", GREEN)
    add_metric(s, 3.55, 3.75, 2.55, 1.05, "Validation mean accuracy", f"{stat['do_fmbs_mean'] * 100:.2f}%", "Phase 5 repeated test", BLUE)
    add_metric(s, 6.45, 3.75, 2.55, 1.05, "Clean validation accuracy", f"{validation['robustness']['do_fmbs'][0]['accuracy'] * 100:.2f}%", "At epsilon 0.00", PURPLE)
    add_metric(s, 9.35, 3.75, 2.55, 1.05, "Statistical significance", "YES", f"p = {stat['p_value']:.2e}", RED)

    s = new_slide(prs, "End-to-End Implementation Flow", "Five-phase DO-FMBS development pipeline")
    implementation_logic = [
        "DO-FMBS was implemented as a sequential pipeline rather than a single model-training activity.",
        "The pipeline begins by preparing multimodal biometric data, then trains a federated baseline model, evaluates the baseline under attacks, introduces privacy and robustness defenses, and validates the defended model against the baseline.",
        "This sequence is important because the attack results provide the technical justification for the defense phase, while the validation phase confirms whether the defended model improves performance and robustness.",
    ]
    add_box(s, 0.35, 1.15, 12.6, 3.0, "Implementation logic", implementation_logic, BLUE, 14)

    s = new_slide(prs, "Mathematical Equations Implemented: Dynamic Optimization Model", "Model formulation used in the study")
    add_equation_card(s, 0.55, 0.95, 5.45, 2.55, "Base model", "Constrained multivariate regression", "Yᵢ = B₀ + B₁ᵢx₁ + B₂ᵢx₂ + B₃ᵢx₃ + B₄x₄ + ε", "System efficiency was evaluated against adversarial inputs, privacy attacks, defense mechanisms, and global server behavior.", BLUE)
    add_equation_card(s, 7.0, 0.95, 5.45, 2.55, "Variables", "Meaning of model inputs", "x₁ = AI,  x₂ = PA,  x₃ = DM,  x₄ = GS", "AI is adversarial input, PA is privacy attack, DM is defense mechanism, and GS is the global server role.", GREEN)
    add_equation_card(s, 0.55, 3.85, 5.45, 2.55, "Phase mapping", "How the model is satisfied", "FMBS efficiency = f(AI, PA, DM, GS)", "Phase 3 covers AI and PA, Phase 4 covers DM, and Phase 2 covers the federated server aggregation role.", ORANGE)
    add_equation_card(s, 7.0, 3.85, 5.45, 2.55, "Residual term", "Unexplained variation", "e = residual model error", "The remaining variation is discussed through robustness, classification, and statistical validation results.", PURPLE)

    s = new_slide(prs, "Mathematical Equations Implemented: Aggregation and Constraints", "Coefficient conditions used to judge model usefulness")
    add_equation_card(s, 0.55, 0.95, 5.45, 2.55, "Aggregated model", "Multiple clients or rounds", "Yᵢ = B₀ + ∑ⁿᵢ₌₁B₁ᵢx₁ + ∑ⁿᵢ₌₁B₂ᵢx₂ + ∑ⁿᵢ₌₁B₃ᵢx₃ + B₄x₄ + ε", "The simulation aggregates results across federated clients and rounds, not from a single isolated client.", BLUE)
    add_equation_card(s, 7.0, 0.95, 5.45, 2.55, "Constraint set", "Required coefficient relationship", "B₁ < B₃;  B₁ < B₄;  B₂ ≤ B₁;  B₂ < B₃;  B₂ < B₄", "The model is useful when defense and server effects are stronger than attack and privacy-risk effects.", RED)
    add_equation_card(s, 0.55, 3.85, 5.45, 2.55, "Defense dominance", "Meaning in the study", "B₃ > B₁  and  B₃ > B₂", "The defense mechanism should contribute more to useful system performance than the attack pressure.", GREEN)
    add_equation_card(s, 7.0, 3.85, 5.45, 2.55, "Server contribution", "Meaning in the study", "B₄ > B₁  and  B₄ > B₂", "The global server should strengthen aggregation while raw biometric data remains decentralized.", PURPLE)

    s = new_slide(prs, "Mathematical Equations Implemented: Evasion Attacks", "Attack equations used during testing")
    add_equation_card(s, 0.55, 0.95, 5.45, 2.55, "FGSM", "Fast Gradient Sign Method", "Xₐdv = X + ε · sign(∇ₓJ(θ, X, y))", "Face features were perturbed in the gradient-sign direction, then the model was checked for identity misclassification.", ORANGE)
    add_equation_card(s, 7.0, 0.95, 5.45, 2.55, "PGD", "Projected Gradient Descent", "Xₐdvᵗ⁺¹ = Πₓ₊ₛ(Xₐdvᵗ + α · sign(∇ₓJ(θ, Xₐdvᵗ, y)))", "Small gradient steps were repeated and projected back into the allowed perturbation range.", RED)
    add_equation_card(s, 0.55, 3.85, 5.45, 2.55, "Attack parameters", "Values used in the project", "ε = 0.01, 0.05, 0.10, 0.20, 0.30;  α = 0.01", "The attack results were reported across these epsilon values to show how vulnerability changed with perturbation strength.", BLUE)
    add_equation_card(s, 7.0, 3.85, 5.45, 2.55, "Attack measurement", "Security interpretation", "Attack success = successful misclassifications / originally correct samples", "This showed how vulnerable the baseline was before defense mechanisms were added.", PURPLE)

    s = new_slide(prs, "Mathematical Equations Implemented: Inference Attack", "Confidence-threshold membership inference used in Phase 3")
    add_equation_card(s, 0.55, 0.95, 5.45, 2.55, "Confidence score", "Maximum prediction probability", "c(x) = max_k p_theta(y = k | x)", "The attacker first measures how confident the baseline model is for each member and non-member sample.", BLUE)
    add_equation_card(s, 7.0, 0.95, 5.45, 2.55, "Membership decision", "Threshold rule", "m_hat(x) = 1 if c(x) >= tau;  m_hat(x) = 0 if c(x) < tau", f"The saved threshold was tau = {member.get('threshold', 0):.2f}; samples above the threshold were inferred as training members.", PURPLE)
    add_equation_card(s, 0.55, 3.85, 5.45, 2.55, "Attack accuracy", "Correct membership guesses", "Attack Accuracy = (1/N) sum_i 1[m_hat_i = m_i]", f"The current saved inference attack accuracy is {pct(member.get('attack_accuracy', 0))}.", GREEN)
    add_equation_card(s, 7.0, 3.85, 5.45, 2.55, "Privacy risk score", "Risk beyond random guessing", "Privacy Risk = max(0, Attack Accuracy - 0.5) x 2", "Because membership inference is a binary decision, 0.5 represents random guessing; only performance above 0.5 is treated as privacy risk.", ORANGE)

    s = new_slide(prs, "Mathematical Equations Implemented: Poisoning Attack", "Label-flipping model poisoning used in Phase 3")
    add_equation_card(s, 0.55, 0.95, 5.45, 2.55, "Poisoned sample count", "Portion of corrupted labels", "n_poison = max(1, floor(rho x N))", f"The saved poison rate was rho = {poison.get('poison_rate', 0):.2f}, producing {poison.get('poisoned_samples', 0)} poisoned samples.", RED)
    add_equation_card(s, 7.0, 0.95, 5.45, 2.55, "Label-flipping rule", "Wrong label assignment", "y'_i = (y_i + 1) mod C", "Selected face and fingerprint labels were shifted to the next class to simulate malicious or corrupted client labels.", PURPLE)
    add_equation_card(s, 0.55, 3.85, 5.45, 2.55, "Poisoned training", "Fine-tuning copied baseline", "theta_poisoned = train(theta_baseline, X, y')", "The baseline was cloned and trained using the poisoned labels, then evaluated on the same clean evaluation split.", BLUE)
    add_equation_card(s, 7.0, 3.85, 5.45, 2.55, "Poisoning impact", "Accuracy degradation", "Accuracy Drop = Acc_clean - Acc_poisoned", f"The saved run dropped from {pct(poison.get('clean_accuracy', 0))} to {pct(poison.get('poisoned_accuracy', 0))}, a drop of {pct(poison.get('accuracy_drop', 0))}.", ORANGE)

    s = new_slide(prs, "Mathematical Equations Implemented: Federated Training", "Training equations used in the baseline model")
    add_equation_card(s, 0.55, 0.95, 5.45, 2.55, "FedAvg", "Federated averaging", "W_globalᵗ⁺¹ = ∑ᴷₖ₌₁(nₖ/n)W_local,kᵗ", "Client updates were weighted by each client's sample count before forming the next global model.", BLUE)
    add_equation_card(s, 7.0, 0.95, 5.45, 2.55, "Loss function", "Sparse cross-entropy", "L = −∑ᴺᵢ₌₁ yᵢ log(ŷᵢ)", "The face and fingerprint output heads minimized classification error during local training.", GREEN)
    add_equation_card(s, 0.55, 3.85, 5.45, 2.55, "Model update", "Gradient-based learning", "W_new = W_old − η∇L", "The optimizer updated model weights in the direction that reduced training loss.", ORANGE)
    add_equation_card(s, 7.0, 3.85, 5.45, 2.55, "Prediction rule", "Class decision", "ŷ = argmax p(y|x)", "The predicted identity was selected from the output class with the highest probability.", PURPLE)

    s = new_slide(prs, "Mathematical Equations Implemented: Privacy and Defense", "Protection equations used in the defended model")
    add_equation_card(s, 0.55, 0.95, 5.45, 2.55, "Differential privacy", "Noisy mechanism", "M(D) = f(D) + N(0, σ²I)", "Gaussian noise was added to reduce information leakage from learned model behavior.", BLUE)
    add_equation_card(s, 7.0, 0.95, 5.45, 2.55, "Noise scale", "Privacy calibration", "σ = Δf · √(2ln(1.25/δ)) / ε", "The privacy noise scale depends on sensitivity, epsilon, and delta.", GREEN)
    add_equation_card(s, 0.55, 3.85, 5.45, 2.55, "Privacy score", "Attack-resistance score", "Privacy Score = 1 − ASR = 1 − SAI/TIA", "Lower successful attack rate produced a higher privacy score.", ORANGE)
    add_equation_card(s, 7.0, 3.85, 5.45, 2.55, "Label poisoning", "Label-flipping rule", "y_poison = (y + 1) mod C", "A controlled portion of labels was flipped to measure model degradation under poisoning.", PURPLE)

    s = new_slide(prs, "Mathematical Equations Implemented: Evaluation Metrics", "Metrics used to report recognition and validation results")
    add_equation_card(s, 0.55, 0.95, 5.45, 2.55, "Accuracy", "Overall correctness", "Accuracy = (TP + TN) / (TP + TN + FP + FN)", "Accuracy summarized the proportion of correct recognition decisions.", BLUE)
    add_equation_card(s, 7.0, 0.95, 5.45, 2.55, "FAR and FRR", "Biometric error rates", "FAR = FP/(FP + TN);  FRR = FN/(TP + FN)", "FAR estimates false acceptance, while FRR estimates false rejection.", RED)
    add_equation_card(s, 0.55, 3.85, 5.45, 2.55, "Precision, recall, F1", "Classification quality", "Precision = TP/(TP+FP);  Recall = TP/(TP+FN);  F1 = 2PR/(P+R)", "Macro precision, recall, and F1-score were used in final classification validation.", GREEN)
    add_equation_card(s, 7.0, 3.85, 5.45, 2.55, "Robustness accuracy", "Accuracy under attack", "Acc(ε) = correct predictions under attack / tested samples", "Baseline and DO-FMBS were compared across increasing FGSM perturbation strengths.", PURPLE)

    s = new_slide(prs, "Mathematical Equations Implemented: Statistical Validation", "Statistical tests used to support the final claim")
    add_equation_card(s, 0.55, 0.95, 5.45, 2.55, "Mean accuracy", "Repeated validation runs", "x̄ = (1/n)∑ⁿᵢ₌₁xᵢ", "Repeated accuracy samples were averaged for both baseline and DO-FMBS.", BLUE)
    add_equation_card(s, 7.0, 0.95, 5.45, 2.55, "Independent t-test", "Significance test", "t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)", "The t-test checked whether the improvement was statistically significant.", ORANGE)
    add_equation_card(s, 0.55, 3.85, 5.45, 2.55, "Effect size", "Cohen's d", "d = (x̄_DO − x̄_base) / s_pooled", "Effect size described how large the improvement was beyond the p-value.", GREEN)
    add_equation_card(s, 7.0, 3.85, 5.45, 2.55, "Accuracy drop", "Poisoning impact", "Drop = Acc_clean − Acc_poisoned", "The poisoning experiment reported how much accuracy was lost after corrupted labels.", RED)

    s = new_slide(prs, "How the Study Was Strengthened")
    add_box(s, 0.7, 1.05, 5.75, 2.35, "Mathematical model", [
        "The regression model, aggregation reformulation, and coefficient constraints are now part of the implementation story.",
        "The equations are connected directly to the phases and validation evidence.",
    ], BLUE, 12)
    add_box(s, 6.9, 1.05, 5.75, 2.35, "Data collection and simulation", [
        "The work proceeded through simulation-based data collection using public face data and synthetic FVC-style fingerprint samples.",
        "Phase 1 documents preprocessing, normalization, train/test split, and non-IID client creation.",
    ], GREEN, 12)
    add_box(s, 0.7, 3.95, 5.75, 2.15, "Attack evaluation", [
        "Adversarial inputs and privacy attacks are answered through FGSM, PGD, membership inference, and poisoning tests.",
        "The attack metrics justify why defense mechanisms are needed.",
    ], ORANGE, 12)
    add_box(s, 6.9, 3.95, 5.75, 2.15, "Model validation", [
        "Validation is answered with robustness testing, classification metrics, and statistical significance.",
        "The deck explains that FAR needs a separate threshold-based biometric verification run.",
    ], PURPLE, 12)

    s = new_slide(prs, "Phase 1: Data Preparation", "Face and fingerprint modalities prepared for federated learning")
    add_metric(s, 0.6, 1.05, 2.3, 0.95, "Face train/test", f"{face['X_train'].shape[0]} / {face['X_test'].shape[0]}", f"{int(face['n_classes'][0])} identities", BLUE)
    add_metric(s, 3.15, 1.05, 2.3, 0.95, "Face features", f"{int(face['n_features'][0])}", "Flattened LFW features", GREEN)
    add_metric(s, 5.7, 1.05, 2.3, 0.95, "Fingerprint train/test", f"{finger['X_train'].shape[0]} / {finger['X_test'].shape[0]}", f"{int(finger['n_classes'][0])} classes", ORANGE)
    add_metric(s, 8.25, 1.05, 2.3, 0.95, "Fingerprint features", f"{int(finger['n_features'][0])}", "FVC-style features", PURPLE)
    add_box(s, 0.6, 2.35, 5.3, 3.8, "Phase 1 process and output", [
        "LFW faces were loaded, flattened, normalized, and split into train and test sets.",
        "Synthetic FVC-style fingerprints were generated and converted into feature vectors.",
        "Prepared samples were distributed across five non-IID federated clients.",
        "Prepared datasets, client-distribution evidence, and sample visuals were produced for the next phase.",
    ], BLUE, 11)
    add_picture_box(s, "do_fmbs_client_distribution.png", 6.25, 2.25, 5.8, 3.95, "Federated client distribution")

    s = new_slide(prs, "Phase 1 Graphics: Multimodal Inputs", "Face and fingerprint samples used by the model")
    add_picture_box(s, "do_fmbs_lfw_faces.png", 0.65, 1.2, 5.8, 5.25, "Face samples from LFW")
    add_picture_box(s, "do_fmbs_fingerprints.png", 6.85, 1.2, 5.8, 5.25, "Synthetic fingerprint samples")

    s = new_slide(prs, "Phase 2: Federated Baseline Learning", "Baseline model trained through client updates and global aggregation")
    add_box(s, 0.65, 1.05, 5.0, 2.3, "Baseline model design", [
        "Each client trained locally using its own face and fingerprint samples.",
        "The global server aggregated model weights without collecting raw biometric records.",
        "Separate face and fingerprint inputs supported multimodal recognition.",
    ], GREEN, 11)
    add_metric(s, 6.05, 1.1, 2.15, 0.95, "Rounds", str(baseline["training_parameters"]["rounds"]), "Federated rounds", BLUE)
    add_metric(s, 8.45, 1.1, 2.15, 0.95, "Local epochs", str(baseline["training_parameters"]["local_epochs"]), "Per client round", PURPLE)
    add_metric(s, 6.05, 2.25, 2.15, 0.95, "Baseline face test", pct(baseline["face_accuracy"]), "Saved metric", GREEN)
    add_metric(s, 8.45, 2.25, 2.15, 0.95, "Baseline FAR", pct(baseline["FAR"]), "Initial estimate", ORANGE)
    add_picture_box(s, "do_fmbs_training_accuracy.png", 1.0, 3.75, 11.3, 2.55, "Training trend across federated rounds")

    s = new_slide(prs, "Phase 2 Process Details: Federated Baseline", "Reference point for attack and defense evaluation")
    add_box(s, 0.55, 1.1, 3.8, 3.0, "Input to Phase 2", [
        "Prepared face_data.npz and fingerprint_data.npz.",
        "Five non-IID client partitions.",
        "Face and fingerprint labels for supervised learning.",
    ], BLUE, 11)
    add_box(s, 4.75, 1.1, 3.8, 3.0, "Training process", [
        "Clients trained locally using their own biometric samples.",
        "The server aggregated updates to create the shared model.",
        "Training ran for 12 federated rounds with 2 local epochs per round.",
    ], GREEN, 11)
    add_box(s, 8.95, 1.1, 3.8, 3.0, "Output of Phase 2", [
        "Saved baseline model: baseline_model.keras.",
        "Saved metrics: baseline_metrics.pkl.",
        "Baseline became the model attacked in Phase 3.",
    ], ORANGE, 11)
    add_box(s, 1.2, 4.6, 10.9, 1.35, "Baseline evidence", [
        f"Face recognition accuracy from Phase 2 was {pct(baseline['face_accuracy'])}; initial FAR was {pct(baseline['FAR'])}; FRR was {pct(baseline['FRR'])}.",
        "These values created the comparison point for attack testing and final validation.",
    ], PURPLE, 12)

    s = new_slide(prs, "Phase 3: Adversarial Attack Simulation", "Baseline attacked before defense construction")
    add_box(s, 0.55, 1.1, 3.8, 2.3, "Evasion attacks", ["FGSM perturbed the face feature vector in one gradient step.", "PGD applied iterative perturbations and represented the stronger evasion test."], ORANGE, 12)
    add_box(s, 4.75, 1.1, 3.8, 2.3, "Inference attack", ["Membership inference tested whether confidence values exposed training membership.", "The saved threshold was 0.70."], PURPLE, 12)
    add_box(s, 8.95, 1.1, 3.8, 2.3, "Poisoning attack", ["Label-flipping poisoning corrupted local training labels.", "The saved run used a 10% poison rate."], RED, 12)
    add_metric(s, 2.0, 4.15, 2.55, 1.05, "FGSM @ ε 0.10", f"{fgsm_y[fgsm_x.index(0.1)]:.2f}%", "Attack success", ORANGE)
    add_metric(s, 5.35, 4.15, 2.55, 1.05, "PGD @ ε 0.10", f"{pgd_y[fgsm_x.index(0.1)]:.2f}%", "Attack success", RED)
    add_metric(s, 8.7, 4.15, 2.55, 1.05, "Membership attack", pct(member["attack_accuracy"]), "Attack accuracy", PURPLE)

    s = new_slide(prs, "Phase 3 Process Details: Attack Implementation", "Each attack tested a different weakness")
    add_box(s, 0.35, 1.05, 6.05, 2.55, "Evasion process", ["Baseline loaded from baseline_model.keras.", "Face test samples converted for gradient calculation.", "FGSM generated one-step perturbations; PGD generated iterative perturbations.", "Success measured originally correct samples becoming misclassified."], ORANGE, 14)
    add_box(s, 6.95, 1.05, 6.0, 2.55, "Inference process", ["Member and non-member samples passed through the baseline.", "Confidence values compared against threshold 0.70.", "Outputs: attack accuracy, true member detection, false member rate, privacy risk."], PURPLE, 14)
    add_box(s, 0.35, 4.05, 6.05, 2.05, "Poisoning process", ["A copy of the baseline was fine-tuned using corrupted labels.", "Poison rate was 10%, equal to 20 poisoned samples.", "The output was accuracy drop after corrupted updates."], RED, 14)
    add_box(s, 6.95, 4.05, 6.0, 2.05, "Phase 3 evidence generated", ["Attack metrics and visual evidence were generated for FGSM, PGD, membership inference, and model poisoning."], BLUE, 14)

    s = new_slide(prs, "Phase 3 Graphics: Evasion Attacks", "FGSM and PGD attack success by epsilon")
    add_line_chart(s, 0.75, 1.25, 7.25, 4.85, "Evasion attack success by epsilon", [str(x) for x in fgsm_x], [("FGSM", fgsm_y, ORANGE), ("PGD", pgd_y, RED)], (0, 110))
    add_box(s, 8.45, 1.35, 4.0, 4.55, "Evasion attack result", [
        "Evasion attacks modified biometric input features at test time.",
        "FGSM used one gradient step; PGD used iterative gradient steps.",
        "At epsilon 0.10, FGSM reached 89.66% success and PGD reached 93.10% success.",
        "The high success rates showed strong baseline vulnerability.",
    ], ORANGE, 11)

    s = new_slide(prs, "Phase 3 Applied Evasion Attack Evidence", "Graphical evidence after applying FGSM and PGD")
    add_box(s, 0.55, 1.05, 3.8, 2.25, "Attack input", [
        "Baseline model from Phase 2.",
        "Face test feature vectors.",
        "Epsilon values: 0.01, 0.05, 0.10, 0.20, and 0.30.",
    ], BLUE, 11)
    add_box(s, 4.75, 1.05, 3.8, 2.25, "Attack operation", [
        "FGSM applied one-step gradient perturbation.",
        "PGD applied repeated projected gradient perturbations.",
        "Each attack attempted to change the predicted identity.",
    ], ORANGE, 11)
    add_box(s, 8.95, 1.05, 3.8, 2.25, "Attack output", [
        f"FGSM success at epsilon 0.10: {fgsm_y[fgsm_x.index(0.1)]:.2f}%.",
        f"PGD success at epsilon 0.10: {pgd_y[fgsm_x.index(0.1)]:.2f}%.",
        "Attack success increased with perturbation strength.",
    ], RED, 11)
    add_line_chart(s, 1.25, 3.75, 10.8, 2.55, "Applied evasion attack results on baseline model", [str(x) for x in fgsm_x], [("FGSM applied", fgsm_y, ORANGE), ("PGD applied", pgd_y, RED)], (0, 110))

    s = new_slide(prs, "Phase 3 Graphics: Membership Inference Attack", "Inference attack metrics from saved run")
    add_bar_chart(s, 0.55, 1.15, 6.65, 4.9, "Membership inference metrics", ["Attack accuracy", "Privacy risk", "True member", "False member", "Member conf.", "Non-member conf."], [("Metrics", [member["attack_accuracy"] * 100, member["privacy_risk_score"] * 100, member["true_member_detection_rate"] * 100, member["false_member_rate"] * 100, member["member_mean_confidence"] * 100, member["nonmember_mean_confidence"] * 100], PURPLE)], (0, 100))
    add_box(s, 7.55, 1.35, 4.9, 4.55, "Inference attack result", [
        "Membership inference tested whether model confidence revealed training membership.",
        f"Attack accuracy was {pct(member['attack_accuracy'])}.",
        f"Privacy risk score was {pct(member['privacy_risk_score'])}.",
        "The selected threshold did not produce a strong membership leak in this run.",
    ], PURPLE, 14)

    s = new_slide(prs, "Phase 3 Applied Inference Attack Evidence", "Graphical evidence after applying membership inference")
    add_box(s, 0.55, 1.05, 3.8, 2.35, "Attack input", [
        "Training-member samples and non-member test samples.",
        "Model confidence scores from the baseline model.",
        f"Membership threshold: {member['threshold']:.2f}.",
    ], BLUE, 11)
    add_box(s, 4.75, 1.05, 3.8, 2.35, "Attack decision rule", [
        "Confidence above threshold is inferred as member.",
        "Confidence below threshold is inferred as non-member.",
        "The attack measures training participation leakage.",
    ], PURPLE, 11)
    add_box(s, 8.95, 1.05, 3.8, 2.35, "Attack output", [
        f"Attack accuracy: {pct(member['attack_accuracy'])}.",
        f"Privacy risk score: {pct(member['privacy_risk_score'])}.",
        f"Member confidence: {pct(member['member_mean_confidence'])}.",
        f"Non-member confidence: {pct(member['nonmember_mean_confidence'])}.",
    ], GREEN, 10.5)
    add_bar_chart(s, 1.25, 3.85, 10.8, 2.45, "Applied membership inference results on baseline model", ["Attack accuracy", "Privacy risk", "Member conf.", "Non-member conf."], [("Applied inference metrics", [member["attack_accuracy"] * 100, member["privacy_risk_score"] * 100, member["member_mean_confidence"] * 100, member["nonmember_mean_confidence"] * 100], PURPLE)], (0, 100))

    s = new_slide(prs, "Phase 3 Graphics: Model Poisoning Attack", "Label-flipping poisoning effect")
    add_bar_chart(s, 0.85, 1.25, 6.6, 4.8, "Label-flipping poisoning result", ["Clean accuracy", "Poisoned accuracy", "Accuracy drop"], [("Poisoning result", [poison["clean_accuracy"] * 100, poison["poisoned_accuracy"] * 100, poison["accuracy_drop"] * 100], RED)], (0, 40))
    add_box(s, 8.0, 1.35, 4.35, 4.55, "Poisoning attack result", [
        f"Poison rate was {pct(poison['poison_rate'], 1)}.",
        f"Poisoned samples: {poison['poisoned_samples']}.",
        f"Accuracy reduced from {pct(poison['clean_accuracy'])} to {pct(poison['poisoned_accuracy'])}.",
        f"Accuracy drop was {pct(poison['accuracy_drop'])}.",
    ], RED, 12)

    s = new_slide(prs, "Phase 3 Combined Attack Evidence", "Attack graphics generated by the implementation")
    add_picture_box(s, "do_fmbs_attack_results.png", 0.65, 1.15, 5.85, 5.25, "FGSM and PGD attack chart")
    add_picture_box(s, "do_fmbs_attack_extended_results.png", 6.85, 1.15, 5.85, 5.25, "Membership inference and poisoning chart")

    s = new_slide(prs, "Phase 4: Defense Mechanism Construction", "Privacy protection and adversarial robustness")
    add_box(s, 0.55, 1.05, 3.8, 2.6, "Differential Privacy", ["Added controlled Gaussian noise during training.", "Reduced sensitive information leakage.", f"Saved epsilon: {final['parameters']['dp_epsilon']}."], BLUE, 12)
    add_box(s, 4.75, 1.05, 3.8, 2.6, "Adversarial Training", ["Used adversarial examples during training.", "Improved recognition stability under perturbation.", f"ART epsilon: {final['parameters']['art_epsilon']}."], ORANGE, 12)
    add_box(s, 8.95, 1.05, 3.8, 2.6, "DO-FMBS", ["Combined DP and ART in the federated multimodal setting.", "Balanced accuracy, privacy, and attack resistance.", f"Saved epochs: {final['parameters']['epochs']}."], GREEN, 12)
    add_picture_box(s, "do_fmbs_comparison_results.png", 2.0, 4.0, 9.3, 2.45, "Phase 4 comparison chart")

    s = new_slide(prs, "Phase 4 Process Details: Defense Training", "Separate and combined protection strategies")
    add_box(s, 0.35, 1.05, 6.05, 2.55, "DP model", ["Introduced Gaussian noise during training.", "Reduced sensitive biometric information exposure.", "Saved epsilon was 0.50."], BLUE, 14)
    add_box(s, 6.95, 1.05, 6.0, 2.55, "ART model", ["Trained with adversarially perturbed examples.", "Reduced sensitivity to adversarial input perturbations.", "Saved epsilon was 0.05."], ORANGE, 14)
    add_box(s, 0.35, 4.05, 6.05, 2.05, "DO-FMBS model", ["Combined Differential Privacy and Adversarial Robustness Training.", "Used the same multimodal federated setting.", "Final model saved as do_fmbs_model.keras."], GREEN, 14)
    add_box(s, 6.95, 4.05, 6.0, 2.05, "Phase 4 comparison logic", ["Baseline, DP, ART, and DO-FMBS were compared using face accuracy, attack success rate, and privacy score.", "This separated accuracy improvement from privacy and robustness behavior."], PURPLE, 14)

    s = new_slide(prs, "Phase 4 Results: Model Comparison", "Accuracy, privacy score, and attack success")
    add_bar_chart(s, 0.65, 1.2, 7.5, 4.95, "Phase 4 comparison", labels, [("Face accuracy", face_acc, GREEN), ("Privacy score", privacy, BLUE), ("Attack success", attack_success, RED)], (0, 80))
    add_box(s, 8.55, 1.35, 3.9, 4.55, "Main findings", [
        f"DO-FMBS achieved the highest face accuracy: {comparison['do_fmbs']['face_accuracy'] * 100:.2f}%.",
        f"ART had the lowest attack success rate: {comparison['art']['attack_success_rate'] * 100:.2f}%.",
        "DO-FMBS is presented as a balanced model that improves accuracy while adding privacy and robustness mechanisms.",
    ], GREEN, 11)

    s = new_slide(prs, "Phase 5: Validation Design", "Robustness, statistics, and classification metrics")
    add_metric(s, 0.65, 1.05, 2.45, 0.95, "Validation samples", str(validation["test_samples"]), "Saved run", BLUE)
    add_metric(s, 3.45, 1.05, 2.45, 0.95, "Baseline mean", pct(stat["baseline_mean"]), f"+/- {stat['baseline_std'] * 100:.2f}", ORANGE)
    add_metric(s, 6.25, 1.05, 2.45, 0.95, "DO-FMBS mean", pct(stat["do_fmbs_mean"]), f"+/- {stat['do_fmbs_std'] * 100:.2f}", GREEN)
    add_metric(s, 9.05, 1.05, 2.45, 0.95, "Effect size", f"{stat['effect_size']:.2f}", "Cohen's d", PURPLE)
    add_box(s, 1.0, 2.6, 11.2, 2.2, "Validation tests performed", [
        "Robustness testing compared baseline and DO-FMBS accuracy under FGSM epsilon values from 0.00 to 0.20.",
        "Statistical validation used repeated accuracy samples to compare the baseline and proposed model.",
        "Classification metrics compared accuracy, macro precision, macro recall, and macro F1-score.",
    ], BLUE, 12)
    add_box(s, 2.1, 5.15, 9.0, 0.95, "Validation conclusion", ["DO-FMBS passed robustness validation and demonstrated statistically significant improvement over the baseline."], GREEN, 13)

    s = new_slide(prs, "Phase 5 Process Details: Validation Outputs", "Final validation outputs")
    add_box(s, 0.35, 1.05, 6.05, 2.55, "Robustness validation", ["Baseline and DO-FMBS were tested at FGSM epsilon 0.00 to 0.20.", "Accuracy and attack success rate recorded at each epsilon.", "The robustness comparison was visualized as graphical evidence."], ORANGE, 14)
    add_box(s, 6.95, 1.05, 6.0, 2.55, "Statistical validation", ["Repeated accuracy samples calculated for both models.", "T-test measured whether improvement was significant.", f"Saved p-value: {stat['p_value']:.2e}."], PURPLE, 14)
    add_box(s, 0.35, 4.05, 6.05, 2.05, "Classification validation", ["Accuracy, macro precision, macro recall, and macro F1-score calculated.", "The final report summarizes classification quality and recognition improvement."], GREEN, 14)
    add_box(s, 6.95, 4.05, 6.0, 2.05, "Phase 5 evidence generated", ["Robustness, statistical, classification, and summary evidence were produced for final validation.", "The report and visuals provide the final validation evidence."], BLUE, 14)

    eps = [r["epsilon"] for r in validation["robustness"]["baseline"]]
    base_acc = [r["accuracy"] * 100 for r in validation["robustness"]["baseline"]]
    do_acc = [r["accuracy"] * 100 for r in validation["robustness"]["do_fmbs"]]
    s = new_slide(prs, "Phase 5 Graphics: Robustness Under FGSM", "DO-FMBS stayed above baseline across tested attack strengths")
    add_line_chart(s, 0.65, 1.15, 7.45, 5.0, "Validation accuracy under FGSM attack", [str(e) for e in eps], [("Baseline", base_acc, RED), ("DO-FMBS", do_acc, GREEN)], (0, 75))
    add_box(s, 8.55, 1.35, 3.9, 4.55, "Validation result", [
        f"Clean input: DO-FMBS {do_acc[0]:.2f}% vs baseline {base_acc[0]:.2f}%.",
        f"At epsilon 0.10: DO-FMBS {do_acc[2]:.2f}% vs baseline {base_acc[2]:.2f}%.",
        f"At epsilon 0.20: baseline {base_acc[-1]:.2f}%, DO-FMBS {do_acc[-1]:.2f}%.",
        "DO-FMBS remained stronger than baseline across all tested FGSM strengths.",
    ], GREEN, 11)

    s = new_slide(prs, "Phase 5 Graphics: Statistics and Classification Metrics", "Final comparison metrics")
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    base_metrics = [class_metrics["baseline"]["accuracy"] * 100, class_metrics["baseline"]["macro_avg_precision"] * 100, class_metrics["baseline"]["macro_avg_recall"] * 100, class_metrics["baseline"]["macro_avg_f1"] * 100]
    do_metrics = [class_metrics["do_fmbs"]["accuracy"] * 100, class_metrics["do_fmbs"]["macro_avg_precision"] * 100, class_metrics["do_fmbs"]["macro_avg_recall"] * 100, class_metrics["do_fmbs"]["macro_avg_f1"] * 100]
    add_bar_chart(s, 0.65, 1.2, 6.2, 4.85, "Classification metrics", metric_labels, [("Baseline", base_metrics, RED), ("DO-FMBS", do_metrics, GREEN)], (0, 75))
    add_picture_box(s, "do_fmbs_validation_statistics.png", 7.25, 1.15, 5.25, 2.35, "Statistical comparison")
    add_picture_box(s, "do_fmbs_validation_summary.png", 7.25, 3.75, 5.25, 2.45, "Validation summary")

    s = new_slide(prs, "Final Model Implementation Summary", "Implementation summary")
    add_box(s, 0.35, 1.08, 12.6, 3.45, "Implementation conclusion", [
        "DO-FMBS was built through five phases: data preparation, federated baseline training, attack simulation, defense construction, and final validation.",
        "Phase 1 produced normalized face and fingerprint datasets distributed across non-IID clients.",
        "Phase 2 trained the baseline; Phase 3 applied FGSM, PGD, membership inference, and poisoning attacks.",
        "Phase 4 introduced DP, ART, and the combined DO-FMBS model.",
        f"Phase 5 confirmed validation mean accuracy improved from {pct(stat['baseline_mean'])} to {pct(stat['do_fmbs_mean'])}.",
    ], GREEN, 15)
    add_box(s, 0.35, 4.76, 12.6, 1.68, "Metric scope", [
        "The current validation focused on classification accuracy, adversarial robustness, statistical significance, and attack behavior.",
        "FAR is a threshold-based biometric verification metric requiring genuine and impostor verification attempts, so it is treated separately from this validation run.",
    ], ORANGE, 16)

    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()

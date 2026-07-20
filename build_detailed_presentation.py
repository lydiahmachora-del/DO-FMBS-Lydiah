import os
import pickle
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "do_fmbs_data"
OUT = ROOT / "DO_FMBS_slide_Presentation.pdf"
SLIDE_COUNT = 0

BG = "#f6f8fb"
INK = "#102a43"
TEXT = "#334e68"
MUTED = "#627d98"
BLUE = "#2f80ed"
GREEN = "#00a896"
ORANGE = "#f59e0b"
RED = "#d64550"
PURPLE = "#7c3aed"
GRAY = "#d8dee9"
BODY_FONT_SIZE = 10.0


def load_pickle(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def load_npz(name):
    return np.load(DATA / name, allow_pickle=True)


def pct(value, decimals=2):
    return f"{value * 100:.{decimals}f}%"


def wrap(text, width=72):
    return "\n".join(textwrap.wrap(str(text), width=width))


def wrapped_lines(text, width):
    return textwrap.wrap(str(text), width=max(16, int(width)))


def add_header(ax, title, subtitle=None, slide_no=None):
    ax.add_patch(Rectangle((0, 0.91), 1, 0.09, color=INK, transform=ax.transAxes))
    ax.text(0.035, 0.957, title, color="white", fontsize=20, fontweight="bold",
            va="center", transform=ax.transAxes)
    if subtitle:
        ax.text(0.035, 0.922, subtitle, color="#d9e2ec", fontsize=9.5,
                va="center", transform=ax.transAxes)
    if slide_no:
        ax.text(0.965, 0.957, str(slide_no), color="#d9e2ec", fontsize=9,
                ha="right", va="center", transform=ax.transAxes)


def add_footer(ax, text=""):
    if not text:
        return
    ax.text(0.035, 0.035, text, color=MUTED, fontsize=7.5, transform=ax.transAxes)
    ax.text(0.965, 0.035, "Lydiah Moraa Machora", color=MUTED, fontsize=7.5,
            ha="right", transform=ax.transAxes)


def add_text_box(ax, x, y, w, h, title, bullets, color=BLUE, font_size=BODY_FONT_SIZE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.012",
                                linewidth=0.9, edgecolor=GRAY, facecolor="white",
                                transform=ax.transAxes))
    ax.add_patch(Rectangle((x, y + h - 0.035), w, 0.035, color=color, transform=ax.transAxes))

    title_size = min(12, max(8.8, 7.5 + w * 6.0))
    title_width = max(18, int(w * 95))
    title_lines = wrapped_lines(title, title_width)
    ax.text(x + 0.018, y + h - 0.054, "\n".join(title_lines), color=INK, fontsize=title_size,
            fontweight="bold", transform=ax.transAxes)

    wrap_width = max(20, int(w * 88))
    prepared = []
    total_lines = 0
    for item in bullets:
        lines = wrapped_lines(item, wrap_width)
        prepared.append(lines)
        total_lines += max(1, len(lines))

    top_gap = 0.075 + 0.024 * max(0, len(title_lines) - 1)
    bottom_gap = 0.018
    bullet_gap = 0.010
    available = max(0.05, h - top_gap - bottom_gap - bullet_gap * max(0, len(prepared) - 1))
    max_font = font_size
    fit_font = min(max_font, available * 540 / max(1, total_lines) / 1.16)
    fit_font = max(5.5, fit_font)
    line_step = fit_font * 1.16 / 540

    yy = y + h - top_gap
    for lines in prepared:
        text = "- " + "\n  ".join(lines)
        ax.text(x + 0.020, yy, text,
                fontsize=fit_font, color=TEXT, va="top", linespacing=1.16,
                transform=ax.transAxes)
        yy -= line_step * len(lines) + bullet_gap


def add_metric(ax, x, y, w, h, label, value, note, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009,rounding_size=0.012",
                                linewidth=0.8, edgecolor=GRAY, facecolor="white",
                                transform=ax.transAxes))
    ax.text(x + 0.015, y + h - 0.032, label, color=TEXT, fontsize=8.2,
            fontweight="bold", transform=ax.transAxes)
    ax.text(x + 0.015, y + 0.043, value, color=color, fontsize=19,
            fontweight="bold", transform=ax.transAxes)
    ax.text(x + 0.015, y + 0.016, note, color=MUTED, fontsize=7.0,
            transform=ax.transAxes)


def add_research_opening_slides(pdf):
    fig, ax = new_slide(
        pdf,
        "Privacy-Preserving Dynamic Optimization Model for FMBS",
        "Mitigating adversarial privacy attacks in federated multimodal biometric systems",
        "01",
    )
    ax.text(0.055, 0.73,
            "A Privacy-Preserving Dynamic Optimization Model for\nFederated Multimodal Biometric Systems",
            fontsize=25, color=INK, fontweight="bold", linespacing=1.1, transform=ax.transAxes)
    add_text_box(ax, 0.055, 0.37, 0.42, 0.24, "Candidate", [
        "Lydiah Moraa Machora",
        "I152/P/9515/23",
        "Master of Science presentation",
    ], BLUE, font_size=12)
    add_text_box(ax, 0.525, 0.37, 0.42, 0.24, "Supervision", [
        "Dr. Richard Omollo, PhD",
        "Dr. Gonza Omoro, PhD",
        "Research direction: privacy, robustness, and federated biometric learning",
    ], GREEN, font_size=12)
    add_text_box(ax, 0.055, 0.14, 0.89, 0.15, "Presentation focus", [
        "The study addresses adversarial privacy attacks, federated learning risks, and defense validation for face and fingerprint biometric modalities.",
    ], ORANGE, font_size=12)
    finish(pdf, fig, ax)

    fig, ax = new_slide(pdf, "Statement of the Problem", "Why the study was necessary", "02")
    add_text_box(ax, 0.035, 0.54, 0.45, 0.27, "Research gap", [
        "Federated Multimodal Biometric Systems support privacy-preserving authentication by keeping sensitive biometric data decentralized.",
        "Federated learning still depends on semi-trusted participants and can expose the system to insider threats and cross-modal vulnerabilities.",
    ], RED, font_size=10.8)
    add_text_box(ax, 0.52, 0.54, 0.445, 0.27, "Adversarial threat", [
        "Attackers can manipulate biometric inputs, infer sensitive information from model behavior, or corrupt model training through poisoned updates.",
        "The major attack categories considered are evasion, membership inference, and model poisoning.",
    ], ORANGE, font_size=10.8)
    add_text_box(ax, 0.035, 0.16, 0.93, 0.27, "Core problem", [
        "The central challenge was the absence of a comprehensive adaptive defense model that can preserve biometric utility, data confidentiality, and robustness under diverse adversarial conditions.",
        "The proposed DO-FMBS model responds by combining dynamic optimization, privacy preservation, adversarial robustness, and federated multimodal learning.",
    ], BLUE, font_size=10.8)
    finish(pdf, fig, ax)

    fig, ax = new_slide(pdf, "Research Objectives", "General and specific objectives guiding the study", "03")
    add_text_box(ax, 0.035, 0.64, 0.93, 0.17, "Main objective", [
        "To develop a privacy-preserving dynamic optimization model for mitigating adversarial attacks in Federated Multimodal Biometric Systems.",
    ], GREEN, font_size=12)
    add_text_box(ax, 0.035, 0.17, 0.45, 0.34, "Specific objectives I", [
        "To examine the types and characteristics of adversarial attacks on Federated Multimodal Biometric Systems.",
        "To assess the impact of adversarial inputs on privacy and performance.",
        "To examine existing privacy-preserving models for Federated Multimodal Biometric Systems.",
    ], BLUE, font_size=10.8)
    add_text_box(ax, 0.52, 0.17, 0.445, 0.34, "Specific objectives II", [
        "To develop a privacy-preserving dynamic optimization model for mitigating adversarial attacks.",
        "To validate the proposed model using robustness, privacy, performance, and statistical evaluation evidence.",
    ], PURPLE, font_size=10.8)
    finish(pdf, fig, ax)

    fig, ax = new_slide(pdf, "Research Questions", "Questions addressed by the implementation and validation", "04")
    add_text_box(ax, 0.035, 0.54, 0.45, 0.27, "Attack and impact questions", [
        "What types and characteristics of adversarial attacks affect Federated Multimodal Biometric Systems?",
        "How do adversarial inputs affect the privacy and performance of Federated Multimodal Biometric Systems?",
    ], ORANGE, font_size=10.8)
    add_text_box(ax, 0.52, 0.54, 0.445, 0.27, "Model development question", [
        "How can a robust privacy-preserving dynamic optimization model be developed for Federated Multimodal Biometric Systems?",
        "Which defense mechanisms can jointly address privacy leakage and adversarial robustness?",
    ], BLUE, font_size=10.8)
    add_text_box(ax, 0.035, 0.16, 0.93, 0.27, "Validation question", [
        "How can the proposed model be validated to demonstrate mitigation against adversarial privacy attacks?",
        "The implementation answers this through attack simulation, defense construction, robustness testing, classification metrics, and statistical validation.",
    ], GREEN, font_size=10.8)
    finish(pdf, fig, ax)

    fig, ax = new_slide(pdf, "Scope of the Study", "Boundaries and evaluation focus", "05")
    add_text_box(ax, 0.035, 0.50, 0.45, 0.31, "Technical scope", [
        "The study focused on privacy attacks against Federated Multimodal Biometric Systems operating in federated learning environments.",
        "It examined evasion, membership inference, and poisoning attacks because these directly affect authentication, privacy, and model integrity.",
    ], BLUE, font_size=10.8)
    add_text_box(ax, 0.52, 0.50, 0.445, 0.31, "Biometric scope", [
        "The multimodal setting used facial recognition and fingerprint biometric modalities.",
        "The implementation relied on publicly available biometric datasets rather than human respondents or field data collection.",
    ], GREEN, font_size=10.8)
    add_text_box(ax, 0.035, 0.17, 0.93, 0.21, "Validation scope", [
        "The model was evaluated through attack impact, privacy-preserving defense behavior, robustness under adversarial inputs, classification performance, and statistical significance.",
    ], PURPLE, font_size=11.2)
    finish(pdf, fig, ax)

    fig, ax = new_slide(pdf, "Theoretical Framework", "The theories supporting model design and analysis", "06")
    add_text_box(ax, 0.035, 0.18, 0.45, 0.62, "Adversarial Machine Learning Theory", [
        "Explains learning under attack, including evasion, inference, and poisoning behavior.",
        "Supports the separation of adversarial inputs and privacy attacks in the model.",
        "Guides attacker-defender simulation, model adaptation, coefficient constraints, and attack-impact quantification.",
    ], RED, font_size=10.8)
    add_text_box(ax, 0.52, 0.18, 0.445, 0.62, "Federated Learning Theory", [
        "Explains decentralized training, local client updates, and global aggregation without centralizing raw biometric data.",
        "Supports the role of the global server as an efficiency and coordination parameter.",
        "Guides privacy-utility analysis, communication-aware robustness, and defense propagation across clients.",
    ], BLUE, font_size=10.8)
    finish(pdf, fig, ax)

    fig, ax = new_slide(pdf, "Comparison of Privacy-Preserving Models", "How the proposed model extends existing approaches", "07")
    add_text_box(ax, 0.035, 0.17, 0.285, 0.63, "Differential Privacy Model", [
        "Adds noise to gradients or outputs to reduce privacy leakage.",
        "Useful against inference risk, but limited against evasion attacks when used alone.",
        "DO-FMBS retains privacy protection while modeling its interaction with adversarial pressure.",
    ], BLUE, font_size=10.0)
    add_text_box(ax, 0.36, 0.17, 0.285, 0.63, "Adversarial Robust Training", [
        "Uses perturbed inputs during training to improve robustness.",
        "Useful for adversarial resilience, but does not fully model privacy trade-offs by itself.",
        "DO-FMBS connects robustness training with privacy-preserving federated learning.",
    ], ORANGE, font_size=10.0)
    add_text_box(ax, 0.685, 0.17, 0.28, 0.63, "Dynamic Optimization Model", [
        "Models adversarial inputs, privacy attacks, defense mechanisms, and global server behavior together.",
        "Uses coefficient constraints and feedback logic to represent defense-attack trade-offs.",
        "Provides a federated multimodal approach for balancing accuracy, privacy, and robustness.",
    ], GREEN, font_size=10.0)
    finish(pdf, fig, ax)


def add_image(ax, path, x, y, w, h, title=None):
    full = ROOT / path
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.01",
                                linewidth=0.8, edgecolor=GRAY, facecolor="white",
                                transform=ax.transAxes))
    if title:
        ax.text(x + 0.012, y + h - 0.03, title, fontsize=9.5, color=INK,
                fontweight="bold", transform=ax.transAxes)
        inner_y = y + 0.015
        inner_h = h - 0.055
    else:
        inner_y = y + 0.015
        inner_h = h - 0.03
    if full.exists():
        img = mpimg.imread(full)
        img_ax = ax.figure.add_axes([x + 0.012, inner_y, w - 0.024, inner_h])
        img_ax.imshow(img)
        img_ax.set_axis_off()
    else:
        ax.text(x + w / 2, y + h / 2, f"Missing image:\n{path}", color=RED,
                ha="center", va="center", fontsize=9, transform=ax.transAxes)


def new_slide(pdf, title, subtitle=None, slide_no=None):
    global SLIDE_COUNT
    SLIDE_COUNT += 1
    fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    add_header(ax, title, subtitle, f"{SLIDE_COUNT:02d}")
    return fig, ax


def finish(pdf, fig, ax, footer=None):
    add_footer(ax, footer or "")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_equation_card(ax, x, y, w, h, phase, title, equation, satisfied, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.012",
                                linewidth=0.9, edgecolor=GRAY, facecolor="white",
                                transform=ax.transAxes))
    ax.add_patch(Rectangle((x, y + h - 0.04), w, 0.04, color=color, transform=ax.transAxes))
    ax.text(x + 0.015, y + h - 0.065, phase, fontsize=8.2, color=MUTED,
            fontweight="bold", transform=ax.transAxes)
    ax.text(x + 0.015, y + h - 0.098, title, fontsize=10.2, color=INK,
            fontweight="bold", transform=ax.transAxes)
    ax.text(x + 0.015, y + h - 0.150, wrap(equation, max(24, int(w * 82))),
            fontsize=8.4, color=color, fontweight="bold", va="top",
            linespacing=1.2, transform=ax.transAxes)
    ax.text(x + 0.015, y + 0.028, wrap("How it was met: " + satisfied, max(25, int(w * 82))),
            fontsize=8.8, color=TEXT, va="bottom", linespacing=1.16,
            transform=ax.transAxes)


def bar_chart(fig, rect, labels, series, title, ylabel="Percent (%)", ylim=None):
    chart = fig.add_axes(rect)
    chart.set_facecolor("white")
    for spine in chart.spines.values():
        spine.set_edgecolor(GRAY)
    x = np.arange(len(labels))
    width = 0.75 / len(series)
    offsets = np.linspace(-0.35 + width / 2, 0.35 - width / 2, len(series))
    for offset, (name, values, color) in zip(offsets, series):
        bars = chart.bar(x + offset, values, width=width, label=name, color=color)
        chart.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1)
    chart.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=8)
    chart.set_xticks(x)
    chart.set_xticklabels(labels, fontsize=8)
    chart.tick_params(axis="y", labelsize=8)
    chart.set_ylabel(ylabel, fontsize=8.5, color=TEXT)
    chart.grid(axis="y", color="#e6ecf2", linewidth=0.8)
    if ylim:
        chart.set_ylim(*ylim)
    chart.legend(frameon=False, fontsize=7.5, loc="upper left")
    return chart


def line_chart(fig, rect, x_values, series, title, ylabel="Percent (%)", ylim=None):
    chart = fig.add_axes(rect)
    chart.set_facecolor("white")
    for spine in chart.spines.values():
        spine.set_edgecolor(GRAY)
    for name, values, color, marker in series:
        chart.plot(x_values, values, marker=marker, linewidth=2.2, label=name, color=color)
        for x, y in zip(x_values, values):
            chart.text(x, y + 2.2, f"{y:.1f}", fontsize=6.5, ha="center", color=color)
    chart.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=8)
    chart.set_xlabel("Attack strength / epsilon", fontsize=8.5, color=TEXT)
    chart.set_ylabel(ylabel, fontsize=8.5, color=TEXT)
    chart.tick_params(labelsize=8)
    chart.grid(color="#e6ecf2", linewidth=0.8)
    if ylim:
        chart.set_ylim(*ylim)
    chart.legend(frameon=False, fontsize=8)
    return chart


def main():
    global SLIDE_COUNT
    SLIDE_COUNT = 0
    baseline = load_pickle("baseline_metrics.pkl")
    attacks = load_pickle("attack_results.pkl")
    final = load_pickle("final_results.pkl")
    validation = load_pickle("validation_results.pkl")
    face = load_npz("face_data.npz")
    finger = load_npz("fingerprint_data.npz")

    comparison = final["comparison"]
    models = ["baseline", "dp", "art", "do_fmbs"]
    model_labels = ["Baseline", "DP", "ART", "DO-FMBS"]
    face_acc = [comparison[m]["face_accuracy"] * 100 for m in models]
    privacy = [comparison[m]["privacy_score"] * 100 for m in models]
    attack_success = [comparison[m]["attack_success_rate"] * 100 for m in models]

    fgsm_x = [r["epsilon"] for r in attacks["fgsm_results"]]
    fgsm_y = [r["success_rate"] * 100 for r in attacks["fgsm_results"]]
    pgd_x = [r["epsilon"] for r in attacks["pgd_results"]]
    pgd_y = [r["success_rate"] * 100 for r in attacks["pgd_results"]]
    member = attacks["membership_inference"]
    poison = attacks["model_poisoning"]
    stat = validation["statistical_test"]
    class_metrics = validation["classification_metrics"]

    with PdfPages(OUT) as pdf:
        add_research_opening_slides(pdf)

        fig, ax = new_slide(pdf, "DO-FMBS: Sequential Model Design and Results",
                            "A phase-by-phase presentation of how the model was built, attacked, defended, and validated",
                            "01")
        ax.text(0.055, 0.78, "Dynamic-Optimization Model for Federated Multimodal Biometric Systems",
                fontsize=23, color=INK, fontweight="bold", transform=ax.transAxes)
        add_metric(ax, 0.055, 0.46, 0.21, 0.12, "Final DO-FMBS accuracy",
                   f"{comparison['do_fmbs']['face_accuracy'] * 100:.2f}%",
                   "Phase 4 comparison", GREEN)
        add_metric(ax, 0.285, 0.46, 0.21, 0.12, "Validation mean accuracy",
                   f"{stat['do_fmbs_mean'] * 100:.2f}%",
                   "Phase 5 repeated test", BLUE)
        add_metric(ax, 0.515, 0.46, 0.21, 0.12, "Clean validation accuracy",
                   f"{validation['robustness']['do_fmbs'][0]['accuracy'] * 100:.2f}%",
                   "At epsilon 0.00", PURPLE)
        add_metric(ax, 0.745, 0.46, 0.21, 0.12, "Statistical significance",
                   "YES" if stat["is_significant"] else "NO",
                   f"p = {stat['p_value']:.2e}", RED if stat["is_significant"] else ORANGE)
        add_text_box(ax, 0.035, 0.13, 0.93, 0.29, "Presentation scope", [
            "This presentation documents the complete DO-FMBS implementation workflow from raw multimodal biometric data preparation to final model validation.",
            "Each phase shows the input data, the process performed, the generated output, and the evidence produced by the implementation.",
            "The attack-analysis section gives separate graphical evidence for evasion attacks, membership inference, and model poisoning so the security evaluation is visible in the slides.",
            "The numerical results are extracted from the saved implementation artifacts: baseline_metrics.pkl, attack_results.pkl, final_results.pkl, and validation_results.pkl.",
        ], GREEN, font_size=10.5)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "End-to-End Implementation Flow",
                            "The model was developed as a five-phase pipeline", "02")
        phases = [
            ("Phase 1", "Data preparation", "Face + fingerprint data, preprocessing, non-IID clients"),
            ("Phase 2", "Federated baseline", "Client training + global aggregation without raw data sharing"),
            ("Phase 3", "Attack simulation", "FGSM, PGD, membership inference, label-flipping poisoning"),
            ("Phase 4", "Defense mechanisms", "DP, ART, and combined DO-FMBS"),
            ("Phase 5", "Validation", "Robustness, statistical testing, classification metrics"),
        ]
        x0, y0, w, h, gap = 0.055, 0.54, 0.165, 0.22, 0.025
        colors = [BLUE, GREEN, ORANGE, PURPLE, RED]
        for i, (phase, title, body) in enumerate(phases):
            x = x0 + i * (w + gap)
            ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.01,rounding_size=0.012",
                                        linewidth=1.0, edgecolor=GRAY, facecolor="white",
                                        transform=ax.transAxes))
            ax.add_patch(Rectangle((x, y0 + h - 0.04), w, 0.04, color=colors[i], transform=ax.transAxes))
            ax.text(x + 0.012, y0 + h - 0.066, phase, fontsize=10, color=INK,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(x + 0.012, y0 + h - 0.102, title, fontsize=9.5, color=TEXT,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(x + 0.012, y0 + h - 0.135, wrap(body, 24), fontsize=7.2,
                    color=TEXT, va="top", linespacing=1.2, transform=ax.transAxes)
            if i < len(phases) - 1:
                ax.add_patch(FancyArrowPatch((x + w + 0.003, y0 + h / 2),
                                             (x + w + gap - 0.004, y0 + h / 2),
                                             arrowstyle="-|>", mutation_scale=12,
                                             linewidth=1.5, color=MUTED,
                                             transform=ax.transAxes))
        add_text_box(ax, 0.035, 0.12, 0.93, 0.26, "Implementation logic", [
            "DO-FMBS was implemented as a sequential pipeline rather than a single model-training activity.",
            "The pipeline begins by preparing multimodal biometric data, then trains a federated baseline model, evaluates the baseline under attacks, introduces privacy and robustness defenses, and validates the defended model against the baseline.",
            "This sequence is important because the attack results provide the technical justification for the defense phase, while the validation phase confirms whether the defended model improves performance and robustness.",
        ], BLUE, font_size=10.5)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Mathematical Equations Implemented: Dynamic Optimization Model",
                            "Model formulation used in the study", "03")
        add_equation_card(ax, 0.055, 0.48, 0.41, 0.32, "Base model", "Constrained multivariate regression",
                          "Yᵢ = B₀ + B₁ᵢx₁ + B₂ᵢx₂ + B₃ᵢx₃ + B₄x₄ + ε",
                          "System efficiency was evaluated against adversarial inputs, privacy attacks, defense mechanisms, and global server behavior.",
                          BLUE)
        add_equation_card(ax, 0.545, 0.48, 0.41, 0.32, "Variables", "Meaning of model inputs",
                          "x₁ = AI,  x₂ = PA,  x₃ = DM,  x₄ = GS",
                          "AI is adversarial input, PA is privacy attack, DM is defense mechanism, and GS is the global server role.",
                          GREEN)
        add_equation_card(ax, 0.055, 0.12, 0.41, 0.32, "Phase mapping", "How the model is satisfied",
                          "FMBS efficiency = f(AI, PA, DM, GS)",
                          "Phase 3 covers AI and PA, Phase 4 covers DM, and Phase 2 covers the federated server aggregation role.",
                          ORANGE)
        add_equation_card(ax, 0.545, 0.12, 0.41, 0.32, "Residual term", "Unexplained variation",
                          "e = residual model error",
                          "The remaining variation is discussed through robustness, classification, and statistical validation results.",
                          PURPLE)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Mathematical Equations Implemented: Aggregation and Constraints",
                            "Coefficient conditions used to judge model usefulness", "04")
        add_equation_card(ax, 0.055, 0.48, 0.41, 0.32, "Aggregated model", "Multiple clients or rounds",
                          "Yᵢ = B₀ + ∑ⁿᵢ₌₁B₁ᵢx₁ + ∑ⁿᵢ₌₁B₂ᵢx₂ + ∑ⁿᵢ₌₁B₃ᵢx₃ + B₄x₄ + ε",
                          "The simulation aggregates results across federated clients and rounds, not from a single isolated client.",
                          BLUE)
        add_equation_card(ax, 0.545, 0.48, 0.41, 0.32, "Constraint set", "Required coefficient relationship",
                          "B₁ < B₃;  B₁ < B₄;  B₂ ≤ B₁;  B₂ < B₃;  B₂ < B₄",
                          "The model is useful when defense and server effects are stronger than attack and privacy-risk effects.",
                          RED)
        add_equation_card(ax, 0.055, 0.12, 0.41, 0.32, "Defense dominance", "Meaning in the study",
                          "B₃ > B₁  and  B₃ > B₂",
                          "The defense mechanism should contribute more to useful system performance than the attack pressure.",
                          GREEN)
        add_equation_card(ax, 0.545, 0.12, 0.41, 0.32, "Server contribution", "Meaning in the study",
                          "B₄ > B₁  and  B₄ > B₂",
                          "The global server should strengthen aggregation while raw biometric data remains decentralized.",
                          PURPLE)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Mathematical Equations Implemented: Evasion Attacks",
                            "Attack equations used during testing", "05")
        add_equation_card(ax, 0.055, 0.48, 0.41, 0.32, "FGSM", "Fast Gradient Sign Method",
                          "Xₐdv = X + ε · sign(∇ₓJ(θ, X, y))",
                          "Face features were perturbed in the gradient-sign direction, then the model was checked for identity misclassification.",
                          ORANGE)
        add_equation_card(ax, 0.545, 0.48, 0.41, 0.32, "PGD", "Projected Gradient Descent",
                          "Xₐdvᵗ⁺¹ = Πₓ₊ₛ(Xₐdvᵗ + α · sign(∇ₓJ(θ, Xₐdvᵗ, y)))",
                          "Small gradient steps were repeated and projected back into the allowed perturbation range.",
                          RED)
        add_equation_card(ax, 0.055, 0.12, 0.41, 0.32, "Attack parameters", "Values used in the project",
                          "ε = 0.01, 0.05, 0.10, 0.20, 0.30;  α = 0.01",
                          "The attack results were reported across these epsilon values to show how vulnerability changed with perturbation strength.",
                          BLUE)
        add_equation_card(ax, 0.545, 0.12, 0.41, 0.32, "Attack measurement", "Security interpretation",
                          "Attack success = successful misclassifications / originally correct samples",
                          "This showed how vulnerable the baseline was before defense mechanisms were added.",
                          PURPLE)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Mathematical Equations Implemented: Inference Attack",
                            "Confidence-threshold membership inference used in Phase 3", "06")
        add_equation_card(ax, 0.055, 0.48, 0.41, 0.32, "Confidence score", "Maximum prediction probability",
                          "c(x) = max_k p_theta(y = k | x)",
                          "The attacker first measures how confident the baseline model is for each member and non-member sample.",
                          BLUE)
        add_equation_card(ax, 0.545, 0.48, 0.41, 0.32, "Membership decision", "Threshold rule",
                          "m_hat(x) = 1 if c(x) >= tau;  m_hat(x) = 0 if c(x) < tau",
                          f"The saved threshold was tau = {member.get('threshold', 0):.2f}; samples above the threshold were inferred as training members.",
                          PURPLE)
        add_equation_card(ax, 0.055, 0.12, 0.41, 0.32, "Attack accuracy", "Correct membership guesses",
                          "Attack Accuracy = (1/N) sum_i 1[m_hat_i = m_i]",
                          f"The current saved inference attack accuracy is {pct(member.get('attack_accuracy', 0))}.",
                          GREEN)
        add_equation_card(ax, 0.545, 0.12, 0.41, 0.32, "Privacy risk score", "Risk beyond random guessing",
                          "Privacy Risk = max(0, Attack Accuracy - 0.5) x 2",
                          "Because membership inference is a binary decision, 0.5 represents random guessing; only performance above 0.5 is treated as privacy risk.",
                          ORANGE)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Mathematical Equations Implemented: Poisoning Attack",
                            "Label-flipping model poisoning used in Phase 3", "07")
        add_equation_card(ax, 0.055, 0.48, 0.41, 0.32, "Poisoned sample count", "Portion of corrupted labels",
                          "n_poison = max(1, floor(rho x N))",
                          f"The saved poison rate was rho = {poison.get('poison_rate', 0):.2f}, producing {poison.get('poisoned_samples', 0)} poisoned samples.",
                          RED)
        add_equation_card(ax, 0.545, 0.48, 0.41, 0.32, "Label-flipping rule", "Wrong label assignment",
                          "y'_i = (y_i + 1) mod C",
                          "Selected face and fingerprint labels were shifted to the next class to simulate malicious or corrupted client labels.",
                          PURPLE)
        add_equation_card(ax, 0.055, 0.12, 0.41, 0.32, "Poisoned training", "Fine-tuning copied baseline",
                          "theta_poisoned = train(theta_baseline, X, y')",
                          "The baseline was cloned and trained using the poisoned labels, then evaluated on the same clean evaluation split.",
                          BLUE)
        add_equation_card(ax, 0.545, 0.12, 0.41, 0.32, "Poisoning impact", "Accuracy degradation",
                          "Accuracy Drop = Acc_clean - Acc_poisoned",
                          f"The saved run dropped from {pct(poison.get('clean_accuracy', 0))} to {pct(poison.get('poisoned_accuracy', 0))}, a drop of {pct(poison.get('accuracy_drop', 0))}.",
                          ORANGE)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Mathematical Equations Implemented: Federated Training",
                            "Training equations used in the baseline model", "06")
        add_equation_card(ax, 0.055, 0.48, 0.41, 0.32, "FedAvg", "Federated averaging",
                          "W_globalᵗ⁺¹ = ∑ᴷₖ₌₁(nₖ/n)W_local,kᵗ",
                          "Client updates were weighted by each client's sample count before forming the next global model.",
                          BLUE)
        add_equation_card(ax, 0.545, 0.48, 0.41, 0.32, "Loss function", "Sparse cross-entropy",
                          "L = −∑ᴺᵢ₌₁ yᵢ log(ŷᵢ)",
                          "The face and fingerprint output heads minimized classification error during local training.",
                          GREEN)
        add_equation_card(ax, 0.055, 0.12, 0.41, 0.32, "Model update", "Gradient-based learning",
                          "W_new = W_old − η∇L",
                          "The optimizer updated model weights in the direction that reduced training loss.",
                          ORANGE)
        add_equation_card(ax, 0.545, 0.12, 0.41, 0.32, "Prediction rule", "Class decision",
                          "ŷ = argmax p(y|x)",
                          "The predicted identity was selected from the output class with the highest probability.",
                          PURPLE)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Mathematical Equations Implemented: Privacy and Defense",
                            "Protection equations used in the defended model", "07")
        add_equation_card(ax, 0.055, 0.48, 0.41, 0.32, "Differential privacy", "Noisy mechanism",
                          "M(D) = f(D) + N(0, σ²I)",
                          "Gaussian noise was added to reduce information leakage from learned model behavior.",
                          BLUE)
        add_equation_card(ax, 0.545, 0.48, 0.41, 0.32, "Noise scale", "Privacy calibration",
                          "σ = Δf · √(2ln(1.25/δ)) / ε",
                          "The privacy noise scale depends on sensitivity, epsilon, and delta.",
                          GREEN)
        add_equation_card(ax, 0.055, 0.12, 0.41, 0.32, "Privacy score", "Attack-resistance score",
                          "Privacy Score = 1 − ASR = 1 − SAI/TIA",
                          "Lower successful attack rate produced a higher privacy score.",
                          ORANGE)
        add_equation_card(ax, 0.545, 0.12, 0.41, 0.32, "Label poisoning", "Label-flipping rule",
                          "y_poison = (y + 1) mod C",
                          "A controlled portion of labels was flipped to measure model degradation under poisoning.",
                          PURPLE)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Mathematical Equations Implemented: Evaluation Metrics",
                            "Metrics used to report recognition and validation results", "08")
        add_equation_card(ax, 0.055, 0.48, 0.41, 0.32, "Accuracy", "Overall correctness",
                          "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
                          "Accuracy summarized the proportion of correct recognition decisions.",
                          BLUE)
        add_equation_card(ax, 0.545, 0.48, 0.41, 0.32, "FAR and FRR", "Biometric error rates",
                          "FAR = FP/(FP + TN);  FRR = FN/(TP + FN)",
                          "FAR estimates false acceptance, while FRR estimates false rejection.",
                          RED)
        add_equation_card(ax, 0.055, 0.12, 0.41, 0.32, "Precision, recall, F1", "Classification quality",
                          "Precision = TP/(TP+FP);  Recall = TP/(TP+FN);  F1 = 2PR/(P+R)",
                          "Macro precision, recall, and F1-score were used in final classification validation.",
                          GREEN)
        add_equation_card(ax, 0.545, 0.12, 0.41, 0.32, "Robustness accuracy", "Accuracy under attack",
                          "Acc(ε) = correct predictions under attack / tested samples",
                          "Baseline and DO-FMBS were compared across increasing FGSM perturbation strengths.",
                          PURPLE)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Mathematical Equations Implemented: Statistical Validation",
                            "Statistical tests used to support the final claim", "09")
        add_equation_card(ax, 0.055, 0.48, 0.41, 0.32, "Mean accuracy", "Repeated validation runs",
                          "x̄ = (1/n)∑ⁿᵢ₌₁xᵢ",
                          "Repeated accuracy samples were averaged for both baseline and DO-FMBS.",
                          BLUE)
        add_equation_card(ax, 0.545, 0.48, 0.41, 0.32, "Independent t-test", "Significance test",
                          "t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)",
                          "The t-test checked whether the improvement was statistically significant.",
                          ORANGE)
        add_equation_card(ax, 0.055, 0.12, 0.41, 0.32, "Effect size", "Cohen's d",
                          "d = (x̄_DO − x̄_base) / s_pooled",
                          "Effect size described how large the improvement was beyond the p-value.",
                          GREEN)
        add_equation_card(ax, 0.545, 0.12, 0.41, 0.32, "Accuracy drop", "Poisoning impact",
                          "Drop = Acc_clean − Acc_poisoned",
                          "The poisoning experiment reported how much accuracy was lost after corrupted labels.",
                          RED)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "How the Study Was Strengthened", None, "10")
        add_text_box(ax, 0.055, 0.49, 0.41, 0.31, "Mathematical model", [
            "The regression model, aggregation reformulation, and coefficient constraints are now part of the implementation story.",
            "The equations are connected directly to the phases and validation evidence.",
        ], BLUE, font_size=12)
        add_text_box(ax, 0.545, 0.49, 0.41, 0.31, "Data collection and simulation", [
            "The work proceeded through simulation-based data collection using public face data and synthetic FVC-style fingerprint samples.",
            "Phase 1 documents preprocessing, normalization, train/test split, and non-IID client creation.",
        ], GREEN, font_size=12)
        add_text_box(ax, 0.055, 0.13, 0.41, 0.28, "Attack evaluation", [
            "Adversarial inputs and privacy attacks are answered through FGSM, PGD, membership inference, and poisoning tests.",
            "The attack metrics justify why defense mechanisms are needed.",
        ], ORANGE, font_size=12)
        add_text_box(ax, 0.545, 0.13, 0.41, 0.28, "Model validation", [
            "Validation is answered with robustness testing, classification metrics, and statistical significance.",
            "The deck explains that FAR needs a separate threshold-based biometric verification run.",
        ], PURPLE, font_size=12)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 1: Data Preparation",
                            "Face and fingerprint modalities were prepared for federated multimodal learning", "03")
        add_metric(ax, 0.055, 0.72, 0.2, 0.11, "Face train/test",
                   f"{face['X_train'].shape[0]} / {face['X_test'].shape[0]}",
                   f"{int(face['n_classes'][0])} identities", BLUE)
        add_metric(ax, 0.285, 0.72, 0.2, 0.11, "Face features",
                   f"{int(face['n_features'][0])}",
                   "Flattened LFW image features", GREEN)
        add_metric(ax, 0.515, 0.72, 0.2, 0.11, "Fingerprint train/test",
                   f"{finger['X_train'].shape[0]} / {finger['X_test'].shape[0]}",
                   f"{int(finger['n_classes'][0])} classes", ORANGE)
        add_metric(ax, 0.745, 0.72, 0.2, 0.11, "Fingerprint features",
                   f"{int(finger['n_features'][0])}",
                   "Synthetic FVC-style images", PURPLE)
        add_text_box(ax, 0.055, 0.16, 0.41, 0.46, "Phase 1 process and output", [
            "Input 1: LFW face images were loaded, resized, flattened into feature vectors, normalized, and separated into training and testing sets.",
            "Input 2: synthetic FVC-style fingerprint samples were generated, converted into numerical feature vectors, and split into training and testing sets.",
            "Federated setup: the prepared samples were partitioned across five non-IID clients to represent decentralized biometric data ownership.",
            "Output: prepared face data, prepared fingerprint data, client-distribution evidence, and sample visuals were produced for the next phase.",
        ], BLUE)
        add_image(ax, "do_fmbs_client_distribution.png", 0.50, 0.16, 0.445, 0.46,
                  "Federated client distribution")
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 1 Graphics: Multimodal Inputs",
                            "The model uses two biometric modalities rather than a single source", "04")
        add_image(ax, "do_fmbs_lfw_faces.png", 0.055, 0.18, 0.43, 0.62,
                  "Face samples from LFW")
        add_image(ax, "do_fmbs_fingerprints.png", 0.515, 0.18, 0.43, 0.62,
                  "Synthetic fingerprint samples")
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 2: Federated Baseline Learning",
                            "A baseline multimodal model was trained through client updates and global aggregation", "05")
        add_text_box(ax, 0.055, 0.49, 0.39, 0.31, "Baseline model design", [
            "Each client trains locally on its available face and fingerprint samples.",
            "The server aggregates model weights, which preserves the federated design because raw biometric records are not centrally shared.",
            "The architecture has separate face and fingerprint inputs, learns modality-specific features, and outputs recognition predictions.",
        ], GREEN)
        add_metric(ax, 0.49, 0.69, 0.19, 0.11, "Rounds",
                   str(baseline["training_parameters"]["rounds"]), "Federated rounds", BLUE)
        add_metric(ax, 0.70, 0.69, 0.19, 0.11, "Local epochs",
                   str(baseline["training_parameters"]["local_epochs"]), "Per client per round", PURPLE)
        add_metric(ax, 0.49, 0.54, 0.19, 0.11, "Baseline face test",
                   pct(baseline["face_accuracy"]), "Saved Phase 2 metric", GREEN)
        add_metric(ax, 0.70, 0.54, 0.19, 0.11, "Baseline FAR",
                   pct(baseline["FAR"]), "Initial FAR estimate", ORANGE)
        add_image(ax, "do_fmbs_training_accuracy.png", 0.08, 0.10, 0.82, 0.34,
                  "Training trend across federated rounds")
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 2 Process Details: Federated Baseline",
                            "The baseline model established the reference point for later attack and defense evaluation", "06")
        add_text_box(ax, 0.055, 0.50, 0.28, 0.30, "Input to Phase 2", [
            "Prepared face_data.npz and fingerprint_data.npz from Phase 1.",
            "Five non-IID client partitions representing decentralized biometric data.",
            "Face and fingerprint labels used for supervised multimodal recognition.",
        ], BLUE)
        add_text_box(ax, 0.365, 0.50, 0.28, 0.30, "Training process", [
            "Each client trained the multimodal model locally using its own face and fingerprint samples.",
            "The global server collected model updates and aggregated weights to create a shared baseline model.",
            "Training was repeated for 12 federated rounds with 2 local epochs per round.",
        ], GREEN)
        add_text_box(ax, 0.675, 0.50, 0.28, 0.30, "Output of Phase 2", [
            "Saved baseline model: baseline_model.keras.",
            "Saved metrics: baseline_metrics.pkl.",
            "Training-progress evidence was generated for the federated rounds.",
            "The baseline became the model attacked in Phase 3 and compared against in Phase 5.",
        ], ORANGE)
        add_text_box(ax, 0.09, 0.18, 0.82, 0.20, "Baseline evidence", [
            f"Face recognition accuracy from the Phase 2 saved metrics was {pct(baseline['face_accuracy'])}.",
            f"The initial FAR estimate in the baseline metrics was {pct(baseline['FAR'])}, while FRR was {pct(baseline['FRR'])}.",
            "These baseline values were not treated as the final proposed result; they were used as the reference point for attack testing and later validation.",
        ], PURPLE)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 3: Adversarial Attack Simulation",
                            "The baseline was deliberately attacked before defenses were introduced", "06")
        add_text_box(ax, 0.055, 0.51, 0.27, 0.29, "Evasion attacks", [
            "FGSM perturbs the face feature vector in one gradient step.",
            "PGD applies iterative perturbations and is treated as the stronger evasion test.",
        ], ORANGE)
        add_text_box(ax, 0.365, 0.51, 0.27, 0.29, "Inference attack", [
            "Membership inference checks whether model confidence can reveal if a sample belonged to training data.",
            "The saved threshold was 0.70.",
        ], PURPLE)
        add_text_box(ax, 0.675, 0.51, 0.27, 0.29, "Poisoning attack", [
            "Label-flipping poisoning corrupts part of the local training data.",
            "The saved run used a 10% poison rate.",
        ], RED)
        add_metric(ax, 0.13, 0.29, 0.2, 0.11, "FGSM @ ε 0.10",
                   f"{fgsm_y[fgsm_x.index(0.1)]:.2f}%", "Attack success", ORANGE)
        add_metric(ax, 0.40, 0.29, 0.2, 0.11, "PGD @ ε 0.10",
                   f"{pgd_y[pgd_x.index(0.1)]:.2f}%", "Attack success", RED)
        add_metric(ax, 0.67, 0.29, 0.2, 0.11, "Membership attack",
                   pct(member["attack_accuracy"]), "Attack accuracy", PURPLE)
        add_text_box(ax, 0.16, 0.10, 0.68, 0.12, "Why Phase 3 matters", [
            "This phase proves that the baseline is vulnerable enough to justify privacy and robustness defenses in Phase 4.",
        ], BLUE, font_size=8)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 3 Process Details: Attack Implementation",
                            "Each attack type tested a different weakness in federated biometric learning", "08")
        add_text_box(ax, 0.035, 0.49, 0.445, 0.31, "Evasion attack process", [
            "The trained baseline model was loaded from baseline_model.keras.",
            "Face test samples were selected and converted to tensors for gradient calculation.",
            "FGSM generated one-step perturbations; PGD generated iterative perturbations.",
            "Attack success was measured as the proportion of originally correct samples that became misclassified.",
        ], ORANGE, font_size=10.5)
        add_text_box(ax, 0.520, 0.49, 0.445, 0.31, "Inference attack process", [
            "Member samples and non-member samples were passed through the baseline model.",
            "The model confidence values were compared against a membership threshold of 0.70.",
            "The attack estimated whether confidence could reveal that a sample was part of the training set.",
            "The output metrics were attack accuracy, true member detection, false member rate, and privacy risk.",
        ], PURPLE, font_size=10.5)
        add_text_box(ax, 0.035, 0.16, 0.445, 0.25, "Poisoning attack process", [
            "A copy of the baseline model was fine-tuned using corrupted local labels.",
            "The saved poison rate was 10%, corresponding to 20 poisoned samples in the run.",
            "The poisoned model was evaluated against the clean baseline evaluation split.",
            "The main output was accuracy drop after corrupted training updates.",
        ], RED, font_size=10.5)
        add_text_box(ax, 0.520, 0.16, 0.445, 0.25, "Phase 3 evidence generated", [
            "The attack phase stored the metrics for FGSM, PGD, membership inference, and model poisoning.",
            "The evasion chart visualized FGSM and PGD results.",
            "The inference and poisoning chart visualized membership inference and label-flipping effects.",
        ], BLUE, font_size=10.5)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 3 Graphics: Evasion Attacks",
                            "FGSM and PGD attack success increased as perturbation strength increased", "07")
        line_chart(fig, [0.08, 0.17, 0.58, 0.58], fgsm_x, [
            ("FGSM success", fgsm_y, ORANGE, "o"),
            ("PGD success", pgd_y, RED, "s"),
        ], "Evasion attack success by epsilon", "Attack success (%)", (0, 110))
        add_text_box(ax, 0.70, 0.18, 0.25, 0.57, "Evasion attack result", [
            "Evasion attacks modify the biometric input features at testing time so that the trained model predicts the wrong identity.",
            "FGSM produced a one-step gradient perturbation, while PGD applied iterative gradient-based perturbations and therefore represented a stronger evasion test.",
            "At epsilon 0.10, FGSM reached 89.66% success and PGD reached 93.10% success.",
            "The high success rates show that the undefended baseline was highly vulnerable to adversarial biometric inputs.",
        ], ORANGE)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 3 Applied Evasion Attack Evidence",
                            "Graphical evidence after applying FGSM and PGD to the baseline model", "08")
        add_text_box(ax, 0.055, 0.53, 0.27, 0.27, "Attack input", [
            "Baseline model from Phase 2.",
            "Face test feature vectors.",
            "Epsilon values: 0.01, 0.05, 0.10, 0.20, and 0.30.",
        ], BLUE)
        add_text_box(ax, 0.365, 0.53, 0.27, 0.27, "Attack operation", [
            "FGSM applied one-step gradient perturbation.",
            "PGD applied repeated projected gradient perturbations.",
            "Each attack attempted to change the predicted identity.",
        ], ORANGE)
        add_text_box(ax, 0.675, 0.53, 0.27, 0.27, "Attack output", [
            f"FGSM success at epsilon 0.10: {fgsm_y[fgsm_x.index(0.1)]:.2f}%.",
            f"PGD success at epsilon 0.10: {pgd_y[pgd_x.index(0.1)]:.2f}%.",
            "The plotted curve shows that attack success increased with perturbation strength.",
        ], RED)
        line_chart(fig, [0.12, 0.12, 0.76, 0.32], fgsm_x, [
            ("FGSM applied", fgsm_y, ORANGE, "o"),
            ("PGD applied", pgd_y, RED, "s"),
        ], "Applied evasion attack results on baseline model", "Attack success (%)", (0, 110))
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 3 Graphics: Membership Inference Attack",
                            "Inference attack results show whether confidence scores leak training membership", "08")
        labels = ["Attack accuracy", "Privacy risk", "True member detection", "False member rate",
                  "Member confidence", "Non-member confidence"]
        values = [
            member["attack_accuracy"] * 100,
            member["privacy_risk_score"] * 100,
            member["true_member_detection_rate"] * 100,
            member["false_member_rate"] * 100,
            member["member_mean_confidence"] * 100,
            member["nonmember_mean_confidence"] * 100,
        ]
        bar_chart(fig, [0.055, 0.18, 0.55, 0.57], labels, [
            ("Membership inference metrics", values, PURPLE),
        ], "Membership inference metrics", "Rate / confidence (%)", (0, 100))
        add_text_box(ax, 0.65, 0.18, 0.315, 0.57, "Inference attack result", [
            "Membership inference tested whether model confidence could reveal whether a biometric sample was part of the training set.",
            f"The attack accuracy was {pct(member['attack_accuracy'])}, which is close to random guessing for a binary membership decision.",
            f"The privacy risk score was {pct(member['privacy_risk_score'])}.",
            "The member confidence was higher than the non-member confidence, but the selected threshold did not produce a strong membership leak in this run.",
        ], PURPLE, font_size=10.5)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 3 Applied Inference Attack Evidence",
                            "Graphical evidence after applying membership inference to the baseline model", "10")
        add_text_box(ax, 0.055, 0.52, 0.28, 0.28, "Attack input", [
            "Training-member samples and non-member test samples.",
            "Model confidence scores from the baseline model.",
            f"Membership threshold: {member['threshold']:.2f}.",
        ], BLUE)
        add_text_box(ax, 0.365, 0.52, 0.28, 0.28, "Attack decision rule", [
            "If confidence is above the threshold, the sample is inferred as a member.",
            "If confidence is below the threshold, it is inferred as a non-member.",
            "The attack measures whether the model leaks training participation.",
        ], PURPLE)
        add_text_box(ax, 0.675, 0.52, 0.28, 0.28, "Attack output", [
            f"Attack accuracy: {pct(member['attack_accuracy'])}.",
            f"Privacy risk score: {pct(member['privacy_risk_score'])}.",
            f"Member confidence: {pct(member['member_mean_confidence'])}; non-member confidence: {pct(member['nonmember_mean_confidence'])}.",
        ], GREEN)
        bar_chart(fig, [0.12, 0.12, 0.76, 0.32], ["Attack accuracy", "Privacy risk", "Member conf.", "Non-member conf."], [
            ("Applied inference metrics", [
                member["attack_accuracy"] * 100,
                member["privacy_risk_score"] * 100,
                member["member_mean_confidence"] * 100,
                member["nonmember_mean_confidence"] * 100,
            ], PURPLE),
        ], "Applied membership inference results on baseline model", "Rate / confidence (%)", (0, 100))
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 3 Graphics: Model Poisoning Attack",
                            "Label-flipping poisoning measured the effect of corrupted local training updates", "09")
        poison_labels = ["Clean accuracy", "Poisoned accuracy", "Accuracy drop"]
        poison_values = [
            poison["clean_accuracy"] * 100,
            poison["poisoned_accuracy"] * 100,
            poison["accuracy_drop"] * 100,
        ]
        bar_chart(fig, [0.10, 0.20, 0.52, 0.53], poison_labels, [
            ("Poisoning result", poison_values, RED),
        ], "Label-flipping poisoning result", "Percent (%)", (0, 40))
        add_text_box(ax, 0.67, 0.20, 0.27, 0.53, "How poisoning was tested", [
            f"Poison rate: {pct(poison['poison_rate'], 1)} of the local training samples.",
            f"Poisoned samples: {poison['poisoned_samples']}.",
            f"The attack reduced accuracy by {pct(poison['accuracy_drop'])}, from {pct(poison['clean_accuracy'])} to {pct(poison['poisoned_accuracy'])}.",
            "This supports the argument that federated biometric learning needs protection not only against test-time attacks, but also against corrupted client updates.",
        ], RED)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 3 Combined Attack Evidence",
                            "Attack graphics generated by the implementation", "10")
        add_image(ax, "do_fmbs_attack_results.png", 0.055, 0.17, 0.43, 0.62,
                  "FGSM and PGD attack chart")
        add_image(ax, "do_fmbs_attack_extended_results.png", 0.515, 0.17, 0.43, 0.62,
                  "Membership inference and poisoning chart")
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 4: Defense Mechanism Construction",
                            "The proposed model combines privacy protection and adversarial robustness", "11")
        add_text_box(ax, 0.055, 0.53, 0.28, 0.27, "Differential Privacy", [
            "Adds controlled Gaussian noise during training.",
            "Goal: reduce sensitive information leakage from learned model behavior.",
            f"Saved epsilon parameter: {final['parameters']['dp_epsilon']}.",
        ], BLUE)
        add_text_box(ax, 0.365, 0.53, 0.28, 0.27, "Adversarial Training", [
            "Uses adversarial examples during training.",
            "Goal: improve recognition stability when inputs are perturbed.",
            f"Saved ART epsilon parameter: {final['parameters']['art_epsilon']}.",
        ], ORANGE)
        add_text_box(ax, 0.675, 0.53, 0.28, 0.27, "DO-FMBS", [
            "Combines DP and ART in the federated multimodal setting.",
            "Goal: balance recognition accuracy, privacy preservation, and attack resistance.",
            f"Saved epochs: {final['parameters']['epochs']}.",
        ], GREEN)
        add_image(ax, "do_fmbs_comparison_results.png", 0.13, 0.11, 0.74, 0.33,
                  "Phase 4 comparison chart")
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 4 Process Details: Defense Training",
                            "The defense phase compared separate and combined protection strategies", "12")
        add_text_box(ax, 0.035, 0.50, 0.445, 0.30, "DP model", [
            "The Differential Privacy variant introduced Gaussian noise during training.",
            "Its purpose was to reduce the amount of sensitive biometric information exposed through learned model behavior.",
            "The saved epsilon value for DP was 0.50.",
        ], BLUE, font_size=10.5)
        add_text_box(ax, 0.520, 0.50, 0.445, 0.30, "ART model", [
            "The Adversarial Robustness Training variant trained with adversarially perturbed examples.",
            "Its purpose was to make the model less sensitive to adversarial input perturbations.",
            "The saved adversarial-training epsilon value was 0.05.",
        ], ORANGE, font_size=10.5)
        add_text_box(ax, 0.035, 0.17, 0.445, 0.25, "DO-FMBS model", [
            "The proposed model combined Differential Privacy and Adversarial Robustness Training.",
            "It was trained in the same multimodal federated setting as the baseline.",
            "The final defended model was saved as do_fmbs_model.keras.",
        ], GREEN, font_size=10.5)
        add_text_box(ax, 0.520, 0.17, 0.445, 0.25, "Phase 4 comparison logic", [
            "The baseline, DP, ART, and DO-FMBS variants were compared using face accuracy, attack success rate, and privacy score.",
            "This made it possible to separate pure accuracy improvement from privacy and robustness behavior.",
            "The Phase 4 outputs included the final comparison metrics, the trained DO-FMBS model, and the comparison chart.",
        ], PURPLE, font_size=10.5)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 4 Results: Model Comparison",
                            "Baseline, DP, ART, and DO-FMBS were compared on accuracy, privacy, and attack success", "12")
        bar_chart(fig, [0.07, 0.18, 0.62, 0.57], model_labels, [
            ("Face accuracy", face_acc, GREEN),
            ("Privacy score", privacy, BLUE),
            ("Attack success", attack_success, RED),
        ], "Phase 4 comparison", "Percent (%)", (0, 80))
        add_text_box(ax, 0.73, 0.18, 0.22, 0.57, "Main findings", [
            f"DO-FMBS achieved the highest face accuracy in the saved comparison: {comparison['do_fmbs']['face_accuracy'] * 100:.2f}%.",
            f"ART had the lowest attack success rate: {comparison['art']['attack_success_rate'] * 100:.2f}%.",
            "The proposed DO-FMBS result should be presented as a balanced model: it improves accuracy strongly while introducing privacy and robustness mechanisms.",
        ], GREEN)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 5: Validation Design",
                            "The final model was validated against the baseline under FGSM robustness tests and statistical checks", "13")
        add_metric(ax, 0.055, 0.70, 0.2, 0.11, "Validation samples",
                   str(validation["test_samples"]), "Saved validation run", BLUE)
        add_metric(ax, 0.285, 0.70, 0.2, 0.11, "Baseline mean",
                   pct(stat["baseline_mean"]), f"+/- {stat['baseline_std'] * 100:.2f}", ORANGE)
        add_metric(ax, 0.515, 0.70, 0.2, 0.11, "DO-FMBS mean",
                   pct(stat["do_fmbs_mean"]), f"+/- {stat['do_fmbs_std'] * 100:.2f}", GREEN)
        add_metric(ax, 0.745, 0.70, 0.2, 0.11, "Effect size",
                   f"{stat['effect_size']:.2f}", "Cohen's d", PURPLE)
        add_text_box(ax, 0.08, 0.33, 0.84, 0.24, "Validation tests performed", [
            "Robustness testing compared baseline and DO-FMBS accuracy under increasing FGSM epsilon values from 0.00 to 0.20.",
            "Statistical validation used repeated accuracy samples to compare the baseline and proposed model.",
            "Classification metrics compared accuracy, macro precision, macro recall, and macro F1-score.",
        ], BLUE)
        add_text_box(ax, 0.16, 0.13, 0.68, 0.12, "Validation conclusion", [
            "The saved report concludes that DO-FMBS passes robustness validation and demonstrates statistically significant improvement over the baseline.",
        ], GREEN)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 5 Process Details: Validation Outputs",
                            "Final validation tested robustness, statistical strength, and recognition quality", "15")
        add_text_box(ax, 0.035, 0.50, 0.445, 0.30, "Robustness validation", [
            "Baseline and DO-FMBS were evaluated under FGSM perturbation strengths of 0.00, 0.05, 0.10, 0.15, and 0.20.",
            "Accuracy and attack success rate were recorded at each epsilon value.",
            "The result was visualized as the robustness comparison chart.",
        ], ORANGE, font_size=10.5)
        add_text_box(ax, 0.520, 0.50, 0.445, 0.30, "Statistical validation", [
            "Repeated accuracy samples were calculated for both the baseline and DO-FMBS.",
            "A t-test measured whether the improvement was statistically significant.",
            f"The saved p-value was {stat['p_value']:.2e}, and the result was significant.",
        ], PURPLE, font_size=10.5)
        add_text_box(ax, 0.035, 0.17, 0.445, 0.25, "Classification validation", [
            "Accuracy, macro precision, macro recall, and macro F1-score were calculated.",
            "The classification metrics show whether the final model improved overall recognition quality, not only robustness.",
            "The summary was saved in validation_results.pkl and validation_report.txt.",
        ], GREEN, font_size=10.5)
        add_text_box(ax, 0.520, 0.17, 0.445, 0.25, "Phase 5 evidence generated", [
            "The validation phase stores robustness, statistical, and classification results.",
            "The validation report contains the detailed result interpretation.",
            "The robustness, statistics, and summary visuals provide the graphical validation evidence.",
        ], BLUE, font_size=10.5)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 5 Graphics: Robustness Under FGSM",
                            "DO-FMBS stayed above the baseline across all tested attack strengths", "14")
        eps = [r["epsilon"] for r in validation["robustness"]["baseline"]]
        base_acc = [r["accuracy"] * 100 for r in validation["robustness"]["baseline"]]
        do_acc = [r["accuracy"] * 100 for r in validation["robustness"]["do_fmbs"]]
        line_chart(fig, [0.07, 0.18, 0.62, 0.57], eps, [
            ("Baseline accuracy", base_acc, RED, "o"),
            ("DO-FMBS accuracy", do_acc, GREEN, "s"),
        ], "Validation accuracy under FGSM attack", "Accuracy (%)", (0, 75))
        add_text_box(ax, 0.73, 0.18, 0.22, 0.57, "Validation result", [
            f"Under clean input conditions, DO-FMBS achieved {do_acc[0]:.2f}% accuracy compared with baseline {base_acc[0]:.2f}%.",
            f"At epsilon 0.10, DO-FMBS achieved {do_acc[2]:.2f}% compared with baseline {base_acc[2]:.2f}%.",
            f"At epsilon 0.20, baseline accuracy dropped to {base_acc[-1]:.2f}%, while DO-FMBS retained {do_acc[-1]:.2f}%.",
            "The validation curve confirms that the proposed model remained stronger than the baseline across all tested FGSM attack strengths.",
        ], GREEN)
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Phase 5 Graphics: Statistics and Classification Metrics",
                            "The final comparison was statistically significant and improved core classification metrics", "15")
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
        base_metrics = [
            class_metrics["baseline"]["accuracy"] * 100,
            class_metrics["baseline"]["macro_avg_precision"] * 100,
            class_metrics["baseline"]["macro_avg_recall"] * 100,
            class_metrics["baseline"]["macro_avg_f1"] * 100,
        ]
        do_metrics = [
            class_metrics["do_fmbs"]["accuracy"] * 100,
            class_metrics["do_fmbs"]["macro_avg_precision"] * 100,
            class_metrics["do_fmbs"]["macro_avg_recall"] * 100,
            class_metrics["do_fmbs"]["macro_avg_f1"] * 100,
        ]
        bar_chart(fig, [0.07, 0.18, 0.46, 0.57], metric_labels, [
            ("Baseline", base_metrics, RED),
            ("DO-FMBS", do_metrics, GREEN),
        ], "Classification metrics", "Percent (%)", (0, 75))
        add_image(ax, "do_fmbs_validation_statistics.png", 0.57, 0.45, 0.36, 0.30,
                  "Statistical comparison")
        add_image(ax, "do_fmbs_validation_summary.png", 0.57, 0.12, 0.36, 0.28,
                  "Validation summary")
        finish(pdf, fig, ax)

        fig, ax = new_slide(pdf, "Final Model Implementation Summary",
                            "Implementation summary", "16")
        add_text_box(ax, 0.035, 0.46, 0.93, 0.34, "Implementation conclusion", [
            "The DO-FMBS model was built through five complete phases: multimodal biometric data preparation, federated baseline training, adversarial and privacy attack simulation, defense mechanism construction, and final validation.",
            "Phase 1 produced normalized face and fingerprint datasets and distributed them across non-IID federated clients. Phase 2 trained the baseline multimodal federated model and produced the first performance metrics.",
            "Phase 3 demonstrated the security problem by applying FGSM, PGD, membership inference, and label-flipping poisoning attacks. Phase 4 addressed the identified weaknesses using Differential Privacy, Adversarial Robustness Training, and the combined DO-FMBS model.",
            f"Phase 5 confirmed that DO-FMBS improved validation mean accuracy from {pct(stat['baseline_mean'])} to {pct(stat['do_fmbs_mean'])}, with a statistically significant improvement.",
        ], GREEN, font_size=10.5)
        add_text_box(ax, 0.035, 0.13, 0.93, 0.25, "Metric scope", [
            "The current validation focused on classification accuracy, adversarial robustness, statistical significance, and attack behavior.",
            "False Acceptance Rate is a threshold-based biometric verification metric that requires genuine and impostor verification attempts. Therefore, FAR is treated separately from the current classification and robustness validation results.",
            "A dedicated verification experiment with thresholds, genuine attempts, impostor attempts, and false-accept counts is required before presenting a newly re-estimated FAR value.",
        ], ORANGE, font_size=11.0)
        finish(pdf, fig, ax)

    print(OUT)


if __name__ == "__main__":
    main()

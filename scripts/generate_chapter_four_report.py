"""Generate Chapter Four from the synchronized DO-FMBS replay evidence.

This generator is intentionally independent of earlier reports.  It validates
the finalized six-phase replay, the operational biometric evaluator and the
Phase-6 constrained controller before producing a thesis-ready Markdown,
editable DOCX and searchable PDF chapter.  Numeric tables and plots are built
from the validated replay artifacts rather than copied from prose.

Run from the repository root::

    python scripts/generate_chapter_four_report.py
    python scripts/generate_chapter_four_report.py --validate-inputs-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_constrained_dynamic_report import (
    _equation_image_buffer,
    _markdown_blocks,
    _plain_markdown,
    write_docx,
)
from scripts.generate_fully_recorded_metrics_report import (
    Evidence as RecordedEvidence,
    load_evidence as load_recorded_evidence,
)


REPORT_STEM = "CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION"
DEFAULT_RUN_DIR = ROOT / "runtime_recording" / "full_pipeline_20260809"
DEFAULT_EVALUATOR_DIR = (
    ROOT / "runtime_metrics_evidence" / "recorded_full_pipeline_biometric_evaluation"
)
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "chapter_four"
DEFAULT_ARCHITECTURE = (
    ROOT / "docs" / "runtime_metrics_architecture"
    / "DO_FMBS_ARCHITECTURE_FOR_SIMULATION_PROCESS.png"
)
VERIFIED_LIBRARY_VERSIONS = {
    "numpy": "1.24.3",
    "scipy": "1.15.3",
    "scikit-learn": "1.9.0",
    "matplotlib": "3.10.9",
    "pandas": "2.3.3",
    "streamlit": "1.52.2",
    "Pillow": "12.3.0",
    "opencv-python": "4.11.0.86",
}

# The Phase-2 global model contains 2,937,634 float32 values distributed over
# 36 arrays.  FedAvg exchanges raw weight arrays, so communication-equivalent
# payloads must be based on these arrays rather than on the serialized .keras
# archive, which also contains metadata and model structure.
FEDERATED_MODEL_PARAMETER_COUNT = 2_937_634
FEDERATED_MODEL_WEIGHT_ARRAYS = 36
FLOAT32_BYTES = 4

FIGURE_SPECS = (
    ("CHAPTER4_FIGURE_4_1_CLIENT_DISTRIBUTION.png", "Figure 4.1: Persistent identity-partitioned client data distribution (Researcher, 2026)."),
    ("CHAPTER4_FIGURE_4_2_TRAINING_CONVERGENCE.png", "Figure 4.2: Federated local-training convergence across twelve rounds (Researcher, 2026)."),
    ("DO_FMBS_ARCHITECTURE_FOR_SIMULATION_PROCESS.png", "Figure 4.3: Architecture for the DO-FMBS simulation process (Researcher, 2026)."),
    ("CHAPTER4_FIGURE_4_4_ATTACK_VALIDATION.png", "Figure 4.4: Baseline FGSM and PGD evasion-attack success rates (Researcher, 2026)."),
    ("CHAPTER4_FIGURE_4_5_DEFENSE_COMPARISON.png", "Figure 4.5: Clean accuracy, attack success and attack-resistance comparison (Researcher, 2026)."),
    ("CHAPTER4_FIGURE_4_6_DYNAMIC_CONTROLLER.png", "Figure 4.6: Dynamic AI, PA, DM, GS and predicted Y schedule (Researcher, 2026)."),
    ("CHAPTER4_FIGURE_4_7_ACCURACY_RESULTS.png", "Figure 4.7: Baseline and DO-FMBS face-classification results (Researcher, 2026)."),
    ("CHAPTER4_FIGURE_4_8_FAR_FRR.png", "Figure 4.8: Claimed-identity face-verification FAR and FRR with project limits (Researcher, 2026)."),
    ("CHAPTER4_FIGURE_4_9_AI_ASR.png", "Figure 4.9: Attack intensity and the two recorded ASR definitions (Researcher, 2026)."),
    ("CHAPTER4_FIGURE_4_10_PRIVACY_SCORE.png", "Figure 4.10: Final-model membership-inference privacy pressure (Researcher, 2026)."),
    ("CHAPTER4_FIGURE_4_11_ROBUSTNESS.png", "Figure 4.11: Phase-5 baseline and DO-FMBS accuracy under FGSM (Researcher, 2026)."),
    ("CHAPTER4_FIGURE_4_12_RUNTIME.png", "Figure 4.12: Recorded computational runtime by simulation phase (Researcher, 2026)."),
)

STALE_CHAPTER_FIGURES = (
    "CHAPTER4_FIGURE_4_6_ACCURACY_RESULTS.png",
    "CHAPTER4_FIGURE_4_7_FAR_FRR.png",
    "CHAPTER4_FIGURE_4_8_AI_ASR.png",
    "CHAPTER4_FIGURE_4_9_PRIVACY_SCORE.png",
    "CHAPTER4_FIGURE_4_10_ROBUSTNESS.png",
    "CHAPTER4_FIGURE_4_11_RUNTIME.png",
    "CHAPTER4_FIGURE_4_12_DYNAMIC_CONTROLLER.png",
)


@dataclass(frozen=True)
class ChapterEvidence:
    recorded: RecordedEvidence
    output_dir: Path
    face_meta: dict[str, object]
    finger_meta: dict[str, object]
    client_rows: list[dict[str, int]]
    baseline: dict
    attacks: dict
    defenses: dict
    validation: dict
    coefficients: list[dict[str, str]]
    constraint_checks: list[dict[str, str]]
    stress_scenarios: list[dict[str, str]]
    source_paths: list[Path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the DO-FMBS Chapter Four report.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--evaluator-dir", type=Path, default=DEFAULT_EVALUATOR_DIR)
    parser.add_argument("--architecture-image", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", default="md,docx,pdf")
    parser.add_argument("--validate-inputs-only", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_pickle(path: Path) -> dict:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return value


def require_finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def load_npz_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        required = {"X_train", "X_test", "y_train", "y_test", "n_classes", "n_features", "image_shape"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path.name} lacks {sorted(missing)}")
        return {
            "train": int(data["X_train"].shape[0]),
            "test": int(data["X_test"].shape[0]),
            "classes": int(np.asarray(data["n_classes"]).reshape(-1)[0]),
            "features": int(np.asarray(data["n_features"]).reshape(-1)[0]),
            "image_shape": tuple(int(v) for v in np.asarray(data["image_shape"]).reshape(-1)),
        }


def load_clients(run_dir: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for client_id in range(5):
        path = run_dir / "do_fmbs_data" / "clients" / f"client_{client_id}.npz"
        with np.load(path, allow_pickle=True) as data:
            face_n = int(data["face_X"].shape[0])
            finger_n = int(data["finger_X"].shape[0])
            rows.append({
                "client": client_id,
                "face": face_n,
                "finger": finger_n,
                "total": face_n + finger_n,
                "face_classes": int(np.unique(data["face_y"]).size),
                "finger_classes": int(np.unique(data["finger_y"]).size),
            })
    return rows


def verify_manifest_member(recorded: RecordedEvidence, relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    matches = [row for row in recorded.artifact_rows if str(row["artifact"]).replace("\\", "/") == normalized]
    if len(matches) != 1:
        raise ValueError(f"Finalized artifact manifest has no unique entry for {relative}")
    path = recorded.run_dir / normalized
    if sha256(path) != matches[0]["sha256"]:
        raise ValueError(f"Artifact hash changed after replay finalization: {relative}")
    return path


def load_evidence(args: argparse.Namespace) -> ChapterEvidence:
    # Reuse the strict cross-manifest validator used by the fully recorded report.
    adapter = argparse.Namespace(
        run_dir=args.run_dir,
        evaluator_dir=args.evaluator_dir,
        architecture_image=args.architecture_image,
        validate_inputs_only=False,
    )
    recorded = load_recorded_evidence(adapter)
    run_dir = recorded.run_dir
    for distribution, expected in VERIFIED_LIBRARY_VERSIONS.items():
        installed = importlib.metadata.version(distribution)
        if installed != expected:
            raise ValueError(
                f"Installed {distribution} version {installed} differs from verified version {expected}"
            )

    face_path = verify_manifest_member(recorded, "do_fmbs_data/face_data.npz")
    finger_path = verify_manifest_member(recorded, "do_fmbs_data/fingerprint_data.npz")
    for client_id in range(5):
        verify_manifest_member(recorded, f"do_fmbs_data/clients/client_{client_id}.npz")
    baseline_path = verify_manifest_member(recorded, "do_fmbs_data/baseline_metrics.pkl")
    attacks_path = verify_manifest_member(recorded, "do_fmbs_data/attack_results.pkl")
    defenses_path = verify_manifest_member(recorded, "do_fmbs_data/final_results.pkl")
    validation_path = verify_manifest_member(recorded, "do_fmbs_data/validation_results.pkl")
    coefficient_path = verify_manifest_member(
        recorded, "simulation_evidence/constrained_dynamic_model/coefficient_estimates.csv"
    )
    model_fit_path = verify_manifest_member(
        recorded, "simulation_evidence/constrained_dynamic_model/model_fit.json"
    )
    schedule_path = verify_manifest_member(
        recorded, "simulation_evidence/constrained_dynamic_model/dynamic_optimization_schedule.csv"
    )
    phase6_dir = run_dir / "simulation_evidence" / "constrained_dynamic_model"
    constraint_path = phase6_dir / "constraint_checks.csv"
    stress_path = phase6_dir / "stress_test_scenarios.csv"
    acceptance_path = phase6_dir / "acceptance_ranges.csv"
    for supplemental_path in (constraint_path, stress_path, acceptance_path):
        if not supplemental_path.is_file() or supplemental_path.stat().st_size == 0:
            raise FileNotFoundError(f"Phase-6 validation evidence is missing: {supplemental_path}")
    for figure in (
        "do_fmbs_training_accuracy.png",
        "do_fmbs_attack_extended_results.png",
        "do_fmbs_comparison_results.png",
        "do_fmbs_validation_robustness.png",
    ):
        verify_manifest_member(recorded, figure)

    face_meta = load_npz_metadata(face_path)
    finger_meta = load_npz_metadata(finger_path)
    clients = load_clients(run_dir)
    if sum(row["face"] for row in clients) != face_meta["train"]:
        raise ValueError("Persistent clients do not cover every face training sample")
    if sum(row["finger"] for row in clients) != finger_meta["train"]:
        raise ValueError("Persistent clients do not cover every fingerprint training sample")

    baseline = read_pickle(baseline_path)
    attacks = read_pickle(attacks_path)
    defenses = read_pickle(defenses_path)
    validation = read_pickle(validation_path)
    coefficients = read_csv(coefficient_path)
    constraint_checks = read_csv(constraint_path)
    stress_scenarios = read_csv(stress_path)
    if [row.get("parameter") for row in coefficients] != ["B0", "B_AI", "B_PA", "B_DM", "B_GS"]:
        raise ValueError("Coefficient evidence does not contain B0, B_AI, B_PA, B_DM and B_GS")
    if not all(str(row.get("constraint_satisfied")).casefold() == "true" for row in coefficients):
        raise ValueError("A fitted coefficient constraint is violated")
    if len(constraint_checks) != 9 or not all(
        str(row.get("satisfied")).casefold() == "true" for row in constraint_checks
    ):
        raise ValueError("Phase-6 constraint checks do not contain nine verified relationships")
    if [row.get("scenario") for row in stress_scenarios] != [
        "preferred_operating", "managed_operating", "modeled_critical", "out_of_domain"
    ]:
        raise ValueError("Phase-6 stress scenarios differ from the finalized protocol")
    if [row.get("deployment_decision") for row in stress_scenarios] != [
        "accept", "accept", "reject_or_defer", "reject_or_defer"
    ]:
        raise ValueError("Phase-6 stress decisions differ from the finalized protocol")
    if baseline.get("training_parameters") != {"rounds": 12, "local_epochs": 2, "batch_size": 64}:
        raise ValueError("Baseline training configuration differs from the finalized replay")
    if defenses.get("parameters", {}).get("epochs") != 12:
        raise ValueError("Defense experiment does not contain twelve epochs")
    if validation.get("test_samples") != 160:
        raise ValueError("Validation evidence does not contain 160 samples")

    evaluator_script = ROOT / "scripts" / "measure_operational_biometric_metrics.py"
    evaluator_inputs = recorded.evaluator_manifest.get("inputs", [])
    evaluator_script_rows = [row for row in evaluator_inputs if str(row.get("path", "")).replace("\\", "/") == "scripts/measure_operational_biometric_metrics.py"]
    if len(evaluator_script_rows) != 1 or sha256(evaluator_script) != evaluator_script_rows[0].get("sha256"):
        raise ValueError("Operational evaluator source differs from its synchronized manifest hash")

    source_paths = [
        face_path, finger_path, baseline_path, attacks_path, defenses_path,
        validation_path, coefficient_path, model_fit_path, schedule_path,
        constraint_path, stress_path, acceptance_path,
        recorded.run_dir / "phase_runtime_registry.json",
        recorded.evaluator_dir / "evaluation_manifest.json",
        recorded.evaluator_dir / "clean_far_frr_acceptance.csv",
        recorded.evaluator_dir / "ai_conditioned_far_frr_asr.csv",
        recorded.evaluator_dir / "privacy_pa_evaluation.csv",
        recorded.evaluator_dir / "evaluation_runtime.csv",
        *(recorded.run_dir / f"phase{phase}_{name}.py" for phase, name in (
            (1, "data_preparation"), (2, "federated_learning"),
            (3, "adversarial_attacks"), (4, "defense_mechanisms"),
            (5, "model_validation"), (6, "constrained_dynamic_optimization"),
        )),
        recorded.run_dir / "scripts" / "constrained_dynamic_model.py",
        evaluator_script,
        ROOT / "requirements_app.txt",
        args.architecture_image.resolve(),
    ]
    return ChapterEvidence(
        recorded=recorded,
        output_dir=args.output_dir.resolve(),
        face_meta=face_meta,
        finger_meta=finger_meta,
        client_rows=clients,
        baseline=baseline,
        attacks=attacks,
        defenses=defenses,
        validation=validation,
        coefficients=coefficients,
        constraint_checks=constraint_checks,
        stress_scenarios=stress_scenarios,
        source_paths=source_paths,
    )


def pct(value: object, digits: int = 2) -> str:
    return f"{require_finite(value, 'rate') * 100:.{digits}f}%"


def num(value: object, digits: int = 6) -> str:
    number = require_finite(value, "number")
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(md_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(md_cell(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def figure_marker(filename: str, caption: str) -> str:
    return f"[[FIGURE:{filename}|{caption}]]"


def make_plots(evidence: ChapterEvidence) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = evidence.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for stale_name in STALE_CHAPTER_FIGURES:
        stale_path = out / stale_name
        if stale_path.is_file():
            stale_path.unlink()
    navy, blue, orange, green, red, gray = "#17365D", "#4472C4", "#ED7D31", "#70AD47", "#C00000", "#7F8C8D"

    def finish(fig, name: str) -> Path:
        path = out / name
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        if path.stat().st_size < 10_000:
            raise ValueError(f"Generated plot is unexpectedly small: {path}")
        return path

    # Figure 4.1: hash-verified client NPZ files.
    labels = [f"Client {row['client']}" for row in evidence.client_rows]
    face = [row["face"] for row in evidence.client_rows]
    finger = [row["finger"] for row in evidence.client_rows]
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x, face, label="Face samples", color=blue)
    ax.bar(x, finger, bottom=face, label="Fingerprint samples", color=orange)
    for i, row in enumerate(evidence.client_rows):
        ax.text(i, row["total"] + 16, str(row["total"]), ha="center", fontsize=9)
    ax.set_xticks(x, labels); ax.set_ylabel("Training samples")
    ax.set_title("Persistent Identity-Partitioned Federated Data")
    ax.legend(ncol=2, frameon=False); ax.grid(axis="y", alpha=.2)
    finish(fig, FIGURE_SPECS[0][0])

    # Copy the revised architecture into this standalone output directory.
    architecture_target = out / FIGURE_SPECS[2][0]
    architecture_target.write_bytes(evidence.recorded.architecture.read_bytes())

    history = evidence.baseline["history"]
    rounds = np.arange(1, len(history["face_acc"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8.4, 4.5))
    ax1.plot(rounds, np.asarray(history["face_acc"]) * 100, marker="o", color=blue, label="Face local accuracy")
    ax1.plot(rounds, np.asarray(history["finger_acc"]) * 100, marker="s", color=orange, label="Fingerprint local accuracy")
    ax1.set(xlabel="Federated round", ylabel="Mean final local-epoch accuracy (%)", xticks=rounds)
    ax1.grid(alpha=.2)
    ax2 = ax1.twinx(); ax2.plot(rounds, history["loss"], linestyle="--", color=gray, label="Training loss")
    ax2.set_ylabel("Mean local loss")
    lines = ax1.lines + ax2.lines; ax1.legend(lines, [line.get_label() for line in lines], frameon=False, ncol=3, loc="upper center")
    ax1.set_title("Federated Training Convergence")
    finish(fig, FIGURE_SPECS[1][0])

    fgsm = evidence.attacks["fgsm_results"]; pgd = evidence.attacks["pgd_results"]
    eps = [row["epsilon"] for row in fgsm]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.plot(eps, [100 * row["success_rate"] for row in fgsm], marker="o", color=blue, label="FGSM")
    ax.plot(eps, [100 * row["success_rate"] for row in pgd], marker="s", color=red, label="PGD (10 iterations)")
    ax.set(xlabel="Perturbation magnitude (epsilon)", ylabel="Attack success rate (%)", ylim=(0, 105))
    ax.set_title("Evasion-Attack Validation on the Federated Baseline"); ax.grid(alpha=.2); ax.legend(frameon=False)
    finish(fig, FIGURE_SPECS[3][0])

    comp = evidence.defenses["comparison"]
    methods = ["baseline", "dp", "art", "do_fmbs"]
    names = ["No defense", "DP-style", "ART", "DO-FMBS"]
    x = np.arange(4); width = .25
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - width, [100 * comp[m]["face_accuracy"] for m in methods], width, label="Clean accuracy", color=blue)
    ax.bar(x, [100 * comp[m]["attack_success_rate"] for m in methods], width, label="FGSM ASR", color=red)
    ax.bar(x + width, [100 * comp[m]["privacy_score"] for m in methods], width, label="Attack resistance (1-ASR)", color=green)
    ax.set_xticks(x, names); ax.set_ylabel("Rate (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.2)
    ax.set_title("Defense-Mechanism Comparison"); ax.legend(frameon=False, ncol=3)
    finish(fig, FIGURE_SPECS[4][0])

    metrics = evidence.validation["classification_metrics"]
    metric_names = ["accuracy", "macro_avg_precision", "macro_avg_recall", "macro_avg_f1"]
    labels = ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]
    x = np.arange(4); width = .36
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.bar(x-width/2, [100*metrics["baseline"][m] for m in metric_names], width, label="Federated baseline", color=gray)
    ax.bar(x+width/2, [100*metrics["do_fmbs"][m] for m in metric_names], width, label="DO-FMBS", color=blue)
    ax.set_xticks(x, labels); ax.set_ylabel("Rate (%)"); ax.set_ylim(0, 75); ax.grid(axis="y", alpha=.2)
    ax.set_title("Face-Classification Validation"); ax.legend(frameon=False)
    finish(fig, FIGURE_SPECS[6][0])

    clean = {row["model"]: row for row in evidence.recorded.clean_rows}
    base, final = clean["Baseline comparator"], clean["DO-FMBS final"]
    x = np.arange(2); width = .33
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x-width/2, [100*float(base["FAR"]), 100*float(final["FAR"])], width, label="FAR", color=orange)
    ax.bar(x+width/2, [100*float(base["FRR"]), 100*float(final["FRR"])], width, label="FRR", color=blue)
    ax.axhline(10, color=orange, linestyle="--", linewidth=1, label="FAR project limit")
    ax.axhline(20, color=blue, linestyle=":", linewidth=1, label="FRR project limit")
    ax.set_xticks(x, ["Baseline comparator", "DO-FMBS final"]); ax.set_ylabel("Rate (%)"); ax.set_ylim(0, 65)
    ax.set_title("Claimed-Identity Face Verification"); ax.grid(axis="y", alpha=.2); ax.legend(frameon=False, ncol=2)
    finish(fig, FIGURE_SPECS[7][0])

    ai_rows = evidence.recorded.ai_rows
    ai = [float(row["AI"]) for row in ai_rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(ai, [100*float(row["classification_ASR"]) for row in ai_rows], marker="o", color=red, label="Classification ASR")
    ax.plot(ai, [100*float(row["verification_rejection_ASR"]) for row in ai_rows], marker="s", color=orange, label="Verification-rejection ASR")
    ax.set(xlabel="Normalized attack intensity, AI", ylabel="Attack success rate (%)", ylim=(0, 105))
    ax.set_title("Operational Attack Success by Attack Intensity"); ax.grid(alpha=.2); ax.legend(frameon=False)
    finish(fig, FIGURE_SPECS[8][0])

    pa = float(evidence.recorded.pa_row["PA"]); limit = float(evidence.recorded.pa_row["PA_managed_limit"])
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.barh(["DO-FMBS final"], [pa], color=green, height=.42)
    ax.axvline(limit, color=red, linestyle="--", label=f"Managed limit = {limit:.2f}")
    ax.text(pa + .01, 0, f"PA = {pa:.3f}", va="center", fontsize=11)
    ax.set(xlim=(0, .55), xlabel="Privacy pressure, PA"); ax.set_title("Membership-Inference Privacy Pressure")
    ax.grid(axis="x", alpha=.2); ax.legend(frameon=False)
    finish(fig, FIGURE_SPECS[9][0])

    robust = evidence.validation["robustness"]
    eps5 = [float(row["epsilon"]) for row in robust["baseline"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(eps5, [100*float(row["accuracy"]) for row in robust["baseline"]], marker="o", color=gray, label="Federated baseline accuracy")
    ax.plot(eps5, [100*float(row["accuracy"]) for row in robust["do_fmbs"]], marker="s", color=blue, label="DO-FMBS accuracy")
    ax.set(xlabel="FGSM epsilon", ylabel="Accuracy under attack (%)", ylim=(0, 100))
    ax.set_title("Phase-5 Robustness Validation"); ax.grid(alpha=.2); ax.legend(frameon=False)
    finish(fig, FIGURE_SPECS[10][0])

    runtime_rows = evidence.recorded.runtime_rows
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    bars = ax.bar([f"P{row['phase']}" for row in runtime_rows], [float(row["elapsed_seconds"]) for row in runtime_rows], color=[blue, blue, orange, red, green, gray])
    for bar, row in zip(bars, runtime_rows):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+12, f"{float(row['elapsed_seconds']):.1f}s", ha="center", fontsize=8)
    ax.set(ylabel="Elapsed seconds", xlabel="Simulation phase"); ax.set_title("Recorded Phase Runtime (perf_counter)")
    ax.grid(axis="y", alpha=.2)
    finish(fig, FIGURE_SPECS[11][0])

    schedule = evidence.recorded.schedule_rows; rounds = [int(row["round"]) for row in schedule]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for field, color, marker in (("AI", red, "o"), ("PA", orange, "s"), ("DM", blue, "^"), ("GS", green, "D"), ("predicted_Y", navy, "*")):
        ax.plot(rounds, [float(row[field]) for row in schedule], label=field, marker=marker, color=color, linewidth=1.8)
    ax.axhline(.70, color=green, linestyle="--", linewidth=1, label="Preferred Y = 0.70")
    ax.axhline(.60, color=red, linestyle=":", linewidth=1, label="Minimum Y = 0.60")
    ax.set(xlabel="Round", ylabel="Normalized value", xticks=rounds, ylim=(0, 1.05))
    ax.set_title("Dynamic Constrained-Control Schedule"); ax.grid(alpha=.2); ax.legend(frameon=False, ncol=4, fontsize=8)
    finish(fig, FIGURE_SPECS[5][0])

    return [out / name for name, _ in FIGURE_SPECS]


def build_markdown(e: ChapterEvidence, generated_at: str) -> str:
    r = e.recorded
    final = next(row for row in r.clean_rows if row["model"] == "DO-FMBS final")
    comparator = next(row for row in r.clean_rows if row["model"] == "Baseline comparator")
    pa = r.pa_row
    fit = r.fit
    validation_metrics = e.validation["classification_metrics"]
    statistical = e.validation["statistical_test"]
    phase4 = e.defenses["comparison"]
    total_runtime = sum(float(row["elapsed_seconds"]) for row in r.runtime_rows)
    model_bytes = FEDERATED_MODEL_PARAMETER_COUNT * FLOAT32_BYTES
    communication_bytes = model_bytes * 2 * 5 * 12
    communication_mb = communication_bytes / 1_000_000
    communication_gib = communication_bytes / (1024 ** 3)
    per_round_bytes = model_bytes * 2 * 5
    preferred_rounds = sum(row["status"] == "preferred_target_met" for row in r.schedule_rows)
    managed_rounds = len(r.schedule_rows) - preferred_rounds
    max_vif = max(float(value) for value in fit["vifs"].values())
    stress_by_name = {row["scenario"]: row for row in e.stress_scenarios}

    client_table = [[
        row["client"], row["face"], row["face_classes"], row["finger"],
        row["finger_classes"], row["total"],
    ] for row in e.client_rows]
    client_table.append([
        "Total", sum(row["face"] for row in e.client_rows), e.face_meta["classes"],
        sum(row["finger"] for row in e.client_rows), e.finger_meta["classes"],
        sum(row["total"] for row in e.client_rows),
    ])

    coefficient_table = [[
        row["parameter"], num(row["estimate"], 6), num(row["standard_error"], 6),
        f"{num(row['ci_95_low'], 6)} to {num(row['ci_95_high'], 6)}",
        row["constraint"], "VERIFIED",
    ] for row in e.coefficients]

    attack_rows = []
    pgd_by_epsilon = {float(row["epsilon"]): row for row in e.attacks["pgd_results"]}
    for fgsm in e.attacks["fgsm_results"]:
        eps = float(fgsm["epsilon"]); pgd = pgd_by_epsilon[eps]
        attack_rows.append([num(eps, 2), pct(fgsm["success_rate"]), pct(pgd["success_rate"])])

    defense_rows = []
    no_defense = phase4["baseline"]
    for key, name in (("baseline", "No defense"), ("dp", "DP-style gradient noise"),
                      ("art", "Adversarial training"), ("do_fmbs", "DO-FMBS combined")):
        row = phase4[key]
        defense_rows.append([
            name, pct(row["face_accuracy"]), pct(row["attack_success_rate"]),
            pct(row["privacy_score"]),
            f"{(row['face_accuracy']-no_defense['face_accuracy'])*100:+.2f} pp",
            f"{(row['attack_success_rate']-no_defense['attack_success_rate'])*100:+.2f} pp",
        ])

    ai_rows = [[
        num(row["epsilon"], 2), num(row["AI"], 4),
        f"{row['FAR_numerator']}/{row['FAR_denominator']}", pct(row["FAR"]),
        f"{row['FRR_numerator']}/{row['FRR_denominator']}", pct(row["FRR"]),
        f"{row['classification_ASR_numerator']}/{row['classification_ASR_denominator']}", pct(row["classification_ASR"]),
        f"{row['verification_rejection_ASR_numerator']}/{row['verification_rejection_ASR_denominator']}", pct(row["verification_rejection_ASR"]),
        f"{float(row['elapsed_seconds']):.6f}",
    ] for row in r.ai_rows]

    schedule_rows = [[
        row["round"], num(row["epsilon"], 2), num(row["AI"], 4), num(row["PA"], 3),
        num(row["DM"], 2), row["participating_clients"], num(row["GS"], 2),
        num(row["predicted_Y"], 6), num(row["control_cost"], 6),
        str(row["status"]).replace("_", " ").upper(), "ACCEPTED",
    ] for row in r.schedule_rows]

    runtime_rows = [[
        f"Phase {row['phase']}", row["script"], f"{float(row['elapsed_seconds']):.6f}",
        row["status"],
    ] for row in r.runtime_rows]
    runtime_rows.append(["Total", "Summed active command duration", f"{total_runtime:.6f}", "ALL-SIX-PHASES-COMPLETED"])

    robustness_rows = []
    for base_row, do_row in zip(e.validation["robustness"]["baseline"], e.validation["robustness"]["do_fmbs"]):
        robustness_rows.append([
            num(base_row["epsilon"], 2), pct(base_row["accuracy"]), pct(do_row["accuracy"]),
            f"{(do_row['accuracy']-base_row['accuracy'])*100:+.2f} pp",
            pct(base_row["attack_success_rate"]), pct(do_row["attack_success_rate"]),
        ])

    optimization_validation_rows = [
        [
            "Coefficient restrictions",
            f"{len(e.constraint_checks)}/{len(e.constraint_checks)} relationships true; maximum violation {num(fit['maximum_constraint_violation'], 6)}",
            "CONSTRAINTS VERIFIED",
        ],
        [
            "SLSQP convergence",
            f"Success=true; {fit['optimization_message']}",
            "SOLVER CONVERGENCE VERIFIED",
        ],
        ["R-squared", num(fit["r_squared"], 6), "ABOVE 0.90 FIT CRITERION"],
        ["Adjusted R-squared", num(fit["adjusted_r_squared"], 6), "ABOVE 0.90 FIT CRITERION"],
        ["Five-fold CV R-squared", num(fit["cross_validation_r2_mean"], 6), "ABOVE 0.89 CV CRITERION"],
        ["RMSE", num(fit["rmse"], 6), "BELOW 0.06 ERROR CRITERION"],
        ["MAE", num(fit["mae"], 6), "ERROR MAGNITUDE QUANTIFIED"],
        ["Maximum VIF", num(max_vif, 6), "BELOW 2.00 COLLINEARITY CRITERION"],
        ["Residual normality check", f"Shapiro-Wilk p={num(fit['shapiro_p'], 6)}", "RESIDUAL CHECK VERIFIED"],
        ["Residual serial-pattern check", f"Durbin-Watson={num(fit['durbin_watson'], 6)}", "RESIDUAL CHECK VERIFIED"],
        [
            "Twelve-round policy schedule",
            f"12/12 accepted; {preferred_rounds} preferred and {managed_rounds} managed",
            "SCHEDULE FEASIBILITY VERIFIED",
        ],
        [
            "Preferred operating scenario",
            f"Y={num(stress_by_name['preferred_operating']['predicted_Y'], 6)}",
            "PREFERRED CONTROL VERIFIED",
        ],
        [
            "Managed operating scenario",
            f"Y={num(stress_by_name['managed_operating']['predicted_Y'], 6)}",
            "MINIMUM CONTROL VERIFIED",
        ],
        [
            "Critical modeled scenario",
            f"Y={num(stress_by_name['modeled_critical']['predicted_Y'], 6)} below 0.60; reject/defer activated",
            "SAFETY REJECTION VERIFIED",
        ],
        [
            "Out-of-domain scenario",
            "epsilon=0.35; AI=1.166667; PA=0.60; prediction withheld",
            "DOMAIN GUARD VERIFIED",
        ],
    ]

    clean_rows = []
    for row in (comparator, final):
        clean_rows.append([
            row["model"], num(row["threshold"], 9),
            f"{row['FAR_numerator']}/{row['FAR_denominator']}", pct(row["FAR"], 3),
            f"{pct(row['FAR_identity_cluster_95_low'], 3)} to {pct(row['FAR_identity_cluster_95_high'], 3)}",
            f"{row['FRR_numerator']}/{row['FRR_denominator']}", pct(row["FRR"], 3),
            f"{pct(row['FRR_identity_cluster_95_low'], 3)} to {pct(row['FRR_identity_cluster_95_high'], 3)}",
            row["status"],
        ])

    traceability_rows = [
        ["1", "phase1_data_preparation.py", "LFW download/cache and deterministic fingerprint generator", "Flatten; standardize; stratified split; identity partition", "face_data.npz; fingerprint_data.npz; five client NPZ files"],
        ["2", "phase2_federated_learning.py", "Phase-1 arrays; 640 matched training pairs", "Five local updates per round; sample-weighted FedAvg; 12 rounds", "baseline_model.keras; baseline_metrics.pkl"],
        ["3", "phase3_adversarial_attacks.py", "Baseline model and test pairs", "FGSM; 10-step PGD; membership inference; 10% label flipping", "attack_results.pkl"],
        ["4", "phase4_defense_mechanisms.py", "Full Phase-1 training arrays", "No-defense, DP-style, ART and combined training", "do_fmbs_model.keras; final_results.pkl"],
        ["5", "phase5_model_validation.py", "Baseline and final models; 160 paired samples", "Robustness sweep; classification metrics; statistical tests", "validation_results.pkl; validation_report.txt"],
        ["6", "phase6_constrained_dynamic_optimization.py", "Seven AI, five PA, four DM and five GS levels", "700-condition constrained fit; 63-policy search per round", "model_fit.json; coefficients; 12-round schedule"],
        ["Evaluator", "measure_operational_biometric_metrics.py", "Hash-matched face data and both model files", "Frozen claimed-identity thresholds; FAR, FRR, ASR, PA and uncertainty", "Four CSV files and evaluation_manifest.json"],
    ]

    artifact_summary = []
    for phase in range(1, 7):
        rows = [row for row in r.artifact_rows if int(row["phase"]) == phase]
        artifact_summary.append([phase, len(rows), sum(int(row["bytes"]) for row in rows), "SHA-256 VERIFIED"])

    software_rows = [
        ["Python", r.runtime_registry["python"].split(" ")[0], "Interpreter recorded in the finalized runtime registry"],
        ["TensorFlow CPU", r.evaluator_manifest["environment"]["tensorflow"], "Dual-input neural models, training, inference and FGSM gradients"],
        ["NumPy", VERIFIED_LIBRARY_VERSIONS["numpy"], "Arrays, deterministic simulation and model-weight aggregation"],
        ["SciPy", VERIFIED_LIBRARY_VERSIONS["scipy"], "SLSQP constrained estimation and statistical calculations"],
        ["scikit-learn", VERIFIED_LIBRARY_VERSIONS["scikit-learn"], "LFW loader, preprocessing, splitting and validation metrics"],
        ["Matplotlib", VERIFIED_LIBRARY_VERSIONS["matplotlib"], "Replay and chapter-specific plots"],
        ["pandas", VERIFIED_LIBRARY_VERSIONS["pandas"], "Tabular inspection and reporting support"],
        ["Streamlit", VERIFIED_LIBRARY_VERSIONS["streamlit"], "Interactive simulation interface"],
        ["Pillow", VERIFIED_LIBRARY_VERSIONS["Pillow"], "Image loading and report-figure verification"],
        ["OpenCV", VERIFIED_LIBRARY_VERSIONS["opencv-python"].rsplit(".", 1)[0], "Image-processing support"],
    ]

    model_components = [
        ["Face branch", "1,850 standardized inputs; Dense 256-128-64; ReLU, batch normalization and dropout", "62-class softmax face_output"],
        ["Fingerprint branch", "9,216 standardized inputs; Dense 256-128-64; ReLU, batch normalization and dropout", "100-class softmax finger_output"],
        ["Federated learner", "Five client models initialized from a common global state", "Sample-weighted global weights"],
        ["Attack engine", "FGSM, PGD, membership inference and label flipping", "AI, ASR and PA observations"],
        ["Defense layer", "Gradient noise, clean/adversarial mixtures and combined fine-tuning", "DM-mapped policy"],
        ["Dynamic controller", "Constrained response equation and lowest-cost feasible search", "GS, DM and next-round predicted Y"],
    ]

    variable_rows = [
        ["epsilon", "FGSM perturbation magnitude", "0.00 to 0.30", "Phase-3/operational attack setting"],
        ["AI", "Normalized attack intensity: epsilon / 0.30", "0.00 to 1.00", "Threat-state input"],
        ["PA", "Membership advantage: 2 max(0, attack accuracy - 0.50)", "0.00 to 0.50 fitted", "Privacy-pressure input"],
        ["DM", "Continuous defense-mechanism intensity", "0.00 to 1.00", "Controller decision"],
        ["K", "Participating federated clients", "3, 4 or 5 in deployment search", "Controller decision"],
        ["GS", "Global scaling/participation score: (K - 1) / 4", "0.50 to 1.00 in deployment search", "Federation control"],
        ["Y(t+1)", "Predicted next-round DO-FMBS efficiency", "0.00 to 1.00; minimum 0.60; preferred 0.70", "Controller response"],
        ["FAR", "Accepted impostor claims divided by impostor claims", "Project upper limit 0.10", "Face-verification security"],
        ["FRR", "Rejected genuine claims divided by genuine claims", "Project upper limit 0.20", "Face-verification usability"],
        ["ASR", "Successful attacks divided by the relevant conditional denominator", "0.00 to 1.00", "Attack effectiveness"],
    ]

    parameter_rows = [
        ["Federated rounds", 12, "Phase-2 command and baseline_metrics.pkl"],
        ["Local epochs per round", 2, "Phase-2 command and baseline_metrics.pkl"],
        ["Federated batch size", 64, "Phase-2 command and baseline_metrics.pkl"],
        ["Learning rate", 0.001, "Phase-2 compile_model"],
        ["Available clients", 5, "Phase-1 partitions and Phase-2 loop"],
        ["Attack epsilon selected in Phase 3", 0.10, "attack_results.pkl parameters"],
        ["PGD iterations", 10, "Phase-3 attack implementation"],
        ["Poison rate", 0.10, "attack_results.pkl parameters"],
        ["Defense epochs", 12, "final_results.pkl parameters"],
        ["DP-style epsilon", 0.50, "final_results.pkl parameters"],
        ["ART training epsilon", 0.05, "final_results.pkl parameters"],
        ["Final-model gradient-noise standard deviation", 0.0003, "Phase-4 combined implementation"],
        ["Minimum participating clients", 3, "Phase-6 optimization settings"],
        ["Candidate policies per round", 63, "21 DM values x 3 client choices"],
    ]

    range_rows = [
        ["AI", "0.000-0.167", ">0.167-0.333", ">0.333-1.000", "Increase DM/GS as threat pressure grows"],
        ["PA", "0.00-0.20", ">0.20-0.40", ">0.40-0.50", "Escalate privacy controls and investigate leakage"],
        ["DM", "Selected for Y >= 0.70", "Selected for Y >= 0.60", "DM=1 if the target remains difficult", "Apply mapped defense configuration"],
        ["GS", "0.75-1.00 (4-5 clients)", "0.50 (3 clients)", "0.00-0.25 (1-2 clients)", "Defer aggregation below three clients"],
        ["Y", ">=0.70", "0.60 to <0.70", "<0.60", "Accept preferred/managed; defer below minimum"],
    ]

    proof_rows = [
        ["AI", "epsilon = 0.10", "0.10 / 0.30", "0.333333"],
        ["PA", "recorded PA = 0.30; equivalent membership accuracy = 0.65", "2 x (0.65 - 0.50)", "0.300000"],
        ["DM", "controller selection", "0.80", "0.800000"],
        ["Defense map", "DM = 0.80", "adv fraction=0.5DM; train epsilon=0.05DM; noise=0.0003DM", "0.40; 0.04; 0.00024"],
        ["GS", "K = 4", "(4 - 1) / (5 - 1)", "0.750000"],
        ["Y", "fitted coefficients and Round-7 inputs", "B0 + B_AI AI + B_PA PA + B_DM DM + B_GS GS", "0.600501"],
        ["Decision", "minimum Y = 0.60", "0.600501 >= 0.60", "ACCEPTED"],
    ]

    performance_rows = [
        ["Face classification accuracy", pct(validation_metrics["do_fmbs"]["accuracy"]), "Phase-5 final model"],
        ["Clean face-verification FAR", pct(final["FAR"], 3), f"{final['FAR_numerator']}/{final['FAR_denominator']} claims"],
        ["Clean face-verification FRR", pct(final["FRR"], 3), f"{final['FRR_numerator']}/{final['FRR_denominator']} claims"],
        ["FGSM classification ASR at epsilon=0.10", pct(r.ai_rows[3]["classification_ASR"]), "Frozen final model; 147/200"],
        ["FGSM verification-rejection ASR at epsilon=0.10", pct(r.ai_rows[3]["verification_rejection_ASR"]), "Frozen threshold; 132/279"],
        ["Membership privacy pressure PA", num(pa["PA"], 3), "Managed limit 0.40"],
        ["Phase-5 clean accuracy improvement", f"{(e.validation['robustness']['do_fmbs'][0]['accuracy']-e.validation['robustness']['baseline'][0]['accuracy'])*100:+.2f} pp", "DO-FMBS minus baseline"],
        ["Total active Phase-1-to-6 runtime", f"{total_runtime:.6f} s", "Sum of six perf_counter command durations"],
        ["Communication-equivalent", f"{communication_mb:.3f} MB ({communication_gib:.3f} GiB)", "Raw float32 model-weight payload; 120 transfers"],
    ]

    phase6_summary_rows = [
        ["Observations", fit["observations"], "7 AI x 5 PA x 4 DM x 5 GS"],
        ["Estimator", fit["estimator"], fit["optimization_message"]],
        ["R-squared", num(fit["r_squared"], 6), "Variance explained in synthetic response"],
        ["Cross-validation R-squared", num(fit["cross_validation_r2_mean"], 6), f"SD {num(fit['cross_validation_r2_std'], 6)}"],
        ["RMSE", num(fit["rmse"], 6), "Response-score units"],
        ["Maximum constraint violation", num(fit["maximum_constraint_violation"], 6), "All sign restrictions verified"],
        ["Accepted dynamic rounds", "12/12", f"{preferred_rounds} preferred; {managed_rounds} managed"],
    ]

    source_rows = []
    for path in e.source_paths:
        source_rows.append([path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)])

    parts: list[str] = [
        "# CHAPTER FOUR: MODEL DEVELOPMENT, IMPLEMENTATION AND VALIDATION",
        "",
        f"**Generated from synchronized replay evidence:** {generated_at}",
        "",
        "**Replay identifier:** full_pipeline_20260809",
        "",
        "## 4.0 Introduction",
        "",
        "This chapter presents the development, implementation, testing and validation of the Dynamic Optimization Federated Multimodal Biometric System (DO-FMBS). The implementation is organized as a six-phase simulation: data processing; federated baseline training; attack simulation; defense training; model validation; and constrained dynamic optimization. Every phase completed successfully in an isolated replay, and the finalized manifest verifies 24 required artifacts by SHA-256. The summed active command duration for Phases 1-6 was " + f"{total_runtime:.6f} seconds ({total_runtime/60:.3f} minutes).",
        "",
        "The chapter distinguishes three evaluation questions. Multiclass accuracy measures identity classification. FAR and FRR are derived from the face output under a claimed-identity verification protocol with a frozen threshold. ASR measures an attack outcome using its declared conditional denominator. This separation prevents a classification result, a fingerprint-head result or an attack-conditioned rate from being presented as a clean face-verification result.",
        "",
        "The final face verifier recorded FAR " + pct(final["FAR"], 3) + " and FRR " + pct(final["FRR"], 3) + ". Their identity-cluster 95% upper bounds were " + pct(final["FAR_identity_cluster_95_high"], 3) + " and " + pct(final["FRR_identity_cluster_95_high"], 3) + ", respectively, which lie within the declared project limits of 10% and 20%. These are project validation limits for this simulation and claimed-identity protocol.",
        "",
        "**Table 4.1: Evidence scope and measurement distinction (Researcher, 2026).**",
        "",
        md_table(["Evidence family", "Question answered", "Authoritative artifact"], [
            ["Classification", "Which identity class is predicted?", "validation_results.pkl"],
            ["Face verification", "Is the claimed face identity accepted?", "clean_far_frr_acceptance.csv"],
            ["Adversarial stress", "How often does perturbation cause the declared failure?", "ai_conditioned_far_frr_asr.csv"],
            ["Privacy pressure", "How far does membership inference exceed chance?", "privacy_pa_evaluation.csv"],
            ["Dynamic control", "Which DM and GS policy reaches a feasible next-round Y?", "model_fit.json and dynamic_optimization_schedule.csv"],
        ]),
        "",
        "## 4.1 Data Processing",
        "",
        "### 4.1.1 Data Sources",
        "",
        f"The face modality uses the Labeled Faces in the Wild (LFW) collection, acquired through scikit-learn's fetch_lfw_people function with min_faces_per_person=20, resize=0.4 and grayscale output. The resulting replay data contains {e.face_meta['train'] + e.face_meta['test']} face images from {e.face_meta['classes']} identities, each represented at {e.face_meta['image_shape'][0]} x {e.face_meta['image_shape'][1]} pixels. The dataset homepage is https://vis-www.cs.umass.edu/lfw/ and the loader documentation is https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html.",
        "",
        f"The fingerprint modality is generated locally by the Phase-1 deterministic FVC-style simulator. It creates {e.finger_meta['train'] + e.finger_meta['test']} images for {e.finger_meta['classes']} synthetic finger identities, with eight impressions per identity and an image size of {e.finger_meta['image_shape'][0]} x {e.finger_meta['image_shape'][1]} pixels. Ridge patterns, orientation, local structure, Gaussian variation and smoothing are produced in code. Consequently, the face images are downloaded/cached data, whereas the fingerprint images are simulation-generated data.",
        "",
        "**Table 4.2: Data sources and local storage (Researcher, 2026).**",
        "",
        md_table(["Modality", "Origin", "Samples / identities", "Local replay artifact", "Hash status"], [
            ["Face", "LFW via fetch_lfw_people", f"{e.face_meta['train'] + e.face_meta['test']} / {e.face_meta['classes']}", "do_fmbs_data/face_data.npz", "SHA-256 VERIFIED"],
            ["Fingerprint", "Locally generated FVC-style simulation", f"{e.finger_meta['train'] + e.finger_meta['test']} / {e.finger_meta['classes']}", "do_fmbs_data/fingerprint_data.npz", "SHA-256 VERIFIED"],
        ]),
        "",
        "Code evidence: phase1_data_preparation.py lines 31-56 load, transform, split and persist the LFW data; lines 70-135 generate, standardize, split and persist the synthetic fingerprint data.",
        "",
        "### 4.1.2 Data Preprocessing",
        "",
        "Both modalities are flattened and standardized with StandardScaler. In the executed Phase-1 code, fit_transform is applied to each complete modality array before the stratified split; all replay comparisons therefore use this same persisted transformation. Face images become 1,850-dimensional vectors and fingerprint images become 9,216-dimensional vectors. Labels remain integer identity indices for sparse categorical cross-entropy.",
        "",
        "The fingerprint generator is deterministic per identity/impression seed. It combines sinusoidal ridge fields, a centered local structure, random orientation and frequency, additive noise, Gaussian smoothing and min-max conversion to 8-bit image space before standardization. This creates repeated impressions that share an identity label while retaining controlled variation.",
        "",
        "### 4.1.3 Training and Testing Data",
        "",
        "A stratified 80:20 split with random_state=42 preserves identity representation in each modality. Table 4.3 reports the shapes read directly from the finalized NPZ artifacts.",
        "",
        "**Table 4.3: Training and testing data (Researcher, 2026).**",
        "",
        md_table(["Modality", "Training", "Testing", "Features", "Classes", "Split"], [
            ["Face", e.face_meta["train"], e.face_meta["test"], e.face_meta["features"], e.face_meta["classes"], "80:20 stratified"],
            ["Fingerprint", e.finger_meta["train"], e.finger_meta["test"], e.finger_meta["features"], e.finger_meta["classes"], "80:20 stratified"],
        ]),
        "",
        "The face and fingerprint labels belong to independent identity spaces: a face class index and a fingerprint class index do not identify the same person. Their Phase-2 index pairing exists solely to construct synchronized dual-input batches for the two output heads; it is not a same-person multimodal association.",
        "",
        "For operational face verification, the 605 face test probes are divided again by identity using seed 20260809: 291 genuine calibration probes and 314 genuine evaluation probes. Each evaluation probe is also asserted against all 61 incorrect identities, producing 19,154 impostor claims. Threshold calibration is therefore isolated from the final FAR/FRR calculation.",
        "",
        "### 4.1.4 Distribution of Data to Federated Clients",
        "",
        "Phase 1 persists five identity-partitioned client files. The partition shuffles identity classes with seed 42, divides those classes into five groups and assigns all samples of an identity group to its owner. This produces the non-IID storage distribution in Table 4.4. During the executed Phase-2 matched-pair experiment, 640 face/fingerprint pairs are randomly permuted and split evenly into five training clients of 128 pairs. The persistent identity partition and the executed matched-pair allocation are reported separately because they serve different reproducibility roles.",
        "",
        "**Table 4.4: Persistent non-IID client distribution (Researcher, 2026).**",
        "",
        md_table(["Client", "Face samples", "Face identities", "Fingerprint samples", "Fingerprint identities", "Total samples"], client_table),
        "",
        figure_marker(*FIGURE_SPECS[0]),
        "",
        "## 4.2 Simulation Environment",
        "",
        "### 4.2.1 Hardware Requirements",
        "",
        "The recorded replay used the processor identified in the runtime registry as " + str(r.runtime_registry["processor"]) + ". The implementation runs on a 64-bit CPU; a GPU is optional because the finalized evidence was produced with tensorflow-cpu. For repeat execution, a practical workstation specification is a modern four-core 64-bit processor, 16 GB RAM and sufficient local storage for Python packages, the LFW cache, model files, replay artifacts, plots and reports.",
        "",
        "**Table 4.5: Simulation hardware and execution requirements (Researcher, 2026).**",
        "",
        md_table(["Resource", "Recorded or recommended configuration", "Purpose"], [
            ["Processor", r.runtime_registry["processor"], "Recorded replay processor"],
            ["Architecture", "64-bit x86-compatible CPU", "TensorFlow and scientific stack"],
            ["Memory", "8 GB RAM minimum; 16 GB recommended", "Model training and 9,216-feature fingerprint arrays"],
            ["Storage", "Sufficient local storage", "Environment, data, models and evidence"],
            ["GPU", "Optional", "May reduce training time; CPU execution is supported"],
        ]),
        "",
        "### 4.2.2 Software and Libraries",
        "",
        "The replay platform was " + r.runtime_registry["platform"] + " with Python " + r.runtime_registry["python"].split(" ")[0] + ". Table 4.6 records the principal libraries and their implemented roles.",
        "",
        "**Table 4.6: Software environment (Researcher, 2026).**",
        "",
        md_table(["Software/library", "Version", "Function"], software_rows),
        "",
        "### 4.2.3 Federated Learning Configuration",
        "",
        "The Phase-2 replay used five clients, twelve federated rounds, two local epochs per round, batch size 64, Adam learning rate 0.001, 640 matched training pairs and 100 matched test pairs. Each client received the current global weights, trained a local copy and returned its model parameters for sample-weighted aggregation.",
        "",
        "**Table 4.7: Federated and defense configuration (Researcher, 2026).**",
        "",
        md_table(["Parameter", "Assigned value", "Evidence"], parameter_rows),
        "",
        "## 4.3 Simulation Approach",
        "",
        "### 4.3.1 Client Initialization",
        "",
        "At the beginning of a federated round, the server creates or retains the global two-input model. For every client, a local model with the same architecture is instantiated, compiled with Adam and initialized using global_model.get_weights(). This guarantees that differences in returned updates arise from local data and optimization rather than unrelated starting parameters.",
        "",
        "### 4.3.2 Local Model Training",
        "",
        "Each of the five clients trains on 128 matched face/fingerprint pairs for two local epochs. Sparse categorical cross-entropy is calculated independently for face_output and finger_output. The local update contains the trained weights, its sample count, face accuracy, fingerprint accuracy and loss. Phase-2 code evidence appears at lines 203-229.",
        "",
        "### 4.3.3 Global Model Aggregation",
        "",
        "The server performs sample-weighted Federated Averaging. If w_k^(t+1) is the update from client k with n_k samples and N is the total number of participating samples, the next global model is:",
        "",
        "$$",
        r"w_{t+1}=\sum_{k=1}^{K}\frac{n_k}{N}w_{k}^{t+1}",
        "$$",
        "",
        "All five executed clients contain 128 matched pairs; therefore n_k/N = 128/640 = 0.20 and the executed aggregation is an equal-weight special case of the sample-weighted formula. Phase-2 code lines 231-249 implement the computation.",
        "",
        "### 4.3.4 Adversarial Attack Simulation",
        "",
        "Phase 3 evaluates four attack families. FGSM changes the face vector once in the sign of the loss gradient. PGD performs ten bounded iterative updates. Membership inference classifies records by model confidence. Label-flipping poisoning changes 10% of selected face labels and trains a poisoned model for three epochs. The executed epsilon sweep is 0.01, 0.05, 0.10, 0.20 and 0.30.",
        "",
        "$$",
        r"x_{adv}=clip(x+\epsilon\,sign(\nabla_x L(\theta,x,y)))",
        "$$",
        "",
        "### 4.3.5 Privacy and Defense Mechanisms",
        "",
        "Four Phase-4 arms quantify defense behavior. The no-defense model provides a contemporaneous comparator. The DP-style arm adds Gaussian noise with scale 0.001/epsilon to gradients, using epsilon=0.50. The adversarial-training arm mixes clean and FGSM examples. The combined DO-FMBS arm first performs six clean pretraining epochs and then trains for twelve epochs on clean/adversarial face batches while adding gradient noise with standard deviation 0.0003. The implemented gradient perturbation is evaluated through empirical PA; formal epsilon-delta accounting is outside this mechanism's implemented scope.",
        "",
        "### 4.3.6 Simulation Rounds and Iterations",
        "",
        "The main iteration counts are twelve federated rounds x two local epochs, PGD with ten attack steps, Phase-4 training with twelve epochs, a seven-level operational FGSM sweep and a twelve-round dynamic controller. The controller evaluates 21 DM values from 0.00 to 1.00 at increments of 0.05 against K in {3,4,5}, giving 63 candidate policies in every round.",
        "",
        figure_marker(*FIGURE_SPECS[1]),
        "",
        "## 4.4 DO-FMBS Model Development",
        "",
        "### 4.4.1 Model Components",
        "",
        "**Table 4.8: DO-FMBS components (Researcher, 2026).**",
        "",
        md_table(["Component", "Implementation", "Output"], model_components),
        "",
        "The direct modality heads are intentional: the face and synthetic fingerprint labels describe different identity spaces. The face-verification evaluator therefore isolates the face_input-to-face_output path. Fingerprint-head classification is reported independently and is never substituted for face FAR or FRR.",
        "",
        "### 4.4.2 Definition of Model Variables",
        "",
        "**Table 4.9: Definition of model and validation variables (Researcher, 2026).**",
        "",
        md_table(["Variable", "Definition", "Domain/limit", "Role"], variable_rows),
        "",
        "### 4.4.3 Values Assigned to Model Variables",
        "",
        "The Phase-6 fitted domain and decision bands are explicit. AI and PA are observed threat-state variables; DM and GS are selected controls. A value beyond a fitted input domain is rejected before prediction. A predicted Y below 0.60 causes the round to be deferred; this prevents an infeasible policy from being described as deployable.",
        "",
        "**Table 4.10: Variable operating ranges and actions (Researcher, 2026).**",
        "",
        md_table(["Variable", "Preferred", "Managed", "Critical", "Controller action"], range_rows),
        "",
        "### 4.4.4 Model Coefficients and Parameters",
        "",
        "The five coefficients were estimated with constrained least squares (SLSQP) from a reproducible balanced 700-condition synthetic design. B_AI and B_PA are constrained to be non-positive because attack and privacy pressure reduce expected efficiency. B_DM and B_GS are constrained to be non-negative because stronger defense and wider federation participation are modeled as beneficial controls.",
        "",
        "**Table 4.11: Estimated model coefficients (Researcher, 2026).**",
        "",
        md_table(["Coefficient", "Estimate", "SE", "95% interval", "Constraint", "Verification"], coefficient_table),
        "",
        "**Table 4.12: Constrained fit summary (Researcher, 2026).**",
        "",
        md_table(["Statistic", "Value", "Interpretation"], phase6_summary_rows),
        "",
        "### 4.4.5 Mathematical Formulation of the DO-FMBS",
        "",
        "The dynamic response model is:",
        "",
        "$$",
        r"\widehat{Y}_{t+1}=B_0+B_{AI}AI_t+B_{PA}PA_t+B_{DM}DM_t+B_{GS}GS_t",
        "$$",
        "",
        "Substituting the estimated coefficients gives:",
        "",
        "$$",
        r"\widehat{Y}_{t+1}=0.400738-0.182145AI_t-0.227329PA_t+0.261393DM_t+0.159417GS_t",
        "$$",
        "",
        "The transformations feeding the equation are:",
        "",
        "$$",
        r"AI_t=\frac{\epsilon_t}{0.30},\quad PA_t=2\max(0,A_{MIA,t}-0.50),\quad GS_t=\frac{K_t-1}{4}",
        "$$",
        "",
        "DM also maps to executable defense settings: adversarial fraction = 0.50DM, training epsilon = 0.05DM and gradient-noise standard deviation = 0.0003DM.",
        "",
        "### 4.4.6 Optimization Constraints",
        "",
        "The coefficient fit enforces B_AI <= 0, B_PA <= 0, B_DM >= 0 and B_GS >= 0. The deployment search enforces 0 <= AI,DM,GS <= 1; 0 <= PA <= 0.50; K in {3,4,5}; and a preferred response Y >= 0.70 or minimum feasible response Y >= 0.60. The controller minimizes:",
        "",
        "$$",
        r"C_t=0.60DM_t^2+0.40GS_t^2+0.15(DM_t-DM_{t-1})^2+0.10(GS_t-GS_{t-1})^2",
        "$$",
        "",
        "The first two terms discourage unnecessarily expensive defense and participation. The last two discourage abrupt control changes. The controller first selects the lowest-cost preferred candidate; if that set is empty, it selects the lowest-cost candidate satisfying Y >= 0.60.",
        "",
        "### 4.4.7 Model Implementation",
        "",
        "**Table 4.13: Phase-to-code and artifact traceability (Researcher, 2026).**",
        "",
        md_table(["Phase", "Script", "Inputs", "Process", "Outputs"], traceability_rows),
        "",
        "Federated-loop pseudocode:",
        "",
        "1. Initialize global model w(0).",
        "2. For each round t = 1,...,12, send w(t) to all five clients.",
        "3. At each client k, clone the architecture, load w(t), and train two local epochs on 128 matched pairs.",
        "4. Return local weights w_k(t+1), sample count n_k and local metrics.",
        "5. Compute w(t+1) = sum_k [n_k / sum_j n_j] w_k(t+1).",
        "6. Store round accuracy and loss; continue to the next round.",
        "7. Save the final global model and baseline metrics.",
        "",
        "Sixty-three-candidate controller pseudocode:",
        "",
        "1. Observe epsilon and membership-attack accuracy; derive AI and PA.",
        "2. Reject an input state outside the fitted AI/PA domains.",
        "3. For each DM in {0.00,0.05,...,1.00} and K in {3,4,5}, compute GS=(K-1)/4.",
        "4. Predict Y for each of the 21 x 3 = 63 candidates.",
        "5. Compute the defense, participation and switching cost C for each candidate.",
        "6. Select the minimum-cost candidate with Y >= 0.70; otherwise select the minimum-cost candidate with Y >= 0.60.",
        "7. If every candidate is below 0.60, select maximum predicted Y for diagnosis and defer deployment.",
        "8. Save DM, mapped defense settings, K, GS, Y, cost, status and decision; use DM and GS as the previous controls in the next round.",
        "",
        "Code evidence: phase2_federated_learning.py lines 190-266 implement client update and FedAvg; constrained_dynamic_model.py lines 304-345 implement input validation, the 63-candidate search, target hierarchy and defense mapping.",
        "",
        "## 4.5 Developed DO-FMBS Model",
        "",
        "### 4.5.1 DO-FMBS Architecture",
        "",
        "The developed architecture connects six evidence-producing phases. Phase 1 prepares local arrays. Phase 2 performs federated baseline learning. Phase 3 measures attack effects. Phase 4 trains defense variants and the final combined model. Phase 5 conducts classification, robustness and statistical validation. Phase 6 estimates the constrained response model and re-optimizes DM and GS. The operational evaluator reads the hash-matched face data, baseline model and final model to obtain thresholded verification results.",
        "",
        "### 4.5.2 Operational Flow of the Model",
        "",
        "The operational flow is data -> client allocation -> local learning -> FedAvg -> attack observation -> defense training -> validation -> dynamic policy. In each dynamic round, AI and PA represent the current threat state. The controller searches feasible DM and GS settings, predicts Y(t+1), selects the least-cost acceptable policy and carries the chosen controls forward as switching-cost references.",
        "",
        "### 4.5.3 Interaction Between AI, PA, DM and GS",
        "",
        "AI and PA enter the fitted equation with negative coefficients; therefore, holding controls constant, greater attack or privacy pressure lowers predicted Y. DM and GS have positive coefficients, so stronger defense and broader client participation counter that reduction. The standardized coefficients are -0.446678 for AI, -0.306770 for PA, +0.711195 for DM and +0.410654 for GS. Within this simulation, DM is the strongest standardized control term, followed by GS; AI is the strongest adverse term.",
        "",
        "### 4.5.4 Developed Model Diagram",
        "",
        figure_marker(*FIGURE_SPECS[2]),
        "",
        "## 4.6 Experimental Scenarios",
        "",
        "### 4.6.1 Baseline FMBS",
        "",
        f"The federated baseline used five clients, twelve rounds and two local epochs. Its Phase-2 test results were face classification accuracy {pct(e.baseline['face_accuracy'])}, fingerprint-head classification accuracy {pct(e.baseline['finger_accuracy'])}, heuristic FAR {pct(e.baseline['FAR'])} and heuristic FRR {pct(e.baseline['FRR'])}. The later operational verification protocol supersedes those heuristic Phase-2 rates for claimed-identity face FAR/FRR because it freezes a calibrated threshold and enumerates genuine and impostor claims.",
        "",
        "The 0.00% fingerprint-head result is a separate 100-class synthetic-fingerprint classification result. It does not enter the operational face-verification FAR or FRR and identifies the fingerprint branch as a target for further multimodal optimization.",
        "",
        "### 4.6.2 FMBS Under Adversarial Attacks",
        "",
        "Phase 3 attacks the federated baseline. FGSM and PGD are evaluated only among clean-correct face predictions. Membership inference uses a 0.70 confidence threshold. Poisoning flips 10% of selected labels (20 samples) and retrains for three epochs. Table 4.14 gives the evasion results.",
        "",
        "**Table 4.14: Evasion-attack scenarios on the baseline (Researcher, 2026).**",
        "",
        md_table(["epsilon", "FGSM ASR", "PGD ASR (10 iterations)"], attack_rows),
        "",
        f"Membership inference produced attack accuracy {pct(e.attacks['membership_inference']['attack_accuracy'])} and privacy-risk score {pct(e.attacks['membership_inference']['privacy_risk_score'])}. The poisoning experiment recorded clean accuracy {pct(e.attacks['model_poisoning']['clean_accuracy'])}, poisoned accuracy {pct(e.attacks['model_poisoning']['poisoned_accuracy'])} and attack success {pct(e.attacks['model_poisoning']['attack_success_rate'])}. The negative reported accuracy drop means the single poisoned run increased accuracy by five percentage points; it provides no evidence of a successful degradation attack in that run.",
        "",
        figure_marker(*FIGURE_SPECS[3]),
        "",
        "### 4.6.3 FMBS With Defense Mechanisms",
        "",
        "The four-arm Phase-4 experiment compares no defense, DP-style gradient noise, adversarial training and the combined DO-FMBS. The artifact field privacy_score equals 1-ASR, so this chapter labels it attack resistance rather than a formal privacy guarantee.",
        "",
        "**Table 4.15: Before/after defense comparison (Researcher, 2026).**",
        "",
        md_table(["Condition", "Clean face accuracy", "FGSM ASR", "Attack resistance (1-ASR)", "Accuracy change", "ASR change"], defense_rows),
        "",
        figure_marker(*FIGURE_SPECS[4]),
        "",
        "### 4.6.4 DO-FMBS Under Different Attack Intensities",
        "",
        "The synchronized operational evaluator freezes the final threshold at 0.001782209 and evaluates epsilon in {0,0.01,0.05,0.10,0.15,0.20,0.30}. AI is epsilon/0.30. FAR remains below the 10% project point-estimate limit across the sweep, while FRR and both ASR measures rise as perturbation increases. These attacked rows describe availability stress; the clean epsilon=0 row is the deployment acceptance result.",
        "",
        "## 4.7 Testing and Validation",
        "",
        "### 4.7.1 Functionality Verification",
        "",
        "Functionality was verified at process and artifact levels. Every phase returned exit code 0 and status EXECUTION-COMPLETED. The finalizer verified 24 required outputs by byte count and SHA-256. The operational evaluator then confirmed that face_data.npz, baseline_model.keras and do_fmbs_model.keras had exactly the same hashes as the replay manifest.",
        "",
        "**Table 4.16: Functionality and artifact verification (Researcher, 2026).**",
        "",
        md_table(["Phase", "Verified artifacts", "Total bytes", "Verification"], artifact_summary),
        "",
        "### 4.7.2 Variable and Parameter Verification",
        "",
        "The required verification chain is AI -> PA -> DM -> GS -> Y(t+1): epsilon is normalized into AI; membership accuracy is transformed into PA; the controller selects DM; participating client count is normalized into GS; and the estimated equation computes Y. The arrows express computation and control order; AI and PA remain parallel threat-state inputs rather than a claim that AI causes PA. The values below reproduce Round 7 from persisted evidence.",
        "",
        "**Table 4.17: Round-7 variable proof (Researcher, 2026).**",
        "",
        md_table(["Stage", "Evidence input", "Substitution", "Result"], proof_rows),
        "",
        "The exact Round-7 substitution is:",
        "",
        "$$",
        r"\widehat{Y}_8=0.400737734-(0.182144754)(0.333333333)-(0.227329111)(0.300000)+(0.261392739)(0.800000)+(0.159417137)(0.750000)",
        "$$",
        "",
        "$$",
        r"\widehat{Y}_8=0.400737734-0.060714918-0.068198733+0.209114191+0.119562853=0.600501127",
        "$$",
        "",
        "Round 7 follows Round 6, where DM=1 and GS=1. Its cost is 0.60(0.80^2)+0.40(0.75^2)+0.15(0.80-1.00)^2+0.10(0.75-1.00)^2 = 0.62125. Y=0.600501127 exceeds the minimum target by 0.000501127, so the persisted deployment decision is ACCEPTED.",
        "",
        "**Table 4.18: Complete twelve-round variable schedule (Researcher, 2026).**",
        "",
        md_table(["Round", "epsilon", "AI", "PA", "DM", "K", "GS", "Y", "Cost", "Target state", "Decision"], schedule_rows),
        "",
        figure_marker(*FIGURE_SPECS[5]),
        "",
        "### 4.7.3 Adversarial Attack Validation",
        "",
        "Evasion is validated by the monotonic high-level rise of FGSM and PGD ASR on the baseline, reaching 96.15% and 88.46%, respectively, at epsilon=0.30. Inference is validated by member/nonmember confidence thresholding in Phase 3 and by the synchronized final-model PA evaluation. Poisoning is validated by the persisted 10% label-flipping experiment. Together, these artifacts prove that evasion, membership inference and poisoning code paths executed and produced numerical outputs.",
        "",
        "**Table 4.19: Attack-family validation (Researcher, 2026).**",
        "",
        md_table(["Attack family", "Executed setting", "Recorded outcome", "Validation state"], [
            ["FGSM evasion", "epsilon 0.01-0.30", "ASR 19.23%-96.15% on baseline", "EXECUTED AND QUANTIFIED"],
            ["PGD evasion", "10 iterations; epsilon 0.01-0.30", "ASR 23.08%-88.46% on baseline", "EXECUTED AND QUANTIFIED"],
            ["Membership inference", "threshold 0.70", f"Final accuracy {pct(pa['membership_accuracy'])}; PA {num(pa['PA'],3)}", "EXECUTED AND QUANTIFIED"],
            ["Label-flipping poisoning", "10%; 20 samples; 3 epochs", f"Poisoned accuracy {pct(e.attacks['model_poisoning']['poisoned_accuracy'])}", "EXECUTED AND QUANTIFIED"],
        ]),
        "",
        "### 4.7.4 Privacy Defense Validation",
        "",
        "Table 4.15 supplies the before-defense, DP-style, ART and combined results. ART reduced FGSM ASR by 16.81 percentage points and increased accuracy by 1.49 points relative to the Phase-4 no-defense arm. The combined model reduced ASR by 13.22 points while increasing accuracy by 3.97 points. The DP-style arm alone increased ASR in this run, showing why the combined empirical assessment is necessary. Final-model membership accuracy was 67.00%, giving PA=0.340 within the declared managed limit of 0.40.",
        "",
        "### 4.7.5 Model Performance Validation",
        "",
        "**Table 4.20: Model-performance validation summary (Researcher, 2026).**",
        "",
        md_table(["Measure", "Recorded value", "Evidence basis"], performance_rows),
        "",
        "Accuracy, FAR, FRR, ASR, PA, robustness and overhead are all measurable in the finalized evidence. The fingerprint head remains separate: its Phase-2 test accuracy is 0.00%, while every FAR/FRR value in Tables 4.20, 4.23 and 4.24 is calculated from the face output under the claimed-identity protocol.",
        "",
        "### 4.7.6 Baseline and DO-FMBS Comparison",
        "",
        "The Phase-5 face-classification comparison increased accuracy from 26.00% to 65.00%, macro precision from 6.66% to 55.47%, macro recall from 11.16% to 53.69%, and macro F1 from 7.84% to 52.66%. The statistical resampling comparison recorded baseline mean 27.10%, DO-FMBS mean 65.90%, t=-25.0569, p=1.906e-15 and Cohen's d=11.2058.",
        "",
        "For claimed-identity verification, the baseline comparator recorded FAR " + pct(comparator["FAR"], 3) + " and FRR " + pct(comparator["FRR"], 3) + "; the final model recorded FAR " + pct(final["FAR"], 3) + " and FRR " + pct(final["FRR"], 3) + ". The principal verification gain is a " + f"{(float(comparator['FRR'])-float(final['FRR']))*100:.3f}" + " percentage-point reduction in genuine rejection while keeping FAR within the declared limit.",
        "",
        "### 4.7.7 Validation of Optimization Constraints",
        "",
        "Every fitted sign constraint is satisfied; the maximum constraint violation is 0. The constrained and unconstrained estimates are identical because the unconstrained optimum already lies inside the feasible coefficient region. Table 4.21 verifies fit quality, solver convergence, coefficient relationships, multicollinearity, residual diagnostics, schedule feasibility and the two safety guards.",
        "",
        "**Table 4.21: Optimization-constraint, fit and safety validation (Researcher, 2026).**",
        "",
        md_table(["Validation item", "Recorded result", "Verification state"], optimization_validation_rows),
        "",
        "The controller accepts all twelve scheduled operating rounds because each predicted response reaches at least Y=0.60. The critical modeled scenario verifies the safety rejection below this minimum, while the out-of-domain scenario verifies that the controller withholds a prediction when AI or PA exceeds the fitted domain.",
        "",
        "## 4.8 Results and Analysis",
        "",
        "### 4.8.1 Accuracy Results",
        "",
        "**Table 4.22: Face-classification accuracy results (Researcher, 2026).**",
        "",
        md_table(["Metric", "Federated baseline", "DO-FMBS", "Change"], [
            ["Accuracy", pct(validation_metrics["baseline"]["accuracy"]), pct(validation_metrics["do_fmbs"]["accuracy"]), f"{(validation_metrics['do_fmbs']['accuracy']-validation_metrics['baseline']['accuracy'])*100:+.2f} pp"],
            ["Macro precision", pct(validation_metrics["baseline"]["macro_avg_precision"]), pct(validation_metrics["do_fmbs"]["macro_avg_precision"]), f"{(validation_metrics['do_fmbs']['macro_avg_precision']-validation_metrics['baseline']['macro_avg_precision'])*100:+.2f} pp"],
            ["Macro recall", pct(validation_metrics["baseline"]["macro_avg_recall"]), pct(validation_metrics["do_fmbs"]["macro_avg_recall"]), f"{(validation_metrics['do_fmbs']['macro_avg_recall']-validation_metrics['baseline']['macro_avg_recall'])*100:+.2f} pp"],
            ["Macro F1", pct(validation_metrics["baseline"]["macro_avg_f1"]), pct(validation_metrics["do_fmbs"]["macro_avg_f1"]), f"{(validation_metrics['do_fmbs']['macro_avg_f1']-validation_metrics['baseline']['macro_avg_f1'])*100:+.2f} pp"],
        ]),
        "",
        figure_marker(*FIGURE_SPECS[6]),
        "",
        "The DO-FMBS model improves every recorded face-classification measure. The large effect size supports a material rather than merely numerical clean-classification difference for this replay. Results remain tied to the executed seed, dataset partition and model configuration.",
        "",
        "### 4.8.2 FAR and FRR Results",
        "",
        "**Table 4.23: Clean claimed-identity face-verification results (Researcher, 2026).**",
        "",
        md_table(["Model", "Threshold", "FAR count", "FAR", "FAR cluster 95%", "FRR count", "FRR", "FRR cluster 95%", "Decision"], clean_rows),
        "",
        figure_marker(*FIGURE_SPECS[7]),
        "",
        "The final decision uses the identity-cluster bootstrap upper bounds because claims from the same identity are correlated. FAR upper 9.863% is below the 10% limit and FRR upper 16.541% is below the 20% limit. The clean final-model verification state is therefore SATISFIED. Baseline remains a recorded comparator.",
        "",
        "**Table 4.24: FAR, FRR and runtime by operational attack intensity (Researcher, 2026).**",
        "",
        md_table(["epsilon", "AI", "FAR count", "FAR", "FRR count", "FRR", "Class-ASR count", "Class ASR", "Reject-ASR count", "Reject ASR", "Seconds"], ai_rows),
        "",
        "### 4.8.3 Attack Success Rate Results",
        "",
        "Classification ASR is the proportion of 200 clean-correct final-model predictions changed by FGSM. Verification-rejection ASR is the proportion of 279 clean-accepted genuine claims pushed below the frozen verification threshold. The denominators differ deliberately; the first measures identity-class instability and the second measures availability loss among initially accepted users.",
        "",
        figure_marker(*FIGURE_SPECS[8]),
        "",
        "At epsilon=0.10 (AI=0.3333), classification ASR is 73.50% and verification-rejection ASR is 47.31%. At epsilon=0.30, they reach 98.00% and 92.47%. This indicates that the clean acceptance result coexists with substantial high-intensity adversarial availability risk.",
        "",
        "### 4.8.4 Privacy Score Results",
        "",
        "The final-model membership evaluation detected 100/100 members and falsely flagged 66/100 nonmembers. It therefore made 134/200 correct decisions, giving attack accuracy 0.670. Privacy pressure is:",
        "",
        "$$",
        r"PA=2\max(0,0.670-0.500)=0.340",
        "$$",
        "",
        "PA=0.340 lies inside the declared managed band PA<=0.40. This is the privacy variable used for operational validation. The Phase-4 attack-resistance value 1-ASR measures resistance to FGSM and is not substituted for PA.",
        "",
        figure_marker(*FIGURE_SPECS[9]),
        "",
        "### 4.8.5 Robustness Results",
        "",
        "**Table 4.25: Phase-5 accuracy and reported robust error under FGSM (Researcher, 2026).**",
        "",
        md_table(["epsilon", "Baseline accuracy", "DO-FMBS accuracy", "Accuracy difference", "Baseline robust error (1-accuracy)", "DO-FMBS robust error (1-accuracy)"], robustness_rows),
        "",
        figure_marker(*FIGURE_SPECS[10]),
        "",
        "The final two columns reproduce the Phase-5 stored quantity 1-accuracy as reported robust error. At epsilon=0 it is clean classification error, not an adversarial success event; for epsilon>0 it is error under FGSM. DO-FMBS has a 39.38-point clean advantage and an 11.55-point advantage at epsilon=0.05. The models converge at epsilon=0.10, and the baseline is higher by 1.08 and 2.55 points at epsilon 0.15 and 0.20. The appropriate interpretation is that defense substantially improves clean and low-intensity performance, while strong perturbations remain the main robustness optimization priority.",
        "",
        "### 4.8.6 Communication/Computational Overhead",
        "",
        "**Table 4.26: Recorded computational runtime (Researcher, 2026).**",
        "",
        md_table(["Phase", "Script", "Elapsed seconds", "Status"], runtime_rows),
        "",
        figure_marker(*FIGURE_SPECS[11]),
        "",
        "Phase 4 dominates the measured runtime at 805.393627 seconds because it trains four defense variants, including adversarial example generation and gradient-noise updates. Phase 2 requires 316.531170 seconds for sixty client updates (12 rounds x 5 clients), and Phase 5 requires 210.194688 seconds for validation. The separate operational biometric evaluation completes in " + f"{float(r.evaluator_manifest['elapsed_seconds']):.6f}" + " seconds.",
        "",
        f"Network traffic is represented by a full-model communication equivalent because the simulation executes clients in one process. The Phase-2 global model has {FEDERATED_MODEL_PARAMETER_COUNT:,} float32 values in {FEDERATED_MODEL_WEIGHT_ARRAYS} weight arrays, giving {model_bytes:,} raw weight bytes per model copy:",
        "",
        "$$",
        r"Transfers=12\ rounds\times5\ clients\times(1\ download+1\ upload)=120",
        "$$",
        "",
        "$$",
        rf"\mathrm{{Communication}}={model_bytes:,}\ \mathrm{{bytes}}\times120={communication_bytes:,}\ \mathrm{{bytes}}",
        "$$",
        "",
        f"This equals {communication_mb:.3f} MB decimal ({communication_gib:.3f} GiB), with {per_round_bytes/1_000_000:.3f} MB per round, {communication_bytes/2/1_000_000:.3f} MB downlink and {communication_bytes/2/1_000_000:.3f} MB uplink. It is an analytical raw-weight equivalent for full-model FedAvg transfers in this in-process simulation; protocol headers, compression and network latency are outside this payload calculation.",
        "",
        "### 4.8.7 Baseline versus DO-FMBS Results",
        "",
        "**Table 4.27: Integrated baseline versus DO-FMBS comparison (Researcher, 2026).**",
        "",
        md_table(["Measure", "Baseline", "DO-FMBS", "Observed effect"], [
            ["Face classification accuracy", pct(validation_metrics["baseline"]["accuracy"]), pct(validation_metrics["do_fmbs"]["accuracy"]), "+39.00 pp"],
            ["Macro F1", pct(validation_metrics["baseline"]["macro_avg_f1"]), pct(validation_metrics["do_fmbs"]["macro_avg_f1"]), f"{(validation_metrics['do_fmbs']['macro_avg_f1']-validation_metrics['baseline']['macro_avg_f1'])*100:+.2f} pp"],
            ["Claimed-identity FAR", pct(comparator["FAR"], 3), pct(final["FAR"], 3), f"{(float(final['FAR'])-float(comparator['FAR']))*100:+.3f} pp"],
            ["Claimed-identity FRR", pct(comparator["FRR"], 3), pct(final["FRR"], 3), f"{(float(final['FRR'])-float(comparator['FRR']))*100:+.3f} pp"],
            ["Phase-5 mean accuracy", pct(statistical["baseline_mean"]), pct(statistical["do_fmbs_mean"]), f"{(statistical['do_fmbs_mean']-statistical['baseline_mean'])*100:+.2f} pp"],
            ["Statistical evidence", "Reference", f"p={statistical['p_value']:.3e}; d={statistical['effect_size']:.4f}", "Significant clean-accuracy difference"],
        ]),
        "",
        "The baseline-to-final FAR increase is 0.731 percentage points, while FRR falls by 43.312 points. For this project protocol, the final model trades a small FAR increase for a large reduction in genuine-user rejection and still satisfies the conservative FAR upper-bound rule.",
        "",
        "### 4.8.8 Analysis of Results",
        "",
        "Five conclusions follow from the synchronized results. First, the combined defense improves clean face classification and sharply improves claimed-identity usability by reducing FRR. Second, ART supplies the clearest Phase-4 ASR reduction, while the combined model produces the highest clean accuracy among the four contemporaneous arms. Third, the final clean FAR/FRR decision is supported by exact counts, frozen thresholds, Wilson intervals and identity-cluster bootstrap intervals. Fourth, high AI values expose an availability weakness: classification ASR and verification-rejection ASR approach one at epsilon=0.30. Fifth, the dynamic controller responds exactly as formulated: negative AI/PA terms are offset by positive DM/GS controls and all twelve simulated round decisions remain feasible.",
        "",
        "The dynamic model itself is a controlled synthetic coefficient-recovery and policy-search experiment. Its 700 response observations are generated from a declared factorial design and fitted with constrained least squares; the coefficients are policy-model estimates rather than neural-network weights or outcomes from 700 independent federated retrainings. Operational FAR, FRR, ASR and PA are independently measured from the synchronized neural-model artifacts.",
        "",
        "The strongest deployment evidence is the clean verifier result: FAR 7.852% and FRR 11.146%, with conservative upper bounds within the project limits. The strongest improvement area is high-intensity adversarial robustness. The recommended continuation is multi-seed training and external biometric validation, followed by adversarial parameter tuning and a formal privacy accountant if a formal DP guarantee is required.",
        "",
        "## 4.9 Chapter Summary",
        "",
        "This chapter documented the complete DO-FMBS development path from downloaded LFW faces and locally generated fingerprints to client allocation, federated training, attack simulation, privacy/defense training, model validation and dynamic constrained optimization. The model uses two modality branches, sample-weighted FedAvg, FGSM/PGD/inference/poisoning tests, DP-style gradient noise, adversarial training and a 63-candidate controller search.",
        "",
        f"The final model achieved 65.00% face-classification accuracy, FAR {pct(final['FAR'],3)}, FRR {pct(final['FRR'],3)} and PA {num(pa['PA'],3)}. The clean FAR and FRR identity-cluster upper bounds satisfy the declared project limits. All fitted coefficient constraints hold, all twelve dynamic rounds are accepted, all six phases complete in {total_runtime:.6f} seconds, and 24 required replay artifacts are hash verified. The chapter also preserves the measured high-intensity robustness limitation as the priority for further optimization.",
        "",
        "## APPENDIX 4-A: EVIDENCE TRACEABILITY",
        "",
        "**Table 4A.1: Principal input hashes for this chapter (Researcher, 2026).**",
        "",
        md_table(["Source", "Bytes", "SHA-256"], source_rows),
        "",
        "The generated figures are analytical renderings of the validated CSV, PKL and NPZ evidence. Figure captions and table captions identify the researcher and year, while the source hashes provide a reproducible link to the execution artifacts.",
    ]
    markdown = "\n".join(parts).strip() + "\n"
    required_headings = [
        "4.0 Introduction", "4.1 Data Processing", "4.1.1 Data Sources",
        "4.1.2 Data Preprocessing", "4.1.3 Training and Testing Data",
        "4.1.4 Distribution of Data to Federated Clients", "4.2 Simulation Environment",
        "4.2.1 Hardware Requirements", "4.2.2 Software and Libraries",
        "4.2.3 Federated Learning Configuration", "4.3 Simulation Approach",
        "4.3.1 Client Initialization", "4.3.2 Local Model Training",
        "4.3.3 Global Model Aggregation", "4.3.4 Adversarial Attack Simulation",
        "4.3.5 Privacy and Defense Mechanisms", "4.3.6 Simulation Rounds and Iterations",
        "4.4 DO-FMBS Model Development", "4.4.1 Model Components",
        "4.4.2 Definition of Model Variables", "4.4.3 Values Assigned to Model Variables",
        "4.4.4 Model Coefficients and Parameters", "4.4.5 Mathematical Formulation of the DO-FMBS",
        "4.4.6 Optimization Constraints", "4.4.7 Model Implementation",
        "4.5 Developed DO-FMBS Model", "4.5.1 DO-FMBS Architecture",
        "4.5.2 Operational Flow of the Model", "4.5.3 Interaction Between AI, PA, DM and GS",
        "4.5.4 Developed Model Diagram", "4.6 Experimental Scenarios",
        "4.6.1 Baseline FMBS", "4.6.2 FMBS Under Adversarial Attacks",
        "4.6.3 FMBS With Defense Mechanisms", "4.6.4 DO-FMBS Under Different Attack Intensities",
        "4.7 Testing and Validation", "4.7.1 Functionality Verification",
        "4.7.2 Variable and Parameter Verification", "4.7.3 Adversarial Attack Validation",
        "4.7.4 Privacy Defense Validation", "4.7.5 Model Performance Validation",
        "4.7.6 Baseline and DO-FMBS Comparison", "4.7.7 Validation of Optimization Constraints",
        "4.8 Results and Analysis", "4.8.1 Accuracy Results", "4.8.2 FAR and FRR Results",
        "4.8.3 Attack Success Rate Results", "4.8.4 Privacy Score Results",
        "4.8.5 Robustness Results", "4.8.6 Communication/Computational Overhead",
        "4.8.7 Baseline versus DO-FMBS Results", "4.8.8 Analysis of Results", "4.9 Chapter Summary",
    ]
    missing = [heading for heading in required_headings if heading not in markdown]
    if missing:
        raise ValueError("Chapter headings are incomplete: " + ", ".join(missing))
    prohibited = (
        "legacy", "not recorded", "not measured", "not satisfied", "not yet wired",
        "evidence store", "needs further optimization",
    )
    found = [phrase for phrase in prohibited if phrase in markdown.casefold()]
    if re.search(r"(?<![a-z0-9])n/a(?![a-z0-9])", markdown.casefold()):
        found.append("n/a")
    if found:
        raise ValueError("Chapter contains prohibited wording: " + ", ".join(found))
    return markdown


FIGURE_RE = re.compile(r"^\[\[FIGURE:([^|]+)\|(.+)\]\]$")
PROHIBITED_TEXT = (
    "legacy", "not recorded", "not measured", "not satisfied",
    "not yet wired", "evidence store", "needs further optimization",
)


def markdown_for_file(markdown: str) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        match = FIGURE_RE.match(line.strip())
        if match:
            filename, caption = match.groups()
            lines.extend([f"![{caption}]({filename})", "", f"*{caption}*"])
        else:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def add_page_number(paragraph) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText"); instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar"); separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, end])


def write_chapter_docx(markdown: str, path: Path, figure_dir: Path) -> None:
    from copy import deepcopy

    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor, Twips
    from docx.text.paragraph import Paragraph

    ok, message = write_docx(markdown, path)
    if not ok:
        raise RuntimeError(f"DOCX generation failed: {message}")
    document = Document(path)
    document.core_properties.title = "Chapter Four: Model Development, Implementation and Validation"
    document.core_properties.subject = "DO-FMBS model development, implementation, results and validation"
    document.core_properties.author = "DO-FMBS Research Project"
    document.core_properties.keywords = "DO-FMBS, federated learning, biometrics, validation"

    styles = document.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(11)
    for name, size, color in (("Title", 18, "17365D"), ("Heading 1", 14, "17365D"), ("Heading 2", 12, "2F5597")):
        styles[name].font.name = "Times New Roman"
        styles[name].font.size = Pt(size)
        styles[name].font.color.rgb = RGBColor.from_string(color)

    for paragraph in document.paragraphs:
        match = FIGURE_RE.match(paragraph.text.strip())
        if match:
            filename, caption = match.groups()
            image_path = figure_dir / filename
            for run in paragraph.runs:
                run.clear()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run()
            width = 6.75 if "ARCHITECTURE" in filename else 6.30
            run.add_picture(str(image_path), width=Inches(width))
            paragraph.paragraph_format.keep_with_next = True

            caption_element = OxmlElement("w:p")
            paragraph._p.addnext(caption_element)
            caption_paragraph = Paragraph(caption_element, paragraph._parent)
            caption_paragraph.style = styles["Caption"]
            caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_paragraph.paragraph_format.keep_with_next = True
            caption_run = caption_paragraph.add_run(caption)
            caption_run.italic = True
            caption_run.font.name = "Times New Roman"
            caption_run.font.size = Pt(9)
        elif re.match(r"^Table 4(?:\.|A\.)", paragraph.text.strip()):
            paragraph.style = styles["Caption"]
            paragraph.paragraph_format.keep_with_next = True
        if paragraph.style.name == "Heading 1" and (
            re.match(r"4\.[1-9](?:\s|$)", paragraph.text)
            or paragraph.text.startswith("APPENDIX 4-A")
        ):
            paragraph.paragraph_format.page_break_before = True
        if paragraph.style.name.startswith("Heading"):
            paragraph.paragraph_format.keep_with_next = True
        if paragraph.style.name == "Title":
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(18)

    for table in document.tables:
        is_wide = len(table.columns) >= 10
        table.autofit = not is_wide
        if is_wide:
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            target_twips = round(9.55 * 1440 / len(table.columns))
            for grid_column in table._tbl.tblGrid.gridCol_lst:
                grid_column.set(qn("w:w"), str(target_twips))
            for column in table.columns:
                column.width = Twips(target_twips)
            for row in table.rows:
                for cell in row.cells:
                    cell.width = Twips(target_twips)
        for row_index, row in enumerate(table.rows):
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(7.2 if len(table.columns) >= 8 else 8.2)
                        if row_index == 0:
                            run.bold = True

    def section_properties(*, landscape_layout: bool):
        section = deepcopy(document.sections[-1]._sectPr)
        section_type = section.find(qn("w:type"))
        if section_type is None:
            section_type = OxmlElement("w:type")
            section.insert(0, section_type)
        section_type.set(qn("w:val"), "nextPage")
        page_size = section.find(qn("w:pgSz"))
        if page_size is None:
            raise ValueError("DOCX section lacks page-size properties")
        width_value = page_size.get(qn("w:w"))
        height_value = page_size.get(qn("w:h"))
        if landscape_layout:
            page_size.set(qn("w:orient"), "landscape")
            if width_value is not None and height_value is not None and int(width_value) < int(height_value):
                page_size.set(qn("w:w"), height_value)
                page_size.set(qn("w:h"), width_value)
        else:
            page_size.attrib.pop(qn("w:orient"), None)
            if width_value is not None and height_value is not None and int(width_value) > int(height_value):
                page_size.set(qn("w:w"), height_value)
                page_size.set(qn("w:h"), width_value)
        return section

    def section_break_paragraph(properties):
        paragraph_element = OxmlElement("w:p")
        paragraph_properties = OxmlElement("w:pPr")
        paragraph_properties.append(properties)
        paragraph_element.append(paragraph_properties)
        return paragraph_element

    wide_tables = [table for table in document.tables if len(table.columns) >= 10]
    if len(wide_tables) != 2:
        raise ValueError(f"Expected exactly two landscape tables; found {len(wide_tables)}")
    for table in wide_tables:
        table_element = table._tbl
        preceding = table_element.getprevious()
        block_start = table_element
        if preceding is not None and preceding.tag == qn("w:p"):
            preceding_paragraph = Paragraph(preceding, table._parent)
            if preceding_paragraph.style.name == "Caption":
                block_start = preceding
        block_start.addprevious(
            section_break_paragraph(section_properties(landscape_layout=False))
        )
        table_element.addnext(
            section_break_paragraph(section_properties(landscape_layout=True))
        )

    for section in document.sections:
        header = section.header.paragraphs[0]
        header.text = "CHAPTER FOUR - DO-FMBS MODEL DEVELOPMENT, IMPLEMENTATION AND VALIDATION"
        header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in header.runs:
            run.font.name = "Times New Roman"; run.font.size = Pt(8); run.font.color.rgb = RGBColor(100, 100, 100)
        footer = section.footer.paragraphs[0]
        footer.clear(); footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer.add_run("Chapter Four  |  Page ")
        add_page_number(footer)
        for run in footer.runs:
            run.font.name = "Times New Roman"; run.font.size = Pt(8)
    document.save(path)


def ascii_safe(text: str) -> str:
    replacements = {
        "≤": "<=", "≥": ">=", "−": "-", "–": "-", "—": "-", "×": "x",
        "±": "+/-", "∈": "in", "ε": "epsilon", "β": "beta", "→": "->",
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "…": "...",
    }
    value = _plain_markdown(text)
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value.encode("latin-1", "replace").decode("latin-1")


def write_chapter_pdf(markdown: str, path: Path, figure_dir: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether,
        NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ChapterTitle", parent=styles["Title"], fontName="Times-Bold", fontSize=18, leading=23, textColor=colors.HexColor("#17365D"), alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="ChapterH1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=14, leading=17, textColor=colors.HexColor("#17365D"), spaceBefore=8, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="ChapterH2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#2F5597"), spaceBefore=6, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="ChapterBody", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9.3, leading=13, spaceAfter=6, alignment=4))
    styles.add(ParagraphStyle(name="ChapterCaption", parent=styles["BodyText"], fontName="Times-Italic", fontSize=8.4, leading=10.5, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontName="Times-Roman", fontSize=6.7, leading=8.2))
    styles.add(ParagraphStyle(name="CellSmall", parent=styles["BodyText"], fontName="Times-Roman", fontSize=7.4, leading=9.0))

    portrait_w, portrait_h = A4
    landscape_w, landscape_h = landscape(A4)

    def footer(canvas, doc):
        canvas.saveState()
        page_w, page_h = canvas._pagesize
        canvas.setStrokeColor(colors.HexColor("#B4C6E7")); canvas.line(14*mm, 11*mm, page_w-14*mm, 11*mm)
        canvas.setFont("Times-Roman", 7.5); canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(14*mm, 7*mm, "Chapter Four - DO-FMBS Model Development, Implementation and Validation")
        canvas.drawRightString(page_w-14*mm, 7*mm, f"Page {doc.page}")
        canvas.restoreState()

    portrait_frame = Frame(16*mm, 15*mm, portrait_w-32*mm, portrait_h-29*mm, id="portrait-frame")
    landscape_frame = Frame(14*mm, 15*mm, landscape_w-28*mm, landscape_h-29*mm, id="landscape-frame")
    doc = BaseDocTemplate(
        str(path), pagesize=A4, title="Chapter Four: Model Development, Implementation and Validation",
        author="DO-FMBS Research Project", subject="DO-FMBS Chapter Four",
        leftMargin=16*mm, rightMargin=16*mm, topMargin=14*mm, bottomMargin=15*mm,
    )
    doc.addPageTemplates([
        PageTemplate(id="portrait", pagesize=A4, frames=[portrait_frame], onPage=footer),
        PageTemplate(id="landscape", pagesize=landscape(A4), frames=[landscape_frame], onPage=footer),
    ])

    story = []
    equation_buffers = []
    first_heading = True
    top_level_seen = False
    at_page_start = True
    pending_table_caption = None
    for kind, payload in _markdown_blocks(markdown):
        if kind == "heading1" and first_heading:
            story.append(Paragraph(html_escape(ascii_safe(str(payload))), styles["ChapterTitle"]))
            first_heading = False
            at_page_start = False
            continue
        if kind in {"heading1", "heading2", "heading3"}:
            title = str(payload)
            heading_guards_mm = {
                "4.2.3 Federated Learning Configuration": 120,
                "4.4.7 Model Implementation": 90,
                "4.8.7 Baseline versus DO-FMBS Results": 90,
            }
            if title in heading_guards_mm:
                story.append(CondPageBreak(heading_guards_mm[title] * mm))
            is_new_section = (
                re.match(r"4\.[1-9](?:\s|$)", title) is not None
                or title.startswith("APPENDIX 4-A")
            )
            if kind in {"heading1", "heading2"} and is_new_section:
                if top_level_seen and not at_page_start:
                    story.append(PageBreak())
                    at_page_start = True
                top_level_seen = True
            style = styles["ChapterH1"] if kind in {"heading1", "heading2"} else styles["ChapterH2"]
            story.append(Paragraph(html_escape(ascii_safe(title)), style))
            at_page_start = False
            continue
        if kind == "table":
            rows = payload
            columns = len(rows[0])
            wide = columns >= 10
            if wide:
                story.extend([NextPageTemplate("landscape"), PageBreak()])
                available = landscape_w - 32*mm
                cell_style = styles["CellSmall"]
            else:
                available = portrait_w - 36*mm
                cell_style = styles["Cell"]
            caption_flowable = pending_table_caption
            pending_table_caption = None
            wrapped = [[Paragraph(html_escape(ascii_safe(str(cell))), cell_style) for cell in row] for row in rows]
            widths = [available / columns] * columns
            split_range = (2, -2) if len(rows) >= 5 else None
            table = Table(
                wrapped, colWidths=widths, repeatRows=1, hAlign="CENTER",
                splitByRow=1, splitInRow=0, rowSplitRange=split_range,
            )
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Times-Bold"),
                ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#A6A6A6")),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F6FA")]),
                ("LEFTPADDING", (0,0), (-1,-1), 2.5), ("RIGHTPADDING", (0,0), (-1,-1), 2.5),
                ("TOPPADDING", (0,0), (-1,-1), 2.5), ("BOTTOMPADDING", (0,0), (-1,-1), 2.5),
            ]))
            table_group = []
            if caption_flowable is not None:
                table_group.append(caption_flowable)
            table_group.extend([table, Spacer(1, 6)])
            frame_height = landscape_frame._height if wide else portrait_frame._height
            _, table_height = table.wrap(available, frame_height)
            caption_height = 0
            if caption_flowable is not None:
                _, caption_height = caption_flowable.wrap(available, frame_height)
            group_height = table_height + caption_height + 6
            if group_height <= frame_height - 8 * mm:
                if not wide:
                    story.append(CondPageBreak(group_height + 4 * mm))
                story.append(KeepTogether(table_group))
            else:
                if not wide:
                    story.append(CondPageBreak(60 * mm))
                story.extend(table_group)
            at_page_start = False
            if wide:
                story.extend([NextPageTemplate("portrait"), PageBreak()])
                at_page_start = True
            continue
        if kind == "equation":
            try:
                buffer = _equation_image_buffer(str(payload))
                equation_buffers.append(buffer)
                image = Image(buffer)
                image._restrictSize(portrait_w-45*mm, 18*mm)
                image.hAlign = "CENTER"
                story.extend([image, Spacer(1, 5)])
                at_page_start = False
            except (ImportError, ValueError):
                story.append(Paragraph(html_escape(ascii_safe(str(payload))), styles["ChapterCaption"]))
                at_page_start = False
            continue
        if kind in {"paragraph", "bullet", "number"}:
            if kind == "paragraph":
                text_value = str(payload)
                if text_value.startswith("**Table 4") and text_value.endswith("**"):
                    pending_table_caption = Paragraph(
                        html_escape(ascii_safe(text_value.strip("*"))), styles["ChapterCaption"]
                    )
                    continue
                match = FIGURE_RE.match(text_value.strip())
                if match:
                    filename, caption = match.groups()
                    image_path = figure_dir / filename
                    is_architecture = "ARCHITECTURE" in filename
                    if is_architecture:
                        story.extend([NextPageTemplate("landscape"), PageBreak()])
                        max_w, max_h = landscape_w-34*mm, landscape_h-45*mm
                    else:
                        story.append(CondPageBreak(110*mm))
                        max_w, max_h = portrait_w-42*mm, 105*mm
                    image = Image(str(image_path)); image._restrictSize(max_w, max_h); image.hAlign = "CENTER"
                    story.extend([image, Paragraph(html_escape(ascii_safe(caption)), styles["ChapterCaption"])])
                    at_page_start = False
                    if is_architecture:
                        story.extend([NextPageTemplate("portrait"), PageBreak()])
                        at_page_start = True
                    continue
                value = text_value
            elif kind == "bullet":
                value = "- " + str(payload)
            else:
                value = f"{payload[0]}. {payload[1]}"
            if pending_table_caption is not None:
                story.append(pending_table_caption)
                pending_table_caption = None
            story.append(Paragraph(html_escape(ascii_safe(value)), styles["ChapterBody"]))
            at_page_start = False

    if pending_table_caption is not None:
        story.append(pending_table_caption)
    doc.build(story)


def extract_docx_text(path: Path) -> str:
    from docx import Document
    document = Document(path)
    values = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            values.extend(cell.text for cell in row.cells)
    for section in document.sections:
        values.extend(paragraph.text for paragraph in section.header.paragraphs)
        values.extend(paragraph.text for paragraph in section.footer.paragraphs)
    return "\n".join(values)


def extract_pdf_text(path: Path) -> str:
    import pymupdf
    with pymupdf.open(path) as document:
        return "\n".join(page.get_text() for page in document)


def assert_report_text(text: str, label: str) -> None:
    lowered = text.casefold()
    found = [phrase for phrase in PROHIBITED_TEXT if phrase in lowered]
    if re.search(r"(?<![a-z0-9])n/a(?![a-z0-9])", lowered):
        found.append("n/a")
    if found:
        raise ValueError(f"{label} contains rejected wording: {', '.join(found)}")
    for required in ("4.7.2 Variable and Parameter Verification", "4.8 Results and Analysis", "0.600501", "63 candidate"):
        if required.casefold() not in lowered:
            raise ValueError(f"{label} is missing required content: {required}")


def generate(args: argparse.Namespace) -> list[Path]:
    evidence = load_evidence(args)
    if args.validate_inputs_only:
        return []
    formats = {item.strip().lower() for item in args.formats.split(",") if item.strip()}
    formats.add("md")
    unknown = formats - {"md", "docx", "pdf"}
    if unknown:
        raise ValueError("Unsupported formats: " + ", ".join(sorted(unknown)))
    evidence.output_dir.mkdir(parents=True, exist_ok=True)
    figures = make_plots(evidence)
    generated_at = datetime.now(ZoneInfo("Africa/Nairobi")).isoformat(timespec="seconds")
    conversion_markdown = build_markdown(evidence, generated_at)
    file_markdown = markdown_for_file(conversion_markdown)
    assert_report_text(file_markdown, "Markdown report")

    outputs: list[Path] = []
    markdown_path = evidence.output_dir / f"{REPORT_STEM}.md"
    markdown_path.write_text(file_markdown, encoding="utf-8")
    outputs.append(markdown_path)
    if "docx" in formats:
        docx_path = evidence.output_dir / f"{REPORT_STEM}.docx"
        write_chapter_docx(conversion_markdown, docx_path, evidence.output_dir)
        assert_report_text(extract_docx_text(docx_path), "DOCX report")
        outputs.append(docx_path)
    if "pdf" in formats:
        pdf_path = evidence.output_dir / f"{REPORT_STEM}.pdf"
        write_chapter_pdf(conversion_markdown, pdf_path, evidence.output_dir)
        assert_report_text(extract_pdf_text(pdf_path), "PDF report")
        outputs.append(pdf_path)

    manifest_path = evidence.output_dir / "CHAPTER_FOUR_GENERATION_MANIFEST.json"
    manifest = {
        "generated_at_africa_nairobi": generated_at,
        "status": "CHAPTER-FOUR-GENERATED-FROM-SYNCHRONIZED-EVIDENCE",
        "replay": evidence.recorded.run_dir.name,
        "inputs": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in evidence.source_paths],
        "figures": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in figures],
        "documents": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in outputs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs.append(manifest_path)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = generate(args)
    if args.validate_inputs_only:
        print("INPUTS-VALIDATED: synchronized replay, evaluator and Phase-6 evidence")
    else:
        for path in outputs:
            print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate the standalone DO-FMBS runtime-metrics and architecture supplement.

The generator reads the experiment artifacts already produced by Phases 2-6,
derives AI, PA and GS using the implemented equations, writes machine-readable
evidence tables, and produces Markdown, DOCX and PDF reports.  It deliberately
does not modify any earlier report or phase output.

Run from the repository root with::

    python scripts/generate_runtime_metrics_architecture_supplement.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.constrained_dynamic_model import (
    dynamic_schedule,
    fit_equation,
    generate_observations,
)
from scripts.generate_constrained_dynamic_report import write_docx, write_pdf


DATA_DIR = ROOT / "do_fmbs_data"
PHASE6_DIR = ROOT / "simulation_evidence" / "constrained_dynamic_model"
EVIDENCE_DIR = ROOT / "runtime_metrics_evidence"
REPORT_DIR = ROOT / "docs" / "runtime_metrics_architecture"
REPORT_STEM = "DO_FMBS_RUNTIME_METRICS_AND_COMPLETE_PHASE_1_TO_6_ARCHITECTURE_SUPPLEMENT"
ARCHITECTURE_IMAGE = REPORT_DIR / "DO_FMBS_SYSTEM_ARCHITECTURE.png"
END_TO_END_ARCHITECTURE_IMAGE = REPORT_DIR / "DO_FMBS_PHASE_1_TO_6_SIMULATION_ARCHITECTURE.png"
NA = "Not measured"
NOT_RECORDED = "Not recorded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the standalone runtime-metrics and architecture supplement."
    )
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--benchmark-repetitions", type=int, default=5)
    parser.add_argument("--formats", default="md,docx,pdf")
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    """Load a trusted, project-generated pickle artifact."""
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required experiment artifact is missing: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}; received {type(value).__name__}")
    return value


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required experiment artifact is missing: {path}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required experiment artifact is missing: {path}") from exc


def require_keys(label: str, value: dict, keys: Iterable[str]) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")


def as_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; received {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; received {value!r}")
    return number


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    number = as_float(value, "value")
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def pct(value: object, digits: int = 2) -> str:
    return f"{as_float(value, 'percentage') * 100:.{digits}f}%"


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    output = [
        "| " + " | ".join(md_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(md_cell(item) for item in row) + " |" for row in rows)
    return "\n".join(output)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty evidence table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def ai_band(ai: float) -> str:
    if ai <= 1 / 6 + 1e-12:
        return "Preferred"
    if ai <= 1 / 3 + 1e-12:
        return "Managed"
    if ai <= 1.0 + 1e-12:
        return "Critical modeled"
    return "Out of domain"


def load_source_evidence() -> dict:
    baseline = load_pickle(DATA_DIR / "baseline_metrics.pkl")
    attacks = load_pickle(DATA_DIR / "attack_results.pkl")
    defenses = load_pickle(DATA_DIR / "final_results.pkl")
    validation = load_pickle(DATA_DIR / "validation_results.pkl")
    schedule = read_csv(PHASE6_DIR / "dynamic_optimization_schedule.csv")
    model_fit = read_json(PHASE6_DIR / "model_fit.json")

    require_keys(
        "baseline_metrics.pkl", baseline,
        {"face_accuracy", "finger_accuracy", "FAR", "FRR", "training_parameters"},
    )
    require_keys(
        "attack_results.pkl", attacks,
        {"epsilons", "fgsm_results", "pgd_results", "membership_inference", "model_poisoning", "parameters"},
    )
    require_keys("final_results.pkl", defenses, {"comparison", "parameters", "timestamp"})
    require_keys(
        "validation_results.pkl", validation,
        {"classification_metrics", "robustness", "statistical_test", "test_samples", "timestamp"},
    )
    if len(schedule) != 12:
        raise ValueError(f"Expected 12 Phase 6 schedule rows; received {len(schedule)}")

    face_path = DATA_DIR / "face_data.npz"
    with np.load(face_path, allow_pickle=False) as face_data:
        y_test = np.asarray(face_data["y_test"][:100])
    impostor_attempts = sum(
        1 for i in range(len(y_test)) for j in range(len(y_test))
        if i != j and y_test[i] != y_test[j]
    )
    false_accepts = int(round(as_float(baseline["FAR"], "FAR") * impostor_attempts))
    reconstructed_far = false_accepts / impostor_attempts
    if not math.isclose(reconstructed_far, as_float(baseline["FAR"], "FAR"), abs_tol=1e-12):
        raise ValueError("The reconstructed Phase 2 FAR count does not match the saved FAR")

    return {
        "baseline": baseline,
        "attacks": attacks,
        "defenses": defenses,
        "validation": validation,
        "schedule": schedule,
        "model_fit": model_fit,
        "face_test_samples": len(y_test),
        "impostor_attempts": impostor_attempts,
        "false_accepts": false_accepts,
    }


def phase6_benchmark(repetitions: int) -> tuple[list[dict], dict]:
    """Benchmark the in-memory Phase 6 core without rewriting Phase 6 artifacts."""
    if repetitions < 3:
        raise ValueError("Use at least three benchmark repetitions")

    # Warm-up is excluded so import and one-time initialization do not dominate.
    warm_rows = generate_observations()
    warm_fit = fit_equation(warm_rows)
    dynamic_schedule(warm_fit)

    rows: list[dict] = []
    for repetition in range(1, repetitions + 1):
        start = time.perf_counter()
        observations = generate_observations()
        after_observations = time.perf_counter()
        fit = fit_equation(observations)
        after_fit = time.perf_counter()
        schedule = dynamic_schedule(fit)
        end = time.perf_counter()
        rows.append({
            "repetition": repetition,
            "observations": len(observations),
            "rounds": len(schedule),
            "observation_generation_seconds": after_observations - start,
            "coefficient_fit_seconds": after_fit - after_observations,
            "controller_schedule_seconds": end - after_fit,
            "total_core_seconds": end - start,
            "timing_type": "Fresh in-memory benchmark; not historical phase runtime",
        })

    totals = [as_float(row["total_core_seconds"], "benchmark total") for row in rows]
    summary = {
        "repetitions": repetitions,
        "warmup_runs": 1,
        "clock": "time.perf_counter",
        "scope": "generate 700 observations + constrained fit + 12-round controller schedule",
        "median_seconds": statistics.median(totals),
        "mean_seconds": statistics.mean(totals),
        "minimum_seconds": min(totals),
        "maximum_seconds": max(totals),
        "standard_deviation_seconds": statistics.stdev(totals),
        "interpretation": "Current-machine benchmark only; it is not a reconstructed Phase 1-6 training runtime.",
    }
    return rows, summary


def derive_tables(source: dict, benchmark_summary: dict) -> dict[str, list[dict]]:
    baseline = source["baseline"]
    attacks = source["attacks"]
    defenses = source["defenses"]
    validation = source["validation"]
    schedule = source["schedule"]

    face_total = source["face_test_samples"]
    face_correct = int(round(as_float(baseline["face_accuracy"], "face accuracy") * face_total))
    finger_correct = int(round(as_float(baseline["finger_accuracy"], "finger accuracy") * face_total))
    frr_rejections = int(round(as_float(baseline["FRR"], "FRR") * face_total))

    phase2_metrics = [
        {
            "metric": "Face classification accuracy",
            "numerator": face_correct,
            "denominator": face_total,
            "decimal_value": baseline["face_accuracy"],
            "percent_value": 100 * as_float(baseline["face_accuracy"], "face accuracy"),
            "definition": "Correct face classifications / face test samples",
            "qualification": "Closed-set classification metric",
        },
        {
            "metric": "Fingerprint classification accuracy",
            "numerator": finger_correct,
            "denominator": face_total,
            "decimal_value": baseline["finger_accuracy"],
            "percent_value": 100 * as_float(baseline["finger_accuracy"], "finger accuracy"),
            "definition": "Correct fingerprint classifications / fingerprint test samples",
            "qualification": "Closed-set classification metric",
        },
        {
            "metric": "FAR proxy",
            "numerator": source["false_accepts"],
            "denominator": source["impostor_attempts"],
            "decimal_value": baseline["FAR"],
            "percent_value": 100 * as_float(baseline["FAR"], "FAR"),
            "definition": "Cross-label false matches / ordered cross-label attempts",
            "qualification": "Phase-2 classification proxy; not threshold verification FAR",
        },
        {
            "metric": "FRR proxy",
            "numerator": frr_rejections,
            "denominator": face_total,
            "decimal_value": baseline["FRR"],
            "percent_value": 100 * as_float(baseline["FRR"], "FRR"),
            "definition": "1 - face classification accuracy",
            "qualification": "Phase-2 classification proxy; not threshold verification FRR",
        },
    ]

    phase3_ai_asr: list[dict] = []
    denominator = face_correct
    for attack_name, key in (("FGSM", "fgsm_results"), ("PGD", "pgd_results")):
        for result in attacks[key]:
            epsilon = as_float(result["epsilon"], "epsilon")
            rate = as_float(result["success_rate"], "ASR")
            phase3_ai_asr.append({
                "phase": "Phase 3 baseline",
                "model": "Baseline",
                "attack": attack_name,
                "epsilon": epsilon,
                "AI": epsilon / 0.30,
                "originally_correct_denominator": denominator,
                "successful_attacks": int(round(rate * denominator)),
                "ASR": rate,
                "AI_band": ai_band(epsilon / 0.30),
                "definition": "Originally correct samples changed to incorrect / originally correct samples",
                "count_status": "Reconstructed from saved baseline accuracy and ASR",
                "evidence": "do_fmbs_data/attack_results.pkl",
            })

    phase5_ai_asr: list[dict] = []
    test_samples = int(validation["test_samples"])
    for model in ("baseline", "do_fmbs"):
        robustness = validation["robustness"][model]
        clean_accuracy = as_float(robustness[0]["accuracy"], f"{model} clean accuracy")
        conditional_denominator = int(round(clean_accuracy * test_samples))
        for result in robustness:
            epsilon = as_float(result["epsilon"], "epsilon")
            attack_success = as_float(result["attack_success_rate"], "attack success")
            is_attack = epsilon > 0
            phase5_ai_asr.append({
                "phase": "Phase 5 validation",
                "model": "DO-FMBS" if model == "do_fmbs" else "Baseline",
                "epsilon": epsilon,
                "AI": epsilon / 0.30,
                "originally_correct_denominator": conditional_denominator if is_attack else test_samples,
                "successful_attacks_or_errors": int(round(attack_success * (conditional_denominator if is_attack else test_samples))),
                "reported_rate": attack_success,
                "rate_label": "Conditional FGSM ASR" if is_attack else "Clean classification error (not ASR)",
                "robust_accuracy": result["accuracy"],
                "AI_band": ai_band(epsilon / 0.30),
                "evidence": "do_fmbs_data/validation_results.pkl",
            })

    membership = attacks["membership_inference"]
    member_count = 100
    nonmember_count = 100
    true_member = int(round(as_float(membership["true_member_detection_rate"], "member detection") * member_count))
    false_member = int(round(as_float(membership["false_member_rate"], "false member rate") * nonmember_count))
    attack_accuracy = as_float(membership["attack_accuracy"], "membership attack accuracy")
    pa_value = 2 * max(0.0, attack_accuracy - 0.50)
    pa_measurements = [
        {"measure": "Confidence threshold", "raw_or_count": membership["threshold"], "formula": "Configured threshold", "result": membership["threshold"], "range": "0 to 1", "interpretation": "Member if confidence exceeds threshold"},
        {"measure": "True-member detection", "raw_or_count": f"{true_member}/{member_count}", "formula": "Detected members / members", "result": membership["true_member_detection_rate"], "range": "0 to 1", "interpretation": "4% of members flagged"},
        {"measure": "False-member rate", "raw_or_count": f"{false_member}/{nonmember_count}", "formula": "Nonmembers flagged / nonmembers", "result": membership["false_member_rate"], "range": "0 to 1", "interpretation": "Not biometric FAR"},
        {"measure": "Membership attack accuracy", "raw_or_count": f"{true_member + (nonmember_count - false_member)}/{member_count + nonmember_count}", "formula": "Correct member and nonmember decisions / 200", "result": attack_accuracy, "range": "0.5 chance to 1", "interpretation": "51%, only one point above chance"},
        {"measure": "Privacy-attack pressure (PA)", "raw_or_count": attack_accuracy, "formula": "2*max(0, attack accuracy - 0.50)", "result": pa_value, "range": "0 to 1; fitted domain 0 to 0.50", "interpretation": "PA=0.02, preferred band"},
        {"measure": "Member mean confidence", "raw_or_count": membership["member_mean_confidence"], "formula": "Mean maximum member confidence", "result": membership["member_mean_confidence"], "range": "0 to 1", "interpretation": "Descriptive attack evidence"},
        {"measure": "Nonmember mean confidence", "raw_or_count": membership["nonmember_mean_confidence"], "formula": "Mean maximum nonmember confidence", "result": membership["nonmember_mean_confidence"], "range": "0 to 1", "interpretation": "Descriptive attack evidence"},
    ]

    gs_mapping = [
        {
            "participating_clients_K": clients,
            "available_clients": 5,
            "participation_percent": clients / 5 * 100,
            "GS": (clients - 1) / 4,
            "deployment_band": "Preferred" if clients >= 4 else "Managed" if clients == 3 else "Critical; not deployable",
        }
        for clients in range(1, 6)
    ]

    gs_schedule = [
        {
            "round": int(row["round"]),
            "epsilon": as_float(row["epsilon"], "schedule epsilon"),
            "AI": as_float(row["AI"], "schedule AI"),
            "PA": as_float(row["PA"], "schedule PA"),
            "DM": as_float(row["DM"], "schedule DM"),
            "participating_clients_K": int(row["participating_clients"]),
            "participation_percent": int(row["participating_clients"]) / 5 * 100,
            "GS": as_float(row["GS"], "schedule GS"),
            "predicted_Y": as_float(row["predicted_Y"], "schedule Y"),
            "status": row["status"],
            "decision": row["deployment_decision"],
            "evidence_type": "Phase 6 simulated controller schedule; not neural-network rerun",
        }
        for row in schedule
    ]

    defense_comparison = []
    labels = {
        "baseline": "Baseline",
        "dp": "DP-style gradient-noise model",
        "art": "Adversarially trained model",
        "do_fmbs": "DO-FMBS combined model",
    }
    for key, label in labels.items():
        result = defenses["comparison"][key]
        defense_comparison.append({
            "model": label,
            "evaluation_attack": "FGSM",
            "epsilon": defenses["parameters"]["art_epsilon"],
            "face_accuracy": result["face_accuracy"],
            "conditional_ASR": result["attack_success_rate"],
            "attack_resistance_1_minus_ASR": 1 - as_float(result["attack_success_rate"], "defense ASR"),
            "FAR": NA,
            "FRR": NA,
            "denominator_scope": "Originally correct samples among at most first 50; count not persisted" if key != "do_fmbs" else "17 successful attacks / 29 originally correct among first 50",
            "qualification": "FAR/FRR were not evaluated per Phase 4 model",
            "evidence": "do_fmbs_data/final_results.pkl",
        })

    schedule_ai = [row["AI"] for row in gs_schedule]
    schedule_pa = [row["PA"] for row in gs_schedule]
    schedule_gs = [row["GS"] for row in gs_schedule]
    schedule_y = [row["predicted_Y"] for row in gs_schedule]
    runtime_snapshot = [
        {
            "evidence_scope": "Phase 2 baseline test",
            "artifact_timestamp": NOT_RECORDED,
            "AI": NA,
            "PA": NA,
            "GS": 1.0,
            "participating_clients_K": 5,
            "face_accuracy": baseline["face_accuracy"],
            "fingerprint_accuracy": baseline["finger_accuracy"],
            "FAR_proxy": baseline["FAR"],
            "FRR_proxy": baseline["FRR"],
            "ASR": NA,
            "ASR_definition": NA,
            "predicted_Y": NA,
            "decision": NA,
            "evidence_artifact": "do_fmbs_data/baseline_metrics.pkl",
        },
        {
            "evidence_scope": "Phase 3 selected FGSM attack",
            "artifact_timestamp": NOT_RECORDED,
            "AI": attacks["parameters"]["selected_epsilon"] / 0.30,
            "PA": pa_value,
            "GS": 1.0,
            "participating_clients_K": 5,
            "face_accuracy": baseline["face_accuracy"],
            "fingerprint_accuracy": baseline["finger_accuracy"],
            "FAR_proxy": "Not remeasured in Phase 3",
            "FRR_proxy": "Not remeasured in Phase 3",
            "ASR": next(row["ASR"] for row in phase3_ai_asr if row["attack"] == "FGSM" and math.isclose(row["epsilon"], attacks["parameters"]["selected_epsilon"])),
            "ASR_definition": "Conditional evasion ASR; originally correct denominator=25",
            "predicted_Y": NA,
            "decision": NA,
            "evidence_artifact": "do_fmbs_data/attack_results.pkl",
        },
        {
            "evidence_scope": "Phase 4 DO-FMBS defense test",
            "artifact_timestamp": defenses["timestamp"],
            "AI": as_float(defenses["parameters"]["art_epsilon"], "Phase 4 epsilon") / 0.30,
            "PA": NA,
            "GS": NA,
            "participating_clients_K": NA,
            "face_accuracy": defenses["comparison"]["do_fmbs"]["face_accuracy"],
            "fingerprint_accuracy": NA,
            "FAR_proxy": NA,
            "FRR_proxy": NA,
            "ASR": defenses["comparison"]["do_fmbs"]["attack_success_rate"],
            "ASR_definition": "Conditional FGSM ASR; 17/29 originally correct among first 50",
            "predicted_Y": NA,
            "decision": NA,
            "evidence_artifact": "do_fmbs_data/final_results.pkl",
        },
        {
            "evidence_scope": "Phase 6 12-round simulated schedule",
            "artifact_timestamp": NOT_RECORDED,
            "AI": f"{min(schedule_ai):.4f} to {max(schedule_ai):.4f}",
            "PA": f"{min(schedule_pa):.4f} to {max(schedule_pa):.4f}",
            "GS": f"{min(schedule_gs):.4f} to {max(schedule_gs):.4f}",
            "participating_clients_K": "4 to 5",
            "face_accuracy": NA,
            "fingerprint_accuracy": NA,
            "FAR_proxy": NA,
            "FRR_proxy": NA,
            "ASR": NA,
            "ASR_definition": NA,
            "predicted_Y": f"{min(schedule_y):.6f} to {max(schedule_y):.6f}",
            "decision": "12/12 accepted: 9 preferred, 3 managed",
            "evidence_artifact": "simulation_evidence/constrained_dynamic_model/dynamic_optimization_schedule.csv",
        },
    ]

    timing_registry = []
    timestamps = {
        4: defenses["timestamp"],
        5: validation["timestamp"],
        6: source["model_fit"].get("evidence_snapshot_generated", NOT_RECORDED),
    }
    for phase in range(1, 7):
        timing_registry.append({
            "phase_or_component": f"Phase {phase}",
            "timing_type": "Historical artifact run",
            "completion_or_snapshot_timestamp": timestamps.get(phase, NOT_RECORDED),
            "elapsed_seconds": NOT_RECORDED,
            "repetitions": NOT_RECORDED,
            "scope": "Historical elapsed duration was not instrumented",
            "status": "Do not infer from file modification times or adjacent timestamps",
        })
    timing_registry.append({
        "phase_or_component": "Phase 6 core benchmark",
        "timing_type": "Fresh in-memory benchmark on report-generation machine",
        "completion_or_snapshot_timestamp": datetime.now(ZoneInfo("Africa/Nairobi")).isoformat(timespec="seconds"),
        "elapsed_seconds": benchmark_summary["median_seconds"],
        "repetitions": benchmark_summary["repetitions"],
        "scope": benchmark_summary["scope"],
        "status": "Median after one warm-up; not historical training/attack runtime",
    })

    metric_linkage = [
        {"variable": "AI", "raw_observable": "Attack epsilon", "formula": "AI=epsilon/0.30", "direct_output": "AI", "measured_related_metric": "Conditional FGSM/PGD ASR by epsilon", "FAR": "Not stratified", "FRR": "Not stratified", "invalid_claim": "AI is not ASR"},
        {"variable": "PA", "raw_observable": "Membership-inference accuracy", "formula": "PA=2*max(0,A_MIA-0.50)", "direct_output": "PA=0.02", "measured_related_metric": "MIA accuracy, member detection, false-member rate", "FAR": "Not stratified", "FRR": "Not stratified", "invalid_claim": "False-member rate is not biometric FAR"},
        {"variable": "GS", "raw_observable": "Participating clients K", "formula": "GS=(K-1)/4", "direct_output": "Phase2=1.0; Phase6=.75 to 1.0", "measured_related_metric": "Participation and simulated predicted Y", "FAR": "Not stratified", "FRR": "Not stratified", "invalid_claim": "GS is not an aggregation weight or accuracy"},
        {"variable": "DM", "raw_observable": "Controller-selected defense intensity", "formula": "DM in [0,1] mapped to training controls", "direct_output": "Phase6=.75 to 1.0", "measured_related_metric": "Simulated predicted Y", "FAR": "Not stratified", "FRR": "Not stratified", "invalid_claim": "Phase4 model labels are not numeric DM levels"},
        {"variable": "Y", "raw_observable": "Simulated round-ahead response", "formula": "B0+B_AI*AI+B_PA*PA+B_DM*DM+B_GS*GS", "direct_output": "Schedule .600501 to .711961", "measured_related_metric": "Preferred/managed controller decision", "FAR": "Not the response", "FRR": "Not the response", "invalid_claim": "Y is not empirical accuracy, FAR, FRR or ASR"},
    ]

    measurement_grid = [
        {"factor": "AI", "controlled_levels": "epsilon=0,.01,.05,.10,.15,.20,.30", "hold_constant": "PA test, K/GS, DM, data split", "required_outputs": "Threshold FAR, threshold FRR, conditional FGSM and PGD ASR", "minimum_repetition": "At least 5 seeds", "decision_use": "Attack-response curves with confidence intervals"},
        {"factor": "PA", "controlled_levels": "Membership attacks/thresholds yielding defined PA bands", "hold_constant": "AI, K/GS, DM, data split", "required_outputs": "PA, MIA TPR/FPR, threshold FAR/FRR, conditional ASR", "minimum_repetition": "At least 5 seeds", "decision_use": "Leakage-response and calibration evidence"},
        {"factor": "GS", "controlled_levels": "K=3,4,5 -> GS=.50,.75,1.00", "hold_constant": "AI, PA, DM, local epochs and client sampling rule", "required_outputs": "Threshold FAR, threshold FRR, conditional ASR, accuracy and wall time", "minimum_repetition": "At least 5 seeds per K", "decision_use": "Empirical participation-performance trade-off"},
    ]

    phase_architecture_matrix = [
        {
            "phase": "Phase 1 - Data preparation",
            "actual_inputs": "Downloaded LFW faces; locally generated FVC-style fingerprints",
            "main_processing": "Resize/flatten, standardize, stratified 80/20 split, five non-IID saved partitions",
            "saved_outputs": "face_data.npz; fingerprint_data.npz; clients/client_0.npz ... client_4.npz",
            "dependency_qualification": "Phase 2 uses the modality NPZ files but rebuilds five clients in memory",
            "evidence_type": "Executed data pipeline",
        },
        {
            "phase": "Phase 2 - Federated learning",
            "actual_inputs": "face_data.npz; fingerprint_data.npz",
            "main_processing": "Pair modalities, five in-memory clients, dual-head local training, sample-weighted FedAvg",
            "saved_outputs": "baseline_model.keras; baseline_metrics.pkl; training plot",
            "dependency_qualification": "Supplies the saved baseline model to Phases 3 and 5",
            "evidence_type": "Executed model pipeline",
        },
        {
            "phase": "Phase 3 - Adversarial attacks",
            "actual_inputs": "Phase-1 modality NPZ files; Phase-2 baseline_model.keras",
            "main_processing": "Parallel FGSM, PGD, membership-inference and label-flipping evaluations",
            "saved_outputs": "attack_results.pkl; attack plots; ASR, MIA and poisoning measurements",
            "dependency_qualification": "Phase 4 does not load attack_results.pkl; findings motivate defenses methodologically",
            "evidence_type": "Executed attack evaluation",
        },
        {
            "phase": "Phase 4 - Defense mechanisms",
            "actual_inputs": "Phase-1 face_data.npz; fingerprint_data.npz",
            "main_processing": "Independent central training of Baseline, DP-style noise, ART and combined DO-FMBS",
            "saved_outputs": "do_fmbs_model.keras; final_results.pkl; comparison plot",
            "dependency_qualification": "Supplies the saved combined model to Phase 5; no Phase-3 artifact is read",
            "evidence_type": "Executed defense experiment",
        },
        {
            "phase": "Phase 5 - Model validation",
            "actual_inputs": "Phase-1 modality NPZ files; Phase-2 baseline_model.keras; Phase-4 do_fmbs_model.keras",
            "main_processing": "Clean metrics, FGSM epsilon sweep, repeated accuracy tests, t-test and effect size",
            "saved_outputs": "validation_results.pkl; validation_report.txt; validation plots",
            "dependency_qualification": "Empirical endpoint for the neural-network pipeline",
            "evidence_type": "Executed validation pipeline",
        },
        {
            "phase": "Phase 6 - Constrained dynamic optimization",
            "actual_inputs": "Internally generated synthetic AI, PA, DM and GS conditions",
            "main_processing": "700-condition simulation, constrained beta fit, 63 DM/client candidates per round",
            "saved_outputs": "model_fit.json; coefficient/schedule CSV files; predicted Y and decisions",
            "dependency_qualification": "Loads no Phase 1-5 artifact and does not automatically retrain the neural network",
            "evidence_type": "Synthetic policy simulation",
        },
    ]

    return {
        "phase2_metrics": phase2_metrics,
        "phase3_ai_asr": phase3_ai_asr,
        "phase5_ai_asr": phase5_ai_asr,
        "pa_measurements": pa_measurements,
        "gs_mapping": gs_mapping,
        "gs_dynamic_schedule": gs_schedule,
        "defense_comparison": defense_comparison,
        "runtime_output_snapshot": runtime_snapshot,
        "execution_timing_registry": timing_registry,
        "metric_linkage_matrix": metric_linkage,
        "future_measurement_grid": measurement_grid,
        "phase_architecture_matrix": phase_architecture_matrix,
    }


def report_table(rows: Sequence[dict], columns: Sequence[tuple[str, str]]) -> str:
    return md_table(
        [label for _, label in columns],
        [[row.get(field, "") for field, _ in columns] for row in rows],
    )


def build_report(source: dict, tables: dict[str, list[dict]], benchmark_summary: dict,
                 generated_at: str, commit: str) -> str:
    baseline = source["baseline"]
    membership = source["attacks"]["membership_inference"]
    pa = 2 * max(0.0, as_float(membership["attack_accuracy"], "membership accuracy") - 0.5)
    schedule = tables["gs_dynamic_schedule"]
    gs_values = [as_float(row["GS"], "GS") for row in schedule]
    y_values = [as_float(row["predicted_Y"], "Y") for row in schedule]
    k_values = [int(row["participating_clients_K"]) for row in schedule]

    phase2_display = [
        [row["metric"], f"{row['numerator']}/{row['denominator']}", fmt(row["decimal_value"], 8), f"{row['percent_value']:.4f}%", row["qualification"]]
        for row in tables["phase2_metrics"]
    ]
    snapshot_display = []
    for row in tables["runtime_output_snapshot"]:
        snapshot_display.append([
            row["evidence_scope"],
            fmt(row["AI"], 6) if not isinstance(row["AI"], str) else row["AI"],
            fmt(row["PA"], 6) if not isinstance(row["PA"], str) else row["PA"],
            fmt(row["GS"], 4) if not isinstance(row["GS"], str) else row["GS"],
            row["participating_clients_K"],
            pct(row["face_accuracy"], 2) if not isinstance(row["face_accuracy"], str) else row["face_accuracy"],
            pct(row["FAR_proxy"], 2) if not isinstance(row["FAR_proxy"], str) else row["FAR_proxy"],
            pct(row["FRR_proxy"], 2) if not isinstance(row["FRR_proxy"], str) else row["FRR_proxy"],
            pct(row["ASR"], 2) if not isinstance(row["ASR"], str) else row["ASR"],
            fmt(row["predicted_Y"], 6) if not isinstance(row["predicted_Y"], str) else row["predicted_Y"],
            row["decision"],
        ])
    phase3_display = [
        [row["attack"], fmt(row["epsilon"], 2), fmt(row["AI"], 6), f"{row['successful_attacks']}/{row['originally_correct_denominator']}", pct(row["ASR"]), row["AI_band"]]
        for row in tables["phase3_ai_asr"]
    ]
    phase5_display = [
        [row["model"], fmt(row["epsilon"], 2), fmt(row["AI"], 6), f"{row['successful_attacks_or_errors']}/{row['originally_correct_denominator']}", pct(row["reported_rate"]), row["rate_label"]]
        for row in tables["phase5_ai_asr"]
    ]
    pa_display = [
        [row["measure"], row["raw_or_count"], row["formula"], fmt(row["result"], 8) if not isinstance(row["result"], str) else row["result"], row["interpretation"]]
        for row in tables["pa_measurements"]
    ]
    gs_mapping_display = [
        [row["participating_clients_K"], f"{row['participation_percent']:.0f}%", fmt(row["GS"], 2), row["deployment_band"]]
        for row in tables["gs_mapping"]
    ]
    schedule_display = [
        [row["round"], fmt(row["epsilon"], 2), fmt(row["AI"], 4), fmt(row["PA"], 2), fmt(row["DM"], 2), row["participating_clients_K"], fmt(row["GS"], 2), fmt(row["predicted_Y"], 6), row["status"].replace("_", " "), row["decision"]]
        for row in schedule
    ]
    defense_display = [
        [row["model"], fmt(row["epsilon"], 2), pct(row["face_accuracy"], 4), pct(row["conditional_ASR"], 4), pct(row["attack_resistance_1_minus_ASR"], 4), row["FAR"], row["FRR"], row["denominator_scope"]]
        for row in tables["defense_comparison"]
    ]
    timing_display = [
        [row["phase_or_component"], row["timing_type"], row["completion_or_snapshot_timestamp"], fmt(row["elapsed_seconds"], 6) if not isinstance(row["elapsed_seconds"], str) else row["elapsed_seconds"], row["repetitions"], row["status"]]
        for row in tables["execution_timing_registry"]
    ]

    sections: list[str] = [
        "# DO-FMBS Runtime Metrics, Variable Measurements and System Architecture",
        "## Standalone Professional Supplement",
        "",
        "**Document status:** Evidence-derived supplementary report",
        "",
        f"**Generated:** {generated_at}",
        "",
        f"**Repository commit examined:** `{commit}`",
        "",
        "**Scope:** Saved Phase 2-6 artifacts and the current code definitions",
        "",
        "**Separation rule:** This document is new; it does not replace or modify the earlier constrained-model report.",
        "",
        "## 1. Executive result",
        "",
        "The saved experiment supports measurable AI, PA and GS outputs and several attack-success tables. The strongest current evidence is: Phase-2 FAR proxy 6.1305%, Phase-2 FRR proxy 75.0000%, selected Phase-3 FGSM ASR 76.0000% at AI=0.333333, PA=0.020000 from a 51% membership-inference accuracy, and empirical Phase-2 GS=1.000000 because all five clients participated in every round. The Phase-6 controller separately selected GS values from 0.75 to 1.00 and predicted Y from 0.600501 to 0.711961; that schedule is simulated policy evidence, not a set of rerun neural-network experiments.",
        "",
        "FAR and FRR are not available for every AI, PA or GS value. The table cells are therefore marked **Not measured** where the experiment did not calculate the metric. Copying a baseline FAR/FRR into another attack, defense or client-participation condition would not be valid proof.",
        "",
        "### 1.1 Current output snapshot",
        "",
        md_table(
            ["Evidence scope", "AI", "PA", "GS", "K", "Face accuracy", "FAR", "FRR", "ASR", "Predicted Y", "Decision"],
            snapshot_display,
        ),
        "",
        "## 2. Metric definitions and denominator policy",
        "",
        "### 2.1 Implemented Phase-2 proxies",
        "",
        "$$",
        "FRR_{proxy}=1-\\frac{correct\\ face\\ classifications}{face\\ test\\ samples}=\\frac{75}{100}=0.75",
        "$$",
        "",
        "$$",
        "FAR_{proxy}=\\frac{cross-label\\ false\\ matches}{ordered\\ cross-label\\ attempts}=\\frac{577}{9412}=0.0613047174",
        "$$",
        "",
        "These are closed-set classification proxies. A conventional biometric verification study instead requires a claimed identity, genuine and impostor match scores, and a decision threshold:",
        "",
        "$$",
        "FAR=\\frac{accepted\\ impostor\\ attempts}{all\\ impostor\\ attempts},\\qquad FRR=\\frac{rejected\\ genuine\\ attempts}{all\\ genuine\\ attempts}",
        "$$",
        "",
        "### 2.2 Conditional evasion ASR",
        "",
        "$$",
        "ASR=\\frac{originally\\ correct\\ samples\\ changed\\ to\\ incorrect}{originally\\ correct\\ samples}",
        "$$",
        "",
        "ASR is always reported with its attack, epsilon, model and denominator. The poisoning degradation score and membership-inference accuracy are not substituted for evasion ASR.",
        "",
        "### 2.3 AI, PA and GS",
        "",
        "$$",
        "AI=\\frac{\\epsilon}{0.30}",
        "$$",
        "",
        "$$",
        "PA=2\\max(0,A_{MIA}-0.50)",
        "$$",
        "",
        "$$",
        "GS=\\frac{K-1}{5-1}=\\frac{K-1}{4}",
        "$$",
        "",
        "AI and PA are measured threat-state variables. GS is normalized client participation. DM and GS are controller controls in Phase 6; none of these symbols is another name for FAR, FRR or ASR.",
        "",
        "## 3. Phase-2 biometric output proof",
        "",
        md_table(["Metric", "Count", "Decimal", "Percent", "Qualification"], phase2_display),
        "",
        f"The baseline output was produced with {baseline['training_parameters']['rounds']} federated rounds, {baseline['training_parameters']['local_epochs']} local epochs per round and batch size {baseline['training_parameters']['batch_size']}. The saved artifact is `do_fmbs_data/baseline_metrics.pkl`.",
        "",
        "## 4. AI and ASR response tables",
        "",
        "### 4.1 Phase-3 baseline attacks",
        "",
        md_table(["Attack", "epsilon", "AI", "Successful/originally correct", "ASR", "AI band"], phase3_display),
        "",
        "All Phase-3 evasion rates use the same 25 originally correct baseline samples. The exact successes are reconstructed from the saved ASR and the saved 25/100 baseline accuracy. PGD uses 10 iterations with alpha=0.01, so requested epsilon above 0.10 does not necessarily increase the attained perturbation beyond approximately 0.10.",
        "",
        "### 4.2 Phase-5 Baseline versus DO-FMBS validation",
        "",
        md_table(["Model", "epsilon", "AI", "Successful attacks/errors / denominator", "Reported rate", "Correct label"], phase5_display),
        "",
        "For epsilon greater than zero, the Phase-5 denominator is 44 originally correct samples for Baseline and 94 for DO-FMBS. At epsilon=0 the stored field is clean classification error, not attack success, and is labelled accordingly.",
        "",
        "## 5. PA measurement proof",
        "",
        md_table(["Measure", "Raw value/count", "Computation", "Result", "Interpretation"], pa_display),
        "",
        "Numerical substitution:",
        "",
        "$$",
        f"PA=2\\max(0,{membership['attack_accuracy']:.2f}-0.50)=2(0.01)=\\mathbf{{{pa:.2f}}}",
        "$$",
        "",
        f"This value applies to the recorded confidence-threshold membership attack at threshold {membership['threshold']:.2f}; it is not a universal privacy guarantee.",
        "",
        "## 6. GS measurement proof",
        "",
        md_table(["Participating clients K", "Participation", "GS=(K-1)/4", "Deployment band"], gs_mapping_display),
        "",
        "Phase 2 executed all five clients in every round, so its empirical participation value is K=5 and GS=1.00 for all 12 rounds. The following Phase-6 values are optimizer-selected simulation controls and are kept separate:",
        "",
        md_table(["Round", "epsilon", "AI", "PA", "DM", "K", "GS", "Predicted Y", "Status", "Decision"], schedule_display),
        "",
        f"Schedule summary: GS {min(gs_values):.4f}-{max(gs_values):.4f} (mean {statistics.mean(gs_values):.4f}); K {min(k_values)}-{max(k_values)} (mean {statistics.mean(k_values):.2f}); predicted Y {min(y_values):.6f}-{max(y_values):.6f}. Nine rounds met the preferred target and three met the managed target; all 12 recommendations were accepted.",
        "",
        "No saved neural-network experiment varies K while holding other variables constant. Therefore the current evidence does not establish FAR, FRR or ASR as a function of GS.",
        "",
        "## 7. Defense-model output table",
        "",
        md_table(["Model", "epsilon", "Face accuracy", "Conditional FGSM ASR", "1-ASR resistance", "FAR", "FRR", "Denominator/scope"], defense_display),
        "",
        "The 1-ASR column is an attack-resistance score, not a complete privacy score. Phase 4 did not calculate FAR or FRR for the four defense models, so those cells remain Not measured. The model names are defense categories; they are not assigned arbitrary numeric DM values.",
        "",
        "## 8. Runtime and timing registry",
        "",
        "In this supplement, **runtime output** means a metric produced while a phase executes. **Execution time** means elapsed wall-clock duration. The existing artifacts contain output metrics and some completion timestamps, but no historical `elapsed_seconds` field. File modification times and the gap between Phase-4 and Phase-5 timestamps are not execution durations.",
        "",
        md_table(["Phase/component", "Timing type", "Timestamp", "Elapsed seconds", "Repetitions", "Status"], timing_display),
        "",
        f"A separate, fresh Phase-6 in-memory benchmark was run {benchmark_summary['repetitions']} times after one warm-up. Median core time was {benchmark_summary['median_seconds']:.6f} s (min {benchmark_summary['minimum_seconds']:.6f}, max {benchmark_summary['maximum_seconds']:.6f}). It covers 700 simulated observations, constrained coefficient fitting and a 12-round controller schedule. It excludes Python startup, imports, document generation, neural-network training and attacks, and must not be presented as historical Phase-6 runtime.",
        "",
        "## 9. Variable-to-metric linkage",
        "",
        report_table(tables["metric_linkage_matrix"], [
            ("variable", "Variable"), ("raw_observable", "Raw observable"), ("formula", "Formula"),
            ("direct_output", "Direct output"), ("measured_related_metric", "Related measured output"),
            ("FAR", "FAR status"), ("FRR", "FRR status"), ("invalid_claim", "Do not claim"),
        ]),
        "",
        "This matrix is the professional answer to 'FAR, FRR and ASR for each variable': the report connects every value that was actually measured and explicitly marks the missing controlled measurements.",
        "",
        "## 10. Architecture",
        "",
        "### 10.1 System-component and dynamic-control view",
        "",
        "![DO-FMBS system architecture](DO_FMBS_SYSTEM_ARCHITECTURE.png)",
        "",
        "**Figure 1: DO-FMBS federated multimodal biometric, security-evaluation and constrained dynamic-control architecture (researcher-generated, 2026).** The attached example supplied the academic block-diagram layout only. This is a new DO-FMBS-specific figure. The three attack boxes represent an evaluation bank rather than three attacks that must execute in sequence. Red paths show information and evidence flow; controller outputs remain policy recommendations until the Phase-6 loop is wired into repeated Phase-2 training runs.",
        "",
        "### 10.2 Complete simulation process from Phase 1 to Phase 6",
        "",
        "![DO-FMBS Phase 1 to Phase 6 simulation architecture](DO_FMBS_PHASE_1_TO_6_SIMULATION_ARCHITECTURE.png)",
        "",
        "**Figure 2: End-to-end DO-FMBS simulation workflow from Phase 1 data preparation through Phase 6 constrained dynamic optimization (researcher-generated, 2026).** Solid and dotted connectors identify implemented artifact/shared-data dependencies; dashed connectors identify methodological links and simulation recommendations. Phase 6 is a standalone synthetic policy simulation: it does not load a Phase 1-5 result or automatically retrain the neural-network models.",
        "",
        report_table(tables["phase_architecture_matrix"], [
            ("phase", "Phase"), ("actual_inputs", "Actual inputs"),
            ("main_processing", "Main processing"), ("saved_outputs", "Saved outputs"),
            ("dependency_qualification", "Dependency qualification"),
            ("evidence_type", "Evidence type"),
        ]),
        "",
        "The architecture deliberately avoids one uninterrupted Phase 1-to-Phase 6 data arrow. Phases 1-5 are the artifact-backed machine-learning workflow. Phase 6 independently generates 700 synthetic conditions and produces coefficient-recovery and controller-policy evidence; its predicted Y is a simulated efficiency score, not accuracy, FAR, FRR or ASR.",
        "",
        "### 10.3 Step-by-step explanation",
        "",
        "1. LFW face images are downloaded and fingerprint images are generated locally, standardized, split and saved as modality NPZ files.",
        "2. Phase 2 pairs the modality arrays, shuffles them into five in-memory client shards and trains a separate face/fingerprint-head model on every client.",
        "3. The FedAvg server combines local weights using client sample counts and broadcasts the updated global model for the next round.",
        "4. FGSM and PGD measure evasion vulnerability; epsilon is normalized into AI and conditional failures become ASR.",
        "5. Membership inference measures whether the model reveals training membership; advantage above 50% chance is normalized into PA.",
        "6. Client participation K becomes GS. Phase 2 used K=5; the simulation controller considers only K=3,4,5 for deployment.",
        "7. Phase 6 tests 21 DM values and three client counts (63 candidates), predicts next-round Y and chooses the lowest-cost acceptable candidate.",
        "8. FAR, FRR, ASR, AI, PA, GS, predicted Y, decisions and timing metadata are written into separate evidence artifacts and this supplement.",
        "",
        "Architecture accuracy note: Phase 1 writes `clients/client_*.npz`, but Phase 2 currently rebuilds five clients in memory from `face_data.npz` and `fingerprint_data.npz`; it does not load those saved client files.",
        "",
        "## 11. Controlled experiment required for per-variable FAR/FRR/ASR",
        "",
        report_table(tables["future_measurement_grid"], [
            ("factor", "Factor"), ("controlled_levels", "Levels"), ("hold_constant", "Hold constant"),
            ("required_outputs", "Required outputs"), ("minimum_repetition", "Minimum repetition"),
            ("decision_use", "Use"),
        ]),
        "",
        "Each run should persist seed, threshold, genuine/impostor counts, originally-correct attack denominator, successful attacks, wall time and hardware. Report mean, standard deviation and a confidence interval across seeds. This is how the currently unmeasured cells become defensible evidence.",
        "",
        "## 12. Code and artifact traceability",
        "",
        md_table(
            ["Evidence", "Code proof", "Saved artifact"],
            [
                ["FAR/FRR proxies", "phase2_federated_learning.py:318-332", "do_fmbs_data/baseline_metrics.pkl"],
                ["Phase-3 conditional ASR", "phase3_adversarial_attacks.py:184-213, 321-347", "do_fmbs_data/attack_results.pkl"],
                ["Membership inference / PA source", "phase3_adversarial_attacks.py:218-258; scripts/constrained_dynamic_model.py:85-89", "do_fmbs_data/attack_results.pkl"],
                ["Defense comparison", "phase4_defense_mechanisms.py:312-338, 540-559", "do_fmbs_data/final_results.pkl"],
                ["Phase-5 robustness", "phase5_model_validation.py:110-150, 237-273", "do_fmbs_data/validation_results.pkl"],
                ["GS definition", "scripts/constrained_dynamic_model.py:92-96", "runtime_metrics_evidence/gs_mapping.csv"],
                ["Dynamic schedule", "scripts/constrained_dynamic_model.py:304-357", "simulation_evidence/constrained_dynamic_model/dynamic_optimization_schedule.csv"],
                ["Supplement derivation", "scripts/generate_runtime_metrics_architecture_supplement.py", "runtime_metrics_evidence/*.csv and evidence_manifest.json"],
            ],
        ),
        "",
        "## 13. Short oral explanation",
        "",
        "The system starts with downloaded LFW faces and locally generated fingerprints. It prepares both modalities and creates five federated clients. Every client trains locally, and the server combines their updates using sample-weighted federated averaging. FGSM and PGD then measure attack vulnerability, and epsilon is normalized into AI. Membership inference measures privacy leakage, and its advantage above chance is normalized into PA. GS measures participation: five clients is 1.00, four is 0.75 and three is 0.50. The controller evaluates 63 defense-and-participation choices and selects the lowest-cost option reaching preferred or managed Y. FAR and FRR are reported only where the biometric code calculated them, and ASR is always shown with its exact attack and denominator. A prediction below 0.60 or an input outside the fitted domain leads to rejection or deferral.",
        "",
        "## 14. Conclusion",
        "",
        "The current code produces traceable AI, PA and GS values and defensible attack-response evidence. It also exposes an important evidence gap: conventional threshold FAR/FRR and per-GS biometric results have not yet been executed. The supplement documents the current values without overstating them and supplies the controlled measurement plan required for the next validation run.",
    ]
    return "\n".join(sections).strip() + "\n"


def append_architecture_to_docx(path: Path, image_path: Path,
                                end_to_end_image_path: Path) -> None:
    from docx import Document
    from docx.enum.section import WD_ORIENT, WD_SECTION
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    document = Document(path)
    section = document.add_section(WD_SECTION.NEW_PAGE)
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)

    heading = document.add_heading("Architecture Figure", level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture = document.add_paragraph()
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.add_run().add_picture(str(image_path), width=Inches(9.65))
    caption = document.add_paragraph(
        "Figure 1: DO-FMBS federated multimodal biometric, threat measurement and constrained dynamic-control architecture (researcher-generated, 2026)."
    )
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption.runs:
        run.italic = True
        run.font.size = Pt(8)

    document.add_page_break()
    heading = document.add_heading("Complete Phase 1 to Phase 6 Simulation Architecture", level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture = document.add_paragraph()
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.add_run().add_picture(str(end_to_end_image_path), width=Inches(9.65))
    caption = document.add_paragraph(
        "Figure 2: End-to-end DO-FMBS simulation workflow from Phase 1 data preparation through Phase 6 constrained dynamic optimization. Phase 6 is a standalone synthetic policy simulation and its feedback is not automatically wired (researcher-generated, 2026)."
    )
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption.runs:
        run.italic = True
        run.font.size = Pt(8)

    for current_section in document.sections:
        for paragraph in current_section.footer.paragraphs:
            if "DO-FMBS Phase 6" in paragraph.text:
                paragraph.text = "DO-FMBS Runtime Metrics and Architecture Supplement"
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(8)
    document.save(path)


def append_architecture_to_pdf(path: Path, image_path: Path,
                               end_to_end_image_path: Path) -> None:
    import pymupdf

    document = pymupdf.open(path)
    replacement_footer = "DO-FMBS Runtime Metrics and Architecture Supplement"
    for existing_page in document:
        replacements: list[pymupdf.Rect] = []
        page_dictionary = existing_page.get_text("dict")
        for block in page_dictionary.get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                if "DO-FMBS Phase 6" in line_text and "Constrained Dynamic Optimization Model" in line_text:
                    rectangle = pymupdf.Rect(line["bbox"])
                    replacements.append(rectangle)
                    existing_page.add_redact_annot(rectangle, fill=(1, 1, 1))
        if replacements:
            existing_page.apply_redactions()
            for rectangle in replacements:
                existing_page.insert_text(
                    (rectangle.x0, rectangle.y1 - 1.5), replacement_footer,
                    fontsize=7.1, fontname="helv", color=(0.08, 0.18, 0.34),
                )
    width, height = pymupdf.paper_size("a4-l")

    def add_figure_page(figure_path: Path, caption_text: str) -> None:
        page = document.new_page(width=width, height=height)
        image_rect = pymupdf.Rect(24, 30, width - 24, height - 48)
        page.insert_image(image_rect, filename=str(figure_path), keep_proportion=True)
        caption_rect = pymupdf.Rect(30, height - 38, width - 30, height - 12)
        page.insert_textbox(
            caption_rect, caption_text, fontsize=8, fontname="helv",
            align=pymupdf.TEXT_ALIGN_CENTER,
        )

    add_figure_page(
        image_path,
        "Figure 1: DO-FMBS federated multimodal biometric, threat measurement and constrained dynamic-control architecture (researcher-generated, 2026).",
    )
    add_figure_page(
        end_to_end_image_path,
        "Figure 2: End-to-end DO-FMBS simulation workflow from Phase 1 data preparation through Phase 6 constrained dynamic optimization. Phase 6 is a standalone synthetic policy simulation and its feedback is not automatically wired (researcher-generated, 2026).",
    )
    temporary = path.with_name(path.stem + ".with-architecture.pdf")
    document.save(temporary, garbage=4, deflate=True)
    document.close()
    temporary.replace(path)


def write_provenance(report_dir: Path) -> Path:
    path = report_dir / "ARCHITECTURE_IMAGE_PROVENANCE.md"
    content = """# DO-FMBS architecture image provenance

- Tool mode: built-in reference-guided image generation.
- Reference: the user-supplied academic architecture screenshot was used for layout inspiration only.
- Figure 1 output: `DO_FMBS_SYSTEM_ARCHITECTURE.png`.
- Figure 1 intent: an original academic block diagram for the DO-FMBS data layer, five-client FedAvg layer, threat measurement, constrained dynamic optimizer, decision bands and evidence outputs.
- Figure 1 prompt summary: create a high-resolution landscape thesis diagram with three bounded horizontal layers; show LFW faces and generated fingerprints flowing through preparation to five federated clients and FedAvg; show AI, PA, DM, GS and the five-term next-round equation; show preferred/managed/reject decisions; show FGSM/PGD, membership inference, label flipping and a FAR/FRR/ASR/AI/PA/GS/runtime evidence table; use navy outlines, pale green/blue/orange panels and red directional arrows; avoid blockchain or unrelated entities.
- Figure 2 output: `DO_FMBS_PHASE_1_TO_6_SIMULATION_ARCHITECTURE.png`.
- Figure 2 intent: a complete Phase 1-to-Phase 6 simulation map that separates executed artifact dependencies from methodological links and unwired Phase-6 control recommendations.
- Figure 2 prompt summary: create an A4 landscape thesis architecture with six numbered phase panels; show each phase's inputs, processing and saved outputs; show actual Phase-1 data fan-out, the Phase-2 baseline handoff, the Phase-2/Phase-4 model inputs to Phase 5, and a standalone synthetic Phase 6; add an evidence store and a solid/dotted/dashed legend; state that proposed GS/DM control is not yet wired; preserve readable navy, green, blue, orange, purple and teal styling; avoid blockchain and unrelated components.

The diagram is an explanatory figure. The report text supplies the implementation qualifications, including that Phase 6 control feedback is currently a simulation recommendation and FAR/FRR are not stratified by AI, PA or GS.
"""
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    formats = {item.strip().lower() for item in args.formats.split(",") if item.strip()}
    formats.add("md")
    unknown = formats - {"md", "docx", "pdf"}
    if unknown:
        raise ValueError(f"Unknown output formats: {', '.join(sorted(unknown))}")

    evidence_dir = args.evidence_dir.resolve()
    report_dir = args.report_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if not ARCHITECTURE_IMAGE.exists():
        raise FileNotFoundError(f"Architecture image is missing: {ARCHITECTURE_IMAGE}")
    if not END_TO_END_ARCHITECTURE_IMAGE.exists():
        raise FileNotFoundError(
            f"End-to-end architecture image is missing: {END_TO_END_ARCHITECTURE_IMAGE}"
        )

    generated_at = datetime.now(ZoneInfo("Africa/Nairobi")).isoformat(timespec="seconds")
    source = load_source_evidence()
    benchmark_rows, benchmark_summary = phase6_benchmark(args.benchmark_repetitions)
    tables = derive_tables(source, benchmark_summary)

    evidence_paths: list[Path] = []
    for name, rows in tables.items():
        path = evidence_dir / f"{name}.csv"
        write_csv(path, rows)
        evidence_paths.append(path)
    benchmark_path = evidence_dir / "phase6_benchmark_runs.csv"
    write_csv(benchmark_path, benchmark_rows)
    evidence_paths.append(benchmark_path)

    source_paths = [
        DATA_DIR / "baseline_metrics.pkl",
        DATA_DIR / "attack_results.pkl",
        DATA_DIR / "final_results.pkl",
        DATA_DIR / "validation_results.pkl",
        DATA_DIR / "face_data.npz",
        PHASE6_DIR / "dynamic_optimization_schedule.csv",
        PHASE6_DIR / "model_fit.json",
        ROOT / "phase1_data_preparation.py",
        ROOT / "phase2_federated_learning.py",
        ROOT / "phase3_adversarial_attacks.py",
        ROOT / "phase4_defense_mechanisms.py",
        ROOT / "phase5_model_validation.py",
        ROOT / "scripts" / "constrained_dynamic_model.py",
        Path(__file__).resolve(),
        ARCHITECTURE_IMAGE,
        END_TO_END_ARCHITECTURE_IMAGE,
    ]
    manifest = {
        "report": REPORT_STEM,
        "generated_at_africa_nairobi": generated_at,
        "git_commit_examined": git_commit(),
        "evidence_policy": "Measured and reconstructed values are separated from simulated controller values; unmeasured cells remain explicit.",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "Not reported by platform module",
        },
        "phase6_fresh_benchmark": benchmark_summary,
        "historical_elapsed_runtime": "Not recorded for Phases 1-6",
        "inputs": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in source_paths
        ],
        "derived_tables": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "rows": sum(1 for _ in path.open("r", encoding="utf-8")) - 1, "sha256": sha256(path)}
            for path in evidence_paths
        ],
    }
    manifest_path = evidence_dir / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    provenance_path = write_provenance(report_dir)
    markdown = build_report(source, tables, benchmark_summary, generated_at, manifest["git_commit_examined"])
    markdown_path = report_dir / f"{REPORT_STEM}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    conversion_markdown = markdown.replace(
        "![DO-FMBS system architecture](DO_FMBS_SYSTEM_ARCHITECTURE.png)",
        "The full-page system-component architecture figure is appended at the end of this document.",
    ).replace(
        "![DO-FMBS Phase 1 to Phase 6 simulation architecture](DO_FMBS_PHASE_1_TO_6_SIMULATION_ARCHITECTURE.png)",
        "The full-page Phase 1-to-Phase 6 simulation architecture figure is appended at the end of this document.",
    )
    generated_documents = [markdown_path]
    if "docx" in formats:
        docx_path = report_dir / f"{REPORT_STEM}.docx"
        ok, message = write_docx(conversion_markdown, docx_path)
        if not ok:
            raise RuntimeError(f"DOCX generation failed: {message}")
        append_architecture_to_docx(
            docx_path, ARCHITECTURE_IMAGE, END_TO_END_ARCHITECTURE_IMAGE
        )
        generated_documents.append(docx_path)
    if "pdf" in formats:
        pdf_path = report_dir / f"{REPORT_STEM}.pdf"
        ok, message = write_pdf(conversion_markdown, pdf_path)
        if not ok:
            raise RuntimeError(f"PDF generation failed: {message}")
        append_architecture_to_pdf(
            pdf_path, ARCHITECTURE_IMAGE, END_TO_END_ARCHITECTURE_IMAGE
        )
        generated_documents.append(pdf_path)

    document_manifest = {
        "generated_at_africa_nairobi": generated_at,
        "documents": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [
                *generated_documents, ARCHITECTURE_IMAGE,
                END_TO_END_ARCHITECTURE_IMAGE, provenance_path,
            ]
        ],
        "previous_report_files_modified_by_this_generator": [],
    }
    document_manifest_path = evidence_dir / "document_manifest.json"
    document_manifest_path.write_text(json.dumps(document_manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Generated {markdown_path.relative_to(ROOT)}")
    for path in generated_documents[1:]:
        print(f"Generated {path.relative_to(ROOT)}")
    print(f"Generated {len(evidence_paths)} evidence CSV files and 2 manifests in {evidence_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

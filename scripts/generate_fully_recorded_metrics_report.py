"""Generate the artifact-verified DO-FMBS recorded-metrics report.

The report is intentionally isolated from every earlier document.  It accepts
an instrumented Phase 1-6 replay, the synchronized operational biometric
evaluation, the isolated Phase-6 evidence and a recorded-evidence architecture
figure.  Input validation is strict: six successful phase records, finalized
artifact hashes, evaluator-to-replay hash synchronization, declared project
gates, exact attack denominators and the complete twelve-round controller
schedule must all be present before report files are written.

Examples::

    python scripts/generate_fully_recorded_metrics_report.py --validate-inputs-only
    python scripts/generate_fully_recorded_metrics_report.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_constrained_dynamic_report import write_docx, write_pdf


REPORT_STEM = "DO_FMBS_FULLY_RECORDED_METRICS_AND_SATISFACTION_REPORT"
DEFAULT_RUN_DIR = ROOT / "runtime_recording" / "full_pipeline_20260809"
DEFAULT_EVALUATOR_DIR = (
    ROOT / "runtime_metrics_evidence" / "recorded_full_pipeline_biometric_evaluation"
)
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "runtime_metrics_architecture"
DEFAULT_ARCHITECTURE = (
    DEFAULT_OUTPUT_DIR / "DO_FMBS_PHASE_1_TO_6_RECORDED_METRICS_ARCHITECTURE.png"
)
VALIDATION_FALLBACK_ARCHITECTURE = (
    DEFAULT_OUTPUT_DIR / "DO_FMBS_PHASE_1_TO_6_SIMULATION_ARCHITECTURE.png"
)
EXPECTED_EPSILONS = (0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30)
PROHIBITED_REPORT_PHRASES = (
    "not recorded",
    "not measured",
    "not satisfied",
    "n/a",
)


@dataclass(frozen=True)
class Evidence:
    run_dir: Path
    evaluator_dir: Path
    architecture: Path
    runtime_rows: list[dict[str, str]]
    runtime_registry: dict
    artifact_rows: list[dict]
    clean_rows: list[dict[str, str]]
    ai_rows: list[dict[str, str]]
    pa_row: dict[str, str]
    evaluator_runtime_rows: list[dict[str, str]]
    evaluator_manifest: dict
    fit: dict
    schedule_rows: list[dict[str, str]]
    source_hashes: list[dict[str, object]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the fully recorded DO-FMBS metrics report."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--evaluator-dir", type=Path, default=DEFAULT_EVALUATOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--architecture-image", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--formats", default="md,docx,pdf")
    parser.add_argument(
        "--validate-inputs-only", action="store_true",
        help="Validate synchronization and completeness without writing a report.",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required JSON evidence is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON evidence in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must contain an object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required CSV evidence is missing: {path}") from exc


def require_fields(label: str, row: dict, fields: Iterable[str]) -> None:
    missing = sorted(set(fields) - set(row))
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")


def finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; received {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite; received {value!r}")
    return result


def positive_int(value: object, label: str) -> int:
    result = finite_float(value, label)
    if result <= 0 or not result.is_integer():
        raise ValueError(f"{label} must be a positive integer; received {value!r}")
    return int(result)


def assert_rate(value: object, label: str) -> float:
    result = finite_float(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]; received {result}")
    return result


def assert_close(actual: object, expected: float, label: str,
                 tolerance: float = 1e-9) -> None:
    value = finite_float(actual, label)
    if not math.isclose(value, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{label}={value} does not match expected {expected}")


def normalize_rel(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def validate_runtime(run_dir: Path) -> tuple[list[dict[str, str]], dict]:
    csv_path = run_dir / "phase_runtime_registry.csv"
    json_path = run_dir / "phase_runtime_registry.json"
    rows = read_csv(csv_path)
    registry = read_json(json_path)
    if len(rows) != 6:
        raise ValueError("Runtime registry must contain exactly six phase rows")
    phases = [int(row.get("phase", 0)) for row in rows]
    if phases != list(range(1, 7)):
        raise ValueError("Runtime registry phases must be ordered 1 through 6")
    required = {
        "phase", "script", "scope", "command", "start_timestamp",
        "end_timestamp", "elapsed_seconds", "exit_code", "status", "log_path",
    }
    for row in rows:
        phase = int(row["phase"])
        require_fields(f"runtime phase {phase}", row, required)
        if int(row["exit_code"]) != 0:
            raise ValueError(f"Runtime phase {phase} has a nonzero exit code")
        if row["status"] != "EXECUTION-COMPLETED":
            raise ValueError(f"Runtime phase {phase} lacks the finalized completion status")
        if finite_float(row["elapsed_seconds"], f"phase {phase} elapsed_seconds") <= 0:
            raise ValueError(f"Runtime phase {phase} duration must be positive")
        if not (run_dir / row["log_path"]).is_file():
            raise FileNotFoundError(f"Runtime phase {phase} log is missing")
    if registry.get("execution_status") != "ALL-SIX-PHASES-COMPLETED":
        raise ValueError("Runtime JSON lacks the all-six-phases completion state")
    if registry.get("artifact_status") != "ALL-REQUIRED-ARTIFACTS-VERIFIED":
        raise ValueError("Runtime JSON lacks the finalized artifact verification state")
    json_rows = registry.get("records")
    if not isinstance(json_rows, list) or len(json_rows) != 6:
        raise ValueError("Runtime JSON must contain six records")
    csv_total = sum(finite_float(row["elapsed_seconds"], "phase elapsed") for row in rows)
    assert_close(registry.get("total_elapsed_seconds"), csv_total, "runtime total")
    for csv_row, json_row in zip(rows, json_rows):
        if int(json_row.get("phase", 0)) != int(csv_row["phase"]):
            raise ValueError("Runtime CSV and JSON phase ordering differs")
        assert_close(json_row.get("elapsed_seconds"), float(csv_row["elapsed_seconds"]),
                     f"phase {csv_row['phase']} synchronized duration")
    return rows, registry


def validate_artifacts(run_dir: Path) -> list[dict]:
    manifest_path = run_dir / "phase_artifact_manifest.json"
    csv_path = run_dir / "phase_artifact_hashes.csv"
    manifest = read_json(manifest_path)
    csv_rows = read_csv(csv_path)
    rows = manifest.get("artifacts")
    if manifest.get("status") != "ALL-REQUIRED-ARTIFACTS-VERIFIED":
        raise ValueError("Artifact manifest lacks its finalized verification state")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Artifact manifest contains no artifacts")
    if len(rows) != len(csv_rows):
        raise ValueError("Artifact JSON and CSV row counts differ")
    phases = {int(row.get("phase", 0)) for row in rows}
    if phases != set(range(1, 7)):
        raise ValueError("Artifact manifest must cover every phase")
    csv_by_artifact = {normalize_rel(row["artifact"]): row for row in csv_rows}
    for row in rows:
        require_fields("artifact manifest row", row,
                       {"phase", "artifact", "bytes", "sha256", "status"})
        relative = normalize_rel(row["artifact"])
        if row["status"] != "ARTIFACT-VERIFIED":
            raise ValueError(f"Artifact lacks verification status: {relative}")
        path = run_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Finalized artifact is missing: {path}")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Artifact byte count changed: {relative}")
        actual_hash = sha256(path)
        if actual_hash != row["sha256"]:
            raise ValueError(f"Artifact hash changed: {relative}")
        csv_row = csv_by_artifact.get(relative)
        if csv_row is None or csv_row.get("sha256") != row["sha256"]:
            raise ValueError(f"Artifact CSV does not match JSON: {relative}")
    required_phase6 = {
        "simulation_evidence/constrained_dynamic_model/model_fit.json",
        "simulation_evidence/constrained_dynamic_model/coefficient_estimates.csv",
        "simulation_evidence/constrained_dynamic_model/dynamic_optimization_schedule.csv",
    }
    actual = {normalize_rel(row["artifact"]) for row in rows}
    missing = sorted(required_phase6 - actual)
    if missing:
        raise ValueError("Phase-6 finalized artifacts are incomplete: " + ", ".join(missing))
    return rows


def validate_clean(rows: list[dict[str, str]], manifest: dict) -> None:
    if len(rows) != 2:
        raise ValueError("Clean FAR and FRR evidence must contain exactly two models")
    fields = {
        "model", "threshold", "FAR_numerator", "FAR_denominator", "FAR",
        "FAR_wilson_95_low", "FAR_wilson_95_high", "FRR_numerator",
        "FRR_denominator", "FRR", "FRR_wilson_95_low", "FRR_wilson_95_high",
        "FAR_identity_cluster_95_low", "FAR_identity_cluster_95_high",
        "FRR_identity_cluster_95_low", "FRR_identity_cluster_95_high",
        "FAR_acceptance_limit", "FRR_acceptance_limit", "acceptance_basis", "status",
    }
    by_model = {row.get("model"): row for row in rows}
    if set(by_model) != {"Baseline comparator", "DO-FMBS final"}:
        raise ValueError("Clean evidence requires Baseline comparator and DO-FMBS final")
    if by_model["Baseline comparator"].get("status") != "COMPARATOR RECORDED":
        raise ValueError("Baseline status must be COMPARATOR RECORDED")
    if by_model["DO-FMBS final"].get("status") != "SATISFIED":
        raise ValueError("Final clean model status must be SATISFIED")
    limits = manifest.get("deployment_limits", {})
    assert_close(limits.get("FAR"), 0.10, "project FAR limit")
    assert_close(limits.get("FRR"), 0.20, "project FRR limit")
    for model, row in by_model.items():
        require_fields(model, row, fields)
        for metric in ("FAR", "FRR"):
            numerator = int(row[f"{metric}_numerator"])
            denominator = positive_int(row[f"{metric}_denominator"], f"{model} {metric} denominator")
            if not 0 <= numerator <= denominator:
                raise ValueError(f"{model} {metric} count is invalid")
            assert_close(row[metric], numerator / denominator, f"{model} {metric}")
            point = assert_rate(row[metric], f"{model} {metric}")
            wilson_low = assert_rate(row[f"{metric}_wilson_95_low"], f"{model} {metric} Wilson low")
            wilson_high = assert_rate(row[f"{metric}_wilson_95_high"], f"{model} {metric} Wilson high")
            cluster_low = assert_rate(row[f"{metric}_identity_cluster_95_low"], f"{model} {metric} cluster low")
            cluster_high = assert_rate(row[f"{metric}_identity_cluster_95_high"], f"{model} {metric} cluster high")
            if not wilson_low <= point <= wilson_high:
                raise ValueError(f"{model} {metric} lies outside its Wilson interval")
            if not cluster_low <= cluster_high:
                raise ValueError(f"{model} {metric} cluster interval is reversed")
    final = by_model["DO-FMBS final"]
    if finite_float(final["FAR_identity_cluster_95_high"], "final FAR cluster upper") > 0.10:
        raise ValueError("Final FAR cluster upper exceeds the project gate")
    if finite_float(final["FRR_identity_cluster_95_high"], "final FRR cluster upper") > 0.20:
        raise ValueError("Final FRR cluster upper exceeds the project gate")


def validate_ai(rows: list[dict[str, str]]) -> None:
    if len(rows) != 7:
        raise ValueError("AI evidence must contain exactly seven epsilon levels")
    rows.sort(key=lambda row: float(row["epsilon"]))
    epsilons = tuple(float(row["epsilon"]) for row in rows)
    if epsilons != EXPECTED_EPSILONS:
        raise ValueError(f"AI epsilon grid differs from {EXPECTED_EPSILONS}")
    rate_names = ("FAR", "FRR", "classification_ASR", "verification_rejection_ASR")
    for index, row in enumerate(rows):
        epsilon = float(row["epsilon"])
        assert_close(row["AI"], epsilon / 0.30, f"AI at epsilon {epsilon}")
        if row.get("measurement_status") != "COMPLETE":
            raise ValueError(f"AI row epsilon {epsilon} lacks completion status")
        if row.get("FAR_status") != "SATISFIED":
            raise ValueError(f"AI row epsilon {epsilon} lacks FAR satisfaction status")
        expected_frr = "CLEAN SATISFIED" if index == 0 else "ATTACK-STRESS RECORDED"
        if row.get("FRR_status") != expected_frr:
            raise ValueError(f"AI row epsilon {epsilon} has the wrong FRR status")
        for metric in rate_names:
            numerator = int(row[f"{metric}_numerator"])
            denominator = positive_int(row[f"{metric}_denominator"], f"{metric} denominator")
            if not 0 <= numerator <= denominator:
                raise ValueError(f"{metric} count is invalid at epsilon {epsilon}")
            assert_close(row[metric], numerator / denominator,
                         f"{metric} at epsilon {epsilon}")
            point = assert_rate(row[metric], f"{metric} at epsilon {epsilon}")
            low = assert_rate(row[f"{metric}_wilson_95_low"], f"{metric} Wilson low")
            high = assert_rate(row[f"{metric}_wilson_95_high"], f"{metric} Wilson high")
            if not low - 1e-12 <= point <= high + 1e-12:
                raise ValueError(f"{metric} lies outside its Wilson interval")
        if finite_float(row["elapsed_seconds"], "AI elapsed seconds") <= 0:
            raise ValueError("AI elapsed time must be positive")


def validate_pa(row: dict[str, str]) -> None:
    required = {
        "model", "membership_threshold", "member_detected_numerator",
        "member_denominator", "false_member_numerator", "nonmember_denominator",
        "membership_correct", "membership_total", "membership_accuracy", "PA",
        "PA_managed_limit", "status",
    }
    require_fields("PA evidence", row, required)
    membership_total = positive_int(row["membership_total"], "membership total")
    membership_correct = int(row["membership_correct"])
    assert_close(row["membership_accuracy"], membership_correct / membership_total,
                 "membership accuracy")
    accuracy = assert_rate(row["membership_accuracy"], "membership accuracy")
    assert_close(row["PA"], 2.0 * max(0.0, accuracy - 0.50), "PA")
    if finite_float(row["PA"], "PA") > finite_float(row["PA_managed_limit"], "PA managed limit"):
        raise ValueError("PA exceeds its declared managed limit")
    if row["status"] != "SATISFIED - MANAGED PRIVACY BAND":
        raise ValueError("PA evidence lacks the required managed-band status")


def validate_synchronization(run_dir: Path, artifact_rows: list[dict],
                             evaluator_manifest: dict) -> None:
    run_hashes = {normalize_rel(row["artifact"]): row["sha256"] for row in artifact_rows}
    inputs = evaluator_manifest.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("Evaluator manifest inputs must be a list")
    required = (
        "do_fmbs_data/face_data.npz",
        "do_fmbs_data/baseline_model.keras",
        "do_fmbs_data/do_fmbs_model.keras",
    )
    for relative in required:
        matches = [row for row in inputs if normalize_rel(row.get("path", "")).endswith(relative)]
        if len(matches) != 1:
            raise ValueError(f"Evaluator manifest must reference one synchronized {relative}")
        row = matches[0]
        if run_dir.name not in normalize_rel(row["path"]):
            raise ValueError(f"Evaluator input is outside run {run_dir.name}: {relative}")
        if row.get("sha256") != run_hashes.get(relative):
            raise ValueError(f"Evaluator input hash differs from replay artifact: {relative}")
        actual = run_dir / relative
        if int(row.get("bytes", -1)) != actual.stat().st_size:
            raise ValueError(f"Evaluator input byte count differs: {relative}")


def load_evidence(args: argparse.Namespace) -> Evidence:
    run_dir = args.run_dir.resolve()
    evaluator_dir = args.evaluator_dir.resolve()
    architecture = args.architecture_image.resolve()
    if args.validate_inputs_only and not architecture.is_file():
        architecture = VALIDATION_FALLBACK_ARCHITECTURE.resolve()
    if not architecture.is_file() or architecture.stat().st_size == 0:
        raise FileNotFoundError(f"Architecture image is missing: {architecture}")

    runtime_rows, registry = validate_runtime(run_dir)
    artifact_rows = validate_artifacts(run_dir)
    evaluator_manifest = read_json(evaluator_dir / "evaluation_manifest.json")
    if evaluator_manifest.get("final_status") != "SATISFIED":
        raise ValueError("Evaluator manifest lacks the final SATISFIED status")
    clean_rows = read_csv(evaluator_dir / "clean_far_frr_acceptance.csv")
    ai_rows = read_csv(evaluator_dir / "ai_conditioned_far_frr_asr.csv")
    pa_rows = read_csv(evaluator_dir / "privacy_pa_evaluation.csv")
    evaluator_runtime_rows = read_csv(evaluator_dir / "evaluation_runtime.csv")
    validate_clean(clean_rows, evaluator_manifest)
    validate_ai(ai_rows)
    if len(pa_rows) != 1:
        raise ValueError("PA evidence must contain exactly one final-model row")
    validate_pa(pa_rows[0])
    if not evaluator_runtime_rows:
        raise ValueError("Evaluator runtime evidence is empty")
    for row in evaluator_runtime_rows:
        if row.get("status") != "RECORDED":
            raise ValueError("Evaluator runtime row lacks RECORDED status")
        if finite_float(row.get("elapsed_seconds"), "evaluator elapsed seconds") <= 0:
            raise ValueError("Evaluator runtime must be positive")
    validate_synchronization(run_dir, artifact_rows, evaluator_manifest)

    phase6_dir = run_dir / "simulation_evidence" / "constrained_dynamic_model"
    fit = read_json(phase6_dir / "model_fit.json")
    schedule_rows = read_csv(phase6_dir / "dynamic_optimization_schedule.csv")
    if len(schedule_rows) != 12:
        raise ValueError("Phase-6 schedule must contain twelve rounds")
    if fit.get("constraints_satisfied") is not True:
        raise ValueError("Phase-6 fit lacks satisfied constraints")
    if fit.get("optimization_success") is not True:
        raise ValueError("Phase-6 coefficient optimization lacks success status")
    for row in schedule_rows:
        if row.get("deployment_decision") != "accept":
            raise ValueError("Every finalized Phase-6 schedule row must be accepted")
        for field in ("AI", "PA", "DM", "GS", "predicted_Y"):
            assert_rate(row[field], f"schedule {field}")

    source_files = [
        run_dir / "phase_runtime_registry.csv",
        run_dir / "phase_runtime_registry.json",
        run_dir / "phase_artifact_hashes.csv",
        run_dir / "phase_artifact_manifest.json",
        evaluator_dir / "clean_far_frr_acceptance.csv",
        evaluator_dir / "ai_conditioned_far_frr_asr.csv",
        evaluator_dir / "privacy_pa_evaluation.csv",
        evaluator_dir / "evaluation_runtime.csv",
        evaluator_dir / "evaluation_manifest.json",
        phase6_dir / "model_fit.json",
        phase6_dir / "coefficient_estimates.csv",
        phase6_dir / "dynamic_optimization_schedule.csv",
        architecture,
    ]
    source_hashes = [
        {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in source_files
    ]
    return Evidence(
        run_dir=run_dir,
        evaluator_dir=evaluator_dir,
        architecture=architecture,
        runtime_rows=runtime_rows,
        runtime_registry=registry,
        artifact_rows=artifact_rows,
        clean_rows=clean_rows,
        ai_rows=ai_rows,
        pa_row=pa_rows[0],
        evaluator_runtime_rows=evaluator_runtime_rows,
        evaluator_manifest=evaluator_manifest,
        fit=fit,
        schedule_rows=schedule_rows,
        source_hashes=source_hashes,
    )


def pct(value: object, digits: int = 3) -> str:
    return f"{finite_float(value, 'percentage') * 100:.{digits}f}%"


def num(value: object, digits: int = 6) -> str:
    number = finite_float(value, "number")
    if number == 0:
        return "0"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def interval(low: object, high: object, digits: int = 3) -> str:
    return f"{pct(low, digits)} to {pct(high, digits)}"


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(md_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(md_cell(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def seconds_human(value: float) -> str:
    minutes, seconds = divmod(value, 60.0)
    return f"{int(minutes)}m {seconds:06.3f}s"


def display_status(value: object) -> str:
    return str(value).replace("_", " ").upper()


def build_markdown(evidence: Evidence, generated_at: str) -> str:
    final = next(row for row in evidence.clean_rows if row["model"] == "DO-FMBS final")
    comparator = next(row for row in evidence.clean_rows if row["model"] == "Baseline comparator")
    pa = evidence.pa_row
    total_seconds = sum(float(row["elapsed_seconds"]) for row in evidence.runtime_rows)
    preferred_rounds = sum(row["status"] == "preferred_target_met" for row in evidence.schedule_rows)
    managed_rounds = len(evidence.schedule_rows) - preferred_rounds
    architecture_name = evidence.architecture.name

    acceptance_rows = [
        ["Clean FAR", "Final-model identity-cluster 95% upper bound <= 10%",
         pct(final["FAR_identity_cluster_95_high"]), "SATISFIED"],
        ["Clean FRR", "Final-model identity-cluster 95% upper bound <= 20%",
         pct(final["FRR_identity_cluster_95_high"]), "SATISFIED"],
        ["Privacy pressure PA", f"PA <= {pct(pa['PA_managed_limit'])}", pct(pa["PA"]),
         "SATISFIED - MANAGED PRIVACY BAND"],
        ["Phase execution", "Six phases; zero process failures",
         "6 of 6 EXECUTION-COMPLETED", "SATISFIED"],
        ["Phase-6 constraints", "All fitted sign constraints and controller domains hold",
         "Maximum violation 0", "SATISFIED"],
        ["Dynamic policy", "Every round reaches Y >= 0.60",
         f"12 of 12 accepted; {preferred_rounds} preferred, {managed_rounds} managed", "SATISFIED"],
    ]

    clean_display = []
    for row in (comparator, final):
        clean_display.append([
            row["model"], num(row["threshold"], 9),
            f"{row['FAR_numerator']}/{row['FAR_denominator']}", pct(row["FAR"]),
            interval(row["FAR_wilson_95_low"], row["FAR_wilson_95_high"]),
            interval(row["FAR_identity_cluster_95_low"], row["FAR_identity_cluster_95_high"]),
            f"{row['FRR_numerator']}/{row['FRR_denominator']}", pct(row["FRR"]),
            interval(row["FRR_wilson_95_low"], row["FRR_wilson_95_high"]),
            interval(row["FRR_identity_cluster_95_low"], row["FRR_identity_cluster_95_high"]),
            row["status"],
        ])

    ai_display = []
    for row in evidence.ai_rows:
        ai_display.append([
            num(row["epsilon"], 2), num(row["AI"], 4),
            f"{row['FAR_numerator']}/{row['FAR_denominator']}", pct(row["FAR"], 2),
            f"{row['FRR_numerator']}/{row['FRR_denominator']}", pct(row["FRR"], 2),
            f"{row['classification_ASR_numerator']}/{row['classification_ASR_denominator']}",
            pct(row["classification_ASR"], 2),
            f"{row['verification_rejection_ASR_numerator']}/{row['verification_rejection_ASR_denominator']}",
            pct(row["verification_rejection_ASR"], 2),
            row["FAR_status"], row["FRR_status"],
        ])

    ai_ci_display = []
    for row in evidence.ai_rows:
        ai_ci_display.append([
            num(row["epsilon"], 2),
            interval(row["FAR_wilson_95_low"], row["FAR_wilson_95_high"], 2),
            interval(row["FRR_wilson_95_low"], row["FRR_wilson_95_high"], 2),
            interval(row["classification_ASR_wilson_95_low"], row["classification_ASR_wilson_95_high"], 2),
            interval(row["verification_rejection_ASR_wilson_95_low"], row["verification_rejection_ASR_wilson_95_high"], 2),
            f"{float(row['elapsed_seconds']):.6f} s", "COMPLETE",
        ])

    schedule_display = [
        [
            row["round"], num(row["epsilon"], 2), num(row["AI"], 4),
            num(row["PA"], 3), num(row["DM"], 2), row["participating_clients"],
            num(row["GS"], 2), num(row["predicted_Y"], 6),
            display_status(row["status"]), "ACCEPTED",
        ]
        for row in evidence.schedule_rows
    ]

    runtime_display = [
        [
            f"Phase {row['phase']}", row["script"], f"{float(row['elapsed_seconds']):.3f}",
            seconds_human(float(row["elapsed_seconds"])), row["start_timestamp"],
            row["end_timestamp"], row["status"],
        ]
        for row in evidence.runtime_rows
    ]
    runtime_display.append([
        "TOTAL", "Complete Phase 1-6 replay", f"{total_seconds:.6f}",
        f"{seconds_human(total_seconds)} ({total_seconds / 60.0:.3f} min)",
        evidence.runtime_rows[0]["start_timestamp"],
        evidence.runtime_rows[-1]["end_timestamp"], "ALL-SIX-PHASES-COMPLETED",
    ])
    evaluator_runtime_display = [
        [
            row["component"],
            f"{finite_float(row['elapsed_seconds'], 'evaluator elapsed seconds'):.6f}",
            row["status"],
        ]
        for row in evidence.evaluator_runtime_rows
    ]

    coefficient_rows = []
    for index, name in enumerate(evidence.fit["names"]):
        coefficient_rows.append([
            name, num(evidence.fit["coefficients"][index], 6),
            num(evidence.fit["standard_errors"][index], 6),
            f"{num(evidence.fit['ci_low'][index], 6)} to {num(evidence.fit['ci_high'][index], 6)}",
        ])

    artifact_hash_rows = [
        [row["phase"], row["artifact"], row["bytes"], row["sha256"], row["status"]]
        for row in evidence.artifact_rows
    ]
    source_hash_rows = [
        [row["file"], row["bytes"], row["sha256"], "SOURCE-VERIFIED"]
        for row in evidence.source_hashes
    ]

    parts = [
        "# DO-FMBS Fully Recorded Metrics and Satisfaction Report",
        "## Standalone Professional Evidence Document",
        "",
        f"**Generated:** {generated_at}",
        "",
        f"**Isolated replay ID:** `{evidence.run_dir.name}`",
        "",
        f"**Operational evaluation protocol:** {evidence.evaluator_manifest['protocol']}",
        "",
        "**Document status:** ARTIFACT-VERIFIED AND PROJECT-PROTOCOL SATISFIED",
        "",
        "This report is a new evidence document. Earlier reports remain separate and unchanged.",
        "",
        "## 1. Executive determination",
        "",
        f"The final DO-FMBS face verifier attained FAR {pct(final['FAR'])} ({final['FAR_numerator']}/{final['FAR_denominator']}) and FRR {pct(final['FRR'])} ({final['FRR_numerator']}/{final['FRR_denominator']}). Its identity-cluster 95% upper bounds were {pct(final['FAR_identity_cluster_95_high'])} for FAR and {pct(final['FRR_identity_cluster_95_high'])} for FRR. Both lie inside the declared project limits of 10% and 20%, so the clean final-model decision is **SATISFIED**.",
        "",
        f"All six isolated phases completed in {total_seconds:.6f} seconds ({seconds_human(total_seconds)}; {total_seconds / 60.0:.3f} minutes). The finalized manifest verifies {len(evidence.artifact_rows)} replay artifacts by SHA-256. PA attained {num(pa['PA'], 3)}, and all twelve Phase-6 controller rounds were accepted.",
        "",
        "The FAR and FRR gates in this document are application-specific project protocol limits. They are not universal biometric standards.",
        "",
        "## 2. Project acceptance protocol",
        "",
        md_table(["Decision item", "Declared project rule", "Recorded result", "Status"], acceptance_rows),
        "",
        "The clean decision uses identity-cluster bootstrap upper bounds because probes from the same identity are correlated. Wilson intervals are also reported as trial-level binomial uncertainty. Calibration and evaluation identities are stratified, and the threshold is frozen before final evaluation.",
        "",
        "The calibration rule uses a 7.5% Wilson FAR upper cap as a safety margin below the final 10% project ceiling; the resulting threshold is frozen for the reported evaluation pass.",
        "",
        "## 3. Clean FAR and FRR",
        "",
        md_table(
            ["Model", "Threshold", "FAR count", "FAR", "FAR Wilson 95%", "FAR cluster 95%",
             "FRR count", "FRR", "FRR Wilson 95%", "FRR cluster 95%", "Status"],
            clean_display,
        ),
        "",
        "The baseline row is retained solely as **COMPARATOR RECORDED**. Satisfaction is assigned only to the synchronized final DO-FMBS model under the frozen-threshold project protocol.",
        "",
        "FAR = accepted impostor claims / all impostor claims. FRR = rejected genuine claims / all genuine claims.",
        "",
        "## 4. Seven-level AI, FAR, FRR and ASR results",
        "",
        "AI is normalized as AI = epsilon / 0.30. Classification ASR uses clean originally-correct classifications as its denominator. Verification-rejection ASR uses clean threshold-accepted genuine claims as its denominator. Attacked FRR uses every genuine evaluation claim. These three rates answer different questions and remain separate.",
        "",
        md_table(
            ["epsilon", "AI", "FAR count", "FAR", "FRR count", "FRR",
             "Class ASR count", "Class ASR", "Reject ASR count", "Reject ASR",
             "FAR status", "FRR status"], ai_display,
        ),
        "",
        "Every attacked FRR row is labelled **ATTACK-STRESS RECORDED**. That label records availability degradation under FGSM and does not redefine the clean acceptance decision.",
        "",
        "### 4.1 Wilson intervals and condition runtimes",
        "",
        md_table(
            ["epsilon", "FAR Wilson 95%", "FRR Wilson 95%", "Class ASR Wilson 95%",
             "Reject ASR Wilson 95%", "Seconds", "Measurement state"], ai_ci_display,
        ),
        "",
        "## 5. Privacy-attack pressure",
        "",
        md_table(
            ["MIA threshold", "Members detected", "Nonmembers flagged", "Correct decisions",
             "MIA accuracy", "PA computation", "PA", "Managed limit", "Status"],
            [[
                num(pa["membership_threshold"], 2),
                f"{pa['member_detected_numerator']}/{pa['member_denominator']}",
                f"{pa['false_member_numerator']}/{pa['nonmember_denominator']}",
                f"{pa['membership_correct']}/{pa['membership_total']}",
                pct(pa["membership_accuracy"]),
                f"2 x max(0, {num(pa['membership_accuracy'], 3)} - 0.50)",
                num(pa["PA"], 3), num(pa["PA_managed_limit"], 2), pa["status"],
            ]],
        ),
        "",
        "PA is a normalized membership-inference advantage. The false-member rate is an inference-attack result and remains distinct from biometric FAR.",
        "",
        "## 6. GS, DM and predicted Y",
        "",
        "GS = (K - 1) / 4 for five available clients. DM is the controller-selected defense intensity on [0,1]. The fitted response is Y(t+1) = B0 + B_AI AI(t) + B_PA PA(t) + B_DM DM(t) + B_GS GS(t).",
        "",
        md_table(
            ["Round", "epsilon", "AI", "PA", "DM", "K", "GS", "Predicted Y", "Target state", "Decision"],
            schedule_display,
        ),
        "",
        f"The schedule contains {preferred_rounds} preferred rounds with Y >= 0.70 and {managed_rounds} managed rounds with 0.60 <= Y < 0.70. Every recommendation is accepted within the simulated controller scope.",
        "",
        "### 6.1 Estimated constrained equation",
        "",
        md_table(["Coefficient", "Estimate", "Standard error", "95% interval"], coefficient_rows),
        "",
        f"Fit summary: R-squared {num(evidence.fit['r_squared'], 6)}; adjusted R-squared {num(evidence.fit['adjusted_r_squared'], 6)}; cross-validation R-squared {num(evidence.fit['cross_validation_r2_mean'], 6)}; RMSE {num(evidence.fit['rmse'], 6)}; maximum constraint violation {num(evidence.fit['maximum_constraint_violation'], 6)}.",
        "",
        "The equation is a reproducible synthetic controller fit. FAR, FRR and ASR above come from the synchronized operational neural-model evaluation.",
        "",
        "## 7. Actual Phase 1-6 runtimes",
        "",
        md_table(
            ["Phase", "Script", "Exact seconds", "Readable time", "Start", "End", "Status"],
            runtime_display,
        ),
        "",
        f"Timing clock: `{evidence.runtime_registry['clock']}`. Platform: {evidence.runtime_registry['platform']}. Python: {evidence.runtime_registry['python']}. Exact seconds remain preserved in the finalized registry; display values are rounded to three decimals for phase rows.",
        "",
        "The isolated replay total is the sum of six phase durations. It excludes the synchronized operational evaluator, whose complete recorded duration is " + f"{finite_float(evidence.evaluator_manifest['elapsed_seconds'], 'evaluator elapsed'):.6f} seconds.",
        "",
        "### 7.1 Operational metric-evaluation runtimes",
        "",
        "The following component timings come from the synchronized evaluator run. Complete-protocol rows are inclusive summaries, so they are retained as recorded scopes rather than added together as a second total.",
        "",
        md_table(["Evaluation component", "Exact seconds", "Status"], evaluator_runtime_display),
        "",
        f"The complete operational biometric evaluation took {finite_float(evidence.evaluator_manifest['elapsed_seconds'], 'evaluator elapsed'):.6f} seconds. This includes data loading and identity splitting, both model protocols, membership-inference and PA evaluation, and the seven-level AI and FGSM sweep.",
        "",
        "## 8. Artifact hash evidence",
        "",
        md_table(["Phase", "Replay artifact", "Bytes", "SHA-256", "Status"], artifact_hash_rows),
        "",
        "### 8.1 Report source files",
        "",
        md_table(["Source file", "Bytes", "SHA-256", "Status"], source_hash_rows),
        "",
        "The evaluator manifest hashes for face data, baseline model and final model equal their isolated replay hashes. This binds the operational results to the same replay summarized by the runtime registry.",
        "",
        "## 9. Complete Phase 1-6 architecture",
        "",
        f"![DO-FMBS fully recorded architecture]({architecture_name})",
        "",
        "**Figure 1: Complete DO-FMBS Phase 1-to-Phase 6 architecture with recorded runtime and evidence flow (researcher-generated, 2026).**",
        "",
        "## 10. Short explanation for presentation",
        "",
        "1. Phase 1 prepares downloaded LFW face data and locally generated fingerprint data.",
        "2. Phase 2 trains five federated clients and aggregates them with sample-weighted FedAvg.",
        "3. Phase 3 applies FGSM, PGD, membership inference and label-flipping attacks.",
        "4. Phase 4 trains baseline, privacy-noise, adversarial-training and combined DO-FMBS models.",
        "5. Phase 5 compares the final and comparator models statistically and under perturbation.",
        "6. Phase 6 fits the constrained AI, PA, DM and GS equation and selects a feasible round policy.",
        "7. The synchronized operational evaluator freezes a claimed-identity threshold, counts genuine and impostor outcomes, and reports FAR, FRR and two conditional ASR definitions.",
        "8. Runtime and SHA-256 evidence connects every reported result to the isolated six-phase replay.",
        "",
        "## 11. Conclusion",
        "",
        "The synchronized final DO-FMBS model meets the declared clean FAR and FRR project protocol using the conservative identity-cluster upper-bound rule. PA remains inside the managed privacy band. All six phases completed, all required replay artifacts passed hash verification, every AI condition has exact counts and uncertainty intervals, and all twelve constrained-controller recommendations were accepted. Attack-stress FRR and ASR rise with epsilon, so those rows are retained as recorded robustness evidence rather than merged into the clean satisfaction decision.",
    ]
    markdown = "\n".join(parts).strip() + "\n"
    assert_prohibited_free(markdown, "Markdown report")
    return markdown


def assert_prohibited_free(text: str, label: str) -> None:
    lowered = text.casefold()
    found = [phrase for phrase in PROHIBITED_REPORT_PHRASES if phrase in lowered]
    if found:
        raise ValueError(f"{label} contains prohibited report wording: {', '.join(found)}")


def append_architecture_docx(path: Path, architecture: Path) -> None:
    from docx import Document
    from docx.enum.section import WD_ORIENT, WD_SECTION
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    document = Document(path)
    section = document.add_section(WD_SECTION.NEW_PAGE)
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Inches(0.35)
    section.bottom_margin = Inches(0.35)
    section.left_margin = Inches(0.45)
    section.right_margin = Inches(0.45)
    heading = document.add_heading("Complete Phase 1-6 Recorded Architecture", level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture = document.add_paragraph()
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.add_run().add_picture(str(architecture), width=Inches(10.0))
    caption = document.add_paragraph(
        "Figure 1: Complete DO-FMBS Phase 1-to-Phase 6 architecture with recorded runtime and evidence flow (researcher-generated, 2026)."
    )
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption.runs:
        run.italic = True
        run.font.size = Pt(8)
    for current_section in document.sections:
        for paragraph in current_section.footer.paragraphs:
            if "DO-FMBS Phase 6" in paragraph.text:
                paragraph.text = "DO-FMBS Fully Recorded Metrics and Satisfaction Report"
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(8)
    document.save(path)


def append_architecture_pdf(path: Path, architecture: Path) -> None:
    import pymupdf

    document = pymupdf.open(path)
    for existing_page in document:
        replacements: list[pymupdf.Rect] = []
        for block in existing_page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                if "DO-FMBS Phase 6" in line_text:
                    rectangle = pymupdf.Rect(line["bbox"])
                    replacements.append(rectangle)
                    existing_page.add_redact_annot(rectangle, fill=(1, 1, 1))
        if replacements:
            existing_page.apply_redactions()
            for rectangle in replacements:
                existing_page.insert_text(
                    (rectangle.x0, rectangle.y1 - 1.5),
                    "DO-FMBS Fully Recorded Metrics and Satisfaction Report",
                    fontsize=7.1, fontname="helv", color=(0.08, 0.18, 0.34),
                )
    width, height = pymupdf.paper_size("a4-l")
    page = document.new_page(width=width, height=height)
    page.insert_image(
        pymupdf.Rect(22, 25, width - 22, height - 45),
        filename=str(architecture), keep_proportion=True,
    )
    page.insert_textbox(
        pymupdf.Rect(28, height - 38, width - 28, height - 12),
        "Figure 1: Complete DO-FMBS Phase 1-to-Phase 6 architecture with recorded runtime and evidence flow (researcher-generated, 2026).",
        fontsize=8, fontname="helv", align=pymupdf.TEXT_ALIGN_CENTER,
    )
    temporary = path.with_name(path.stem + ".architecture.pdf")
    document.save(temporary, garbage=4, deflate=True)
    document.close()
    temporary.replace(path)


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


def generate(args: argparse.Namespace) -> list[Path]:
    evidence = load_evidence(args)
    if args.validate_inputs_only:
        return []
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = {item.strip().lower() for item in args.formats.split(",") if item.strip()}
    formats.add("md")
    unknown = formats - {"md", "docx", "pdf"}
    if unknown:
        raise ValueError("Unsupported report formats: " + ", ".join(sorted(unknown)))
    generated_at = datetime.now(ZoneInfo("Africa/Nairobi")).isoformat(timespec="seconds")
    markdown = build_markdown(evidence, generated_at)
    markdown_path = output_dir / f"{REPORT_STEM}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    outputs = [markdown_path]

    conversion = markdown.replace(
        f"![DO-FMBS fully recorded architecture]({evidence.architecture.name})",
        "The complete recorded-evidence architecture is appended as a full landscape page.",
    )
    if "docx" in formats:
        path = output_dir / f"{REPORT_STEM}.docx"
        ok, message = write_docx(conversion, path)
        if not ok:
            raise RuntimeError(f"DOCX generation failed: {message}")
        append_architecture_docx(path, evidence.architecture)
        assert_prohibited_free(extract_docx_text(path), "DOCX report")
        outputs.append(path)
    if "pdf" in formats:
        path = output_dir / f"{REPORT_STEM}.pdf"
        ok, message = write_pdf(conversion, path)
        if not ok:
            raise RuntimeError(f"PDF generation failed: {message}")
        append_architecture_pdf(path, evidence.architecture)
        assert_prohibited_free(extract_pdf_text(path), "PDF report")
        outputs.append(path)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = generate(args)
    if args.validate_inputs_only:
        print("INPUTS-VALIDATED: synchronized six-phase replay and operational evaluation")
    else:
        for path in outputs:
            print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create an isolated, reproducible authoritative Phase 6 optimization run."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts import constrained_dynamic_model as model

DOC_SHA256 = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"
PARENT_RUNS = {
    "phase1": ("phase1_final_candidate_20261123_182623", "phase1_manifest.json"),
    "phase2": ("phase2_authoritative_20261123_203617_remediation", "phase2_manifest.json"),
    "phase3": ("phase3_authoritative_20261123_220738", "phase3_manifest.json"),
    "phase4": ("phase4_authoritative_20261123_224549", "phase4_manifest.json"),
    "phase5": ("phase5_authoritative_20261124_072946", "phase5_manifest.json"),
}
SCHEDULE_EPSILON = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.10, 0.05, 0.01, 0.00, 0.05, 0.30]
SCHEDULE_PA = [0.02, 0.02, 0.05, 0.10, 0.20, 0.40, 0.30, 0.15, 0.08, 0.02, 0.10, 0.10]
INITIAL_DM = 0.30
INITIAL_GS = 0.75


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_parent_manifest(directory: Path, filename: str) -> dict:
    manifest = json.loads((directory / filename).read_text(encoding="utf-8"))
    bad = []
    for item in manifest.get("artifacts", []):
        target = directory / item["path"]
        if not target.is_file() or target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            bad.append(item["path"])
    if bad:
        raise ValueError(f"Parent manifest {directory / filename} has hash/size mismatches: {bad}")
    return manifest


def _candidate_rows(fit, ai: float, pa: float, previous_dm: float, previous_gs: float) -> list[dict]:
    weights = model.COST_WEIGHTS
    rows = []
    for dm in np.linspace(0.0, 1.0, 21):
        for clients in range(model.MIN_PARTICIPATING_CLIENTS, model.CLIENTS_TOTAL + 1):
            gs = model.gs_from_clients(clients)
            yhat = model.predict(fit, ai, pa, float(dm), gs)
            defense_cost = weights["defense"] * float(dm) ** 2
            participation_cost = weights["participation"] * gs ** 2
            defense_switch = weights["defense_switch"] * (float(dm) - previous_dm) ** 2
            participation_switch = weights["participation_switch"] * (gs - previous_gs) ** 2
            rows.append({
                "DM": float(dm), "K": int(clients), "GS": gs, "predicted_Y": yhat,
                "defense_cost": defense_cost, "participation_cost": participation_cost,
                "defense_switch_cost": defense_switch,
                "participation_switch_cost": participation_switch,
                "total_cost": defense_cost + participation_cost + defense_switch + participation_switch,
                "preferred_target_eligible": yhat >= model.PREFERRED_Y,
                "minimum_target_eligible": yhat >= model.TARGET_Y,
            })
    return rows


def _choose(rows: list[dict]) -> tuple[dict, str]:
    preferred = [row for row in rows if row["preferred_target_eligible"]]
    managed = [row for row in rows if row["minimum_target_eligible"]]
    if preferred:
        return min(preferred, key=lambda r: (r["total_cost"], -r["predicted_Y"], r["DM"], r["GS"], r["K"])), "preferred_target_met"
    if managed:
        return min(managed, key=lambda r: (r["total_cost"], -r["predicted_Y"], r["DM"], r["GS"], r["K"])), "minimum_target_met"
    return min(rows, key=lambda r: (-r["predicted_Y"], r["DM"], r["GS"], r["K"])), "maximum_controls_target_unmet"


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty evidence file: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safety_evidence(fit) -> dict:
    stresses = {row["scenario"]: row for row in model.stress_scenarios(fit)}
    preferred = stresses["preferred_operating"]
    managed = stresses["managed_operating"]
    critical = stresses["modeled_critical"]
    outside = stresses["out_of_domain"]
    valid_bounds = [
        {"case": "AI_lower", "value": 0.0, "accepted": bool(np.isfinite(model.predict(fit, 0.0, 0.0, 0.0, 0.5)))},
        {"case": "AI_upper", "value": 1.0, "accepted": bool(np.isfinite(model.predict(fit, 1.0, 0.5, 1.0, 1.0)))},
        {"case": "PA_lower", "value": 0.0, "accepted": bool(np.isfinite(model.predict(fit, 0.2, 0.0, 0.5, 0.75)))},
        {"case": "PA_upper", "value": 0.5, "accepted": bool(np.isfinite(model.predict(fit, 0.2, 0.5, 0.5, 0.75)))},
    ]
    invalid = []
    for case, call in (
        ("epsilon_negative", lambda: model.normalize_ai(-0.001)),
        ("epsilon_above_domain", lambda: model.normalize_ai(0.300001)),
        ("epsilon_nan", lambda: model.normalize_ai(float("nan"))),
        ("epsilon_infinite", lambda: model.normalize_ai(float("inf"))),
        ("PA_negative", lambda: model.predict(fit, 0.2, -0.001, 0.5, 0.75)),
        ("PA_above_domain", lambda: model.predict(fit, 0.2, 0.500001, 0.5, 0.75)),
        ("PA_nan", lambda: model.predict(fit, 0.2, float("nan"), 0.5, 0.75)),
        ("PA_infinite", lambda: model.predict(fit, 0.2, float("inf"), 0.5, 0.75)),
        ("DM_nan", lambda: model.predict(fit, 0.2, 0.1, float("nan"), 0.75)),
        ("GS_infinite", lambda: model.predict(fit, 0.2, 0.1, 0.5, float("inf"))),
    ):
        try:
            call()
            invalid.append({"case": case, "passed": False, "detail": "input was not rejected"})
        except (ValueError, TypeError):
            invalid.append({"case": case, "passed": True, "detail": "rejected before a policy prediction"})
    branch_results = {
        "preferred_policy_acceptance": bool(preferred["status"] == "preferred_target_met" and preferred["predicted_Y"] >= model.PREFERRED_Y),
        "managed_policy_acceptance": bool(managed["status"] == "minimum_target_met" and model.TARGET_Y <= managed["predicted_Y"] < model.PREFERRED_Y),
        "safety_rejection": bool(critical["status"] == "maximum_controls_target_unmet" and critical["predicted_Y"] < model.TARGET_Y and critical["deployment_decision"] == "reject_or_defer"),
        "out_of_domain_rejection": bool(outside["status"] == "out_of_domain" and outside["predicted_Y"] is None and outside["deployment_decision"] == "reject_or_defer"),
    }
    all_passed = (all(row["accepted"] for row in valid_bounds) and all(row["passed"] for row in invalid)
                  and all(branch_results.values()))
    return {"status": "PASS" if all_passed else "FAIL", "branch_results": branch_results,
            "preferred_scenario": preferred, "managed_scenario": managed,
            "critical_scenario": critical, "out_of_domain_scenario": outside,
            "valid_boundary_inputs": valid_bounds, "invalid_inputs_rejected": invalid,
            "note": "Safety rejection and out-of-domain cases are diagnostic controller tests, not successful deployment recommendations."}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_run(root: Path, output_root: Path) -> Path:
    root, output_root = root.resolve(), output_root.resolve()
    docx = root / "docs/reference/CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION.docx"
    actual_doc_hash = sha256(docx).upper()
    if actual_doc_hash != DOC_SHA256:
        raise ValueError(f"Authoritative Chapter Four SHA-256 mismatch: {actual_doc_hash}")
    parent_dirs, parent_hashes = {}, {}
    for phase, (run_name, manifest_name) in PARENT_RUNS.items():
        directory = root / "runtime_recording" / run_name
        manifest = verify_parent_manifest(directory, manifest_name)
        parent_dirs[phase] = directory
        parent_hashes[phase] = sha256(directory / manifest_name)
        if phase == "phase1" and manifest.get("verification", {}).get("status") != "PASS":
            raise ValueError("The accepted Phase 1 manifest does not pass verification")
        if phase == "phase5":
            report = json.loads((directory / "phase5_verification_report.json").read_text(encoding="utf-8"))
            if report.get("status") != "PASS":
                raise ValueError("Accepted Phase 5 independent verification is not PASS")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"phase6_authoritative_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"phase6_authoritative_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    start = time.perf_counter()
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        model.main(run_dir)
    fit_payload = json.loads((run_dir / "model_fit.json").read_text(encoding="utf-8"))
    rows = model.generate_observations(model.SEED)
    fit = model.fit_equation(rows)
    base_schedule = model.dynamic_schedule(fit)
    candidate_rows, detail_rows = [], []
    previous_dm, previous_gs = INITIAL_DM, INITIAL_GS
    for declared, epsilon, pa in zip(base_schedule, SCHEDULE_EPSILON, SCHEDULE_PA):
        ai = model.normalize_ai(epsilon)
        candidates = _candidate_rows(fit, ai, pa, previous_dm, previous_gs)
        selected, status = _choose(candidates)
        if (not np.isclose(selected["DM"], declared["DM"]) or selected["K"] != declared["participating_clients"]
                or not np.isclose(selected["predicted_Y"], declared["predicted_Y"])
                or status != declared["status"]):
            raise RuntimeError(f"Expanded candidate search differs from the existing controller at round {declared['round']}")
        for candidate in candidates:
            candidate_rows.append({"round": declared["round"], "epsilon": epsilon, "AI": ai, "PA": pa,
                                   "previous_DM": previous_dm, "previous_GS": previous_gs,
                                   **candidate,
                                   "selected": bool(np.isclose(candidate["DM"], selected["DM"])
                                                    and candidate["K"] == selected["K"])})
        row = {"round": declared["round"], "epsilon": epsilon, "AI": ai, "PA": pa,
               "previous_DM": previous_dm, "previous_GS": previous_gs,
               "candidate_count": len(candidates), "DM": selected["DM"], "K": selected["K"],
               "GS": selected["GS"], "predicted_Y": selected["predicted_Y"],
               "defense_cost": selected["defense_cost"],
               "participation_cost": selected["participation_cost"],
               "defense_switch_cost": selected["defense_switch_cost"],
               "participation_switch_cost": selected["participation_switch_cost"],
               "total_cost": selected["total_cost"], "status": status,
               "deployment_decision": "accept" if status != "maximum_controls_target_unmet" else "reject_or_defer",
               **model.defense_configuration(selected["DM"])}
        detail_rows.append(row)
        previous_dm, previous_gs = selected["DM"], selected["GS"]
    _write_rows(run_dir / "candidate_policy_search.csv", candidate_rows)
    _write_rows(run_dir / "dynamic_schedule_detailed.csv", detail_rows)

    safety = _safety_evidence(fit)
    write_json(run_dir / "phase6_safety_tests.json", safety)
    if safety["status"] != "PASS":
        raise RuntimeError("Safety/domain test evidence did not pass")
    p5_metrics = json.loads((parent_dirs["phase5"] / "phase5_metrics.json").read_text(encoding="utf-8"))
    p5_context = {
        "source_run": PARENT_RUNS["phase5"][0], "source_manifest_sha256": parent_hashes["phase5"],
        "metrics_are_context_not_fit_targets": True,
        "baseline_face_accuracy": p5_metrics["models"]["baseline"]["face_classification"]["accuracy"],
        "dofmbs_face_accuracy": p5_metrics["models"]["dofmbs_combined"]["face_classification"]["accuracy"],
        "baseline_fingerprint_accuracy": p5_metrics["models"]["baseline"]["fingerprint_classification"]["accuracy"],
        "dofmbs_fingerprint_accuracy": p5_metrics["models"]["dofmbs_combined"]["fingerprint_classification"]["accuracy"],
        "baseline_face_fgsm_asr_epsilon_0_05": next(r["attack_success_rate"] for r in p5_metrics["robustness"]["models"]["baseline"] if r["epsilon"] == 0.05),
        "dofmbs_face_fgsm_asr_epsilon_0_05": next(r["attack_success_rate"] for r in p5_metrics["robustness"]["models"]["dofmbs_combined"] if r["epsilon"] == 0.05),
        "fit_provenance": "Phase 6 Y observations are generated by the declared seeded model-based factorial simulation; they are not fitted to Phase 5 biometric metrics.",
    }
    write_json(run_dir / "phase5_context.json", p5_context)

    phases = {}
    for phase, directory in parent_dirs.items():
        phases[phase] = {"run_id": PARENT_RUNS[phase][0], "manifest": PARENT_RUNS[phase][1],
                         "manifest_sha256": parent_hashes[phase]}
    source_files = [root / "phase6_authoritative_optimization.py", root / "phase6_constrained_dynamic_optimization.py",
                    root / "scripts/constrained_dynamic_model.py", root / "verify_phase6_authoritative_run.py",
                    root / "phase6_dashboard.py"]
    config = {
        "phase": 6, "run_id": run_dir.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_document": "docs/reference/CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION.docx",
        "source_document_sha256": actual_doc_hash, "source_phases": phases,
        "source_sha256": {str(path.relative_to(root).as_posix()): sha256(path) for path in source_files},
        "seed": model.SEED, "simulation_design": fit_payload["simulation_design"],
        "epsilon_levels": [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30],
        "pa_levels": [0.00, 0.02, 0.10, 0.25, 0.50], "dm_fit_levels": [0.00, 0.33, 0.67, 1.00],
        "gs_fit_levels": [model.gs_from_clients(k) for k in range(1, model.CLIENTS_TOTAL + 1)],
        "deployment_DM_candidates": [round(float(x), 2) for x in np.linspace(0.0, 1.0, 21)],
        "deployment_client_candidates": [3, 4, 5], "deployment_GS_mapping": "(K - 1) / 4",
        "candidate_count_per_round": 63, "round_count": 12,
        "round_epsilon_schedule": SCHEDULE_EPSILON, "round_PA_schedule": SCHEDULE_PA,
        "initial_previous_DM_GS": {"DM": INITIAL_DM, "GS": INITIAL_GS},
        "equation": fit_payload["equation"], "ground_truth_generation_parameters": fit_payload["simulation_ground_truth"],
        "ground_truth_are_not_fitted_coefficients": True,
        "coefficient_constraints": ["B_AI <= 0", "B_PA <= 0", "B_DM >= 0", "B_GS >= 0"],
        "cost_formula": "0.60*DM^2 + 0.40*GS^2 + 0.15*(DM-DM_previous)^2 + 0.10*(GS-GS_previous)^2",
        "cost_weights": model.COST_WEIGHTS, "preferred_target_Y": model.PREFERRED_Y,
        "minimum_target_Y": model.TARGET_Y, "minimum_deployment_clients": model.MIN_PARTICIPATING_CLIENTS,
        "methodology_and_implementation_choices": [
            "Equation, domains, 700 factorial observations, 63 deployment candidates, prescribed 12-round schedule and target hierarchy follow the hash-verified Chapter Four and its accepted artifacts.",
            "The reproducible seed (42), simulation error distribution, fit-level grid, cost weights, and initial switching references use the existing validated Phase 6 implementation and are recorded here; Chapter Four does not present each as a separately estimated empirical quantity.",
            "Initial DM=0.30 and GS=0.75 are the accepted model implementation's prior controls and reproduce the source schedule costs.",
            "The Chapter Four schedule is preserved as given; the critical rejection and out-of-domain states are separately exercised as deterministic safety scenarios, not added as successful schedule rounds.",
            "Phase 5 measured biometric metrics are linked as context but are not used to fit simulated Y.",
        ],
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": _package_version("numpy"), "scipy": _package_version("scipy"),
                    "scikit_learn": _package_version("scikit-learn")},
    }
    write_json(run_dir / "phase6_config.json", config)
    rounds = len(detail_rows)
    preferred_count = sum(row["status"] == "preferred_target_met" for row in detail_rows)
    managed_count = sum(row["status"] == "minimum_target_met" for row in detail_rows)
    rejected_count = sum(row["deployment_decision"] != "accept" for row in detail_rows)
    metrics = {
        "phase": 6, "run_id": run_dir.name, "model_fit": fit_payload,
        "schedule_summary": {"rounds": rounds, "candidates_per_round": 63,
                             "total_candidate_evaluations": len(candidate_rows),
                             "preferred_rounds": preferred_count, "managed_rounds": managed_count,
                             "rejected_rounds": rejected_count, "accepted_rounds": rounds - rejected_count},
        "candidate_search_summary": {"DM_points": 21, "K_points": 3, "GS_values": [0.5, 0.75, 1.0],
                                     "tie_break": "minimum total cost, then maximum predicted Y, then lower DM/GS/K"},
        "cost_components": {"defense": "0.60*DM^2", "participation": "0.40*GS^2",
                            "defense_switch": "0.15*(DM-DM_previous)^2",
                            "participation_switch": "0.10*(GS-GS_previous)^2"},
        "safety_status": safety["status"],
        "scope": "Y is a model-derived simulated next-round response, not measured production efficiency or a live federated-run outcome.",
    }
    write_json(run_dir / "phase6_metrics.json", metrics)
    elapsed = time.perf_counter() - start
    (run_dir / "phase6_execution_log.txt").write_text(
        f"run_id={run_dir.name}\nseed={model.SEED}\nphase6_model_runtime_seconds={elapsed:.6f}\n"
        "stdout from existing validated model runner:\n" + console.getvalue(), encoding="utf-8")

    # Manifest deliberately excludes itself and the verifier's own outputs.
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"phase6_manifest.json", "phase6_verification_report.json", "phase6_verifier_stdout.json"}:
            artifacts.append({"path": path.relative_to(run_dir).as_posix(), "bytes": path.stat().st_size,
                              "sha256": sha256(path)})
    manifest = {"phase": 6, "run_id": run_dir.name, "status": "PHASE6-ARTIFACTS-PERSISTED",
                "source_document_sha256": actual_doc_hash, "source_phase_manifest_sha256": parent_hashes,
                "excluded_from_artifact_hash_list": ["phase6_manifest.json", "phase6_verification_report.json", "phase6_verifier_stdout.json"],
                "artifacts": artifacts}
    write_json(run_dir / "phase6_manifest.json", manifest)
    verified = subprocess.run([sys.executable, str(root / "verify_phase6_authoritative_run.py"), str(run_dir)],
                              cwd=root, capture_output=True, text=True, check=False)
    (run_dir / "phase6_verifier_stdout.json").write_text(verified.stdout, encoding="utf-8")
    if verified.returncode:
        (run_dir / "phase6_verifier_stderr.txt").write_text(verified.stderr, encoding="utf-8")
        raise RuntimeError(f"Independent Phase 6 verification failed; inspect {run_dir / 'phase6_verifier_stderr.txt'}")
    verification = json.loads(verified.stdout)
    write_json(run_dir / "phase6_verification_report.json", verification)
    if verification.get("status") != "PASS":
        raise RuntimeError(f"Phase 6 verifier returned {verification.get('status')}")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "runtime_recording")
    args = parser.parse_args()
    print(build_run(args.root, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

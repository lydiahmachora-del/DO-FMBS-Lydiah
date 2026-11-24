"""Independent, artifact-based verification of an authoritative Phase 6 run."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


DOC_SHA256 = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"
PARENTS = {
    "phase1": ("phase1_final_candidate_20261123_182623", "phase1_manifest.json"),
    "phase2": ("phase2_authoritative_20261123_203617_remediation", "phase2_manifest.json"),
    "phase3": ("phase3_authoritative_20261123_220738", "phase3_manifest.json"),
    "phase4": ("phase4_authoritative_20261123_224549", "phase4_manifest.json"),
    "phase5": ("phase5_authoritative_20261124_072946", "phase5_manifest.json"),
}
EPSILONS = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.10, 0.05, 0.01, 0.00, 0.05, 0.30]
PA_VALUES = [0.02, 0.02, 0.05, 0.10, 0.20, 0.40, 0.30, 0.15, 0.08, 0.02, 0.10, 0.10]
WEIGHTS = {"defense": 0.60, "participation": 0.40, "defense_switch": 0.15, "participation_switch": 0.10}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _check(checks: list[dict], name: str, condition: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def _verify_parent(root: Path, phase: str, config: dict, checks: list[dict]) -> None:
    run_name, manifest_name = PARENTS[phase]
    directory = root / "runtime_recording" / run_name
    manifest_path = directory / manifest_name
    manifest = _json(manifest_path)
    bad = []
    for item in manifest.get("artifacts", []):
        target = directory / item["path"]
        if not target.is_file() or target.stat().st_size != item["bytes"] or _sha(target) != item["sha256"]:
            bad.append(item["path"])
    _check(checks, f"Parent {phase} artifact manifest", not bad,
           f"run={run_name}; listed={len(manifest.get('artifacts', []))}; invalid={bad}")
    saved = config["source_phases"][phase]
    _check(checks, f"Parent {phase} lineage", saved["run_id"] == run_name
           and saved["manifest"] == manifest_name and saved["manifest_sha256"] == _sha(manifest_path),
           f"manifest_sha256={saved.get('manifest_sha256')}")
    if phase == "phase5":
        _check(checks, "Phase 5 independent acceptance", _json(directory / "phase5_verification_report.json").get("status") == "PASS",
               "accepted Phase 5 verifier report")


def verify(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    root = run_dir.parents[1]
    config = _json(run_dir / "phase6_config.json")
    fit = _json(run_dir / "model_fit.json")
    metrics = _json(run_dir / "phase6_metrics.json")
    manifest = _json(run_dir / "phase6_manifest.json")
    checks: list[dict] = []
    _check(checks, "Run identity", run_dir.name == config["run_id"] == metrics["run_id"] == manifest["run_id"], run_dir.name)

    bad_files = []
    for item in manifest.get("artifacts", []):
        path = run_dir / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            bad_files.append(item["path"])
    _check(checks, "Phase 6 artifact manifest", not bad_files and len(manifest.get("artifacts", [])) >= 12,
           f"listed={len(manifest.get('artifacts', []))}; invalid={bad_files}")
    expected_excluded = {"phase6_manifest.json", "phase6_verification_report.json", "phase6_verifier_stdout.json"}
    _check(checks, "Manifest excludes only self-referential verifier outputs",
           set(manifest["excluded_from_artifact_hash_list"]) == expected_excluded,
           str(manifest["excluded_from_artifact_hash_list"]))

    doc = root / config["source_document"]
    _check(checks, "Authoritative source identity", _sha(doc).upper() == DOC_SHA256 == config["source_document_sha256"], DOC_SHA256)
    for phase in PARENTS:
        _verify_parent(root, phase, config, checks)
    for relative, expected in config["source_sha256"].items():
        path = root / relative
        _check(checks, f"Source hash {relative}", path.is_file() and _sha(path) == expected, expected)

    observations = _csv(run_dir / "simulation_observations.csv")
    truth = {"B0": 0.40, "B_AI": -0.18, "B_PA": -0.22, "B_DM": 0.26, "B_GS": 0.16}
    rng = np.random.default_rng(42)
    eps_levels, pa_levels, dm_levels = [0.00, .01, .05, .10, .15, .20, .30], [0.00, .02, .10, .25, .50], [0.00, .33, .67, 1.00]
    gs_levels = [0.0, .25, .50, .75, 1.0]
    expected_rows = []
    obs_num = 0
    for epsilon in eps_levels:
        ai = epsilon / .30
        for pa in pa_levels:
            for dm in dm_levels:
                for gs in gs_levels:
                    obs_num += 1
                    error = float(rng.normal(0.0, .025))
                    y = np.clip(truth["B0"] + truth["B_AI"]*ai + truth["B_PA"]*pa
                                + truth["B_DM"]*dm + truth["B_GS"]*gs + error, 0.0, 1.0)
                    expected_rows.append((epsilon, ai, pa, dm, gs, error, float(y)))
    try:
        actual = np.asarray([[float(row[key]) for key in ("epsilon", "AI", "PA", "DM", "GS", "simulation_error", "Y")]
                             for row in observations])
        expected = np.asarray(expected_rows)
        simulation_matches = len(observations) == 700 and np.allclose(actual, expected, rtol=0, atol=1e-14)
    except (KeyError, ValueError):
        simulation_matches = False
    _check(checks, "Seeded 700-row factorial simulation", simulation_matches,
           f"rows={len(observations)}; levels=7x5x4x5; seed=42")

    matrix = np.column_stack((np.ones(700), actual[:, 1:5]))
    y = actual[:, 6]
    ols = np.linalg.lstsq(matrix, y, rcond=None)[0]
    estimates = np.asarray(fit["coefficients"], dtype=float)
    signs = (estimates[1] <= 1e-12 and estimates[2] <= 1e-12 and estimates[3] >= -1e-12 and estimates[4] >= -1e-12)
    _check(checks, "Constrained estimates versus independent least squares", signs and np.allclose(estimates, ols, atol=1e-8, rtol=0),
           f"estimates={estimates.tolist()}; max_abs_OLS_difference={float(np.max(np.abs(estimates-ols))):.3g}")
    predicted = matrix @ estimates
    residuals = y - predicted
    r2 = 1.0 - float(np.sum(residuals**2) / np.sum((y-y.mean())**2))
    adjusted = 1.0 - (1-r2) * (699/695)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    _check(checks, "Fit statistics independently recomputed",
           np.isclose(r2, fit["r_squared"]) and np.isclose(adjusted, fit["adjusted_r_squared"])
           and np.isclose(rmse, fit["rmse"]) and np.isclose(mae, fit["mae"]),
           f"R2={r2:.8f}; adjusted={adjusted:.8f}; RMSE={rmse:.8f}; MAE={mae:.8f}")
    _check(checks, "Solver and constraint evidence", fit["optimization_success"] is True
           and fit["maximum_constraint_violation"] <= 1e-10 and fit["active_constraints"] == [],
           f"solver={fit['estimator']}; violation={fit['maximum_constraint_violation']}")
    _check(checks, "Fit diagnostics are finite", all(np.isfinite(float(fit[k])) for k in
           ("r_squared", "adjusted_r_squared", "rmse", "mae", "shapiro_p", "durbin_watson",
            "cross_validation_r2_mean", "cross_validation_rmse_mean")) and max(fit["vifs"].values()) < 2,
           f"CV R2={fit['cross_validation_r2_mean']:.6f}; max VIF={max(fit['vifs'].values()):.6f}")

    candidate_rows = _csv(run_dir / "candidate_policy_search.csv")
    schedule = _csv(run_dir / "dynamic_schedule_detailed.csv")
    original_schedule = _csv(run_dir / "dynamic_optimization_schedule.csv")
    _check(checks, "Twelve-round schedule and full 63-policy search",
           len(schedule) == len(original_schedule) == 12 and len(candidate_rows) == 12*63
           and all(sum(int(row["round"]) == r for row in candidate_rows) == 63 for r in range(1, 13)),
           f"rounds={len(schedule)}; total candidates={len(candidate_rows)}")
    previous_dm, previous_gs = .30, .75
    detail_by_round = {int(row["round"]): row for row in schedule}
    old_by_round = {int(row["round"]): row for row in original_schedule}
    cost_ok = True
    selection_ok = True
    schedule_ok = True
    for r in range(1, 13):
        row = detail_by_round[r]
        if not (np.isclose(float(row["epsilon"]), EPSILONS[r-1]) and np.isclose(float(row["PA"]), PA_VALUES[r-1])
                and np.isclose(float(row["AI"]), EPSILONS[r-1]/.30)
                and np.isclose(float(row["previous_DM"]), previous_dm)
                and np.isclose(float(row["previous_GS"]), previous_gs)
                and int(row["candidate_count"]) == 63):
            schedule_ok = False
        candidates = [c for c in candidate_rows if int(c["round"]) == r]
        for c in candidates:
            dm, gs, yhat = float(c["DM"]), float(c["GS"]), float(c["predicted_Y"])
            components = [WEIGHTS["defense"]*dm**2, WEIGHTS["participation"]*gs**2,
                          WEIGHTS["defense_switch"]*(dm-previous_dm)**2,
                          WEIGHTS["participation_switch"]*(gs-previous_gs)**2]
            calculated_y = float(estimates @ np.asarray([1.0, float(row["AI"]), float(row["PA"]), dm, gs]))
            eligible_preferred = yhat >= .70
            eligible_minimum = yhat >= .60
            if (not np.allclose(components, [float(c[k]) for k in ("defense_cost", "participation_cost", "defense_switch_cost", "participation_switch_cost")], atol=1e-12)
                    or not np.isclose(sum(components), float(c["total_cost"]), atol=1e-12)
                    or not np.isclose(yhat, calculated_y, atol=1e-12)
                    or (str(c["preferred_target_eligible"]).lower() == "true") != eligible_preferred
                    or (str(c["minimum_target_eligible"]).lower() == "true") != eligible_minimum):
                cost_ok = False
        preferred = [c for c in candidates if str(c["preferred_target_eligible"]).lower() == "true"]
        managed = [c for c in candidates if str(c["minimum_target_eligible"]).lower() == "true"]
        pool = preferred or managed
        if pool:
            best = min(pool, key=lambda c: (float(c["total_cost"]), -float(c["predicted_Y"]),
                                            float(c["DM"]), float(c["GS"]), int(c["K"])))
            expected_status = "preferred_target_met" if preferred else "minimum_target_met"
        else:
            best = min(candidates, key=lambda c: (-float(c["predicted_Y"]), float(c["DM"]), float(c["GS"]), int(c["K"])))
            expected_status = "maximum_controls_target_unmet"
        if (not np.isclose(float(row["DM"]), float(best["DM"])) or int(row["K"]) != int(best["K"])
                or not np.isclose(float(row["GS"]), float(best["GS"]))
                or not np.isclose(float(row["predicted_Y"]), float(best["predicted_Y"]))
                or row["status"] != expected_status or str(best["selected"]).lower() != "true"):
            selection_ok = False
        old = old_by_round[r]
        if (not np.isclose(float(old["DM"]), float(row["DM"]))
                or int(old["participating_clients"]) != int(row["K"])
                or not np.isclose(float(old["predicted_Y"]), float(row["predicted_Y"]))
                or not np.isclose(float(old["control_cost"]), float(row["total_cost"]))):
            schedule_ok = False
        previous_dm, previous_gs = float(row["DM"]), float(row["GS"])
    _check(checks, "Every candidate prediction, cost component and eligibility", cost_ok,
           "Recomputed all 756 rows from beta, threat state, controls and switching references")
    _check(checks, "Target hierarchy and minimum-cost policy selection", selection_ok,
           "Selected candidate is the minimum-cost feasible preferred policy, otherwise managed, otherwise max-Y diagnostic")
    _check(checks, "Prescribed schedule and switching-state carry-forward", schedule_ok,
           "12 epsilon/PA states match Chapter Four and selected DM/GS carry into the next round")

    safety = _json(run_dir / "phase6_safety_tests.json")
    _check(checks, "Preferred, managed, safety-reject and domain-reject branches", safety["status"] == "PASS"
           and all(safety["branch_results"].values()), json.dumps(safety["branch_results"], sort_keys=True))
    _check(checks, "Boundary acceptance and invalid/nonfinite input rejection",
           all(x["accepted"] for x in safety["valid_boundary_inputs"])
           and all(x["passed"] for x in safety["invalid_inputs_rejected"]),
           f"accepted_bounds={len(safety['valid_boundary_inputs'])}; rejected_invalid={len(safety['invalid_inputs_rejected'])}")
    summary = metrics["schedule_summary"]
    observed_counts = {"preferred_rounds": sum(row["status"] == "preferred_target_met" for row in schedule),
                       "managed_rounds": sum(row["status"] == "minimum_target_met" for row in schedule),
                       "rejected_rounds": sum(row["deployment_decision"] != "accept" for row in schedule)}
    _check(checks, "Persisted schedule summary", summary["rounds"] == 12 and summary["total_candidate_evaluations"] == 756
           and all(summary[k] == v for k, v in observed_counts.items()), str({**summary, **observed_counts}))

    # Confirm that artifact provenance distinguishes simulation parameters from fitted betas.
    _check(checks, "Synthetic response-model scope", fit["observations"] == 700
           and "not neural-network weights" in fit["scope_statement"].lower()
           and config["ground_truth_are_not_fitted_coefficients"] is True,
           fit["scope_statement"])
    return {"phase": 6, "run_id": run_dir.name, "status": "PASS", "independent": True,
            "check_count": len(checks), "failed_checks": [], "checks": checks}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), sort_keys=True))

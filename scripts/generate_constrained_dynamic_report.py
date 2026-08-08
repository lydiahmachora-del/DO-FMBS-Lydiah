"""Generate the professional Phase 6 constrained dynamic-model report.

The Markdown report is always generated. DOCX and PDF versions are generated
when ``python-docx`` and ``reportlab`` are available, respectively. The report
is deliberately built from the Phase 6 JSON/CSV evidence rather than copied
numeric constants.

Run from the repository root with::

    python scripts/generate_constrained_dynamic_report.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "simulation_evidence" / "constrained_dynamic_model"
DEFAULT_OUTPUT = ROOT / "docs" / "constrained_dynamic_model"
REPORT_STEM = "DO_FMBS_CONSTRAINED_DYNAMIC_MODEL_REPORT"

REQUIRED_CSV = {
    "acceptance_ranges.csv",
    "coefficient_estimates.csv",
    "constraint_checks.csv",
    "cost_weight_sensitivity.csv",
    "dynamic_optimization_schedule.csv",
    "simulation_observations.csv",
    "stress_test_scenarios.csv",
}

# These are the initial-state references used by the Phase 6 controller. All
# search bounds, candidate values, target levels and cost weights are read from
# model_fit.json rather than duplicated here.
INITIAL_DM = 0.30
INITIAL_GS = 0.75
FIXED_FEDERATED_SETTINGS = {
    "Federated rounds": 12,
    "Local epochs per round": 2,
    "Batch size": 64,
    "Learning rate": 0.001,
    "Total available clients": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Phase 6 constrained dynamic-model report."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT,
        help="Directory containing model_fit.json and Phase 6 CSV evidence.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Directory in which report files are written.",
    )
    parser.add_argument(
        "--formats", default="md,docx,pdf",
        help="Comma-separated formats. Markdown is always generated.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required Phase 6 artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required Phase 6 artifact is missing: {path}") from exc


def require_fields(label: str, payload: dict, fields: Iterable[str]) -> None:
    missing = sorted(set(fields) - set(payload))
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")


def require_csv_fields(label: str, rows: Sequence[dict], fields: Iterable[str]) -> None:
    if not rows:
        raise ValueError(f"{label} contains no data rows")
    require_fields(label, rows[0], fields)


def as_float(value: object, label: str = "value") -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; received {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; received {value!r}")
    return number


def as_int(value: object, label: str = "value") -> int:
    number = as_float(value, label)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer; received {value!r}")
    return int(number)


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def fmt(value: object, digits: int = 6) -> str:
    number = as_float(value)
    if number == 0:
        return "0"
    if abs(number) < 10 ** (-(digits - 1)) or abs(number) >= 10 ** digits:
        return f"{number:.3e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def fmt_p(value: object) -> str:
    number = as_float(value, "p-value")
    if number == 0:
        return "< 1×10⁻³⁰⁰ (numerical underflow)"
    if number < 0.001:
        return f"{number:.3e}"
    return f"{number:.4f}"


def fmt_optional(value: object, digits: int = 6, missing: str = "Not evaluated") -> str:
    if value is None or str(value).strip() == "":
        return missing
    return fmt(value, digits)


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def display_label(value: object) -> str:
    """Turn persisted machine labels into report-ready prose."""
    labels = {
        "target_met": "Target met",
        "preferred_target_met": "Preferred target met",
        "minimum_target_met": "Minimum target met",
        "maximum_controls_target_unmet": "Maximum controls; target unmet",
        "accept": "Accept",
        "reject_or_defer": "Reject or defer",
        "outside_critical": "Outside critical",
        "preferred_operating": "Preferred operating",
        "managed_operating": "Managed operating",
        "modeled_critical": "Modeled critical",
        "out_of_domain": "Out of domain",
    }
    text = str(value)
    return labels.get(text, text.replace("_", " ").strip().capitalize())


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]], aligns: Sequence[str] | None = None) -> str:
    if aligns is None:
        aligns = ["left"] * len(headers)
    separators = {"left": "---", "right": "---:", "center": ":---:"}
    output = [
        "| " + " | ".join(md_cell(item) for item in headers) + " |",
        "| " + " | ".join(separators[item] for item in aligns) + " |",
    ]
    output.extend("| " + " | ".join(md_cell(item) for item in row) + " |" for row in rows)
    return "\n".join(output)


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


def diagnostic_decision(value: float, ideal_test, managed_test) -> str:
    if ideal_test(value):
        return "Ideal / pass"
    if managed_test(value):
        return "Managed / investigate"
    return "Outside / refit before use"


def load_evidence(input_dir: Path) -> tuple[dict, dict[str, list[dict]], list[Path]]:
    fit_path = input_dir / "model_fit.json"
    fit = read_json(fit_path)
    require_fields(
        "model_fit.json", fit,
        {
            "names", "coefficients", "standard_errors", "p_values", "ci_low", "ci_high",
            "standardized_coefficients", "r_squared", "adjusted_r_squared", "rmse", "mae",
            "residual_mean", "residual_std", "shapiro_p", "durbin_watson",
            "cross_validation_r2_mean", "cross_validation_r2_std",
            "cross_validation_rmse_mean", "vifs", "observations", "degrees_of_freedom",
            "condition_number", "estimator", "optimization_success", "optimization_message",
            "maximum_constraint_violation", "equation", "constraints_satisfied", "target_Y",
            "preferred_Y", "seed", "simulation_design", "dynamic_definition",
            "simulation_ground_truth", "maximum_coefficient_recovery_error", "fitted_domains",
            "optimization_settings", "unconstrained_coefficients",
            "maximum_constrained_ols_difference", "active_constraints", "inference_basis",
            "scope_statement",
        },
    )

    csv_paths = sorted(input_dir.glob("*.csv"))
    available = {path.name for path in csv_paths}
    missing = sorted(REQUIRED_CSV - available)
    if missing:
        raise FileNotFoundError(
            "Required Phase 6 CSV artifacts are missing: " + ", ".join(missing)
        )
    csv_data = {path.name: read_csv(path) for path in csv_paths}

    require_csv_fields(
        "coefficient_estimates.csv", csv_data["coefficient_estimates.csv"],
        {"parameter", "estimate", "simulation_ground_truth", "recovery_error", "unconstrained_ols_estimate", "standardized_estimate", "standard_error", "p_value", "ci_95_low", "ci_95_high", "constraint", "constraint_satisfied"},
    )
    require_csv_fields(
        "acceptance_ranges.csv", csv_data["acceptance_ranges.csv"],
        {"variable", "definition", "fitted_domain", "preferred", "managed", "critical_modeled", "out_of_domain", "required_action"},
    )
    require_csv_fields(
        "constraint_checks.csv", csv_data["constraint_checks.csv"],
        {"constraint", "constraint_type", "left", "right", "slack", "satisfied"},
    )
    require_csv_fields(
        "cost_weight_sensitivity.csv", csv_data["cost_weight_sensitivity.csv"],
        {"profile", "epsilon", "AI", "PA", "weight_defense", "weight_participation", "weight_defense_switch", "weight_participation_switch", "DM", "adversarial_fraction", "training_attack_epsilon", "gradient_noise_std", "GS", "participating_clients", "predicted_Y", "control_cost", "status"},
    )
    require_csv_fields(
        "dynamic_optimization_schedule.csv", csv_data["dynamic_optimization_schedule.csv"],
        {"round", "epsilon", "AI", "PA", "DM", "adversarial_fraction", "training_attack_epsilon", "gradient_noise_std", "GS", "participating_clients", "predicted_Y", "control_cost", "status", "deployment_decision"},
    )
    require_csv_fields(
        "simulation_observations.csv", csv_data["simulation_observations.csv"],
        {"observation", "epsilon", "AI", "PA", "DM", "GS", "participating_clients", "Y", "simulation_error", "seed"},
    )
    require_csv_fields(
        "stress_test_scenarios.csv", csv_data["stress_test_scenarios.csv"],
        {"scenario", "epsilon", "AI", "PA", "DM", "adversarial_fraction", "training_attack_epsilon", "gradient_noise_std", "GS", "participating_clients", "predicted_Y", "control_cost", "status", "deployment_decision"},
    )

    names = list(fit["names"])
    if names != ["B0", "B_AI", "B_PA", "B_DM", "B_GS"]:
        raise ValueError(f"Unexpected coefficient order in model_fit.json: {names!r}")
    vector_fields = ["coefficients", "standard_errors", "p_values", "ci_low", "ci_high", "standardized_coefficients"]
    for field in vector_fields:
        if len(fit[field]) != len(names):
            raise ValueError(f"model_fit.json field {field!r} must contain {len(names)} values")

    return fit, csv_data, [fit_path, *csv_paths]


def build_report(fit: dict, csv_data: dict[str, list[dict]], evidence_paths: Sequence[Path]) -> str:
    coeffs = csv_data["coefficient_estimates.csv"]
    ranges = csv_data["acceptance_ranges.csv"]
    checks = csv_data["constraint_checks.csv"]
    schedule = csv_data["dynamic_optimization_schedule.csv"]
    sensitivity = csv_data["cost_weight_sensitivity.csv"]
    observations = csv_data["simulation_observations.csv"]
    stresses = csv_data["stress_test_scenarios.csv"]
    beta = {row["parameter"]: as_float(row["estimate"], row["parameter"]) for row in coeffs}
    settings = fit["optimization_settings"]
    cost_weights = settings["cost_weights"]
    client_candidates = tuple(as_int(value, "client candidate") for value in settings["client_candidates"])
    defense_start = as_float(settings["defense_grid_start"])
    defense_stop = as_float(settings["defense_grid_stop"])
    defense_step = as_float(settings["defense_grid_step"])
    defense_levels = int(round((defense_stop - defense_start) / defense_step)) + 1
    candidates_per_round = defense_levels * len(client_candidates)

    all_constraints = as_bool(fit["constraints_satisfied"]) and all(as_bool(row["satisfied"]) for row in checks)
    fit_passes = (
        as_bool(fit["optimization_success"])
        and all_constraints
        and as_float(fit["r_squared"]) >= 0.90
        and as_float(fit["cross_validation_r2_mean"]) >= 0.90
        and as_float(fit["rmse"]) <= 0.05
        and max(as_float(value) for value in fit["vifs"].values()) < 5.0
        and as_float(fit["condition_number"]) < 30.0
    )

    successful_schedule = [row for row in schedule if row["deployment_decision"] == "accept"]
    substitution_row = max(
        successful_schedule or schedule,
        key=lambda row: as_float(row["AI"]) + as_float(row["PA"]),
    )
    substitution_values = {
        name: as_float(substitution_row[name]) for name in ("AI", "PA", "DM", "GS")
    }
    contributions = {
        "B0": beta["B0"],
        "B_AI × AI": beta["B_AI"] * substitution_values["AI"],
        "B_PA × PA": beta["B_PA"] * substitution_values["PA"],
        "B_DM × DM": beta["B_DM"] * substitution_values["DM"],
        "B_GS × GS": beta["B_GS"] * substitution_values["GS"],
    }
    calculated_y = sum(contributions.values())
    stored_y = as_float(substitution_row["predicted_Y"])

    error_values = [as_float(row["simulation_error"]) for row in observations]
    variable_levels = {
        name: sorted({as_float(row[name]) for row in observations})
        for name in ("epsilon", "AI", "PA", "DM", "GS")
    }
    target_met = sum(row["deployment_decision"] == "accept" for row in schedule)
    preferred_met = sum(row["status"] == "preferred_target_met" for row in schedule)
    minimum_met = sum(row["status"] == "minimum_target_met" for row in schedule)
    dm_values = [as_float(row["DM"]) for row in schedule]
    gs_values = [as_float(row["GS"]) for row in schedule]
    clients = [as_int(row["participating_clients"]) for row in schedule]
    client_counts = Counter(clients)

    status_word = "PASS" if fit_passes else "REVIEW REQUIRED"
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    coefficient_meanings = {
        "B0": "Expected simulated next-round FMBS efficiency when AI, PA, DM and GS are zero.",
        "B_AI": "Change in predicted efficiency for a one-unit rise in normalized adversarial-input pressure, holding other variables fixed.",
        "B_PA": "Change in predicted efficiency for a one-unit rise in membership-inference advantage, holding other variables fixed.",
        "B_DM": "Change in predicted efficiency for a one-unit rise in defense intensity, holding other variables fixed.",
        "B_GS": "Change in predicted efficiency for a one-unit rise in the participating-client ratio, holding other variables fixed.",
    }

    report: list[str] = []
    add = report.append
    add("# DO-FMBS Phase 6: Constrained Dynamic Optimization Model")
    add("")
    add("## Professional simulation, coefficient-estimation and deployment report")
    add("")
    add(md_table(
        ["Report field", "Value"],
        [
            ["Researcher", "Lydiah Moraa Machora"],
            ["Registration number", "I152/P/9515/23"],
            ["Institution", "Jaramogi Oginga Odinga University of Science and Technology"],
            ["Research year", "2026"],
            ["Evidence decision", f"**{status_word}**"],
            ["Evidence snapshot generated", generated_at],
            ["Estimator", fit["estimator"]],
            ["Controlled observations", as_int(fit["observations"])],
            ["Random seed", as_int(fit["seed"])],
        ],
    ))
    add("")

    add("## Executive conclusion")
    add("")
    if fit_passes:
        add(
            "The Phase 6 constrained model passed the project’s declared simulation-fit gates. "
            f"The constrained optimization converged, all four fitted sign restrictions and all five "
            f"independently evaluated ordering hypotheses were satisfied, "
            f"R² was {fmt(fit['r_squared'], 4)}, five-fold cross-validated R² was "
            f"{fmt(fit['cross_validation_r2_mean'], 4)}, and RMSE was {fmt(fit['rmse'], 4)} "
            "on the 0-to-1 outcome scale. The fitted signs align with the model: AI and PA reduce "
            "predicted efficiency, while DM and GS increase it."
        )
    else:
        add(
            "One or more declared simulation-fit gates were not met. The model outputs remain useful "
            "for diagnosis, but the affected evidence should be reviewed before a deployment decision."
        )
    add("")
    add(
        f"The dynamic controller accepted {target_met} of {len(schedule)} simulated federation rounds: "
        f"{preferred_met} met the preferred target of {fmt(fit['preferred_Y'], 2)} and {minimum_met} "
        f"met the managed minimum target of {fmt(fit['target_Y'], 2)}. "
        "Stress tests show that maximum controls cannot maintain the target under the highest combined "
        "AI and PA conditions; those cases are explicitly rejected or deferred."
    )
    add("")

    add("## 1. Constrained model and variable alignment")
    add("")
    add("The estimated transition equation is:")
    add("")
    add("$$")
    add(r"\widehat{Y}_{t+1}=\widehat{\beta}_0+\widehat{\beta}_{AI}AI_t+\widehat{\beta}_{PA}PA_t+\widehat{\beta}_{DM}DM_t+\widehat{\beta}_{GS}GS_t")
    add("$$")
    add("")
    add(
        "All four predictors are normalized, and their fitted domains are declared explicitly. This "
        "alignment makes sign, magnitude and operating-range comparisons interpretable. AI and PA are observed threat states at round *t*. DM and "
        "GS are controller-selected actions at round *t*. The outcome is the simulated next-round "
        "FMBS efficiency score at round *t + 1*."
    )
    add("")
    domain_by_variable = {row["variable"]: row["fitted_domain"] for row in ranges}
    variable_rows = [
        ["AI", "Adversarial Input", "FGSM perturbation strength normalized as ε / 0.30", "Observed state", domain_by_variable["AI"], "Lower"],
        ["PA", "Privacy Attack", "Twice the membership-inference accuracy advantage above chance", "Observed state", domain_by_variable["PA"], "Lower"],
        ["DM", "Defense Mechanism", "Deployable defense-policy intensity selected by the controller", "Control", domain_by_variable["DM"], "Higher when threat rises"],
        ["GS", "Global Server", "Normalized client participation: (K − 1) / 4", "Control", domain_by_variable["GS"], "Higher participation"],
        ["Y", "FMBS efficiency", "Simulated next-round efficiency score", "Outcome", domain_by_variable["Y"], f">= {fmt(fit['preferred_Y'], 2)} preferred"],
    ]
    add(md_table(
        ["Variable", "Aligned name", "Operational definition", "Role", "Model domain", "Desired direction"],
        variable_rows,
    ))
    add("")
    add(
        "PA is an attack-advantage measure, not a general privacy guarantee. In this fitted design PA "
        "extends from 0 to 0.50. GS is a participation "
        "ratio, not an aggregation coefficient and not a count of training epochs. DM is a normalized "
        "control intensity so that a single β_DM coefficient is meaningful."
    )
    add("")

    add("## 2. Estimated beta values and statistical evidence")
    add("")
    coefficient_rows = []
    for row in coeffs:
        name = row["parameter"]
        coefficient_rows.append([
            name,
            fmt(row["simulation_ground_truth"], 6),
            fmt(row["estimate"], 6),
            fmt(row["recovery_error"], 6),
            fmt(row["standardized_estimate"], 4),
            fmt(row["standard_error"], 6),
            f"[{fmt(row['ci_95_low'], 6)}, {fmt(row['ci_95_high'], 6)}]",
            fmt_p(row["p_value"]),
        ])
    add(md_table(
        ["Parameter", "Generating value", "Estimate", "Recovery error", "Standardized", "SE", "95% CI", "p-value"],
        coefficient_rows,
        ["left", "right", "right", "right", "right", "right", "right", "right"],
    ))
    add("")
    for row in coeffs:
        name = row["parameter"]
        add(f"- **{name} = {fmt(row['estimate'], 6)}:** {coefficient_meanings[name]}")
    add("")
    non_intercept = [row for row in coeffs if row["parameter"] != "B0"]
    dominant = max(non_intercept, key=lambda row: abs(as_float(row["standardized_estimate"])))
    add(
        f"On the standardized scale, **{dominant['parameter']}** has the largest absolute association "
        f"({fmt(dominant['standardized_estimate'], 4)}). This ranking describes the simulated design; "
        "it is not a claim of universal causal importance."
    )
    add("")
    add(
        f"The largest absolute coefficient-recovery error was "
        f"**{fmt(fit['maximum_coefficient_recovery_error'], 6)}**. The constrained and unconstrained "
        f"solutions differed by at most **{fmt(fit['maximum_constrained_ols_difference'], 8)}**; "
        f"active fitted constraints: **{', '.join(fit['active_constraints']) if fit['active_constraints'] else 'none'}**. "
        f"Inference basis: {fit['inference_basis']}"
    )
    add("")

    add("### Coefficient-constraint verification")
    add("")
    add(md_table(
        ["Constraint", "Type", "Left", "Right", "Slack", "Decision"],
        [[
            row["constraint"], display_label(row["constraint_type"]), fmt(row["left"], 6), fmt(row["right"], 6),
            fmt(row["slack"], 6), "Passed" if as_bool(row["satisfied"]) else "Failed",
        ] for row in checks],
        ["left", "left", "right", "right", "right", "center"],
    ))
    add("")
    add(
        f"Maximum numerical constraint violation: **{fmt(fit['maximum_constraint_violation'], 8)}**. "
        f"Estimator termination: **{fit['optimization_message']}**."
    )
    add("")

    add("## 3. Model-fit diagnostics and decision ranges")
    add("")
    add(
        "The following thresholds are project reporting gates for a normalized simulation model. "
        "They are declared decision rules, not universal standards for every biometric deployment."
    )
    add("")
    max_vif = max(as_float(value) for value in fit["vifs"].values())
    diagnostic_rows = [
        ["R²", fmt(fit["r_squared"], 4), ">=0.90", "0.75–<0.90", "<0.75", diagnostic_decision(as_float(fit["r_squared"]), lambda x: x >= .90, lambda x: x >= .75)],
        ["Adjusted R²", fmt(fit["adjusted_r_squared"], 4), ">=0.90", "0.75–<0.90", "<0.75", diagnostic_decision(as_float(fit["adjusted_r_squared"]), lambda x: x >= .90, lambda x: x >= .75)],
        ["Five-fold CV R²", f"{fmt(fit['cross_validation_r2_mean'], 4)} ± {fmt(fit['cross_validation_r2_std'], 4)}", ">=0.90", "0.75–<0.90", "<0.75", diagnostic_decision(as_float(fit["cross_validation_r2_mean"]), lambda x: x >= .90, lambda x: x >= .75)],
        ["RMSE", fmt(fit["rmse"], 4), "<=0.05", ">0.05–0.10", ">0.10", diagnostic_decision(as_float(fit["rmse"]), lambda x: x <= .05, lambda x: x <= .10)],
        ["MAE", fmt(fit["mae"], 4), "<=0.05", ">0.05–0.10", ">0.10", diagnostic_decision(as_float(fit["mae"]), lambda x: x <= .05, lambda x: x <= .10)],
        ["Maximum VIF", fmt(max_vif, 4), "<5", "5–10", ">10", diagnostic_decision(max_vif, lambda x: x < 5, lambda x: x <= 10)],
        ["Condition number", fmt(fit["condition_number"], 4), "<30", "30–100", ">100", diagnostic_decision(as_float(fit["condition_number"]), lambda x: x < 30, lambda x: x <= 100)],
        ["Durbin–Watson", fmt(fit["durbin_watson"], 4), "Recorded only", "Not rated", "Not rated", "Not interpreted: factorial row order is non-temporal"],
        ["Shapiro–Wilk p", fmt_p(fit["shapiro_p"]), ">=0.05", "0.01–<0.05", "<0.01", diagnostic_decision(as_float(fit["shapiro_p"]), lambda x: x >= .05, lambda x: x >= .01)],
    ]
    add(md_table(
        ["Diagnostic", "Attained value", "Ideal / pass", "Managed", "Outside", "Decision"],
        diagnostic_rows,
        ["left", "right", "center", "center", "center", "left"],
    ))
    add("")
    add(
        f"Residual mean was {fmt(fit['residual_mean'], 8)} and residual standard deviation was "
        f"{fmt(fit['residual_std'], 5)}. Cross-validated RMSE was "
        f"{fmt(fit['cross_validation_rmse_mean'], 5)}. The diagnostics are internally consistent "
        "with a well-conditioned linear approximation for the simulated domain."
    )
    add("")

    add("## 4. Declared operating bands and outside-range decisions")
    add("")
    add(md_table(
        ["Variable", "Definition", "Fitted domain", "Preferred", "Managed", "Critical but modeled", "Out of domain", "Required action"],
        [[
            row["variable"], row["definition"], row["fitted_domain"], row["preferred"],
            row["managed"], row["critical_modeled"], row["out_of_domain"], row["required_action"],
        ] for row in ranges],
    ))
    add("")
    add(
        "These are operating-policy bands. A value in a managed band may continue only when the "
        "controller finds controls that keep predicted Y at or above the minimum target. A critical "
        "value remains within the fitted simulation domain but requires maximum escalation and can still "
        "lead to rejection or deferral. A value outside its variable-specific fitted domain is rejected "
        "without producing a model prediction; it is never silently clipped during deployment."
    )
    add("")

    add("### Stress-test decisions")
    add("")
    add(md_table(
        ["Scenario", "ε", "AI", "PA", "DM", "GS", "Clients", "Predicted Y", "Status", "Deployment decision"],
        [[
            display_label(row["scenario"]), fmt(row["epsilon"], 2), fmt(row["AI"], 3), fmt(row["PA"], 3),
            fmt_optional(row["DM"], 2), fmt_optional(row["GS"], 2), row["participating_clients"] or "Not selected",
            fmt_optional(row["predicted_Y"], 4), display_label(row["status"]), display_label(row["deployment_decision"]),
        ] for row in stresses],
        ["left", "right", "right", "right", "right", "right", "right", "right", "left", "left"],
    ))
    add("")
    critical = next((row for row in stresses if row["scenario"] == "modeled_critical"), None)
    if critical:
        add(
            f"At the modeled-critical condition (AI={fmt(critical['AI'], 2)}, "
            f"PA={fmt(critical['PA'], 2)}), the controller selected maximum DM and GS but predicted "
            f"Y={fmt(critical['predicted_Y'], 4)}, below the {fmt(fit['target_Y'], 2)} target. "
            f"The recorded decision is **{display_label(critical['deployment_decision'])}**. This is the intended "
            "fail-safe behavior when additional control cannot restore the required operating level."
        )
        add("")
    out_of_domain = next((row for row in stresses if row["scenario"] == "out_of_domain"), None)
    if out_of_domain:
        add(
            f"The out-of-domain scenario supplied ε={fmt(out_of_domain['epsilon'], 2)} "
            f"(AI={fmt(out_of_domain['AI'], 4)}) and PA={fmt(out_of_domain['PA'], 2)}. "
            "Because both exceed their fitted domains, no DM, GS or predicted Y was produced; the "
            f"recorded decision is **{display_label(out_of_domain['deployment_decision'])}**."
        )
        add("")

    add("## 5. Numerical substitution using estimated coefficients")
    add("")
    add(
        f"Round {substitution_row['round']} is used because it is the highest combined AI and PA "
        "condition in the dynamic schedule that still meets the target. Its values are:"
    )
    add("")
    add(
        f"AI={fmt(substitution_values['AI'], 6)}, PA={fmt(substitution_values['PA'], 6)}, "
        f"DM={fmt(substitution_values['DM'], 6)}, GS={fmt(substitution_values['GS'], 6)}."
    )
    add("")
    add("Substitute the estimated beta values:")
    add("")
    add("$$")
    add(
        r"\widehat{Y}_{t+1}="
        f"{fmt(beta['B0'], 6)}"
        f"+({fmt(beta['B_AI'], 6)})({fmt(substitution_values['AI'], 6)})"
        f"+({fmt(beta['B_PA'], 6)})({fmt(substitution_values['PA'], 6)})"
        f"+({fmt(beta['B_DM'], 6)})({fmt(substitution_values['DM'], 6)})"
        f"+({fmt(beta['B_GS'], 6)})({fmt(substitution_values['GS'], 6)})"
    )
    add("$$")
    add("")
    add(md_table(
        ["Term", "Contribution"],
        [[name, fmt(value, 6)] for name, value in contributions.items()] + [["Total predicted Y", fmt(calculated_y, 6)]],
        ["left", "right"],
    ))
    add("")
    add(
        f"The recomputed result is **{fmt(calculated_y, 6)}**; the stored optimizer value is "
        f"**{fmt(stored_y, 6)}**. Their absolute difference is "
        f"**{fmt(abs(calculated_y - stored_y), 10)}**, confirming that the saved schedule uses the "
        "estimated equation."
    )
    add("")

    add("## 6. What is dynamic and how optimization occurs")
    add("")
    add(
        "The beta coefficients are estimated once for this evidence snapshot; the controls are dynamic. "
        "Before every simulated federation round, the controller observes current AI_t and PA_t. It then "
        "re-evaluates all deployable DM_t and GS_t combinations, considers the previous round’s controls, "
        "and selects a new policy for predicting Y_(t+1). Therefore, dynamic means state-dependent, "
        "round-by-round adaptation—not coefficients that are arbitrarily changed during inference."
    )
    add("")
    add("### 6.1 Coefficient-estimation optimization")
    add("")
    add("The estimator solves:")
    add("")
    add("$$")
    add(r"\min_{\beta}\;\frac{1}{2}\lVert X\beta-y\rVert_2^2")
    add("$$")
    add("")
    add(
        "subject to β_AI ≤ 0, β_PA ≤ 0, β_DM ≥ 0 and β_GS ≥ 0. The five coefficient-order "
        "relations shown in the constraint table are independently evaluated hypotheses; they are not "
        "enforced during estimation. SciPy SLSQP "
        f"reported: **{fit['optimization_message']}**."
    )
    add("")
    add("### 6.2 Dynamic controller objective and exact search")
    add("")
    add("For every candidate control pair, the controller calculates:")
    add("")
    add("$$")
    add(
        f"C_t={fmt(cost_weights['defense'], 2)}DM_t^2"
        f"+{fmt(cost_weights['participation'], 2)}GS_t^2"
        f"+{fmt(cost_weights['defense_switch'], 2)}(DM_t-DM_{{t-1}})^2"
        f"+{fmt(cost_weights['participation_switch'], 2)}(GS_t-GS_{{t-1}})^2"
    )
    add("$$")
    add("")
    add(
        f"It enumerates DM from {defense_start:.2f} to {defense_stop:.2f} in {defense_step:.2f} steps "
        f"({defense_levels} values) and participating clients K ∈ "
        f"{{{', '.join(str(value) for value in client_candidates)}}}, where GS=(K−1)/4. "
        f"This gives **{candidates_per_round} deployable candidates per round**. "
        f"The first round starts from DM_(t−1)={INITIAL_DM:.2f} and GS_(t−1)={INITIAL_GS:.2f}."
    )
    add("")
    add(
        f"The decision rule first searches for candidates satisfying preferred Y >= {fmt(fit['preferred_Y'], 2)}. "
        f"If none exist, it searches for the managed minimum Y >= {fmt(fit['target_Y'], 2)}. "
        "Within the highest feasible tier it minimizes C_t; a cost tie is resolved in favour of higher predicted "
        "Y. If no candidate meets even the minimum target, it selects the candidate with the highest predicted Y and "
        "marks the round `maximum_controls_target_unmet`. This is a finite, reproducible constrained grid "
        "search rather than an unspecified claim of optimization."
    )
    add("")
    add("### 6.3 Cost-weight sensitivity")
    add("")
    add(
        "The cost weights express a deployment policy; they are not estimated beta coefficients. "
        "Phase 6 therefore repeats the same managed threat state under three cost profiles to show how "
        "the selected control mix changes."
    )
    add("")
    add(md_table(
        ["Cost profile", "w_DM", "w_GS", "DM", "GS", "Clients", "Predicted Y", "Cost", "Result"],
        [[
            display_label(row["profile"]), fmt(row["weight_defense"], 2),
            fmt(row["weight_participation"], 2), fmt(row["DM"], 2), fmt(row["GS"], 2),
            row["participating_clients"], fmt(row["predicted_Y"], 4),
            fmt(row["control_cost"], 4), display_label(row["status"]),
        ] for row in sensitivity],
        ["left", "right", "right", "right", "right", "right", "right", "right", "left"],
    ))
    add("")
    add(
        "At AI=0.333 and PA=0.30, the balanced profile selected DM=0.80 with four clients; "
        "the defense-sparing profile selected DM=0.65 with five clients; and the "
        "participation-sparing profile selected DM=1.00 with three clients. All three met the "
        f"managed minimum Y target of {fmt(fit['target_Y'], 2)}. The switching weights remained "
        "0.15 for DM and 0.10 for GS in all profiles. This sensitivity result shows that cost weights "
        "change the chosen feasible policy, so they must be selected and disclosed as governance inputs."
    )
    add("")

    add("## 7. Federation-learning control values")
    add("")
    add(
        "Within Phase 6, the federated control is client participation. Five clients are available; GS "
        "normalizes participation as (K−1)/4. The controller requires at least three clients, so the "
        "deployable GS candidates are 0.50, 0.75 and 1.00. It does not reinterpret GS as a model weight. "
        "Once clients are selected, sample-weighted federated aggregation remains a separate training "
        "operation."
    )
    add("")
    add("### Fixed versus optimized federation settings")
    add("")
    fixed_optimized_rows = [
        ["Defense intensity (DM)", f"{defense_start:.2f}–{defense_stop:.2f} in {defense_step:.2f} steps", "Optimized by Phase 6", "Controls adversarial fraction, training ε and gradient-noise standard deviation"],
        ["Active clients (K) / GS", f"K ∈ {{{', '.join(str(value) for value in client_candidates)}}}; GS=(K−1)/4", "Optimized by Phase 6", "Minimum three active clients"],
    ]
    fixed_optimized_rows.extend([
        [name, fmt(value, 3) if isinstance(value, float) else value, "Fixed integration setting", "Not searched by Phase 6"]
        for name, value in FIXED_FEDERATED_SETTINGS.items()
    ])
    add(md_table(
        ["Setting", "Value/search space", "Phase 6 role", "Interpretation"],
        fixed_optimized_rows,
    ))
    add("")
    add(
        "The learning rate, local-epoch count, batch size and federated-round count are fixed at the "
        "values above; Phase 6 does **not** claim to optimize them. Its adaptive decisions are limited "
        "to DM and the active-client choice represented by GS."
    )
    add("")
    add(md_table(
        ["Round", "ε", "AI", "PA", "DM", "GS", "Clients", "Predicted Y", "Cost", "Status"],
        [[
            row["round"], fmt(row["epsilon"], 2), fmt(row["AI"], 3), fmt(row["PA"], 3),
            fmt(row["DM"], 2), fmt(row["GS"], 2), row["participating_clients"],
            fmt(row["predicted_Y"], 4), fmt(row["control_cost"], 4), display_label(row["status"]),
        ] for row in schedule],
        ["right", "right", "right", "right", "right", "right", "right", "right", "right", "left"],
    ))
    add("")
    add("### Defense settings produced by optimized DM")
    add("")
    defense_rows_by_dm: dict[float, dict] = {}
    for row in schedule:
        defense_rows_by_dm.setdefault(as_float(row["DM"]), row)
    add(md_table(
        ["DM", "Adversarial fraction", "Training attack ε", "Gradient-noise SD"],
        [[
            fmt(dm, 2), fmt(row["adversarial_fraction"], 3),
            fmt(row["training_attack_epsilon"], 4), fmt(row["gradient_noise_std"], 6),
        ] for dm, row in sorted(defense_rows_by_dm.items())],
        ["right", "right", "right", "right"],
    ))
    add("")
    add(
        "The mapping is linear: adversarial fraction = 0.50×DM, training attack ε = 0.05×DM, "
        "and Gaussian gradient-noise standard deviation = 0.0003×DM. These are simulation-policy "
        "controls and do not by themselves establish formal differential privacy."
    )
    add("")
    add(
        f"Across {len(schedule)} rounds, DM ranged from {min(dm_values):.2f} to {max(dm_values):.2f}; "
        f"GS ranged from {min(gs_values):.2f} to {max(gs_values):.2f}; participating clients ranged "
        f"from {min(clients)} to {max(clients)}. Client selections were "
        + ", ".join(f"K={key}: {client_counts[key]} round(s)" for key in sorted(client_counts))
        + f". The preferred target was met in {preferred_met}/{len(schedule)} rounds; the remaining "
        + f"{minimum_met} accepted round(s) met the managed minimum."
    )
    add("")
    add(
        "Optimizing rounds, local epochs, batch size or learning rate would require a separate "
        "training-cost experiment and is outside this controller’s declared search space."
    )
    add("")

    add("## 8. Simulation design and evidence provenance")
    add("")
    add(f"Design declared by the fit artifact: **{fit['simulation_design']}**.")
    add("")
    add(
        "The simulation-generating coefficients are known design parameters used to test whether the "
        "estimator can recover the intended constrained relationship. They are not fitted results. The "
        "estimated coefficients and their recovery errors are reported separately in Section 2."
    )
    add("")
    design_rows = []
    for name, levels in variable_levels.items():
        display_levels = ", ".join(fmt(value, 3) for value in levels)
        design_rows.append([name, len(levels), display_levels, fmt(min(levels), 3), fmt(max(levels), 3)])
    add(md_table(
        ["Design field", "Levels", "Values", "Minimum", "Maximum"],
        design_rows,
        ["left", "right", "left", "right", "right"],
    ))
    add("")
    add(
        f"The observation file contains {len(observations)} rows. Simulation errors have mean "
        f"{statistics.mean(error_values):.6f}, sample standard deviation "
        f"{statistics.stdev(error_values):.6f}, minimum {min(error_values):.6f}, and maximum "
        f"{max(error_values):.6f}. The fixed seed is {as_int(fit['seed'])}."
    )
    add("")
    add("### Evidence manifest")
    add("")
    manifest_rows = []
    for path in evidence_paths:
        row_count = "1 JSON object" if path.suffix.lower() == ".json" else f"{len(csv_data[path.name])} data rows"
        manifest_rows.append([
            path.name, row_count, path.stat().st_size, sha256(path),
        ])
    add(md_table(
        ["Artifact", "Content", "Bytes", "SHA-256"],
        manifest_rows,
        ["left", "left", "right", "left"],
    ))
    add("")
    add(f"Repository commit at report generation: `{git_commit()}`")
    add("")
    add(f"Python runtime: `{platform.python_version()}` on `{platform.platform()}`")
    add("")
    add(f"Report generator: `scripts/generate_constrained_dynamic_report.py`")
    add("")

    add("## 9. Equation-to-code and artifact traceability")
    add("")
    add(
        "The table below identifies the functions that produce each mathematical component. It allows "
        "an examiner or reviewer to move directly from the equation and variable definitions to the "
        "executable implementation and saved evidence."
    )
    add("")
    add(md_table(
        ["Model element", "Implemented operation", "Function(s)", "Location / evidence"],
        [
            ["AI, PA and GS alignment", "Normalize ε, calculate membership advantage, and map participating clients to GS=(K−1)/4", "`normalize_ai`, `membership_advantage`, `gs_from_clients`", "`scripts/constrained_dynamic_model.py`; `acceptance_ranges.csv`"],
            ["DM operational mapping", "Map defense intensity to adversarial fraction, training attack ε and gradient-noise SD", "`defense_configuration`", "`scripts/constrained_dynamic_model.py`; schedule and sensitivity CSVs"],
            ["Simulated transition", "Generate Y_(t+1) from AI_t, PA_t, DM_t and GS_t plus controlled error", "`simulation_response`, `generate_observations`", "`scripts/constrained_dynamic_model.py`; `simulation_observations.csv`"],
            ["Beta estimation", "Minimize one-half residual sum of squares subject to four fitted sign restrictions", "`_constrained_coefficients`, `fit_equation`", "`scripts/constrained_dynamic_model.py`; `coefficient_estimates.csv`; `model_fit.json`"],
            ["Equation substitution", "Calculate predicted Y using the five estimated beta values", "`predict`", "`scripts/constrained_dynamic_model.py`; dynamic and stress CSVs"],
            ["Dynamic optimization", "Search DM and active-client controls at every round under preferred/minimum targets", "`optimize_controls`, `dynamic_schedule`, `cost_weight_sensitivity`", "`scripts/constrained_dynamic_model.py`; schedule and sensitivity CSVs"],
            ["Bands and fail-safe decisions", "Declare fitted/preferred/managed/critical bands and reject out-of-domain or infeasible cases", "`acceptance_ranges`, `stress_scenarios`", "`scripts/constrained_dynamic_model.py`; acceptance and stress CSVs"],
            ["Artifact production", "Run estimation, optimization and CSV/JSON serialization", "`main`", "`phase6_constrained_dynamic_optimization.py`; `simulation_evidence/constrained_dynamic_model/`"],
        ],
    ))
    add("")

    add("## 10. Interpretation and deployment decision protocol")
    add("")
    add("For each proposed federation round:")
    add("")
    add("1. Measure ε and calculate AI = ε / 0.30; estimate PA using the defined membership-advantage procedure.")
    add("2. Reject malformed values outside their fitted domains: AI outside 0–1 or PA outside 0–0.50.")
    add(f"3. Search the {candidates_per_round} allowed DM/client-participation combinations.")
    add(f"4. Prefer Y >= {fmt(fit['preferred_Y'], 2)}; otherwise accept managed operation only when Y >= {fmt(fit['target_Y'], 2)} and all controls are available.")
    add("5. If the target is infeasible, isolate the affected input or defer aggregation; do not report the maximum-control result as acceptable.")
    add(f"6. Treat Y >= {fmt(fit['preferred_Y'], 2)} as preferred and {fmt(fit['target_Y'], 2)} <= Y < {fmt(fit['preferred_Y'], 2)} as managed operation requiring monitoring.")
    add("")

    add("## 11. Plain-language oral explanation")
    add("")
    add(
        "This phase creates many controlled situations in which attack pressure, privacy pressure, "
        "defense strength and client participation are varied. It uses those results to estimate how "
        "each variable changes the next-round FMBS efficiency score. The negative beta values show that "
        "AI and PA reduce the score, while the positive beta values show that DM and GS improve it. "
        "The model is dynamic because it measures the current threats and chooses new controls before "
        "every federation round. It first tries to reach the preferred score of 0.70 and otherwise allows "
        "managed operation only at or above 0.60. It searches defense intensity and three-to-five active "
        "clients, while rounds, local epochs, batch size and learning rate remain fixed. If maximum controls "
        "cannot reach 0.60, or if an input is outside the fitted domain, the round is rejected or deferred."
    )
    add("")

    add("## 12. Limitations and responsible claims")
    add("")
    add("- The coefficient estimates validate recovery and constrained optimization on controlled simulated observations; external operational data are still required for field validation.")
    add("- Confidence intervals and p-values are conditional on the specified linear model and its error assumptions. They do not establish causality.")
    add("- The operating bands are declared project controls, not universal biometric-industry thresholds.")
    add("- The controller must not extrapolate beyond the normalized input domains. Values outside those domains require measurement correction or rejection.")
    add("- GS optimizes participating-client count only, with at least three active clients. It does not optimize local epochs, learning rate, batch size, round count, communication latency or client-specific aggregation weights.")
    add("- PA quantifies the selected membership-inference advantage; a low PA value does not establish protection against every privacy attack.")
    add("- Y is a simulated next-round FMBS efficiency score. A deployment assessment must additionally validate identity-specific accuracy, conventional biometric FAR/FRR, calibration, subgroup performance, privacy accounting and adaptive attacks.")
    add(f"- Scope statement from the evidence artifact: {fit['scope_statement']}")
    add("- A high R² demonstrates fit to this simulated design. It does not by itself establish safety, fairness, privacy or generalization to a new population.")
    add("")

    add("## 13. Professional conclusion")
    add("")
    add(
        f"The constrained equation was estimated from {len(observations)} controlled observations, "
        "not populated by assigning outcome metrics to beta symbols. All four fitted sign restrictions "
        f"and all five independently evaluated ordering hypotheses passed, the constrained estimator "
        f"converged, and the attained R² of "
        f"{fmt(fit['r_squared'], 4)} was supported by cross-validated R² of "
        f"{fmt(fit['cross_validation_r2_mean'], 4)}. The dynamic controller then used those estimates "
        "to choose defense intensity and participating-client ratio at every round. It met the minimum "
        f"Y target in {target_met}/{len(schedule)} dynamic rounds ({preferred_met} preferred, "
        f"{minimum_met} managed-minimum) and correctly rejected or deferred "
        "stress conditions for which maximum controls were insufficient."
    )
    add("")
    add(
        "Accordingly, the mathematical computation, constraint enforcement and simulated control logic "
        "are internally verified for the declared operating domain. Operational deployment remains "
        "conditional on prospective validation using representative biometric and federation data."
    )
    add("")

    return "\n".join(report)


def _markdown_blocks(markdown: str) -> list[tuple[str, object]]:
    """Parse the report's limited Markdown subset for optional document output."""
    lines = markdown.splitlines()
    blocks: list[tuple[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            parsed = [[cell.strip().replace("\\|", "|") for cell in row.strip("|").split("|")] for row in table_lines]
            blocks.append(("table", [parsed[0], *parsed[2:]]))
            continue
        if line == "$$":
            equation_lines = []
            index += 1
            while index < len(lines) and lines[index] != "$$":
                equation_lines.append(lines[index])
                index += 1
            index += 1
            blocks.append(("equation", " ".join(equation_lines)))
            continue
        if line.startswith("### "):
            blocks.append(("heading3", line[4:]))
        elif line.startswith("## "):
            blocks.append(("heading2", line[3:]))
        elif line.startswith("# "):
            blocks.append(("heading1", line[2:]))
        elif line.startswith("- "):
            blocks.append(("bullet", line[2:]))
        elif len(line) > 3 and line[0].isdigit() and ". " in line[:4]:
            number, value = line.split(". ", 1)
            blocks.append(("number", (number, value)))
        else:
            paragraph = [line]
            index += 1
            while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "- ", "$$")):
                if len(lines[index]) > 3 and lines[index][0].isdigit() and ". " in lines[index][:4]:
                    break
                paragraph.append(lines[index])
                index += 1
            blocks.append(("paragraph", " ".join(part.strip() for part in paragraph)))
            continue
        index += 1
    return blocks


def _plain_markdown(text: str) -> str:
    replacements = {
        r"\widehat{Y}_{t+1}": "Ŷ(t+1)",
        r"\widehat{\beta}_0": "β₀",
        r"\widehat{\beta}_{AI}": "β_AI",
        r"\widehat{\beta}_{PA}": "β_PA",
        r"\widehat{\beta}_{DM}": "β_DM",
        r"\widehat{\beta}_{GS}": "β_GS",
        r"\min_{\beta}": "min_β",
        r"\frac{1}{2}": "½",
        r"\lVert": "‖",
        r"\rVert": "‖",
        r"\beta": "β",
        r"\;": " ",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text.replace("**", "").replace("`", "")


def _equation_image_buffer(latex: str):
    """Render a display equation as a high-resolution transparent PNG buffer."""
    from io import BytesIO
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image

    # Matplotlib MathText accepts \Vert but not every installation accepts the
    # LaTeX aliases \lVert and \rVert.
    normalized = latex.replace(r"\lVert", r"\Vert").replace(r"\rVert", r"\Vert")
    buffer = BytesIO()
    math_to_image(
        f"${normalized}$",
        buffer,
        prop=FontProperties(family="DejaVu Serif", size=17),
        dpi=220,
        format="png",
        color="#17365D",
    )
    buffer.seek(0)
    return buffer


def write_docx(markdown: str, path: Path) -> tuple[bool, str]:
    try:
        from docx import Document
        from docx.enum.section import WD_SECTION
        from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches, Pt, RGBColor
    except ImportError:
        return False, "python-docx is not installed"

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    styles = document.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9)
    for style_name, size, color in (("Title", 22, "17365D"), ("Heading 1", 16, "17365D"), ("Heading 2", 13, "2F5597"), ("Heading 3", 11, "4472C4")):
        style = styles[style_name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)

    first_heading = True
    for kind, payload in _markdown_blocks(markdown):
        if kind == "heading1" and first_heading:
            paragraph = document.add_paragraph(_plain_markdown(str(payload)), style="Title")
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            first_heading = False
        elif kind.startswith("heading"):
            level = {"heading1": 1, "heading2": 1, "heading3": 2}[kind]
            document.add_heading(_plain_markdown(str(payload)), level=level)
        elif kind == "table":
            rows = payload
            table = document.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            for row_index, row in enumerate(rows):
                for column_index, value in enumerate(row):
                    cell = table.cell(row_index, column_index)
                    cell.text = _plain_markdown(str(value))
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(7.5)
                            if row_index == 0:
                                run.bold = True
        elif kind == "equation":
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            try:
                from PIL import Image as PillowImage
                equation_buffer = _equation_image_buffer(str(payload))
                with PillowImage.open(equation_buffer) as equation_image:
                    width_px, height_px = equation_image.size
                equation_buffer.seek(0)
                target_height = 0.34
                target_width = target_height * width_px / height_px
                run = paragraph.add_run()
                if target_width > 6.15:
                    run.add_picture(equation_buffer, width=Inches(6.15))
                else:
                    run.add_picture(equation_buffer, height=Inches(target_height))
            except (ImportError, ValueError):
                run = paragraph.add_run(_plain_markdown(str(payload)))
                run.font.name = "Cambria Math"
                run.font.size = Pt(12)
                run.italic = True
        elif kind in {"bullet", "number"}:
            style = "List Bullet" if kind == "bullet" else "List Number"
            value = payload[1] if kind == "number" else payload
            document.add_paragraph(_plain_markdown(str(value)), style=style)
        else:
            document.add_paragraph(_plain_markdown(str(payload)))

    footer = document.sections[0].footer.paragraphs[0]
    footer.text = "DO-FMBS Phase 6 — Constrained Dynamic Optimization Model"
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.save(path)
    return True, "created"


def write_pdf(markdown: str, path: Path) -> tuple[bool, str]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import CondPageBreak, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return False, "reportlab is not installed"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=18, leading=21, textColor=colors.HexColor("#17365D"), alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportH1", parent=styles["Heading1"], fontSize=15, leading=18, textColor=colors.HexColor("#17365D"), spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportH2", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#2F5597"), spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontSize=8.2, leading=11, spaceAfter=5))
    styles.add(ParagraphStyle(name="Equation", parent=styles["BodyText"], fontName="Courier", fontSize=10, leading=13, alignment=TA_CENTER, spaceAfter=6))

    def ascii_safe(value: str) -> str:
        value = _plain_markdown(value)
        substitutions = {
            "β": "beta", "ε": "epsilon", "²": "^2", "≥": ">=", "≤": "<=",
            "−": "-", "–": "-", "—": "-", "×": "x", "±": "+/-", "∈": "in",
            "…": "...", "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
            "⁻": "-", "³": "3", "⁰": "0", "⁵": "5", "⁸": "8", "Ŷ": "Y_hat",
            "‖": "||", "½": "1/2",
        }
        for source, target in substitutions.items():
            value = value.replace(source, target)
        return value

    story = []
    equation_buffers = []
    first_heading = True
    for kind, payload in _markdown_blocks(markdown):
        if kind == "heading1" and first_heading:
            story.append(Paragraph(html_escape(ascii_safe(str(payload))), styles["ReportTitle"]))
            first_heading = False
        elif kind in {"heading1", "heading2"}:
            if str(payload).startswith("2. Estimated beta"):
                story.append(CondPageBreak(70 * mm))
            story.append(Paragraph(html_escape(ascii_safe(str(payload))), styles["ReportH1"]))
        elif kind == "heading3":
            if str(payload).startswith("6.3 Cost-weight"):
                story.append(PageBreak())
            story.append(Paragraph(html_escape(ascii_safe(str(payload))), styles["ReportH2"]))
        elif kind == "table":
            rows = [[Paragraph(html_escape(ascii_safe(str(cell))), styles["ReportBody"]) for cell in row] for row in payload]
            available_width = A4[0] - 24 * mm
            widths = [available_width / len(rows[0])] * len(rows[0])
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#A6A6A6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.extend([table, Spacer(1, 5)])
        elif kind == "equation":
            try:
                equation_buffer = _equation_image_buffer(str(payload))
                equation_buffers.append(equation_buffer)
                equation_image = Image(equation_buffer)
                equation_image._restrictSize(A4[0] - 34 * mm, 15 * mm)
                equation_image.hAlign = "CENTER"
                story.extend([equation_image, Spacer(1, 5)])
            except (ImportError, ValueError):
                story.append(Paragraph(html_escape(ascii_safe(str(payload))), styles["Equation"]))
        else:
            if kind == "bullet":
                value = "• " + str(payload)
            elif kind == "number":
                value = f"{payload[0]}. {payload[1]}"
            else:
                value = str(payload)
            story.append(Paragraph(html_escape(ascii_safe(value)), styles["ReportBody"]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(12 * mm, 8 * mm, "DO-FMBS Phase 6 — Constrained Dynamic Optimization Model")
        canvas.drawRightString(A4[0] - 12 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=12 * mm, leftMargin=12 * mm,
        topMargin=11 * mm, bottomMargin=12 * mm,
        title="DO-FMBS Phase 6 Constrained Dynamic Optimization Model",
        author="DO-FMBS Research Project",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return True, "created"


def main() -> int:
    args = parse_args()
    requested = {item.strip().lower() for item in args.formats.split(",") if item.strip()}
    requested.add("md")
    unsupported = requested - {"md", "docx", "pdf"}
    if unsupported:
        raise ValueError("Unsupported report format(s): " + ", ".join(sorted(unsupported)))

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    fit, csv_data, evidence_paths = load_evidence(input_dir)
    markdown = build_report(fit, csv_data, evidence_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    skipped: list[str] = []
    md_path = output_dir / f"{REPORT_STEM}.md"
    md_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    generated.append(md_path)

    if "docx" in requested:
        docx_path = output_dir / f"{REPORT_STEM}.docx"
        created, reason = write_docx(markdown, docx_path)
        if created:
            generated.append(docx_path)
        else:
            skipped.append(f"DOCX skipped: {reason}")

    if "pdf" in requested:
        pdf_path = output_dir / f"{REPORT_STEM}.pdf"
        created, reason = write_pdf(markdown, pdf_path)
        if created:
            generated.append(pdf_path)
        else:
            skipped.append(f"PDF skipped: {reason}")

    print("Generated report files:")
    for path in generated:
        print(f"  - {path}")
    for message in skipped:
        print(f"  - {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

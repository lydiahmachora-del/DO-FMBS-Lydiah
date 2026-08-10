"""DO-FMBS constrained dynamic simulation, coefficient estimation and control.

This module creates a reproducible factorial simulation, estimates the displayed
dynamic equation, checks coefficient constraints, and solves a constrained
defense/federation policy at each federated round.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulation_evidence" / "constrained_dynamic_model"
SEED = 42
EPSILON_MAX = 0.30
CLIENTS_TOTAL = 5
MIN_PARTICIPATING_CLIENTS = 3
PA_FIT_MAX = 0.50
TARGET_Y = 0.60
PREFERRED_Y = 0.70
GROUND_TRUTH = {
    "B0": 0.40,
    "B_AI": -0.18,
    "B_PA": -0.22,
    "B_DM": 0.26,
    "B_GS": 0.16,
}
COST_WEIGHTS = {
    "defense": 0.60,
    "participation": 0.40,
    "defense_switch": 0.15,
    "participation_switch": 0.10,
}


@dataclass(frozen=True)
class FitResult:
    names: list[str]
    coefficients: list[float]
    standard_errors: list[float]
    t_statistics: list[float]
    p_values: list[float]
    ci_low: list[float]
    ci_high: list[float]
    standardized_coefficients: list[float]
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    mae: float
    residual_mean: float
    residual_std: float
    shapiro_w: float
    shapiro_p: float
    durbin_watson: float
    cross_validation_r2_mean: float
    cross_validation_r2_std: float
    cross_validation_rmse_mean: float
    vifs: dict[str, float]
    observations: int
    degrees_of_freedom: int
    condition_number: float
    estimator: str
    optimization_success: bool
    optimization_message: str
    residual_sum_of_squares: float
    maximum_constraint_violation: float
    unconstrained_coefficients: list[float]
    maximum_constrained_ols_difference: float
    active_constraints: list[str]
    inference_basis: str


def normalize_ai(epsilon: float) -> float:
    """Map FGSM epsilon to AI without hiding out-of-domain inputs."""
    if not 0.0 <= epsilon <= EPSILON_MAX:
        raise ValueError(f"epsilon must be between 0 and {EPSILON_MAX}; got {epsilon}")
    return float(epsilon / EPSILON_MAX)


def membership_advantage(attack_accuracy: float) -> float:
    """Convert threshold-attack accuracy above chance to a [0,1] pressure."""
    if not 0.0 <= attack_accuracy <= 1.0:
        raise ValueError("membership attack accuracy must be between 0 and 1")
    return float(2.0 * max(0.0, attack_accuracy - 0.5))


def gs_from_clients(participating_clients: int) -> float:
    """Normalize one-to-five-client participation so one client maps to zero."""
    if participating_clients not in range(1, CLIENTS_TOTAL + 1):
        raise ValueError(f"participating_clients must be between 1 and {CLIENTS_TOTAL}")
    return float((participating_clients - 1) / (CLIENTS_TOTAL - 1))


def defense_configuration(dm: float) -> dict:
    """Map the abstract defense control to reproducible training settings.

    DM=1 reproduces the combined Phase 4 policy: half of each combined batch is
    adversarial, FGSM training epsilon is 0.05, and Gaussian gradient-noise
    standard deviation is 0.0003. Lower intensities interpolate those controls.
    This is a simulation policy and does not claim formal differential privacy.
    """
    if not 0.0 <= dm <= 1.0:
        raise ValueError("DM must be between 0 and 1")
    return {
        "adversarial_fraction": 0.50 * dm,
        "training_attack_epsilon": 0.05 * dm,
        "gradient_noise_std": 0.0003 * dm,
    }


def simulation_response(ai: float, pa: float, dm: float, gs: float,
                        noise: float = 0.0) -> float:
    """Noisy constrained transition used to validate coefficient recovery.

    AI and PA are states observed at round t; DM and GS are controls applied at
    round t; this response is simulated FMBS efficiency Y at round t+1. Bounds
    keep that synthetic response in its declared [0,1] domain.
    """
    next_round_performance = (
        GROUND_TRUTH["B0"] + GROUND_TRUTH["B_AI"] * ai
        + GROUND_TRUTH["B_PA"] * pa + GROUND_TRUTH["B_DM"] * dm
        + GROUND_TRUTH["B_GS"] * gs + noise
    )
    return float(np.clip(next_round_performance, 0.0, 1.0))


def generate_observations(seed: int = SEED) -> list[dict]:
    """Generate a balanced, reproducible 700-condition factorial simulation."""
    rng = np.random.default_rng(seed)
    epsilon_levels = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    pa_levels = [0.00, 0.02, 0.10, 0.25, 0.50]
    dm_levels = [0.00, 0.33, 0.67, 1.00]
    gs_levels = [gs_from_clients(clients) for clients in range(1, CLIENTS_TOTAL + 1)]
    rows: list[dict] = []
    observation = 0
    for epsilon in epsilon_levels:
        ai = normalize_ai(epsilon)
        for pa in pa_levels:
            for dm in dm_levels:
                for gs in gs_levels:
                    observation += 1
                    stochastic_error = float(rng.normal(0.0, 0.025))
                    y = simulation_response(ai, pa, dm, gs, stochastic_error)
                    rows.append({
                        "observation": observation,
                        "epsilon": epsilon,
                        "AI": ai,
                        "PA": pa,
                        "DM": dm,
                        "GS": gs,
                        "participating_clients": int(round(gs * (CLIENTS_TOTAL - 1) + 1)),
                        "Y": y,
                        "simulation_error": stochastic_error,
                        "seed": seed,
                    })
    return rows


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - float(np.sum((y - prediction) ** 2)) / denominator


def _vif(values: np.ndarray, names: list[str]) -> dict[str, float]:
    result = {}
    for index, name in enumerate(names):
        target = values[:, index]
        others = np.delete(values, index, axis=1)
        design = np.column_stack([np.ones(len(values)), others])
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        r2 = _r2(target, design @ beta)
        result[name] = float(1.0 / max(1e-12, 1.0 - r2))
    return result


def _constrained_coefficients(design: np.ndarray, y: np.ndarray) -> tuple:
    """Fit sign-constrained coefficients and return solver metadata."""
    ols = np.linalg.lstsq(design, y, rcond=None)[0]
    # Fitted restrictions are limited to directional hypotheses. B_PA <= B_AI
    # remains an independently evaluated presentation hypothesis.
    constraint_matrix = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],   # B_AI <= 0
        [0.0, 0.0, 1.0, 0.0, 0.0],   # B_PA <= 0
        [0.0, 0.0, 0.0, 1.0, 0.0],   # B_DM >= 0
        [0.0, 0.0, 0.0, 0.0, 1.0],   # B_GS >= 0
    ])
    lower = np.array([-np.inf, -np.inf, 0.0, 0.0])
    upper = np.array([0.0, 0.0, np.inf, np.inf])

    def objective(candidate: np.ndarray) -> float:
        error = design @ candidate - y
        return float(0.5 * error @ error)

    solution = optimize.minimize(
        objective,
        ols,
        jac=lambda candidate: design.T @ (design @ candidate - y),
        constraints=[optimize.LinearConstraint(constraint_matrix, lower, upper)],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not solution.success:
        raise RuntimeError(f"Constrained coefficient estimation failed: {solution.message}")
    constrained_values = constraint_matrix @ solution.x
    lower_violation = np.maximum(lower - constrained_values, 0.0)
    upper_violation = np.maximum(constrained_values - upper, 0.0)
    maximum_violation = float(max(lower_violation.max(), upper_violation.max()))
    constraint_names = ["B_AI <= 0", "B_PA <= 0", "B_DM >= 0", "B_GS >= 0"]
    active = [name for name, value in zip(constraint_names, constrained_values)
              if abs(float(value)) <= 1e-8]
    return solution.x, ols, solution, maximum_violation, active


def fit_equation(rows: list[dict]) -> FitResult:
    """Estimate B0...B4 by sign-constrained least squares with diagnostics."""
    predictor_names = ["AI", "PA", "DM", "GS"]
    predictors = np.array([[row[name] for name in predictor_names] for row in rows], dtype=float)
    y = np.array([row["Y"] for row in rows], dtype=float)
    design = np.column_stack([np.ones(len(rows)), predictors])
    beta, ols, solution, maximum_violation, active_constraints = _constrained_coefficients(design, y)
    prediction = design @ beta
    residual = y - prediction
    n, parameters = design.shape
    dof = n - parameters
    sse = float(residual @ residual)
    variance = sse / dof
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    t_values = beta / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_values), dof)
    critical = float(stats.t.ppf(0.975, dof))
    ci_low = beta - critical * standard_errors
    ci_high = beta + critical * standard_errors
    r_squared = _r2(y, prediction)
    adjusted = 1.0 - (1.0 - r_squared) * (n - 1) / dof
    standardized = np.empty(parameters)
    standardized[0] = 0.0
    standardized[1:] = beta[1:] * predictors.std(axis=0, ddof=1) / y.std(ddof=1)

    # Deterministic five-fold cross-validation.
    indices = np.random.default_rng(SEED).permutation(n)
    folds = np.array_split(indices, 5)
    cv_r2, cv_rmse = [], []
    for fold in folds:
        train = np.setdiff1d(indices, fold, assume_unique=True)
        fold_beta, _, _, _, _ = _constrained_coefficients(design[train], y[train])
        fold_prediction = design[fold] @ fold_beta
        cv_r2.append(_r2(y[fold], fold_prediction))
        cv_rmse.append(float(np.sqrt(np.mean((y[fold] - fold_prediction) ** 2))))

    shapiro_w, shapiro_p = stats.shapiro(residual)
    durbin_watson = float(np.sum(np.diff(residual) ** 2) / np.sum(residual ** 2))
    constrained_ols_difference = float(np.max(np.abs(beta - ols)))
    inference_basis = (
        "Classical OLS covariance is reported because the constrained solution "
        "equals the unconstrained OLS solution and no fitted sign constraint is active."
        if not active_constraints and constrained_ols_difference <= 1e-8
        else "Classical OLS inference is descriptive only because a fitted constraint is active."
    )
    return FitResult(
        names=["B0", "B_AI", "B_PA", "B_DM", "B_GS"],
        coefficients=beta.tolist(), standard_errors=standard_errors.tolist(),
        t_statistics=t_values.tolist(), p_values=p_values.tolist(),
        ci_low=ci_low.tolist(), ci_high=ci_high.tolist(),
        standardized_coefficients=standardized.tolist(),
        r_squared=float(r_squared), adjusted_r_squared=float(adjusted),
        rmse=float(np.sqrt(np.mean(residual ** 2))),
        mae=float(np.mean(np.abs(residual))), residual_mean=float(np.mean(residual)),
        residual_std=float(np.std(residual, ddof=1)), shapiro_w=float(shapiro_w),
        shapiro_p=float(shapiro_p), durbin_watson=durbin_watson,
        cross_validation_r2_mean=float(np.mean(cv_r2)),
        cross_validation_r2_std=float(np.std(cv_r2, ddof=1)),
        cross_validation_rmse_mean=float(np.mean(cv_rmse)),
        vifs=_vif(predictors, predictor_names), observations=n,
        degrees_of_freedom=dof, condition_number=float(np.linalg.cond(design)),
        estimator="constrained least squares (SLSQP)",
        optimization_success=bool(solution.success),
        optimization_message=str(solution.message),
        residual_sum_of_squares=sse,
        maximum_constraint_violation=maximum_violation,
        unconstrained_coefficients=ols.tolist(),
        maximum_constrained_ols_difference=constrained_ols_difference,
        active_constraints=active_constraints,
        inference_basis=inference_basis,
    )


def predict(fit: FitResult, ai: float, pa: float, dm: float, gs: float) -> float:
    domains = {"AI": (ai, 0.0, 1.0), "PA": (pa, 0.0, PA_FIT_MAX),
               "DM": (dm, 0.0, 1.0), "GS": (gs, 0.0, 1.0)}
    violations = [name for name, (value, low, high) in domains.items()
                  if not low <= value <= high]
    if violations:
        raise ValueError(f"out-of-domain predictor(s): {', '.join(violations)}")
    b = fit.coefficients
    return float(b[0] + b[1] * ai + b[2] * pa + b[3] * dm + b[4] * gs)


def optimize_controls(fit: FitResult, ai: float, pa: float, previous_dm: float,
                      previous_gs: float, target: float = TARGET_Y,
                      preferred: float = PREFERRED_Y,
                      cost_weights: dict | None = None) -> dict:
    """Choose lowest-cost controls, preferring Y>=0.70 then requiring Y>=0.60."""
    # Validate the threat state before constructing any numerical decision.
    predict(fit, ai, pa, 0.0, gs_from_clients(MIN_PARTICIPATING_CLIENTS))
    weights = COST_WEIGHTS if cost_weights is None else cost_weights
    candidates = []
    for dm in np.linspace(0.0, 1.0, 21):
        for clients in range(MIN_PARTICIPATING_CLIENTS, CLIENTS_TOTAL + 1):
            gs = gs_from_clients(clients)
            predicted = predict(fit, ai, pa, float(dm), gs)
            cost = (
                weights["defense"] * float(dm) ** 2
                + weights["participation"] * gs ** 2
                + weights["defense_switch"] * (float(dm) - previous_dm) ** 2
                + weights["participation_switch"] * (gs - previous_gs) ** 2
            )
            candidates.append((cost, -predicted, float(dm), gs, clients, predicted))
    preferred_candidates = [candidate for candidate in candidates if candidate[-1] >= preferred]
    minimum_candidates = [candidate for candidate in candidates if candidate[-1] >= target]
    if preferred_candidates:
        chosen = min(preferred_candidates, key=lambda value: (value[0], value[1]))
        status = "preferred_target_met"
    elif minimum_candidates:
        chosen = min(minimum_candidates, key=lambda value: (value[0], value[1]))
        status = "minimum_target_met"
    else:
        chosen = min(candidates, key=lambda value: value[1])
        status = "maximum_controls_target_unmet"
    cost, _, dm, gs, clients, predicted = chosen
    configuration = defense_configuration(dm)
    return {
        "DM": dm, **configuration, "GS": gs, "participating_clients": clients,
        "predicted_Y": predicted, "control_cost": cost, "status": status,
    }


def dynamic_schedule(fit: FitResult) -> list[dict]:
    """Run a 12-round dynamic controller under changing threat pressure."""
    epsilon_path = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.10, 0.05, 0.01, 0.00, 0.05, 0.10]
    pa_path = [0.02, 0.02, 0.05, 0.10, 0.20, 0.40, 0.30, 0.15, 0.08, 0.02, 0.10, 0.20]
    previous_dm, previous_gs = 0.30, gs_from_clients(4)
    rows = []
    for round_number, (epsilon, pa) in enumerate(zip(epsilon_path, pa_path), 1):
        ai = normalize_ai(epsilon)
        control = optimize_controls(fit, ai, pa, previous_dm, previous_gs)
        rows.append({"round": round_number, "epsilon": epsilon, "AI": ai,
                     "PA": pa, **control,
                     "deployment_decision": "accept" if control["status"] != "maximum_controls_target_unmet" else "reject_or_defer"})
        previous_dm, previous_gs = control["DM"], control["GS"]
    return rows


def cost_weight_sensitivity(fit: FitResult) -> list[dict]:
    """Re-optimize one managed threat state under three declared cost policies."""
    profiles = {
        "balanced": COST_WEIGHTS,
        "defense_sparing": {
            "defense": 1.00, "participation": 0.20,
            "defense_switch": 0.15, "participation_switch": 0.10,
        },
        "participation_sparing": {
            "defense": 0.30, "participation": 0.80,
            "defense_switch": 0.15, "participation_switch": 0.10,
        },
    }
    epsilon, pa = 0.10, 0.30
    ai = normalize_ai(epsilon)
    rows = []
    for profile, weights in profiles.items():
        control = optimize_controls(
            fit, ai, pa, previous_dm=0.30, previous_gs=gs_from_clients(4),
            cost_weights=weights,
        )
        rows.append({
            "profile": profile, "epsilon": epsilon, "AI": ai, "PA": pa,
            **{f"weight_{name}": value for name, value in weights.items()},
            **control,
        })
    return rows


def stress_scenarios(fit: FitResult) -> list[dict]:
    """Show controller behaviour inside and outside declared operating bands."""
    scenarios = [
        ("preferred_operating", 0.05, 0.10),
        ("managed_operating", 0.10, 0.30),
        ("modeled_critical", 0.20, 0.50),
        ("out_of_domain", 0.35, 0.60),
    ]
    rows = []
    previous_dm, previous_gs = 0.30, gs_from_clients(4)
    for name, epsilon, pa in scenarios:
        ai = epsilon / EPSILON_MAX
        if epsilon > EPSILON_MAX or pa > PA_FIT_MAX:
            rows.append({
                "scenario": name, "epsilon": epsilon, "AI": ai, "PA": pa,
                "DM": None, "adversarial_fraction": None,
                "training_attack_epsilon": None, "gradient_noise_std": None,
                "GS": None, "participating_clients": None, "predicted_Y": None,
                "control_cost": None, "status": "out_of_domain",
                "deployment_decision": "reject_or_defer",
            })
            continue
        control = optimize_controls(fit, ai, pa, previous_dm, previous_gs)
        accepted = control["status"] in {"preferred_target_met", "minimum_target_met"}
        rows.append({
            "scenario": name, "epsilon": epsilon, "AI": ai, "PA": pa,
            **control, "deployment_decision": "accept" if accepted else "reject_or_defer",
        })
        previous_dm, previous_gs = control["DM"], control["GS"]
    return rows


def constraint_checks(fit: FitResult) -> list[dict]:
    b = dict(zip(fit.names, fit.coefficients))
    definitions = [
        ("B_AI < B_DM", "evaluated hypothesis", True, b["B_AI"], b["B_DM"], b["B_DM"] - b["B_AI"]),
        ("B_AI < B_GS", "evaluated hypothesis", True, b["B_AI"], b["B_GS"], b["B_GS"] - b["B_AI"]),
        ("B_PA <= B_AI", "evaluated hypothesis", False, b["B_PA"], b["B_AI"], b["B_AI"] - b["B_PA"]),
        ("B_PA < B_DM", "evaluated hypothesis", True, b["B_PA"], b["B_DM"], b["B_DM"] - b["B_PA"]),
        ("B_PA < B_GS", "evaluated hypothesis", True, b["B_PA"], b["B_GS"], b["B_GS"] - b["B_PA"]),
        ("B_AI <= 0", "fitted sign restriction", False, b["B_AI"], 0.0, -b["B_AI"]),
        ("B_PA <= 0", "fitted sign restriction", False, b["B_PA"], 0.0, -b["B_PA"]),
        ("B_DM >= 0", "fitted sign restriction", False, b["B_DM"], 0.0, b["B_DM"]),
        ("B_GS >= 0", "fitted sign restriction", False, b["B_GS"], 0.0, b["B_GS"]),
    ]
    return [{"constraint": name, "constraint_type": constraint_type,
             "left": left, "right": right,
             "slack": slack,
             "satisfied": bool(slack > 1e-12 if strict else slack >= -1e-12)}
            for name, constraint_type, strict, left, right, slack in definitions]


def acceptance_ranges() -> list[dict]:
    return [
        {"variable": "AI", "definition": "epsilon / 0.30", "fitted_domain": "0.000–1.000 (epsilon 0–0.30)", "preferred": "0.000–0.167", "managed": ">0.167–0.333", "critical_modeled": ">0.333–1.000", "out_of_domain": ">1.000", "required_action": "Increase controls in the critical band; reject without prediction outside the fitted domain."},
        {"variable": "PA", "definition": "2 × max(0, membership-attack accuracy − 0.50)", "fitted_domain": "0.00–0.50", "preferred": "0.00–0.20", "managed": ">0.20–0.40", "critical_modeled": ">0.40–0.50", "out_of_domain": ">0.50", "required_action": "Escalate leakage investigation; reject without prediction outside the fitted domain."},
        {"variable": "DM", "definition": "continuous defense policy intensity", "fitted_domain": "0.00–1.00", "preferred": "controller-selected to meet Y >= 0.70", "managed": "controller-selected to meet Y >= 0.60", "critical_modeled": "maximum DM=1 when target is infeasible", "out_of_domain": "<0 or >1", "required_action": "Apply the mapped defense settings; reject invalid values."},
        {"variable": "GS", "definition": "(participating clients − 1) / 4", "fitted_domain": "0.00–1.00 (1–5 clients)", "preferred": "0.75–1.00 (4–5 clients)", "managed": "0.50 (3 clients)", "critical_modeled": "0.00–0.25 (1–2 clients; not deployable)", "out_of_domain": "<0 or >1", "required_action": "Require at least three clients; otherwise defer aggregation."},
        {"variable": "Y", "definition": "simulated next-round FMBS efficiency score", "fitted_domain": "0.00–1.00", "preferred": ">=0.70", "managed": "0.60–<0.70", "critical_modeled": "<0.60", "out_of_domain": "<0 or >1", "required_action": "Reject or defer a round below 0.60."},
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = generate_observations()
    fit = fit_equation(rows)
    schedule = dynamic_schedule(fit)
    sensitivity = cost_weight_sensitivity(fit)
    stresses = stress_scenarios(fit)
    ranges = acceptance_ranges()
    checks = constraint_checks(fit)
    coefficient_rows = []
    for index, name in enumerate(fit.names):
        coefficient_rows.append({
            "parameter": name, "estimate": fit.coefficients[index],
            "simulation_ground_truth": GROUND_TRUTH[name],
            "recovery_error": fit.coefficients[index] - GROUND_TRUTH[name],
            "unconstrained_ols_estimate": fit.unconstrained_coefficients[index],
            "standardized_estimate": fit.standardized_coefficients[index],
            "standard_error": fit.standard_errors[index],
            "t_statistic": fit.t_statistics[index], "p_value": fit.p_values[index],
            "ci_95_low": fit.ci_low[index], "ci_95_high": fit.ci_high[index],
            "constraint": {"B_AI": "<=0", "B_PA": "<=0", "B_DM": ">=0", "B_GS": ">=0"}.get(name, "unrestricted"),
            "constraint_satisfied": True if name == "B0" else (
                fit.coefficients[index] <= 0 if name in {"B_AI", "B_PA"} else fit.coefficients[index] >= 0),
        })
    write_csv(OUT / "simulation_observations.csv", rows)
    write_csv(OUT / "coefficient_estimates.csv", coefficient_rows)
    write_csv(OUT / "dynamic_optimization_schedule.csv", schedule)
    write_csv(OUT / "cost_weight_sensitivity.csv", sensitivity)
    write_csv(OUT / "stress_test_scenarios.csv", stresses)
    write_csv(OUT / "acceptance_ranges.csv", ranges)
    write_csv(OUT / "constraint_checks.csv", checks)
    payload = asdict(fit)
    payload["equation"] = "Y_hat_(t+1) = B0 + B_AI*AI_t + B_PA*PA_t + B_DM*DM_t + B_GS*GS_t"
    payload["constraints_satisfied"] = all(row["satisfied"] for row in checks)
    payload["target_Y"] = TARGET_Y
    payload["preferred_Y"] = PREFERRED_Y
    payload["seed"] = SEED
    payload["simulation_design"] = "7 AI x 5 PA x 4 DM x 5 GS = 700 balanced observations"
    payload["simulation_ground_truth"] = GROUND_TRUTH
    payload["maximum_coefficient_recovery_error"] = max(
        abs(estimate - GROUND_TRUTH[name])
        for name, estimate in zip(fit.names, fit.coefficients)
    )
    payload["fitted_domains"] = {
        "epsilon": [0.0, EPSILON_MAX], "AI": [0.0, 1.0],
        "PA": [0.0, PA_FIT_MAX], "DM": [0.0, 1.0], "GS": [0.0, 1.0],
        "participating_clients": [1, CLIENTS_TOTAL], "Y": [0.0, 1.0],
    }
    payload["optimization_settings"] = {
        "minimum_participating_clients": MIN_PARTICIPATING_CLIENTS,
        "total_clients": CLIENTS_TOTAL,
        "defense_grid_start": 0.0, "defense_grid_stop": 1.0,
        "defense_grid_step": 0.05,
        "client_candidates": list(range(MIN_PARTICIPATING_CLIENTS, CLIENTS_TOTAL + 1)),
        "cost_weights": COST_WEIGHTS,
        "preferred_target_Y": PREFERRED_Y, "minimum_target_Y": TARGET_Y,
    }
    payload["dynamic_definition"] = (
        "AI_t and PA_t are measured before each round; DM_t and GS_t are re-optimized; "
        "Y_(t+1) is predicted. The controller first seeks Y>=0.70, then Y>=0.60, "
        "requires at least three clients, and rejects infeasible or out-of-domain states."
    )
    payload["scope_statement"] = (
        "This is a reproducible synthetic constrained-coefficient recovery and adaptive-policy "
        "demonstration. Coefficients are simulation estimates, not neural-network weights or "
        "estimates from 700 independent federated training runs."
    )
    (OUT / "model_fit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"coefficients": dict(zip(fit.names, fit.coefficients)),
                      "r_squared": fit.r_squared, "adjusted_r_squared": fit.adjusted_r_squared,
                      "cv_r2": fit.cross_validation_r2_mean,
                      "constraints_satisfied": payload["constraints_satisfied"],
                      "rounds_accepted": sum(row["deployment_decision"] == "accept" for row in schedule),
                      "rounds_preferred": sum(row["status"] == "preferred_target_met" for row in schedule)}, indent=2))


if __name__ == "__main__":
    main()

"""Read-only Phase 6 dashboard backed exclusively by persisted run artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_phase6_authoritative(run_root: Path) -> None:
    st.title("Phase 6 · Dynamic optimization")
    st.caption("Constrained response-model fit and twelve-round defense/participation schedule.")
    runs = sorted((path for path in run_root.glob("phase6_authoritative_*")
                   if (path / "phase6_manifest.json").is_file()),
                  key=lambda path: path.stat().st_mtime, reverse=True)
    if not runs:
        st.info("No saved Phase 6 optimization run is available yet.")
        return
    name = st.selectbox("Optimization run", [path.name for path in runs], key="phase6_authoritative_run")
    run = run_root / name
    config = _json(run / "phase6_config.json")
    fit = _json(run / "model_fit.json")
    metrics = _json(run / "phase6_metrics.json")
    report = _json(run / "phase6_verification_report.json")
    manifest = _json(run / "phase6_manifest.json")
    if not config or not fit or not metrics:
        st.error("The selected Phase 6 run is missing required persisted evidence.")
        return

    rows, bad = [], []
    for artifact in manifest.get("artifacts", []):
        path = run / artifact["path"]
        try:
            good = path.stat().st_size == artifact["bytes"] and _sha(path) == artifact["sha256"]
        except OSError:
            good = False
        rows.append({"Artifact": artifact["path"], "Bytes": artifact["bytes"],
                     "Current SHA-256 check": "PASS" if good else "FAIL"})
        if not good:
            bad.append(artifact["path"])
    if report.get("status") != "PASS" or bad or len(rows) < 12:
        st.error("The saved optimization evidence needs review before interpretation.")
    st.caption(f"Run `{name}` · seed {config.get('seed')} · {fit.get('observations')} controlled observations")

    st.subheader("Fitted dynamic response model")
    st.latex(r"\hat{Y}_{t+1}=\beta_0+\beta_{AI}AI_t+\beta_{PA}PA_t+\beta_{DM}DM_t+\beta_{GS}GS_t")
    coefficients = pd.DataFrame([{"Coefficient": key, "Estimate": estimate,
                                  "95% CI": f"[{fit['ci_low'][i]:.6f}, {fit['ci_high'][i]:.6f}]",
                                  "Standardized estimate": fit["standardized_coefficients"][i],
                                  "p-value": fit["p_values"][i]}
                                 for i, (key, estimate) in enumerate(zip(fit["names"], fit["coefficients"]))])
    st.dataframe(coefficients, width="stretch", hide_index=True)
    values = st.columns(5)
    for column, title, value in zip(values, ("R²", "Adjusted R²", "5-fold CV R²", "RMSE", "MAE"),
                                    (fit["r_squared"], fit["adjusted_r_squared"], fit["cross_validation_r2_mean"], fit["rmse"], fit["mae"])):
        column.metric(title, f"{value:.5f}")
    with st.expander("Fit and residual diagnostics"):
        st.json({"Estimator": fit["estimator"], "solver_success": fit["optimization_success"],
                 "maximum constraint violation": fit["maximum_constraint_violation"],
                 "VIF": fit["vifs"], "Shapiro-Wilk W / p": [fit["shapiro_w"], fit["shapiro_p"]],
                 "Durbin-Watson": fit["durbin_watson"], "CV R² SD": fit["cross_validation_r2_std"],
                 "CV RMSE": fit["cross_validation_rmse_mean"], "inference basis": fit["inference_basis"]}, expanded=True)
    st.caption("The response is model-derived from controlled simulation observations; it is not a measured deployment outcome.")

    st.subheader("Controller search and objective")
    a, b, c = st.columns(3)
    a.metric("Policies per round", "21 × 3 = 63")
    b.metric("Candidate evaluations", str(metrics["schedule_summary"]["total_candidate_evaluations"]))
    c.metric("Rounds", str(metrics["schedule_summary"]["rounds"]))
    st.code(config["cost_formula"], language="text")
    st.caption("The controller minimizes cost subject to the response targets (0.70, then 0.60) and client-count/domain constraints. If no candidate qualifies, the round is deferred.")

    try:
        schedule = pd.read_csv(run / "dynamic_schedule_detailed.csv")
        candidates = pd.read_csv(run / "candidate_policy_search.csv")
    except (OSError, ValueError) as error:
        st.error("Controller schedule data could not be loaded.")
        with st.expander("Technical details"):
            st.code(str(error))
        return
    st.subheader("Prescribed twelve-round policy schedule")
    display = schedule[["round", "epsilon", "AI", "PA", "previous_DM", "previous_GS", "DM", "K", "GS",
                        "predicted_Y", "defense_cost", "participation_cost", "defense_switch_cost",
                        "participation_switch_cost", "total_cost", "status", "deployment_decision"]]
    st.dataframe(display, width="stretch", hide_index=True)
    plot_left, plot_right = st.columns(2)
    plot_left.line_chart(schedule, x="round", y=["AI", "PA"],
                         x_label="Federated round", y_label="Threat state", height=300)
    plot_right.line_chart(schedule, x="round", y=["DM", "GS"],
                          x_label="Federated round", y_label="Control setting", height=300)
    st.line_chart(schedule, x="round", y="predicted_Y", x_label="Federated round",
                  y_label="Predicted response Y", color="#167d83", height=280)
    round_number = st.selectbox("Inspect all candidates in a round", list(range(1, 13)), key="phase6_candidate_round")
    with st.expander(f"Round {round_number}: all 63 candidate policies", expanded=False):
        st.dataframe(candidates[candidates["round"] == round_number], width="stretch", hide_index=True)

    safety = _json(run / "phase6_safety_tests.json")
    with st.expander("Constraint, domain, and safety checks"):
        branch_rows = [{"Scenario": label.replace("_", " ").title(), "Passed": passed}
                       for label, passed in safety.get("branch_results", {}).items()]
        st.dataframe(pd.DataFrame(branch_rows), width="stretch", hide_index=True)
        st.caption("Critical in-domain and out-of-domain cases are evaluated as rejection conditions.")
        st.write("Boundary inputs")
        st.dataframe(pd.DataFrame(safety.get("valid_boundary_inputs", [])), width="stretch", hide_index=True)
        st.write("Rejected invalid inputs")
        st.dataframe(pd.DataFrame(safety.get("invalid_inputs_rejected", [])), width="stretch", hide_index=True)
    with st.expander("Phase 5 context"):
        st.json(_json(run / "phase5_context.json"), expanded=False)
        st.caption("Phase 5 measures are contextual inputs; they are not observations of the simulated response variable.")
    with st.expander("Evidence integrity and verification"):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.json(report, expanded=False)
        st.caption(f"Chapter Four source SHA-256: `{config.get('source_document_sha256')}`")

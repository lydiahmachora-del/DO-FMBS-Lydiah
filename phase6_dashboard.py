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
    st.title("Phase 6 · Dynamic Optimization (Authoritative)")
    st.caption("Final research phase · persisted model-fit and controller evidence only; this page never regenerates the simulation.")
    runs = sorted((path for path in run_root.glob("phase6_authoritative_*")
                   if (path / "phase6_manifest.json").is_file()),
                  key=lambda path: path.stat().st_mtime, reverse=True)
    if not runs:
        st.info("No authoritative Phase 6 run is available yet. Start phase6_authoritative_optimization.py to create a fresh isolated run.")
        return
    name = st.selectbox("Saved Phase 6 run", [path.name for path in runs], key="phase6_authoritative_run")
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
    if report.get("status") == "PASS" and not bad and len(rows) >= 12:
        st.success(f"Independent verification PASS · {report.get('check_count')} checks · manifest PASS · `{name}`")
    else:
        st.error(f"Evidence integrity needs attention · verifier={report.get('status', 'missing')} · invalid artifacts={len(bad)}")
    st.caption(f"Authoritative Chapter Four SHA-256: `{config.get('source_document_sha256')}` · seed {config.get('seed')} · {fit.get('observations')} controlled synthetic observations")

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
    st.info("The 700-row model-derived simulation tests coefficient recovery; predicted Y is not measured real-world efficiency and was not fitted from the Phase 5 biometric metrics.")

    st.subheader("Controller search and objective")
    a, b, c = st.columns(3)
    a.metric("Policies per round", "21 × 3 = 63")
    b.metric("Candidate evaluations", str(metrics["schedule_summary"]["total_candidate_evaluations"]))
    c.metric("Rounds", str(metrics["schedule_summary"]["rounds"]))
    st.code(config["cost_formula"], language="text")
    st.caption("The controller first chooses the minimum-cost candidate with predicted Y ≥ 0.70; if none exists, it chooses the minimum-cost candidate with Y ≥ 0.60. Otherwise it reports the maximum-response diagnostic and rejects/defer; it is never counted as an accepted policy.")

    try:
        schedule = pd.read_csv(run / "dynamic_schedule_detailed.csv")
        candidates = pd.read_csv(run / "candidate_policy_search.csv")
    except (OSError, ValueError) as error:
        st.error(f"Could not load saved controller artifacts: {error}")
        return
    st.subheader("Prescribed twelve-round policy schedule")
    display = schedule[["round", "epsilon", "AI", "PA", "previous_DM", "previous_GS", "DM", "K", "GS",
                        "predicted_Y", "defense_cost", "participation_cost", "defense_switch_cost",
                        "participation_switch_cost", "total_cost", "status", "deployment_decision"]]
    st.dataframe(display, width="stretch", hide_index=True)
    plot = schedule.set_index("round")[["AI", "PA", "DM", "GS", "predicted_Y"]]
    st.line_chart(plot)
    round_number = st.selectbox("Inspect all candidates in a round", list(range(1, 13)), key="phase6_candidate_round")
    with st.expander(f"Round {round_number}: all 63 candidate policies", expanded=False):
        st.dataframe(candidates[candidates["round"] == round_number], width="stretch", hide_index=True)

    st.subheader("Safety and domain tests")
    safety = _json(run / "phase6_safety_tests.json")
    branch_rows = [{"Test": label.replace("_", " ").title(), "Passed": passed}
                   for label, passed in safety.get("branch_results", {}).items()]
    st.dataframe(pd.DataFrame(branch_rows), width="stretch", hide_index=True)
    st.caption("The critical in-domain scenario and the out-of-domain scenario are separate rejection tests, not successful schedule decisions.")
    st.write("Accepted boundary cases")
    st.dataframe(pd.DataFrame(safety.get("valid_boundary_inputs", [])), width="stretch", hide_index=True)
    st.write("Rejected invalid/non-finite inputs")
    st.dataframe(pd.DataFrame(safety.get("invalid_inputs_rejected", [])), width="stretch", hide_index=True)

    st.subheader("Phase 5 evidence linked as context")
    st.json(_json(run / "phase5_context.json"), expanded=False)
    st.caption("Phase 5 recognition, verification and attack metrics are context only. They are not treated as observations of simulated Y or as proof of live deployment behavior.")
    st.subheader("Artifact integrity and independent verifier")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.json(report, expanded=False)
    st.warning("This is a controlled simulation and decision-policy demonstration. It does not guarantee biometric security, formal privacy, or safe operation in a live federation.")

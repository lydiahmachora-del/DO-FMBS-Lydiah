"""Read-only presentation of persisted Phase 5 validation evidence."""
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


def _pct(value: object) -> str:
    return f"{100 * float(value):.2f}%"


def render_phase5_authoritative(run_root: Path) -> None:
    st.title("Phase 5 · Model validation")
    st.caption("Classification, claimed-identity verification, robustness, paired comparisons, and computation measurements.")
    runs = sorted((path for path in run_root.glob("phase5_authoritative_*")
                   if (path / "phase5_manifest.json").is_file()),
                  key=lambda path: path.stat().st_mtime, reverse=True)
    if not runs:
        st.info("No saved Phase 5 validation run is available yet.")
        return
    chosen = st.selectbox("Validation run", [path.name for path in runs], key="phase5_authoritative_run")
    run = run_root / chosen
    metrics = _json(run / "phase5_metrics.json")
    config = _json(run / "phase5_config.json")
    verification = _json(run / "phase5_verification_report.json")
    manifest = _json(run / "phase5_manifest.json")
    if not metrics or not config:
        st.error("The selected run is missing its persisted configuration or metrics.")
        return

    integrity = []
    all_good = bool(manifest.get("artifacts"))
    for item in manifest.get("artifacts", []):
        target = run / item["path"]
        try:
            good = target.stat().st_size == item["bytes"] and _sha(target) == item["sha256"]
        except OSError:
            good = False
        all_good &= good
        integrity.append({"Artifact": item["path"], "Bytes": item["bytes"], "SHA-256": item["sha256"],
                          "Current check": "PASS" if good else "FAIL"})
    st.caption(f"Run `{chosen}` · baseline `{config.get('source_phase2_run')}` · defense `{config.get('source_phase4_run')}`")
    if verification.get("status") != "PASS" or not all_good:
        st.error("The saved validation evidence needs review before interpretation.")

    models = metrics.get("models", {})
    cls_rows = []
    for model_key, display in (("baseline", "Federated baseline"), ("dofmbs_combined", "DO-FMBS combined defense")):
        entry = models.get(model_key, {})
        for modality, key in (("Face", "face_classification"), ("Fingerprint", "fingerprint_classification")):
            score = entry.get(key, {})
            cls_rows.append({"Model": display, "Modality": modality, "Correct / N": f"{score.get('correct', 0)} / {score.get('samples', 0)}",
                             "Accuracy": _pct(score.get("accuracy", 0)), "Macro precision": _pct(score.get("macro_precision", 0)),
                             "Macro recall": _pct(score.get("macro_recall", 0)), "Macro F1": _pct(score.get("macro_f1", 0))})
    st.subheader("Clean classification · same 100 test records per modality")
    st.dataframe(pd.DataFrame(cls_rows), width="stretch", hide_index=True)
    accuracy_rows = []
    for model_key, label in (("baseline", "Federated baseline"), ("dofmbs_combined", "Combined defense")):
        for modality in ("face", "fingerprint"):
            accuracy_rows.append({"Model": label, "Modality": modality.title(),
                                  "Accuracy (%)": 100 * models.get(model_key, {}).get(f"{modality}_classification", {}).get("accuracy", 0)})
    st.bar_chart(pd.DataFrame(accuracy_rows), x="Model", y="Accuracy (%)", color="Modality",
                 x_label="Model", y_label="Accuracy (%)", height=310)
    st.caption("Face and fingerprint accuracy are separate identity-classification tasks.")

    st.subheader("Robustness · face-only FGSM")
    rob_rows = []
    for name, results in metrics.get("robustness", {}).get("models", {}).items():
        for item in results:
            rob_rows.append({"Model": "Federated baseline" if name == "baseline" else "DO-FMBS combined defense",
                             "Epsilon": f"{item['epsilon']:.2f}", "Clean-correct eligible": item["clean_correct_eligible"],
                             "Attack successes": item["attack_successes"], "ASR": _pct(item["attack_success_rate"]),
                             "Resistance (1−ASR)": _pct(item["attack_resistance_1_minus_asr"]),
                             "Accuracy under attack": _pct(item["accuracy_under_attack"])})
    if rob_rows:
        robust_frame = pd.DataFrame(rob_rows)
        st.dataframe(robust_frame, width="stretch", hide_index=True)
        plot = robust_frame.assign(**{"ASR (%)": robust_frame["ASR"].str.rstrip("%").astype(float)})
        st.line_chart(plot, x="Epsilon", y="ASR (%)", color="Model",
                      x_label="FGSM perturbation ε", y_label="Attack success rate (%)", height=300)

    st.subheader("Paired statistical comparison")
    statistical_rows = []
    for modality, result in metrics.get("statistical_comparisons", {}).items():
        statistical_rows.append({"Modality": modality.title(), "Baseline accuracy": _pct(result["baseline_accuracy"]),
                                 "DO-FMBS accuracy": _pct(result["candidate_accuracy"]),
                                 "Paired difference": f"{result['difference_percentage_points']:.2f} pp",
                                 "Identity-cluster 95% CI": str([round(100 * v, 2) for v in result["paired_cluster_bootstrap_95"]]) + " pp",
                                 "Permutation p": f"{result['p_value']:.5f}", "Significant at α=.05": result["significant"]})
    st.dataframe(pd.DataFrame(statistical_rows), width="stretch", hide_index=True)
    st.caption("Primary test: paired identity-cluster sign-flip permutation; record-level McNemar is a sensitivity analysis.")

    st.subheader("Operational claimed-identity verification")
    verify_rows = []
    for model_key, display in (("baseline", "Federated baseline"), ("dofmbs_combined", "DO-FMBS combined defense")):
        for modality in ("face", "fingerprint"):
            item = models.get(model_key, {}).get(f"{modality}_verification", {})
            result = item.get("evaluation", {})
            verify_rows.append({"Model": display, "Modality": modality.title(), "Frozen threshold": item.get("threshold"),
                                "Genuine / impostor claims": f"{result.get('genuine_attempts')} / {result.get('impostor_attempts')}",
                                "FAR": _pct(result.get("FAR", 0)), "FRR": _pct(result.get("FRR", 0)),
                                "TAR": _pct(result.get("TAR", 0)), "AUC": f"{result.get('ROC_AUC', 0):.4f}",
                                "EER": _pct(result.get("EER", 0))})
    st.dataframe(pd.DataFrame(verify_rows), width="stretch", hide_index=True)
    st.caption("Each model and modality uses a threshold calibrated on Phase 1 claims and frozen before evaluation.")

    st.subheader("Computation and overhead")
    overhead = metrics.get("computational_measurements", {})
    comp_rows = []
    for key, label in (("baseline", "Federated baseline"), ("dofmbs_combined", "DO-FMBS combined defense")):
        item = overhead.get(key, {})
        latency = item.get("inference_latency", {})
        comp_rows.append({"Model": label, "Parameters": item.get("model_parameters"), "Checkpoint bytes": item.get("checkpoint_size_bytes"),
                          "Load seconds": item.get("checkpoint_load_seconds_measured"),
                          "100-record p50 seconds": latency.get("batch_latency_seconds_p50"),
                          "100-record p95 seconds": latency.get("batch_latency_seconds_p95")})
    st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)
    estimate = overhead.get("phase2_raw_float32_weight_transfer_equivalent", {})
    with st.expander("Measurement notes"):
        st.caption(f"Recorded training times: Phase 2 {overhead.get('phase2_training_seconds_existing_record')} s; Phase 4 {overhead.get('phase4_training_and_evaluation_seconds_existing_record')} s. Raw float32 transfer estimate: {estimate.get('mib')} MiB; this is derived, not measured network traffic. Memory measurement: {overhead.get('memory_usage', {}).get('status', 'unavailable')}.")
    with st.expander("Evidence integrity and verification"):
        st.dataframe(pd.DataFrame(integrity), width="stretch", hide_index=True)
        st.json(verification, expanded=False)

"""Streamlit evidence dashboard for the synchronized DO-FMBS replay."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "do_fmbs_data" / "final_evidence_manifest.json"

st.set_page_config(page_title="DO-FMBS Scientific Evidence", page_icon="🔐", layout="wide")


@st.cache_data(show_spinner=False)
def load_manifest(path: str, modified_ns: int) -> dict:
    del modified_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percent(value: object, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def source_caption(path: str, stage: str) -> None:
    st.caption(f"Source artifact: `{path}` · Experimental stage: {stage}")


def classification_panel(title: str, values: dict, source: str) -> None:
    st.subheader(title)
    columns = st.columns(5)
    columns[0].metric("Accuracy", percent(values["accuracy"]))
    columns[1].metric("Macro precision", percent(values["macro_precision"]))
    columns[2].metric("Macro recall", percent(values["macro_recall"]))
    columns[3].metric("Macro F1", percent(values["macro_f1"]))
    columns[4].metric("Weighted F1", percent(values["weighted_f1"]))
    low, high = values["accuracy_wilson_95"]
    st.write(
        f"Correct decisions: **{values['correct']}/{values['samples']}**. "
        f"Accuracy 95% Wilson CI: **[{percent(low)}, {percent(high)}]**."
    )
    source_caption(source, "Phase 5 final classification test")


def verification_panel(title: str, values: dict, source: str) -> None:
    final = values["final_test"]
    st.subheader(title)
    columns = st.columns(5)
    columns[0].metric("FAR", percent(final["FAR"], 3))
    columns[1].metric("FRR", percent(final["FRR"], 3))
    columns[2].metric("TAR / GAR", percent(final["TAR"], 3))
    columns[3].metric("ROC AUC", f"{float(final['ROC_AUC']):.4f}")
    columns[4].metric("EER", percent(final["EER"], 3))
    st.write(
        f"FAR = **{final['impostor_accepts']}/{final['impostor_attempts']}**; "
        f"FRR = **{final['genuine_rejects']}/{final['genuine_attempts']}**; "
        f"frozen threshold = **{float(final['threshold']):.6f}**."
    )
    st.write(
        f"FAR 95% Wilson CI: [{percent(final['FAR_wilson_95'][0], 3)}, "
        f"{percent(final['FAR_wilson_95'][1], 3)}]; "
        f"FRR 95% Wilson CI: [{percent(final['FRR_wilson_95'][0], 3)}, "
        f"{percent(final['FRR_wilson_95'][1], 3)}]."
    )
    st.caption(values["threshold_rule"] + ". " + values["accept_rule"] + ".")
    source_caption(source, "Phase 5 calibration and independent final verification test")


def artifact_path(manifest: dict, suffix: str) -> Path | None:
    for item in manifest.get("artifacts", []):
        if item["path"].endswith(suffix):
            return ROOT / item["path"]
    return None


st.title("DO-FMBS Scientific Evidence Dashboard")
st.write("Dynamic Optimization Federated Multimodal Biometric System — synchronized experimental evidence")

if not MANIFEST_PATH.exists():
    st.error("No authoritative replay manifest is available. Run the full Phase 1–6 replay and manifest builder first.")
    st.code(
        ".\\venv\\Scripts\\python.exe scripts\\run_isolated_instrumented_pipeline.py\n"
        ".\\venv\\Scripts\\python.exe scripts\\build_final_evidence_manifest.py",
        language="powershell",
    )
    st.stop()

manifest = load_manifest(str(MANIFEST_PATH), MANIFEST_PATH.stat().st_mtime_ns)
st.success(f"{manifest['status']} · replay `{manifest['replay_id']}`")
st.caption(
    f"Manifest: `{MANIFEST_PATH.relative_to(ROOT)}` · Git commit: `{manifest['git']['commit']}` · "
    f"Working tree dirty when recorded: {manifest['git']['working_tree_dirty_at_manifest_creation']}"
)

tabs = st.tabs([
    "Study & data", "Federated learning", "Classification", "Verification",
    "Adversarial robustness", "Privacy & defenses", "Dynamic optimization", "Traceability",
])

with tabs[0]:
    st.header("Study scope and dataset evidence")
    st.info(manifest["methodological_scope"]["identity_spaces"])
    dataset_rows = []
    for modality in ("face", "fingerprint"):
        row = manifest["dataset_statistics"][modality]
        dataset_rows.append({
            "Modality": modality.title(), "Source": row["source"], "Identities": row["identities"],
            "Total": row["total"], "Training": row["train"], "Calibration": row["calibration"],
            "Final test": row["test"], "Scaler fit": row["scaler_fit"],
        })
    st.dataframe(pd.DataFrame(dataset_rows), use_container_width=True, hide_index=True)
    quality = manifest["dataset_statistics"]["fingerprint"]["quality_gate"]
    st.subheader("Fingerprint identity-separability gate")
    columns = st.columns(4)
    columns[0].metric("Same-identity mean", f"{quality['same_identity_similarity_mean']:.3f}")
    columns[1].metric("Different-identity mean", f"{quality['different_identity_similarity_mean']:.3f}")
    columns[2].metric("Inter/intra ratio", f"{quality['inter_to_intra_distance_ratio']:.2f}")
    columns[3].metric("Nearest-neighbour identity", percent(quality["nearest_neighbour_identity_accuracy"]))
    st.write(f"Gate decision: **{'PASS' if quality['passed'] else 'FAIL'}**")
    source_caption("do_fmbs_data/fingerprint_quality_diagnostics.json (inside authoritative replay)", "Phase 1 pre-training gate")
    for suffix in ("fingerprint_same_identity_grid.png", "fingerprint_similarity_distributions.png"):
        image = artifact_path(manifest, suffix)
        if image and image.exists():
            st.image(str(image), use_container_width=True)

with tabs[1]:
    st.header("Federated learning")
    fed = manifest["federated_learning"]
    parameters = fed["training_parameters"]
    st.write(fed["architecture_scope"] + ".")
    st.write(f"Partition: {fed['federated_partition']}.")
    columns = st.columns(4)
    columns[0].metric("Clients", str(fed["clients"]))
    columns[1].metric("Rounds", str(parameters["rounds"]))
    columns[2].metric("Local epochs", str(parameters["local_epochs"]))
    columns[3].metric("Batch size", str(parameters["batch_size"]))
    st.dataframe(pd.DataFrame([
        {"Modality": "Face", **fed["face_classification"]},
        {"Modality": "Fingerprint", **fed["fingerprint_classification"]},
    ]), use_container_width=True, hide_index=True)
    source_caption("do_fmbs_data/phase2_metrics.json (inside authoritative replay)", "Phase 2 federated baseline")
    image = artifact_path(manifest, "federated_training_convergence.png")
    if image and image.exists():
        st.image(str(image), use_container_width=True)

with tabs[2]:
    st.header("Final independent classification")
    metrics = manifest["final_metrics"]
    classification_panel("Face Classification", metrics["face_classification"], "face_classification_report.json")
    classification_panel("Fingerprint Classification", metrics["fingerprint_classification"], "fingerprint_classification_report.json")
    cols = st.columns(2)
    for column, suffix, caption in zip(cols, ("face_confusion_matrix.png", "fingerprint_confusion_matrix.png"),
                                        ("Face confusion matrix", "Fingerprint confusion matrix")):
        image = artifact_path(manifest, suffix)
        if image and image.exists():
            column.image(str(image), caption=caption, use_container_width=True)

with tabs[3]:
    st.header("Claimed-identity verification")
    verification_panel("Face Verification", manifest["final_metrics"]["face_verification"], "face_verification_report.json")
    verification_panel("Fingerprint Verification", manifest["final_metrics"]["fingerprint_verification"], "fingerprint_verification_report.json")
    for modality in ("face", "fingerprint"):
        st.subheader(f"{modality.title()} verification figures")
        cols = st.columns(3)
        for column, suffix in zip(cols, (
            f"{modality}_genuine_impostor_distribution.png", f"{modality}_verification_roc.png",
            f"{modality}_far_frr_threshold.png",
        )):
            image = artifact_path(manifest, suffix)
            if image and image.exists():
                column.image(str(image), use_container_width=True)

with tabs[4]:
    st.header("Adversarial Robustness")
    st.warning(manifest["methodological_scope"]["adversarial"])
    phase3 = manifest["attacks"]
    st.subheader("FGSM and PGD attack simulation")
    st.dataframe(pd.DataFrame(phase3["evasion"]), use_container_width=True, hide_index=True)
    st.subheader("Final-model face FGSM robustness")
    robust = manifest["final_metrics"]["robustness"]
    rows = []
    for model_name in ("baseline", "do_fmbs"):
        rows.extend({"model": model_name, **row} for row in robust[model_name])
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    source_caption("phase3_attack_results.json and face_robustness_report.json", "Phases 3 and 5; face branch only")
    image = artifact_path(manifest, "face_robustness_validation.png")
    if image and image.exists():
        st.image(str(image), use_container_width=True)

with tabs[5]:
    st.header("Privacy / Defense Results")
    defense = manifest["defenses"]
    st.warning(defense["privacy_terminology"])
    rows = []
    for model_name, values in defense["comparison"].items():
        rows.append({
            "Model": model_name, "Face accuracy": values["face_classification"]["accuracy"],
            "Fingerprint accuracy": values["fingerprint_classification"]["accuracy"],
            "Face FGSM ASR": values["face_fgsm"]["attack_success_rate"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Membership inference")
    st.json(manifest["attacks"]["membership_inference"], expanded=False)
    st.subheader("Label-flipping poisoning")
    st.json(manifest["attacks"]["label_flipping_poisoning"], expanded=False)
    source_caption("phase4_defense_results.json", "Phase 4 defense comparison")
    image = artifact_path(manifest, "defense_comparison.png")
    if image and image.exists():
        st.image(str(image), use_container_width=True)

with tabs[6]:
    st.header("Dynamic Optimization")
    fit = manifest["dynamic_optimization"]["model_fit"]
    st.info(fit["scope_statement"])
    columns = st.columns(4)
    columns[0].metric("Observations", str(fit["observations"]))
    columns[1].metric("R²", f"{fit['r_squared']:.4f}")
    columns[2].metric("Adjusted R²", f"{fit['adjusted_r_squared']:.4f}")
    columns[3].metric("RMSE", f"{fit['rmse']:.4f}")
    st.subheader("Constrained coefficients")
    coefficient_rows = [
        {"Parameter": name, "Estimate": estimate, "95% CI low": low, "95% CI high": high}
        for name, estimate, low, high in zip(fit["names"], fit["coefficients"], fit["ci_low"], fit["ci_high"])
    ]
    st.dataframe(pd.DataFrame(coefficient_rows), use_container_width=True, hide_index=True)
    st.write(f"Constraints satisfied: **{fit['constraints_satisfied']}**. Optimization success: **{fit['optimization_success']}**.")
    source_caption(manifest["dynamic_optimization"]["artifact_directory"] + "/model_fit.json", "Phase 6 controlled simulation")

with tabs[7]:
    st.header("Scientific traceability")
    st.write(f"Authoritative replay root: `{manifest['authoritative_replay_root']}`")
    st.write(f"Recorded Phase 1–6 runtime: **{manifest['runtime']['total_seconds']:.2f} seconds**")
    runtime_rows = [{key: row.get(key) for key in ("phase", "script", "command", "elapsed_seconds", "exit_code", "status")}
                    for row in manifest["runtime"]["records"]]
    st.dataframe(pd.DataFrame(runtime_rows), use_container_width=True, hide_index=True)
    st.subheader("Hashed authoritative artifacts")
    st.dataframe(pd.DataFrame(manifest["artifacts"]), use_container_width=True, hide_index=True)
    st.download_button(
        "Download final evidence manifest",
        data=json.dumps(manifest, indent=2),
        file_name="final_evidence_manifest.json",
        mime="application/json",
    )

"""DO-FMBS phase-by-phase simulation workspace and evidence dashboard."""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from phase1_evidence import inspect_phase1_artifacts


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "do_fmbs_data" / "final_evidence_manifest.json"
RUNNER = ROOT / "scripts" / "run_isolated_instrumented_pipeline.py"
AUDITOR = ROOT / "scripts" / "verify_replay_evidence.py"
MANIFEST_BUILDER = ROOT / "scripts" / "build_final_evidence_manifest.py"
RUN_ROOT = ROOT / "runtime_recording"

PHASES = [
    {
        "number": 1, "name": "Data preparation", "page": "Phase 1 · Data",
        "purpose": "Prepare the face and fingerprint datasets, fit preprocessing on training data, and verify the synthetic fingerprint identities before training.",
        "inputs": "LFW face images and seeded fingerprint identity templates.",
        "outputs": "Train/calibration/test partitions, train-only scalers, client partitions, and the fingerprint quality gate.",
        "prerequisite": "None — first phase in a new replay.",
    },
    {
        "number": 2, "name": "Federated learning", "page": "Phase 2 · Federated",
        "purpose": "Train the independent face and fingerprint identity heads across five class-exclusive non-IID clients and aggregate the local model parameters.",
        "inputs": "Phase 1 partitions, scalers, and exact persisted client files.",
        "outputs": "Federated baseline model, round history, modality losses, and convergence evidence.",
        "prerequisite": "Phase 1 completed successfully.",
    },
    {
        "number": 3, "name": "Attack simulation", "page": "Phase 3 · Attacks",
        "purpose": "Measure face-branch FGSM/PGD evasion, confidence-based membership inference, and the controlled label-flipping scenario.",
        "inputs": "Phase 2 baseline and the persisted training/test partitions.",
        "outputs": "Attack settings, denominators, success rates, and attack plots.",
        "prerequisite": "Phase 2 completed successfully.",
    },
    {
        "number": 4, "name": "Defense mechanisms", "page": "Phase 4 · Defenses",
        "purpose": "Compare the baseline, DP-inspired gradient noise, adversarial training, and the combined two-modality objective.",
        "inputs": "Phase 2 baseline and Phase 3 attack configuration.",
        "outputs": "Per-modality defense metrics, combined losses, and fingerprint-gradient checks.",
        "prerequisite": "Phase 3 completed successfully.",
    },
    {
        "number": 5, "name": "Model validation", "page": "Phase 5 · Validation",
        "purpose": "Evaluate classification and claimed-identity verification separately for each modality, then report uncertainty and paired comparisons.",
        "inputs": "Phase 2 baseline, Phase 4 final model, and the untouched calibration/test partitions.",
        "outputs": "Classification reports, frozen verification thresholds, FAR/FRR, confidence intervals, ROC curves, and robustness evidence.",
        "prerequisite": "Phase 4 completed successfully.",
    },
    {
        "number": 6, "name": "Dynamic optimization", "page": "Phase 6 · Optimization",
        "purpose": "Fit the declared constrained simulation model, check its diagnostics, and calculate the round-specific control schedule.",
        "inputs": "Completed phase evidence and the declared seeded factorial simulation.",
        "outputs": "Coefficients, constraints, diagnostics, sensitivity analysis, and the optimization schedule.",
        "prerequisite": "Phase 5 completed successfully.",
    },
]
PAGE_OPTIONS = ["Overview", *[phase["page"] for phase in PHASES], "Evidence & provenance"]

st.set_page_config(page_title="DO-FMBS | Simulation workspace", page_icon="🧬", layout="wide")
st.markdown("""
<style>
    :root { --navy:#10283f; --blue:#1e4b70; --teal:#087f8c; --ink:#172b3a; --muted:#667887; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#10283f 0%,#173c59 100%); }
    [data-testid="stSidebar"] * { color:#eef5f7; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color:#edf5f7; }
    .hero { padding:1.5rem 1.8rem; border-radius:16px; color:white; background:linear-gradient(115deg,#10283f,#1e5873); margin-bottom:1rem; }
    .hero h1 { color:white; margin:0 0 .35rem 0; font-size:2rem; }
    .hero p { color:#d5e5ec; margin:0; }
    .eyebrow { color:#82d2d0; text-transform:uppercase; letter-spacing:.12em; font-size:.74rem; font-weight:700; }
    .phase-card { background:#f5f8fa; border:1px solid #e1e8ed; border-radius:12px; padding:1rem 1.1rem; min-height:132px; }
    .phase-card h4 { color:#173c59; margin:.15rem 0 .45rem 0; }
    .phase-card p { color:#526574; font-size:.9rem; margin:0; }
    .small-muted { color:#667887; font-size:.86rem; }
    div[data-testid="stMetric"] { background:#f7fafb; border:1px solid #e5ecef; padding:.7rem .85rem; border-radius:10px; }
    .stButton > button[kind="primary"] { background:#087f8c; border-color:#087f8c; }
    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        background:#087f8c !important; color:#ffffff !important;
        border:1px solid #62c5cb !important; font-weight:700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        background:#066a75 !important; border-color:#8be0df !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:focus-visible {
        outline:3px solid #f4c95d !important; outline-offset:2px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_json(path_string: str, modified_ns: int) -> dict:
    del modified_ns
    return json.loads(Path(path_string).read_text(encoding="utf-8"))


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return read_json(str(path), path.stat().st_mtime_ns)
    except (OSError, json.JSONDecodeError):
        return None


def phase1_artifact_root(manifest: dict | None, run_dir: Path | None) -> Path | None:
    if run_dir is not None:
        return run_dir / "do_fmbs_data"
    if manifest:
        replay_root = manifest.get("authoritative_replay_root")
        if replay_root:
            return ROOT / replay_root / "do_fmbs_data"
    return None


def phase1_signature(data_dir: Path) -> tuple:
    names = ["phase1_metadata.json", "face_data.npz", "fingerprint_data.npz", "face_scaler.npz",
             "fingerprint_scaler.npz", "client_partition.json"]
    names.extend(f"clients/client_{index}.npz" for index in range(5))
    signature = []
    for name in names:
        path = data_dir / name
        try:
            info = path.stat()
            signature.append((name, info.st_mtime_ns, info.st_size))
        except OSError:
            signature.append((name, None, None))
    return tuple(signature)


@st.cache_data(show_spinner="Verifying saved Phase 1 artifacts...")
def cached_phase1_inspection(data_dir_string: str, signature: tuple) -> dict:
    del signature
    return inspect_phase1_artifacts(Path(data_dir_string))


@st.cache_data(show_spinner=False)
def cached_sha256(path_string: str, modified_ns: int, size: int) -> str:
    del modified_ns, size
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase1_artifact_status_rows(data_dir: Path, manifest: dict | None,
                                inspection: dict) -> list[dict[str, str]]:
    artifacts = [
        ("Face dataset", "face_data.npz", "face"),
        ("Fingerprint dataset", "fingerprint_data.npz", "fingerprint"),
        ("Face scaler", "face_scaler.npz", "face_scaler"),
        ("Fingerprint scaler", "fingerprint_scaler.npz", "fingerprint_scaler"),
        ("Client partition metadata", "client_partition.json", "clients"),
        *[(f"Client {index} data", f"clients/client_{index}.npz", "clients") for index in range(5)],
        ("Phase 1 metadata", "phase1_metadata.json", "metadata"),
        ("Fingerprint identity templates", "fingerprint_identity_templates.json", "unverified"),
        ("Fingerprint quality diagnostics", "fingerprint_quality_diagnostics.json", "quality"),
        ("Face training preview", "figures/face_training_samples.png", "unverified"),
        ("Fingerprint identity preview", "figures/fingerprint_same_identity_grid.png", "unverified"),
        ("Fingerprint quality preview", "figures/fingerprint_similarity_distributions.png", "unverified"),
        ("Client distribution figure", "figures/federated_client_distribution.png", "unverified"),
    ]
    checks = {row["check"]: row["status"] == "PASS" for row in inspection.get("checks", [])}
    manifest_entries = manifest.get("artifacts", []) if manifest else []
    rows = []
    for label, relative, verification_key in artifacts:
        path = data_dir / relative
        if not path.is_file():
            status = "Missing"
        else:
            manifest_entry = next((item for item in manifest_entries
                                   if item.get("path", "").endswith(f"/do_fmbs_data/{relative.replace(chr(92), '/')}")
                                   or item.get("path", "").endswith(f"do_fmbs_data/{relative.replace(chr(92), '/')}")
                                   or item.get("path", "").endswith(f"/{relative.replace(chr(92), '/')}")
                                   ), None)
            if manifest_entry:
                try:
                    info = path.stat()
                    actual = cached_sha256(str(path), info.st_mtime_ns, info.st_size)
                    status = "Present · manifest hash verified" if actual == manifest_entry.get("sha256") else "Present · hash mismatch"
                except OSError:
                    status = "Present · hash not verified"
            elif verification_key in {"face", "fingerprint"}:
                status = "Present · structural checks passed" if checks.get(f"{verification_key} structure") and checks.get(f"{verification_key} split coverage") else "Present · verification failed"
            elif verification_key.endswith("_scaler"):
                modality = verification_key.removesuffix("_scaler")
                status = "Present · training-only fit verified" if checks.get(f"{modality} scaler") else "Present · verification failed"
            elif verification_key == "clients":
                status = "Present · client consistency verified" if inspection.get("clients_verified") else "Present · verification failed"
            elif verification_key == "metadata":
                status = "Present · metadata consistency checked" if checks.get("Phase 1 metadata") else "Present · verification failed"
            elif verification_key == "quality":
                payload = load_json(path)
                status = "Present · quality gate passed" if payload and payload.get("passed") is True else "Present · not verified / gate failed"
            else:
                status = "Present · not content-verified"
        rows.append({"Artifact": label, "File": relative, "Status": status})
    return rows


def percent(value: object, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def selected_run_dir() -> Path | None:
    raw = st.session_state.get("active_run_dir")
    if not raw:
        return None
    candidate = Path(raw).resolve()
    try:
        candidate.relative_to(RUN_ROOT.resolve())
    except ValueError:
        st.session_state.pop("active_run_dir", None)
        return None
    return candidate


def current_manifest() -> tuple[dict | None, Path | None]:
    run_dir = selected_run_dir()
    if run_dir is not None:
        candidate = run_dir / "do_fmbs_data" / "final_evidence_manifest.json"
        payload = load_json(candidate)
        if payload and payload.get("status") == "AUTHORITATIVE-SYNCHRONIZED-REPLAY-COMPLETE":
            return payload, candidate
        return None, None
    payload = load_json(MANIFEST_PATH)
    if payload and payload.get("status") == "AUTHORITATIVE-SYNCHRONIZED-REPLAY-COMPLETE":
        return payload, MANIFEST_PATH
    return None, None


def registry_for(run_dir: Path | None, manifest: dict | None) -> tuple[list[dict], str | None]:
    if run_dir is not None:
        path = run_dir / "phase_runtime_registry.json"
        payload = load_json(path)
        return (payload or {}).get("records", []), str(path) if path.exists() else None
    if manifest:
        return manifest.get("runtime", {}).get("records", []), None
    return [], None


def completed_phases(records: list[dict]) -> set[int]:
    return {
        int(row["phase"])
        for row in records
        if int(row.get("exit_code", 1)) == 0
        and row.get("status") == "EXECUTION-COMPLETED"
    }


def artifact_path(manifest: dict | None, suffix: str) -> Path | None:
    if not manifest:
        return None
    for item in manifest.get("artifacts", []):
        if item.get("path", "").endswith(suffix):
            path = ROOT / item["path"]
            return path if path.exists() else None
    return None


def phase_artifact_path(run_dir: Path | None, manifest: dict | None, phase: int, suffix: str) -> Path | None:
    if manifest:
        return artifact_path(manifest, suffix)
    if run_dir:
        path = run_dir / "do_fmbs_data" / suffix
        return path if path.exists() else None
    return None


def run_streamed(command: list[str], title: str) -> bool:
    output = st.empty()
    lines: list[str] = []
    status = st.status(title, expanded=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip())
            output.code("\n".join(lines[-45:]), language="text")
        return_code = process.wait()
    except OSError as error:
        status.update(label=f"Could not start: {title}", state="error")
        st.error(str(error))
        return False
    if return_code == 0:
        status.update(label=f"Completed: {title}", state="complete", expanded=False)
        return True
    status.update(label=f"Failed: {title} (exit {return_code})", state="error", expanded=True)
    return False


def start_new_replay() -> Path:
    stamp = datetime.now(ZoneInfo("Africa/Nairobi")).strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"interactive_replay_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = RUN_ROOT / f"interactive_replay_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    st.session_state["active_run_dir"] = str(run_dir)
    st.session_state["selected_page"] = PHASES[0]["page"]
    return run_dir


def execute_phase(phase_number: int, run_dir: Path) -> bool:
    command = [
        sys.executable, "-u", str(RUNNER), "--run-dir", str(run_dir),
        "--start-phase", str(phase_number), "--end-phase", str(phase_number), "--resume",
    ]
    if not run_streamed(command, f"Phase {phase_number}: {PHASES[phase_number - 1]['name']}"):
        return False
    if phase_number != 6:
        st.session_state["selected_page"] = PHASES[phase_number]["page"]
        st.rerun()
        return True

    return finalize_replay(run_dir)


def finalize_replay(run_dir: Path) -> bool:
    if not run_streamed(
        [sys.executable, "-u", str(AUDITOR), "--run-dir", str(run_dir)],
        "Independent replay audit",
    ):
        return False
    manifest_path = run_dir / "do_fmbs_data" / "final_evidence_manifest.json"
    if not run_streamed(
        [sys.executable, "-u", str(MANIFEST_BUILDER), "--run-dir", str(run_dir), "--publish-path", str(manifest_path)],
        "Build synchronized evidence manifest",
    ):
        return False
    st.success("All six phases completed. Independent audit passed and the replay manifest is ready.")
    st.rerun()
    return True


def metric_card_row(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def render_federated_history(history: dict, manifest: dict | None, run_dir: Path | None) -> None:
    if not history or not history.get("round"):
        return
    convergence = artifact_path(manifest, "federated_training_convergence.png") if manifest else None
    if convergence is None and run_dir:
        candidate = run_dir / "do_fmbs_data" / "figures" / "federated_training_convergence.png"
        convergence = candidate if candidate.exists() else None
    if convergence:
        st.image(str(convergence), caption="Modality-specific federated training convergence", use_container_width=True)
    else:
        chart = pd.DataFrame({
            "Face weighted local accuracy": history["face_accuracy"],
            "Fingerprint weighted local accuracy": history["fingerprint_accuracy"],
        }, index=history["round"])
        chart.index.name = "Federated round"
        st.line_chart(chart, height=280)
    accuracy_table = pd.DataFrame({
        "Federated round": history["round"],
        "Face weighted local accuracy": [percent(value) for value in history["face_accuracy"]],
        "Fingerprint weighted local accuracy": [percent(value) for value in history["fingerprint_accuracy"]],
    })
    loss_table = pd.DataFrame({
        "Federated round": history["round"],
        "Face cross-entropy loss": [f"{value:.4f}" for value in history["face_loss"]],
        "Fingerprint cross-entropy loss": [f"{value:.4f}" for value in history["fingerprint_loss"]],
    })
    st.subheader("Values behind the convergence graphs")
    st.markdown("#### Weighted local accuracy by round")
    st.dataframe(accuracy_table, use_container_width=True, hide_index=True)
    st.download_button("Download accuracy values (CSV)", data=accuracy_table.to_csv(index=False),
                       file_name="federated_convergence_accuracy.csv", mime="text/csv",
                       key="download_federated_accuracy_values")
    st.markdown("#### Cross-entropy loss by round")
    st.dataframe(loss_table, use_container_width=True, hide_index=True)
    st.download_button("Download loss values (CSV)", data=loss_table.to_csv(index=False),
                       file_name="federated_convergence_loss.csv", mime="text/csv",
                       key="download_federated_loss_values")
    st.caption("For each round and modality, both plotted metrics are alpha-weighted averages of the five clients' final local-epoch metrics. The combined training objective is 0.5 × face loss + 0.5 × fingerprint loss. These are training-history values, not final test metrics.")


def render_phase_result(phase_number: int, manifest: dict | None, run_dir: Path | None) -> None:
    if not manifest:
        if run_dir:
            st.info("This replay is still in progress. Phase results appear here after this phase finishes.")
        else:
            st.info("The accepted replay manifest is not available. Start a new replay to generate phase evidence.")
        return

    if phase_number == 1:
        data_dir = phase1_artifact_root(manifest, run_dir)
        if data_dir is None:
            st.info("Start a replay to inspect its persisted Phase 1 datasets and evidence.")
            return
        inspection = cached_phase1_inspection(str(data_dir), phase1_signature(data_dir))
        metadata = load_json(data_dir / "phase1_metadata.json") or {}
        st.subheader("Dataset acquisition and provenance")
        st.write("Face data uses public LFW samples loaded with scikit-learn. Fingerprints are generated locally from deterministic synthetic identity templates; no fingerprint dataset is downloaded or imported.")
        st.caption("LFW loader cache/download outcome was not recorded for this replay. This page reads saved artifacts only and will not trigger a dataset fetch.")

        summaries = inspection.get("datasets", {})
        overview_rows = []
        for modality in ("face", "fingerprint"):
            value = summaries.get(modality)
            if not value:
                continue
            recorded = metadata.get(modality, {})
            overview_rows.append({
                "Modality": modality.title(),
                "Data source": recorded.get("source", value["source"]),
                "Acquisition": value["acquisition"],
                "Samples": value["samples"], "Identities": value["identities"],
                "Image dimensions": " × ".join(map(str, value["image_shape"])),
                "Flattened features": value["features"],
                "Representation": value["representation"],
            })
        st.subheader("Dataset overview")
        if overview_rows:
            st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No readable Phase 1 dataset archives are available in this replay yet.")

        st.subheader("Preprocessing actually performed")
        left, right = st.columns(2)
        left.markdown("**Face**\n\n1. Load LFW through `fetch_lfw_people`.\n2. Create stratified train/calibration/test partitions.\n3. Flatten 50 × 37 grayscale images to 1,850 features.\n4. Fit `StandardScaler` on training samples only; transform train, calibration, and test data.\n5. Persist transformed arrays, original images, labels, source indices, and scaler statistics.")
        right.markdown("**Fingerprint**\n\n1. Generate deterministic identity templates and eight varied impressions per identity.\n2. Run the identity-separability quality gate.\n3. Split impressions 0–4 / 5 / 6–7 into train/calibration/test.\n4. Flatten 96 × 96 grayscale images to 9,216 features.\n5. Fit `StandardScaler` on training samples only; transform all partitions and persist arrays, labels, indices, and scaler statistics.")
        seeds = metadata.get("seeds", {})
        partition_metadata = load_json(data_dir / "client_partition.json") or {}
        seed_cols = st.columns(4)
        for column, label, value in zip(seed_cols,
                                        ("Global seed", "Face split seed", "Client partition seed", "Fingerprint identity / impression seeds"),
                                        (seeds.get("global", "Not recorded"), seeds.get("face_split", "Not recorded"),
                                         partition_metadata.get("seed", "Not recorded"),
                                         f"{seeds.get('fingerprint_identity_base', '—')} / {seeds.get('fingerprint_impression_base', '—')}")):
            column.metric(label, str(value))

        st.subheader("Train / calibration / test distribution")
        split_rows = [{"Modality": name.title(), **value.get("splits", {})}
                      for name, value in summaries.items()]
        if split_rows:
            split_frame = pd.DataFrame(split_rows).set_index("Modality")
            st.bar_chart(split_frame, height=240)
            st.dataframe(split_frame.reset_index(), use_container_width=True, hide_index=True)
        st.caption("Face partitions are stratified 70/15/15 (split seed shown above). Fingerprints use five training, one calibration, and two final-test impressions per identity.")

        st.subheader("Persisted face examples")
        previews = inspection.get("face_previews", [])
        if previews:
            for offset in range(0, len(previews), 4):
                columns = st.columns(min(4, len(previews) - offset))
                for column, preview in zip(columns, previews[offset:offset + 4]):
                    column.image(preview["image"], caption=preview["caption"], use_container_width=True)
        else:
            st.info("Face sample previews are unavailable because the persisted raw training images are missing.")

        st.subheader("Synthetic fingerprint identity quality")
        quality = (metadata.get("fingerprint", {}).get("quality_gate")
                   or load_json(data_dir / "fingerprint_quality_diagnostics.json") or {})
        if quality:
            metric_card_row([
                ("Same identity mean", f"{quality.get('same_identity_similarity_mean', 0):.3f}"),
                ("Different identity mean", f"{quality.get('different_identity_similarity_mean', 0):.3f}"),
                ("Inter/intra distance ratio", f"{quality.get('inter_to_intra_distance_ratio', 0):.2f}"),
                ("Nearest-neighbour accuracy", percent(quality.get("nearest_neighbour_identity_accuracy", 0))),
                ("Gate", "PASS" if quality.get("passed") else "FAIL / unknown"),
            ])
        finger_cols = st.columns(2)
        for column, relative, title in zip(finger_cols,
                                           ("figures/fingerprint_same_identity_grid.png", "figures/fingerprint_similarity_distributions.png"),
                                           ("Persisted synthetic impressions", "Persisted identity-separability plot")):
            path = data_dir / relative
            if path.is_file():
                column.image(str(path), caption=title, use_container_width=True)
            else:
                column.info(f"Optional artifact missing: {Path(relative).name}")

        st.subheader("Identity-exclusive client allocation")
        if inspection.get("clients"):
            client_frame = pd.DataFrame(inspection["clients"])
            st.bar_chart(client_frame.set_index("Client")[
                ["Face samples", "Fingerprint samples", "Face identities", "Fingerprint identities"]
            ], height=280)
            st.dataframe(client_frame, use_container_width=True, hide_index=True)
            st.caption("Bars use counts read from the saved five client NPZ files. Unequal face sample counts are preserved; face and fingerprint identities are allocated independently.")
        else:
            st.warning("Client distribution cannot be shown because persisted client artifacts are unavailable.")

        st.subheader("Phase 1 verification")
        if inspection.get("status") == "PASS":
            st.success("All available Phase 1 structural, split, scaler, metadata, and client checks passed.")
        else:
            st.warning("Some Phase 1 evidence is missing or failed verification; review the check table before relying on it.")
        st.dataframe(pd.DataFrame(inspection.get("checks", [])), use_container_width=True, hide_index=True)

        st.subheader("Phase 1 artifact status")
        st.caption("A file is not marked verified based on existence alone. Accepted replay files are checked against manifest SHA-256 when a matching manifest entry exists; otherwise status reflects the specific structural check shown.")
        st.dataframe(pd.DataFrame(phase1_artifact_status_rows(data_dir, manifest, inspection)),
                     use_container_width=True, hide_index=True)

    elif phase_number == 2:
        fed = manifest["federated_learning"]
        params = fed["training_parameters"]
        st.subheader("Federated configuration")
        metric_card_row([("Clients", str(fed["clients"])), ("Rounds", str(params["rounds"])),
                         ("Local epochs", str(params["local_epochs"])), ("Batch size", str(params["batch_size"])),
                         ("Learning rate", str(params["learning_rate"]))])
        st.caption(fed["federated_partition"] + ". Labels remain independent across the two modalities.")
        rows = []
        for modality, result in (("Face", fed["face_classification"]), ("Fingerprint", fed["fingerprint_classification"])):
            rows.append({"Modality": modality, "Samples": result["samples"], "Classes": result["classes"],
                         "Accuracy": percent(result["accuracy"]), "Test cross-entropy": f"{result['loss']:.4f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        history = fed.get("history", {})
        render_federated_history(history, manifest, run_dir)
        image = artifact_path(manifest, "federated_client_distribution.png")
        if image:
            st.image(str(image), caption="Class-exclusive non-IID client allocation", use_container_width=True)

    elif phase_number == 3:
        attacks = manifest["attacks"]
        st.warning("FGSM/PGD evasion measurements apply to the face branch only.")
        st.subheader("Evasion experiments")
        st.dataframe(pd.DataFrame(attacks.get("evasion", [])), use_container_width=True, hide_index=True)
        left, right = st.columns(2)
        left.markdown("#### Membership inference")
        left.json(attacks.get("membership_inference", {}), expanded=False)
        right.markdown("#### Label-flipping poisoning")
        right.json(attacks.get("label_flipping_poisoning", {}), expanded=False)
        image = artifact_path(manifest, "face_adversarial_attacks.png")
        if image:
            st.image(str(image), caption="Face-branch adversarial evaluation", use_container_width=True)

    elif phase_number == 4:
        defense = manifest["defenses"]
        st.info("Combined objective: 0.5 × face loss + 0.5 × fingerprint loss. Gaussian gradient noise is DP-inspired; no formal privacy budget is claimed.")
        rows = []
        for name, value in defense["comparison"].items():
            rows.append({"Configuration": name.replace("_", " ").title(),
                         "Face accuracy": percent(value["face_classification"]["accuracy"]),
                         "Fingerprint accuracy": percent(value["fingerprint_classification"]["accuracy"]),
                         "Face FGSM ASR": percent(value["face_fgsm"]["attack_success_rate"])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("The replay records nonzero fingerprint-head gradients during combined-loss training.")
        image = artifact_path(manifest, "defense_comparison.png")
        if image:
            st.image(str(image), caption="Modality-specific defense comparison", use_container_width=True)

    elif phase_number == 5:
        metrics = manifest["final_metrics"]
        st.subheader("Classification")
        rows = []
        for modality in ("face", "fingerprint"):
            result = metrics[f"{modality}_classification"]
            rows.append({"Modality": modality.title(), "Correct": f"{result['correct']}/{result['samples']}",
                         "Accuracy": percent(result["accuracy"]), "Macro F1": percent(result["macro_f1"]),
                         "Weighted F1": percent(result["weighted_f1"])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.subheader("Claimed-identity verification")
        rows = []
        for modality in ("face", "fingerprint"):
            result = metrics[f"{modality}_verification"]["final_test"]
            rows.append({"Modality": modality.title(), "FAR": percent(result["FAR"], 3),
                         "FRR": percent(result["FRR"], 3), "TAR": percent(result["TAR"], 3),
                         "AUC": f"{result['ROC_AUC']:.4f}", "EER": percent(result["EER"], 3),
                         "Threshold": f"{result['threshold']:.6f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Each threshold was selected on calibration data and frozen before the final test. Classification accuracy is a separate task from FAR/FRR.")
        cols = st.columns(2)
        for col, suffix, title in zip(cols, ("face_confusion_matrix.png", "fingerprint_confusion_matrix.png"),
                                      ("Face confusion matrix", "Fingerprint confusion matrix")):
            image = artifact_path(manifest, suffix)
            if image:
                col.image(str(image), caption=title, use_container_width=True)

    elif phase_number == 6:
        fit = manifest["dynamic_optimization"]["model_fit"]
        st.info(fit["scope_statement"])
        metric_card_row([("Observations", str(fit["observations"])), ("R²", f"{fit['r_squared']:.4f}"),
                         ("Adjusted R²", f"{fit['adjusted_r_squared']:.4f}"),
                         ("Cross-validated R²", f"{fit['cross_validation_r2_mean']:.4f}"),
                         ("RMSE", f"{fit['rmse']:.4f}")])
        rows = [{"Coefficient": name, "Estimate": estimate, "95% CI": f"[{low:.4f}, {high:.4f}]"}
                for name, estimate, low, high in zip(fit["names"], fit["coefficients"], fit["ci_low"], fit["ci_high"])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        schedule = phase_artifact_path(run_dir, manifest, 6, "dynamic_optimization_schedule.csv")
        if schedule:
            frame = pd.read_csv(schedule)
            if "round" in frame:
                st.line_chart(frame.set_index("round"), height=300)
        st.caption(f"Coefficient constraints satisfied: {fit['constraints_satisfied']}. Optimization success: {fit['optimization_success']}.")


def render_overview(manifest: dict | None, records: list[dict], run_dir: Path | None) -> None:
    st.markdown('<div class="hero"><div class="eyebrow">Research simulation workspace</div><h1>DO-FMBS</h1><p>Six phases, traceable inputs, visible execution, and evidence from one replay.</p></div>', unsafe_allow_html=True)
    done = completed_phases(records)
    if manifest:
        state_text = "Completed replay"
        replay_id = manifest["replay_id"]
        runtime = manifest.get("runtime", {}).get("total_seconds", 0.0)
    elif run_dir:
        state_text = "Replay in progress"
        replay_id = run_dir.name
        runtime = sum(float(row.get("elapsed_seconds", 0)) for row in records)
    else:
        state_text = "Ready to begin"
        replay_id = "No active replay"
        runtime = 0.0
    metric_card_row([("Replay status", state_text), ("Replay ID", replay_id),
                     ("Phase progress", f"{len(done)}/6"), ("Recorded runtime", f"{runtime:.1f} sec")])
    st.progress(len(done) / 6, text=f"Simulation pipeline: {len(done)} of 6 phases complete")
    st.markdown("### Phase workflow")
    for offset in range(0, 6, 3):
        cols = st.columns(3)
        for col, phase in zip(cols, PHASES[offset:offset + 3]):
            number = phase["number"]
            status_text = "Complete" if number in done else ("Ready" if number == (len(done) + 1) else "Waiting")
            col.markdown(
                f'<div class="phase-card"><div class="eyebrow">Phase {number} · {status_text}</div>'
                f'<h4>{phase["name"]}</h4><p>{phase["purpose"]}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown("### Research guardrails")
    st.write("Face and fingerprint branches have separate identity spaces. Fingerprints are synthetic and identity-persistent. Evasion results are face-branch only. Gradient noise is DP-inspired and has no formal epsilon/delta guarantee.")
    if run_dir is None:
        st.button("Start a new phase-by-phase simulation", type="primary", use_container_width=True,
                  on_click=start_new_replay)
    elif len(done) == 6 and manifest:
        st.success("The active replay completed all six phases and passed the independent evidence audit.")
    elif done:
        next_phase = min(max(done) + 1, 6)
        st.info(f"Next phase: {PHASES[next_phase - 1]['name']}. Select it in the sidebar to review its process and run it.")
    else:
        st.info("A new replay folder is ready. Select Phase 1 in the sidebar to begin data preparation.")


def render_phase_page(phase: dict, manifest: dict | None, run_dir: Path | None,
                      records: list[dict], completed: set[int]) -> None:
    number = phase["number"]
    record = next((row for row in records if int(row["phase"]) == number), None)
    done = number in completed
    st.markdown(f'<div class="eyebrow">Phase {number} of 6</div>', unsafe_allow_html=True)
    st.title(phase["name"])
    st.write(phase["purpose"])
    columns = st.columns([2, 2, 2, 1.2])
    columns[0].markdown("**Prerequisite**\n\n" + phase["prerequisite"])
    columns[1].markdown("**Inputs**\n\n" + phase["inputs"])
    columns[2].markdown("**Outputs**\n\n" + phase["outputs"])
    columns[3].metric("Status", "Complete" if done else ("Failed" if record and record.get("exit_code", 0) else "Waiting"))

    run_config = manifest.get("phase_parameters", {}) if manifest else {}
    config_key = f"phase{number}"
    configuration = run_config.get(config_key)
    if configuration:
        with st.expander("Recorded configuration", expanded=False):
            st.json(configuration, expanded=False)

    phase_is_next = number == 1 or all(index in completed for index in range(1, number))
    if run_dir is None:
        st.warning("To execute phases in order, start a new replay from Overview. The accepted replay shown elsewhere remains available as read-only evidence.")
    elif phase_is_next:
        if done:
            st.success(f"Phase {number} is complete in replay `{run_dir.name}`.")
            st.caption("This phase is locked for the current replay to preserve downstream consistency. Start another replay to recompute it.")
        else:
            st.info("The phase runner will stream its console output here. Each phase writes into this replay's isolated folder.")
            if st.button(f"Run Phase {number}: {phase['name']}", type="primary"):
                execute_phase(number, run_dir)
        if number == 6 and done and not manifest:
            st.warning("Phase 6 finished, but audit or manifest publication did not. Retry finalization without rerunning model phases.")
            if st.button("Retry audit and publish evidence", type="primary"):
                finalize_replay(run_dir)
    else:
        waiting_for = next((index for index in range(1, number) if index not in completed), None)
        st.warning(f"Complete Phase {waiting_for} first. The workflow keeps phase inputs and evidence in sequence.")

    if record:
        st.caption(f"Last run: {record.get('start_timestamp', 'time unavailable')} · {float(record.get('elapsed_seconds', 0)):.1f} sec · exit {record.get('exit_code')}")
        log = run_dir / record.get("log_path", f"phase{number}_console.log") if run_dir else None
        if log and log.exists():
            with st.expander("Phase console log", expanded=not done):
                try:
                    st.code("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]), language="text")
                except OSError:
                    st.caption("Log file could not be read.")

    if done and manifest:
        st.divider()
        render_phase_result(number, manifest, run_dir)
    elif done and run_dir:
        st.divider()
        st.subheader("Phase output")
        output_names = {
            1: "phase1_metadata.json", 2: "phase2_metrics.json", 3: "phase3_attack_results.json",
            4: "phase4_defense_results.json", 5: "phase5_validation.json",
        }
        if number in output_names:
            payload = load_json(run_dir / "do_fmbs_data" / output_names[number])
            if payload:
                if number == 2 and payload.get("history"):
                    render_federated_history(payload["history"], None, run_dir)
                    st.dataframe(pd.DataFrame([
                        {"Modality": "Face", "Accuracy": percent(payload["face_classification"]["accuracy"]),
                         "Test loss": f"{payload['face_classification']['loss']:.4f}"},
                        {"Modality": "Fingerprint", "Accuracy": percent(payload["fingerprint_classification"]["accuracy"]),
                         "Test loss": f"{payload['fingerprint_classification']['loss']:.4f}"},
                    ]), use_container_width=True, hide_index=True)
                else:
                    st.json(payload, expanded=False)
        else:
            st.info("Phase 6 outputs are saved in the replay's simulation_evidence/constrained_dynamic_model folder.")


def render_evidence(manifest: dict | None, records: list[dict], manifest_path: Path | None) -> None:
    st.title("Evidence & provenance")
    if not manifest:
        st.info("A synchronized manifest is published after all six phases and the independent audit complete.")
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        return
    st.success(f"{manifest['status']} · replay `{manifest['replay_id']}`")
    st.write(f"Manifest location: `{manifest_path}`")
    st.write(f"Recorded total phase runtime: **{manifest['runtime']['total_seconds']:.2f} seconds**")
    st.subheader("Phase execution records")
    runtime_rows = [{key: row.get(key) for key in ("phase", "script", "scope", "elapsed_seconds", "exit_code", "status")}
                    for row in manifest["runtime"]["records"]]
    st.dataframe(pd.DataFrame(runtime_rows), use_container_width=True, hide_index=True)
    st.subheader("Independent replay audit")
    st.json(manifest.get("independent_audit", {}), expanded=False)
    st.subheader("Hashed artifacts")
    st.dataframe(pd.DataFrame(manifest.get("artifacts", [])), use_container_width=True, hide_index=True)
    st.download_button("Download evidence manifest", data=json.dumps(manifest, indent=2),
                       file_name="final_evidence_manifest.json", mime="application/json")


manifest, manifest_path = current_manifest()
run_dir = selected_run_dir()
records, _registry_path = registry_for(run_dir, manifest)
done = completed_phases(records)

with st.sidebar:
    st.markdown("<div style='padding:.5rem 0 1rem'><div style='font-size:1.25rem;font-weight:800'>DO-FMBS</div><div style='font-size:.8rem;color:#c5d9e2'>Simulation & evidence workspace</div></div>", unsafe_allow_html=True)
    if manifest:
        st.caption(f"Evidence: {manifest['replay_id']}")
    elif run_dir:
        st.caption(f"Active: {run_dir.name}")
    else:
        st.caption("No active simulation")
    st.progress(len(done) / 6, text=f"{len(done)} / 6 phases")
    for phase in PHASES:
        number = phase["number"]
        marker = "✓" if number in done else ("▶" if number == (len(done) + 1) else "·")
        st.caption(f"{marker}  Phase {number} — {phase['name']}")
    st.divider()
    st.radio("Navigate the workflow", PAGE_OPTIONS, key="selected_page", label_visibility="collapsed")
    st.divider()
    if run_dir:
        st.caption("New simulations run in isolated folders; the accepted evidence manifest stays intact until you select a completed replay.")
        st.button("Start another replay", use_container_width=True, on_click=start_new_replay)
    elif manifest:
        st.button("Start a new simulation", use_container_width=True, on_click=start_new_replay)
    st.markdown("<div style='font-size:.76rem;color:#c5d9e2;padding-top:1rem'>Lydiah Moraa Machora<br>JOOUST · Master's research</div>", unsafe_allow_html=True)

selected_page = st.session_state.get("selected_page", "Overview")
if selected_page == "Overview":
    render_overview(manifest, records, run_dir)
elif selected_page == "Evidence & provenance":
    render_evidence(manifest, records, manifest_path)
else:
    phase = next(item for item in PHASES if item["page"] == selected_page)
    render_phase_page(phase, manifest, run_dir, records, done)

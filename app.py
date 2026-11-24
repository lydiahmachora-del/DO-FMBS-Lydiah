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
import numpy as np
import streamlit as st

from phase1_evidence import inspect_phase1_artifacts
from phase1_reimplementation_evidence import inspect_phase1 as inspect_rebuilt_phase1
from phase3_dashboard import render_phase3_authoritative
from phase4_dashboard import render_phase4_authoritative
from phase5_dashboard import render_phase5_authoritative
from phase6_dashboard import render_phase6_authoritative
from phase_configuration import (
    CONFIG_VERSION, PHASE_SCHEMAS, changed_parameters, cli_arguments,
    defaults_for_all_phases, phase_defaults, validate_phase_config,
)
from research_consolidation_dashboard import ACCEPTED_RUNS, render_research_consolidation


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "do_fmbs_data" / "final_evidence_manifest.json"
RUNNER = ROOT / "run_interactive_pipeline.py"
AUDITOR = ROOT / "scripts" / "verify_replay_evidence.py"
MANIFEST_BUILDER = ROOT / "scripts" / "build_final_evidence_manifest.py"
RUN_ROOT = ROOT / "runtime_recording"
ACTIVE_RUN_PATH = RUN_ROOT / ".active_interactive_run"

PHASES = [
    {
        "number": 1, "name": "Data preparation", "page": "Run Phase 1 · Data preparation",
        "purpose": "Prepare the face and fingerprint datasets, fit preprocessing on training data, and verify the synthetic fingerprint identities before training.",
        "inputs": "LFW face images and seeded fingerprint identity templates.",
        "outputs": "Train/calibration/test partitions, train-only scalers, client partitions, and the fingerprint quality gate.",
        "prerequisite": "None — first phase in a new replay.",
    },
    {
        "number": 2, "name": "Federated learning", "page": "Run Phase 2 · Federated baseline",
        "purpose": "Train the independent face and fingerprint identity heads across five class-exclusive non-IID clients and aggregate the local model parameters.",
        "inputs": "Phase 1 partitions, scalers, and exact persisted client files.",
        "outputs": "Federated baseline model, round history, modality losses, and convergence evidence.",
        "prerequisite": "Phase 1 completed successfully.",
    },
    {
        "number": 3, "name": "Attack simulation", "page": "Run Phase 3 · Attack experiments",
        "purpose": "Measure face-branch FGSM/PGD evasion, confidence-based membership inference, and the controlled label-flipping scenario.",
        "inputs": "Phase 2 baseline and the persisted training/test partitions.",
        "outputs": "Attack settings, denominators, success rates, and attack plots.",
        "prerequisite": "Phase 2 completed successfully.",
    },
    {
        "number": 4, "name": "Defense mechanisms", "page": "Run Phase 4 · Defenses",
        "purpose": "Compare the baseline, DP-inspired gradient noise, adversarial training, and the combined two-modality objective.",
        "inputs": "Phase 2 baseline and Phase 3 attack configuration.",
        "outputs": "Per-modality defense metrics, combined losses, and fingerprint-gradient checks.",
        "prerequisite": "Phase 3 completed successfully.",
    },
    {
        "number": 5, "name": "Model validation", "page": "Run Phase 5 · Validation",
        "purpose": "Evaluate classification and claimed-identity verification separately for each modality, then report uncertainty and paired comparisons.",
        "inputs": "Phase 2 baseline, Phase 4 final model, and the untouched calibration/test partitions.",
        "outputs": "Classification reports, frozen verification thresholds, FAR/FRR, confidence intervals, ROC curves, and robustness evidence.",
        "prerequisite": "Phase 4 completed successfully.",
    },
    {
        "number": 6, "name": "Dynamic optimization", "page": "Run Phase 6 · Dynamic optimization",
        "purpose": "Fit the declared constrained simulation model, check its diagnostics, and calculate the round-specific control schedule.",
        "inputs": "Completed phase evidence and the declared seeded factorial simulation.",
        "outputs": "Coefficients, constraints, diagnostics, sensitivity analysis, and the optimization schedule.",
        "prerequisite": "Phase 5 completed successfully.",
    },
]
RESEARCH_COMPARISON_PAGE = "Cross-Phase Analysis"
RESEARCH_SCOPE_PAGE = "Research Scope & Limitations"
SIMULATION_PAGE = "Simulation"
PHASE1_REBUILD_PAGE = "Phase 1 · Data preparation"
PHASE2_AUTHORITATIVE_PAGE = "Phase 2 · Federated baseline"
PHASE3_AUTHORITATIVE_PAGE = "Phase 3 · Attack experiments"
PHASE4_AUTHORITATIVE_PAGE = "Phase 4 · Defense mechanisms"
PHASE5_AUTHORITATIVE_PAGE = "Phase 5 · Model validation"
PHASE6_AUTHORITATIVE_PAGE = "Phase 6 · Dynamic optimization"
PAGE_OPTIONS = [
    "Overview", SIMULATION_PAGE, RESEARCH_COMPARISON_PAGE, PHASE1_REBUILD_PAGE,
    PHASE2_AUTHORITATIVE_PAGE, PHASE3_AUTHORITATIVE_PAGE,
    PHASE4_AUTHORITATIVE_PAGE, PHASE5_AUTHORITATIVE_PAGE,
    PHASE6_AUTHORITATIVE_PAGE, RESEARCH_SCOPE_PAGE,
]
DIAGNOSTIC_OPTIONS = ["Choose a workflow view", *[phase["page"] for phase in PHASES], "Evidence & provenance"]

st.set_page_config(page_title="DO-FMBS | Simulation workspace", page_icon="🧬", layout="wide")
st.markdown("""
<style>
    :root { --navy:#17364e; --blue:#285875; --teal:#167d83; --ink:#1c2d3a; --muted:#5d6e79; --line:#dce5ea; }
    .block-container { padding-top:1.35rem; padding-bottom:2.5rem; max-width:1480px; }
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:var(--ink); }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color:var(--ink); }
    .hero { padding:1.1rem 1.35rem; border-radius:10px; color:var(--ink); background:#edf4f5; border:1px solid #dce8e9; margin-bottom:.8rem; }
    .hero h1 { color:var(--navy); margin:0 0 .25rem 0; font-size:1.8rem; }
    .hero p { color:#4f626e; margin:0; }
    .eyebrow { color:#176f79; text-transform:uppercase; letter-spacing:.08em; font-size:.72rem; font-weight:700; }
    .phase-card { background:#ffffff; border:1px solid var(--line); border-radius:9px; padding:.85rem .95rem; min-height:158px; height:158px; box-sizing:border-box; }
    .phase-card h4 { color:var(--navy); margin:.15rem 0 .4rem 0; }
    .phase-card p { color:#4e606d; font-size:.86rem; margin:0; line-height:1.42; }
    .small-muted { color:var(--muted); font-size:.86rem; }
    div[data-testid="stMetric"] { background:#ffffff; border:1px solid var(--line); padding:.6rem .75rem; border-radius:9px; }
    [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] { border:1px solid var(--line); border-radius:10px; background:#fff; padding:.25rem; }
    .stButton > button[kind="primary"] { background:#176f79; border-color:#176f79; }
    [data-testid="stDialog"] [role="dialog"] {
        width:min(92vw, 860px) !important; max-width:calc(100vw - 2rem) !important;
        max-height:88vh !important; overflow-y:auto !important; box-sizing:border-box;
        overscroll-behavior:contain; scrollbar-gutter:stable;
    }
    [data-testid="stDialog"] [role="dialog"] > div { min-width:0; }
    [data-testid="stDialog"] [data-testid="stExpander"] { margin:.2rem 0; }
    [data-testid="stDialog"] [data-testid="stCaptionContainer"] p { margin-bottom:.15rem; }
    [data-testid="stDialog"] .st-key-phase_configuration_footer {
        position:sticky; bottom:0; z-index:20; padding:.65rem .25rem .25rem;
        margin-top:.35rem; background:var(--background-color, #ffffff);
        border-top:1px solid var(--line, #dce5ea);
    }
    [data-testid="stDialog"] .st-key-phase_configuration_footer button {
        min-height:2.65rem; white-space:normal;
    }
    @media (max-height: 760px) {
        [data-testid="stDialog"] [role="dialog"] { max-height:82vh !important; }
    }
    h1, h2, h3 { color:#17364e; letter-spacing:-.015em; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        background:#167d83 !important; color:#ffffff !important;
        border:1px solid #167d83 !important; font-weight:700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        background:#126c71 !important; border-color:#126c71 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:focus-visible {
        outline:3px solid #f4c95d !important; outline-offset:2px !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] { background:#ffffff; }
    @media (max-width: 900px) {
        .phase-card { min-height:180px; height:auto; }
        .block-container { padding-left:1rem; padding-right:1rem; }
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
    candidates = [st.session_state.get("active_run_dir")]
    if ACTIVE_RUN_PATH.is_file():
        try:
            persisted = ACTIVE_RUN_PATH.read_text(encoding="utf-8").strip()
            if persisted:
                candidates.append(persisted)
        except OSError:
            pass
    for raw in candidates:
        if not raw:
            continue
        raw_path = Path(raw)
        candidate = (raw_path if raw_path.is_absolute() else RUN_ROOT / raw_path).resolve()
        try:
            candidate.relative_to(RUN_ROOT.resolve())
        except ValueError:
            continue
        if candidate.is_dir():
            st.session_state["active_run_dir"] = str(candidate)
            return candidate
    st.session_state.pop("active_run_dir", None)
    return None


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


def run_streamed(command: list[str], title: str, phase_progress=None) -> bool:
    lines: list[str] = []
    status = st.status(f"Running · {title}", expanded=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        with st.expander("Technical execution log", expanded=True):
            st.caption("Live output from the running phase. New lines appear automatically as the process reports progress.")
            output = st.empty()
            for line in process.stdout:
                lines.append(line.rstrip())
                output.code("\n".join(lines[-45:]), language="text")
                if phase_progress and line.startswith("[TIMING] Phase "):
                    parts = line.split()
                    if len(parts) > 3 and parts[2].isdigit():
                        phase_number = int(parts[2])
                        if " started at " in line:
                            phase_progress(phase_number, "running")
                        elif " ended with " in line:
                            phase_progress(phase_number, "completed" if "EXECUTION-COMPLETED" in line else "failed")
        return_code = process.wait()
    except OSError as error:
        status.update(label=f"Error · {title}", state="error", expanded=False)
        st.error("The simulation process could not be started.")
        with st.expander("Technical details"):
            st.code(str(error))
        return False
    if return_code == 0:
        status.update(label=f"Completed · {title}", state="complete", expanded=False)
        return True
    status.update(label=f"Error · {title}", state="error", expanded=False)
    st.error("This phase ended with an execution error. Review the technical log for details.")
    with st.expander("Technical details"):
        st.write(f"Process exit code: {return_code}")
        st.code("\n".join(lines[-80:]), language="text")
    return False


def start_new_replay() -> Path:
    stamp = datetime.now(ZoneInfo("Africa/Nairobi")).strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"interactive_replay_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = RUN_ROOT / f"interactive_replay_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    ACTIVE_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_RUN_PATH.write_text(run_dir.name, encoding="utf-8")
    st.session_state["active_run_dir"] = str(run_dir)
    st.session_state["pending_page_navigation"] = {
        "selected_page": SIMULATION_PAGE,
        "diagnostic_page": DIAGNOSTIC_OPTIONS[0],
    }
    return run_dir


def queue_page_navigation(selected_page: str, diagnostic_page: str = DIAGNOSTIC_OPTIONS[0]) -> None:
    st.session_state["pending_page_navigation"] = {
        "selected_page": selected_page,
        "diagnostic_page": diagnostic_page,
    }


def read_run_configuration(run_dir: Path | None) -> dict[str, dict]:
    path = run_dir / "run_config.json" if run_dir else None
    payload = load_json(path) if path else None
    phases = (payload or {}).get("phases", {})
    return {f"phase{phase}": validate_phase_config(phase, phases.get(f"phase{phase}", {}))
            for phase in range(1, 7)}


def save_run_configuration(run_dir: Path, configurations: dict[str, dict]) -> Path:
    validated = {f"phase{phase}": validate_phase_config(
        phase, configurations.get(f"phase{phase}", {})) for phase in range(1, 7)}
    payload = {"run_id": run_dir.name, "configuration_version": CONFIG_VERSION,
               "created_at": datetime.now(ZoneInfo("Africa/Nairobi")).isoformat(timespec="seconds"),
               "default_preset": "research_defaults", "phases": validated}
    path = run_dir / "run_config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _form_key(phase: int, parameter: str) -> str:
    return f"phase_config_{phase}_{parameter}"


def open_phase_configuration(phase_number: int | None, run_dir: Path | None = None) -> None:
    st.session_state["pending_phase_configuration"] = "all" if phase_number is None else phase_number
    st.session_state["pending_configuration_run_dir"] = str(run_dir) if run_dir else ""
    configurations = read_run_configuration(run_dir) if phase_number is not None else defaults_for_all_phases()
    selected_phases = range(1, 7) if phase_number is None else [phase_number]
    for phase in selected_phases:
        for parameter, value in configurations[f"phase{phase}"].items():
            st.session_state[_form_key(phase, parameter)] = value if not isinstance(value, list) else ", ".join(map(str, value))


def request_configuration_reset(phase_number: int | None) -> None:
    st.session_state["pending_configuration_reset"] = "all" if phase_number is None else int(phase_number)


@st.dialog("Configure research phase", width="large")
def phase_configuration_dialog(phase_number: int | None) -> None:
    all_phases = phase_number is None
    selected_phases = range(1, 7) if all_phases else [int(phase_number)]
    reset_pending = st.session_state.pop("pending_configuration_reset", None)
    if reset_pending is not None:
        reset_phases = range(1, 7) if reset_pending == "all" else [int(reset_pending)]
        for phase in reset_phases:
            for name, value in phase_defaults(phase).items():
                st.session_state[_form_key(phase, name)] = value if not isinstance(value, list) else ", ".join(map(str, value))
    st.caption("Research defaults are prefilled. Modified values are validated and saved with this isolated execution.")
    if all_phases:
        st.markdown("### Configure a new run")
        st.caption("Each phase keeps its own settings. All outputs stay together in one isolated run.")
    else:
        phase = PHASES[int(phase_number) - 1]
        st.markdown(f"### Phase {phase_number} · {phase['name']}")
        st.caption(phase["purpose"])
    for phase in selected_phases:
        with st.expander(f"Phase {phase} · {PHASES[phase - 1]['name']}", expanded=not all_phases):
            values = {}
            defaults = phase_defaults(phase)
            fields = list(PHASE_SCHEMAS[phase].items())
            for row_start in range(0, len(fields), 2):
                field_columns = st.columns(2, gap="medium")
                for column, (name, spec) in zip(field_columns, fields[row_start:row_start + 2]):
                    key = _form_key(phase, name)
                    default = defaults[name]
                    if key not in st.session_state:
                        st.session_state[key] = default if not isinstance(default, list) else ", ".join(map(str, default))
                    help_text = (f"{spec['description']} Valid range: {spec['min']} to {spec['max']} "
                                 f"{spec['unit']}. Research default: {default}.")
                    if spec["type"] == "int":
                        column.number_input(spec["label"], min_value=int(spec["min"]), max_value=int(spec["max"]),
                                            step=1, key=key, help=help_text)
                    elif spec["type"] == "float":
                        step = 0.0001 if spec["min"] < 0.001 else (0.001 if spec["min"] < 0.1 else 0.01)
                        column.number_input(spec["label"], min_value=float(spec["min"]), max_value=float(spec["max"]),
                                            step=step, format="%.6f", key=key, help=help_text)
                    else:
                        column.text_input(f"{spec['label']} ({spec['unit']})", key=key,
                                          help=help_text + " Enter comma-separated numeric values in ascending order.")
                    values[name] = st.session_state[key]
            try:
                normalized = {name: ([float(v.strip()) for v in value.split(",") if v.strip()]
                                     if PHASE_SCHEMAS[phase][name]["type"] == "float_list" else value)
                              for name, value in values.items()}
                normalized = validate_phase_config(phase, normalized)
                changed = changed_parameters(phase, normalized)
                st.caption("Modified from research defaults: " + (", ".join(changed) if changed else "None"))
                with st.expander("Configuration preview", expanded=False):
                    st.json({"phase": phase, "values": normalized,
                             "changed_parameters": changed}, expanded=True)
            except (TypeError, ValueError) as error:
                st.error(f"Configuration validation: {error}")

    with st.container(key="phase_configuration_footer"):
        controls = st.columns(3)
        controls[0].button("Reset to research defaults", key="reset_phase_config",
                           on_click=request_configuration_reset, args=(phase_number,))
        if controls[1].button("Cancel", key="cancel_phase_config"):
            st.session_state.pop("pending_phase_configuration", None)
            st.rerun()
        submit_label = "Configure and Run All Phases" if all_phases else f"Run Phase {phase_number}"
        submitted = controls[2].button(submit_label, type="primary", key="submit_phase_config")
    if not submitted:
        return
    try:
        configurations = defaults_for_all_phases() if all_phases else read_run_configuration(
            Path(st.session_state["pending_configuration_run_dir"])
        ) if st.session_state.get("pending_configuration_run_dir") else defaults_for_all_phases()
        for phase in selected_phases:
            values = {name: st.session_state[_form_key(phase, name)] for name in PHASE_SCHEMAS[phase]}
            for name, spec in PHASE_SCHEMAS[phase].items():
                if spec["type"] == "float_list":
                    try:
                        values[name] = [float(value.strip()) for value in values[name].split(",") if value.strip()]
                    except ValueError as error:
                        raise ValueError(f"{name} must contain comma-separated numbers") from error
            configurations[f"phase{phase}"] = validate_phase_config(phase, values)
    except (TypeError, ValueError) as error:
        st.error(f"Cannot start: {error}")
        return
    if all_phases:
        run_dir = start_new_replay()
    else:
        raw_run_dir = st.session_state.get("pending_configuration_run_dir", "")
        run_dir = Path(raw_run_dir) if raw_run_dir else start_new_replay()
    save_run_configuration(run_dir, configurations)
    st.session_state.pop("pending_phase_configuration", None)
    queue_page_navigation(SIMULATION_PAGE)
    st.session_state["pending_phase_execution"] = {
        "phase": None if all_phases else int(phase_number),
        "run_dir": str(run_dir),
    }
    st.rerun()


def clear_diagnostic_page() -> None:
    st.session_state["diagnostic_page"] = DIAGNOSTIC_OPTIONS[0]


def execute_phase(phase_number: int, run_dir: Path, rerun_after: bool = True) -> bool:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        save_run_configuration(run_dir, read_run_configuration(run_dir))
    command = [
        sys.executable, "-u", str(RUNNER), "--run-dir", str(run_dir),
        "--start-phase", str(phase_number), "--end-phase", str(phase_number), "--resume",
        "--config-file", str(config_path),
    ]
    if not run_streamed(command, f"Phase {phase_number}: {PHASES[phase_number - 1]['name']}"):
        return False
    queue_page_navigation(SIMULATION_PAGE, PHASES[phase_number - 1]["page"])
    if phase_number == 6:
        return finalize_replay(run_dir, rerun_after=rerun_after)
    if rerun_after:
        st.rerun()
    return True


def view_phase_results(phase_number: int) -> None:
    queue_page_navigation(SIMULATION_PAGE, PHASES[phase_number - 1]["page"])


def start_and_run_phase_one() -> None:
    run_dir = start_new_replay()
    execute_phase(1, run_dir, rerun_after=False)


def return_to_simulation() -> None:
    queue_page_navigation(SIMULATION_PAGE)


def run_all_phases(configurations: dict[str, dict] | None = None,
                   run_dir: Path | None = None) -> None:
    run_dir = run_dir or start_new_replay()
    configurations = configurations or defaults_for_all_phases()
    config_path = save_run_configuration(run_dir, configurations)
    progress = st.progress(0, text="Preparing a new isolated run")
    stage = st.empty()

    def show_phase_progress(phase_number: int, state: str) -> None:
        phase_name = PHASES[phase_number - 1]["name"]
        completed_before = phase_number - 1
        pending_after = 6 - phase_number
        if state == "running":
            progress.progress(completed_before / 6, text=f"Phase {phase_number} of 6 · {phase_name}")
            stage.info(f"Running Phase {phase_number}: {phase_name} · {completed_before} complete · {pending_after} pending")
        elif state == "completed":
            progress.progress(phase_number / 6, text=f"{phase_number} of 6 phases complete")
            stage.caption(f"Phase {phase_number} complete · {phase_name}")
        else:
            stage.error(f"Phase {phase_number} failed · {phase_name}")

    command = [sys.executable, "-u", str(RUNNER), "--run-dir", str(run_dir),
               "--start-phase", "1", "--end-phase", "6", "--resume",
               "--config-file", str(config_path)]
    if run_streamed(command, "Six-phase simulation", phase_progress=show_phase_progress):
        progress.progress(1.0, text="Six phases complete · publishing evidence")
        stage.success("All six phases completed. Finalizing this isolated run.")
        finalize_replay(run_dir, rerun_after=False)


def completed_interactive_runs(phase: int | None = None) -> list[Path]:
    matches = []
    for path in RUN_ROOT.glob("interactive_replay_*"):
        if not path.is_dir() or not (path / "run_config.json").is_file():
            continue
        registry = load_json(path / "phase_runtime_registry.json") or {}
        completed = {int(row.get("phase", 0)) for row in registry.get("records", [])
                     if row.get("status") == "EXECUTION-COMPLETED" and int(row.get("exit_code", 1)) == 0}
        if phase is None or phase in completed:
            matches.append(path)
    return sorted(matches, key=lambda item: item.stat().st_mtime, reverse=True)


def phase_run_metrics(run_dir: Path, phase: int) -> dict:
    data_dir = run_dir / "do_fmbs_data"
    if phase == 1:
        payload = load_json(data_dir / "phase1_metadata.json") or {}
        return {"face_samples": payload.get("face", {}).get("total"),
                "fingerprint_samples": payload.get("fingerprint", {}).get("total"),
                "fingerprint_identities": payload.get("fingerprint", {}).get("identities"),
                "impressions_per_identity": payload.get("fingerprint", {}).get("impressions_per_identity")}
    if phase == 2:
        payload = load_json(data_dir / "phase2_metrics.json") or {}
        return {"rounds": len(payload.get("history", {}).get("round", [])),
                "face_accuracy": payload.get("face_classification", {}).get("accuracy"),
                "fingerprint_accuracy": payload.get("fingerprint_classification", {}).get("accuracy"),
                "training_parameters": payload.get("training_parameters")}
    if phase == 3:
        payload = load_json(data_dir / "phase3_attack_results.json") or {}
        return {"face_attack_samples": payload.get("parameters", {}).get("face_attack_samples"),
                "attack_settings": payload.get("parameters"),
                "membership_accuracy": payload.get("membership_inference", {}).get("attack_accuracy"),
                "evasion_rows": len(payload.get("evasion", []))}
    if phase == 4:
        payload = load_json(data_dir / "phase4_defense_results.json") or {}
        return {"defense_conditions": list(payload.get("comparison", {})),
                "parameters": payload.get("parameters"),
                "face_accuracy": {name: item.get("face_classification", {}).get("accuracy")
                                  for name, item in payload.get("comparison", {}).items()}}
    if phase == 5:
        payload = load_json(data_dir / "phase5_validation.json") or {}
        return {"evaluation": payload.get("evaluation"),
                "bootstrap_repetitions": payload.get("bootstrap_repetitions"),
                "face_accuracy": payload.get("face", {}).get("classification", {}).get("accuracy"),
                "fingerprint_accuracy": payload.get("fingerprint", {}).get("classification", {}).get("accuracy")}
    payload = load_json(run_dir / "simulation_evidence" / "constrained_dynamic_model" / "model_fit.json") or {}
    schedule_path = run_dir / "simulation_evidence" / "constrained_dynamic_model" / "dynamic_optimization_schedule.csv"
    schedule_count = len(pd.read_csv(schedule_path)) if schedule_path.is_file() else 0
    return {"controller_rounds": schedule_count, "observations": payload.get("observations"),
            "candidate_policies_per_round": payload.get("optimization_settings", {}).get("candidate_policies_per_round"),
            "r_squared": payload.get("r_squared"),
            "optimization_settings": payload.get("optimization_settings")}


def render_run_comparison() -> None:
    with st.expander("Compare interactive executions", expanded=True):
        if len(completed_interactive_runs()) < 2:
            st.info("Complete at least two interactive runs containing the same phase to compare actual configurations and outputs.")
            return
        phase_options = [phase for phase in range(1, 7) if len(completed_interactive_runs(phase)) >= 2]
        if not phase_options:
            st.info("No phase has two completed interactive executions yet.")
            return
        phase = st.selectbox("Phase to compare", phase_options,
                             format_func=lambda value: f"Phase {value} · {PHASES[value - 1]['name']}")
        candidates = completed_interactive_runs(phase)
        options = [str(path) for path in candidates]
        selected = st.multiselect("Select two or more completed runs", options,
                                  default=options[:2], format_func=lambda value: Path(value).name)
        if len(selected) < 2:
            st.info("Select at least two runs for comparison.")
            return
        rows = []
        for raw_path in selected:
            path = Path(raw_path)
            config = load_json(path / f"phase{phase}_config.json") or {}
            registry = load_json(path / "phase_runtime_registry.json") or {}
            record = next((row for row in registry.get("records", []) if int(row.get("phase", 0)) == phase), {})
            rows.append({"Run ID": path.name, "Timestamp": record.get("start_timestamp"),
                         "Status": record.get("status", config.get("status")),
                         "Changed parameters": ", ".join(config.get("changed_parameters", [])) or "None",
                         "Configuration": json.dumps(config.get("configuration", {}), sort_keys=True),
                         "Metrics from artifacts": json.dumps(phase_run_metrics(path, phase), sort_keys=True)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Differences are descriptive; they do not establish causality unless runs are controlled and stochastic seeds are considered.")


def render_simulation_page(records: list[dict], run_dir: Path | None, manifest: dict | None) -> None:
    st.title("Simulation")
    st.caption("Run all six phases in a new isolated run, or continue one phase at a time with its prerequisites enforced.")
    if run_dir:
        st.info(f"Current run: `{run_dir.name}`. A new full run creates a different run ID and leaves this run and all accepted results intact.")
        st.button("Run All Phases · New Run", type="primary", key="run_all_phases",
                  on_click=open_phase_configuration, args=(None,))
        if st.button("Start a new run", key="new_interactive_run"):
            start_new_replay()
            st.rerun()
    else:
        st.info("No interactive run is selected. A run started here receives its own folder, configuration, seeds, and phase logs.")
        all_col, one_col = st.columns(2)
        all_col.button("Run All Phases · New Run", type="primary", key="run_all_phases_empty",
                       on_click=open_phase_configuration, args=(None,))
        if one_col.button("Start a run for individual phases", key="new_interactive_run_empty"):
            start_new_replay()
            st.rerun()

    done = completed_phases(records)
    failed = {int(row.get("phase", 0)) for row in records if int(row.get("exit_code") or 0) != 0}
    if run_dir:
        elapsed = sum(float(row.get("elapsed_seconds", 0)) for row in records)
        cards = st.columns(3)
        cards[0].metric("Current run", "In progress" if len(done) < 6 else "Completed")
        cards[1].metric("Phases complete", f"{len(done)} / 6")
        cards[2].metric("Recorded phase time", f"{elapsed:.1f} s")
        st.caption(f"Run ID: `{run_dir.name}`")
        st.progress(len(done) / 6, text=f"Workflow stages · {len(done)} of 6 complete")
    elif manifest:
        st.caption(f"Accepted results are available in Cross-Phase Analysis · `{manifest.get('replay_id', 'saved run')}`")

    st.subheader("Phase controls and results")
    for offset in range(0, 6, 3):
        columns = st.columns(3)
        for column, phase in zip(columns, PHASES[offset:offset + 3]):
            number = phase["number"]
            phase_done = number in done
            prerequisite_met = number == 1 or all(index in done for index in range(1, number))
            status_text = "Complete" if phase_done else ("Error" if number in failed else ("Ready" if prerequisite_met else "Waiting"))
            column.markdown(
                f'<div class="phase-card"><div class="eyebrow">Phase {number} · {status_text}</div>'
                f'<h4>{phase["name"]}</h4><p>{phase["purpose"]}</p></div>',
                unsafe_allow_html=True,
            )
            if phase_done:
                column.button("View Results", key=f"view_phase_{number}",
                              on_click=view_phase_results, args=(number,), use_container_width=True)
            elif run_dir and prerequisite_met:
                column.button(f"Run Phase {number}", key=f"run_phase_{number}",
                              on_click=open_phase_configuration, args=(number, run_dir), type="primary",
                              use_container_width=True)
            elif not run_dir and number == 1:
                column.button("Start run · Phase 1", key="run_phase_1_empty", type="primary",
                              on_click=open_phase_configuration, args=(1, None))
            elif not run_dir:
                column.button("Waiting for Phase 1", key=f"phase_waiting_{number}_without_run",
                              disabled=True, use_container_width=True)
                column.caption("Start a new run and complete Phase 1 first.")
            else:
                column.button("Waiting for prerequisite", key=f"phase_waiting_{number}", disabled=True,
                              use_container_width=True)
                if not prerequisite_met:
                    column.caption(f"Available after Phase {number - 1} completes.")
    render_run_comparison()
    if len(done) == 6 and manifest:
        st.success("This interactive run is complete. Use the View Results buttons above to inspect its saved phase outputs; Cross-Phase Analysis presents the accepted research runs.")


def finalize_replay(run_dir: Path, rerun_after: bool = True) -> bool:
    manifest_path = run_dir / "interactive_experiment_manifest.json"
    existing_manifest = load_json(manifest_path)
    if existing_manifest and existing_manifest.get("status") == "COMPLETED-EXPLORATORY-NOT-AUTHORITATIVE":
        st.success("All six exploratory phases completed. Configuration and artifact hashes are recorded separately from accepted research evidence.")
        if rerun_after:
            st.rerun()
        return True
    registry = load_json(run_dir / "phase_runtime_registry.json") or {}
    records = registry.get("records", [])
    complete = {int(row.get("phase", 0)) for row in records
                if row.get("status") == "EXECUTION-COMPLETED" and int(row.get("exit_code", 1)) == 0}
    if complete != set(range(1, 7)):
        st.error("The exploratory run manifest was not written because all six phases have not completed successfully.")
        return False
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "interactive_experiment_manifest.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append({"path": path.relative_to(run_dir).as_posix(), "bytes": path.stat().st_size,
                          "sha256": digest})
    payload = {"status": "COMPLETED-EXPLORATORY-NOT-AUTHORITATIVE",
               "run_id": run_dir.name, "created_at": datetime.now(ZoneInfo("Africa/Nairobi")).isoformat(),
               "configuration_version": CONFIG_VERSION,
               "scientific_status": "Exploratory configurable execution; not an accepted thesis run or independent scientific audit.",
               "configuration": load_json(run_dir / "run_config.json"),
               "phase_records": records, "artifacts": artifacts,
               "validation": {"all_six_phases_completed": True,
                              "artifact_inventory_with_sha256": True,
                              "independent_authoritative_audit": False}}
    (run_dir / "interactive_experiment_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    st.success("All six exploratory phases completed. Configuration and artifact hashes are recorded separately from accepted research evidence.")
    if rerun_after:
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
        st.info("This interactive replay uses its own preparation workflow. For the Chapter Four reproduction and accepted persisted results, see the Phase 1 research page and Research Scope & Limitations.")
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
    st.markdown('<div class="hero"><div class="eyebrow">Federated multimodal biometric research</div><h1>DO-FMBS</h1><p>Data preparation, federated learning, security evaluation, defense, validation, and dynamic optimization.</p></div>', unsafe_allow_html=True)
    done = completed_phases(records)
    if manifest:
        state_text = "Completed"
        replay_id = manifest["replay_id"]
        runtime = manifest.get("runtime", {}).get("total_seconds", 0.0)
    elif run_dir:
        state_text = "In progress"
        replay_id = run_dir.name
        runtime = sum(float(row.get("elapsed_seconds", 0)) for row in records)
    else:
        state_text = "Not started"
        replay_id = "No interactive replay selected"
        runtime = 0.0
    metric_card_row([("Run status", state_text), ("Phases completed", f"{len(done)} / 6"),
                     ("Recorded phase time", f"{runtime:.1f} s")])
    st.caption(f"Interactive run ID: `{replay_id}`")
    if run_dir or manifest:
        st.progress(len(done) / 6, text=f"Interactive workflow · {len(done)} of 6 phases completed")
    else:
        st.caption("Accepted results are available under Cross-Phase Analysis. Use Simulation to start a new isolated run.")
    st.markdown("### Research workflow")
    overview_text = {
        1: "Prepare face and synthetic fingerprint data, preprocessing, and client partitions.",
        2: "Train and aggregate the two modality-specific identity models across five clients.",
        3: "Evaluate face-branch evasion, membership inference, and label-flipping scenarios.",
        4: "Compare gradient-noise, adversarial, and combined defense conditions.",
        5: "Assess classification, verification, robustness, and statistical comparisons.",
        6: "Fit the constrained response model and inspect its twelve-round policy schedule.",
    }
    for offset in range(0, 6, 3):
        cols = st.columns(3)
        for col, phase in zip(cols, PHASES[offset:offset + 3]):
            number = phase["number"]
            failed = next((row for row in records if int(row.get("phase", 0)) == number and int(row.get("exit_code") or 0) != 0), None)
            status_text = "Error" if failed else ("Completed" if number in done else ("Queued" if number == (len(done) + 1) else "Waiting"))
            col.markdown(
                f'<div class="phase-card"><div class="eyebrow">Phase {number} · {status_text}</div>'
                f'<h4>{phase["name"]}</h4><p>{overview_text[number]}</p></div>',
                unsafe_allow_html=True,
            )
    if run_dir is None and not manifest:
        st.info("Open Simulation to start one phase or run the complete six-phase workflow in a new isolated run.")
    elif len(done) == 6 and manifest:
        st.info("This interactive replay is complete. Its execution record is available in Technical diagnostics.")
    elif done:
        next_phase = min(max(done) + 1, 6)
        st.info(f"Next step: {PHASES[next_phase - 1]['name']}.")
    else:
        st.info("A new run folder is ready. Open Technical diagnostics → Run Phase 1 to begin the sequential workflow.")

    st.markdown("### Research snapshots")
    phase2_dir = RUN_ROOT / ACCEPTED_RUNS[2][0]
    phase6_dir = RUN_ROOT / ACCEPTED_RUNS[6][0]
    try:
        round_metrics = pd.read_csv(phase2_dir / "round_metrics.csv")
        schedule = pd.read_csv(phase6_dir / "dynamic_schedule_detailed.csv")
        chart_left, chart_right = st.columns(2)
        accuracy = round_metrics.copy()
        accuracy[["test_face_output_accuracy", "test_finger_output_accuracy"]] *= 100
        chart_left.subheader("Federated test accuracy")
        chart_left.line_chart(accuracy, x="round", y=["test_face_output_accuracy", "test_finger_output_accuracy"],
                              x_label="Federated round", y_label="Accuracy (%)",
                              color=["#285875", "#167d83"], height=270)
        chart_right.subheader("Phase 6 predicted response")
        chart_right.line_chart(schedule, x="round", y="predicted_Y", x_label="Controller round",
                              y_label="Predicted response Y", color="#167d83", height=270)
        st.caption("Both charts are drawn from the saved accepted phase runs; they do not trigger training or optimization.")
    except (OSError, ValueError, KeyError) as error:
        st.info("Research snapshots are not available from the saved phase files.")
        with st.expander("Details"):
            st.caption(str(error))

    st.button("View Cross-Phase Analysis", key="overview_cross_phase",
              on_click=lambda: (st.session_state.update({"selected_page": RESEARCH_COMPARISON_PAGE,
                                                        "diagnostic_page": DIAGNOSTIC_OPTIONS[0]})))


def render_phase_page(phase: dict, manifest: dict | None, run_dir: Path | None,
                      records: list[dict], completed: set[int]) -> None:
    st.button("Back to Simulation", key=f"back_to_simulation_{phase['number']}",
              on_click=return_to_simulation)
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
    phase_status = "Completed" if done else ("Error" if record and int(record.get("exit_code") or 0) != 0 else "Waiting")
    columns[3].metric("Status", phase_status)

    run_config = manifest.get("phase_parameters", {}) if manifest else {}
    config_key = f"phase{number}"
    configuration = run_config.get(config_key)
    if configuration:
        with st.expander("Recorded configuration", expanded=False):
            st.json(configuration, expanded=False)
    actual_config = load_json(run_dir / f"phase{number}_config.json") if run_dir else None
    if actual_config:
        with st.expander("Submitted configuration and backend provenance", expanded=True):
            st.write(f"Status: **{actual_config.get('status', 'recorded')}** · schema {actual_config.get('configuration_version', 'unknown')}")
            st.write("Changed parameters: " + (", ".join(actual_config.get("changed_parameters", [])) or "None; research defaults used"))
            st.json(actual_config.get("configuration", {}), expanded=True)

    phase_is_next = number == 1 or all(index in completed for index in range(1, number))
    if run_dir is None:
        if number == 1:
            st.info("No fresh interactive run is active. Start a new run from Simulation, then trigger Phase 1 to begin.")
            if st.button("Start a new run for Phase 1", type="primary", key="phase_1_start_fresh_run"):
                start_new_replay()
                st.rerun()
        else:
            required = ", ".join(f"Phase {index}" for index in range(1, number))
            st.info(f"No fresh interactive run is active. Start one from Simulation and complete {required} in order before Phase {number}.")
    elif phase_is_next:
        if done:
            st.success(f"Phase {number} completed in run `{run_dir.name}`.")
            st.caption("This run is retained as recorded. Start a new run to execute the workflow again.")
        else:
            st.caption("Results are saved to this run's isolated folder.")
            st.button(f"Configure & Run Phase {number}", type="primary",
                      on_click=open_phase_configuration, args=(number, run_dir))
        if number == 6 and done and not (run_dir / "interactive_experiment_manifest.json").is_file():
            st.error("Phase 6 completed, but the exploratory run manifest is missing.")
            if st.button("Rebuild exploratory run manifest", type="primary"):
                finalize_replay(run_dir)
    else:
        waiting_for = next((index for index in range(1, number) if index not in completed), None)
        st.info(f"Trigger and complete Phase {waiting_for} in this run before running Phase {number}.")

    if record:
        st.caption(f"Last run: {record.get('start_timestamp', 'time unavailable')} · {float(record.get('elapsed_seconds', 0)):.1f} sec · exit {record.get('exit_code')}")
        log = run_dir / record.get("log_path", f"phase{number}_console.log") if run_dir else None
        if log and log.exists():
            with st.expander("Technical execution log", expanded=True):
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
        elif number == 6:
            output_dir = run_dir / "simulation_evidence" / "constrained_dynamic_model"
            fit = load_json(output_dir / "model_fit.json")
            schedule_path = output_dir / "dynamic_optimization_schedule.csv"
            if fit and schedule_path.is_file():
                schedule = pd.read_csv(schedule_path)
                metric_card_row([("Simulation observations", str(fit.get("observations", "—"))),
                                 ("Controller rounds", str(len(schedule))),
                                 ("Candidate policies / round", str(fit.get("optimization_settings", {}).get("candidate_policies_per_round", "—"))),
                                 ("R²", f"{float(fit.get('r_squared', 0.0)):.4f}")])
                st.line_chart(schedule.set_index("round")[["predicted_Y", "control_cost"]], height=300)
                st.dataframe(schedule, use_container_width=True, hide_index=True)
                st.caption("Synthetic constrained-response simulation; values are generated by this exploratory run's saved configuration, not by the federated model.")
            else:
                st.info("Optimization outputs are not available for this run.")
        else:
            st.info("Optimization outputs are saved with this run's simulation evidence.")


def render_evidence(manifest: dict | None, records: list[dict], manifest_path: Path | None) -> None:
    st.title("Interactive run diagnostics")
    st.caption("Detailed execution records and artifact provenance for the sequential interactive run.")
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


def render_research_scope() -> None:
    st.title("Research Scope & Limitations")
    st.caption("Interpretation boundaries for the experiments and results shown in this application.")
    st.subheader("Research scope")
    st.markdown("""
    - Face and fingerprint branches use independent identity populations; paired rows synchronize computation and do not denote matched people.
    - Fingerprint images are generated synthetic ridge impressions, not samples from a public or operational fingerprint database.
    - Phase 3 evasion and Phase 5 robustness experiments perturb face features; they do not establish fingerprint attack robustness.
    - The gradient-noise defense is DP-inspired. No formal differential-privacy guarantee or privacy budget is reported.
    - Phase 1 reproduces the Chapter Four order in which `StandardScaler` is fitted before the train/test split; held-out distribution statistics therefore enter preprocessing.
    - Phase 6 uses controlled synthetic observations. Its predicted response and policy schedule are simulation outputs, not measurements from a live federation.
    - Verification metrics describe claimed-identity matching under the recorded protocol; they do not measure liveness or presentation-attack detection.
    """)
    st.subheader("Interpretation")
    st.write("Read classification, verification, attack, defense, and optimization results as separate outcomes with their recorded denominators and evaluation protocols. Each phase page provides the relevant configuration and saved evidence.")


def render_phase1_rebuild() -> None:
    st.title("Phase 1 · Chapter Four (Authoritative)")
    st.caption("Face and synthetic fingerprint datasets, preprocessing, verification partitions, and identity-exclusive client allocation.")
    runs = sorted((path for pattern in ("phase1_reimplementation_*", "phase1_final_candidate_*")
                   for path in RUN_ROOT.glob(pattern)
                   if (path / "phase1_manifest.json").is_file()),
                  key=lambda path: path.stat().st_mtime, reverse=True)
    labels = [path.name for path in runs]
    current = st.session_state.get("phase1_rebuild_dir")
    if current and Path(current).is_dir() and Path(current).name not in labels:
        labels.insert(0, Path(current).name)
    if labels:
        selected = st.selectbox("Saved Phase 1 replay", labels,
                                index=labels.index(Path(current).name) if current and Path(current).name in labels else 0)
        data_dir = RUN_ROOT / selected
    else:
        data_dir = Path(current) if current else None

    if st.button("Start final-candidate Phase 1 replay", type="primary", key="start_source_faithful_phase1"):
        stamp = datetime.now(ZoneInfo("Africa/Nairobi")).strftime("%Y%m%d_%H%M%S")
        output = RUN_ROOT / f"phase1_final_candidate_{stamp}"
        suffix = 1
        while output.exists():
            output = RUN_ROOT / f"phase1_final_candidate_{stamp}_{suffix}"
            suffix += 1
        if run_streamed([sys.executable, "-u", str(ROOT / "phase1_reimplementation.py"),
                           "--output-dir", str(output)], "Phase 1 · Data preparation"):
            st.session_state["phase1_rebuild_dir"] = str(output)
            st.rerun()
        return

    if data_dir is None or not (data_dir / "phase1_manifest.json").is_file():
        st.info("No saved Phase 1 run is available yet.")
        return

    result = inspect_rebuilt_phase1(data_dir)
    manifest = result.get("manifest", {})
    metadata = manifest.get("metadata", {})
    config = manifest.get("configuration", {})
    seed_rows = []
    for source_name, details in config.get("seeds", {}).items():
        seed_rows.append({"Random source": source_name,
                          "Seed": details.get("value", details.get("base")),
                          "Purpose": details.get("purpose"),
                          "Derivation": details.get("formula", "")})
    with st.expander("Configuration and provenance"):
        st.write(f"Run `{manifest.get('replay_id', 'unknown')}` · source document SHA-256 `{metadata.get('source_document_sha256', 'not recorded')}`")
        st.dataframe(pd.DataFrame(seed_rows), use_container_width=True, hide_index=True)
        st.json(config.get("lfw_acquisition", {}), expanded=False)
        st.json(config.get("generator_provenance", {}), expanded=False)
        st.caption(f"Methodology clarification: {config.get('methodology_clarification', {}).get('document', 'not recorded')}")
        st.json(config.get("fingerprint_generation", {}), expanded=False)

    rows = []
    for modality in ("face", "fingerprint"):
        item = metadata.get(modality, {})
        if item:
            rows.append({"Modality": modality.title(), "Source": item.get("source"),
                         "Samples": item.get("samples"), "Identities": item.get("identities"),
                         "Image dimensions": " × ".join(map(str, item.get("image_shape", []))),
                         "Features": item.get("features"), "Train": item.get("train"), "Test": item.get("test")})
    st.subheader("Persisted datasets and splits")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Preprocessing")
    st.caption(metadata.get("preprocessing", "Preprocessing metadata is unavailable."))

    st.subheader("Verification partitions")
    verification_rows = []
    for modality in ("face", "fingerprint"):
        item = metadata.get(modality, {})
        if modality == "face":
            verification_rows.append({"Modality": "Face", "Eligible identities": item.get("identities"),
                                      "Calibration genuine": item.get("verification_calibration_genuine"),
                                      "Evaluation genuine": item.get("verification_evaluation_genuine"),
                                      "Evaluation impostor claims": item.get("evaluation_impostor_claims")})
        else:
            verification_rows.append({"Modality": "Fingerprint", "Eligible identities": item.get("eligible_verification_identities"),
                                      "Calibration genuine": item.get("calibration_genuine"),
                                      "Evaluation genuine": item.get("evaluation_genuine"),
                                      "Calibration impostor claims": item.get("calibration_impostor_claims"),
                                      "Evaluation impostor claims": item.get("evaluation_impostor_claims")})
    st.dataframe(pd.DataFrame(verification_rows), use_container_width=True, hide_index=True)

    st.subheader("Persistent identity-exclusive clients")
    client_frame = pd.DataFrame(result.get("clients", []))
    if not client_frame.empty:
        st.bar_chart(client_frame, x="Client", y=["Face samples", "Fingerprint samples"],
                     x_label="Client", y_label="Training samples", height=300,
                     color=["#285875", "#167d83"])
        st.dataframe(client_frame, use_container_width=True, hide_index=True)
    else:
        st.warning("Client evidence is unavailable or failed validation.")

    with st.expander("Identity mappings and sample images"):
        try:
            with np.load(data_dir / "face_data.npz", allow_pickle=False) as face_data:
                face_mapping = pd.DataFrame({
                    "Label": np.arange(len(face_data["target_names"]), dtype=int),
                    "LFW identity": face_data["target_names"].astype(str),
                })
            with np.load(data_dir / "fingerprint_data.npz", allow_pickle=False) as finger_data:
                finger_labels = np.unique(np.concatenate((finger_data["y_train"], finger_data["y_test"])))
                finger_preview = finger_data["images_train"][0]
            left, right = st.columns(2)
            left.markdown("**Face identity → label**")
            left.dataframe(face_mapping, use_container_width=True, hide_index=True, height=320)
            right.markdown("**Synthetic fingerprint identity → label**")
            right.dataframe(pd.DataFrame({"Identity ID": finger_labels.astype(int),
                                          "Label": finger_labels.astype(int)}),
                            use_container_width=True, hide_index=True, height=320)
            face_data = np.load(data_dir / "face_data.npz", allow_pickle=False)
            image_left, image_right = st.columns(2)
            image_left.image(face_data["images_train"][0], caption="Face training sample", clamp=True)
            image_right.image(finger_preview, caption="Synthetic fingerprint training sample", clamp=True)
            face_data.close()
        except (OSError, KeyError, ValueError) as error:
            st.error("Saved identity mappings or sample images could not be loaded.")
            with st.expander("Technical details"):
                st.code(str(error))
    with st.expander("Verification and artifact details"):
        if result.get("status") != "PASS":
            st.error("One or more persisted data checks need review.")
        st.dataframe(pd.DataFrame(result.get("checks", [])), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(manifest.get("artifacts", [])), use_container_width=True, hide_index=True)


def render_phase2_authoritative() -> None:
    st.title("Phase 2 · Federated baseline")
    st.caption("Train face and fingerprint identity classifiers across five federated clients and inspect their round-by-round convergence.")
    phase1_runs = sorted(
        (path for path in RUN_ROOT.glob("phase1_final_candidate_*")
         if (path / "phase1_manifest.json").is_file()
         and (path / "phase1_verification_report.json").is_file()),
        key=lambda path: path.stat().st_mtime, reverse=True,
    )
    if not phase1_runs:
        st.info("A verified Phase 1 dataset run is required before federated training.")
        return

    selected_phase1 = st.selectbox(
        "Phase 1 dataset run", [path.name for path in phase1_runs],
        index=next((i for i, path in enumerate(phase1_runs)
                    if path.name == "phase1_final_candidate_20261123_182623"), 0),
        key="phase2_source_phase1",
    )
    phase1_dir = RUN_ROOT / selected_phase1
    phase1_manifest = load_json(phase1_dir / "phase1_manifest.json") or {}
    phase1_report = load_json(phase1_dir / "phase1_verification_report.json") or {}
    source_ok = (phase1_report.get("status") == "PASS"
                 and phase1_manifest.get("verification", {}).get("status") == "PASS")
    st.caption(f"Dataset source: `{selected_phase1}` · {'verified inputs' if source_ok else 'input verification needs review'}")
    if not source_ok:
        st.error("This Phase 1 run has not passed its saved verification; training is unavailable.")

    phase2_runs = sorted(
        (path for path in RUN_ROOT.glob("phase2_authoritative_*")
         if (path / "phase2_manifest.json").is_file()),
        key=lambda path: path.stat().st_mtime, reverse=True,
    )
    if source_ok and st.button("Run Phase 2", type="primary", key="start_authoritative_phase2"):
        stamp = datetime.now(ZoneInfo("Africa/Nairobi")).strftime("%Y%m%d_%H%M%S")
        output_dir = RUN_ROOT / f"phase2_authoritative_{stamp}"
        suffix = 1
        while output_dir.exists():
            output_dir = RUN_ROOT / f"phase2_authoritative_{stamp}_{suffix}"
            suffix += 1
        command = [sys.executable, "-u", str(ROOT / "phase2_authoritative_training.py"),
                   "--phase1-dir", str(phase1_dir), "--output-dir", str(output_dir)]
        if run_streamed(command, "Phase 2 · Federated baseline"):
            st.session_state["phase2_authoritative_run"] = output_dir.name
            st.rerun()
        return

    if not phase2_runs:
        st.info("No saved Phase 2 run is available yet. Running this phase creates a new timestamped run.")
        return
    chosen_run = st.selectbox(
        "Experiment run", [path.name for path in phase2_runs],
        index=next((i for i, path in enumerate(phase2_runs)
                    if path.name == st.session_state.get("phase2_authoritative_run")), 0),
        key="phase2_saved_run",
    )
    run_path = RUN_ROOT / chosen_run
    metrics = load_json(run_path / "phase2_metrics.json")
    config = load_json(run_path / "phase2_config.json")
    verification = load_json(run_path / "phase2_verification_report.json")
    run_manifest = load_json(run_path / "phase2_manifest.json") or {}
    if not metrics or not config or not verification:
        st.error("This saved Phase 2 run is missing required persisted evidence.")
        return

    st.caption(f"Run `{chosen_run}` · dataset source `{metrics.get('phase1_replay_id', 'unknown')}`")
    if verification.get("status") != "PASS":
        st.error("The saved training evidence needs review before interpretation.")
    with st.expander("Training configuration"):
        st.json({key: config.get(key) for key in (
            "clients", "rounds", "local_epochs", "batch_size", "learning_rate", "dropout_rate",
            "loss", "loss_weights", "pairing", "random_seeds", "environment", "determinism",
        )}, expanded=False)

    face = metrics.get("face_classification", {})
    finger = metrics.get("fingerprint_classification", {})
    cards = st.columns(4)
    cards[0].metric("Federated clients", str(config.get("clients", "—")))
    cards[1].metric("Communication rounds", str(config.get("rounds", "—")))
    cards[2].metric("Face test accuracy", percent(face.get("accuracy", 0)))
    cards[3].metric("Fingerprint test accuracy", percent(finger.get("accuracy", 0)))
    st.caption("Face and fingerprint results use separate identity label spaces.")

    try:
        round_frame = pd.read_csv(run_path / "round_metrics.csv")
        left, right = st.columns(2)
        left.subheader("Test accuracy by round")
        left.line_chart(round_frame, x="round", y=["test_face_output_accuracy", "test_finger_output_accuracy"],
                        x_label="Federated round", y_label="Accuracy", height=310,
                        color=["#285875", "#167d83"])
        right.subheader("Cross-entropy loss by round")
        right.line_chart(round_frame, x="round", y=["test_face_output_loss", "test_finger_output_loss"],
                         x_label="Federated round", y_label="Cross-entropy loss", height=310,
                         color=["#285875", "#167d83"])
        with st.expander("Round-level training and test values", expanded=False):
            st.dataframe(round_frame, use_container_width=True, hide_index=True)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not read round metrics: {error}")

    st.subheader("Five-client allocation")
    allocation = load_json(run_path / "client_allocation.json") or {}
    client_rows = [{
        "Client": client.get("client_id"),
        "Paired records": client.get("paired_sample_count"),
        "Face identity classes": len(set(client.get("face_labels", []))),
        "Fingerprint identity classes": len(set(client.get("fingerprint_labels", []))),
        "Face label space": "separate 0–61",
        "Fingerprint label space": "separate 0–99",
    } for client in allocation.get("clients", [])]
    st.dataframe(pd.DataFrame(client_rows), use_container_width=True, hide_index=True)
    with st.expander("Local client training histories"):
        histories = load_json(run_path / "client_training_histories.json") or []
        rounds_available = sorted({int(item["round"]) for item in histories})
        if rounds_available:
            selected_round = st.select_slider("Federated round", options=rounds_available,
                                              value=rounds_available[-1], key="phase2_history_round")
            local_rows = []
            for item in histories:
                if int(item["round"]) == selected_round:
                    last_epoch = {key: values[-1] for key, values in item.get("epoch_history", {}).items() if values}
                    local_rows.append({"Round": selected_round, "Client": item["client_id"],
                                       "Samples": item["sample_count"], **last_epoch})
            st.dataframe(pd.DataFrame(local_rows), use_container_width=True, hide_index=True)

    st.subheader("Global model evaluation")
    st.dataframe(pd.DataFrame([
        {"Modality": "Face", "Correct / test rows": f"{face.get('correct', 0)} / {face.get('samples', 0)}",
         "Classes": face.get("classes"), "Accuracy": percent(face.get("accuracy", 0))},
        {"Modality": "Fingerprint", "Correct / test rows": f"{finger.get('correct', 0)} / {finger.get('samples', 0)}",
         "Classes": finger.get("classes"), "Accuracy": percent(finger.get("accuracy", 0))},
    ]), use_container_width=True, hide_index=True)
    with st.expander("Evidence integrity and verification"):
        st.json(verification, expanded=False)
        artifact_rows = []
        for artifact in run_manifest.get("artifacts", []):
            path = run_path / artifact["path"]
            try:
                info = path.stat()
                actual = cached_sha256(str(path), info.st_mtime_ns, info.st_size)
                status = "PASS" if actual == artifact.get("sha256") and info.st_size == artifact.get("bytes") else "FAIL"
            except OSError:
                status = "FAIL"
            artifact_rows.append({"Artifact": artifact["path"], "Current check": status})
        st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)


manifest, manifest_path = current_manifest()
run_dir = selected_run_dir()
records, _registry_path = registry_for(run_dir, manifest)
done = completed_phases(records)
pending_navigation = st.session_state.pop("pending_page_navigation", None)
if pending_navigation:
    st.session_state["selected_page"] = pending_navigation["selected_page"]
    st.session_state["diagnostic_page"] = pending_navigation["diagnostic_page"]

with st.sidebar:
    if st.session_state.get("selected_page") not in PAGE_OPTIONS:
        st.session_state.pop("selected_page", None)
    if st.session_state.get("diagnostic_page") not in DIAGNOSTIC_OPTIONS:
        st.session_state["diagnostic_page"] = DIAGNOSTIC_OPTIONS[0]
    st.markdown("<div style='padding:.4rem 0 1rem'><div style='font-size:1.2rem;font-weight:800;color:#17364e'>DO-FMBS</div><div style='font-size:.82rem;color:#526574'>Federated biometric research</div></div>", unsafe_allow_html=True)
    if run_dir:
        st.caption(f"Interactive run · {run_dir.name}")
        st.progress(len(done) / 6, text=f"Workflow · {len(done)} of 6 phases")
    elif manifest:
        st.caption(f"Interactive run · {manifest.get('replay_id', 'completed')}")
        st.progress(len(done) / 6, text=f"Workflow · {len(done)} of 6 phases")
    else:
        st.caption("Saved results are available in Cross-Phase Analysis.")
    st.radio("Research navigation", PAGE_OPTIONS, key="selected_page",
             on_change=clear_diagnostic_page, label_visibility="visible")
    st.button("New Interactive Run", type="primary", use_container_width=True,
              on_click=start_new_replay)
    with st.expander("Technical diagnostics", expanded=False):
        st.selectbox("Run or inspect a phase", DIAGNOSTIC_OPTIONS,
                     key="diagnostic_page", label_visibility="visible")
        st.caption("Execution records and logs for interactive runs.")
    st.markdown("<div style='font-size:.76rem;color:#526574;padding-top:.8rem'>Lydiah Moraa Machora<br>JOOUST · Master's research</div>", unsafe_allow_html=True)

diagnostic_page = st.session_state.get("diagnostic_page", DIAGNOSTIC_OPTIONS[0])
selected_page = diagnostic_page if diagnostic_page != DIAGNOSTIC_OPTIONS[0] else st.session_state.get("selected_page", "Overview")
if selected_page == "Overview":
    render_overview(manifest, records, run_dir)
elif selected_page == SIMULATION_PAGE:
    render_simulation_page(records, run_dir, manifest)
elif selected_page == RESEARCH_SCOPE_PAGE:
    render_research_scope()
elif selected_page == "Evidence & provenance":
    render_evidence(manifest, records, manifest_path)
elif selected_page == RESEARCH_COMPARISON_PAGE:
    render_research_consolidation(RUN_ROOT, ROOT / "docs" / "FINAL_DO_FMBS_RESEARCH_SIMULATION_AND_DEMONSTRATION_REPORT.md")
elif selected_page in {
    PHASE1_REBUILD_PAGE, PHASE2_AUTHORITATIVE_PAGE, PHASE3_AUTHORITATIVE_PAGE,
    PHASE4_AUTHORITATIVE_PAGE, PHASE5_AUTHORITATIVE_PAGE, PHASE6_AUTHORITATIVE_PAGE,
}:
    phase_number = {
        PHASE1_REBUILD_PAGE: 1, PHASE2_AUTHORITATIVE_PAGE: 2,
        PHASE3_AUTHORITATIVE_PAGE: 3, PHASE4_AUTHORITATIVE_PAGE: 4,
        PHASE5_AUTHORITATIVE_PAGE: 5, PHASE6_AUTHORITATIVE_PAGE: 6,
    }[selected_page]
    render_phase_page(PHASES[phase_number - 1], manifest, run_dir, records, done)
else:
    if selected_page in {phase["page"] for phase in PHASES}:
        phase = next(item for item in PHASES if item["page"] == selected_page)
        render_phase_page(phase, manifest, run_dir, records, done)

pending_configuration = st.session_state.get("pending_phase_configuration")
if pending_configuration is not None:
    phase_configuration_dialog(None if pending_configuration == "all" else int(pending_configuration))

pending_execution = st.session_state.pop("pending_phase_execution", None)
if pending_execution:
    execution_run_dir = Path(pending_execution["run_dir"])
    execution_phase = pending_execution["phase"]
    if execution_phase is None:
        st.info(f"Submitted configuration for all six phases · run `{execution_run_dir.name}`. Execution is starting below.")
        run_all_phases(read_run_configuration(execution_run_dir), run_dir=execution_run_dir)
    else:
        phase_name = PHASES[int(execution_phase) - 1]["name"]
        st.info(f"Submitted configuration for Phase {execution_phase} · {phase_name} · run `{execution_run_dir.name}`. Execution is starting below.")
        execute_phase(int(execution_phase), execution_run_dir)

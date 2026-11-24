"""Read-only Streamlit rendering for persisted authoritative Phase 3 evidence."""
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


def _percent(value) -> str:
    return f"{100 * float(value):.2f}%" if value is not None else "n/a"


def render_phase3_authoritative(run_root: Path) -> None:
    st.title("Phase 3 · Attack experiments")
    st.caption("Face-branch evasion, membership inference, and label-flipping experiments against the saved federated baseline.")
    runs = sorted((p for p in run_root.glob("phase3_authoritative_*")
                   if (p / "phase3_manifest.json").is_file()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        st.info("No saved Phase 3 experiment is available yet.")
        return
    names = [p.name for p in runs]
    chosen = st.selectbox("Experiment run", names, key="phase3_authoritative_run")
    run_path = run_root / chosen
    config, metrics = _json(run_path / "phase3_config.json"), _json(run_path / "phase3_metrics.json")
    verification, manifest = (_json(run_path / "phase3_verification_report.json"),
                              _json(run_path / "phase3_manifest.json"))
    if not config or not metrics:
        st.error("This saved Phase 3 run is missing its persisted configuration or metrics.")
        return
    current_verifier = Path(__file__).resolve().with_name("verify_phase3_authoritative_run.py")
    recorded_verifier_hash = config.get("source_sha256", {}).get(current_verifier.name)
    verifier_matches = current_verifier.is_file() and recorded_verifier_hash == _sha(current_verifier)
    st.caption(f"Run `{chosen}` · baseline `{config.get('source_phase2_run', 'not recorded')}` · clean-correct face records {metrics.get('clean_correct_face_denominator', 'n/a')}/100")
    st.caption("Attack inputs: standardized face features. Fingerprint inputs are unchanged.")

    evasion = metrics.get("evasion", [])
    for family, title in (("FGSM", "FGSM evasion · face branch"), ("PGD", "PGD evasion · face branch")):
        st.subheader(title)
        rows = [r for r in evasion if r.get("attack") == family]
        if family == "PGD":
            st.caption("10 fixed updates × alpha 0.01, no random start. The trajectory can move at most 0.10 in L-infinity distance, even when the declared projection bound is 0.20 or 0.30.")
        columns = {
            "Epsilon / projection bound": "epsilon_projection_bound",
            "Clean-correct N": "clean_correct_denominator",
            "Successes": "attack_successes",
            "ASR": "attack_success_rate",
            "Accuracy under attack (N=100)": "accuracy_under_attack_all_100",
            "Max L-infinity change": "perturbation_linf_max",
            "Projection bound satisfied": "projection_bound_satisfied",
        }
        if family == "PGD":
            columns = {**columns, "Step size": "step_size", "Iterations": "iterations"}
        view = [{label: (_percent(row.get(key)) if key in ("attack_success_rate", "accuracy_under_attack_all_100")
                         else row.get(key)) for label, key in columns.items()} for row in rows]
        if rows:
            chart_rows = pd.DataFrame([{"Epsilon": row["epsilon_projection_bound"],
                                        "Attack success rate": row["attack_success_rate"]}
                                       for row in rows])
            st.line_chart(chart_rows, x="Epsilon", y="Attack success rate",
                          x_label="Perturbation bound ε", y_label="Attack success rate",
                          color="#167d83", height=300)
        st.dataframe(pd.DataFrame(view), width="stretch", hide_index=True)

    st.subheader("Membership inference · fixed confidence threshold")
    mia = metrics.get("membership_inference", {})
    st.dataframe(pd.DataFrame([{
        "Threshold": mia.get("threshold"), "Members": mia.get("members"), "Nonmembers": mia.get("nonmembers"),
        "Members called member": f"{mia.get('true_member_detections')}/100",
        "Nonmembers falsely called member": f"{mia.get('false_member_detections')}/100",
        "False negatives": mia.get("false_negatives"), "Attack accuracy": _percent(mia.get("attack_accuracy", 0)),
        "Empirical PA (not a guarantee)": mia.get("privacy_quantity_PA"),
    }]), width="stretch", hide_index=True)

    st.subheader("Label-flipping poisoning proxy · face labels")
    poison = metrics.get("label_flipping", {})
    st.dataframe(pd.DataFrame([{
        "Selected samples": poison.get("clean_sample_count"), "Poison rate": _percent(poison.get("poison_rate", 0)),
        "Labels flipped": poison.get("flipped_labels"), "Retraining epochs": poison.get("epochs"),
        "Clean accuracy": _percent(poison.get("clean_accuracy", 0)),
        "Poisoned accuracy": _percent(poison.get("poisoned_accuracy", 0)),
        "Change (pp)": poison.get("accuracy_change_percentage_points"),
        "Clamped degradation ASR": _percent(poison.get("clamped_degradation_poisoning_asr", 0)),
        "Outcome": poison.get("interpretation"),
    }]), width="stretch", hide_index=True)

    st.subheader("Chapter Four reference context")
    fgsm, pgd_rows = ([r for r in evasion if r.get("attack") == name] for name in ("FGSM", "PGD"))
    comparison = [
        {"Experiment": "FGSM", "Chapter Four reference successes": "5/26, 16/26, 19/26, 25/26, 25/26",
         "Current successes": ", ".join(f"{r['attack_successes']}/{r['clean_correct_denominator']}" for r in fgsm)},
        {"Experiment": "PGD", "Chapter Four reference successes": "6/26, 16/26, 23/26, 23/26, 23/26",
         "Current successes": ", ".join(f"{r['attack_successes']}/{r['clean_correct_denominator']}" for r in pgd_rows)},
        {"Experiment": "Membership", "Chapter Four reference": "101/200 correct; PA 0.01",
         "Current": f"{mia.get('attack_accuracy', 0):.3f} accuracy; PA {mia.get('privacy_quantity_PA', 0):.3f}"},
        {"Experiment": "Label flipping", "Chapter Four reference": "26% → 31%; +5 pp; ASR 0%",
         "Current": f"{_percent(poison.get('clean_accuracy', 0))} → {_percent(poison.get('poisoned_accuracy', 0))}; {poison.get('accuracy_change_percentage_points', 0):+.2f} pp"},
    ]
    st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)
    with st.expander("Configuration and evidence details"):
        st.json(config.get("attacks", {}), expanded=False)
        artifact_rows = []
        for item in manifest.get("artifacts", []):
            path = run_path / item["path"]
            try:
                actual = _sha(path)
                okay = path.stat().st_size == item["bytes"] and actual == item["sha256"]
            except OSError:
                okay = False
            artifact_rows.append({"Artifact": item["path"], "Current hash check": "PASS" if okay else "FAIL"})
        st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)
        st.json(verification, expanded=False)
        st.caption(f"Recorded verifier: {verification.get('status', 'unavailable')} · verifier source matches current acceptance script: {'yes' if verifier_matches else 'no'} · source model SHA-256: `{config.get('source_phase2_model_sha256', 'missing')}`")

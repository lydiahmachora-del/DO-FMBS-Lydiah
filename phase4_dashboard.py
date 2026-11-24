"""Read-only Phase 4 comparison of persisted defense runs."""
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def render_phase4_authoritative(run_root: Path) -> None:
    st.title("Phase 4 · Defense mechanisms")
    st.caption("Read-only view of saved defense checkpoints and evaluation evidence. This page never trains or regenerates a run.")
    runs = sorted((p for p in run_root.glob("phase4_authoritative_*")
                   if (p / "phase4_manifest.json").is_file()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        st.info("No authoritative Phase 4 run is saved yet. The historical Phase 4 workflow remains available in the legacy simulation view.")
        return
    chosen = st.selectbox("Saved Phase 4 run", [p.name for p in runs], key="phase4_authoritative_run")
    path = run_root / chosen
    config, metrics = _json(path / "phase4_config.json"), _json(path / "phase4_metrics.json")
    verification, manifest = (_json(path / "phase4_verification_report.json"),
                              _json(path / "phase4_manifest.json"))
    if not config or not metrics:
        st.error("This run is missing its persisted configuration or metrics.")
        return
    artifact_ok = True
    for entry in manifest.get("artifacts", []):
        target = path / entry["path"]
        try:
            artifact_ok &= target.stat().st_size == entry["bytes"] and _sha(target) == entry["sha256"]
        except OSError:
            artifact_ok = False
    if verification.get("status") == "PASS" and artifact_ok:
        st.success(f"Independent verification PASS · run `{chosen}` · source Phase 2 `{config.get('source_phase2_run')}`")
    else:
        st.error(f"Verification/artifact status requires review · verifier={verification.get('status', 'missing')} · artifact hashes={'PASS' if artifact_ok else 'FAIL'}")
    st.caption(f"Baseline checkpoint SHA-256 `{config.get('source_phase2_checkpoint_sha256', 'missing')}` · Phase 3 reference `{config.get('source_phase3_run', 'missing')}`")

    states = metrics.get("results", {})
    rows = []
    for name, result in states.items():
        attack = result.get("face_fgsm", {})
        rows.append({"Model state": name.replace("_", " ").title(),
                     "Face accuracy": result.get("face", {}).get("accuracy"),
                     "Fingerprint accuracy": result.get("fingerprint", {}).get("accuracy"),
                     "Face FGSM ASR": attack.get("attack_success_rate"),
                     "Clean-correct N": attack.get("clean_correct_denominator"),
                     "Attack successes": attack.get("attack_successes"),
                     "Evaluation N": attack.get("samples")})
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        st.subheader("Four-state comparison")
        display = comparison.copy()
        for column in ("Face accuracy", "Fingerprint accuracy", "Face FGSM ASR"):
            display[column] = display[column].map(lambda x: f"{100 * float(x):.2f}%")
        st.dataframe(display, width="stretch", hide_index=True)
        st.bar_chart(comparison.set_index("Model state")[["Face accuracy", "Fingerprint accuracy", "Face FGSM ASR"]])

    training = config.get("training", {})
    defenses = config.get("defenses", {})
    st.subheader("Executed settings")
    st.json({"optimizer": training.get("optimizer"), "learning_rate": training.get("learning_rate"),
             "batch_size": training.get("batch_size"), "defense_epochs": training.get("epochs_each_defense"),
             "combined_clean_pretraining_epochs": training.get("combined_clean_pretraining_epochs"),
             "combined_loss_weights": training.get("loss_weights"),
             "DP-inspired Gaussian noise": defenses.get("dp_inspired"),
             "face-only adversarial training": defenses.get("face_adversarial"),
             "combined defense": defenses.get("combined")}, expanded=False)
    st.warning("DP-inspired Gaussian gradient noise is not a formal differential-privacy guarantee: this run has no per-example clipping or privacy accountant. FGSM training perturbs face features only and does not establish fingerprint robustness.")
    st.caption("Face and fingerprint labels describe independent identity populations. Positional pairing supports the two-input model computation; it is not cross-modal person identity fusion.")

    grad = _json(path / "gradient_evidence.json")
    if grad:
        st.subheader("Combined-defense gradient evidence")
        st.dataframe(pd.DataFrame([{"Branch": branch.title(), "Mean gradient L2": grad.get("combined", {}).get("face_gradient_norm_mean" if branch == "face" else "fingerprint_gradient_norm_mean"),
                                    "Model update L2 (epoch-total)": grad.get("combined", {}).get("parameter_update_l2_total") if branch == "face" else None}
                                   for branch in ("face", "fingerprint")]), width="stretch", hide_index=True)
        st.caption("Both branch norms are measured from the equal-weight combined objective; the combined checkpoint is jointly fine-tuned, not assembled from separately trained weights.")

    st.subheader("Artifact integrity")
    integrity = []
    for item in manifest.get("artifacts", []):
        target = path / item["path"]
        try:
            good = target.stat().st_size == item["bytes"] and _sha(target) == item["sha256"]
        except OSError:
            good = False
        integrity.append({"Artifact": item["path"], "Bytes": item["bytes"], "SHA-256": item["sha256"],
                          "Current check": "PASS" if good else "FAIL"})
    st.dataframe(pd.DataFrame(integrity), width="stretch", hide_index=True)
    st.json(verification, expanded=False)

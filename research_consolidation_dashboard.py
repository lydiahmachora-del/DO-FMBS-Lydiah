"""Read-only comparison and export view for accepted DO-FMBS phase evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st


ACCEPTED_RUNS = {
    1: ("phase1_final_candidate_20261123_182623", "phase1_manifest.json", "phase1_verification_report.json"),
    2: ("phase2_authoritative_20261123_203617_remediation", "phase2_manifest.json", "phase2_verification_report.json"),
    3: ("phase3_authoritative_20261123_220738", "phase3_manifest.json", "phase3_verification_report.json"),
    4: ("phase4_authoritative_20261123_224549", "phase4_manifest.json", "phase4_verification_report.json"),
    5: ("phase5_authoritative_20261124_072946", "phase5_manifest.json", "phase5_verification_report.json"),
    6: ("phase6_authoritative_20261124_080204", "phase6_manifest.json", "phase6_verification_report.json"),
}


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_accepted_runs(run_root: Path) -> tuple[dict[int, dict], list[dict]]:
    """Load fixed accepted run IDs and check every manifest-listed artifact."""
    loaded: dict[int, dict] = {}
    report_rows: list[dict] = []
    for phase, (run_id, manifest_name, verifier_name) in ACCEPTED_RUNS.items():
        folder = run_root / run_id
        manifest = _json(folder / manifest_name)
        verifier = _json(folder / verifier_name)
        bad: list[str] = []
        entries = manifest.get("artifacts", [])
        manifest_run_id = manifest.get("run_id", manifest.get("replay_id"))
        identity_ok = manifest_run_id == run_id and manifest.get("phase") == phase
        if not identity_ok:
            bad.append("manifest run/phase identity mismatch")
        if not entries:
            bad.append("manifest has no artifacts")
        for entry in entries:
            artifact = folder / entry.get("path", "")
            try:
                artifact.resolve().relative_to(folder.resolve())
                if artifact.stat().st_size != entry.get("bytes") or _sha256(artifact) != entry.get("sha256"):
                    bad.append(entry.get("path", "<unknown>"))
            except (OSError, ValueError):
                bad.append(entry.get("path", "<missing>"))
        verifier_status = verifier.get("status", "UNKNOWN")
        ok = verifier_status == "PASS" and not bad
        loaded[phase] = {
            "run_id": run_id,
            "folder": folder,
            "manifest": manifest,
            "verifier": verifier,
            "metrics": _json(folder / f"phase{phase}_metrics.json"),
            "metadata": _json(folder / "phase1_metadata.json") if phase == 1 else {},
            "fit": _json(folder / "model_fit.json") if phase == 6 else {},
            "integrity_ok": ok,
            "bad_artifacts": bad,
        }
        report_rows.append({
            "Phase": phase,
            "Accepted run": run_id,
            "Independent verifier": verifier_status,
            "Manifest artifacts": len(entries),
            "Current SHA-256 checks": "PASS" if not bad and entries else "FAIL",
            "Manifest identity": "PASS" if identity_ok else "FAIL",
            "Dashboard accepted": "YES" if ok else "NO",
        })
    return loaded, report_rows


def _metric_rows(data: dict[int, dict]) -> list[dict]:
    p1 = data[1]["metadata"]
    p2 = data[2]["metrics"]
    p3 = data[3]["metrics"]
    p4 = data[4]["metrics"]
    p5 = data[5]["metrics"]
    p6 = data[6]["metrics"].get("model_fit", {})
    summary = [
        {"Phase": "1 · Data", "Evidence snapshot": f"Face {p1.get('face', {}).get('samples', '—')} samples / {p1.get('face', {}).get('identities', '—')} identities; fingerprint {p1.get('fingerprint', {}).get('samples', '—')} / {p1.get('fingerprint', {}).get('identities', '—')} synthetic identities"},
        {"Phase": "2 · Federated baseline", "Evidence snapshot": f"12 rounds; test accuracy face {p2.get('face_classification', {}).get('accuracy', float('nan')):.1%}, fingerprint {p2.get('fingerprint_classification', {}).get('accuracy', float('nan')):.1%}"},
        {"Phase": "3 · Attacks", "Evidence snapshot": f"Face FGSM ε=0.05 ASR {next((x.get('attack_success_rate', 0) for x in p3.get('evasion', []) if x.get('attack') == 'FGSM' and x.get('epsilon_projection_bound') == 0.05), float('nan')):.1%}; MIA accuracy {p3.get('membership_inference', {}).get('attack_accuracy', float('nan')):.1%}"},
        {"Phase": "4 · Defenses", "Evidence snapshot": f"Baseline → combined: face {p4.get('results', {}).get('federated_baseline', {}).get('face', {}).get('accuracy', float('nan')):.1%} → {p4.get('results', {}).get('combined_dofmbs_defense', {}).get('face', {}).get('accuracy', float('nan')):.1%}; fingerprint {p4.get('results', {}).get('federated_baseline', {}).get('fingerprint', {}).get('accuracy', float('nan')):.1%} → {p4.get('results', {}).get('combined_dofmbs_defense', {}).get('fingerprint', {}).get('accuracy', float('nan')):.1%}"},
        {"Phase": "5 · Validation", "Evidence snapshot": f"Combined face {p5.get('models', {}).get('dofmbs_combined', {}).get('face_classification', {}).get('accuracy', float('nan')):.1%}, fingerprint {p5.get('models', {}).get('dofmbs_combined', {}).get('fingerprint_classification', {}).get('accuracy', float('nan')):.1%}; paired face p={p5.get('statistical_comparisons', {}).get('face', {}).get('p_value', float('nan')):.4g}"},
        {"Phase": "6 · Optimization", "Evidence snapshot": f"Controlled simulation; R² {p6.get('r_squared', float('nan')):.4f}, CV R² {p6.get('cross_validation_r2_mean', float('nan')):.4f}; {data[6]['metrics'].get('schedule_summary', {}).get('rounds', '—')} prescribed rounds"},
    ]
    return summary


def render_research_consolidation(run_root: Path, report_path: Path) -> None:
    st.title("DO-FMBS · Research simulation and comparison")
    st.caption("A read-only, evidence-led view across the six accepted phase runs. Historical models and metrics are loaded from disk; this page does not train or alter them.")
    evidence, audit_rows = inspect_accepted_runs(run_root)
    integrity = all(item["integrity_ok"] for item in evidence.values())
    if integrity:
        st.success("All six accepted phase runs pass their recorded independent verification and current manifest SHA-256 checks.")
    else:
        failed = [f"Phase {phase}" for phase, item in evidence.items() if not item["integrity_ok"]]
        st.error("Evidence check failed for " + ", ".join(failed) + ". Do not use affected values as verified results.")
    st.info("Demo mode = browse and compare these persisted results. To execute a fresh simulation, use **Start a new simulation** in the sidebar; it creates an isolated replay and leaves accepted evidence intact.")

    summary = _metric_rows(evidence)
    st.subheader("Six-phase research workflow")
    st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)

    p2 = evidence[2]["folder"]
    try:
        rounds = pd.read_csv(p2 / "round_metrics.csv")
    except (OSError, ValueError) as error:
        rounds = pd.DataFrame()
        st.warning(f"Could not read persisted federated round history: {error}")
    if not rounds.empty:
        st.subheader("Federated training · saved round values")
        st.caption("Test metrics are plotted by federated round. These are checkpoint evaluations, not independent test sets per round.")
        left, right = st.columns(2)
        left.line_chart(rounds.set_index("round")[["test_face_output_accuracy", "test_finger_output_accuracy"]])
        right.line_chart(rounds.set_index("round")[["test_face_output_loss", "test_finger_output_loss"]])
        st.dataframe(rounds[["round", "test_face_output_accuracy", "test_finger_output_accuracy", "test_face_output_loss", "test_finger_output_loss"]], width="stretch", hide_index=True)
        st.download_button("Download Phase 2 round values (CSV)", rounds.to_csv(index=False).encode("utf-8"), "phase2_round_metrics.csv", "text/csv")

    st.subheader("Validation comparison · Phase 5")
    p5 = evidence[5]["metrics"]
    comparison_rows = []
    for model_key, label in (("baseline", "Federated baseline"), ("dofmbs_combined", "DO-FMBS combined defense")):
        models = p5.get("models", {}).get(model_key, {})
        for modality in ("face", "fingerprint"):
            score = models.get(f"{modality}_classification", {})
            comparison_rows.append({"Model": label, "Modality": modality.title(), "Correct / N": f"{score.get('correct', '—')} / {score.get('samples', '—')}", "Accuracy": score.get("accuracy")})
    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
    stats = p5.get("statistical_comparisons", {})
    st.caption(f"Paired identity-cluster test: face Δ={stats.get('face', {}).get('difference_percentage_points', float('nan')):+.2f} pp (p={stats.get('face', {}).get('p_value', float('nan')):.5f}); fingerprint Δ={stats.get('fingerprint', {}).get('difference_percentage_points', float('nan')):+.2f} pp (p={stats.get('fingerprint', {}).get('p_value', float('nan')):.5f}).")

    st.subheader("Run provenance and integrity")
    st.dataframe(pd.DataFrame(audit_rows), width="stretch", hide_index=True)
    st.download_button("Download six-phase evidence summary (CSV)", pd.DataFrame(summary).to_csv(index=False).encode("utf-8"), "dofmbs_six_phase_summary.csv", "text/csv")
    package = {
        "title": "DO-FMBS accepted six-phase evidence index",
        "interpretation": "Values are read from accepted phase-specific runs; see the report for scope and limitations.",
        "phases": [
            {"phase": phase, "run_id": data["run_id"], "independent_verifier": data["verifier"].get("status"),
             "manifest_sha256_checks": "PASS" if data["integrity_ok"] else "FAIL",
             "summary": summary[phase - 1]["Evidence snapshot"], "metrics": data["metrics"]}
            for phase, data in evidence.items()
        ],
    }
    st.download_button("Download evidence index (JSON)", json.dumps(package, indent=2, allow_nan=False).encode("utf-8"), "dofmbs_six_phase_evidence.json", "application/json")
    if report_path.is_file():
        st.download_button("Download final research report (Markdown)", report_path.read_bytes(), report_path.name, "text/markdown")
    st.warning("Scope: the fingerprint branch uses generated synthetic identities; face/fingerprint labels are independent populations; adversarial robustness is face-only; gradient noise is not a formal differential-privacy guarantee; Phase 6 is a controlled model-derived simulation, not deployment evidence.")

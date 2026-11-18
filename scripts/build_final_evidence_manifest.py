"""Build the single authoritative manifest for one completed Phase 1-6 replay."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runtime_recording" / "final_scientific_replay_20260907"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--publish-path", type=Path, default=ROOT / "do_fmbs_data" / "final_evidence_manifest.json")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    import hashlib

    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip() or "unavailable"
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip())
    return commit, dirty


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_dir = run_dir / "do_fmbs_data"
    registry = read_json(run_dir / "phase_runtime_registry.json")
    records = registry.get("records", [])
    if [int(row["phase"]) for row in records] != [1, 2, 3, 4, 5, 6]:
        raise RuntimeError("Authoritative manifest requires exactly one ordered record for Phases 1-6")
    if any(int(row["exit_code"]) != 0 or row["status"] != "EXECUTION-COMPLETED" for row in records):
        raise RuntimeError("All six phases must complete with exit code zero")

    phase1 = read_json(data_dir / "phase1_metadata.json")
    phase2 = read_json(data_dir / "phase2_metrics.json")
    phase3 = read_json(data_dir / "phase3_attack_results.json")
    phase4 = read_json(data_dir / "phase4_defense_results.json")
    phase5 = read_json(data_dir / "phase5_validation.json")
    independent_audit = read_json(data_dir / "independent_replay_audit.json")
    replay_id = run_dir.name
    if independent_audit.get("status") != "INDEPENDENT-REPLAY-AUDIT-PASSED" or independent_audit.get("replay_id") != replay_id:
        raise RuntimeError("Independent replay audit is missing, failed or belongs to another replay")
    for number, payload in enumerate((phase1, phase2, phase3, phase4, phase5), start=1):
        if payload.get("replay_id") != replay_id:
            raise RuntimeError(f"Phase {number} replay mismatch: {payload.get('replay_id')} != {replay_id}")

    phase6_dir = run_dir / "simulation_evidence" / "constrained_dynamic_model"
    phase6 = read_json(phase6_dir / "model_fit.json")
    artifact_paths = sorted(
        [path for path in data_dir.rglob("*") if path.is_file() and path.name != "final_evidence_manifest.json"]
        + [path for path in phase6_dir.rglob("*") if path.is_file()]
        + [run_dir / "phase_runtime_registry.json", run_dir / "phase_runtime_registry.csv"]
        + sorted(run_dir.glob("phase*_console.log"))
    )
    artifacts = [
        {"path": rel(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in artifact_paths
    ]
    commit, dirty = git_state()
    dependencies = {}
    for package in ("tensorflow-cpu", "numpy", "scipy", "scikit-learn", "matplotlib", "pandas", "python-docx", "PyMuPDF"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not installed"

    manifest = {
        "schema_version": "2.0",
        "status": "AUTHORITATIVE-SYNCHRONIZED-REPLAY-COMPLETE",
        "replay_id": replay_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_replay_root": rel(run_dir),
        "git": {"commit": commit, "working_tree_dirty_at_manifest_creation": dirty},
        "environment": {"python": sys.version, "platform": platform.platform(), "dependencies": dependencies,
                        "determinism_note": "Fixed seeds and TensorFlow deterministic operations requested; cross-hardware bitwise identity is not guaranteed."},
        "seeds": phase1["seeds"],
        "phase_parameters": {
            "phase1": {"face_split": "70/15/15 stratified", "fingerprint_split": "5/1/2 impressions per identity"},
            "phase2": phase2["training_parameters"], "phase3": phase3["parameters"],
            "phase4": phase4["parameters"],
            "phase5": {"bootstrap_repetitions": phase5["bootstrap_repetitions"],
                       "verification_threshold_rule": phase5["face"]["verification"]["threshold_rule"]},
            "phase6": phase6,
        },
        "dataset_statistics": {"face": phase1["face"], "fingerprint": phase1["fingerprint"]},
        "federated_learning": phase2,
        "attacks": phase3,
        "defenses": phase4,
        "final_metrics": {
            "face_classification": phase5["face"]["classification"],
            "fingerprint_classification": phase5["fingerprint"]["classification"],
            "face_verification": phase5["face"]["verification"],
            "fingerprint_verification": phase5["fingerprint"]["verification"],
            "statistical_validation": phase5["statistical_validation"],
            "robustness": phase5["robustness"],
        },
        "dynamic_optimization": {"model_fit": phase6, "artifact_directory": rel(phase6_dir)},
        "runtime": {"clock": registry.get("clock"), "records": records,
                    "total_seconds": float(sum(float(row["elapsed_seconds"]) for row in records))},
        "independent_audit": independent_audit,
        "methodological_scope": {
            "identity_spaces": "independent face and fingerprint populations; no person-level multimodal fusion claim",
            "fingerprints": "synthetic within-study evidence, not an FVC or external benchmark",
            "adversarial": "FGSM and PGD evasion evidence applies to the face branch only",
            "privacy": "DP-inspired Gaussian gradient noise without per-example clipping or a privacy accountant",
            "phase6": "controlled simulation/model-derived optimization rather than production deployment evidence",
        },
        "artifacts": artifacts,
    }
    target_in_run = data_dir / "final_evidence_manifest.json"
    for target in (target_in_run, args.publish_path.resolve()):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Authoritative evidence manifest: {args.publish_path.resolve()}")
    print(f"Replay: {replay_id}; hashed artifacts: {len(artifacts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate an isolated Phase 1-6 replay and finalize its runtime evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runtime_recording" / "full_pipeline_20260809"
EXPECTED = {
    1: [
        "do_fmbs_data/face_data.npz", "do_fmbs_data/fingerprint_data.npz",
        *[f"do_fmbs_data/clients/client_{index}.npz" for index in range(5)],
    ],
    2: [
        "do_fmbs_data/baseline_model.keras", "do_fmbs_data/baseline_metrics.pkl",
        "do_fmbs_training_accuracy.png",
    ],
    3: [
        "do_fmbs_data/attack_results.pkl", "do_fmbs_attack_results.png",
        "do_fmbs_attack_extended_results.png",
    ],
    4: [
        "do_fmbs_data/do_fmbs_model.keras", "do_fmbs_data/final_results.pkl",
        "do_fmbs_comparison_results.png",
    ],
    5: [
        "do_fmbs_data/validation_results.pkl", "do_fmbs_data/validation_report.txt",
        "do_fmbs_validation_robustness.png", "do_fmbs_validation_statistics.png",
        "do_fmbs_validation_summary.png",
    ],
    6: [
        "simulation_evidence/constrained_dynamic_model/model_fit.json",
        "simulation_evidence/constrained_dynamic_model/coefficient_estimates.csv",
        "simulation_evidence/constrained_dynamic_model/dynamic_optimization_schedule.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize isolated Phase 1-6 evidence.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    registry_json = run_dir / "phase_runtime_registry.json"
    payload = json.loads(registry_json.read_text(encoding="utf-8"))
    records = payload["records"]
    if {int(row["phase"]) for row in records} != set(range(1, 7)):
        raise RuntimeError("Runtime registry must contain all six phases")
    artifact_rows: list[dict] = []
    for record in records:
        phase = int(record["phase"])
        if int(record["exit_code"]) != 0:
            raise RuntimeError(f"Phase {phase} exit code is not zero")
        record["status"] = "EXECUTION-COMPLETED"
        for relative in EXPECTED[phase]:
            path = run_dir / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Phase {phase} required artifact is missing: {path}")
            artifact_rows.append({
                "phase": phase,
                "artifact": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "status": "ARTIFACT-VERIFIED",
            })
    payload["records"] = records
    payload["execution_status"] = "ALL-SIX-PHASES-COMPLETED"
    payload["artifact_status"] = "ALL-REQUIRED-ARTIFACTS-VERIFIED"
    payload["total_elapsed_seconds"] = sum(float(row["elapsed_seconds"]) for row in records)
    registry_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    registry_csv = run_dir / "phase_runtime_registry.csv"
    fields = [
        "phase", "script", "scope", "command", "start_timestamp",
        "end_timestamp", "elapsed_seconds", "exit_code", "status", "log_path",
    ]
    with registry_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in records)

    artifact_csv = run_dir / "phase_artifact_hashes.csv"
    with artifact_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(artifact_rows[0]))
        writer.writeheader()
        writer.writerows(artifact_rows)
    artifact_manifest = {
        "status": "ALL-REQUIRED-ARTIFACTS-VERIFIED",
        "artifacts": artifact_rows,
    }
    (run_dir / "phase_artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("ALL-SIX-PHASES-COMPLETED")
    print("ALL-REQUIRED-ARTIFACTS-VERIFIED")
    print(f"Total elapsed seconds: {payload['total_elapsed_seconds']:.6f}")


if __name__ == "__main__":
    main()

"""Independently verify split, gradient and biometric-metric arithmetic for a replay."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runtime_recording" / "final_scientific_replay_20260907"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_disjoint(*arrays: np.ndarray) -> None:
    sets = [set(np.asarray(values, dtype=int)) for values in arrays]
    for index, left in enumerate(sets):
        for right in sets[index + 1:]:
            if left & right:
                raise AssertionError("Split overlap detected")


def main() -> int:
    args = parse_args()
    data_dir = args.run_dir.resolve() / "do_fmbs_data"
    checks = []
    for modality in ("face", "fingerprint"):
        with np.load(data_dir / f"{modality}_data.npz", allow_pickle=False) as data:
            assert_disjoint(data["train_indices"], data["calibration_indices"], data["test_indices"])
            all_indices = np.concatenate([data["train_indices"], data["calibration_indices"], data["test_indices"]])
            if len(np.unique(all_indices)) != len(all_indices):
                raise AssertionError(f"{modality} split does not have unique source observations")
            with np.load(data_dir / f"{modality}_scaler.npz", allow_pickle=False) as scaler:
                np.testing.assert_array_equal(scaler["fitted_train_indices"], data["train_indices"])
            test_count = len(data["y_test"])
        matrix = np.loadtxt(data_dir / f"{modality}_confusion_matrix.csv", delimiter=",")
        if int(matrix.sum()) != test_count:
            raise AssertionError(f"{modality} confusion matrix total does not equal final test samples")
        report = read_json(data_dir / f"{modality}_verification_report.json")
        if not report["threshold_frozen_before_test"]:
            raise AssertionError(f"{modality} threshold was not frozen")
        with np.load(data_dir / f"{modality}_verification_scores.npz", allow_pickle=False) as scores:
            genuine = scores["test_genuine"]
            impostor = scores["test_impostor"]
        final = report["final_test"]
        threshold = float(final["threshold"])
        false_accepts = int(np.count_nonzero(impostor >= threshold))
        false_rejects = int(np.count_nonzero(genuine < threshold))
        if (false_accepts, len(impostor)) != (final["impostor_accepts"], final["impostor_attempts"]):
            raise AssertionError(f"{modality} FAR counts do not reproduce")
        if (false_rejects, len(genuine)) != (final["genuine_rejects"], final["genuine_attempts"]):
            raise AssertionError(f"{modality} FRR counts do not reproduce")
        if not np.isclose(final["FAR"], false_accepts / len(impostor)):
            raise AssertionError(f"{modality} FAR percentage does not reproduce")
        if not np.isclose(final["FRR"], false_rejects / len(genuine)):
            raise AssertionError(f"{modality} FRR percentage does not reproduce")
        checks.append({"check": f"{modality} split/scaler/classification/verification arithmetic", "status": "PASS"})

    with np.load(data_dir / "fingerprint_data.npz", allow_pickle=False) as fingerprint:
        for split, expected in (("train", 5), ("calibration", 1), ("test", 2)):
            _, counts = np.unique(fingerprint[f"y_{split}"], return_counts=True)
            if len(counts) != 100 or not np.all(counts == expected):
                raise AssertionError(f"Fingerprint {split} identity counts are invalid")
    quality = read_json(data_dir / "fingerprint_quality_diagnostics.json")
    if not quality["passed"]:
        raise AssertionError("Fingerprint quality gate did not pass")
    defense = read_json(data_dir / "phase4_defense_results.json")
    if not defense["fingerprint_gradient_norms"] or not all(
        float(value) > 0 for value in defense["fingerprint_gradient_norms"].values()
    ):
        raise AssertionError("Fingerprint gradients were absent")
    if defense["parameters"]["privacy_accountant"] or defense["parameters"]["epsilon_delta_reported"]:
        raise AssertionError("Phase 4 inaccurately claims formal privacy accounting")
    checks.extend([
        {"check": "fingerprint 5/1/2 identity protocol", "status": "PASS"},
        {"check": "fingerprint pre-training separability", "status": "PASS"},
        {"check": "combined-loss fingerprint gradients", "status": "PASS"},
        {"check": "DP-inspired terminology boundary", "status": "PASS"},
    ])
    output = data_dir / "independent_replay_audit.json"
    output.write_text(json.dumps({
        "status": "INDEPENDENT-REPLAY-AUDIT-PASSED", "replay_id": args.run_dir.resolve().name,
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Independent replay audit: PASS ({len(checks)} checks)")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

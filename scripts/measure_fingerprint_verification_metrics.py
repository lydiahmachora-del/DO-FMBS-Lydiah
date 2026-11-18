"""Measure claimed-identity verification metrics for the fingerprint branch.

The existing operational evaluator measures the face branch.  This companion
evaluator applies the same frozen-threshold design to the saved fingerprint
output of the baseline and final DO-FMBS models.  It keeps the synthetic
fingerprint protocol separate from the face protocol: only identities with two
test impressions can contribute one calibration and one evaluation probe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "runtime_recording" / "full_pipeline_20260809" / "do_fmbs_data"
DEFAULT_OUTPUT = (
    ROOT / "runtime_metrics_evidence" / "recorded_full_pipeline_fingerprint_verification_evaluation"
)
SPLIT_SEED = 20260809
BOOTSTRAP_SEED = 951523
CALIBRATION_FAR_UPPER_CAP = 0.075
DEPLOYMENT_FAR_LIMIT = 0.10
DEPLOYMENT_FRR_LIMIT = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure threshold-based fingerprint verification metrics for DO-FMBS."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument(
        "--calibration-far-upper-cap",
        type=float,
        default=CALIBRATION_FAR_UPPER_CAP,
        help="Calibration-only Wilson FAR upper cap used to freeze each threshold.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, total: int,
           z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided 95% Wilson binomial interval."""
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def paired_identity_split(labels: np.ndarray, *, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one calibration and one evaluation impression per eligible identity.

    The persisted fingerprint test split contains one or two impressions per
    synthetic identity.  An identity with a single test impression cannot be
    used in a disjoint calibration/evaluation protocol, so it is excluded from
    the claimed-identity trial set rather than reused in both partitions.
    """
    rng = np.random.default_rng(seed)
    calibration: list[int] = []
    evaluation: list[int] = []
    eligible_identities: list[int] = []
    for identity in np.unique(labels):
        indices = np.flatnonzero(labels == identity).copy()
        if len(indices) < 2:
            continue
        rng.shuffle(indices)
        calibration.append(int(indices[0]))
        evaluation.append(int(indices[1]))
        eligible_identities.append(int(identity))
    if not calibration or len(calibration) != len(evaluation):
        raise ValueError("Fingerprint protocol requires at least one identity with two test impressions")
    return (
        np.array(sorted(calibration), dtype=int),
        np.array(sorted(evaluation), dtype=int),
        np.array(sorted(eligible_identities), dtype=np.int64),
    )


def trial_scores(probabilities: np.ndarray, indices: np.ndarray,
                 labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = probabilities[indices]
    truth = labels[indices]
    genuine = selected[np.arange(len(indices)), truth]
    mask = np.ones(selected.shape, dtype=bool)
    mask[np.arange(len(indices)), truth] = False
    impostor = selected[mask]
    return genuine, impostor


def select_threshold(impostor_scores: np.ndarray,
                     upper_cap: float = CALIBRATION_FAR_UPPER_CAP) -> float:
    """Freeze the lowest calibration threshold whose Wilson FAR upper bound fits the cap."""
    values = np.unique(np.sort(impostor_scores))
    if values.size == 0:
        raise ValueError("Fingerprint threshold calibration requires impostor scores")
    threshold = float(np.nextafter(values[-1], np.inf))
    for candidate in values[::-1]:
        false_accepts = int(np.count_nonzero(impostor_scores >= candidate))
        if wilson(false_accepts, len(impostor_scores))[1] <= upper_cap:
            threshold = float(candidate)
        else:
            break
    return threshold


def identity_cluster_interval(probabilities: np.ndarray, labels: np.ndarray,
                              indices: np.ndarray, threshold: float,
                              repetitions: int) -> dict[str, tuple[float, float]]:
    """Bootstrap by claimed identity to account for clustered claim trials."""
    identities = np.unique(labels[indices])
    far_successes: list[int] = []
    far_totals: list[int] = []
    frr_successes: list[int] = []
    frr_totals: list[int] = []
    for identity in identities:
        group = indices[labels[indices] == identity]
        genuine, impostor = trial_scores(probabilities, group, labels)
        far_successes.append(int(np.count_nonzero(impostor >= threshold)))
        far_totals.append(len(impostor))
        frr_successes.append(int(np.count_nonzero(genuine < threshold)))
        frr_totals.append(len(genuine))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selections = rng.integers(0, len(identities), size=(repetitions, len(identities)))
    far_successes_array = np.asarray(far_successes)
    far_totals_array = np.asarray(far_totals)
    frr_successes_array = np.asarray(frr_successes)
    frr_totals_array = np.asarray(frr_totals)
    far_values = far_successes_array[selections].sum(axis=1) / far_totals_array[selections].sum(axis=1)
    frr_values = frr_successes_array[selections].sum(axis=1) / frr_totals_array[selections].sum(axis=1)
    return {
        "FAR": tuple(float(value) for value in np.quantile(far_values, [0.025, 0.975])),
        "FRR": tuple(float(value) for value in np.quantile(frr_values, [0.025, 0.975])),
    }


def rate_record(metric: str, successes: int, total: int) -> dict[str, float | int]:
    low, high = wilson(successes, total)
    return {
        f"{metric}_numerator": successes,
        f"{metric}_denominator": total,
        metric: successes / total,
        f"{metric}_wilson_95_low": low,
        f"{metric}_wilson_95_high": high,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def input_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def output_record(path: Path) -> dict[str, object]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_far_upper_cap < DEPLOYMENT_FAR_LIMIT:
        raise ValueError("Calibration FAR upper cap must be positive and below the project FAR limit")
    if args.bootstrap_repetitions < 1_000:
        raise ValueError("Use at least 1,000 identity-cluster bootstrap repetitions")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(ZoneInfo("Africa/Nairobi"))
    total_start = time.perf_counter()

    load_start = time.perf_counter()
    with np.load(data_dir / "fingerprint_data.npz", allow_pickle=False) as data:
        x_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int64)
        n_classes = int(np.asarray(data["n_classes"]).reshape(-1)[0])
    calibration_indices, evaluation_indices, eligible_ids = paired_identity_split(y_test)
    all_identities = np.unique(y_test)
    excluded_ids = np.setdiff1d(all_identities, eligible_ids)
    data_load_seconds = time.perf_counter() - load_start

    if len(eligible_ids) != 60 or len(excluded_ids) != 40:
        raise RuntimeError("Unexpected fingerprint two-impression identity distribution")
    if len(calibration_indices) != len(evaluation_indices) or len(evaluation_indices) != len(eligible_ids):
        raise RuntimeError("Fingerprint calibration/evaluation partitions are inconsistent")

    clean_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = [
        {
            "component": "Fingerprint data load and paired identity split",
            "elapsed_seconds": data_load_seconds,
            "status": "RECORDED",
        }
    ]
    model_specs = [
        ("Baseline comparator", data_dir / "baseline_model.keras"),
        ("DO-FMBS final", data_dir / "do_fmbs_model.keras"),
    ]

    for model_label, model_path in model_specs:
        protocol_start = time.perf_counter()
        model_load_start = time.perf_counter()
        model = tf.keras.models.load_model(model_path, compile=False)
        fingerprint_model = tf.keras.Model(
            model.get_layer("finger_input").input,
            model.get_layer("finger_output").output,
        )
        model_load_seconds = time.perf_counter() - model_load_start

        inference_start = time.perf_counter()
        probabilities = fingerprint_model.predict(x_test, batch_size=64, verbose=0)
        inference_seconds = time.perf_counter() - inference_start
        if probabilities.shape != (len(x_test), n_classes):
            raise RuntimeError("Fingerprint output dimensions do not match the persisted fingerprint data")

        calibration_genuine, calibration_impostor = trial_scores(probabilities, calibration_indices, y_test)
        threshold_start = time.perf_counter()
        threshold = select_threshold(calibration_impostor, args.calibration_far_upper_cap)
        calibration_false_accepts = int(np.count_nonzero(calibration_impostor >= threshold))
        calibration_false_rejects = int(np.count_nonzero(calibration_genuine < threshold))
        threshold_seconds = time.perf_counter() - threshold_start

        evaluation_start = time.perf_counter()
        evaluation_genuine, evaluation_impostor = trial_scores(probabilities, evaluation_indices, y_test)
        false_accepts = int(np.count_nonzero(evaluation_impostor >= threshold))
        false_rejects = int(np.count_nonzero(evaluation_genuine < threshold))
        far_data = rate_record("FAR", false_accepts, len(evaluation_impostor))
        frr_data = rate_record("FRR", false_rejects, len(evaluation_genuine))
        cluster = identity_cluster_interval(
            probabilities, y_test, evaluation_indices, threshold, args.bootstrap_repetitions
        )
        evaluation_seconds = time.perf_counter() - evaluation_start
        final_model = model_label == "DO-FMBS final"
        gate_satisfied = (
            cluster["FAR"][1] <= DEPLOYMENT_FAR_LIMIT
            and cluster["FRR"][1] <= DEPLOYMENT_FRR_LIMIT
        )
        status = (
            "WITHIN PROJECT LIMITS" if final_model and gate_satisfied
            else "FINGERPRINT BRANCH IMPROVEMENT REQUIRED" if final_model
            else "COMPARATOR RECORDED"
        )
        clean_rows.append({
            "model": model_label,
            "modality": "Fingerprint",
            "protocol": "Claimed-identity synthetic-fingerprint verification; one calibration and one evaluation impression per eligible identity",
            "split_seed": SPLIT_SEED,
            "test_images_total": len(y_test),
            "synthetic_identities_total": len(all_identities),
            "eligible_two_impression_identities": len(eligible_ids),
            "excluded_single_impression_identities": len(excluded_ids),
            "calibration_genuine_trials": len(calibration_genuine),
            "calibration_impostor_trials": len(calibration_impostor),
            "threshold": threshold,
            "threshold_rule": (
                "Lowest calibration threshold with FAR Wilson 95% upper <= "
                f"{args.calibration_far_upper_cap:.3f}"
            ),
            "calibration_FAR_numerator": calibration_false_accepts,
            "calibration_FAR_denominator": len(calibration_impostor),
            "calibration_FAR": calibration_false_accepts / len(calibration_impostor),
            "calibration_FRR_numerator": calibration_false_rejects,
            "calibration_FRR_denominator": len(calibration_genuine),
            "calibration_FRR": calibration_false_rejects / len(calibration_genuine),
            **far_data,
            **frr_data,
            "FAR_identity_cluster_95_low": cluster["FAR"][0],
            "FAR_identity_cluster_95_high": cluster["FAR"][1],
            "FRR_identity_cluster_95_low": cluster["FRR"][0],
            "FRR_identity_cluster_95_high": cluster["FRR"][1],
            "FAR_acceptance_limit": DEPLOYMENT_FAR_LIMIT,
            "FRR_acceptance_limit": DEPLOYMENT_FRR_LIMIT,
            "acceptance_basis": "Identity-cluster bootstrap 95% upper bounds",
            "status": status,
        })
        runtime_rows.extend([
            {"component": f"{model_label} fingerprint model load", "elapsed_seconds": model_load_seconds, "status": "RECORDED"},
            {"component": f"{model_label} fingerprint inference", "elapsed_seconds": inference_seconds, "status": "RECORDED"},
            {"component": f"{model_label} fingerprint threshold calibration", "elapsed_seconds": threshold_seconds, "status": "RECORDED"},
            {"component": f"{model_label} fingerprint evaluation and bootstrap", "elapsed_seconds": evaluation_seconds, "status": "RECORDED"},
            {"component": f"{model_label} fingerprint complete protocol", "elapsed_seconds": time.perf_counter() - protocol_start, "status": "RECORDED"},
        ])

    total_seconds = time.perf_counter() - total_start
    runtime_rows.append({
        "component": "Complete fingerprint verification evaluation",
        "elapsed_seconds": total_seconds,
        "status": "RECORDED",
    })
    output_csv = output_dir / "fingerprint_far_frr_acceptance.csv"
    runtime_csv = output_dir / "fingerprint_evaluation_runtime.csv"
    write_csv(output_csv, clean_rows)
    write_csv(runtime_csv, runtime_rows)

    final = next(row for row in clean_rows if row["model"] == "DO-FMBS final")
    ended = datetime.now(ZoneInfo("Africa/Nairobi"))
    manifest = {
        "protocol": "DO-FMBS claimed-identity synthetic-fingerprint verification",
        "scope": (
            "Fingerprint branch only; 60 two-impression synthetic identities are split into "
            "disjoint calibration and evaluation probes. This is not an external fingerprint benchmark."
        ),
        "started_africa_nairobi": started.isoformat(timespec="milliseconds"),
        "ended_africa_nairobi": ended.isoformat(timespec="milliseconds"),
        "elapsed_seconds": total_seconds,
        "split_seed": SPLIT_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "calibration_FAR_Wilson_upper_cap": args.calibration_far_upper_cap,
        "deployment_limits": {"FAR": DEPLOYMENT_FAR_LIMIT, "FRR": DEPLOYMENT_FRR_LIMIT},
        "accept_rule": "claimed fingerprint identity accepted when finger softmax score >= frozen branch threshold",
        "impostor_policy": "all 99 wrong fingerprint identity claims per evaluation probe",
        "eligible_identity_policy": "one calibration and one evaluation impression for every identity with two test impressions",
        "final_status": final["status"],
        "final_status_basis": (
            "Final fingerprint FAR and FRR identity-cluster 95% upper bounds are within the declared limits"
            if final["status"] == "WITHIN PROJECT LIMITS"
            else "Final fingerprint identity-cluster upper bounds identify the branch metrics requiring improvement"
        ),
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or "unreported",
        },
        "inputs": [
            input_record(data_dir / "fingerprint_data.npz"),
            input_record(data_dir / "baseline_model.keras"),
            input_record(data_dir / "do_fmbs_model.keras"),
            input_record(Path(__file__).resolve()),
        ],
        "outputs": [output_record(output_csv), output_record(runtime_csv)],
    }
    (output_dir / "fingerprint_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Fingerprint verification evidence written to {output_dir.relative_to(ROOT)}")
    print(f"Elapsed seconds: {total_seconds:.6f}")
    print(f"Final fingerprint verification status: {final['status']}")


if __name__ == "__main__":
    main()

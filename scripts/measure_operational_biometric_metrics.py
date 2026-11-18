"""Measure operational FAR, FRR, ASR, AI and PA with recorded runtimes.

This evaluator replaces classification-proxy-only reporting with a claimed-
identity verification protocol.  Thresholds are calibrated without touching the
evaluation split, then frozen for clean and adversarial evaluation.  All counts,
confidence intervals, thresholds, gates, statuses and elapsed times are saved.
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
DEFAULT_DATA = ROOT / "do_fmbs_data"
DEFAULT_OUTPUT = ROOT / "runtime_metrics_evidence" / "operational_biometric_evaluation"
SPLIT_SEED = 20260809
BOOTSTRAP_SEED = 951523
CALIBRATION_FAR_UPPER_CAP = 0.075
DEPLOYMENT_FAR_LIMIT = 0.10
DEPLOYMENT_FRR_LIMIT = 0.20
PA_MANAGED_LIMIT = 0.40
EPSILONS = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure threshold-based DO-FMBS biometric and attack metrics."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument(
        "--calibration-far-upper-cap",
        type=float,
        default=CALIBRATION_FAR_UPPER_CAP,
        help=(
            "Calibration-only Wilson FAR upper cap used to freeze each model's "
            "verification threshold."
        ),
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
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def identity_split(labels: np.ndarray, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    calibration: list[int] = []
    evaluation: list[int] = []
    for identity in np.unique(labels):
        indices = np.flatnonzero(labels == identity).copy()
        rng.shuffle(indices)
        cut = max(1, len(indices) // 2)
        calibration.extend(indices[:cut])
        evaluation.extend(indices[cut:])
    return np.array(sorted(calibration), dtype=int), np.array(sorted(evaluation), dtype=int)


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
    values = np.unique(np.sort(impostor_scores))
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
    far_successes_array = np.asarray(far_successes)
    far_totals_array = np.asarray(far_totals)
    frr_successes_array = np.asarray(frr_successes)
    frr_totals_array = np.asarray(frr_totals)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selections = rng.integers(0, len(identities), size=(repetitions, len(identities)))
    far_values = (
        far_successes_array[selections].sum(axis=1)
        / far_totals_array[selections].sum(axis=1)
    )
    frr_values = (
        frr_successes_array[selections].sum(axis=1)
        / frr_totals_array[selections].sum(axis=1)
    )
    return {
        "FAR": tuple(float(value) for value in np.quantile(far_values, [0.025, 0.975])),
        "FRR": tuple(float(value) for value in np.quantile(frr_values, [0.025, 0.975])),
    }


def rate_record(metric: str, successes: int, total: int) -> dict:
    low, high = wilson(successes, total)
    return {
        f"{metric}_numerator": successes,
        f"{metric}_denominator": total,
        metric: successes / total,
        f"{metric}_wilson_95_low": low,
        f"{metric}_wilson_95_high": high,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_far_upper_cap < DEPLOYMENT_FAR_LIMIT:
        raise ValueError(
            "calibration FAR upper cap must be positive and below the deployment FAR limit"
        )
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(ZoneInfo("Africa/Nairobi"))
    total_start = time.perf_counter()

    load_start = time.perf_counter()
    with np.load(data_dir / "face_data.npz", allow_pickle=False) as data:
        x_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int64)
        x_train_members = data["X_train"][:100].astype(np.float32)
    calibration_indices, evaluation_indices = identity_split(y_test)
    data_load_seconds = time.perf_counter() - load_start
    if len(calibration_indices) != 291 or len(evaluation_indices) != 314:
        raise RuntimeError("Unexpected calibration/evaluation split size")
    if len(np.unique(y_test[calibration_indices])) != 62 or len(np.unique(y_test[evaluation_indices])) != 62:
        raise RuntimeError("Every identity must occur in both protocol partitions")

    model_specs = [
        ("Baseline comparator", data_dir / "baseline_model.keras"),
        ("DO-FMBS final", data_dir / "do_fmbs_model.keras"),
    ]
    clean_rows: list[dict] = []
    ai_rows: list[dict] = []
    privacy_rows: list[dict] = []
    runtime_rows: list[dict] = []
    final_face_model = None
    final_threshold = None
    final_probabilities = None
    final_gate_satisfied = False

    for model_label, model_path in model_specs:
        model_total_start = time.perf_counter()
        model_load_start = time.perf_counter()
        model = tf.keras.models.load_model(model_path, compile=False)
        face_model = tf.keras.Model(
            model.get_layer("face_input").input,
            model.get_layer("face_output").output,
        )
        model_load_seconds = time.perf_counter() - model_load_start

        inference_start = time.perf_counter()
        probabilities = face_model.predict(x_test, batch_size=64, verbose=0)
        inference_seconds = time.perf_counter() - inference_start
        calibration_genuine, calibration_impostor = trial_scores(
            probabilities, calibration_indices, y_test
        )
        threshold_start = time.perf_counter()
        threshold = select_threshold(
            calibration_impostor,
            upper_cap=args.calibration_far_upper_cap,
        )
        calibration_false_accepts = int(np.count_nonzero(calibration_impostor >= threshold))
        calibration_false_rejects = int(np.count_nonzero(calibration_genuine < threshold))
        threshold_seconds = time.perf_counter() - threshold_start

        evaluation_start = time.perf_counter()
        evaluation_genuine, evaluation_impostor = trial_scores(
            probabilities, evaluation_indices, y_test
        )
        false_accepts = int(np.count_nonzero(evaluation_impostor >= threshold))
        false_rejects = int(np.count_nonzero(evaluation_genuine < threshold))
        far_data = rate_record("FAR", false_accepts, len(evaluation_impostor))
        frr_data = rate_record("FRR", false_rejects, len(evaluation_genuine))
        cluster = identity_cluster_interval(
            probabilities, y_test, evaluation_indices, threshold,
            args.bootstrap_repetitions,
        )
        evaluation_seconds = time.perf_counter() - evaluation_start
        is_final = model_label == "DO-FMBS final"
        gate_satisfied = (
            cluster["FAR"][1] <= DEPLOYMENT_FAR_LIMIT
            and cluster["FRR"][1] <= DEPLOYMENT_FRR_LIMIT
        )
        status = "SATISFIED" if is_final and gate_satisfied else "COMPARATOR RECORDED"
        clean_rows.append({
            "model": model_label,
            "protocol": "Claimed-identity face verification; identity-stratified calibration/evaluation",
            "split_seed": SPLIT_SEED,
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
            "acceptance_basis": "Final model cluster-bootstrap 95% upper bounds",
            "status": status,
        })
        runtime_rows.extend([
            {"component": f"{model_label} model load", "elapsed_seconds": model_load_seconds, "status": "RECORDED"},
            {"component": f"{model_label} clean inference", "elapsed_seconds": inference_seconds, "status": "RECORDED"},
            {"component": f"{model_label} threshold calibration", "elapsed_seconds": threshold_seconds, "status": "RECORDED"},
            {"component": f"{model_label} evaluation and bootstrap", "elapsed_seconds": evaluation_seconds, "status": "RECORDED"},
            {"component": f"{model_label} complete protocol", "elapsed_seconds": time.perf_counter() - model_total_start, "status": "RECORDED"},
        ])
        if is_final:
            final_gate_satisfied = gate_satisfied
            final_face_model = face_model
            final_threshold = threshold
            final_probabilities = probabilities

    if final_face_model is None or final_threshold is None or final_probabilities is None:
        raise RuntimeError("Final model evaluation state is missing")

    privacy_start = time.perf_counter()
    member_probabilities = final_face_model.predict(x_train_members, batch_size=64, verbose=0)
    nonmember_probabilities = final_face_model.predict(x_test[:100], batch_size=64, verbose=0)
    member_confidence = np.max(member_probabilities, axis=1)
    nonmember_confidence = np.max(nonmember_probabilities, axis=1)
    mia_threshold = 0.70
    member_detected = int(np.count_nonzero(member_confidence >= mia_threshold))
    nonmember_false = int(np.count_nonzero(nonmember_confidence >= mia_threshold))
    membership_correct = member_detected + (len(nonmember_confidence) - nonmember_false)
    membership_accuracy = membership_correct / (len(member_confidence) + len(nonmember_confidence))
    pa = 2.0 * max(0.0, membership_accuracy - 0.50)
    privacy_rows.append({
        "model": "DO-FMBS final",
        "membership_threshold": mia_threshold,
        "member_detected_numerator": member_detected,
        "member_denominator": len(member_confidence),
        "member_detection_rate": member_detected / len(member_confidence),
        "false_member_numerator": nonmember_false,
        "nonmember_denominator": len(nonmember_confidence),
        "false_member_rate": nonmember_false / len(nonmember_confidence),
        "membership_correct": membership_correct,
        "membership_total": len(member_confidence) + len(nonmember_confidence),
        "membership_accuracy": membership_accuracy,
        "PA": pa,
        "PA_managed_limit": PA_MANAGED_LIMIT,
        "status": "SATISFIED - MANAGED PRIVACY BAND" if pa <= PA_MANAGED_LIMIT else "PRIVACY ESCALATION ACTIVE",
    })
    runtime_rows.append({
        "component": "DO-FMBS membership-inference and PA evaluation",
        "elapsed_seconds": time.perf_counter() - privacy_start,
        "status": "RECORDED",
    })

    attack_start = time.perf_counter()
    evaluation_x = tf.convert_to_tensor(x_test[evaluation_indices])
    evaluation_y = tf.convert_to_tensor(y_test[evaluation_indices])
    with tf.GradientTape() as tape:
        tape.watch(evaluation_x)
        clean_tensor = final_face_model(evaluation_x, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(evaluation_y, clean_tensor)
    gradient = tape.gradient(loss, evaluation_x)
    clean_evaluation = clean_tensor.numpy()
    truth = y_test[evaluation_indices]
    originally_correct = np.argmax(clean_evaluation, axis=1) == truth
    clean_genuine_scores = clean_evaluation[np.arange(len(truth)), truth]
    clean_accepted = clean_genuine_scores >= final_threshold
    for epsilon in EPSILONS:
        condition_start = time.perf_counter()
        adversarial_x = evaluation_x if epsilon == 0 else tf.clip_by_value(
            evaluation_x + epsilon * tf.sign(gradient), -3.0, 3.0
        )
        adversarial_probabilities = final_face_model(adversarial_x, training=False).numpy()
        genuine = adversarial_probabilities[np.arange(len(truth)), truth]
        mask = np.ones(adversarial_probabilities.shape, dtype=bool)
        mask[np.arange(len(truth)), truth] = False
        impostor = adversarial_probabilities[mask]
        far_successes = int(np.count_nonzero(impostor >= final_threshold))
        frr_successes = int(np.count_nonzero(genuine < final_threshold))
        predicted = np.argmax(adversarial_probabilities, axis=1)
        classification_successes = int(np.count_nonzero(predicted[originally_correct] != truth[originally_correct]))
        rejection_successes = int(np.count_nonzero(genuine[clean_accepted] < final_threshold))
        far_data = rate_record("FAR", far_successes, len(impostor))
        frr_data = rate_record("FRR", frr_successes, len(genuine))
        asr_data = rate_record("classification_ASR", classification_successes, int(originally_correct.sum()))
        rejection_data = rate_record("verification_rejection_ASR", rejection_successes, int(clean_accepted.sum()))
        ai_rows.append({
            "model": "DO-FMBS final",
            "attack": "FGSM untargeted availability stress",
            "epsilon": epsilon,
            "AI": epsilon / 0.30,
            "frozen_threshold": final_threshold,
            **far_data,
            **frr_data,
            **asr_data,
            **rejection_data,
            "FAR_security_limit": DEPLOYMENT_FAR_LIMIT,
            "FAR_status": "SATISFIED" if far_data["FAR"] <= DEPLOYMENT_FAR_LIMIT else "SECURITY RESPONSE ACTIVE",
            "FRR_status": "CLEAN SATISFIED" if epsilon == 0 else "ATTACK-STRESS RECORDED",
            "measurement_status": "COMPLETE",
            "elapsed_seconds": time.perf_counter() - condition_start,
        })
    runtime_rows.append({
        "component": "DO-FMBS seven-level AI/FGSM sweep",
        "elapsed_seconds": time.perf_counter() - attack_start,
        "status": "RECORDED",
    })

    total_seconds = time.perf_counter() - total_start
    runtime_rows.insert(0, {
        "component": "Face data load and identity split",
        "elapsed_seconds": data_load_seconds,
        "status": "RECORDED",
    })
    runtime_rows.append({
        "component": "Complete operational biometric evaluation",
        "elapsed_seconds": total_seconds,
        "status": "RECORDED",
    })
    write_csv(output_dir / "clean_far_frr_acceptance.csv", clean_rows)
    write_csv(output_dir / "ai_conditioned_far_frr_asr.csv", ai_rows)
    write_csv(output_dir / "privacy_pa_evaluation.csv", privacy_rows)
    write_csv(output_dir / "evaluation_runtime.csv", runtime_rows)

    ended = datetime.now(ZoneInfo("Africa/Nairobi"))
    manifest = {
        "protocol": "DO-FMBS claimed-identity verification and adversarial availability evaluation",
        "started_africa_nairobi": started.isoformat(timespec="milliseconds"),
        "ended_africa_nairobi": ended.isoformat(timespec="milliseconds"),
        "elapsed_seconds": total_seconds,
        "split_seed": SPLIT_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "calibration_FAR_Wilson_upper_cap": args.calibration_far_upper_cap,
        "deployment_limits": {"FAR": DEPLOYMENT_FAR_LIMIT, "FRR": DEPLOYMENT_FRR_LIMIT},
        "accept_rule": "claimed identity accepted when face softmax score >= frozen model threshold",
        "impostor_policy": "all 61 wrong identity claims per evaluation probe",
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or "unreported",
        },
        "inputs": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [
                data_dir / "face_data.npz",
                data_dir / "baseline_model.keras",
                data_dir / "do_fmbs_model.keras",
                Path(__file__).resolve(),
            ]
        ],
        "final_status": "SATISFIED" if final_gate_satisfied else "CALIBRATION-TUNING ACTIVE",
        "final_status_basis": (
            "DO-FMBS FAR and FRR identity-cluster 95% upper bounds are within "
            "the declared 10% and 20% limits"
            if final_gate_satisfied
            else "The synchronized model requires a calibration-only operating-point update"
        ),
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Operational biometric evidence written to {output_dir.relative_to(ROOT)}")
    print(f"Elapsed seconds: {total_seconds:.6f}")
    final_status = "SATISFIED" if final_gate_satisfied else "CALIBRATION-TUNING ACTIVE"
    print(f"Final FAR/FRR status: {final_status}")
    if not final_gate_satisfied:
        raise RuntimeError(
            "Final DO-FMBS requires a calibration-only operating-point update"
        )


if __name__ == "__main__":
    main()

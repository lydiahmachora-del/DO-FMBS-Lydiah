"""Phase 5: independent classification, verification and robustness validation."""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from do_fmbs_core import (
    claimed_identity_scores,
    csv_write,
    json_dump,
    modality_probabilities,
    select_eer_threshold,
    set_global_determinism,
    threshold_curve,
    verification_metrics,
    wilson_interval,
)
from phase4_defense_mechanisms import face_fgsm, output_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("do_fmbs_data"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--robustness-epsilons", default="0.00,0.05,0.10,0.15,0.20")
    return parser.parse_args()


def classification_summary(labels: np.ndarray, probabilities: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    correct = int(np.count_nonzero(predictions == labels))
    low, high = wilson_interval(correct, len(labels))
    summary = {
        "samples": int(len(labels)), "correct": correct,
        "accuracy": float(report["accuracy"]), "accuracy_wilson_95": [low, high],
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
    return summary, predictions, confusion_matrix(labels, predictions)


def save_classification_artifacts(data_dir: Path, modality: str, labels: np.ndarray,
                                  predictions: np.ndarray, matrix: np.ndarray,
                                  summary: dict) -> list[str]:
    figures = data_dir / "figures"
    figures.mkdir(exist_ok=True)
    json_path = data_dir / f"{modality}_classification_report.json"
    csv_path = data_dir / f"{modality}_confusion_matrix.csv"
    figure_path = figures / f"{modality}_confusion_matrix.png"
    json_dump(json_path, summary)
    np.savetxt(csv_path, matrix, fmt="%d", delimiter=",")
    size = 8.0 if modality == "face" else 9.2
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    shown = np.log1p(matrix)
    image = ax.imshow(shown, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.045, label="log(1 + count)")
    ax.set(xlabel="Predicted identity", ylabel="True identity",
           title=f"{modality.title()} Classification Confusion Matrix (n={len(labels)})")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [str(json_path), str(csv_path), str(figure_path)]


def verification_protocol(model, modality: str, calibration_x: np.ndarray,
                          calibration_y: np.ndarray, test_x: np.ndarray,
                          test_y: np.ndarray, repetitions: int) -> tuple[dict, dict[str, np.ndarray]]:
    calibration_prob = modality_probabilities(model, modality, calibration_x)
    test_prob = modality_probabilities(model, modality, test_x)
    calibration_genuine, calibration_impostor, _ = claimed_identity_scores(calibration_prob, calibration_y)
    threshold, calibration_eer = select_eer_threshold(calibration_genuine, calibration_impostor)
    calibration_metrics = verification_metrics(
        calibration_prob, calibration_y, threshold, bootstrap_repetitions=repetitions
    )
    final_metrics = verification_metrics(
        test_prob, test_y, threshold, bootstrap_repetitions=repetitions
    )
    test_genuine, test_impostor, _ = claimed_identity_scores(test_prob, test_y)
    payload = {
        "modality": modality,
        "score": "softmax probability assigned to the claimed enrolled identity",
        "accept_rule": "accept when claimed-identity score is greater than or equal to the frozen threshold",
        "threshold_rule": "equal-error-rate operating point selected on the calibration partition only",
        "threshold_frozen_before_test": True,
        "calibration": calibration_metrics,
        "calibration_selected_point": calibration_eer,
        "final_test": final_metrics,
    }
    arrays = {
        "calibration_genuine": calibration_genuine,
        "calibration_impostor": calibration_impostor,
        "test_genuine": test_genuine,
        "test_impostor": test_impostor,
        "test_probabilities": test_prob,
    }
    return payload, arrays


def save_verification_artifacts(data_dir: Path, modality: str, payload: dict,
                                arrays: dict[str, np.ndarray]) -> list[str]:
    figures = data_dir / "figures"
    json_path = data_dir / f"{modality}_verification_report.json"
    counts_path = data_dir / f"{modality}_verification_counts.csv"
    scores_path = data_dir / f"{modality}_verification_scores.npz"
    json_dump(json_path, payload)
    final = payload["final_test"]
    csv_write(counts_path, [{
        "modality": modality, "threshold": final["threshold"],
        "genuine_accepts": final["genuine_accepts"], "genuine_rejects": final["genuine_rejects"],
        "genuine_attempts": final["genuine_attempts"], "impostor_accepts": final["impostor_accepts"],
        "impostor_rejects": final["impostor_rejects"], "impostor_attempts": final["impostor_attempts"],
        "FAR": final["FAR"], "FRR": final["FRR"], "TAR": final["TAR"],
        "ROC_AUC": final["ROC_AUC"], "EER": final["EER"],
    }])
    np.savez_compressed(scores_path, **arrays)

    score_figure = figures / f"{modality}_genuine_impostor_distribution.png"
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    ax.hist(arrays["test_impostor"], bins=60, density=True, alpha=0.70, label="Impostor claims", color="#C44E52")
    ax.hist(arrays["test_genuine"], bins=30, density=True, alpha=0.70, label="Genuine claims", color="#4C72B0")
    ax.axvline(final["threshold"], color="black", linestyle="--", label="Frozen threshold")
    ax.set(xlabel="Claimed-identity score", ylabel="Density",
           title=f"{modality.title()} Verification Score Distributions")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(score_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    roc_figure = figures / f"{modality}_verification_roc.png"
    truth = np.concatenate([np.ones(len(arrays["test_genuine"])), np.zeros(len(arrays["test_impostor"]))])
    scores = np.concatenate([arrays["test_genuine"], arrays["test_impostor"]])
    fpr, tpr, _ = roc_curve(truth, scores)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot(fpr, tpr, label=f"AUC = {final['ROC_AUC']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False acceptance rate", ylabel="True acceptance rate",
           title=f"{modality.title()} Verification ROC")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(roc_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    curve_figure = figures / f"{modality}_far_frr_threshold.png"
    rows = threshold_curve(arrays["calibration_genuine"], arrays["calibration_impostor"])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot([r["threshold"] for r in rows], [100 * r["FAR"] for r in rows], label="Calibration FAR")
    ax.plot([r["threshold"] for r in rows], [100 * r["FRR"] for r in rows], label="Calibration FRR")
    ax.axvline(final["threshold"], color="black", linestyle="--", label="Selected threshold")
    ax.set(xlabel="Decision threshold", ylabel="Rate (%)",
           title=f"{modality.title()} Calibration Operating Point")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(curve_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [str(json_path), str(counts_path), str(scores_path), str(score_figure), str(roc_figure), str(curve_figure)]


def face_robustness(model, face_x: np.ndarray, labels: np.ndarray,
                    finger_dim: int, epsilons: list[float]) -> list[dict[str, object]]:
    neutral = np.zeros((len(face_x), finger_dim), np.float32)
    clean_outputs = model({"face_input": face_x, "finger_input": neutral}, training=False)
    clean_pred = np.argmax(np.asarray(output_tensor(clean_outputs, "face_output", 0)), axis=1)
    eligible = clean_pred == labels
    rows = []
    for epsilon in epsilons:
        attacked = face_x if epsilon == 0 else face_fgsm(model, face_x, neutral, labels, epsilon)
        outputs = model({"face_input": attacked, "finger_input": neutral}, training=False)
        predictions = np.argmax(np.asarray(output_tensor(outputs, "face_output", 0)), axis=1)
        successful = int(np.count_nonzero(eligible & (predictions != labels)))
        rows.append({"epsilon": epsilon, "modality": "face", "attack": "FGSM",
                     "accuracy_all_samples": float(np.mean(predictions == labels)),
                     "eligible_clean_correct": int(np.count_nonzero(eligible)),
                     "successful_attacks": successful,
                     "attack_success_rate": float(successful / max(np.count_nonzero(eligible), 1))})
    return rows


def mcnemar_record(baseline_pred, final_pred, labels, modality: str) -> dict[str, object]:
    baseline_correct = baseline_pred == labels
    final_correct = final_pred == labels
    baseline_only = int(np.count_nonzero(baseline_correct & ~final_correct))
    final_only = int(np.count_nonzero(~baseline_correct & final_correct))
    discordant = baseline_only + final_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(baseline_only, final_only), discordant, 0.5).pvalue)
    return {"modality": modality, "test": "exact McNemar/binomial test on paired test decisions",
            "baseline_only_correct": baseline_only, "final_only_correct": final_only,
            "discordant": discordant, "p_value": p_value, "significant_at_0_05": p_value < 0.05}


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 identity-bootstrap repetitions")
    set_global_determinism()
    started = time.perf_counter()
    import tensorflow as tf

    with np.load(args.data_dir / "face_data.npz", allow_pickle=False) as face:
        face_cal, face_cal_y = face["X_calibration"].astype(np.float32), face["y_calibration"]
        face_test, face_test_y = face["X_test"].astype(np.float32), face["y_test"]
    with np.load(args.data_dir / "fingerprint_data.npz", allow_pickle=False) as finger:
        finger_cal, finger_cal_y = finger["X_calibration"].astype(np.float32), finger["y_calibration"]
        finger_test, finger_test_y = finger["X_test"].astype(np.float32), finger["y_test"]
    baseline = tf.keras.models.load_model(args.data_dir / "baseline_model.keras")
    final_model = tf.keras.models.load_model(args.data_dir / "do_fmbs_model.keras")

    artifacts = []
    modality_payloads = {}
    statistical = []
    predictions = {}
    for modality, calibration_x, calibration_y, test_x, test_y in (
        ("face", face_cal, face_cal_y, face_test, face_test_y),
        ("fingerprint", finger_cal, finger_cal_y, finger_test, finger_test_y),
    ):
        final_prob = modality_probabilities(final_model, modality, test_x)
        baseline_prob = modality_probabilities(baseline, modality, test_x)
        summary, final_pred, matrix = classification_summary(test_y, final_prob)
        baseline_summary, baseline_pred, _ = classification_summary(test_y, baseline_prob)
        summary["baseline_comparison"] = baseline_summary
        summary["classes"] = int(final_prob.shape[1])
        summary["identity_space"] = f"independent {modality} identities"
        artifacts.extend(save_classification_artifacts(
            args.data_dir, modality, test_y, final_pred, matrix, summary
        ))
        verification, arrays = verification_protocol(
            final_model, modality, calibration_x, calibration_y, test_x, test_y,
            args.bootstrap_repetitions,
        )
        artifacts.extend(save_verification_artifacts(args.data_dir, modality, verification, arrays))
        modality_payloads[modality] = {"classification": summary, "verification": verification}
        statistical.append(mcnemar_record(baseline_pred, final_pred, test_y, modality))
        predictions[modality] = (baseline_pred, final_pred)

    epsilons = [float(value) for value in args.robustness_epsilons.split(",")]
    finger_dim = int(final_model.get_layer("finger_input").input.shape[-1])
    robustness = {
        "scope": "FGSM evaluation on the face classification branch only",
        "baseline": face_robustness(baseline, face_test, face_test_y, finger_dim, epsilons),
        "do_fmbs": face_robustness(final_model, face_test, face_test_y, finger_dim, epsilons),
    }
    json_dump(args.data_dir / "face_robustness_report.json", robustness)
    artifacts.append(str(args.data_dir / "face_robustness_report.json"))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for name, marker in (("baseline", "o"), ("do_fmbs", "s")):
        rows = robustness[name]
        ax.plot([r["epsilon"] for r in rows], [100 * r["accuracy_all_samples"] for r in rows],
                marker=marker, label=name.replace("_", " ").title())
    ax.set(xlabel="FGSM epsilon", ylabel="Face classification accuracy (%)",
           title="Final Face-Branch Robustness Validation")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    robustness_figure = args.data_dir / "figures" / "face_robustness_validation.png"
    fig.savefig(robustness_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(robustness_figure))

    payload = {
        "phase": 5, "replay_id": os.environ.get("DO_FMBS_REPLAY_ID", "unspecified"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity_space_statement": "Face and fingerprint classes are independent populations and are never merged.",
        "face": modality_payloads["face"], "fingerprint": modality_payloads["fingerprint"],
        "statistical_validation": statistical, "robustness": robustness,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "artifacts": artifacts, "runtime_seconds": time.perf_counter() - started,
    }
    json_dump(args.data_dir / "phase5_validation.json", payload)
    print(f"Face classification accuracy: {payload['face']['classification']['accuracy']:.4f}")
    print(f"Fingerprint classification accuracy: {payload['fingerprint']['classification']['accuracy']:.4f}")
    for modality in ("face", "fingerprint"):
        result = payload[modality]["verification"]["final_test"]
        print(f"{modality.title()} FAR={result['impostor_accepts']}/{result['impostor_attempts']}={result['FAR']:.4%}; "
              f"FRR={result['genuine_rejects']}/{result['genuine_attempts']}={result['FRR']:.4%}")
    print(f"Phase 5 complete in {payload['runtime_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run authoritative Phase 5 validation on accepted Phase 2 and Phase 4 models."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

from do_fmbs_core import modality_probabilities, select_eer_threshold, wilson_interval
from phase3_authoritative_attacks import fgsm as phase3_fgsm

DOC_SHA256 = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"
P1_RUN = "phase1_final_candidate_20261123_182623"
P2_RUN = "phase2_authoritative_20261123_203617_remediation"
P3_RUN = "phase3_authoritative_20261123_220738"
P4_RUN = "phase4_authoritative_20261123_224549"
P5_EVAL_N = 100
EPSILONS = (0.00, 0.05, 0.10, 0.15, 0.20)
ALPHA = 0.05
BOOTSTRAP_REPETITIONS = 2000
PERMUTATION_REPETITIONS = 20000
SEED = 20261124
LATENCY_REPETITIONS = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_manifest(directory: Path, filename: str) -> dict:
    manifest = json.loads((directory / filename).read_text(encoding="utf-8"))
    bad = []
    for artifact in manifest.get("artifacts", []):
        item = directory / artifact["path"]
        if not item.is_file() or item.stat().st_size != artifact["bytes"] or sha256(item) != artifact["sha256"]:
            bad.append(artifact["path"])
    if bad:
        raise ValueError(f"Invalid source run artifacts in {filename}: {bad}")
    return manifest


def dependency_versions() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "scikit-learn", "keras", "tensorflow-cpu"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = "not-installed"
    try:
        values["tensorflow"] = importlib.metadata.version("tensorflow")
    except importlib.metadata.PackageNotFoundError:
        values["tensorflow"] = values.get("tensorflow-cpu", "not-installed")
    return values


def outputs(model, face: np.ndarray, finger: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = model({"face_input": np.asarray(face, np.float32),
                        "finger_input": np.asarray(finger, np.float32)}, training=False)
    if isinstance(prediction, dict):
        return np.asarray(prediction["face_output"]), np.asarray(prediction["finger_output"])
    return np.asarray(prediction[0]), np.asarray(prediction[1])


def classification_summary(y_true: np.ndarray, probabilities: np.ndarray, class_count: int) -> tuple[dict, np.ndarray]:
    labels = np.arange(class_count, dtype=np.int64)
    pred = np.asarray(probabilities).argmax(axis=1)
    report = classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)
    correct = int(np.count_nonzero(pred == y_true))
    ci = wilson_interval(correct, len(y_true))
    result = {
        "samples": int(len(y_true)), "correct": correct, "incorrect": int(len(y_true) - correct),
        "accuracy": float(accuracy_score(y_true, pred)), "accuracy_wilson_95": list(ci),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_average_class_rule": "all model output classes included, including classes with zero support",
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).astype(int).tolist(),
        "per_class": {str(i): {key: float(report[str(i)][key]) for key in ("precision", "recall", "f1-score", "support")}
                      for i in labels},
    }
    return result, pred.astype(np.int64)


def _map_source_indices(source_indices: np.ndarray, requested: np.ndarray) -> np.ndarray:
    positions = {int(value): pos for pos, value in enumerate(source_indices)}
    try:
        return np.asarray([positions[int(value)] for value in requested], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Claim probe is not present in held-out data: {error}") from error


def _claim_bundle(model, modality: str, calibration_x: np.ndarray, calibration_y: np.ndarray,
                  calibration_source_indices: np.ndarray, eval_x: np.ndarray, eval_y: np.ndarray,
                  eval_source_indices: np.ndarray, claims: dict[str, np.ndarray], classes: int) -> tuple[dict, dict]:
    calibration_prob = modality_probabilities(model, modality, calibration_x)
    evaluation_prob = modality_probabilities(model, modality, eval_x)
    cal_pos = _map_source_indices(calibration_source_indices, claims["calibration_claim_probe_indices"])
    eval_pos = _map_source_indices(eval_source_indices, claims["evaluation_claim_probe_indices"])
    cal_claim_ids = claims["calibration_claim_identity"].astype(np.int64)
    eval_claim_ids = claims["evaluation_claim_identity"].astype(np.int64)
    cal_genuine_mask = claims["calibration_claim_is_genuine"].astype(bool)
    eval_genuine_mask = claims["evaluation_claim_is_genuine"].astype(bool)
    if (np.any(cal_claim_ids < 0) or np.any(cal_claim_ids >= classes)
            or np.any(eval_claim_ids < 0) or np.any(eval_claim_ids >= classes)):
        raise ValueError(f"{modality} claim IDs exceed the model class range")
    cal_scores = calibration_prob[cal_pos, cal_claim_ids]
    eval_scores = evaluation_prob[eval_pos, eval_claim_ids]
    cal_genuine, cal_impostor = cal_scores[cal_genuine_mask], cal_scores[~cal_genuine_mask]
    threshold, cal_point = select_eer_threshold(cal_genuine, cal_impostor)
    final = verification_summary(eval_scores, eval_genuine_mask, threshold)
    final["identity_cluster_bootstrap_95"] = verification_bootstrap(
        evaluation_prob, eval_y, eval_source_indices, claims["evaluation_probe_indices"], threshold,
        BOOTSTRAP_REPETITIONS, SEED + (11 if modality == "face" else 23))
    final["genuine_attempts"] = int(eval_genuine_mask.sum())
    final["impostor_attempts"] = int((~eval_genuine_mask).sum())
    final["genuine_accepts"] = int(np.count_nonzero(eval_scores[eval_genuine_mask] >= threshold))
    final["genuine_rejects"] = int(final["genuine_attempts"] - final["genuine_accepts"])
    final["impostor_accepts"] = int(np.count_nonzero(eval_scores[~eval_genuine_mask] >= threshold))
    final["impostor_rejects"] = int(final["impostor_attempts"] - final["impostor_accepts"])
    ci_far = wilson_interval(final["impostor_accepts"], final["impostor_attempts"])
    ci_frr = wilson_interval(final["genuine_rejects"], final["genuine_attempts"])
    final["FAR_wilson_95"] = list(ci_far)
    final["FRR_wilson_95"] = list(ci_frr)
    payload = {
        "modality": modality,
        "score": "model softmax probability assigned to the claimed enrolled identity",
        "acceptance_rule": "accept if claimed-identity score >= threshold",
        "threshold_rule": "minimize absolute calibration FAR-FRR difference; threshold selected on Phase 1 calibration claims and frozen before evaluation claims",
        "threshold_frozen_before_evaluation": True,
        "threshold": float(threshold), "calibration_eer_point": cal_point,
        "calibration_claims": {"genuine": int(cal_genuine_mask.sum()), "impostor": int((~cal_genuine_mask).sum()),
                               "genuine_scores": cal_genuine.astype(np.float32).tolist(),
                               "impostor_scores": cal_impostor.astype(np.float32).tolist()},
        "evaluation": final,
    }
    evidence = {
        "calibration_probabilities": calibration_prob.astype(np.float32),
        "evaluation_probabilities": evaluation_prob.astype(np.float32),
        "calibration_claim_probe_ids": claims["calibration_claim_probe_indices"],
        "calibration_claim_identities": cal_claim_ids,
        "calibration_is_genuine": cal_genuine_mask,
        "calibration_claim_scores": cal_scores.astype(np.float32),
        "evaluation_claim_probe_ids": claims["evaluation_claim_probe_indices"],
        "evaluation_claim_identities": eval_claim_ids,
        "evaluation_is_genuine": eval_genuine_mask,
        "evaluation_claim_scores": eval_scores.astype(np.float32),
        "evaluation_probe_ids": claims["evaluation_probe_indices"],
        "evaluation_labels": eval_y,
    }
    return payload, evidence


def verification_summary(scores: np.ndarray, genuine_mask: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score
    accepted = scores >= threshold
    genuine_n = int(genuine_mask.sum())
    impostor_n = int((~genuine_mask).sum())
    rejects = int(np.count_nonzero(genuine_mask & ~accepted))
    accepts = int(np.count_nonzero(~genuine_mask & accepted))
    truth = genuine_mask.astype(np.int8)
    fpr, tpr, roc_thresholds = roc_curve(truth, scores)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return {"threshold": float(threshold), "genuine_attempts": genuine_n,
            "impostor_attempts": impostor_n, "genuine_accepts": genuine_n - rejects,
            "genuine_rejects": rejects, "impostor_accepts": accepts,
            "impostor_rejects": impostor_n - accepts,
            "FAR": float(accepts / impostor_n), "FRR": float(rejects / genuine_n),
            "TAR": float((genuine_n - rejects) / genuine_n),
            "ROC_AUC": float(roc_auc_score(truth, scores)),
            "EER": float((fpr[i] + fnr[i]) / 2.0), "EER_threshold": float(roc_thresholds[i])}


def verification_bootstrap(probabilities: np.ndarray, labels: np.ndarray, source_indices: np.ndarray,
                           probe_ids: np.ndarray,
                           threshold: float, repetitions: int, seed: int) -> dict:
    classes = probabilities.shape[1]
    rows = _map_source_indices(source_indices, probe_ids)
    probe_labels = labels[rows]
    identities = np.unique(probe_labels)
    rng = np.random.default_rng(seed)
    cluster_counts = {}
    for identity in identities:
        ids = np.flatnonzero(probe_labels == identity)
        genuine_scores = probabilities[rows[ids], identity]
        accepted_impostors = np.zeros(len(ids), dtype=np.int64)
        for j, row in enumerate(rows[ids]):
            accepted_impostors[j] = int(np.count_nonzero(np.delete(probabilities[row], identity) >= threshold))
        cluster_counts[int(identity)] = (int(np.count_nonzero(genuine_scores < threshold)),
                                         int(accepted_impostors.sum()), len(ids))
    far, frr = np.empty(repetitions), np.empty(repetitions)
    for b in range(repetitions):
        sampled = rng.choice(identities, size=len(identities), replace=True)
        rejected = accepted = genuine_total = 0
        for identity in sampled:
            reject_i, accept_i, n_i = cluster_counts[int(identity)]
            rejected += reject_i; accepted += accept_i; genuine_total += n_i
        far[b] = accepted / (genuine_total * (classes - 1))
        frr[b] = rejected / genuine_total
    return {"resampling_unit": "identity cluster; each sampled identity brings all its evaluation probes and paired claims",
            "repetitions": repetitions, "seed": seed,
            "FAR_95": np.quantile(far, [0.025, 0.975]).astype(float).tolist(),
            "FRR_95": np.quantile(frr, [0.025, 0.975]).astype(float).tolist()}


def paired_cluster_comparison(y: np.ndarray, baseline_pred: np.ndarray, candidate_pred: np.ndarray,
                              seed: int) -> dict:
    identities = np.unique(y)
    deltas = np.asarray([np.count_nonzero((candidate_pred == y) & (y == identity))
                         - np.count_nonzero((baseline_pred == y) & (y == identity))
                         for identity in identities], dtype=np.int64)
    n = len(y)
    observed = int(deltas.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=(PERMUTATION_REPETITIONS, len(deltas)))
    permuted = signs @ deltas
    p_value = float((1 + np.count_nonzero(np.abs(permuted) >= abs(observed))) /
                    (PERMUTATION_REPETITIONS + 1))
    rng = np.random.default_rng(seed + 1)
    boot = np.empty(BOOTSTRAP_REPETITIONS)
    cluster_rows = [np.flatnonzero(y == identity) for identity in identities]
    for i in range(BOOTSTRAP_REPETITIONS):
        selected = rng.integers(0, len(identities), len(identities))
        sampled_rows = np.concatenate([cluster_rows[j] for j in selected])
        boot[i] = (np.count_nonzero(candidate_pred[sampled_rows] == y[sampled_rows])
                   - np.count_nonzero(baseline_pred[sampled_rows] == y[sampled_rows])) / len(sampled_rows)
    base_correct = baseline_pred == y
    candidate_correct = candidate_pred == y
    base_only = int(np.count_nonzero(base_correct & ~candidate_correct))
    candidate_only = int(np.count_nonzero(candidate_correct & ~base_correct))
    discordant = base_only + candidate_only
    mcnemar_p = 1.0 if not discordant else float(binomtest(min(base_only, candidate_only), discordant).pvalue)
    diff = observed / n
    return {"test": "paired identity-cluster sign-flip permutation on accuracy difference",
            "unit": "identity cluster; rows and paired baseline/candidate predictions kept together",
            "identities": int(len(identities)), "samples": int(n),
            "baseline_only_correct": base_only, "candidate_only_correct": candidate_only,
            "paired_accuracy_difference": float(diff), "difference_percentage_points": float(100 * diff),
            "paired_cluster_bootstrap_95": np.quantile(boot, [0.025, 0.975]).astype(float).tolist(),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS, "permutation_repetitions": PERMUTATION_REPETITIONS,
            "seed": seed, "significance_level": ALPHA, "p_value": p_value,
            "significant": bool(p_value < ALPHA),
            "effect_size": "paired accuracy risk difference with identity-cluster bootstrap CI",
            "mcnemar_record_level_sensitivity_p": mcnemar_p,
            "mcnemar_note": "exact paired-record sensitivity analysis; clustered permutation is primary because identities may repeat"}


def robustness(model, name: str, face: np.ndarray, finger: np.ndarray, labels: np.ndarray,
               clean_pred: np.ndarray) -> tuple[list[dict], dict[str, np.ndarray]]:
    clean_mask = clean_pred == labels
    denom = int(clean_mask.sum())
    if denom == 0:
        raise RuntimeError(f"{name}: no clean-correct face samples; ASR undefined")
    rows, evidence = [], {}
    for epsilon in EPSILONS:
        adv = face.copy() if epsilon == 0 else phase3_fgsm(model, face, finger, labels, epsilon)
        attacked_prob, _ = outputs(model, adv, finger)
        pred = attacked_prob.argmax(axis=1)
        success = clean_mask & (pred != labels)
        rows.append({"epsilon": epsilon, "attack": "FGSM", "modality": "face",
                     "clean_correct_eligible": denom, "attack_successes": int(success.sum()),
                     "attack_success_rate": float(success.sum() / denom),
                     "attack_resistance_1_minus_asr": float(1.0 - success.sum() / denom),
                     "evaluation_samples": int(len(labels)),
                     "accuracy_under_attack": float(np.mean(pred == labels)),
                     "robust_error_1_minus_accuracy": float(1 - np.mean(pred == labels)),
                     "perturbation_linf_max": float(np.max(np.abs(adv - face), initial=0.0)),
                     "projection_bound_satisfied": bool(np.max(np.abs(adv - face), initial=0.0) <= epsilon + 1e-6)})
        evidence[f"{name}_eps_{epsilon:.2f}_adversarial_face"] = adv.astype(np.float32)
        evidence[f"{name}_eps_{epsilon:.2f}_predictions"] = pred.astype(np.int64)
        evidence[f"{name}_eps_{epsilon:.2f}_success_mask"] = success
    return rows, evidence


def latency_measurements(model, face: np.ndarray, finger: np.ndarray) -> dict:
    import tensorflow as tf
    inputs = {"face_input": tf.convert_to_tensor(face, tf.float32),
              "finger_input": tf.convert_to_tensor(finger, tf.float32)}
    for _ in range(3):
        model(inputs, training=False)
    durations = []
    for _ in range(LATENCY_REPETITIONS):
        start = time.perf_counter()
        result = model(inputs, training=False)
        if isinstance(result, dict):
            _ = result["face_output"].numpy(), result["finger_output"].numpy()
        else:
            _ = result[0].numpy(), result[1].numpy()
        durations.append(time.perf_counter() - start)
    return {"evaluation_records_per_call": len(face), "warmup_calls": 3,
            "timed_repetitions": LATENCY_REPETITIONS,
            "batch_latency_seconds_p50": float(np.median(durations)),
            "batch_latency_seconds_p95": float(np.quantile(durations, 0.95)),
            "per_record_latency_seconds_p50_derived": float(np.median(durations) / len(face)),
            "measurement": "wall-clock synchronous model call; TensorFlow CPU runtime; 100 paired records"}


def artifact_manifest(run_dir: Path, parents: dict) -> dict:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"phase5_manifest.json", "phase5_verification_report.json",
                                               "phase5_verifier_stdout.json"}:
            artifacts.append({"path": path.relative_to(run_dir).as_posix(), "bytes": path.stat().st_size,
                              "sha256": sha256(path)})
    return {"phase": 5, "run_id": run_dir.name, "status": "PHASE5-ARTIFACTS-PERSISTED",
            "source_document_sha256": DOC_SHA256, "parent_manifests": parents,
            "verification_report_excluded": "Independent verifier report and stdout are excluded to avoid self-referential hashes; verifier status and checks are persisted in those outputs.",
            "artifacts": artifacts}


def run(root: Path, output_root: Path) -> Path:
    import tensorflow as tf
    root, output_root = root.resolve(), output_root.resolve()
    docx = root / "docs" / "reference" / "CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION.docx"
    if sha256(docx).upper() != DOC_SHA256:
        raise ValueError("Authoritative Chapter Four DOCX hash does not match the accepted project hash")
    runtime = root / "runtime_recording"
    p1, p2, p3, p4 = (runtime / name for name in (P1_RUN, P2_RUN, P3_RUN, P4_RUN))
    p1m = verify_manifest(p1, "phase1_manifest.json")
    p2m = verify_manifest(p2, "phase2_manifest.json")
    p3m = verify_manifest(p3, "phase3_manifest.json")
    p4m = verify_manifest(p4, "phase4_manifest.json")
    if p1m.get("verification", {}).get("status") != "PASS" or p1m.get("replay_id") != P1_RUN:
        raise ValueError("Phase 1 source is not the accepted passing replay")
    if p2m.get("run_id") != P2_RUN or p3m.get("run_id") != P3_RUN or p4m.get("run_id") != P4_RUN:
        raise ValueError("A Phase 1–4 run identifier does not match its required accepted run")
    p3cfg = json.loads((p3 / "phase3_config.json").read_text(encoding="utf-8"))
    if p3cfg.get("source_phase2_model_sha256") != sha256(p2 / "final_global_model.keras"):
        raise ValueError("Phase 3 does not reference this accepted Phase 2 checkpoint")
    p4_report = json.loads((p4 / "phase4_verification_report.json").read_text(encoding="utf-8"))
    if p4_report.get("status") != "PASS":
        raise ValueError("Phase 4 authoritative run has no passing independent verification")
    p4config = json.loads((p4 / "phase4_config.json").read_text(encoding="utf-8"))
    if p4config.get("source_phase2_checkpoint_sha256") != sha256(p2 / "final_global_model.keras"):
        raise ValueError("Phase 4 combined candidate does not trace to the accepted Phase 2 baseline")
    baseline_path = p2 / "final_global_model.keras"
    candidate_path = p4 / "combined_model.keras"
    p2_model_entry = next(a for a in p2m["artifacts"] if a["path"] == "final_global_model.keras")
    p4_candidate_entry = next(a for a in p4m["artifacts"] if a["path"] == "combined_model.keras")
    if sha256(baseline_path) != p2_model_entry["sha256"] or sha256(candidate_path) != p4_candidate_entry["sha256"]:
        raise ValueError("A selected model checkpoint does not match its authoritative manifest")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = output_root / f"phase5_authoritative_{stamp}"
    suffix = 1
    while out.exists():
        out = output_root / f"phase5_authoritative_{stamp}_{suffix}"; suffix += 1
    out.mkdir(parents=True)
    snapshot = out / "source_snapshot"; snapshot.mkdir()
    source_files = [Path(__file__).resolve(), Path(__file__).resolve().with_name("verify_phase5_authoritative_run.py"),
                    Path(__file__).resolve().with_name("phase3_authoritative_attacks.py"),
                    Path(__file__).resolve().with_name("do_fmbs_core.py")]
    for source in source_files:
        shutil.copy2(source, snapshot / source.name)
    shutil.copy2(baseline_path, out / "baseline_model.keras")
    shutil.copy2(candidate_path, out / "combined_dofmbs_model.keras")
    started = time.perf_counter()

    with np.load(p1 / "face_data.npz", allow_pickle=False) as f, np.load(p1 / "fingerprint_data.npz", allow_pickle=False) as g, \
            np.load(p4 / "evaluation_records.npz", allow_pickle=False) as p4eval:
        arrays = {
            "face_x": f["X_test"].astype(np.float32), "face_y": f["y_test"].astype(np.int64),
            "face_source_indices": f["test_indices"].astype(np.int64),
            "finger_x": g["X_test"].astype(np.float32), "finger_y": g["y_test"].astype(np.int64),
            "finger_source_indices": g["test_indices"].astype(np.int64),
            "face_eval_x": p4eval["face_test_x"].astype(np.float32),
            "face_eval_y": p4eval["face_test_y"].astype(np.int64),
            "finger_eval_x": p4eval["finger_test_x"].astype(np.float32),
            "finger_eval_y": p4eval["finger_test_y"].astype(np.int64),
            "face_eval_ids": p4eval["face_eval_ids"].astype(np.int64),
            "finger_eval_ids": p4eval["finger_eval_ids"].astype(np.int64),
            "face_classes": int(f["n_classes"][0]), "finger_classes": int(g["n_classes"][0]),
        }
    if not (np.array_equal(arrays["face_eval_x"], arrays["face_x"][:P5_EVAL_N])
            and np.array_equal(arrays["face_eval_y"], arrays["face_y"][:P5_EVAL_N])
            and np.array_equal(arrays["finger_eval_x"], arrays["finger_x"][:P5_EVAL_N])
            and np.array_equal(arrays["finger_eval_y"], arrays["finger_y"][:P5_EVAL_N])
            and np.array_equal(arrays["face_eval_ids"], arrays["face_source_indices"][:P5_EVAL_N])
            and np.array_equal(arrays["finger_eval_ids"], arrays["finger_source_indices"][:P5_EVAL_N])):
        raise ValueError("Phase 4 evaluation records differ from the first 100 accepted Phase 1 test rows")
    np.savez_compressed(out / "evaluation_records.npz", **{k: v for k, v in arrays.items() if isinstance(v, np.ndarray)})

    with np.load(p1 / "verification" / "face_claims.npz", allow_pickle=False) as z:
        face_claims = {k: z[k].copy() for k in z.files}
    with np.load(p1 / "verification" / "fingerprint_claims.npz", allow_pickle=False) as z:
        finger_claims = {k: z[k].copy() for k in z.files}
    np.savez_compressed(out / "verification_claim_records.npz",
                        **{f"face_{k}": v for k, v in face_claims.items()},
                        **{f"fingerprint_{k}": v for k, v in finger_claims.items()})

    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    model_load_times = {}
    t = time.perf_counter(); baseline = tf.keras.models.load_model(baseline_path, compile=False)
    model_load_times["baseline"] = time.perf_counter() - t
    t = time.perf_counter(); candidate = tf.keras.models.load_model(candidate_path, compile=False)
    model_load_times["dofmbs_combined"] = time.perf_counter() - t
    modelset = {"baseline": baseline, "dofmbs_combined": candidate}
    metrics, evidence, stats, robustness_results, robustness_evidence, performance = {}, {}, {}, {}, {}, {}
    for name, model in modelset.items():
        face_prob, finger_prob = outputs(model, arrays["face_eval_x"], arrays["finger_eval_x"])
        face_summary, face_pred = classification_summary(arrays["face_eval_y"], face_prob, arrays["face_classes"])
        finger_summary, finger_pred = classification_summary(arrays["finger_eval_y"], finger_prob, arrays["finger_classes"])
        metrics[name] = {"face_classification": face_summary, "fingerprint_classification": finger_summary}
        evidence[f"{name}_face_probabilities"] = face_prob.astype(np.float32)
        evidence[f"{name}_fingerprint_probabilities"] = finger_prob.astype(np.float32)
        evidence[f"{name}_face_predictions"] = face_pred
        evidence[f"{name}_fingerprint_predictions"] = finger_pred
        stats[name] = {}
        for modality, x, y, source_ids, claims, nclasses in (
            ("face", arrays["face_x"], arrays["face_y"], arrays["face_source_indices"], face_claims, arrays["face_classes"]),
            ("fingerprint", arrays["finger_x"], arrays["finger_y"], arrays["finger_source_indices"], finger_claims, arrays["finger_classes"]),
        ):
            verification, ver_evidence = _claim_bundle(model, modality, x, y, source_ids,
                                                        x, y, source_ids, claims, nclasses)
            metrics[name][f"{modality}_verification"] = verification
            evidence.update({f"{name}_{modality}_{key}": val for key, val in ver_evidence.items()})
        robust_rows, attack_evidence = robustness(model, name, arrays["face_eval_x"], arrays["finger_eval_x"],
                                                   arrays["face_eval_y"], face_pred)
        robustness_results[name] = robust_rows
        robustness_evidence.update(attack_evidence)
        performance[name] = {
            "model_parameters": int(model.count_params()),
            "checkpoint_size_bytes": int((baseline_path if name == "baseline" else candidate_path).stat().st_size),
            "checkpoint_load_seconds_measured": float(model_load_times[name]),
            "inference_latency": latency_measurements(model, arrays["face_eval_x"], arrays["finger_eval_x"]),
        }

    comparisons = {}
    for modality, labels_key in (("face", "face_eval_y"), ("fingerprint", "finger_eval_y")):
        y = arrays[labels_key]
        if modality == "face":
            bpred, cpred = evidence["baseline_face_predictions"], evidence["dofmbs_combined_face_predictions"]
        else:
            bpred, cpred = evidence["baseline_fingerprint_predictions"], evidence["dofmbs_combined_fingerprint_predictions"]
        comparisons[modality] = paired_cluster_comparison(y, bpred, cpred,
                                                           SEED + (31 if modality == "face" else 37))
        comparisons[modality]["baseline_accuracy"] = metrics["baseline"]["face_classification" if modality == "face" else "fingerprint_classification"]["accuracy"]
        comparisons[modality]["candidate_accuracy"] = metrics["dofmbs_combined"]["face_classification" if modality == "face" else "fingerprint_classification"]["accuracy"]

    all_robustness = {name: rows for name, rows in robustness_results.items()}
    np.savez_compressed(out / "classification_and_verification_evidence.npz", **evidence)
    np.savez_compressed(out / "robustness_evidence.npz", **robustness_evidence)
    phase2_config = json.loads((p2 / "phase2_config.json").read_text(encoding="utf-8"))
    phase4_runtime = json.loads((p4 / "phase4_runtime.json").read_text(encoding="utf-8"))
    param_count = performance["baseline"]["model_parameters"]
    p2_transfers = int(phase2_config["rounds"] * phase2_config["clients"] * 2)
    payload_bytes = int(param_count * 4 * p2_transfers)
    overhead = {
        "baseline": performance["baseline"], "dofmbs_combined": performance["dofmbs_combined"],
        "phase2_training_seconds_existing_record": float(phase2_config["runtime_seconds"]),
        "phase4_training_and_evaluation_seconds_existing_record": float(phase4_runtime["training_and_evaluation_seconds"]),
        "phase2_raw_float32_weight_transfer_equivalent": {
            "bytes": payload_bytes, "mib": payload_bytes / (1024 ** 2), "transfers": p2_transfers,
            "derivation": "model parameter count x 4 raw float32 bytes x 12 rounds x 5 clients x download/upload",
            "status": "derived payload estimate; no network transfer was measured"},
        "memory_usage": {"value": None, "status": "not reliably available in this execution environment"},
    }
    run_id = out.name
    elapsed = time.perf_counter() - started
    result = {
        "phase": 5, "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_phase1_run": P1_RUN, "source_phase2_run": P2_RUN,
        "source_phase3_run": P3_RUN, "source_phase4_run": P4_RUN,
        "evaluation": {"classification_and_robustness_records": P5_EVAL_N,
                       "face_record_ids": arrays["face_eval_ids"].astype(int).tolist(),
                       "fingerprint_record_ids": arrays["finger_eval_ids"].astype(int).tolist(),
                       "same_records_across_models": True, "face_fingerprint_identity_correspondence": False},
        "models": metrics,
        "statistical_comparisons": comparisons,
        "robustness": {"attack": "Phase-3-compatible untargeted FGSM in standardized face feature space",
                       "fingerprint_attack": "unsupported/not evaluated; fingerprint inputs unchanged",
                       "epsilon_values": list(EPSILONS), "models": all_robustness},
        "computational_measurements": overhead,
        "runtime_seconds": float(elapsed),
    }
    write_json(out / "phase5_metrics.json", result)
    np.savez_compressed(out / "prediction_labels.npz",
                        face_labels=arrays["face_eval_y"], fingerprint_labels=arrays["finger_eval_y"],
                        face_record_ids=arrays["face_eval_ids"], fingerprint_record_ids=arrays["finger_eval_ids"],
                        baseline_face_predictions=evidence["baseline_face_predictions"],
                        candidate_face_predictions=evidence["dofmbs_combined_face_predictions"],
                        baseline_fingerprint_predictions=evidence["baseline_fingerprint_predictions"],
                        candidate_fingerprint_predictions=evidence["dofmbs_combined_fingerprint_predictions"])
    config = {
        "phase": 5, "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_document_sha256": DOC_SHA256,
        "source_phase1_run": P1_RUN, "source_phase1_manifest_sha256": sha256(p1 / "phase1_manifest.json"),
        "source_phase2_run": P2_RUN, "source_phase2_manifest_sha256": sha256(p2 / "phase2_manifest.json"),
        "baseline_checkpoint_sha256": sha256(baseline_path),
        "source_phase3_run": P3_RUN, "source_phase3_manifest_sha256": sha256(p3 / "phase3_manifest.json"),
        "source_phase4_run": P4_RUN, "source_phase4_manifest_sha256": sha256(p4 / "phase4_manifest.json"),
        "candidate_checkpoint_sha256": sha256(candidate_path),
        "source_snapshot_sha256": {p.name: sha256(snapshot / p.name) for p in source_files},
        "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                             capture_output=True, text=True).stdout.strip(),
        "dependencies": dependency_versions(), "platform": platform.platform(),
        "seeds": {"global_determinism": SEED, "classification_cluster_bootstrap": SEED + 1,
                  "face_sign_flip": SEED + 31, "fingerprint_sign_flip": SEED + 37,
                  "face_verification_bootstrap": SEED + 11, "fingerprint_verification_bootstrap": SEED + 23},
        "methodology": {
            "classification_scope": "first 100 positional paired test rows from accepted Phase 1/4 records; face and fingerprint labels independent",
            "robustness_epsilon_values": list(EPSILONS),
            "robustness_epsilon_source": "Chapter Four Phase 5 range 0.00–0.20; step values follow existing Phase 5 interface 0.05",
            "attack": "Phase 3 FGSM; untargeted face-only gradient sign step, no feature clipping, paired fingerprint held unchanged",
            "asr_denominator": "each model's clean-correct face rows, fixed across the epsilon sweep",
            "verification_claims": "accepted Phase 1 saved claim arrays, claim enumeration and calibration/evaluation partitions; independent thresholds per model and modality",
            "verification_threshold_selection": "minimum absolute calibration FAR-FRR difference; threshold frozen before evaluation claims",
            "statistical_test": "paired identity-cluster sign-flip permutation on accuracy difference; cluster-bootstrap paired CI",
            "statistical_alpha": ALPHA, "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "permutation_repetitions": PERMUTATION_REPETITIONS,
            "verification_bootstrap": "identity cluster, preserving every probe and claim associated with the sampled identity",
            "technical_decisions": ["Use Phase 4's verified first-100 paired evaluation records for clean classification and robustness.",
                                    "Evaluate both face and fingerprint classification on the same first-100 paired records; do not infer shared identities.",
                                    "Use Phase 1's full frozen calibration/evaluation claim sets for operational verification; thresholds are estimated separately for each model.",
                                    "Report fixed 62/100 output-class macro averages, including zero-support classes, for like-for-like class averaging."],
        },
        "limitations": ["Face and fingerprint populations are independent; no person-level cross-modal fusion claim.",
                        "FGSM robustness pipeline perturbs face features only; fingerprint adversarial robustness is unavailable.",
                        "Softmax claim scoring is a closed-set claimed-identity protocol, not liveness/spoof detection.",
                        "The fingerprint data are synthetic; findings do not establish real fingerprint spoof resistance.",
                        "Inference latency was measured on the current CPU runtime and small fixed batch; it is not deployment performance.",
                        "No memory metric or network communication measurement was available; the raw weight payload is derived only."],
    }
    write_json(out / "phase5_config.json", config)
    write_json(out / "phase5_execution_log.json", {"run_id": run_id, "runtime_seconds": elapsed,
               "timing_scope": "wall time from immediately before loading the two checkpoints through creation of classification and robustness evidence archives",
               "excludes": ["parent-manifest preflight", "checkpoint copying", "final metrics/config/manifest serialization", "independent verifier"],
               "note": "This is a scoped execution interval, not total end-to-end wall time. No figures were generated."})
    parents = {"phase1_manifest_sha256": config["source_phase1_manifest_sha256"],
               "phase2_manifest_sha256": config["source_phase2_manifest_sha256"],
               "phase3_manifest_sha256": config["source_phase3_manifest_sha256"],
               "phase4_manifest_sha256": config["source_phase4_manifest_sha256"]}
    write_json(out / "phase5_manifest.json", artifact_manifest(out, parents))
    del modelset, baseline, candidate
    tf.keras.backend.clear_session(); gc.collect()
    verifier_path = Path(__file__).resolve().with_name("verify_phase5_authoritative_run.py")
    verified = subprocess.run([sys.executable, str(verifier_path), str(out)], cwd=root,
                              capture_output=True, text=True, check=False)
    (out / "phase5_verifier_stdout.json").write_text(verified.stdout, encoding="utf-8")
    if verified.returncode != 0:
        (out / "phase5_verifier_stderr.txt").write_text(verified.stderr, encoding="utf-8")
        raise RuntimeError(f"Phase 5 independent verifier failed; inspect {out / 'phase5_verifier_stderr.txt'}")
    verification = json.loads(verified.stdout)
    write_json(out / "phase5_verification_report.json", verification)
    # The verifier stdout and report are documented outputs, intentionally
    # excluded from the self-referential artifact inventory.
    if verification.get("status") != "PASS":
        raise RuntimeError(f"Phase 5 independent verification failed: {verification.get('failed_checks')}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "runtime_recording")
    args = parser.parse_args()
    run_dir = run(args.root, args.output_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproducible Phase 4 defense comparison on the accepted Phase 2 model."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

SOURCE_DOC_SHA256 = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"
PHASE1_RUN = "phase1_final_candidate_20261123_182623"
PHASE2_RUN = "phase2_authoritative_20261123_203617_remediation"
PHASE3_RUN = "phase3_authoritative_20261123_220738"
EVAL_N = 100
SEED = 42
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.001
FGSM_EPSILON = 0.05
DP_EPSILON = 0.5
DP_NOISE_STDDEV = 0.001 / DP_EPSILON
COMBINED_NOISE_STDDEV = 0.0003


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def assert_manifest_files(directory: Path, manifest_name: str) -> dict:
    manifest_path = directory / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = []
    for artifact in manifest.get("artifacts", []):
        item = directory / artifact["path"]
        if not item.is_file() or item.stat().st_size != artifact["bytes"] or sha256(item) != artifact["sha256"]:
            bad.append(artifact["path"])
    if bad:
        raise ValueError(f"Source run {manifest_name} has invalid artifacts: {bad}")
    return manifest


def dependencies() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "scikit-learn", "keras", "tensorflow-cpu"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    try:
        result["tensorflow"] = importlib.metadata.version("tensorflow")
    except importlib.metadata.PackageNotFoundError:
        result["tensorflow"] = result.get("tensorflow-cpu", "not-installed")
    return result


def model_outputs(model, face: np.ndarray, finger: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = model({"face_input": np.asarray(face, np.float32),
                    "finger_input": np.asarray(finger, np.float32)}, training=False)
    if isinstance(output, dict):
        return np.asarray(output["face_output"]), np.asarray(output["finger_output"])
    return np.asarray(output[0]), np.asarray(output[1])


def face_fgsm(model, face: np.ndarray, finger: np.ndarray, labels: np.ndarray,
              epsilon: float = FGSM_EPSILON) -> tuple[np.ndarray, np.ndarray]:
    """Phase-3-compatible untargeted sign step on standardized face features."""
    import tensorflow as tf
    x = tf.convert_to_tensor(face, tf.float32)
    paired_finger = tf.convert_to_tensor(finger, tf.float32)
    y = tf.convert_to_tensor(labels, tf.int64)
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model({"face_input": x, "finger_input": paired_finger}, training=False)
        face_prob = prediction["face_output"] if isinstance(prediction, dict) else prediction[0]
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, face_prob))
    gradient = tape.gradient(loss, x)
    if gradient is None:
        raise RuntimeError("FGSM face-input gradient is missing")
    return (x + tf.cast(epsilon, tf.float32) * tf.sign(gradient)).numpy(), gradient.numpy()


def _named_branch_norms(variables, gradients, tf) -> dict[str, float]:
    result = {"face": 0.0, "fingerprint": 0.0, "shared": 0.0}
    for variable, gradient in zip(variables, gradients):
        if gradient is None:
            continue
        name = variable.name.lower()
        key = "face" if "face_" in name else ("fingerprint" if "finger_" in name else "shared")
        result[key] += float(tf.reduce_sum(tf.square(tf.cast(gradient, tf.float64))).numpy())
    return {key: float(np.sqrt(value)) for key, value in result.items()}


def _objective(face_y, finger_y, output, tf):
    face_prob = output["face_output"] if isinstance(output, dict) else output[0]
    finger_prob = output["finger_output"] if isinstance(output, dict) else output[1]
    face_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(face_y, face_prob))
    finger_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(finger_y, finger_prob))
    return 0.5 * face_loss + 0.5 * finger_loss, face_loss, finger_loss


def train_epoch(model, arrays: dict, *, seed: int, adversarial: bool,
                noise_stddev: float, noise_generator, optimizer, epoch: int) -> tuple[dict, dict, dict | None]:
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(arrays["face_train_y"]))
    face_losses, finger_losses, total_losses = [], [], []
    noise_sq, noise_n, noise_abs_max = 0.0, 0, 0.0
    grad_norm_sums = {key: 0.0 for key in ("face", "fingerprint", "shared")}
    update_norm_sq = 0.0
    adversarial_proof = None
    for start in range(0, len(order), BATCH_SIZE):
        rows = order[start:start + BATCH_SIZE]
        face = arrays["face_train_x"][rows]
        fy = arrays["face_train_y"][rows]
        finger = arrays["finger_train_x"][rows]
        py = arrays["finger_train_y"][rows]
        if adversarial:
            clean_face = face.copy()
            adv, _ = face_fgsm(model, face, finger, fy)
            if adversarial_proof is None:
                adversarial_proof = {"clean_face": clean_face, "adversarial_face": adv.copy(),
                                     "face_labels": fy.copy(), "fingerprint_inputs": finger.copy(),
                                     "fingerprint_labels": py.copy(), "source_rows": rows.copy()}
            face = np.concatenate((face, adv))
            fy = np.concatenate((fy, fy))
            finger = np.concatenate((finger, finger))
            py = np.concatenate((py, py))
        variables = model.trainable_variables
        before = [tf.identity(v) for v in variables]
        with tf.GradientTape() as tape:
            output = model({"face_input": face, "finger_input": finger}, training=True)
            objective, face_loss, finger_loss = _objective(fy, py, output, tf)
        gradients = tape.gradient(objective, variables)
        norms = _named_branch_norms(variables, gradients, tf)
        for key in grad_norm_sums:
            grad_norm_sums[key] += norms[key]
        noisy_gradients = []
        for gradient in gradients:
            if gradient is None:
                noisy_gradients.append(None)
            elif noise_stddev:
                noise = noise_generator.normal(tf.shape(gradient), stddev=noise_stddev, dtype=gradient.dtype)
                noise_sq += float(tf.reduce_sum(tf.square(tf.cast(noise, tf.float64))).numpy())
                noise_n += int(tf.size(noise).numpy())
                noise_abs_max = max(noise_abs_max, float(tf.reduce_max(tf.abs(noise)).numpy()))
                noisy_gradients.append(gradient + noise)
            else:
                noisy_gradients.append(gradient)
        optimizer.apply_gradients((g, v) for g, v in zip(noisy_gradients, variables) if g is not None)
        for old, variable in zip(before, variables):
            update_norm_sq += float(tf.reduce_sum(tf.square(tf.cast(variable - old, tf.float64))).numpy())
        total_losses.append(float(objective.numpy()))
        face_losses.append(float(face_loss.numpy()))
        finger_losses.append(float(finger_loss.numpy()))
    batches = len(total_losses)
    history = {"epoch": epoch, "total_loss": float(np.mean(total_losses)),
               "face_loss": float(np.mean(face_losses)), "fingerprint_loss": float(np.mean(finger_losses)),
               "face_gradient_norm_mean": float(grad_norm_sums["face"] / batches),
               "fingerprint_gradient_norm_mean": float(grad_norm_sums["fingerprint"] / batches),
               "shared_gradient_norm_mean": float(grad_norm_sums["shared"] / batches),
               "parameter_update_l2_epoch": float(np.sqrt(update_norm_sq))}
    noise_info = {"noise_samples": noise_n, "noise_realized_stddev": float(np.sqrt(noise_sq / noise_n)) if noise_n else 0.0,
                  "noise_abs_max": noise_abs_max}
    return history, noise_info, adversarial_proof


def _class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> dict:
    labels = np.arange(class_count)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
            "per_class": {str(k): {m: float(v[m]) for m in ("precision", "recall", "f1-score", "support")}
                          for k, v in report.items() if str(k).isdigit()},
            "macro_avg": {m: float(report["macro avg"][m]) for m in ("precision", "recall", "f1-score")},
            "weighted_avg": {m: float(report["weighted avg"][m]) for m in ("precision", "recall", "f1-score")}}


def evaluate(model, arrays: dict, state: str) -> tuple[dict, dict]:
    face_prob, finger_prob = model_outputs(model, arrays["face_test_x"], arrays["finger_test_x"])
    face_pred, finger_pred = face_prob.argmax(axis=1), finger_prob.argmax(axis=1)
    adv, gradient = face_fgsm(model, arrays["face_test_x"], arrays["finger_test_x"], arrays["face_test_y"])
    adv_face_prob, _ = model_outputs(model, adv, arrays["finger_test_x"])
    adv_pred = adv_face_prob.argmax(axis=1)
    clean_mask = face_pred == arrays["face_test_y"]
    success_mask = clean_mask & (adv_pred != arrays["face_test_y"])
    denominator = int(clean_mask.sum())
    if denominator == 0:
        raise RuntimeError(f"{state}: FGSM ASR is undefined because there are no clean-correct face samples")
    metrics = {
        "state": state,
        "face": _class_metrics(arrays["face_test_y"], face_pred, arrays["face_class_count"]),
        "fingerprint": _class_metrics(arrays["finger_test_y"], finger_pred, arrays["finger_class_count"]),
        "face_fgsm": {"epsilon": FGSM_EPSILON, "samples": int(len(face_pred)),
                      "clean_correct_denominator": denominator,
                      "attack_successes": int(success_mask.sum()),
                      "attack_success_rate": float(success_mask.sum() / denominator),
                      "accuracy_under_attack_all_samples": float(np.mean(adv_pred == arrays["face_test_y"])),
                      "perturbation_linf_max": float(np.max(np.abs(adv - arrays["face_test_x"]))),
                      "gradient_l2": float(np.linalg.norm(gradient.astype(np.float64)))},
    }
    evidence = {"face_labels": arrays["face_test_y"], "fingerprint_labels": arrays["finger_test_y"],
                "face_predictions": face_pred, "fingerprint_predictions": finger_pred,
                "face_probabilities": face_prob.astype(np.float32),
                "fingerprint_probabilities": finger_prob.astype(np.float32),
                "face_adversarial_predictions": adv_pred,
                "face_adversarial_features": adv.astype(np.float32),
                "face_clean_correct_mask": clean_mask,
                "face_attack_success_mask": success_mask,
                "face_evaluation_ids": arrays["face_eval_ids"],
                "fingerprint_evaluation_ids": arrays["finger_eval_ids"]}
    return metrics, evidence


def _artifact_manifest(run_dir: Path, parent_manifests: dict) -> dict:
    artifacts = [{"path": p.relative_to(run_dir).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "phase4_manifest.json"]
    return {"phase": 4, "run_id": run_dir.name, "status": "PHASE4-ARTIFACTS-PERSISTED",
            "source_document_sha256": SOURCE_DOC_SHA256, "parent_manifests": parent_manifests,
            "artifacts": artifacts}


def run(root: Path, output_root: Path) -> Path:
    import tensorflow as tf
    root, output_root = root.resolve(), output_root.resolve()
    p1 = root / "runtime_recording" / PHASE1_RUN
    p2 = root / "runtime_recording" / PHASE2_RUN
    p3 = root / "runtime_recording" / PHASE3_RUN
    docx = root / "docs" / "reference" / "CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION.docx"
    if sha256(docx).upper() != SOURCE_DOC_SHA256:
        raise ValueError("The Chapter Four source document is missing or its SHA-256 has changed")
    p1m = assert_manifest_files(p1, "phase1_manifest.json")
    p2m = assert_manifest_files(p2, "phase2_manifest.json")
    p3m = assert_manifest_files(p3, "phase3_manifest.json")
    p3cfg = json.loads((p3 / "phase3_config.json").read_text(encoding="utf-8"))
    if p1m.get("verification", {}).get("status") != "PASS" or p1m.get("replay_id") != PHASE1_RUN:
        raise ValueError("Phase 1 source is not the accepted passing replay")
    if p2m.get("run_id") != PHASE2_RUN or p3m.get("run_id") != PHASE3_RUN:
        raise ValueError("Phase 2/3 selected run identity mismatch")
    if p3cfg.get("source_phase2_model_sha256") != sha256(p2 / "final_global_model.keras"):
        raise ValueError("Phase 3 evaluation does not reference the selected Phase 2 checkpoint")
    if p3m.get("source_document_sha256") != SOURCE_DOC_SHA256:
        raise ValueError("Phase 3 source document is not authoritative")
    if json.loads((p3 / "phase3_verification_report.json").read_text(encoding="utf-8")).get("status") != "PASS":
        raise ValueError("Selected Phase 3 evaluation has not passed independent verification")
    p2entry = next(a for a in p2m["artifacts"] if a["path"] == "final_global_model.keras")
    baseline_path = p2 / "final_global_model.keras"
    if sha256(baseline_path) != p2entry["sha256"]:
        raise ValueError("Phase 2 baseline checkpoint does not match its accepted manifest")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"phase4_authoritative_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"phase4_authoritative_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir()
    verifier = Path(__file__).resolve().with_name("verify_phase4_authoritative_run.py")
    sources = [Path(__file__).resolve(), verifier,
               Path(__file__).resolve().with_name("phase3_authoritative_attacks.py"),
               Path(__file__).resolve().with_name("phase2_authoritative_training.py")]
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
    shutil.copy2(baseline_path, run_dir / "baseline_model.keras")

    with np.load(p1 / "face_data.npz", allow_pickle=False) as f, np.load(p1 / "fingerprint_data.npz", allow_pickle=False) as g:
        arrays = {
            "face_train_x": f["X_train"][:640].astype(np.float32), "face_train_y": f["y_train"][:640].astype(np.int64),
            "finger_train_x": g["X_train"][:640].astype(np.float32), "finger_train_y": g["y_train"][:640].astype(np.int64),
            "face_test_x": f["X_test"][:EVAL_N].astype(np.float32), "face_test_y": f["y_test"][:EVAL_N].astype(np.int64),
            "finger_test_x": g["X_test"][:EVAL_N].astype(np.float32), "finger_test_y": g["y_test"][:EVAL_N].astype(np.int64),
            "face_eval_ids": f["test_indices"][:EVAL_N].astype(np.int64),
            "finger_eval_ids": g["test_indices"][:EVAL_N].astype(np.int64),
            "face_class_count": int(f["n_classes"][0]), "finger_class_count": int(g["n_classes"][0]),
        }
    np.savez_compressed(run_dir / "evaluation_records.npz", **{k: v for k, v in arrays.items() if isinstance(v, np.ndarray)})
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    baseline = tf.keras.models.load_model(baseline_path, compile=False)
    states, histories, training_evidence, evaluation_evidence, training_proofs = {}, {}, {}, {}, {}
    start_time = time.perf_counter()
    definitions = {
        "dp_inspired": {"adversarial": False, "noise": DP_NOISE_STDDEV, "seed": 4201},
        "face_adversarial": {"adversarial": True, "noise": 0.0, "seed": 4202},
        "combined": {"adversarial": True, "noise": COMBINED_NOISE_STDDEV, "seed": 4203},
    }
    for name, definition in definitions.items():
        print(f"Phase 4 training: {name}", flush=True)
        model = tf.keras.models.load_model(baseline_path, compile=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        noise_generator = tf.random.Generator.from_seed(definition["seed"])
        epoch_history, noise_history = [], []
        if name == "combined":
            for epoch in range(6):
                row, _, _ = train_epoch(model, arrays, seed=5000 + epoch, adversarial=False,
                                     noise_stddev=0.0, noise_generator=noise_generator,
                                     optimizer=optimizer, epoch=epoch + 1)
                row["stage"] = "six_epoch_clean_pretraining"
                epoch_history.append(row)
            model.save(run_dir / "combined_pretrained_model.keras")
        for epoch in range(EPOCHS):
            row, noise, proof = train_epoch(model, arrays, seed=definition["seed"] * 100 + epoch,
                                     adversarial=definition["adversarial"], noise_stddev=definition["noise"],
                                     noise_generator=noise_generator, optimizer=optimizer, epoch=epoch + 1)
            row["stage"] = "defense_finetuning"
            epoch_history.append(row)
            noise_history.append(noise)
            if proof is not None and name not in training_proofs:
                training_proofs[name] = proof
            if (epoch + 1) % 3 == 0:
                print(f"  {name} epoch {epoch+1}/{EPOCHS}: loss={row['total_loss']:.4f}", flush=True)
        path = run_dir / f"{name}_model.keras"
        model.save(path)
        states[name] = model
        histories[name] = epoch_history
        training_evidence[name] = {"noise_stddev_configured": definition["noise"],
                                   "noise_epochs": noise_history,
                                   "face_training_epsilon": FGSM_EPSILON if definition["adversarial"] else None,
                                   "face_gradient_norm_mean": float(np.mean([r["face_gradient_norm_mean"] for r in epoch_history if r["stage"] == "defense_finetuning"])),
                                   "fingerprint_gradient_norm_mean": float(np.mean([r["fingerprint_gradient_norm_mean"] for r in epoch_history if r["stage"] == "defense_finetuning"])),
                                   "parameter_update_l2_total": float(sum(r["parameter_update_l2_epoch"] for r in epoch_history))}

    np.savez_compressed(run_dir / "adversarial_training_examples.npz",
                        **{f"{state}_{field}": value for state, proof in training_proofs.items()
                           for field, value in proof.items()})

    metrics, baseline_evidence = evaluate(baseline, arrays, "federated_baseline")
    evaluation_evidence["baseline"] = baseline_evidence
    all_metrics = {"federated_baseline": metrics}
    for name, model in states.items():
        state_name = {"dp_inspired": "dp_inspired_gaussian_noise", "face_adversarial": "face_adversarial_training", "combined": "combined_dofmbs_defense"}[name]
        metrics, evidence = evaluate(model, arrays, state_name)
        all_metrics[state_name] = metrics
        evaluation_evidence[name] = evidence
    np.savez_compressed(run_dir / "evaluation_predictions.npz",
                        **{f"{state}_{key}": value for state, payload in evaluation_evidence.items()
                           for key, value in payload.items()})
    (run_dir / "training_histories.json").write_text(json.dumps(histories, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "gradient_evidence.json").write_text(json.dumps(training_evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_json(run_dir / "phase4_metrics.json", {"phase": 4, "run_id": run_dir.name,
               "evaluation_samples": EVAL_N, "results": all_metrics})
    config = {
        "phase": 4, "run_id": run_dir.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_document_sha256": SOURCE_DOC_SHA256,
        "source_phase1_run": PHASE1_RUN, "source_phase1_manifest_sha256": sha256(p1 / "phase1_manifest.json"),
        "source_phase2_run": PHASE2_RUN, "source_phase2_manifest_sha256": sha256(p2 / "phase2_manifest.json"),
        "source_phase2_checkpoint": "final_global_model.keras", "source_phase2_checkpoint_sha256": sha256(baseline_path),
        "source_phase2_config_sha256": sha256(p2 / "phase2_config.json"),
        "source_phase3_run": PHASE3_RUN, "source_phase3_manifest_sha256": sha256(p3 / "phase3_manifest.json"),
        "source_phase3_evaluation_reference": "first 100 Phase 1 X_test rows per modality; same positional pairing and labels as Phase 3 FGSM",
        "source_phase3_fgsm_epsilon": 0.05,
        "source_sha256": {p.name: sha256(snapshot / p.name) for p in sources},
        "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip(),
        "dependencies": dependencies(), "platform": platform.platform(),
        "random_seeds": {"global_determinism": SEED, **{k: v["seed"] for k, v in definitions.items()},
                         "combined_clean_pretraining_epoch_formula": "5000 + zero_based_epoch"},
        "training": {"optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
                     "face_training_rows": 640, "fingerprint_training_rows": 640,
                     "epochs_each_defense": EPOCHS, "combined_clean_pretraining_epochs": 6,
                     "combined_defense_epochs_after_pretraining": EPOCHS,
                     "loss": "sparse categorical cross entropy; independent face/fingerprint identity heads",
                     "loss_weights": {"face": 0.5, "fingerprint": 0.5},
                     "adversarial_mixture": "concatenate clean and FGSM face batches; duplicate unchanged fingerprint inputs/labels; retain original face labels"},
        "defenses": {
            "dp_inspired": {"method": "Gaussian noise added to each non-null model gradient before Adam updates",
                            "historical_epsilon_parameter": DP_EPSILON, "noise_scale_formula": "0.001 / epsilon",
                            "noise_distribution": "independent Gaussian N(0, sigma^2)", "gradient_noise_stddev": DP_NOISE_STDDEV,
                            "gradient_clipping": False, "privacy_accountant": False, "formal_dp_claim": False},
            "face_adversarial": {"attack": "Phase-3 untargeted FGSM on standardized face-feature inputs",
                                 "epsilon": FGSM_EPSILON, "training_data": "50% clean and 50% adversarial face rows per paired batch",
                                 "fingerprint_optimization_claim": False},
            "combined": {"method": "six clean pretraining epochs, then face FGSM mixed batches and Gaussian gradient noise",
                         "gradient_noise_stddev": COMBINED_NOISE_STDDEV, "face_fgsm_epsilon": FGSM_EPSILON,
                         "loss_weights": {"face": 0.5, "fingerprint": 0.5},
                         "both_modality_losses_active": True},
        },
        "evaluation": {"rows": EVAL_N, "same_records_for_all_states": True,
                       "face_test_positions": list(range(EVAL_N)), "fingerprint_test_positions": list(range(EVAL_N)),
                       "face_and_fingerprint_identities_correspond": False,
                       "fgsm_asr_denominator": "each model's clean-correct face samples, same Phase 3 convention",
                       "fgsm_attack": "one untargeted sign-gradient step, epsilon=0.05, no clipping; fingerprint rows unchanged"},
        "limitations": ["Gaussian gradient perturbation has no clipping/accountant and is not formal differential privacy.",
                        "Adversarial examples are face-only; fingerprint adversarial robustness is not tested by this defense.",
                        "Face and fingerprint identity populations are independent; row pairing is computational only.",
                        "Combined objective optimizes separate modality classifiers and is not person-level cross-modal identity fusion."],
    }
    write_json(run_dir / "phase4_config.json", config)
    elapsed = time.perf_counter() - start_time
    write_json(run_dir / "phase4_runtime.json", {"training_and_evaluation_seconds": elapsed})
    parent_hashes = {"phase1_manifest_sha256": config["source_phase1_manifest_sha256"],
                     "phase2_manifest_sha256": config["source_phase2_manifest_sha256"],
                     "phase3_manifest_sha256": config["source_phase3_manifest_sha256"]}
    write_json(run_dir / "phase4_verification_report.json", {"status": "PENDING", "checks": [], "failed_checks": []})
    write_json(run_dir / "phase4_manifest.json", _artifact_manifest(run_dir, parent_hashes))
    # Avoid retaining four TensorFlow graphs and model copies while the
    # independent verifier reloads and audits the saved checkpoints.
    del states, baseline, evaluation_evidence, histories, training_evidence, training_proofs
    tf.keras.backend.clear_session()
    gc.collect()
    from verify_phase4_authoritative_run import verify
    result = verify(run_dir)
    write_json(run_dir / "phase4_verification_report.json", result)
    write_json(run_dir / "phase4_manifest.json", _artifact_manifest(run_dir, parent_hashes))
    result = verify(run_dir)
    if result["status"] != "PASS":
        write_json(run_dir / "phase4_verification_report.json", result)
        raise RuntimeError(f"Phase 4 independent verification failed: {result['failed_checks']}")
    write_json(run_dir / "phase4_verification_report.json", result)
    write_json(run_dir / "phase4_manifest.json", _artifact_manifest(run_dir, parent_hashes))
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "runtime_recording")
    args = parser.parse_args()
    print(run(args.root, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

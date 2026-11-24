"""Run auditable Phase 3 attacks against the accepted Phase 2 global model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SOURCE_DOC_SHA256 = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"
EPSILONS = (0.01, 0.05, 0.10, 0.20, 0.30)
PHASE2_RUN = "phase2_authoritative_20261123_203617_remediation"
PHASE1_RUN = "phase1_final_candidate_20261123_182623"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_manifest_files(directory: Path, manifest_name: str) -> tuple[dict, list[str]]:
    manifest_path = directory / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = []
    for item in manifest.get("artifacts", []):
        target = directory / item["path"]
        if not target.is_file() or target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            bad.append(item["path"])
    if bad:
        raise ValueError(f"Invalid {manifest_name} artifacts: {bad}")
    return manifest, bad


def dependency_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "scikit-learn", "keras"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    try:
        result["tensorflow"] = importlib.metadata.version("tensorflow")
    except importlib.metadata.PackageNotFoundError:
        try:
            result["tensorflow"] = importlib.metadata.version("tensorflow-cpu")
        except importlib.metadata.PackageNotFoundError:
            result["tensorflow"] = "not-installed"
    try:
        result["tensorflow_cpu_distribution"] = importlib.metadata.version("tensorflow-cpu")
    except importlib.metadata.PackageNotFoundError:
        result["tensorflow_cpu_distribution"] = "not-installed"
    return result


def outputs(model, face: np.ndarray, finger: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = model({"face_input": face, "finger_input": finger}, training=False)
    if isinstance(prediction, dict):
        return np.asarray(prediction["face_output"]), np.asarray(prediction["finger_output"])
    return np.asarray(prediction[0]), np.asarray(prediction[1])


def fgsm(model, face: np.ndarray, finger: np.ndarray, labels: np.ndarray, epsilon: float) -> np.ndarray:
    """Untargeted sign-gradient step in standardized feature space; no clipping."""
    import tensorflow as tf

    x = tf.convert_to_tensor(face, dtype=tf.float32)
    paired_finger = tf.convert_to_tensor(finger, dtype=tf.float32)
    y = tf.convert_to_tensor(labels, dtype=tf.int64)
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model({"face_input": x, "finger_input": paired_finger}, training=False)
        probability = prediction["face_output"] if isinstance(prediction, dict) else prediction[0]
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, probability))
    gradient = tape.gradient(loss, x)
    if gradient is None:
        raise RuntimeError("Face-input loss gradient is missing")
    return (x + tf.cast(epsilon, x.dtype) * tf.sign(gradient)).numpy()


def pgd(model, face: np.ndarray, finger: np.ndarray, labels: np.ndarray,
        epsilon: float, alpha: float = 0.01, iterations: int = 10) -> np.ndarray:
    """Fixed-step untargeted PGD, no random start, L-infinity projection only."""
    import tensorflow as tf

    original = tf.convert_to_tensor(face, dtype=tf.float32)
    paired_finger = tf.convert_to_tensor(finger, dtype=tf.float32)
    y = tf.convert_to_tensor(labels, dtype=tf.int64)
    adversarial = tf.identity(original)
    # Ten alpha-sized steps cannot mathematically travel farther than 0.10;
    # include that protocol-implied ceiling in the numerical projection.
    effective_bound = min(float(epsilon), float(alpha) * int(iterations))
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial)
            prediction = model({"face_input": adversarial, "finger_input": paired_finger}, training=False)
            probability = prediction["face_output"] if isinstance(prediction, dict) else prediction[0]
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, probability))
        gradient = tape.gradient(loss, adversarial)
        if gradient is None:
            raise RuntimeError("Face-input loss gradient is missing")
        adversarial = adversarial + tf.cast(alpha, adversarial.dtype) * tf.sign(gradient)
        candidate = (original + tf.clip_by_value(adversarial - original, -effective_bound, effective_bound)).numpy()
        origin = original.numpy()
        # Correct one-ULP float32 round-off so persisted coordinate differences
        # do not exceed the mathematical L-infinity projection bound.
        candidate64, origin64 = candidate.astype(np.float64), origin.astype(np.float64)
        candidate = np.clip(candidate64, origin64 - effective_bound, origin64 + effective_bound).astype(np.float32)
        over = np.abs(candidate.astype(np.float64) - origin64) > effective_bound
        while np.any(over):
            candidate[over] = np.nextafter(candidate[over], origin[over])
            over = np.abs(candidate.astype(np.float64) - origin64) > effective_bound
        adversarial = tf.convert_to_tensor(candidate, dtype=original.dtype)
    return adversarial.numpy()


def deterministic_membership_indices(train_count: int, test_count: int, count: int = 100,
                                     seed: int = 7303) -> tuple[np.ndarray, np.ndarray]:
    if train_count < count or test_count < count:
        raise ValueError(f"Need {count} disjoint train/test records, got {train_count}/{test_count}")
    rng = np.random.default_rng(seed)
    return rng.choice(train_count, count, replace=False), rng.choice(test_count, count, replace=False)


def flip_selected_labels(labels: np.ndarray, selected_indices: np.ndarray, flip_count: int,
                         class_count: int, seed: int = 3302) -> tuple[np.ndarray, np.ndarray]:
    if flip_count < 0 or flip_count > len(selected_indices):
        raise ValueError("flip_count must be between zero and selected sample count")
    rng = np.random.default_rng(seed)
    flipped_positions = rng.choice(len(selected_indices), flip_count, replace=False)
    mask = np.zeros(len(labels), dtype=bool)
    actual = selected_indices[flipped_positions]
    mask[actual] = True
    poisoned = np.array(labels, copy=True)
    poisoned[actual] = (poisoned[actual] + 1) % class_count
    return poisoned, mask


def _metrics_for_evasion(model, clean_face, paired_finger, labels, selected_ids, output_dir):
    clean_prob, _ = outputs(model, clean_face, paired_finger)
    clean_pred = np.argmax(clean_prob, axis=1)
    correct_mask = clean_pred == labels
    selected = np.flatnonzero(correct_mask)
    if not len(selected):
        raise RuntimeError("No clean-correct face samples; conditional ASR is undefined")
    attack_face = clean_face[selected]
    attack_finger = paired_finger[selected]
    attack_labels = labels[selected]
    evidence = {
        "evaluation_ids": selected_ids.astype(np.int64), "face_clean": clean_face.astype(np.float32),
        "fingerprint_unchanged": paired_finger.astype(np.float32), "labels": labels.astype(np.int64),
        "clean_predictions": clean_pred.astype(np.int64), "clean_correct_mask": correct_mask,
        "clean_correct_indices": selected.astype(np.int64),
    }
    metrics = []
    for family in ("FGSM", "PGD"):
        for epsilon in EPSILONS:
            adversarial = (fgsm(model, clean_face, paired_finger, labels, epsilon)
                           if family == "FGSM" else pgd(model, clean_face, paired_finger,
                                                        labels, epsilon, alpha=0.01, iterations=10))
            attacked_prob, _ = outputs(model, adversarial, paired_finger)
            attacked_pred = np.argmax(attacked_prob, axis=1)
            conditional_success = attacked_pred[selected] != attack_labels
            delta = adversarial - clean_face
            norms = np.max(np.abs(delta), axis=1)
            bound_ok = bool(np.all(norms <= epsilon + 1e-6))
            if family == "PGD" and epsilon > 0.10:
                if np.max(norms, initial=0.0) > 0.10 + 1e-5:
                    raise RuntimeError("Fixed-step PGD exceeded its 0.10 traversed-distance ceiling")
            prefix = f"{family.lower()}_eps_{epsilon:.2f}"
            evidence[f"{prefix}_adversarial_face"] = adversarial.astype(np.float32)
            evidence[f"{prefix}_predictions"] = attacked_pred.astype(np.int64)
            evidence[f"{prefix}_success_mask"] = conditional_success
            metrics.append({
                "attack": family, "modality": "face", "epsilon_projection_bound": epsilon,
                "step_size": None if family == "FGSM" else 0.01,
                "iterations": 1 if family == "FGSM" else 10,
                "initialization": "clean input; no random start",
                "clean_correct_denominator": int(len(selected)),
                "attack_successes": int(np.count_nonzero(conditional_success)),
                "attack_success_rate": float(np.mean(conditional_success)),
                "accuracy_under_attack_on_conditional_subset": float(np.mean(attacked_pred[selected] == attack_labels)),
                "accuracy_under_attack_all_100": float(np.mean(attacked_pred == labels)),
                "perturbation_linf_mean": float(np.mean(norms)),
                "perturbation_linf_median": float(np.median(norms)),
                "perturbation_linf_max": float(np.max(norms, initial=0.0)),
                "perturbation_linf_min": float(np.min(norms)) if len(norms) else 0.0,
                "max_abs_perturbation": float(np.max(np.abs(delta), initial=0.0)),
                "projection_bound_satisfied": bound_ok,
                "actual_traversal_ceiling": 0.10 if family == "PGD" else epsilon,
            })
    np.savez_compressed(output_dir / "evasion_evidence.npz", **evidence)
    return metrics, int(len(selected))


def _manifest(directory: Path, run_id: str, parent_manifests: dict[str, str]) -> dict:
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "phase3_manifest.json":
            artifacts.append({"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size,
                              "sha256": sha256(path)})
    return {"phase": 3, "run_id": run_id, "status": "PHASE3-ARTIFACTS-PERSISTED",
            "source_document_sha256": SOURCE_DOC_SHA256, "parent_manifests": parent_manifests,
            "artifacts": artifacts}


def run(phase1_dir: Path, phase2_dir: Path, output_root: Path) -> Path:
    import tensorflow as tf
    from phase2_authoritative_training import compile_phase2_model

    phase1_dir, phase2_dir, output_root = phase1_dir.resolve(), phase2_dir.resolve(), output_root.resolve()
    p1_manifest, _ = verify_manifest_files(phase1_dir, "phase1_manifest.json")
    p2_manifest, _ = verify_manifest_files(phase2_dir, "phase2_manifest.json")
    p2_config = json.loads((phase2_dir / "phase2_config.json").read_text(encoding="utf-8"))
    if p1_manifest.get("replay_id") != PHASE1_RUN or p2_manifest.get("run_id") != PHASE2_RUN:
        raise ValueError("Phase 1/2 directories do not identify the accepted authoritative runs")
    if p2_config.get("phase1_manifest_sha256") != sha256(phase1_dir / "phase1_manifest.json"):
        raise ValueError("Phase 2 does not reference the selected accepted Phase 1 manifest")
    if p2_config.get("chapter_four_sha256") != SOURCE_DOC_SHA256:
        raise ValueError("Phase 2 source document hash is not the authoritative Chapter Four hash")
    model_entry = next((x for x in p2_manifest["artifacts"] if x["path"] == "final_global_model.keras"), None)
    model_path = phase2_dir / "final_global_model.keras"
    if not model_entry or sha256(model_path) != model_entry["sha256"]:
        raise ValueError("Authoritative Phase 2 model is missing or does not match its manifest")
    import phase2_authoritative_training as p2_source
    if p2_config.get("source_sha256", {}).get("phase2_authoritative_training.py") != sha256(
            phase2_dir / p2_config["source_snapshot_directory"] / "phase2_authoritative_training.py"):
        raise ValueError("Phase 2 source snapshot does not match its recorded hash")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_id = f"phase3_authoritative_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / run_id
    suffix = 1
    while output_dir.exists():
        output_dir = output_root / f"{run_id}_{suffix}"
        suffix += 1
    output_dir.mkdir()
    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir()
    verifier_source = Path(__file__).resolve().with_name("verify_phase3_authoritative_run.py")
    for source in (Path(__file__).resolve(), Path(__file__).resolve().with_name("do_fmbs_core.py"),
                   Path(p2_source.__file__).resolve(), verifier_source):
        shutil.copy2(source, snapshot_dir / source.name)

    p1_config = json.loads((phase1_dir / "phase1_config.json").read_text(encoding="utf-8"))
    with np.load(phase1_dir / "face_data.npz", allow_pickle=False) as d:
        face_train, face_train_y = d["X_train"], d["y_train"]
        face_test, face_test_y, face_test_ids = d["X_test"], d["y_test"], d["test_indices"]
        face_train_ids = d["train_indices"]
    with np.load(phase1_dir / "fingerprint_data.npz", allow_pickle=False) as d:
        finger_train, finger_train_y = d["X_train"], d["y_train"]
        finger_test, finger_test_y, finger_test_ids = d["X_test"], d["y_test"], d["test_indices"]

    # This is the persisted Phase 2 positional pairing, not an assertion that modalities share identities.
    paired_face_test = np.asarray(face_test[:100], dtype=np.float32)
    paired_finger_test = np.asarray(finger_test[:100], dtype=np.float32)
    paired_face_y = np.asarray(face_test_y[:100], dtype=np.int64)
    paired_finger_y = np.asarray(finger_test_y[:100], dtype=np.int64)
    eval_ids = np.asarray(face_test_ids[:100], dtype=np.int64)
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    baseline = tf.keras.models.load_model(model_path, compile=False)
    model_shapes = baseline.input_shape
    if isinstance(model_shapes, dict):
        face_shape, finger_shape = model_shapes["face_input"], model_shapes["finger_input"]
    else:
        face_shape, finger_shape = model_shapes
    if face_shape[-1] != 1850 or finger_shape[-1] != 9216:
        raise ValueError("Persisted model dimensions do not match the accepted Phase 1 modalities")

    evasion_metrics, clean_correct = _metrics_for_evasion(
        baseline, paired_face_test, paired_finger_test, paired_face_y, eval_ids, output_dir)

    member_i, nonmember_i = deterministic_membership_indices(640, 160, 100, seed=7303)
    member_prob, _ = outputs(baseline, np.asarray(face_train[:640][member_i], np.float32),
                             np.asarray(finger_train[:640][member_i], np.float32))
    nonmember_prob, _ = outputs(baseline, np.asarray(face_test[:160][nonmember_i], np.float32),
                                np.asarray(finger_test[:160][nonmember_i], np.float32))
    member_conf, nonmember_conf = np.max(member_prob, axis=1), np.max(nonmember_prob, axis=1)
    confidence = np.concatenate([member_conf, nonmember_conf])
    truth_member = np.concatenate([np.ones(100, dtype=bool), np.zeros(100, dtype=bool)])
    threshold = 0.70
    decisions = confidence >= threshold
    correct_decisions = decisions == truth_member
    tp = int(np.count_nonzero(decisions[:100]))
    fp = int(np.count_nonzero(decisions[100:]))
    fn, tn = 100 - tp, 100 - fp
    mia_metrics = {
        "attack": "fixed confidence-threshold membership inference", "modality": "face",
        "threshold": threshold, "threshold_comparison": "confidence >= 0.70 means member",
        "members": 100, "nonmembers": 100, "true_member_detections": tp,
        "false_member_detections": fp, "false_negatives": fn, "true_negatives": tn,
        "attack_accuracy": float(np.mean(correct_decisions)),
        "member_true_positive_rate": tp / 100, "nonmember_false_positive_rate": fp / 100,
        "false_negative_rate": fn / 100,
        "privacy_quantity_PA": float(2 * max(0.0, float(np.mean(correct_decisions)) - 0.50)),
        "privacy_quantity_definition": "2 * max(0, membership attack accuracy - 0.50), empirical attack pressure only; not a formal guarantee",
        "reference_Chapter_Four": {"members_called_member": "2/100", "nonmembers_called_member": "1/100",
                                    "accuracy": 0.505, "privacy_quantity": 0.01},
    }
    np.savez_compressed(output_dir / "membership_evidence.npz",
                        member_record_ids=np.asarray(face_train_ids[:640][member_i], np.int64),
                        nonmember_record_ids=np.asarray(face_test_ids[:160][nonmember_i], np.int64),
                        member_phase1_rows=member_i, nonmember_phase1_rows=nonmember_i,
                        true_membership=truth_member, confidence=confidence,
                        attack_decisions=decisions, correct_decisions=correct_decisions,
                        face_true_labels=np.concatenate([face_train_y[:640][member_i], face_test_y[:160][nonmember_i]]).astype(np.int64),
                        fingerprint_paired_rows=np.concatenate([member_i, nonmember_i]).astype(np.int64))

    # Exact Phase 2 training population: first 640 rows of each modality; choose 200 without replacement.
    poison_rng = np.random.default_rng(3301)
    selected = poison_rng.choice(640, 200, replace=False)
    face_y200 = np.asarray(face_train_y[:640][selected], dtype=np.int64)
    poison_face_y, flip_mask_all = flip_selected_labels(
        np.asarray(face_train_y[:640], dtype=np.int64), selected, 20, 62, seed=3302)
    flip_mask = flip_mask_all[selected]
    if int(np.count_nonzero(flip_mask)) != 20:
        raise RuntimeError("Poison protocol did not flip exactly 20 of 200 selected face labels")
    poison_face_y200 = poison_face_y[selected]
    finger_y200 = np.asarray(finger_train_y[:640][selected], dtype=np.int64)
    clean_model_path = output_dir / "clean_baseline_clone.keras"
    shutil.copy2(model_path, clean_model_path)
    clean_clone = tf.keras.models.load_model(clean_model_path, compile=False)
    compile_phase2_model(clean_clone, learning_rate=0.001)
    clean_face_probability, _ = outputs(clean_clone, paired_face_test, paired_finger_test)
    clean_eval_pred = np.argmax(clean_face_probability, axis=1)
    clean_accuracy = float(np.mean(clean_eval_pred == paired_face_y))
    poison_model = tf.keras.models.load_model(clean_model_path, compile=False)
    compile_phase2_model(poison_model, learning_rate=0.001)
    tf.keras.utils.set_random_seed(3303)
    history_obj = poison_model.fit(
        {"face_input": np.asarray(face_train[:640][selected], np.float32),
         "finger_input": np.asarray(finger_train[:640][selected], np.float32)},
        {"face_output": poison_face_y200, "finger_output": finger_y200},
        epochs=3, batch_size=64, shuffle=True, verbose=0,
    )
    poison_probability, _ = outputs(poison_model, paired_face_test, paired_finger_test)
    poison_eval_pred = np.argmax(poison_probability, axis=1)
    poison_accuracy = float(np.mean(poison_eval_pred == paired_face_y))
    accuracy_change = poison_accuracy - clean_accuracy
    poisoning_metrics = {
        "attack": "face-label-flipping data-poisoning proxy", "modality": "face labels only",
        "clean_sample_count": 200, "poisoning_sample_count": 200, "poison_rate": 0.10,
        "flipped_labels": 20, "epochs": 3, "batch_size": 64, "learning_rate": 0.001,
        "selection_seed": 3301, "flip_seed": 3302, "training_seed": 3303,
        "selected_phase1_train_positions": selected.astype(int).tolist(),
        "clean_accuracy": clean_accuracy, "poisoned_accuracy": poison_accuracy,
        "accuracy_change_percentage_points": 100 * accuracy_change,
        "clamped_degradation_poisoning_asr": float(max(0.0, clean_accuracy - poison_accuracy)),
        "asr_definition": "max(0, clean accuracy - poisoned accuracy), the Chapter Four code's clamped degradation-based quantity",
        "interpretation": "degraded" if accuracy_change < 0 else ("improved" if accuracy_change > 0 else "unchanged"),
        "reference_Chapter_Four": {"clean_accuracy": 0.26, "poisoned_accuracy": 0.31,
                                    "change_percentage_points": 5.0, "clamped_asr": 0.0},
        "scope_limitation": "This is a centralized label-flipping fine-tuning proxy, not malicious federated client-update aggregation.",
    }
    poison_model.save(output_dir / "poisoned_face_label_clone.keras")
    np.savez_compressed(output_dir / "poisoning_evidence.npz",
                        selected_phase1_train_positions=selected.astype(np.int64),
                        selected_face_record_ids=np.asarray(face_train_ids[:640][selected], np.int64),
                        original_labels=face_y200, poisoned_labels=poison_face_y200,
                        flip_mask=flip_mask, face_test_record_ids=eval_ids,
                        clean_test_labels=paired_face_y, clean_predictions=clean_eval_pred.astype(np.int64),
                        poisoned_predictions=poison_eval_pred.astype(np.int64),
                        clean_correct_mask=(clean_eval_pred == paired_face_y),
                        poisoned_correct_mask=(poison_eval_pred == paired_face_y),
                        paired_fingerprint_train_labels=finger_y200)
    json_write(output_dir / "poisoning_training_history.json",
               {key: [float(v) for v in values] for key, values in history_obj.history.items()})

    config = {
        "phase": 3, "run_id": output_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_document_sha256": SOURCE_DOC_SHA256,
        "source_phase1_run": phase1_dir.name, "source_phase1_manifest_sha256": sha256(phase1_dir / "phase1_manifest.json"),
        "source_phase2_run": phase2_dir.name, "source_phase2_manifest_sha256": sha256(phase2_dir / "phase2_manifest.json"),
        "source_phase2_model": "final_global_model.keras", "source_phase2_model_sha256": sha256(model_path),
        "source_phase2_config_sha256": sha256(phase2_dir / "phase2_config.json"),
        "source_sha256": {p.name: sha256(snapshot_dir / p.name) for p in snapshot_dir.iterdir() if p.is_file()},
        "dependencies": dependency_versions(), "platform": platform.platform(),
        "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                                           check=True, capture_output=True, text=True).stdout.strip(),
        "random_seeds": {"determinism": 42, "membership_selection": 7303,
                         "poison_selection": 3301, "poison_flip": 3302, "poison_training": 3303},
        "data_references": {
            "face_data_sha256": sha256(phase1_dir / "face_data.npz"),
            "fingerprint_data_sha256": sha256(phase1_dir / "fingerprint_data.npz"),
            "face_member_population": "first 640 Phase 1 face X_train rows used by Phase 2; selected 100 distinct rows",
            "face_nonmember_population": "Phase 1 face X_test; selected 100 distinct rows",
            "evasion_population": "first 100 Phase 1 X_test rows for both modalities, matching Phase 2 test pairing",
            "poison_population": "first 640 Phase 1 train rows per modality; 200 positions selected without replacement",
            "cross_modal_identity_correspondence": False,
        },
        "attacks": {
            "fgsm": {"epsilons": list(EPSILONS), "target": "face only", "loss": "untargeted sparse categorical cross entropy", "input_space": "standardized flattened face feature coordinates", "clipping": "none"},
            "pgd": {"epsilons": list(EPSILONS), "iterations": 10, "step_size": 0.01,
                    "initialization": "clean input; no random start", "projection": "L-infinity around original face input; no feature clipping", "max_traversal_linf": 0.10, "target": "face only"},
            "membership": {"threshold": 0.70, "members": 100, "nonmembers": 100,
                           "decision": "max face class confidence >= 0.70 implies member", "threshold_fitted": False},
            "label_flipping": {"selected_samples": 200, "poison_rate": 0.10, "flipped_labels": 20,
                               "target": "face labels only", "epochs": 3, "optimizer": "Adam", "learning_rate": 0.001},
        },
        "limitations": ["Standardized-coordinate feature attack; no perceptual-image interpretation.",
                        "Fingerprint is unchanged in evasion tests; no fingerprint robustness conclusion.",
                        "Pairing is positional and does not represent real cross-modal identity matches.",
                        "Empirical membership PA is not a formal privacy or differential-privacy guarantee.",
                        "Label-flipping experiment is a centralized proxy, not malicious federated aggregation."],
    }
    json_write(output_dir / "phase3_config.json", config)
    json_write(output_dir / "phase3_metrics.json", {
        "phase": 3, "run_id": output_dir.name,
        "source_phase2_run": phase2_dir.name, "source_phase2_model_sha256": sha256(model_path),
        "evaluation_face_rows": 100, "evaluation_fingerprint_rows_unchanged": 100,
        "clean_correct_face_denominator": clean_correct, "clean_face_accuracy": float(np.mean(np.argmax(outputs(baseline, paired_face_test, paired_finger_test)[0], axis=1) == paired_face_y)),
        "evasion": evasion_metrics, "membership_inference": mia_metrics,
        "label_flipping": poisoning_metrics,
    })
    json_write(output_dir / "phase3_verification_report.json", {"status": "PENDING", "checks": []})
    manifest = _manifest(output_dir, output_dir.name, {
        "phase1_manifest_sha256": sha256(phase1_dir / "phase1_manifest.json"),
        "phase2_manifest_sha256": sha256(phase2_dir / "phase2_manifest.json"),
    })
    manifest_path = output_dir / "phase3_manifest.json"
    json_write(manifest_path, manifest)
    # The independent verifier writes a deterministic report; refresh the
    # artifact manifest afterward so that report bytes are also hash-bound.
    from verify_phase3_authoritative_run import verify as independent_verify
    first_check = independent_verify(output_dir)
    json_write(output_dir / "phase3_verification_report.json", first_check)
    json_write(manifest_path, _manifest(output_dir, output_dir.name, manifest["parent_manifests"]))
    final_check = independent_verify(output_dir)
    if final_check["status"] != "PASS":
        raise RuntimeError(f"Independent Phase 3 verification failed: {final_check['failed_checks']}")
    json_write(output_dir / "phase3_verification_report.json", final_check)
    # Report serialization is stable and included in the final manifest.
    json_write(manifest_path, _manifest(output_dir, output_dir.name, manifest["parent_manifests"]))
    if independent_verify(output_dir)["status"] != "PASS":
        raise RuntimeError("Final hash-bound Phase 3 verification did not pass")
    return output_dir


def finalize_pending_run(run_dir: Path) -> Path:
    """Finalize a newly generated run after a verifier-only correction, without retraining."""
    from verify_phase3_authoritative_run import verify as independent_verify

    run_dir = run_dir.resolve()
    if not run_dir.name.startswith("phase3_authoritative_"):
        raise ValueError("Refusing to finalize a directory that is not a Phase 3 authoritative run")
    config_path = run_dir / "phase3_config.json"
    manifest_path = run_dir / "phase3_manifest.json"
    report_path = run_dir / "phase3_verification_report.json"
    if not config_path.is_file() or not manifest_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Run is missing configuration, manifest, or verifier report")
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    if previous.get("status") not in {"PENDING", "FAIL"}:
        raise ValueError("Refusing to finalize a run that already has a completed verifier report")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verifier = Path(__file__).resolve().with_name("verify_phase3_authoritative_run.py")
    shutil.copy2(verifier, run_dir / "source_snapshot" / verifier.name)
    config.setdefault("source_sha256", {})[verifier.name] = sha256(run_dir / "source_snapshot" / verifier.name)
    json_write(config_path, config)
    parents = json.loads(manifest_path.read_text(encoding="utf-8"))["parent_manifests"]
    json_write(report_path, {"status": "PENDING", "checks": [], "failed_checks": []})
    json_write(manifest_path, _manifest(run_dir, run_dir.name, parents))
    result = independent_verify(run_dir)
    json_write(report_path, result)
    json_write(manifest_path, _manifest(run_dir, run_dir.name, parents))
    final_result = independent_verify(run_dir)
    if final_result["status"] != "PASS":
        json_write(report_path, final_result)
        json_write(manifest_path, _manifest(run_dir, run_dir.name, parents))
        raise RuntimeError(f"Final Phase 3 verification failed: {final_result['failed_checks']}")
    json_write(report_path, final_result)
    json_write(manifest_path, _manifest(run_dir, run_dir.name, parents))
    if independent_verify(run_dir)["status"] != "PASS":
        raise RuntimeError("Hash-bound final Phase 3 verification did not pass")
    return run_dir


def refresh_environment_metadata(run_dir: Path) -> Path:
    """Correct package-distribution aliases in an already completed run record."""
    from verify_phase3_authoritative_run import verify as independent_verify

    run_dir = run_dir.resolve()
    config_path, report_path, manifest_path = (run_dir / "phase3_config.json",
                                               run_dir / "phase3_verification_report.json",
                                               run_dir / "phase3_manifest.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("Only a completed passing run can receive an environment metadata correction")
    if config.get("run_id") != run_dir.name:
        raise ValueError("Run identity does not match configuration")
    actual = dependency_versions()
    previous = dict(config.get("dependencies", {}))
    config["dependencies"] = actual
    config["environment_metadata_correction"] = {
        "reason": "The tensorflow Python distribution metadata key was absent although tensorflow-cpu was installed.",
        "previous_metadata": previous,
        "corrected_metadata": actual,
        "note": "Training and inference used the installed TensorFlow CPU runtime; this corrects package metadata only and does not alter experiment artifacts.",
    }
    json_write(config_path, config)
    parents = json.loads(manifest_path.read_text(encoding="utf-8"))["parent_manifests"]
    json_write(manifest_path, _manifest(run_dir, run_dir.name, parents))
    corrected = independent_verify(run_dir)
    if corrected["status"] != "PASS":
        raise RuntimeError(f"Environment metadata correction invalidated verification: {corrected['failed_checks']}")
    json_write(report_path, corrected)
    json_write(manifest_path, _manifest(run_dir, run_dir.name, parents))
    if independent_verify(run_dir)["status"] != "PASS":
        raise RuntimeError("Hash-bound report failed after metadata correction")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=Path("runtime_recording") / PHASE1_RUN)
    parser.add_argument("--phase2-dir", type=Path, default=Path("runtime_recording") / PHASE2_RUN)
    parser.add_argument("--output-root", type=Path, default=Path("runtime_recording"))
    parser.add_argument("--finalize-existing-dir", type=Path,
                        help="Finalize an incomplete, newly created Phase 3 run without retraining")
    parser.add_argument("--refresh-environment-metadata", type=Path,
                        help="Correct installed dependency aliases in a completed Phase 3 run")
    args = parser.parse_args()
    if args.finalize_existing_dir and args.refresh_environment_metadata:
        parser.error("Choose one finalization option")
    if args.finalize_existing_dir:
        directory = finalize_pending_run(args.finalize_existing_dir)
    elif args.refresh_environment_metadata:
        directory = refresh_environment_metadata(args.refresh_environment_metadata)
    else:
        directory = run(args.phase1_dir, args.phase2_dir, args.output_root)
    print(directory.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

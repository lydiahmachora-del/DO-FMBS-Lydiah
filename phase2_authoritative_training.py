"""Chapter Four Phase 2 federated training on a verified Phase 1 replay."""
from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import accuracy_score

from do_fmbs_core import json_dump, set_global_determinism


SOURCE_DOCUMENT_SHA256 = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"
CONFIG = {
    "phase": 2,
    "research_methodology": "two independent modality-specific identity classifiers; paired rows are computational, not person-level matches",
    "clients": 5,
    "rounds": 12,
    "local_epochs": 2,
    "batch_size": 64,
    "learning_rate": 0.001,
    "dropout_rate": 0.20,
    "loss": "sparse_categorical_crossentropy, independently per output",
    "loss_weights": {"face_output": 0.5, "finger_output": 0.5},
    "pairing": {
        "rule": "array position; face and fingerprint identity labels remain separate",
        "train_face_rows": "first 640 rows of Phase 1 X_train/y_train",
        "train_fingerprint_rows": "all 640 rows of Phase 1 X_train/y_train",
        "test_face_rows": "first 100 rows of Phase 1 X_test/y_test",
        "test_fingerprint_rows": "first 100 rows of Phase 1 X_test/y_test",
        "client_allocation": "permute the 640 paired training row indices, then split into five contiguous groups of 128",
        "client_allocation_seed": 42,
        "researcher_approval_date": "2026-11-23",
    },
    "architecture": {
        "face_input_features": 1850,
        "fingerprint_input_features": 9216,
        "face_hidden_units": [256, 128, 64],
        "fingerprint_hidden_units": [256, 128, 64],
        "activation": "relu",
        "batch_normalization": "after each hidden Dense layer",
        "dropout": "after third batch normalization in each modality branch",
        "face_output_name": "face_output",
        "face_classes": 62,
        "finger_output_name": "finger_output",
        "fingerprint_classes": 100,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_training_sources(output_dir: Path) -> dict[str, str]:
    """Persist and hash the exact local sources used to produce this run."""
    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    sources = {
        "phase2_authoritative_training.py": Path(__file__).resolve(),
        "do_fmbs_core.py": Path(__file__).resolve().with_name("do_fmbs_core.py"),
    }
    hashes: dict[str, str] = {}
    for name, source_path in sources.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"Required training source is missing: {source_path}")
        target = snapshot_dir / name
        shutil.copyfile(source_path, target)
        hashes[name] = sha256_file(target)
    return hashes


def array_digest(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def fedavg(client_weights: list[list[np.ndarray]], sample_counts: list[int]) -> list[np.ndarray]:
    """Sample-count-weight model tensors, including BatchNorm state tensors."""
    if not client_weights or len(client_weights) != len(sample_counts):
        raise ValueError("Each client weight list must have one positive sample count")
    if any(int(count) <= 0 for count in sample_counts):
        raise ValueError("FedAvg sample counts must be positive")
    width = len(client_weights[0])
    if any(len(weights) != width for weights in client_weights):
        raise ValueError("Client models have different numbers of weight tensors")
    total = float(sum(sample_counts))
    result: list[np.ndarray] = []
    for tensor_index in range(width):
        reference = np.asarray(client_weights[0][tensor_index])
        if any(np.asarray(weights[tensor_index]).shape != reference.shape for weights in client_weights):
            raise ValueError(f"Client weight shape mismatch at tensor {tensor_index}")
        averaged = np.zeros(reference.shape, dtype=np.result_type(reference.dtype, np.float64))
        for weights, count in zip(client_weights, sample_counts):
            averaged += np.asarray(weights[tensor_index], dtype=averaged.dtype) * (count / total)
        result.append(averaged.astype(reference.dtype, copy=False))
    return result


def phase2_indices(face_train_count: int, finger_train_count: int,
                   face_test_count: int, finger_test_count: int,
                   allocation_seed: int = 42) -> dict[str, object]:
    if face_train_count < 640 or finger_train_count < 640:
        raise ValueError("At least 640 training rows per modality are required")
    if face_test_count < 100 or finger_test_count < 100:
        raise ValueError("At least 100 test rows per modality are required")
    permutation = np.random.RandomState(allocation_seed).permutation(640).astype(np.int64)
    clients = [chunk for chunk in np.array_split(permutation, 5)]
    if [len(chunk) for chunk in clients] != [128] * 5:
        raise RuntimeError("Phase 2 client allocation is not exactly 128 paired records each")
    return {
        "train_face": np.arange(640, dtype=np.int64),
        "train_fingerprint": np.arange(640, dtype=np.int64),
        "test_face": np.arange(100, dtype=np.int64),
        "test_fingerprint": np.arange(100, dtype=np.int64),
        "client_allocation_permutation": permutation,
        "client_indices": clients,
    }


def pair_phase1_arrays(phase1_dir: Path, allocation_seed: int = 42) -> dict[str, object]:
    """Load validated arrays and build the researcher-approved positional pairs."""
    manifest_path = phase1_dir / "phase1_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Phase 1 manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != 1 or manifest.get("replay_id") != phase1_dir.name:
        raise ValueError("Phase 1 manifest identity does not match its directory")
    metadata = manifest.get("metadata", {})
    if metadata.get("source_document_sha256") != SOURCE_DOCUMENT_SHA256:
        raise ValueError("Phase 1 replay is not tied to the authoritative Chapter Four source")
    if manifest.get("verification", {}).get("status") != "PASS":
        raise ValueError("Phase 1 replay does not have a passing saved verification status")

    for artifact in manifest.get("artifacts", []):
        path = phase1_dir / artifact["path"]
        if not path.is_file() or path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Phase 1 artifact missing or size-mismatched: {artifact['path']}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Phase 1 artifact hash mismatch: {artifact['path']}")

    with np.load(phase1_dir / "face_data.npz", allow_pickle=False) as face:
        face_train_x, face_train_y = face["X_train"].copy(), face["y_train"].copy()
        face_test_x, face_test_y = face["X_test"].copy(), face["y_test"].copy()
        n_face_classes = int(face["n_classes"][0])
    with np.load(phase1_dir / "fingerprint_data.npz", allow_pickle=False) as finger:
        finger_train_x, finger_train_y = finger["X_train"].copy(), finger["y_train"].copy()
        finger_test_x, finger_test_y = finger["X_test"].copy(), finger["y_test"].copy()
        n_finger_classes = int(finger["n_classes"][0])

    if (len(face_train_y) < 640 or len(finger_train_y) != 640
            or len(face_test_y) < 100 or len(finger_test_y) < 100):
        raise ValueError("Phase 1 arrays do not satisfy the prescribed Phase 2 pair counts")
    if face_train_x.shape[1] != 1850 or finger_train_x.shape[1] != 9216:
        raise ValueError("Unexpected modality feature dimensions")
    if n_face_classes != 62 or n_finger_classes != 100:
        raise ValueError("Unexpected modality class counts")

    indices = phase2_indices(len(face_train_y), len(finger_train_y),
                             len(face_test_y), len(finger_test_y), allocation_seed)
    train_face_rows = indices["train_face"]
    train_finger_rows = indices["train_fingerprint"]
    test_face_rows = indices["test_face"]
    test_finger_rows = indices["test_fingerprint"]
    client_indices = indices["client_indices"]

    return {
        "face_train_x": face_train_x[train_face_rows].astype(np.float32, copy=False),
        "face_train_y": face_train_y[train_face_rows].astype(np.int64, copy=False),
        "finger_train_x": finger_train_x[train_finger_rows].astype(np.float32, copy=False),
        "finger_train_y": finger_train_y[train_finger_rows].astype(np.int64, copy=False),
        "face_test_x": face_test_x[test_face_rows].astype(np.float32, copy=False),
        "face_test_y": face_test_y[test_face_rows].astype(np.int64, copy=False),
        "finger_test_x": finger_test_x[test_finger_rows].astype(np.float32, copy=False),
        "finger_test_y": finger_test_y[test_finger_rows].astype(np.int64, copy=False),
        "client_indices": client_indices,
        "client_allocation_permutation": indices["client_allocation_permutation"],
        "source_rows": {
            "train_face": train_face_rows,
            "train_fingerprint": train_finger_rows,
            "test_face": test_face_rows,
            "test_fingerprint": test_finger_rows,
        },
        "class_counts": {"face": n_face_classes, "fingerprint": n_finger_classes},
    }


def create_phase2_model(face_features: int = 1850, fingerprint_features: int = 9216,
                        face_classes: int = 62, fingerprint_classes: int = 100,
                        dropout_rate: float = 0.20):
    import tensorflow as tf

    def branch(inputs, prefix: str):
        value = inputs
        for index, units in enumerate((256, 128, 64), start=1):
            value = tf.keras.layers.Dense(units, activation="relu", name=f"{prefix}_dense_{index}")(value)
            value = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn_{index}")(value)
        return tf.keras.layers.Dropout(dropout_rate, name=f"{prefix}_dropout")(value)

    face_input = tf.keras.Input((face_features,), name="face_input")
    finger_input = tf.keras.Input((fingerprint_features,), name="finger_input")
    face_embedding = branch(face_input, "face")
    finger_embedding = branch(finger_input, "finger")
    face_output = tf.keras.layers.Dense(face_classes, activation="softmax", name="face_output")(face_embedding)
    finger_output = tf.keras.layers.Dense(fingerprint_classes, activation="softmax", name="finger_output")(finger_embedding)
    return tf.keras.Model(
        inputs={"face_input": face_input, "finger_input": finger_input},
        outputs={"face_output": face_output, "finger_output": finger_output},
        name="dofmbs_phase2_independent_dual_branch",
    )


def compile_phase2_model(model, learning_rate: float = 0.001):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"face_output": "sparse_categorical_crossentropy",
              "finger_output": "sparse_categorical_crossentropy"},
        loss_weights={"face_output": 0.5, "finger_output": 0.5},
        metrics={"face_output": ["accuracy"], "finger_output": ["accuracy"]},
    )
    return model


def save_weight_arrays(path: Path, weights: list[np.ndarray]) -> None:
    np.savez_compressed(path, **{f"weight_{i:03d}": value for i, value in enumerate(weights)})


def model_summary(model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append, expand_nested=True, show_trainable=True)
    return "\n".join(lines) + "\n"


def evaluate_global(model, arrays: dict[str, object], prefix: str) -> dict[str, float]:
    x = {"face_input": arrays[f"{prefix}face_x"], "finger_input": arrays[f"{prefix}finger_x"]}
    y = {"face_output": arrays[f"{prefix}face_y"], "finger_output": arrays[f"{prefix}finger_y"]}
    return {key: float(value) for key, value in model.evaluate(x, y, batch_size=64, verbose=0, return_dict=True).items()}


def dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scikit-learn", "tensorflow-cpu", "keras"):
        try:
            key = "tensorflow" if name == "tensorflow-cpu" else name
            versions[key] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "tensorflow-cpu":
                try:
                    import tensorflow as tf
                    versions["tensorflow"] = tf.__version__
                except ImportError:
                    versions["tensorflow"] = "not-installed"
            else:
                versions[name] = "not-installed"
    try:
        versions["tensorflow_cpu_distribution"] = importlib.metadata.version("tensorflow-cpu")
    except importlib.metadata.PackageNotFoundError:
        versions["tensorflow_cpu_distribution"] = "not-installed"
    return versions


def correct_runtime_provenance(run_dir: Path) -> None:
    """Correct dependency-package naming and preserve an explicit correction trail."""
    config_path = run_dir / "phase2_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    actual_versions = dependency_versions()
    correction = {
        "corrected_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "TensorFlow is installed from the tensorflow-cpu distribution; querying the tensorflow distribution name alone returned not-installed.",
        "verified_versions": {key: actual_versions.get(key) for key in
                              ("python", "numpy", "scikit-learn", "tensorflow", "tensorflow_cpu_distribution", "keras")},
        "determinism_note": "The run called set_global_determinism(42), which requests TensorFlow op determinism. TensorFlow 2.13 does not expose the status query used in the initial run metadata, so the old false value meant unconfirmed, not confirmed-disabled.",
        "training_artifacts_changed": False,
        "source_sha256_at_training_time_preserved": config.get("source_sha256", {}).get("phase2_authoritative_training.py"),
    }
    config["environment"] = actual_versions
    config["determinism"] = {
        "tensorflow_op_determinism_requested": True,
        "runtime_state_query_supported": False,
        "note": correction["determinism_note"],
    }
    config["provenance_correction"] = correction
    json_dump(config_path, config)
    json_dump(run_dir / "phase2_provenance_correction.json", correction)
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "phase2_manifest.json":
            artifacts.append({"path": path.relative_to(run_dir).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest_path = run_dir / "phase2_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = artifacts
    manifest["provenance_correction"] = True
    json_dump(manifest_path, manifest)


def source_revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_training(phase1_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing Phase 2 run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot_training_sources(output_dir)
    started = time.perf_counter()
    set_global_determinism(42)
    import tensorflow as tf
    try:
        tf.config.experimental.enable_op_determinism()
        deterministic_ops = True
    except Exception:
        deterministic_ops = False

    arrays = pair_phase1_arrays(phase1_dir, allocation_seed=42)
    n_face, n_finger = arrays["class_counts"]["face"], arrays["class_counts"]["fingerprint"]
    model_factory = lambda: create_phase2_model(face_classes=n_face, fingerprint_classes=n_finger)
    global_model = compile_phase2_model(model_factory())
    model_text = model_summary(global_model)
    (output_dir / "model_summary.txt").write_text(model_text, encoding="utf-8")
    initial_weights = global_model.get_weights()
    initial_hash = array_digest(initial_weights)
    save_weight_arrays(output_dir / "initial_global_weights.npz", initial_weights)

    client_ids = arrays["client_indices"]
    client_payload = {
        "schema_version": 1,
        "pairing_rule": CONFIG["pairing"]["rule"],
        "allocation_seed": 42,
        "clients": [{"client_id": i, "paired_sample_count": len(indices),
                     "pair_positions": indices.astype(int).tolist(),
                     "face_source_rows": arrays["source_rows"]["train_face"][indices].astype(int).tolist(),
                     "fingerprint_source_rows": arrays["source_rows"]["train_fingerprint"][indices].astype(int).tolist(),
                     "face_labels": arrays["face_train_y"][indices].astype(int).tolist(),
                     "fingerprint_labels": arrays["finger_train_y"][indices].astype(int).tolist()}
                    for i, indices in enumerate(client_ids)],
    }
    json_dump(output_dir / "client_allocation.json", client_payload)
    pairing = {key: value.astype(int).tolist() for key, value in arrays["source_rows"].items()}
    pairing.update({"client_allocation_permutation": arrays["client_allocation_permutation"].astype(int).tolist(),
                    "face_train_labels": arrays["face_train_y"].astype(int).tolist(),
                    "fingerprint_train_labels": arrays["finger_train_y"].astype(int).tolist(),
                    "face_test_labels": arrays["face_test_y"].astype(int).tolist(),
                    "fingerprint_test_labels": arrays["finger_test_y"].astype(int).tolist(),
                    "identity_correspondence": False,
                    "researcher_approved_pairing": CONFIG["pairing"]})
    json_dump(output_dir / "pairing_metadata.json", pairing)
    np.savez_compressed(output_dir / "pairing_indices.npz", **{
        key: value for key, value in arrays["source_rows"].items()},
        client_allocation_permutation=arrays["client_allocation_permutation"])

    histories: list[dict[str, object]] = []
    initialization_records: list[dict[str, object]] = []
    round_rows: list[dict[str, float | int]] = []
    rounds_dir = output_dir / "global_weights_by_round"
    rounds_dir.mkdir()

    for round_number in range(1, 13):
        global_before = global_model.get_weights()
        global_before_hash = array_digest(global_before)
        client_weights: list[list[np.ndarray]] = []
        client_counts: list[int] = []
        for client_id, paired_indices in enumerate(client_ids):
            local_model = compile_phase2_model(model_factory())
            local_model.set_weights(global_before)
            local_initial_hash = array_digest(local_model.get_weights())
            matches_global = local_initial_hash == global_before_hash
            initialization_records.append({
                "round": round_number, "client_id": client_id,
                "global_weight_sha256_before_round": global_before_hash,
                "client_weight_sha256_immediately_after_initialization": local_initial_hash,
                "initialization_matches_global": matches_global,
                "sample_count": int(len(paired_indices)),
            })
            if not matches_global:
                raise RuntimeError(f"Client {client_id} did not initialize from round {round_number} global weights")

            local_arrays = {
                "face_train_x": arrays["face_train_x"][paired_indices],
                "face_train_y": arrays["face_train_y"][paired_indices],
                "finger_train_x": arrays["finger_train_x"][paired_indices],
                "finger_train_y": arrays["finger_train_y"][paired_indices],
            }
            tf.keras.utils.set_random_seed(42 + 100 * round_number + client_id)
            fitted = local_model.fit(
                {"face_input": local_arrays["face_train_x"], "finger_input": local_arrays["finger_train_x"]},
                {"face_output": local_arrays["face_train_y"], "finger_output": local_arrays["finger_train_y"]},
                epochs=2, batch_size=64, shuffle=True, verbose=0,
            )
            histories.append({
                "round": round_number, "client_id": client_id, "sample_count": int(len(paired_indices)),
                "initial_global_weight_sha256": global_before_hash,
                "epoch_history": {name: [float(v) for v in values] for name, values in fitted.history.items()},
            })
            client_weights.append(local_model.get_weights())
            client_counts.append(int(len(paired_indices)))
            del local_model

        aggregation_weights = [count / sum(client_counts) for count in client_counts]
        global_after = fedavg(client_weights, client_counts)
        global_model.set_weights(global_after)
        round_path = rounds_dir / f"round_{round_number:02d}_global_weights.npz"
        save_weight_arrays(round_path, global_after)
        # Evaluate both heads on the paired training rows and the fixed 100-row holdout.
        train_eval = global_model.evaluate(
            {"face_input": arrays["face_train_x"], "finger_input": arrays["finger_train_x"]},
            {"face_output": arrays["face_train_y"], "finger_output": arrays["finger_train_y"]},
            batch_size=64, verbose=0, return_dict=True,
        )
        test_eval = global_model.evaluate(
            {"face_input": arrays["face_test_x"], "finger_input": arrays["finger_test_x"]},
            {"face_output": arrays["face_test_y"], "finger_output": arrays["finger_test_y"]},
            batch_size=64, verbose=0, return_dict=True,
        )
        row: dict[str, float | int] = {"round": round_number}
        for prefix, values in (("train", train_eval), ("test", test_eval)):
            row.update({f"{prefix}_{key}": float(value) for key, value in values.items()})
        row["client_sample_counts"] = ",".join(map(str, client_counts))  # type: ignore[assignment]
        row["client_fedavg_weights"] = ",".join(f"{weight:.8f}" for weight in aggregation_weights)  # type: ignore[assignment]
        round_rows.append(row)
        print(f"Round {round_number:02d}/12: test_face_acc={float(test_eval['face_output_accuracy']):.4f}; "
              f"test_fingerprint_acc={float(test_eval['finger_output_accuracy']):.4f}", flush=True)

    with (output_dir / "round_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = list(round_rows[0])
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(round_rows)
    json_dump(output_dir / "client_training_histories.json", histories)
    json_dump(output_dir / "client_initialization_evidence.json", initialization_records)

    predictions = global_model.predict(
        {"face_input": arrays["face_test_x"], "finger_input": arrays["finger_test_x"]},
        batch_size=64, verbose=0,
    )
    test_face_prob = predictions["face_output"]
    test_finger_prob = predictions["finger_output"]
    face_pred = np.argmax(test_face_prob, axis=1).astype(np.int64)
    finger_pred = np.argmax(test_finger_prob, axis=1).astype(np.int64)
    np.savez_compressed(
        output_dir / "final_evaluation_predictions.npz",
        face_labels=arrays["face_test_y"], face_predictions=face_pred,
        face_probabilities=test_face_prob,
        fingerprint_labels=arrays["finger_test_y"], fingerprint_predictions=finger_pred,
        fingerprint_probabilities=test_finger_prob,
    )
    final_path = output_dir / "final_global_model.keras"
    global_model.save(final_path)
    final_weights = global_model.get_weights()
    final_hash = array_digest(final_weights)
    final_face_accuracy = float(accuracy_score(arrays["face_test_y"], face_pred))
    final_finger_accuracy = float(accuracy_score(arrays["finger_test_y"], finger_pred))

    phase1_manifest_hash = sha256_file(phase1_dir / "phase1_manifest.json")
    config = {
        **CONFIG,
        "run_id": output_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase1_replay_id": phase1_dir.name,
        "phase1_manifest_sha256": phase1_manifest_hash,
        "chapter_four_sha256": SOURCE_DOCUMENT_SHA256,
        "random_seeds": {
            "client_allocation": 42,
            "initial_model_and_global_determinism": 42,
            "client_training_formula": "42 + 100 * round_number + client_id",
        },
        "environment": dependency_versions(),
        "platform": platform.platform(),
        "source_revision": source_revision(),
        "source_sha256": source_hashes,
        "source_snapshot_directory": "source_snapshot",
        "determinism": {
            "tensorflow_op_determinism_enabled": deterministic_ops,
            "note": "Exact bit-for-bit reproduction may vary with TensorFlow, BLAS, hardware, and CPU instruction implementations.",
        },
        "initial_global_weight_sha256": initial_hash,
        "final_global_weight_sha256": final_hash,
        "runtime_seconds": time.perf_counter() - started,
    }
    json_dump(output_dir / "phase2_config.json", config)
    results = {
        "phase": 2,
        "run_id": output_dir.name,
        "phase1_replay_id": phase1_dir.name,
        "architecture_scope": CONFIG["research_methodology"],
        "paired_test_records": 100,
        "face_classification": {"samples": 100, "classes": 62, "accuracy": final_face_accuracy,
                                 "correct": int(np.sum(face_pred == arrays["face_test_y"]))},
        "fingerprint_classification": {"samples": 100, "classes": 100, "accuracy": final_finger_accuracy,
                                        "correct": int(np.sum(finger_pred == arrays["finger_test_y"]))},
        "history": {
            "round": [int(row["round"]) for row in round_rows],
            "face_accuracy": [float(row["test_face_output_accuracy"]) for row in round_rows],
            "fingerprint_accuracy": [float(row["test_finger_output_accuracy"]) for row in round_rows],
            "face_loss": [float(row["test_face_output_loss"]) for row in round_rows],
            "fingerprint_loss": [float(row["test_finger_output_loss"]) for row in round_rows],
        },
        "runtime_seconds": config["runtime_seconds"],
        "result_interpretation": "Separate modality classification only; no person-level multimodal accuracy, adversarial robustness, or privacy claim.",
    }
    json_dump(output_dir / "phase2_metrics.json", results)

    verification = {
        "status": "PASS",
        "checks": [
            {"check": "Phase 1 manifest/hash verification", "status": "PASS", "detail": "All source manifest artifacts were rehashed before loading."},
            {"check": "Paired records and independent labels", "status": "PASS", "detail": "640 train pairs, 100 test pairs; modality labels retained separately."},
            {"check": "Five equal client allocations", "status": "PASS", "detail": "5 clients x 128 rows = 640; client permutation seed 42."},
            {"check": "Round/client initialization lineage", "status": "PASS" if len(initialization_records) == 60 and all(item["initialization_matches_global"] for item in initialization_records) else "FAIL", "detail": "Each of 60 local starts is hashed and compared with that round's global state."},
            {"check": "Training schedule", "status": "PASS" if len(round_rows) == 12 and len(histories) == 60 else "FAIL", "detail": "12 rounds, 5 clients per round, 2 local epochs per client."},
            {"check": "Sample-count weighted aggregation", "status": "PASS" if all(len(row["client_fedavg_weights"].split(",")) == 5 for row in round_rows) else "FAIL", "detail": "Weights computed from each client sample count (128 each), never from metrics."},
            {"check": "Separate model outputs and evaluation", "status": "PASS" if len(face_pred) == 100 and len(finger_pred) == 100 else "FAIL", "detail": "Independent 62-class face and 100-class fingerprint outputs and labels."},
        ],
        "face_accuracy": final_face_accuracy,
        "fingerprint_accuracy": final_finger_accuracy,
        "baseline_reference": {"face_approximate": 0.26, "fingerprint": 0.0, "is_guaranteed": False},
    }
    verification["status"] = "PASS" if all(item["status"] == "PASS" for item in verification["checks"]) else "FAIL"
    json_dump(output_dir / "phase2_verification_report.json", verification)

    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "phase2_manifest.json":
            artifacts.append({"path": path.relative_to(output_dir).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    json_dump(output_dir / "phase2_manifest.json", {
        "schema_version": 1, "phase": 2, "run_id": output_dir.name,
        "phase1_replay_id": phase1_dir.name,
        "status": verification["status"], "artifacts": artifacts,
    })
    print(f"Phase 2 run saved: {output_dir}", flush=True)
    print(f"Face test accuracy: {final_face_accuracy:.4f}; fingerprint test accuracy: {final_finger_accuracy:.4f}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, required=True,
                        help="Verified Phase 1 replay directory (for example runtime_recording/phase1_final_candidate_...).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New, non-existing Phase 2 run directory; existing paths are never overwritten.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_training(args.phase1_dir.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

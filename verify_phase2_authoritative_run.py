"""Independent, read-only audit of a persisted authoritative Phase 2 run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_DOC = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_digest(path):
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            value = np.ascontiguousarray(archive[name])
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(value.shape).encode("ascii"))
            digest.update(value.tobytes())
    return digest.hexdigest()


def audit(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "phase2_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "phase2_config.json").read_text(encoding="utf-8"))
    checks = []

    def record(name, passed, evidence):
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    artifacts_bad = []
    for artifact in manifest["artifacts"]:
        path = run_dir / artifact["path"]
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            artifacts_bad.append(artifact["path"])
    record("Phase 2 manifest hashes", not artifacts_bad, f"{len(manifest['artifacts'])} artifacts; bad={artifacts_bad}")
    record("Run and source identity", manifest["run_id"] == run_dir.name == config["run_id"]
           and config["chapter_four_sha256"] == SOURCE_DOC, run_dir.name)

    source_hashes = config.get("source_sha256", {})
    snapshot_dir = run_dir / config.get("source_snapshot_directory", "source_snapshot")
    source_bad = []
    for name, expected_hash in source_hashes.items():
        path = snapshot_dir / name
        if not path.is_file() or sha256(path) != expected_hash:
            source_bad.append(name)
    source_ok = bool(source_hashes) and not source_bad
    record("Training source snapshot hashes", source_ok,
           f"{len(source_hashes)} source files; mismatches={source_bad}")

    phase1_dir = run_dir.parent / config["phase1_replay_id"]
    phase1_manifest_path = phase1_dir / "phase1_manifest.json"
    phase1_manifest = json.loads(phase1_manifest_path.read_text(encoding="utf-8"))
    p1_bad = []
    for item in phase1_manifest["artifacts"]:
        path = phase1_dir / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            p1_bad.append(item["path"])
    p1_ok = not p1_bad and sha256(phase1_manifest_path) == config["phase1_manifest_sha256"]
    record("Phase 1 source integrity", p1_ok, f"{phase1_dir.name}; mismatches={p1_bad}")

    pairs = json.loads((run_dir / "pairing_metadata.json").read_text(encoding="utf-8"))
    clients = json.loads((run_dir / "client_allocation.json").read_text(encoding="utf-8"))["clients"]
    positions = [int(i) for client in clients for i in client["pair_positions"]]
    allocation_ok = len(clients) == 5 and all(c["paired_sample_count"] == 128 for c in clients)
    allocation_ok &= sorted(positions) == list(range(640))
    pair_ok = (len(pairs["train_face"]) == len(pairs["train_fingerprint"]) == 640
               and len(pairs["test_face"]) == len(pairs["test_fingerprint"]) == 100
               and pairs["identity_correspondence"] is False)
    record("Pairing and equal client allocation", pair_ok and allocation_ok,
           f"640 train pairs; 100 test pairs; client counts={[c['paired_sample_count'] for c in clients]}")
    separate_labels = (all(0 <= y < 62 for c in clients for y in c["face_labels"])
                       and all(0 <= y < 100 for c in clients for y in c["fingerprint_labels"]))
    record("Independent label spaces", separate_labels, "face 0..61; fingerprint 0..99")

    initialization = json.loads((run_dir / "client_initialization_evidence.json").read_text(encoding="utf-8"))
    global_hashes = {r: weights_digest(run_dir / "global_weights_by_round" / f"round_{r:02d}_global_weights.npz")
                     for r in range(1, 13)}
    lineage = len(initialization) == 60 and all(x["initialization_matches_global"] for x in initialization)
    lineage &= all(x["global_weight_sha256_before_round"] == (weights_digest(run_dir / "initial_global_weights.npz")
                            if x["round"] == 1 else global_hashes[x["round"] - 1]) for x in initialization)
    record("Client initialization lineage", bool(lineage), "60 starts compared with persisted preceding global weights")

    with (run_dir / "round_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rounds = list(csv.DictReader(stream))
    histories = json.loads((run_dir / "client_training_histories.json").read_text(encoding="utf-8"))
    schedule = (len(rounds) == 12 and len(histories) == 60
                and all(len(x["epoch_history"]["loss"]) == 2 for x in histories))
    fedavg_ok = all(row["client_sample_counts"] == "128,128,128,128,128"
                    and np.allclose([float(x) for x in row["client_fedavg_weights"].split(",")], [.2] * 5,
                                    rtol=0, atol=1e-8) for row in rounds)
    record("Federated schedule and sample-weighted FedAvg", schedule and fedavg_ok,
           "12 rounds; 5 clients; 2 local epochs; aggregation weights from counts")

    with np.load(run_dir / "final_evaluation_predictions.npz", allow_pickle=False) as data:
        prediction_shapes = (data["face_labels"].shape == data["face_predictions"].shape == (100,)
                             and data["fingerprint_labels"].shape == data["fingerprint_predictions"].shape == (100,)
                             and data["face_probabilities"].shape == (100, 62)
                             and data["fingerprint_probabilities"].shape == (100, 100))
        face_acc = float(np.mean(data["face_labels"] == data["face_predictions"]))
        finger_acc = float(np.mean(data["fingerprint_labels"] == data["fingerprint_predictions"]))
    results = json.loads((run_dir / "phase2_metrics.json").read_text(encoding="utf-8"))
    prediction_shapes &= (np.isclose(face_acc, results["face_classification"]["accuracy"])
                          and np.isclose(finger_acc, results["fingerprint_classification"]["accuracy"]))
    record("Separate held-out evaluation", bool(prediction_shapes),
           f"face {face_acc:.4f} and fingerprint {finger_acc:.4f}; 100 rows each")

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(run_dir / "final_global_model.keras", compile=False)
        architecture = (set(model.input_names) == {"face_input", "finger_input"}
                        and set(model.output_names) == {"face_output", "finger_output"}
                        and [model.get_layer(f"face_dense_{i}").units for i in (1, 2, 3)] == [256, 128, 64]
                        and [model.get_layer(f"finger_dense_{i}").units for i in (1, 2, 3)] == [256, 128, 64]
                        and model.get_layer("face_output").units == 62
                        and model.get_layer("finger_output").units == 100
                        and model.get_layer("face_dropout").rate == model.get_layer("finger_dropout").rate == .20)
    except Exception as error:
        architecture, detail = False, str(error)
    else:
        detail = "reloaded Keras model has prescribed branch sizes, heads, and dropout"
    record("Saved model architecture", architecture, detail)

    failures = [x["check"] for x in checks if x["status"] != "PASS"]
    return {"run_id": run_dir.name, "status": "PASS" if not failures else "FAIL",
            "checks": checks, "failed_checks": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    result = audit(parser.parse_args().run_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

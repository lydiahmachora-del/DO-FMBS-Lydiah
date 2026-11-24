"""Independent acceptance checks for persisted Phase 4 defense experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

SOURCE_DOC_SHA256 = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _manifest(directory: Path, name: str) -> tuple[dict, list[str]]:
    data = json.loads((directory / name).read_text(encoding="utf-8"))
    bad = []
    for item in data.get("artifacts", []):
        p = directory / item["path"]
        if not p.is_file() or p.stat().st_size != item["bytes"] or sha256(p) != item["sha256"]:
            bad.append(item["path"])
    return data, bad


def _outputs(model, face, finger):
    out = model({"face_input": np.asarray(face, np.float32),
                 "finger_input": np.asarray(finger, np.float32)}, training=False)
    if isinstance(out, dict):
        return np.asarray(out["face_output"]), np.asarray(out["finger_output"])
    return np.asarray(out[0]), np.asarray(out[1])


def verify(run_dir: Path) -> dict:
    import tensorflow as tf
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "phase4_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "phase4_metrics.json").read_text(encoding="utf-8"))
    histories = json.loads((run_dir / "training_histories.json").read_text(encoding="utf-8"))
    gradient_info = json.loads((run_dir / "gradient_evidence.json").read_text(encoding="utf-8"))
    manifest, bad_artifacts = _manifest(run_dir, "phase4_manifest.json")
    checks = []

    def record(name, passed, evidence):
        checks.append({"check": name, "status": "PASS" if bool(passed) else "FAIL", "evidence": str(evidence)})

    record("Run identity and authoritative source", run_dir.name == config.get("run_id") == metrics.get("run_id") == manifest.get("run_id")
           and config.get("phase") == 4 and config.get("source_document_sha256") == SOURCE_DOC_SHA256
           and manifest.get("source_document_sha256") == SOURCE_DOC_SHA256, run_dir.name)
    record("Phase 4 manifest artifacts", not bad_artifacts and bool(manifest.get("artifacts")),
           f"{len(manifest.get('artifacts', []))} listed; mismatches={bad_artifacts}")
    snapshot_bad = [name for name, digest in config.get("source_sha256", {}).items()
                    if not (run_dir / "source_snapshot" / name).is_file()
                    or sha256(run_dir / "source_snapshot" / name) != digest]
    record("Source code snapshot hashes", bool(config.get("source_sha256")) and not snapshot_bad, snapshot_bad)

    root = run_dir.parent.parent
    p1 = root / "runtime_recording" / config["source_phase1_run"]
    p2 = root / "runtime_recording" / config["source_phase2_run"]
    p3 = root / "runtime_recording" / config["source_phase3_run"]
    p1m, p1bad = _manifest(p1, "phase1_manifest.json")
    p2m, p2bad = _manifest(p2, "phase2_manifest.json")
    p3m, p3bad = _manifest(p3, "phase3_manifest.json")
    p1path, p2path, p3path = (p1 / "phase1_manifest.json", p2 / "phase2_manifest.json", p3 / "phase3_manifest.json")
    record("Accepted Phase 1 source unchanged", not p1bad and p1m.get("verification", {}).get("status") == "PASS"
           and sha256(p1path) == config.get("source_phase1_manifest_sha256"), p1bad)
    baseline_path = p2 / config["source_phase2_checkpoint"]
    baseline_entry = next((x for x in p2m.get("artifacts", []) if x["path"] == config["source_phase2_checkpoint"]), {})
    baseline_hash_before = sha256(baseline_path)
    record("Accepted Phase 2 checkpoint and manifest unchanged", not p2bad and baseline_hash_before == baseline_entry.get("sha256")
           == config.get("source_phase2_checkpoint_sha256") and sha256(p2 / "phase2_manifest.json") == config.get("source_phase2_manifest_sha256"), baseline_hash_before)
    record("Accepted Phase 3 evidence unchanged", not p3bad and p3m.get("run_id") == config.get("source_phase3_run")
           and sha256(p3path) == config.get("source_phase3_manifest_sha256")
           and json.loads((p3 / "phase3_config.json").read_text(encoding="utf-8")).get("source_phase2_model_sha256") == baseline_hash_before,
           f"Phase 3 run {p3m.get('run_id')}; mismatches={p3bad}")

    with np.load(p1 / "face_data.npz", allow_pickle=False) as f, np.load(p1 / "fingerprint_data.npz", allow_pickle=False) as g:
        expected = {"face_train_x": f["X_train"][:640].astype(np.float32),
                    "face_train_y": f["y_train"][:640].astype(np.int64),
                    "finger_train_x": g["X_train"][:640].astype(np.float32),
                    "finger_train_y": g["y_train"][:640].astype(np.int64),
                    "face_test_x": f["X_test"][:100].astype(np.float32),
                    "face_test_y": f["y_test"][:100].astype(np.int64),
                    "finger_test_x": g["X_test"][:100].astype(np.float32),
                    "finger_test_y": g["y_test"][:100].astype(np.int64),
                    "face_eval_ids": f["test_indices"][:100].astype(np.int64),
                    "finger_eval_ids": g["test_indices"][:100].astype(np.int64)}
        face_classes, finger_classes = int(f["n_classes"][0]), int(g["n_classes"][0])
        phase3_eval = np.load(p3 / "evasion_evidence.npz", allow_pickle=False)
        p3_rows_ok = (np.array_equal(phase3_eval["face_clean"], expected["face_test_x"])
                      and np.array_equal(phase3_eval["fingerprint_unchanged"], expected["finger_test_x"])
                      and np.array_equal(phase3_eval["labels"], expected["face_test_y"])
                      and np.array_equal(phase3_eval["evaluation_ids"], expected["face_eval_ids"]))
    with np.load(run_dir / "evaluation_records.npz", allow_pickle=False) as records:
        persisted_rows_ok = all(np.array_equal(records[k], v) for k, v in expected.items())
    record("Identical Phase 1/3 evaluation records and labels", p3_rows_ok and persisted_rows_ok,
           "First 100 stored test positions for each modality; exact IDs/features/labels")

    weights = config.get("training", {})
    record("Declared comparable training schedule", weights.get("epochs_each_defense") == 12
           and weights.get("combined_clean_pretraining_epochs") == 6
           and weights.get("combined_defense_epochs_after_pretraining") == 12
           and weights.get("batch_size") == 64 and weights.get("optimizer") == "Adam"
           and weights.get("loss_weights") == {"face": 0.5, "fingerprint": 0.5}, str(weights))
    definitions = {"dp_inspired": ("dp_inspired_model.keras", "dp_inspired_gaussian_noise"),
                   "face_adversarial": ("face_adversarial_model.keras", "face_adversarial_training"),
                   "combined": ("combined_model.keras", "combined_dofmbs_defense")}
    expected_states = {"federated_baseline", *(v[1] for v in definitions.values())}
    state_names = set(metrics.get("results", {}))
    record("Exactly four model states", state_names == expected_states,
           sorted(state_names))
    with np.load(run_dir / "evaluation_predictions.npz", allow_pickle=False) as ev:
        baseline_copy = run_dir / "baseline_model.keras"
        record("Baseline checkpoint copy is byte-identical", sha256(baseline_copy) == baseline_hash_before,
               sha256(baseline_copy))
        baseline = tf.keras.models.load_model(baseline_copy, compile=False)
        models = {"federated_baseline": baseline}
        for key, (filename, state_name) in definitions.items():
            model_path = run_dir / filename
            model = tf.keras.models.load_model(model_path, compile=False)
            models[state_name] = model
            baseline_weights, tuned_weights = baseline.get_weights(), model.get_weights()
            changed = any(not np.array_equal(a, b) for a, b in zip(baseline_weights, tuned_weights))
            record(f"{state_name} checkpoint loads and updates baseline parameters", changed,
                   f"weights changed={changed}; {len(tuned_weights)} tensors")
            hist = histories.get(key, [])
            expected_len = 18 if key == "combined" else 12
            record(f"{state_name} training history complete", len(hist) == expected_len
                   and all(np.isfinite(row["total_loss"]) and row["parameter_update_l2_epoch"] > 0 for row in hist),
                   f"{len(hist)} epochs")
        model_eval_ok = True
        asr_ok = True
        from phase3_authoritative_attacks import fgsm as phase3_fgsm
        for key, model in models.items():
            state = "federated_baseline" if key == "federated_baseline" else key
            stored_key = "baseline" if key == "federated_baseline" else next(k for k, v in definitions.items() if v[1] == key)
            face_prob, finger_prob = _outputs(model, expected["face_test_x"], expected["finger_test_x"])
            face_pred, finger_pred = face_prob.argmax(1), finger_prob.argmax(1)
            face_label, finger_label = expected["face_test_y"], expected["finger_test_y"]
            result = metrics["results"][state]
            model_eval_ok &= (np.array_equal(ev[f"{stored_key}_face_predictions"], face_pred)
                              and np.array_equal(ev[f"{stored_key}_fingerprint_predictions"], finger_pred)
                              and np.array_equal(ev[f"{stored_key}_face_labels"], face_label)
                              and np.array_equal(ev[f"{stored_key}_fingerprint_labels"], finger_label)
                              and np.isclose(result["face"]["accuracy"], accuracy_score(face_label, face_pred))
                              and np.isclose(result["fingerprint"]["accuracy"], accuracy_score(finger_label, finger_pred))
                              and np.array_equal(np.asarray(result["face"]["confusion_matrix"]), confusion_matrix(face_label, face_pred, labels=np.arange(face_classes)))
                              and np.array_equal(np.asarray(result["fingerprint"]["confusion_matrix"]), confusion_matrix(finger_label, finger_pred, labels=np.arange(finger_classes))))
            adv = ev[f"{stored_key}_face_adversarial_features"]
            delta = adv - expected["face_test_x"]
            reference_adv = phase3_fgsm(model, expected["face_test_x"], expected["finger_test_x"],
                                         expected["face_test_y"], epsilon=0.05)
            attacked_prob, _ = _outputs(model, adv, expected["finger_test_x"])
            attacked_pred = attacked_prob.argmax(1)
            clean_correct = face_pred == face_label
            successes = clean_correct & (attacked_pred != face_label)
            m = result["face_fgsm"]
            asr_ok &= (np.allclose(adv, reference_adv, rtol=0, atol=1e-7)
                       and np.all(np.abs(delta) <= 0.05 + 1e-6)
                       and np.array_equal(ev[f"{stored_key}_face_adversarial_predictions"], attacked_pred)
                       and np.array_equal(ev[f"{stored_key}_face_clean_correct_mask"], clean_correct)
                       and np.array_equal(ev[f"{stored_key}_face_attack_success_mask"], successes)
                       and m["clean_correct_denominator"] == int(clean_correct.sum())
                       and m["attack_successes"] == int(successes.sum())
                       and np.isclose(m["attack_success_rate"], successes.sum() / max(1, clean_correct.sum()))
                       and m["samples"] == 100 and m["epsilon"] == 0.05)
        record("Reloaded predictions, labels, accuracy and confusion matrices", model_eval_ok,
               "All four checkpoints independently re-predicted against the same persisted records")
        record("FGSM perturbations, predictions and conditional ASR", asr_ok,
               "Regenerated with the Phase 3 FGSM function for each checkpoint; per-model clean-correct denominator")

    with np.load(run_dir / "adversarial_training_examples.npz", allow_pickle=False) as proof:
        from phase3_authoritative_attacks import fgsm as phase3_fgsm
        proof_ok = True
        for state in ("face_adversarial", "combined"):
            model_path = baseline_copy if state == "face_adversarial" else run_dir / "combined_pretrained_model.keras"
            source_model = tf.keras.models.load_model(model_path, compile=False)
            clean = proof[f"{state}_clean_face"]
            finger_batch = proof[f"{state}_fingerprint_inputs"]
            labels = proof[f"{state}_face_labels"]
            source_rows = proof[f"{state}_source_rows"]
            reference = phase3_fgsm(source_model, clean, finger_batch, labels, epsilon=0.05)
            original_finger = expected["finger_train_x"][source_rows]
            original_finger_y = expected["finger_train_y"][source_rows]
            original_face_y = expected["face_train_y"][source_rows]
            proof_ok &= (np.array_equal(labels, original_face_y)
                         and np.array_equal(finger_batch, original_finger)
                         and np.array_equal(proof[f"{state}_fingerprint_labels"], original_finger_y)
                         and np.allclose(proof[f"{state}_adversarial_face"], reference, rtol=0, atol=1e-7)
                         and np.allclose(proof[f"{state}_adversarial_face"] - clean, 0.05 * np.sign(reference - clean), atol=1e-7))
        record("Adversarial-training examples use Phase 3 FGSM and preserve paired labels/inputs",
               proof_ok, "Training batches independently regenerated from baseline and combined warm-up checkpoint")

    dp = config["defenses"]["dp_inspired"]
    dp_noise = gradient_info["dp_inspired"]
    dp_realized = [x["noise_realized_stddev"] for x in dp_noise["noise_epochs"]]
    noise_applied = all(x["noise_samples"] > 0 and x["noise_abs_max"] > 0 for x in dp_noise["noise_epochs"])
    scale_ok = (dp["gradient_noise_stddev"] == 0.002 and dp["historical_epsilon_parameter"] == 0.5
                and dp["noise_scale_formula"] == "0.001 / epsilon"
                and all(0.75 * 0.002 < x < 1.25 * 0.002 for x in dp_realized))
    record("DP-inspired Gaussian noise sampled at recorded scale and added during training",
           noise_applied and scale_ok and dp_noise["noise_stddev_configured"] == 0.002,
           f"configured sigma=0.002; epoch realized sigmas={dp_realized}")
    record("Face-only adversarial training scope", config["defenses"]["face_adversarial"]["epsilon"] == 0.05
           and config["defenses"]["face_adversarial"]["fingerprint_optimization_claim"] is False,
           "Face FGSM only; paired fingerprint rows/labels duplicated unchanged")
    combined = config["defenses"]["combined"]
    combined_evidence = gradient_info["combined"]
    branch_history = [r for r in histories["combined"] if r["stage"] == "defense_finetuning"]
    branch_grad_ok = (all(r["face_gradient_norm_mean"] > 0 and r["fingerprint_gradient_norm_mean"] > 0
                          and r["parameter_update_l2_epoch"] > 0 for r in branch_history)
                      and combined_evidence["face_gradient_norm_mean"] > 0
                      and combined_evidence["fingerprint_gradient_norm_mean"] > 0
                      and combined_evidence["parameter_update_l2_total"] > 0)
    combined_noise = combined_evidence["noise_epochs"]
    record("Combined objective trains both branches with nonzero gradients and updates",
           combined.get("loss_weights") == {"face": 0.5, "fingerprint": 0.5}
           and branch_grad_ok and len(combined_noise) == 12
           and all(x["noise_samples"] > 0 and x["noise_realized_stddev"] > 0 for x in combined_noise),
           f"face gradient L2={combined_evidence['face_gradient_norm_mean']:.6g}; fingerprint gradient L2={combined_evidence['fingerprint_gradient_norm_mean']:.6g}")
    record("No formal differential-privacy claim", dp.get("gradient_clipping") is False
           and dp.get("privacy_accountant") is False and dp.get("formal_dp_claim") is False,
           "No epsilon/delta guarantee is reported")
    baseline_hash_after = sha256(baseline_path)
    record("Phase 2 original checkpoint remains unchanged", baseline_hash_before == baseline_hash_after
           == config["source_phase2_checkpoint_sha256"], baseline_hash_after)
    failures = [x["check"] for x in checks if x["status"] != "PASS"]
    return {"run_id": run_dir.name, "status": "PASS" if not failures else "FAIL",
            "checks": checks, "failed_checks": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

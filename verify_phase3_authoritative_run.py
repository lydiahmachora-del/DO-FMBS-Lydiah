"""Independent numerical and provenance verification for a persisted Phase 3 run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_DOC_SHA256 = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"
EPSILONS = (0.01, 0.05, 0.10, 0.20, 0.30)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_hash_manifest(directory: Path, filename: str) -> tuple[dict, list[str]]:
    path = directory / filename
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest.get("artifacts", []):
        target = directory / item["path"]
        if not target.is_file() or target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            failures.append(item["path"])
    return manifest, failures


def _model_outputs(model, face, finger):
    result = model({"face_input": np.asarray(face, np.float32),
                    "finger_input": np.asarray(finger, np.float32)}, training=False)
    if isinstance(result, dict):
        return np.asarray(result["face_output"]), np.asarray(result["finger_output"])
    return np.asarray(result[0]), np.asarray(result[1])


def verify(run_dir: Path) -> dict:
    import tensorflow as tf

    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "phase3_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "phase3_metrics.json").read_text(encoding="utf-8"))
    p3_manifest, p3_bad = verify_hash_manifest(run_dir, "phase3_manifest.json")
    checks = []

    def record(name: str, passed: bool, evidence: str):
        checks.append({"check": name, "status": "PASS" if bool(passed) else "FAIL", "evidence": evidence})

    p1_dir = run_dir.parent / config["source_phase1_run"]
    p2_dir = run_dir.parent / config["source_phase2_run"]
    p1_manifest, p1_bad = verify_hash_manifest(p1_dir, "phase1_manifest.json")
    p2_manifest, p2_bad = verify_hash_manifest(p2_dir, "phase2_manifest.json")
    record("Phase 3 artifact hashes", not p3_bad, f"{len(p3_manifest.get('artifacts', []))} listed; mismatches={p3_bad}")
    identity_ok = (run_dir.name == config["run_id"] == metrics["run_id"] == p3_manifest["run_id"]
                   and config["phase"] == 3 and config["source_document_sha256"] == SOURCE_DOC_SHA256
                   and p3_manifest["source_document_sha256"] == SOURCE_DOC_SHA256)
    record("Run identity and authoritative Chapter Four source", identity_ok, run_dir.name)
    record("Phase 1 source manifest", not p1_bad and sha256(p1_dir / "phase1_manifest.json") == config["source_phase1_manifest_sha256"],
           f"{p1_manifest.get('replay_id')}; mismatches={p1_bad}")
    record("Phase 2 source manifest", not p2_bad and sha256(p2_dir / "phase2_manifest.json") == config["source_phase2_manifest_sha256"],
           f"{p2_manifest.get('run_id')}; mismatches={p2_bad}")

    source_bad = []
    for name, expected in config.get("source_sha256", {}).items():
        path = run_dir / "source_snapshot" / name
        if not path.is_file() or sha256(path) != expected:
            source_bad.append(name)
    record("Phase 3 source snapshot hashes", bool(config.get("source_sha256")) and not source_bad,
           f"{len(config.get('source_sha256', {}))} source files; mismatches={source_bad}")

    model_path = p2_dir / "final_global_model.keras"
    baseline_item = next((x for x in p2_manifest["artifacts"] if x["path"] == "final_global_model.keras"), None)
    baseline_before = sha256(model_path) if model_path.is_file() else "missing"
    baseline_ok = baseline_item is not None and baseline_before == baseline_item["sha256"] == config["source_phase2_model_sha256"]
    record("Authoritative Phase 2 model identity", baseline_ok, baseline_before)

    with np.load(p1_dir / "face_data.npz", allow_pickle=False) as face, \
            np.load(p1_dir / "fingerprint_data.npz", allow_pickle=False) as finger:
        face_train, face_train_y = face["X_train"], face["y_train"]
        face_test, face_test_y, face_test_ids = face["X_test"], face["y_test"], face["test_indices"]
        face_train_ids = face["train_indices"]
        finger_train, finger_train_y = finger["X_train"], finger["y_train"]
        finger_test, finger_test_y = finger["X_test"], finger["y_test"]

    tf.keras.utils.set_random_seed(42)
    baseline = tf.keras.models.load_model(model_path, compile=False)
    evasion_checks = []
    with np.load(run_dir / "evasion_evidence.npz", allow_pickle=False) as e:
        clean_face = e["face_clean"]
        finger_pair = e["fingerprint_unchanged"]
        labels = e["labels"]
        clean_pred_saved = e["clean_predictions"]
        clean_prob, _ = _model_outputs(baseline, clean_face, finger_pair)
        clean_pred_actual = np.argmax(clean_prob, axis=1)
        clean_mask = clean_pred_actual == labels
        clean_subset = np.flatnonzero(clean_mask)
        row_identity_ok = (len(labels) == 100 and np.array_equal(e["evaluation_ids"], face_test_ids[:100])
                           and np.array_equal(clean_face, face_test[:100])
                           and np.array_equal(finger_pair, finger_test[:100])
                           and np.array_equal(labels, face_test_y[:100]))
        prediction_ok = np.array_equal(clean_pred_saved, clean_pred_actual)
        evasion_checks.append(row_identity_ok and prediction_ok)
        record("Evasion source rows and clean predictions", row_identity_ok and prediction_ok,
               f"100 paired Phase 1 test positions; recalculated clean-correct={len(clean_subset)}")
        record("Clean-correct conditional denominator", np.array_equal(e["clean_correct_mask"], clean_mask)
               and np.array_equal(e["clean_correct_indices"], clean_subset)
               and metrics["clean_correct_face_denominator"] == int(clean_mask.sum()),
               f"{int(clean_mask.sum())}/100 clean-correct")
        expected_eps = list(EPSILONS)
        cfg_eps = config["attacks"]["fgsm"]["epsilons"]
        record("Declared epsilon sweep", cfg_eps == expected_eps and config["attacks"]["pgd"]["epsilons"] == expected_eps,
               str(cfg_eps))
        record("PGD fixed-step protocol", config["attacks"]["pgd"]["iterations"] == 10
               and config["attacks"]["pgd"]["step_size"] == 0.01
               and config["attacks"]["pgd"]["initialization"] == "clean input; no random start"
               and config["attacks"]["pgd"]["max_traversal_linf"] == 0.10,
               "10 x 0.01; no random start; maximum traversal 0.10")
        for family in ("FGSM", "PGD"):
            for epsilon in EPSILONS:
                prefix = f"{family.lower()}_eps_{epsilon:.2f}"
                adv = e[f"{prefix}_adversarial_face"]
                saved_pred = e[f"{prefix}_predictions"]
                saved_success = e[f"{prefix}_success_mask"]
                delta = adv - clean_face
                norms = np.max(np.abs(delta), axis=1)
                allowed = epsilon if family == "FGSM" else min(epsilon, 0.10)
                bound_ok = bool(np.all(norms <= allowed + 1e-6))
                attacked_prob, _ = _model_outputs(baseline, adv, finger_pair)
                actual_pred = np.argmax(attacked_prob, axis=1)
                actual_success = actual_pred[clean_subset] != labels[clean_subset]
                row = next(r for r in metrics["evasion"] if r["attack"] == family and
                           np.isclose(r["epsilon_projection_bound"], epsilon))
                recomputed_asr = float(np.mean(actual_success))
                passed = (bound_ok and np.array_equal(saved_pred, actual_pred)
                          and np.array_equal(saved_success, actual_success)
                          and row["clean_correct_denominator"] == len(clean_subset)
                          and row["attack_successes"] == int(actual_success.sum())
                          and np.isclose(row["attack_success_rate"], recomputed_asr)
                          and np.isclose(row["accuracy_under_attack_all_100"], np.mean(actual_pred == labels))
                          and np.isclose(row["accuracy_under_attack_on_conditional_subset"],
                                         np.mean(actual_pred[clean_subset] == labels[clean_subset]))
                          and np.isclose(row["perturbation_linf_max"], np.max(norms, initial=0.0), atol=1e-6)
                          and row["projection_bound_satisfied"])
                evasion_checks.append(passed)
        record("Evasion predictions, ASR, and perturbation projections", all(evasion_checks[1:]),
               "10 attack configurations independently re-predicted and recalculated")
        record("Paired fingerprint inputs unchanged", row_identity_ok
               and np.array_equal(finger_pair, finger_test[:100]),
               "All evasion model calls use the exact persisted Phase 1 fingerprint test rows, unchanged")

    with np.load(run_dir / "membership_evidence.npz", allow_pickle=False) as m:
        member_rows, nonmember_rows = m["member_phase1_rows"], m["nonmember_phase1_rows"]
        member_ids, nonmember_ids = m["member_record_ids"], m["nonmember_record_ids"]
        member_confidence = np.max(_model_outputs(baseline, face_train[:640][member_rows],
                                                  finger_train[:640][member_rows])[0], axis=1)
        nonmember_confidence = np.max(_model_outputs(baseline, face_test[nonmember_rows],
                                                     finger_test[nonmember_rows])[0], axis=1)
        confidence = np.concatenate([member_confidence, nonmember_confidence])
        truth = np.concatenate([np.ones(100, bool), np.zeros(100, bool)])
        decisions = confidence >= 0.70
        correct = decisions == truth
        tp, fp = int(decisions[:100].sum()), int(decisions[100:].sum())
        mia = metrics["membership_inference"]
        mia_ok = (len(np.unique(member_ids)) == len(np.unique(nonmember_ids)) == 100
                  and not set(member_ids.tolist()).intersection(nonmember_ids.tolist())
                  and np.array_equal(member_ids, face_train_ids[:640][member_rows])
                  and np.array_equal(nonmember_ids, face_test_ids[nonmember_rows])
                  and np.array_equal(m["true_membership"], truth)
                  and np.allclose(m["confidence"], confidence, rtol=0, atol=1e-7)
                  and np.array_equal(m["attack_decisions"], decisions)
                  and np.array_equal(m["correct_decisions"], correct)
                  and mia["threshold"] == 0.70 and mia["members"] == mia["nonmembers"] == 100
                  and mia["true_member_detections"] == tp and mia["false_member_detections"] == fp
                  and mia["false_negatives"] == 100 - tp and mia["true_negatives"] == 100 - fp
                  and np.isclose(mia["attack_accuracy"], np.mean(correct))
                  and np.isclose(mia["privacy_quantity_PA"], 2 * max(0, np.mean(correct) - 0.5)))
        record("Membership threshold, counts, and privacy quantity", mia_ok,
               f"threshold=0.70; members={tp}/100 detected; nonmembers={fp}/100 falsely detected; accuracy={np.mean(correct):.4f}")

    with np.load(run_dir / "poisoning_evidence.npz", allow_pickle=False) as p:
        selected = p["selected_phase1_train_positions"]
        original = face_train_y[:640][selected]
        poisoned_labels = p["poisoned_labels"]
        flip = p["flip_mask"]
        selected_ids = p["selected_face_record_ids"]
        exact_eval = (np.array_equal(p["face_test_record_ids"], face_test_ids[:100])
                      and np.array_equal(p["clean_test_labels"], face_test_y[:100]))
        unique_selection = len(selected) == 200 and len(np.unique(selected)) == 200
        label_values_ok = (np.array_equal(p["original_labels"], original)
                           and np.array_equal(selected_ids, face_train_ids[:640][selected])
                           and int(flip.sum()) == 20 and np.array_equal(poisoned_labels[~flip], original[~flip])
                           and np.array_equal(poisoned_labels[flip], (original[flip] + 1) % 62))
        clean_clone = tf.keras.models.load_model(run_dir / "clean_baseline_clone.keras", compile=False)
        poison_model = tf.keras.models.load_model(run_dir / "poisoned_face_label_clone.keras", compile=False)
        clean_prob, _ = _model_outputs(clean_clone, face_test[:100], finger_test[:100])
        poison_prob, _ = _model_outputs(poison_model, face_test[:100], finger_test[:100])
        clean_pred, poison_pred = np.argmax(clean_prob, axis=1), np.argmax(poison_prob, axis=1)
        clean_acc = float(np.mean(clean_pred == face_test_y[:100]))
        poison_acc = float(np.mean(poison_pred == face_test_y[:100]))
        poison_metrics = metrics["label_flipping"]
        persisted_ok = (np.array_equal(p["clean_predictions"], clean_pred)
                        and np.array_equal(p["poisoned_predictions"], poison_pred)
                        and np.array_equal(p["clean_correct_mask"], clean_pred == face_test_y[:100])
                        and np.array_equal(p["poisoned_correct_mask"], poison_pred == face_test_y[:100]))
        poison_metrics_ok = (np.isclose(poison_metrics["clean_accuracy"], clean_acc)
                             and np.isclose(poison_metrics["poisoned_accuracy"], poison_acc)
                             and np.isclose(poison_metrics["accuracy_change_percentage_points"], 100 * (poison_acc - clean_acc))
                             and np.isclose(poison_metrics["clamped_degradation_poisoning_asr"], max(0, clean_acc - poison_acc)))
        record("Label flipping selection and exact changed labels", unique_selection and exact_eval and label_values_ok,
               f"{len(selected)} selected; {int(flip.sum())}/200 labels changed; face-only labels")
        record("Poison checkpoints and matched evaluation metrics", persisted_ok and poison_metrics_ok,
               f"clean={clean_acc:.4f}; poisoned={poison_acc:.4f}; change={100*(poison_acc-clean_acc):+.2f} pp")
        record("Poisoning scope and epochs", poison_metrics["epochs"] == 3 and poison_metrics["poison_rate"] == 0.10
               and poison_metrics["flipped_labels"] == 20 and config["attacks"]["label_flipping"]["target"] == "face labels only",
               "3 epochs; 10%; 20/200; face labels only")

    record("Phase 1 and Phase 2 input references", config["data_references"]["cross_modal_identity_correspondence"] is False
           and config["source_phase1_manifest_sha256"] == sha256(p1_dir / "phase1_manifest.json")
           and config["source_phase2_model_sha256"] == sha256(model_path),
           "Accepted data and model hashes match the run configuration")
    baseline_after = sha256(model_path)
    record("Phase 2 model remains unchanged", baseline_before == baseline_after == config["source_phase2_model_sha256"], baseline_after)
    failures = [c["check"] for c in checks if c["status"] != "PASS"]
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

"""Phase 3: scoped attacks against the trained DO-FMBS baseline.

FGSM and PGD perturb only the face branch.  This phase does not claim evidence
of fingerprint adversarial robustness.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from do_fmbs_core import compile_dual_branch_model, json_dump, modality_probabilities, set_global_determinism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("do_fmbs_data"))
    parser.add_argument("--epsilons", default="0.01,0.05,0.10,0.20,0.30")
    parser.add_argument("--pgd-iterations", type=int, default=10)
    parser.add_argument("--poison-rate", type=float, default=0.10)
    parser.add_argument("--poison-epochs", type=int, default=3)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def output_tensor(outputs, name: str, index: int):
    return outputs[name] if isinstance(outputs, dict) else outputs[index]


def fgsm_face(model, face, finger, labels, epsilon: float):
    import tensorflow as tf

    face_tensor = tf.convert_to_tensor(face, tf.float32)
    finger_tensor = tf.convert_to_tensor(finger, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(face_tensor)
        outputs = model({"face_input": face_tensor, "finger_input": finger_tensor}, training=False)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            labels, output_tensor(outputs, "face_output", 0)
        ))
    gradient = tape.gradient(loss, face_tensor)
    return tf.clip_by_value(face_tensor + epsilon * tf.sign(gradient), -6.0, 6.0)


def pgd_face(model, face, finger, labels, epsilon: float, iterations: int):
    import tensorflow as tf

    original = tf.convert_to_tensor(face, tf.float32)
    adversarial = tf.identity(original)
    alpha = epsilon / max(iterations // 2, 1)
    finger_tensor = tf.convert_to_tensor(finger, tf.float32)
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial)
            outputs = model({"face_input": adversarial, "finger_input": finger_tensor}, training=False)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                labels, output_tensor(outputs, "face_output", 0)
            ))
        gradient = tape.gradient(loss, adversarial)
        adversarial = adversarial + alpha * tf.sign(gradient)
        adversarial = tf.clip_by_value(adversarial, original - epsilon, original + epsilon)
        adversarial = tf.clip_by_value(adversarial, -6.0, 6.0)
    return adversarial


def face_probabilities(model, face, finger) -> np.ndarray:
    outputs = model({"face_input": face, "finger_input": finger}, training=False)
    return np.asarray(output_tensor(outputs, "face_output", 0))


def evaluate_evasion(model, face, finger, labels, epsilons, pgd_iterations):
    clean_prob = face_probabilities(model, face, finger)
    clean_pred = np.argmax(clean_prob, axis=1)
    eligible = clean_pred == labels
    rows = []
    for attack in ("FGSM", "PGD"):
        for epsilon in epsilons:
            if attack == "FGSM":
                adversarial = fgsm_face(model, face, finger, labels, epsilon)
            else:
                adversarial = pgd_face(model, face, finger, labels, epsilon, pgd_iterations)
            attacked_pred = np.argmax(face_probabilities(model, adversarial, finger), axis=1)
            successes = int(np.count_nonzero(eligible & (attacked_pred != labels)))
            rows.append({
                "attack": attack, "modality": "face", "epsilon": float(epsilon),
                "clean_correct": int(np.count_nonzero(eligible)),
                "successful_attacks": successes,
                "attack_success_rate": float(successes / max(np.count_nonzero(eligible), 1)),
                "attacked_accuracy_all_samples": float(np.mean(attacked_pred == labels)),
            })
    return rows


def membership_inference_face(model, train_face, calibration_face) -> dict[str, float | int | str]:
    train_conf = np.max(modality_probabilities(model, "face", train_face), axis=1)
    nonmember_conf = np.max(modality_probabilities(model, "face", calibration_face), axis=1)
    n = min(len(train_conf), len(nonmember_conf))
    rng = np.random.default_rng(42)
    train_conf = rng.choice(train_conf, n, replace=False)
    nonmember_conf = rng.choice(nonmember_conf, n, replace=False)
    truth = np.concatenate([np.ones(n), np.zeros(n)])
    score = np.concatenate([train_conf, nonmember_conf])
    candidates = np.unique(score)
    accuracies = [np.mean((score >= threshold) == truth) for threshold in candidates]
    index = int(np.argmax(accuracies))
    return {
        "attack": "confidence-threshold membership inference", "modality": "face",
        "members": n, "nonmembers": n, "selected_threshold": float(candidates[index]),
        "attack_accuracy": float(accuracies[index]), "ROC_AUC": float(roc_auc_score(truth, score)),
        "membership_advantage": float(max(0.0, 2.0 * accuracies[index] - 1.0)),
        "threshold_selection_scope": "descriptive attack operating point on the Phase 3 attack set",
    }


def poisoning_attack(model, face_x, face_y, finger_x, finger_y, face_test, face_test_y,
                     finger_test, finger_test_y, rate: float, epochs: int) -> dict[str, object]:
    import tensorflow as tf

    poisoned = compile_dual_branch_model(tf.keras.models.clone_model(model), learning_rate=5e-4)
    poisoned.set_weights(model.get_weights())
    n = max(len(face_y), len(finger_y))
    face_order = np.resize(np.arange(len(face_y)), n)
    finger_order = np.resize(np.arange(len(finger_y)), n)
    py_face = face_y[face_order].copy()
    py_finger = finger_y[finger_order].copy()
    rng = np.random.default_rng(3301)
    count = max(1, int(rate * n))
    selected = rng.choice(n, count, replace=False)
    py_face[selected] = (py_face[selected] + 1) % int(model.get_layer("face_output").units)
    py_finger[selected] = (py_finger[selected] + 1) % int(model.get_layer("finger_output").units)
    poisoned.fit(
        {"face_input": face_x[face_order], "finger_input": finger_x[finger_order]},
        {"face_output": py_face, "finger_output": py_finger},
        epochs=epochs, batch_size=64, verbose=0,
    )
    before_face = np.mean(np.argmax(modality_probabilities(model, "face", face_test), axis=1) == face_test_y)
    after_face = np.mean(np.argmax(modality_probabilities(poisoned, "face", face_test), axis=1) == face_test_y)
    before_finger = np.mean(np.argmax(modality_probabilities(model, "fingerprint", finger_test), axis=1) == finger_test_y)
    after_finger = np.mean(np.argmax(modality_probabilities(poisoned, "fingerprint", finger_test), axis=1) == finger_test_y)
    return {
        "attack": "central label-flipping poisoning simulation", "modality": "face and fingerprint labels",
        "poison_rate": rate, "poisoned_synchronized_records": count, "epochs": epochs,
        "face_accuracy_before": float(before_face), "face_accuracy_after": float(after_face),
        "face_accuracy_drop": float(before_face - after_face),
        "fingerprint_accuracy_before": float(before_finger), "fingerprint_accuracy_after": float(after_finger),
        "fingerprint_accuracy_drop": float(before_finger - after_finger),
    }


def main() -> int:
    args = parse_args()
    set_global_determinism()
    started = time.perf_counter()
    epsilons = sorted({float(value) for value in args.epsilons.split(",")})
    import tensorflow as tf

    with np.load(args.data_dir / "face_data.npz", allow_pickle=False) as face:
        face_train, face_train_y = face["X_train"], face["y_train"]
        face_cal = face["X_calibration"]
        face_test, face_test_y = face["X_test"], face["y_test"]
    with np.load(args.data_dir / "fingerprint_data.npz", allow_pickle=False) as finger:
        finger_train, finger_train_y = finger["X_train"], finger["y_train"]
        finger_test, finger_test_y = finger["X_test"], finger["y_test"]
    model = tf.keras.models.load_model(args.data_dir / "baseline_model.keras")
    limit = len(face_test) if args.max_test_samples <= 0 else min(args.max_test_samples, len(face_test))
    face_attack = face_test[:limit].astype(np.float32)
    face_attack_y = face_test_y[:limit]
    finger_dim = int(model.get_layer("finger_input").input.shape[-1])
    neutral_finger = np.zeros((limit, finger_dim), dtype=np.float32)

    evasion = evaluate_evasion(
        model, face_attack, neutral_finger, face_attack_y, epsilons, args.pgd_iterations
    )
    membership = membership_inference_face(model, face_train, face_cal)
    poisoning = poisoning_attack(
        model, face_train, face_train_y, finger_train, finger_train_y,
        face_test, face_test_y, finger_test, finger_test_y, args.poison_rate, args.poison_epochs,
    )
    results = {
        "phase": 3, "replay_id": os.environ.get("DO_FMBS_REPLAY_ID", "unspecified"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope_statement": "FGSM and PGD perturb the face input only; no fingerprint evasion-robustness claim is made.",
        "evasion": evasion, "membership_inference": membership, "label_flipping_poisoning": poisoning,
        "parameters": {"epsilons": epsilons, "pgd_iterations": args.pgd_iterations,
                       "poison_rate": args.poison_rate, "poison_epochs": args.poison_epochs,
                       "face_attack_samples": limit},
        "runtime_seconds": time.perf_counter() - started,
    }
    json_dump(args.data_dir / "phase3_attack_results.json", results)
    with (args.data_dir / "attack_results.pkl").open("wb") as stream:
        pickle.dump(results, stream)

    figures = args.data_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    for attack, marker in (("FGSM", "o"), ("PGD", "s")):
        rows = [row for row in evasion if row["attack"] == attack]
        ax.plot([row["epsilon"] for row in rows], [100 * row["attack_success_rate"] for row in rows],
                marker=marker, label=f"{attack}—face")
    ax.set(xlabel="Perturbation bound (epsilon)", ylabel="Attack success rate (%)",
           title="Face-Branch Adversarial Evasion")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(figures / "face_adversarial_attacks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Phase 3 complete in {results['runtime_seconds']:.2f}s; attacks explicitly scoped by modality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

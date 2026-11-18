"""Phase 4: honest DP-inspired and face-adversarial defense training."""
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
from sklearn.metrics import accuracy_score, log_loss

from do_fmbs_core import compile_dual_branch_model, json_dump, modality_probabilities, set_global_determinism


FACE_LOSS_WEIGHT = 0.5
FINGERPRINT_LOSS_WEIGHT = 0.5
GAUSSIAN_NOISE_STDDEV = 0.0003
ADVERSARIAL_EPSILON = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("do_fmbs_data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-stddev", type=float, default=GAUSSIAN_NOISE_STDDEV)
    parser.add_argument("--adversarial-epsilon", type=float, default=ADVERSARIAL_EPSILON)
    return parser.parse_args()


def output_tensor(outputs, name: str, index: int):
    return outputs[name] if isinstance(outputs, dict) else outputs[index]


def compute_combined_loss(face_labels, fingerprint_labels, outputs,
                          face_weight: float = FACE_LOSS_WEIGHT,
                          fingerprint_weight: float = FINGERPRINT_LOSS_WEIGHT):
    """Return the declared equal-weight dual-modality objective."""
    import tensorflow as tf

    face_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        face_labels, output_tensor(outputs, "face_output", 0)
    ))
    fingerprint_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        fingerprint_labels, output_tensor(outputs, "finger_output", 1)
    ))
    total = face_weight * face_loss + fingerprint_weight * fingerprint_loss
    return total, face_loss, fingerprint_loss


def face_fgsm(model, face, finger, labels, epsilon: float):
    import tensorflow as tf

    face = tf.convert_to_tensor(face, tf.float32)
    finger = tf.convert_to_tensor(finger, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(face)
        outputs = model({"face_input": face, "finger_input": finger}, training=False)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            labels, output_tensor(outputs, "face_output", 0)
        ))
    gradient = tape.gradient(loss, face)
    return tf.clip_by_value(face + epsilon * tf.sign(gradient), -6.0, 6.0)


def synchronized_epoch_arrays(face_x, face_y, finger_x, finger_y, seed: int):
    rng = np.random.default_rng(seed)
    n = max(len(face_y), len(finger_y))
    face_order = np.resize(rng.permutation(len(face_y)), n)
    finger_order = np.resize(rng.permutation(len(finger_y)), n)
    return face_x[face_order], face_y[face_order], finger_x[finger_order], finger_y[finger_order]


def fingerprint_gradient_norms(model, face_x, face_y, finger_x, finger_y) -> dict[str, float]:
    import tensorflow as tf

    with tf.GradientTape() as tape:
        outputs = model({"face_input": face_x, "finger_input": finger_x}, training=True)
        total, _, _ = compute_combined_loss(face_y, finger_y, outputs)
    gradients = tape.gradient(total, model.trainable_variables)
    result = {}
    for variable, gradient in zip(model.trainable_variables, gradients):
        if "finger_" in variable.name:
            result[variable.name] = 0.0 if gradient is None else float(tf.linalg.global_norm([gradient]).numpy())
    return result


def fine_tune(model, face_x, face_y, finger_x, finger_y, *, epochs: int,
              batch_size: int, adversarial: bool, epsilon: float,
              noise_stddev: float) -> list[dict[str, float]]:
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    history = []
    for epoch in range(epochs):
        ex, ey, px, py = synchronized_epoch_arrays(
            face_x, face_y, finger_x, finger_y, 4400 + epoch
        )
        losses, face_losses, finger_losses = [], [], []
        for start in range(0, len(ey), batch_size):
            stop = min(start + batch_size, len(ey))
            bf, byf, bp, byp = ex[start:stop], ey[start:stop], px[start:stop], py[start:stop]
            if adversarial:
                attacked = face_fgsm(model, bf, bp, byf, epsilon).numpy()
                bf = np.concatenate([bf, attacked])
                bp = np.concatenate([bp, bp])
                byf = np.concatenate([byf, byf])
                byp = np.concatenate([byp, byp])
            with tf.GradientTape() as tape:
                outputs = model({"face_input": bf, "finger_input": bp}, training=True)
                loss, face_loss, finger_loss = compute_combined_loss(byf, byp, outputs)
            gradients = tape.gradient(loss, model.trainable_variables)
            if noise_stddev > 0:
                gradients = [
                    None if gradient is None else gradient + tf.random.normal(
                        tf.shape(gradient), stddev=noise_stddev, dtype=gradient.dtype
                    )
                    for gradient in gradients
                ]
            optimizer.apply_gradients(
                (gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables)
                if gradient is not None
            )
            losses.append(float(loss.numpy()))
            face_losses.append(float(face_loss.numpy()))
            finger_losses.append(float(finger_loss.numpy()))
        history.append({"epoch": epoch + 1, "total_loss": float(np.mean(losses)),
                        "face_loss": float(np.mean(face_losses)),
                        "fingerprint_loss": float(np.mean(finger_losses))})
        print(f"  epoch {epoch + 1:02d}/{epochs}: total={history[-1]['total_loss']:.4f}, "
              f"face={history[-1]['face_loss']:.4f}, fingerprint={history[-1]['fingerprint_loss']:.4f}")
    return history


def evaluate_model(model, face_x, face_y, finger_x, finger_y, epsilon: float) -> dict[str, object]:
    face_prob = modality_probabilities(model, "face", face_x)
    finger_prob = modality_probabilities(model, "fingerprint", finger_x)
    face_pred = np.argmax(face_prob, axis=1)
    finger_pred = np.argmax(finger_prob, axis=1)
    neutral_finger = np.zeros((len(face_x), finger_x.shape[1]), dtype=np.float32)
    adversarial = face_fgsm(model, face_x, neutral_finger, face_y, epsilon)
    attacked_pred = np.argmax(np.asarray(output_tensor(model(
        {"face_input": adversarial, "finger_input": neutral_finger}, training=False
    ), "face_output", 0)), axis=1)
    eligible = face_pred == face_y
    successful = int(np.count_nonzero(eligible & (attacked_pred != face_y)))
    return {
        "face_classification": {"accuracy": float(accuracy_score(face_y, face_pred)),
                                "loss": float(log_loss(face_y, face_prob))},
        "fingerprint_classification": {"accuracy": float(accuracy_score(finger_y, finger_pred)),
                                       "loss": float(log_loss(finger_y, finger_prob))},
        "face_fgsm": {"epsilon": epsilon, "eligible_clean_correct": int(np.count_nonzero(eligible)),
                      "successful_attacks": successful,
                      "attack_success_rate": float(successful / max(np.count_nonzero(eligible), 1)),
                      "attacked_accuracy_all_samples": float(np.mean(attacked_pred == face_y))},
    }


def clone_from(model):
    import tensorflow as tf

    clone = tf.keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    return clone


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.noise_stddev < 0:
        raise ValueError("Invalid defense-training parameters")
    set_global_determinism()
    started = time.perf_counter()
    import tensorflow as tf

    with np.load(args.data_dir / "face_data.npz", allow_pickle=False) as face:
        face_train, face_train_y = face["X_train"].astype(np.float32), face["y_train"]
        face_test, face_test_y = face["X_test"].astype(np.float32), face["y_test"]
    with np.load(args.data_dir / "fingerprint_data.npz", allow_pickle=False) as finger:
        finger_train, finger_train_y = finger["X_train"].astype(np.float32), finger["y_train"]
        finger_test, finger_test_y = finger["X_test"].astype(np.float32), finger["y_test"]
    baseline = tf.keras.models.load_model(args.data_dir / "baseline_model.keras")

    models = {"baseline": baseline}
    histories = {}
    configurations = {
        "dp_inspired": {"adversarial": False, "noise_stddev": args.noise_stddev},
        "adversarial_training": {"adversarial": True, "noise_stddev": 0.0},
        "do_fmbs_combined": {"adversarial": True, "noise_stddev": args.noise_stddev},
    }
    for name, configuration in configurations.items():
        print(f"Training {name}")
        model = clone_from(baseline)
        histories[name] = fine_tune(
            model, face_train, face_train_y, finger_train, finger_train_y,
            epochs=args.epochs, batch_size=args.batch_size,
            adversarial=configuration["adversarial"], epsilon=args.adversarial_epsilon,
            noise_stddev=configuration["noise_stddev"],
        )
        models[name] = model

    gradient_face, gradient_face_y, gradient_finger, gradient_finger_y = synchronized_epoch_arrays(
        face_train, face_train_y, finger_train, finger_train_y, 9981
    )
    gradient_norms = fingerprint_gradient_norms(
        models["do_fmbs_combined"], gradient_face[:args.batch_size], gradient_face_y[:args.batch_size],
        gradient_finger[:args.batch_size], gradient_finger_y[:args.batch_size],
    )
    if not gradient_norms or not all(value > 0 for value in gradient_norms.values()):
        raise RuntimeError(f"Fingerprint branch did not receive expected gradients: {gradient_norms}")

    comparison = {
        name: evaluate_model(model, face_test, face_test_y, finger_test, finger_test_y,
                             args.adversarial_epsilon)
        for name, model in models.items()
    }
    models["do_fmbs_combined"].save(args.data_dir / "do_fmbs_model.keras")
    results = {
        "phase": 4, "replay_id": os.environ.get("DO_FMBS_REPLAY_ID", "unspecified"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "defense_scope": "adversarial examples perturb face inputs only; both modality losses are optimized",
        "privacy_terminology": "DP-inspired Gaussian gradient-noise defense; no formal (epsilon, delta)-DP guarantee",
        "combined_objective": {"face_loss_weight": FACE_LOSS_WEIGHT,
                               "fingerprint_loss_weight": FINGERPRINT_LOSS_WEIGHT},
        "parameters": {"epochs": args.epochs, "batch_size": args.batch_size,
                       "gaussian_gradient_noise_stddev": args.noise_stddev,
                       "face_adversarial_epsilon": args.adversarial_epsilon,
                       "per_example_clipping": False, "privacy_accountant": False,
                       "epsilon_delta_reported": False},
        "fingerprint_gradient_norms": gradient_norms,
        "comparison": comparison, "training_history": histories,
        "runtime_seconds": time.perf_counter() - started,
    }
    json_dump(args.data_dir / "phase4_defense_results.json", results)
    with (args.data_dir / "final_results.pkl").open("wb") as stream:
        pickle.dump(results, stream)

    figures = args.data_dir / "figures"
    figures.mkdir(exist_ok=True)
    labels = list(comparison)
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - width, [100 * comparison[n]["face_classification"]["accuracy"] for n in labels], width, label="Face accuracy")
    ax.bar(x, [100 * comparison[n]["fingerprint_classification"]["accuracy"] for n in labels], width, label="Fingerprint accuracy")
    ax.bar(x + width, [100 * comparison[n]["face_fgsm"]["attack_success_rate"] for n in labels], width, label="Face FGSM ASR")
    ax.set(xticks=x, xticklabels=[name.replace("_", "\n") for name in labels], ylabel="Percent",
           title="Defense Comparison by Modality and Metric")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figures / "defense_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Phase 4 complete in {results['runtime_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

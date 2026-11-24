"""Phase 2: five-client non-IID federated learning for two identity spaces."""
from __future__ import annotations

import argparse
import json
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

from do_fmbs_core import (
    compile_dual_branch_model,
    create_dual_branch_model,
    json_dump,
    modality_probabilities,
    set_global_determinism,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("do_fmbs_data"))
    return parser.parse_args()


def balanced_batch_records(client: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, ...]:
    """Synchronize independent modality records without asserting subject pairing."""
    rng = np.random.default_rng(seed)
    n = max(len(client["face_y"]), len(client["finger_y"]))
    face_order = np.resize(rng.permutation(len(client["face_y"])), n)
    finger_order = np.resize(rng.permutation(len(client["finger_y"])), n)
    return (
        client["face_X"][face_order].astype(np.float32),
        client["face_y"][face_order],
        client["finger_X"][finger_order].astype(np.float32),
        client["finger_y"][finger_order],
    )


def federated_train(model_factory, clients: list[dict[str, np.ndarray]], rounds: int,
                    local_epochs: int, batch_size: int, learning_rate: float = 0.001,
                    seed: int = 42) -> tuple[object, dict[str, list[float]]]:
    global_model = compile_dual_branch_model(model_factory(), learning_rate=learning_rate)
    history = {"round": [], "face_accuracy": [], "fingerprint_accuracy": [],
               "face_loss": [], "fingerprint_loss": [], "total_loss": []}
    total_face = sum(len(client["face_y"]) for client in clients)
    total_finger = sum(len(client["finger_y"]) for client in clients)

    for round_index in range(rounds):
        updates = []
        for client_id, client in enumerate(clients):
            local = compile_dual_branch_model(model_factory(), learning_rate=learning_rate)
            local.set_weights(global_model.get_weights())
            face_x, face_y, finger_x, finger_y = balanced_batch_records(
                100_000 + (seed - 42) + round_index * 100 + client_id
            )
            fitted = local.fit(
                {"face_input": face_x, "finger_input": finger_x},
                {"face_output": face_y, "finger_output": finger_y},
                epochs=local_epochs, batch_size=batch_size, verbose=0, shuffle=False,
            )
            weight = 0.5 * len(client["face_y"]) / total_face + 0.5 * len(client["finger_y"]) / total_finger
            updates.append((weight, local.get_weights(), fitted.history))

        aggregated = [np.zeros_like(value) for value in updates[0][1]]
        for weight, values, _ in updates:
            for index, value in enumerate(values):
                aggregated[index] += weight * value
        global_model.set_weights(aggregated)
        history["round"].append(round_index + 1)
        for metric, source in (
            ("face_accuracy", "face_output_accuracy"),
            ("fingerprint_accuracy", "finger_output_accuracy"),
            ("face_loss", "face_output_loss"),
            ("fingerprint_loss", "finger_output_loss"),
            ("total_loss", "loss"),
        ):
            history[metric].append(float(np.average(
                [record[2][source][-1] for record in updates],
                weights=[record[0] for record in updates],
            )))
        print(
            f"Round {round_index + 1:02d}/{rounds}: "
            f"face_acc={history['face_accuracy'][-1]:.4f}, "
            f"fingerprint_acc={history['fingerprint_accuracy'][-1]:.4f}"
        )
    return global_model, history


def main() -> int:
    args = parse_args()
    if min(args.rounds, args.local_epochs, args.batch_size, args.learning_rate) <= 0:
        raise ValueError("Rounds, local epochs, batch size and learning rate must be positive")
    set_global_determinism(args.seed)
    started = time.perf_counter()
    data_dir = args.data_dir
    with np.load(data_dir / "face_data.npz", allow_pickle=False) as face:
        face_test, face_test_y = face["X_test"], face["y_test"]
        face_features, face_classes = int(face["n_features"][0]), int(face["n_classes"][0])
    with np.load(data_dir / "fingerprint_data.npz", allow_pickle=False) as finger:
        finger_test, finger_test_y = finger["X_test"], finger["y_test"]
        finger_features, finger_classes = int(finger["n_features"][0]), int(finger["n_classes"][0])

    partition = json.loads((data_dir / "client_partition.json").read_text(encoding="utf-8"))
    clients = []
    for row in partition["clients"]:
        with np.load(data_dir / "clients" / f"client_{row['client_id']}.npz", allow_pickle=False) as data:
            clients.append({name: data[name] for name in data.files})
    if len(clients) < 2:
        raise RuntimeError(f"At least two persisted federated clients are required, found {len(clients)}")

    factory = lambda: create_dual_branch_model(
        face_features, finger_features, face_classes, finger_classes
    )
    model, history = federated_train(
        factory, clients, args.rounds, args.local_epochs, args.batch_size,
        args.learning_rate, args.seed,
    )

    face_prob = modality_probabilities(model, "face", face_test)
    finger_prob = modality_probabilities(model, "fingerprint", finger_test)
    face_pred = np.argmax(face_prob, axis=1)
    finger_pred = np.argmax(finger_prob, axis=1)
    metrics = {
        "phase": 2,
        "replay_id": os.environ.get("DO_FMBS_REPLAY_ID", "unspecified"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_scope": "two independent modality-specific identity classifiers; no common-subject fusion",
        "batch_pairing_scope": "face and fingerprint records are synchronized only for computation; labels remain independent",
        "federated_partition": partition["strategy"],
        "clients": len(clients),
        "training_parameters": {"rounds": args.rounds, "local_epochs": args.local_epochs,
                                "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                                "seed": args.seed,
                                "loss_weights": {"face": 0.5, "fingerprint": 0.5}},
        "face_classification": {
            "samples": int(len(face_test_y)), "classes": face_classes,
            "accuracy": float(accuracy_score(face_test_y, face_pred)),
            "loss": float(log_loss(face_test_y, face_prob, labels=np.arange(face_classes))),
        },
        "fingerprint_classification": {
            "samples": int(len(finger_test_y)), "classes": finger_classes,
            "accuracy": float(accuracy_score(finger_test_y, finger_pred)),
            "loss": float(log_loss(finger_test_y, finger_prob, labels=np.arange(finger_classes))),
        },
        "history": history,
        "authoritative_far_frr": "Phase 5 threshold-based verification only",
        "runtime_seconds": time.perf_counter() - started,
    }
    model.save(data_dir / "baseline_model.keras")
    json_dump(data_dir / "phase2_metrics.json", metrics)
    # Backward-compatible serialization contains explicit modality metrics and no
    # ambiguous generic FAR/FRR keys.
    with (data_dir / "baseline_metrics.pkl").open("wb") as stream:
        pickle.dump(metrics, stream)

    figures = data_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(history["round"], history["face_accuracy"], marker="o", label="Face")
    axes[0].plot(history["round"], history["fingerprint_accuracy"], marker="s", label="Fingerprint")
    axes[0].set(xlabel="Federated round", ylabel="Weighted local accuracy", title="Federated Training Accuracy")
    axes[0].legend(frameon=False)
    axes[1].plot(history["round"], history["face_loss"], marker="o", label="Face")
    axes[1].plot(history["round"], history["fingerprint_loss"], marker="s", label="Fingerprint")
    axes[1].set(xlabel="Federated round", ylabel="Cross-entropy loss", title="Modality-Specific Training Loss")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(figures / "federated_training_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Face test accuracy: {metrics['face_classification']['accuracy']:.4f}")
    print(f"Fingerprint test accuracy: {metrics['fingerprint_classification']['accuracy']:.4f}")
    print(f"Phase 2 complete in {metrics['runtime_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

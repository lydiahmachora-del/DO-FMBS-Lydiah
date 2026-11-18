"""Measure PGD resistance before and after the recorded DO-FMBS defense.

This post-run evaluator is deliberately separate from Phase 4.  Phase 4 saved
FGSM comparison metrics only, while this script loads the hash-verified
baseline and final defended models from the isolated replay and evaluates both
with the same untargeted, face-input PGD protocol.  It does not retrain or
alter either model.

The result is written to a local evidence directory so Chapter Four can show a
genuine PGD baseline-versus-defended comparison alongside the original FGSM
comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runtime_recording" / "full_pipeline_20260809"
DEFAULT_OUTPUT_DIR = ROOT / "runtime_metrics_evidence" / "pgd_defense_comparison"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_epsilons(value: str) -> list[float]:
    epsilons = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not epsilons or any(epsilon <= 0 for epsilon in epsilons):
        raise argparse.ArgumentTypeError("--epsilons must contain positive comma-separated values")
    if epsilons != sorted(set(epsilons)):
        raise argparse.ArgumentTypeError("--epsilons must be unique and ascending")
    return epsilons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epsilons", type=parse_epsilons, default=[0.01, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=160)
    return parser.parse_args()


def face_probabilities(model, face, finger):
    outputs = model({"face_input": face, "finger_input": finger}, training=False)
    if isinstance(outputs, dict):
        return outputs["face_output"]
    return outputs[0]


def pgd_examples(tf, model, face, finger, labels, *, epsilon: float, alpha: float, iterations: int):
    """Create untargeted Linf PGD examples for an entire fixed test batch."""
    original = tf.convert_to_tensor(face, dtype=tf.float32)
    adversarial = tf.identity(original)
    finger_tensor = tf.convert_to_tensor(finger, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(labels, dtype=tf.int64)
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial)
            probabilities = face_probabilities(model, adversarial, finger_tensor)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(label_tensor, probabilities)
            )
        gradient = tape.gradient(loss, adversarial)
        adversarial = adversarial + alpha * tf.sign(gradient)
        adversarial = tf.minimum(tf.maximum(adversarial, original - epsilon), original + epsilon)
        adversarial = tf.clip_by_value(adversarial, -3.0, 3.0)
    return adversarial


def evaluate_model(tf, model, name: str, face, finger, labels, *, epsilons: Iterable[float], alpha: float, iterations: int):
    started = time.perf_counter()
    face_tensor = tf.convert_to_tensor(face, dtype=tf.float32)
    finger_tensor = tf.convert_to_tensor(finger, dtype=tf.float32)
    clean_predictions = np.argmax(face_probabilities(model, face_tensor, finger_tensor).numpy(), axis=1)
    clean_correct = clean_predictions == labels
    clean_correct_n = int(np.count_nonzero(clean_correct))
    total_n = int(labels.size)
    rows: list[dict[str, object]] = []
    for epsilon in epsilons:
        attack_started = time.perf_counter()
        adversarial = pgd_examples(
            tf, model, face, finger, labels, epsilon=epsilon, alpha=alpha, iterations=iterations
        )
        attacked_predictions = np.argmax(face_probabilities(model, adversarial, finger_tensor).numpy(), axis=1)
        successes = clean_correct & (attacked_predictions != labels)
        success_n = int(np.count_nonzero(successes))
        asr = success_n / clean_correct_n if clean_correct_n else 0.0
        rows.append({
            "model": name,
            "attack": "PGD-untargeted-face",
            "epsilon": epsilon,
            "alpha": alpha,
            "iterations": iterations,
            "total_test_pairs": total_n,
            "clean_correct_pairs": clean_correct_n,
            "clean_face_accuracy": float(np.mean(clean_correct)),
            "successful_attacks": success_n,
            "conditional_ASR": asr,
            "conditional_robust_accuracy": 1.0 - asr,
            "attack_elapsed_seconds": time.perf_counter() - attack_started,
            "status": "MEASURED",
        })
    return rows, time.perf_counter() - started


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "model", "attack", "epsilon", "alpha", "iterations", "total_test_pairs",
        "clean_correct_pairs", "clean_face_accuracy", "successful_attacks", "conditional_ASR",
        "conditional_robust_accuracy", "attack_elapsed_seconds", "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    colors = {"Federated baseline": "#7F8C8D", "Final DO-FMBS defense": "#4472C4"}
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for name, model_rows in by_model.items():
        model_rows.sort(key=lambda row: float(row["epsilon"]))
        ax.plot(
            [float(row["epsilon"]) for row in model_rows],
            [100.0 * float(row["conditional_ASR"]) for row in model_rows],
            marker="o", linewidth=2, label=name, color=colors.get(name),
        )
    ax.set_xlabel("PGD perturbation magnitude (epsilon)")
    ax.set_ylabel("Conditional PGD attack success rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Post-run PGD Comparison: Baseline versus Final DO-FMBS Defense")
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.alpha <= 0 or args.iterations < 1 or args.max_pairs < 1:
        raise ValueError("--alpha, --iterations and --max-pairs must be positive")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.random.set_seed(42)
    np.random.seed(42)
    run_dir = args.run_dir.resolve()
    data_dir = run_dir / "do_fmbs_data"
    input_paths = {
        "face_data": data_dir / "face_data.npz",
        "fingerprint_data": data_dir / "fingerprint_data.npz",
        "baseline_model": data_dir / "baseline_model.keras",
        "do_fmbs_model": data_dir / "do_fmbs_model.keras",
    }
    for label, path in input_paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing {label}: {path}")

    with np.load(input_paths["face_data"], allow_pickle=True) as face_data:
        face_test = face_data["X_test"].astype(np.float32)
        face_labels = face_data["y_test"].astype(np.int64)
    with np.load(input_paths["fingerprint_data"], allow_pickle=True) as finger_data:
        finger_test = finger_data["X_test"].astype(np.float32)
        finger_labels = finger_data["y_test"].astype(np.int64)
    available = min(len(face_test), len(finger_test), args.max_pairs)
    face_test, face_labels = face_test[:available], face_labels[:available]
    finger_test, finger_labels = finger_test[:available], finger_labels[:available]

    baseline = tf.keras.models.load_model(input_paths["baseline_model"], compile=False)
    defended = tf.keras.models.load_model(input_paths["do_fmbs_model"], compile=False)
    baseline_rows, baseline_elapsed = evaluate_model(
        tf, baseline, "Federated baseline", face_test, finger_test, face_labels,
        epsilons=args.epsilons, alpha=args.alpha, iterations=args.iterations,
    )
    defended_rows, defended_elapsed = evaluate_model(
        tf, defended, "Final DO-FMBS defense", face_test, finger_test, face_labels,
        epsilons=args.epsilons, alpha=args.alpha, iterations=args.iterations,
    )
    rows = baseline_rows + defended_rows

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pgd_defense_comparison.csv"
    plot_path = output_dir / "pgd_defense_comparison.png"
    manifest_path = output_dir / "pgd_defense_comparison_manifest.json"
    write_csv(csv_path, rows)
    write_plot(plot_path, rows)
    manifest = {
        "protocol": "Post-run face-input untargeted PGD comparison; same first 160 paired Phase-1 test rows for both saved models.",
        "scope": "Baseline model from Phase 2 versus final combined DO-FMBS defense model from Phase 4. This is a post-run evaluation; it does not retrain either model or claim a Phase-4 four-arm PGD comparison.",
        "attack": {
            "name": "PGD-untargeted-face",
            "epsilons": args.epsilons,
            "alpha": args.alpha,
            "iterations": args.iterations,
            "clip_range": [-3.0, 3.0],
            "ASR_denominator": "Clean-correct face predictions for each evaluated model.",
        },
        "evaluation": {
            "first_test_pairs": available,
            "face_labels": "Phase-1 face test labels",
            "fingerprint_inputs": "Paired fingerprint test vectors retained as the second model input; attacks perturb face inputs only.",
        },
        "timing_seconds": {
            "baseline": baseline_elapsed,
            "final_do_fmbs": defended_elapsed,
            "total": baseline_elapsed + defended_elapsed,
        },
        "inputs": [
            {"label": label, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for label, path in input_paths.items()
        ],
        "outputs": [],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in (csv_path, plot_path):
        manifest["outputs"].append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"PGD defense comparison completed on {available} paired test rows.")
    for row in rows:
        print(
            f"{row['model']} eps={float(row['epsilon']):.2f}: "
            f"ASR={100 * float(row['conditional_ASR']):.2f}% "
            f"({row['successful_attacks']}/{row['clean_correct_pairs']})"
        )
    print(f"Evidence written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

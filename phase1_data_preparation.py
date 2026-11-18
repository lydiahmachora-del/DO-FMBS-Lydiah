"""Phase 1: leakage-free data preparation and fingerprint quality gating."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people

from do_fmbs_core import (
    CLIENT_PARTITION_SEED,
    FACE_SPLIT_SEED,
    FINGERPRINT_IDENTITY_SEED,
    FINGERPRINT_IMPRESSION_SEED,
    fingerprint_impression_split,
    fingerprint_separability_diagnostics,
    fit_training_scaler,
    generate_fingerprint_dataset,
    json_dump,
    non_iid_identity_partitions,
    save_scaler,
    set_global_determinism,
    stratified_three_way_split,
    validate_disjoint_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("do_fmbs_data"))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--fingerprint-identities", type=int, default=100)
    parser.add_argument("--fingerprint-impressions", type=int, default=8)
    return parser.parse_args()


def save_similarity_figure(path: Path, same: np.ndarray, different: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bins = np.linspace(-0.4, 1.0, 60)
    ax.hist(different, bins=bins, density=True, alpha=0.68, color="#C44E52", label="Different identity")
    ax.hist(same, bins=bins, density=True, alpha=0.68, color="#4C72B0", label="Same identity")
    ax.set(xlabel="Ridge-descriptor cosine similarity", ylabel="Density",
           title="Synthetic Fingerprint Identity-Separability Gate")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fingerprint_grids(output_root: Path, images: np.ndarray) -> None:
    fig, axes = plt.subplots(4, 8, figsize=(12, 6.4))
    for identity in range(4):
        for impression in range(8):
            ax = axes[identity, impression]
            ax.imshow(images[identity * 8 + impression], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"F{identity:03d}/I{impression}", fontsize=7)
            ax.axis("off")
    fig.suptitle("Persistent Synthetic Finger Identities: Eight Controlled Impressions Each", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_root / "fingerprint_same_identity_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(8, 6.4))
    for index, ax in enumerate(axes.flat):
        ax.imshow(images[index * 8], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Finger {index:03d}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Between-Identity Synthetic Fingerprint Comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_root / "fingerprint_identity_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_face_grid(path: Path, images: np.ndarray, labels: np.ndarray, names: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(9.2, 7.0))
    for index, ax in enumerate(axes.flat):
        ax.imshow(images[index], cmap="gray")
        ax.set_title(str(names[labels[index]]), fontsize=7)
        ax.axis("off")
    fig.suptitle("LFW Face Samples (Training Partition)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.fingerprint_identities != 100 or args.fingerprint_impressions != 8:
        raise ValueError("The final study protocol requires exactly 100 identities and eight impressions")
    set_global_determinism()
    started = time.perf_counter()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "clients").mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("DO-FMBS Phase 1: leakage-free data preparation")
    print("[1/5] Loading public LFW face data")
    lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.4, color=False)
    face_images = np.asarray(lfw.images, dtype=np.float32)
    face_labels = np.asarray(lfw.target, dtype=np.int64)
    face_train, face_cal, face_test = stratified_three_way_split(
        face_labels, calibration_size=0.15, test_size=0.15, seed=FACE_SPLIT_SEED
    )
    validate_disjoint_splits(face_train, face_cal, face_test)
    face_flat = face_images.reshape(len(face_images), -1).astype(np.float64)
    face_scaler, face_scaled = fit_training_scaler(face_flat, face_train)
    save_scaler(output / "face_scaler.npz", face_scaler, face_train)
    np.savez_compressed(
        output / "face_data.npz",
        X_train=face_scaled[face_train], X_calibration=face_scaled[face_cal], X_test=face_scaled[face_test],
        y_train=face_labels[face_train], y_calibration=face_labels[face_cal], y_test=face_labels[face_test],
        images_train=face_images[face_train], images_calibration=face_images[face_cal], images_test=face_images[face_test],
        train_indices=face_train, calibration_indices=face_cal, test_indices=face_test,
        n_classes=np.asarray([len(lfw.target_names)]), n_features=np.asarray([face_flat.shape[1]]),
        image_shape=np.asarray(face_images.shape[1:]), target_names=np.asarray(lfw.target_names),
        scaler_fit_partition=np.asarray(["training"]), split_seed=np.asarray([FACE_SPLIT_SEED]),
    )
    save_face_grid(figures / "face_training_samples.png", face_images[face_train], face_labels[face_train], lfw.target_names)

    print("[2/5] Generating identity-persistent synthetic fingerprints")
    finger_images, finger_labels, impression_ids, templates = generate_fingerprint_dataset(
        args.fingerprint_identities, args.fingerprint_impressions
    )
    diagnostics, same_scores, different_scores = fingerprint_separability_diagnostics(
        finger_images, finger_labels
    )
    json_dump(output / "fingerprint_quality_diagnostics.json", diagnostics)
    np.savez_compressed(
        output / "fingerprint_similarity_scores.npz",
        same_identity=same_scores.astype(np.float32),
        different_identity=different_scores.astype(np.float32),
    )
    json_dump(output / "fingerprint_identity_templates.json", templates)
    save_similarity_figure(figures / "fingerprint_similarity_distributions.png", same_scores, different_scores)
    save_fingerprint_grids(figures, finger_images)
    if not diagnostics["passed"]:
        raise RuntimeError(
            "Fingerprint identity-separability gate failed; training is scientifically invalid. "
            + json.dumps(diagnostics, sort_keys=True)
        )

    print("[3/5] Creating fixed 5/1/2 fingerprint train/calibration/test split")
    finger_train, finger_cal, finger_test = fingerprint_impression_split(impression_ids)
    validate_disjoint_splits(finger_train, finger_cal, finger_test)
    finger_flat = finger_images.reshape(len(finger_images), -1).astype(np.float64)
    finger_scaler, finger_scaled = fit_training_scaler(finger_flat, finger_train)
    save_scaler(output / "fingerprint_scaler.npz", finger_scaler, finger_train)
    np.savez_compressed(
        output / "fingerprint_data.npz",
        X_train=finger_scaled[finger_train], X_calibration=finger_scaled[finger_cal], X_test=finger_scaled[finger_test],
        y_train=finger_labels[finger_train], y_calibration=finger_labels[finger_cal], y_test=finger_labels[finger_test],
        images_train=finger_images[finger_train], images_calibration=finger_images[finger_cal], images_test=finger_images[finger_test],
        impression_train=impression_ids[finger_train], impression_calibration=impression_ids[finger_cal], impression_test=impression_ids[finger_test],
        train_indices=finger_train, calibration_indices=finger_cal, test_indices=finger_test,
        n_classes=np.asarray([args.fingerprint_identities]), n_features=np.asarray([finger_flat.shape[1]]),
        image_shape=np.asarray(finger_images.shape[1:]), scaler_fit_partition=np.asarray(["training"]),
        identity_seed_base=np.asarray([FINGERPRINT_IDENTITY_SEED]), impression_seed_base=np.asarray([FINGERPRINT_IMPRESSION_SEED]),
    )

    print("[4/5] Persisting the exact non-IID identity allocation used by Phase 2")
    face_client_indices = non_iid_identity_partitions(face_labels[face_train], args.clients, CLIENT_PARTITION_SEED)
    finger_client_indices = non_iid_identity_partitions(finger_labels[finger_train], args.clients, CLIENT_PARTITION_SEED + 1)
    client_rows = []
    for client_id, (fi, pi) in enumerate(zip(face_client_indices, finger_client_indices)):
        np.savez_compressed(
            output / "clients" / f"client_{client_id}.npz",
            face_X=face_scaled[face_train][fi], face_y=face_labels[face_train][fi],
            finger_X=finger_scaled[finger_train][pi], finger_y=finger_labels[finger_train][pi],
            face_source_indices=face_train[fi], finger_source_indices=finger_train[pi],
        )
        client_rows.append({
            "client_id": client_id, "face_samples": int(len(fi)), "fingerprint_samples": int(len(pi)),
            "face_identities": [int(v) for v in np.unique(face_labels[face_train][fi])],
            "fingerprint_identities": [int(v) for v in np.unique(finger_labels[finger_train][pi])],
        })
    json_dump(output / "client_partition.json", {
        "strategy": "class-exclusive non-IID allocation; modality identity spaces partitioned independently",
        "seed": CLIENT_PARTITION_SEED, "clients": client_rows,
    })
    x = np.arange(args.clients)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x - 0.18, [r["face_samples"] for r in client_rows], 0.36, label="Face")
    ax.bar(x + 0.18, [r["fingerprint_samples"] for r in client_rows], 0.36, label="Fingerprint")
    ax.set(xlabel="Federated client", ylabel="Unique training samples", xticks=x,
           title="Identity-Exclusive Non-IID Client Allocation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figures / "federated_client_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("[5/5] Saving Phase 1 provenance")
    metadata = {
        "phase": 1, "replay_id": os.environ.get("DO_FMBS_REPLAY_ID", "unspecified"),
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(),
        "seeds": {"global": 42, "face_split": FACE_SPLIT_SEED,
                  "fingerprint_identity_base": FINGERPRINT_IDENTITY_SEED,
                  "fingerprint_impression_base": FINGERPRINT_IMPRESSION_SEED,
                  "client_partition": CLIENT_PARTITION_SEED},
        "face": {"source": "Labeled Faces in the Wild via scikit-learn",
                 "identities": int(len(lfw.target_names)), "total": int(len(face_images)),
                 "train": int(len(face_train)), "calibration": int(len(face_cal)), "test": int(len(face_test)),
                 "image_shape": list(face_images.shape[1:]), "scaler_fit": "training partition only"},
        "fingerprint": {"source": "locally generated identity-persistent synthetic fingerprints (not FVC)",
                        "identities": args.fingerprint_identities,
                        "impressions_per_identity": args.fingerprint_impressions, "total": int(len(finger_images)),
                        "train": int(len(finger_train)), "calibration": int(len(finger_cal)), "test": int(len(finger_test)),
                        "split_protocol": "impressions 0-4 train, 5 calibration, 6-7 final test",
                        "scaler_fit": "training partition only", "quality_gate": diagnostics},
        "runtime_seconds": time.perf_counter() - started,
    }
    json_dump(output / "phase1_metadata.json", metadata)
    print(f"Phase 1 complete in {metadata['runtime_seconds']:.2f}s")
    print(f"Fingerprint quality gate: PASS; NN={diagnostics['nearest_neighbour_identity_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

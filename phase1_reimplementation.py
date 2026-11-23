"""Source-faithful Phase 1 preparation for the 20260809 Chapter Four replay."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from do_fmbs_core import (
    FINGERPRINT_GENERATION_CONFIG,
    FINGERPRINT_IDENTITY_SEED,
    FINGERPRINT_IMPRESSION_SEED,
    generate_fingerprint_dataset,
    sha256,
)


REPLAY_ID = "full_pipeline_20260809"
SOURCE_DOCUMENT_SHA256 = "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"
SPLIT_SEED = 42
CLIENT_SEED = 42
FACE_VERIFICATION_SEED = 20260809
FINGERPRINT_IDENTITY_SEED_BASE = FINGERPRINT_IDENTITY_SEED
FINGERPRINT_IMPRESSION_SEED_BASE = FINGERPRINT_IMPRESSION_SEED
LFW_ACQUISITION = {
    "min_faces_per_person": 20,
    "resize": 0.4,
    "color": False,
    "funneled": True,
    "slice_": {"rows": [70, 195], "columns": [78, 172]},
    "download_if_missing": True,
    "data_home": None,
}


def lfw_loader_kwargs() -> dict[str, object]:
    """Translate the JSON-safe acquisition record to sklearn loader arguments."""
    settings = dict(LFW_ACQUISITION)
    crop = settings["slice_"]
    assert isinstance(crop, dict)
    settings["slice_"] = (slice(*crop["rows"]), slice(*crop["columns"]))
    return settings
EXPECTED_FACE_CLIENT_SAMPLES = [441, 315, 490, 329, 843]
EXPECTED_FINGER_CLIENT_SAMPLES = [140, 140, 120, 120, 120]


def stratified_split(labels: np.ndarray, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels), dtype=np.int64)
    train, test = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def scale_complete_array(images: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    flat = images.reshape(len(images), -1).astype(np.float64)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(flat)
    return transformed.astype(np.float32), scaler


def identity_client_groups(labels: np.ndarray, clients: int = 5, seed: int = CLIENT_SEED) -> list[np.ndarray]:
    """Reproduce the source allocation: legacy RandomState permutation, by identity."""
    classes = np.unique(labels)
    groups = np.array_split(np.random.RandomState(seed).permutation(classes), clients)
    return [np.flatnonzero(np.isin(labels, identity_group)) for identity_group in groups]


def face_verification_claims(
    test_indices: np.ndarray, labels: np.ndarray, seed: int = FACE_VERIFICATION_SEED
) -> dict[str, np.ndarray]:
    calibration, evaluation = train_test_split(
        np.asarray(test_indices, dtype=np.int64), test_size=314,
        random_state=seed, stratify=labels[test_indices],
    )
    identities = np.unique(labels)

    def claims(probes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        true_labels = labels[probes]
        imp_probe = np.repeat(probes, len(identities) - 1)
        imp_claim = np.concatenate([identities[identities != label] for label in true_labels])
        genuine_probe = probes
        genuine_claim = true_labels
        return (np.concatenate([genuine_probe, imp_probe]),
                np.concatenate([genuine_claim, imp_claim]),
                np.concatenate([np.ones(len(genuine_probe), dtype=np.bool_),
                                np.zeros(len(imp_probe), dtype=np.bool_)]))

    cal_probe, cal_claim, cal_is_genuine = claims(calibration)
    eval_probe, eval_claim, eval_is_genuine = claims(evaluation)
    return {
        "calibration_probe_indices": calibration,
        "evaluation_probe_indices": evaluation,
        "calibration_claim_probe_indices": cal_probe,
        "calibration_claim_identity": cal_claim.astype(np.int64),
        "calibration_claim_is_genuine": cal_is_genuine,
        "evaluation_claim_probe_indices": eval_probe,
        "evaluation_claim_identity": eval_claim.astype(np.int64),
        "evaluation_claim_is_genuine": eval_is_genuine,
    }


def fingerprint_verification_claims(
    test_indices: np.ndarray, labels: np.ndarray, seed: int = FACE_VERIFICATION_SEED
) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    calibration: list[int] = []
    evaluation: list[int] = []
    eligible: list[int] = []
    for identity in np.unique(labels):
        indices = np.asarray(test_indices[labels[test_indices] == identity], dtype=np.int64)
        if len(indices) == 2:
            chosen = rng.permutation(indices)
            calibration.append(int(chosen[0]))
            evaluation.append(int(chosen[1]))
            eligible.append(int(identity))

    identities = np.unique(labels)

    def claims(probes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        true_labels = labels[probes]
        imp_probe = np.repeat(probes, len(identities) - 1)
        imp_claim = np.concatenate([identities[identities != label] for label in true_labels])
        return (np.concatenate([probes, imp_probe]),
                np.concatenate([true_labels, imp_claim]).astype(np.int64),
                np.concatenate([np.ones(len(probes), dtype=np.bool_),
                                np.zeros(len(imp_probe), dtype=np.bool_)]))

    cal = np.asarray(calibration, dtype=np.int64)
    evl = np.asarray(evaluation, dtype=np.int64)
    cal_probe, cal_claim, cal_genuine = claims(cal)
    eval_probe, eval_claim, eval_genuine = claims(evl)
    return {
        "eligible_identities": np.asarray(eligible, dtype=np.int64),
        "calibration_probe_indices": cal,
        "evaluation_probe_indices": evl,
        "calibration_claim_probe_indices": cal_probe,
        "calibration_claim_identity": cal_claim,
        "calibration_claim_is_genuine": cal_genuine,
        "evaluation_claim_probe_indices": eval_probe,
        "evaluation_claim_identity": eval_claim,
        "evaluation_claim_is_genuine": eval_genuine,
    }


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_scaler(path: Path, scaler: StandardScaler) -> None:
    np.savez_compressed(path, mean=scaler.mean_, var=scaler.var_, scale=scaler.scale_,
                        n_samples_seen=np.asarray([scaler.n_samples_seen_]))


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "phase1_manifest.json"):
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                     "sha256": sha256(path)})
    return rows


def build_phase1(output: Path) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Phase 1 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "clients").mkdir()
    (output / "verification").mkdir()

    lfw = fetch_lfw_people(**lfw_loader_kwargs())
    face_images = np.asarray(lfw.images)
    face_labels = np.asarray(lfw.target, dtype=np.int64)
    face_scaled, face_scaler = scale_complete_array(face_images)
    face_train, face_test = stratified_split(face_labels)
    np.savez_compressed(
        output / "face_data.npz", X_train=face_scaled[face_train], X_test=face_scaled[face_test],
        y_train=face_labels[face_train], y_test=face_labels[face_test],
        images_train=face_images[face_train], images_test=face_images[face_test],
        train_indices=face_train, test_indices=face_test,
        target_names=np.asarray(lfw.target_names, dtype=str),
        n_classes=np.asarray([len(lfw.target_names)], dtype=np.int64),
        n_features=np.asarray([face_scaled.shape[1]], dtype=np.int64),
        image_shape=np.asarray(face_images.shape[1:], dtype=np.int64),
    )
    _save_scaler(output / "face_scaler.npz", face_scaler)

    finger_images, finger_labels, impression_ids, templates = generate_fingerprint_dataset()
    finger_scaled, finger_scaler = scale_complete_array(finger_images)
    finger_train, finger_test = stratified_split(finger_labels)
    np.savez_compressed(
        output / "fingerprint_data.npz", X_train=finger_scaled[finger_train], X_test=finger_scaled[finger_test],
        y_train=finger_labels[finger_train], y_test=finger_labels[finger_test],
        images_train=finger_images[finger_train], images_test=finger_images[finger_test],
        train_indices=finger_train, test_indices=finger_test,
        impression_train=impression_ids[finger_train], impression_test=impression_ids[finger_test],
        n_classes=np.asarray([len(np.unique(finger_labels))], dtype=np.int64),
        n_features=np.asarray([finger_scaled.shape[1]], dtype=np.int64),
        image_shape=np.asarray(finger_images.shape[1:], dtype=np.int64),
    )
    _save_scaler(output / "fingerprint_scaler.npz", finger_scaler)
    _json(output / "fingerprint_identity_templates.json", templates)

    face_groups = identity_client_groups(face_labels[face_train])
    finger_groups = identity_client_groups(finger_labels[finger_train])
    clients = []
    for client_id, (face_local, finger_local) in enumerate(zip(face_groups, finger_groups)):
        np.savez_compressed(
            output / "clients" / f"client_{client_id}.npz",
            face_X=face_scaled[face_train[face_local]], face_y=face_labels[face_train[face_local]],
            face_source_indices=face_train[face_local],
            finger_X=finger_scaled[finger_train[finger_local]], finger_y=finger_labels[finger_train[finger_local]],
            finger_source_indices=finger_train[finger_local],
        )
        clients.append({"client_id": client_id,
                        "face_samples": int(len(face_local)), "face_identities": int(len(np.unique(face_labels[face_train[face_local]]))),
                        "fingerprint_samples": int(len(finger_local)), "fingerprint_identities": int(len(np.unique(finger_labels[finger_train[finger_local]])))})

    face_verification = face_verification_claims(face_test, face_labels)
    finger_verification = fingerprint_verification_claims(finger_test, finger_labels)
    np.savez_compressed(output / "verification" / "face_claims.npz", **face_verification)
    np.savez_compressed(output / "verification" / "fingerprint_claims.npz", **finger_verification)
    _json(output / "client_partition.json", {"seed": CLIENT_SEED, "strategy": "identity-exclusive class groups using RandomState permutation", "clients": clients})

    replay_id = output.name
    seed_config = {
        "lfw_loader": {"seed": None, "purpose": "Dataset acquisition/decoding; fetch_lfw_people has no random-state parameter."},
        "fingerprint_identity": {**FINGERPRINT_GENERATION_CONFIG["identity_seed"], "purpose": "Identity-persistent template characteristics."},
        "fingerprint_impression": {**FINGERPRINT_GENERATION_CONFIG["impression_seed"], "purpose": "Deterministic per-impression acquisition variation."},
        "stratified_train_test_split": {"value": SPLIT_SEED, "purpose": "scikit-learn train_test_split(random_state=42, stratify=labels), test_size=0.2."},
        "face_verification_partition": {"value": FACE_VERIFICATION_SEED, "purpose": "Stratified calibration/evaluation partition of held-out face probes."},
        "fingerprint_verification_partition": {"value": FACE_VERIFICATION_SEED, "purpose": "NumPy RandomState permutation to assign held-out impressions to calibration/evaluation."},
        "persistent_client_identity_partition": {"value": CLIENT_SEED, "purpose": "NumPy RandomState permutation before splitting identities across five clients."},
        "python_random_module": {"value": None, "purpose": "Not used by Phase 1 data preparation."},
        "numpy_global_random_state": {"value": None, "purpose": "Not used by Phase 1 data preparation; fingerprint generation uses the explicit per-identity and per-impression generators above."},
    }
    config = {
        "replay_id": replay_id,
        "source_document": "docs/reference/CHAPTER_FOUR_MODEL_DEVELOPMENT_IMPLEMENTATION_AND_VALIDATION.docx",
        "source_document_sha256": SOURCE_DOCUMENT_SHA256,
        "methodology_clarification": {
            "document": "docs/reference/PHASE1_FINGERPRINT_METHODOLOGY_CLARIFICATIONS.md",
            "approval_date": "2026-11-23",
            "authority": "Researcher-approved clarification of details not fully specified in the original Chapter Four DOCX.",
        },
        "lfw_acquisition": LFW_ACQUISITION,
        "fingerprint_generation": FINGERPRINT_GENERATION_CONFIG,
        "seeds": seed_config,
        "preprocessing_order": ["per-impression min-max conversion to uint8", "flatten each modality", "fit StandardScaler on the complete modality array", "stratified 80:20 train/test split"],
        "known_preprocessing_limitation": "StandardScaler is fit before splitting and therefore incorporates held-out distribution statistics (full-array scaler leakage), reproduced to match Chapter Four.",
    }
    _json(output / "phase1_config.json", config)

    metadata = {
        "replay_id": replay_id, "source_document": config["source_document"],
        "source_document_sha256": SOURCE_DOCUMENT_SHA256,
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "seeds": seed_config,
        "lfw_acquisition": LFW_ACQUISITION,
        "preprocessing": "Per-impression min-max conversion to uint8, then flatten; StandardScaler.fit_transform on the complete modality array before stratified train_test_split (Chapter Four order; full-array scaler leakage).",
        "face": {"source": "LFW via scikit-learn fetch_lfw_people with persisted acquisition settings",
                 "samples": int(len(face_images)), "identities": int(len(np.unique(face_labels))),
                 "image_shape": list(face_images.shape[1:]), "features": int(face_scaled.shape[1]),
                 "train": int(len(face_train)), "test": int(len(face_test)),
                 "verification_calibration_genuine": int(len(face_verification["calibration_probe_indices"])),
                 "verification_evaluation_genuine": int(len(face_verification["evaluation_probe_indices"])),
                 "evaluation_impostor_claims": int(np.count_nonzero(~face_verification["evaluation_claim_is_genuine"]))},
        "fingerprint": {"source": "locally generated deterministic identity-persistent synthetic ridge patterns; not a public fingerprint dataset",
                        "samples": int(len(finger_images)), "identities": int(len(np.unique(finger_labels))),
                        "impressions_per_identity": 8, "image_shape": list(finger_images.shape[1:]),
                        "normalization": config["fingerprint_generation"]["normalization"],
                        "identity_to_label": [{"identity_id": int(identity), "label": int(identity)} for identity in np.unique(finger_labels)],
                        "features": int(finger_scaled.shape[1]), "train": int(len(finger_train)), "test": int(len(finger_test)),
                        "eligible_verification_identities": int(len(finger_verification["eligible_identities"])),
                        "calibration_genuine": int(len(finger_verification["calibration_probe_indices"])),
                        "evaluation_genuine": int(len(finger_verification["evaluation_probe_indices"])),
                        "calibration_impostor_claims": int(np.count_nonzero(~finger_verification["calibration_claim_is_genuine"])),
                        "evaluation_impostor_claims": int(np.count_nonzero(~finger_verification["evaluation_claim_is_genuine"]))},
        "clients": clients,
    }
    _json(output / "phase1_metadata.json", metadata)
    manifest = {"schema_version": 3, "phase": 1, "status": "ARTIFACTS_WRITTEN_REQUIRES_INDEPENDENT_VERIFICATION",
                "replay_id": replay_id, "metadata": metadata, "configuration": config,
                "artifacts": _artifact_inventory(output)}
    _json(output / "phase1_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New, isolated output directory; non-empty directories are never overwritten")
    args = parser.parse_args()
    result = build_phase1(args.output_dir)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir),
                      "artifacts": len(result["artifacts"])}, indent=2))
    print("Run phase1_reimplementation_evidence.py --data-dir <output-dir> for independent checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

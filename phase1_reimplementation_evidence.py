"""Independent checks for the source-faithful Phase 1 replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from do_fmbs_core import generate_fingerprint_dataset, sha256
from phase1_reimplementation import (
    EXPECTED_FACE_CLIENT_SAMPLES,
    EXPECTED_FINGER_CLIENT_SAMPLES,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _check(checks: list[dict[str, str]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def _independent_face_claims(test_indices: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    calibration, evaluation = train_test_split(
        np.asarray(test_indices, dtype=np.int64), test_size=314,
        random_state=20260809, stratify=labels[test_indices],
    )
    identities = np.unique(labels)
    result = {}
    for name, probes in (("calibration", calibration), ("evaluation", evaluation)):
        true_ids = labels[probes]
        impostor_probes = np.repeat(probes, len(identities) - 1)
        impostor_ids = np.concatenate([identities[identities != identity] for identity in true_ids])
        result[f"{name}_claim_probe_indices"] = np.concatenate((probes, impostor_probes))
        result[f"{name}_claim_identity"] = np.concatenate((true_ids, impostor_ids)).astype(np.int64)
        result[f"{name}_claim_is_genuine"] = np.concatenate((np.ones(len(probes), bool), np.zeros(len(impostor_probes), bool)))
    result["calibration_probe_indices"] = calibration
    result["evaluation_probe_indices"] = evaluation
    return result


def _independent_fingerprint_claims(test_indices: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    random = np.random.RandomState(20260809)
    calibration, evaluation, eligible = [], [], []
    for identity in np.unique(labels):
        samples = np.asarray(test_indices[labels[test_indices] == identity], dtype=np.int64)
        if len(samples) == 2:
            pair = random.permutation(samples)
            calibration.append(int(pair[0])); evaluation.append(int(pair[1])); eligible.append(int(identity))
    identities = np.unique(labels)
    cal, evl = np.asarray(calibration, dtype=np.int64), np.asarray(evaluation, dtype=np.int64)
    result = {"eligible_identities": np.asarray(eligible, dtype=np.int64),
              "calibration_probe_indices": cal, "evaluation_probe_indices": evl}
    for name, probes in (("calibration", cal), ("evaluation", evl)):
        true_ids = labels[probes]
        impostor_probes = np.repeat(probes, len(identities) - 1)
        impostor_ids = np.concatenate([identities[identities != identity] for identity in true_ids])
        result[f"{name}_claim_probe_indices"] = np.concatenate((probes, impostor_probes))
        result[f"{name}_claim_identity"] = np.concatenate((true_ids, impostor_ids)).astype(np.int64)
        result[f"{name}_claim_is_genuine"] = np.concatenate((np.ones(len(probes), bool), np.zeros(len(impostor_probes), bool)))
    return result


def _modality(data_dir: Path, modality: str, total: int, identities: int,
              shape: tuple[int, int], train_n: int, test_n: int,
              checks: list[dict[str, str]]) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None]:
    path = data_dir / f"{modality}_data.npz"
    try:
        data = _load(path)
        train_i, test_i = data["train_indices"], data["test_indices"]
        labels = np.empty(total, dtype=np.int64)
        labels[train_i], labels[test_i] = data["y_train"], data["y_test"]
        images = np.empty((total, *shape), dtype=data["images_train"].dtype)
        images[train_i], images[test_i] = data["images_train"], data["images_test"]
        flat = images.reshape(total, -1).astype(np.float64)
        expected_train, expected_test = train_test_split(
            np.arange(total), test_size=0.2, random_state=42, stratify=labels
        )
        split_ok = (len(train_i) == train_n and len(test_i) == test_n
                    and np.array_equal(train_i, expected_train)
                    and np.array_equal(test_i, expected_test)
                    and not np.intersect1d(train_i, test_i).size
                    and np.array_equal(np.sort(np.concatenate([train_i, test_i])), np.arange(total)))
        _check(checks, f"{modality} dimensions and split", split_ok and data["X_train"].shape == (train_n, flat.shape[1])
               and data["X_test"].shape == (test_n, flat.shape[1]) and tuple(data["image_shape"]) == shape,
               f"samples={total}; train/test={len(train_i)}/{len(test_i)}; features={flat.shape[1]}; image_shape={shape}")
        identity_ok = len(np.unique(labels)) == identities and len(data["target_names"] if modality == "face" else np.unique(labels)) == identities
        _check(checks, f"{modality} identity count", identity_ok,
               f"observed {len(np.unique(labels))}; expected {identities}")
        scaler = StandardScaler().fit(flat)
        transformed = scaler.transform(flat).astype(np.float32)
        scaled_ok = (np.allclose(data["X_train"], transformed[train_i], rtol=2e-5, atol=2e-5)
                     and np.allclose(data["X_test"], transformed[test_i], rtol=2e-5, atol=2e-5))
        with np.load(data_dir / f"{modality}_scaler.npz", allow_pickle=False) as saved:
            scaler_ok = all(np.allclose(saved[key], getattr(scaler, attr), rtol=1e-10, atol=1e-10)
                            for key, attr in (("mean", "mean_"), ("var", "var_"), ("scale", "scale_")))
        _check(checks, f"{modality} full-array scaler", scaled_ok and scaler_ok,
               "Saved scaler and transformed train/test arrays match a fresh full-array StandardScaler fit.")
        return data, labels
    except (OSError, KeyError, ValueError, IndexError) as error:
        _check(checks, f"{modality} dataset load", False, str(error))
        return None, None


def inspect_phase1(data_dir: Path) -> dict[str, object]:
    checks: list[dict[str, str]] = []
    try:
        manifest = json.loads((data_dir / "phase1_manifest.json").read_text(encoding="utf-8"))
        metadata = manifest.get("metadata", {})
        config = manifest.get("configuration", {})
        _check(checks, "Manifest identity and source document", manifest.get("phase") == 1
               and manifest.get("replay_id") == data_dir.name
               and metadata.get("replay_id") == data_dir.name
               and metadata.get("source_document_sha256") == "E3D90110AC8D4B81C7D70246FA02D3C510352DC064AF6C44792DE35E44577F5A"
               and config.get("source_document_sha256") == metadata.get("source_document_sha256"),
               f"replay_id={manifest.get('replay_id')!r}")
        seed_config = config.get("seeds", {})
        lfw_settings = config.get("lfw_acquisition", {})
        lfw_config_ok = (lfw_settings.get("min_faces_per_person") == 20
                         and lfw_settings.get("resize") == 0.4
                         and lfw_settings.get("slice_") == {
                             "rows": [70, 195], "columns": [78, 172]
                         })
        seeds_ok = (
            seed_config.get("fingerprint_identity", {}).get("base") == 73001
            and seed_config.get("fingerprint_impression", {}).get("base") == 91009
            and seed_config.get("stratified_train_test_split", {}).get("value") == 42
            and seed_config.get("face_verification_partition", {}).get("value") == 20260809
            and seed_config.get("fingerprint_verification_partition", {}).get("value") == 20260809
            and seed_config.get("persistent_client_identity_partition", {}).get("value") == 42
        )
        fingerprint_config = config.get("fingerprint_generation", {})
        normalization = fingerprint_config.get("normalization", {})
        clarification = config.get("methodology_clarification", {})
        perturb = fingerprint_config.get("impression_perturbations", {})
        normalization_ok = (
            normalization.get("scope") == "each completed impression independently"
            and normalization.get("formula") == "(pixel - impression_min) / (impression_max - impression_min) * 255"
            and normalization.get("quantization") == "numpy.rint: nearest integer, ties to even, then uint8"
            and normalization.get("constant_impression") == "all-zero uint8 image"
            and normalization.get("final_clipping", "").startswith("none;")
        )
        clarification_ok = (
            clarification.get("approval_date") == "2026-11-23"
            and clarification.get("document") == "docs/reference/PHASE1_FINGERPRINT_METHODOLOGY_CLARIFICATIONS.md"
            and perturb.get("orientation_delta_degrees") == [-1.5, 1.5]
            and perturb.get("frequency_fractional_delta") == [-0.01, 0.01]
            and "not fitted" in perturb.get("assumption", "")
        )
        _check(checks, "Persisted acquisition, conversion, and seed provenance",
               bool(lfw_config_ok and seeds_ok and normalization_ok and clarification_ok),
               "Verified LFW acquisition arguments, seeds, per-impression conversion/quantization, clarification reference, and approved perturbation bounds.")
        hash_checks = []
        manifest_artifacts = [item for item in manifest.get("artifacts", [])
                              if item.get("path") != "phase1_verification_report.json"]
        for item in manifest_artifacts:
            path = data_dir / item["path"]
            hash_checks.append(path.is_file() and path.stat().st_size == item["bytes"]
                               and sha256(path) == item["sha256"])
        _check(checks, "Artifact existence and SHA-256", bool(hash_checks) and all(hash_checks),
               f"verified {sum(hash_checks)}/{len(hash_checks)} listed Phase 1 data/configuration artifacts")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        manifest = {}
        _check(checks, "Manifest and artifact hashes", False, str(error))

    face, face_labels = _modality(data_dir, "face", 3023, 62, (50, 37), 2418, 605, checks)
    finger, finger_labels = _modality(data_dir, "fingerprint", 800, 100, (96, 96), 640, 160, checks)
    if face is not None and face_labels is not None:
        expected = _independent_face_claims(face["test_indices"], face_labels)
        try:
            actual = _load(data_dir / "verification" / "face_claims.npz")
            valid = all(np.array_equal(actual[key], value) for key, value in expected.items())
            valid &= len(actual["calibration_probe_indices"]) == 291
            valid &= len(actual["evaluation_probe_indices"]) == 314
            valid &= np.count_nonzero(~actual["evaluation_claim_is_genuine"]) == 19154
            _check(checks, "Face verification partitions and claims", bool(valid),
                   f"calibration/evaluation genuine={len(actual['calibration_probe_indices'])}/{len(actual['evaluation_probe_indices'])}; evaluation impostor claims={np.count_nonzero(~actual['evaluation_claim_is_genuine'])}")
        except (OSError, KeyError, ValueError) as error:
            _check(checks, "Face verification partitions and claims", False, str(error))
    if finger is not None and finger_labels is not None:
        expected = _independent_fingerprint_claims(finger["test_indices"], finger_labels)
        try:
            actual = _load(data_dir / "verification" / "fingerprint_claims.npz")
            valid = all(np.array_equal(actual[key], value) for key, value in expected.items())
            valid &= len(actual["eligible_identities"]) == 60
            valid &= len(actual["calibration_probe_indices"]) == len(actual["evaluation_probe_indices"]) == 60
            valid &= np.count_nonzero(~actual["calibration_claim_is_genuine"]) == 5940
            valid &= np.count_nonzero(~actual["evaluation_claim_is_genuine"]) == 5940
            _check(checks, "Fingerprint verification partitions and claims", bool(valid),
                   f"eligible={len(actual['eligible_identities'])}; calibration/evaluation genuine={len(actual['calibration_probe_indices'])}/{len(actual['evaluation_probe_indices'])}; impostor claims={np.count_nonzero(~actual['evaluation_claim_is_genuine'])}")
        except (OSError, KeyError, ValueError) as error:
            _check(checks, "Fingerprint verification partitions and claims", False, str(error))
        if finger is not None:
            try:
                images = np.empty((800, 96, 96), dtype=finger["images_train"].dtype)
                images[finger["train_indices"]] = finger["images_train"]
                images[finger["test_indices"]] = finger["images_test"]
                regenerated, regenerated_labels, _impressions, _templates = generate_fingerprint_dataset()
                _check(checks, "Fingerprint generation reproducibility", np.array_equal(images, regenerated),
                       "All 800 persisted raw impressions match a fresh deterministic per-impression min-max regeneration.")
                pixel_ok = (images.dtype == np.uint8 and np.all(images.min(axis=(1, 2)) == 0)
                            and np.all(images.max(axis=(1, 2)) == 255))
                _check(checks, "Fingerprint per-impression uint8 conversion", bool(pixel_ok),
                       f"dtype={images.dtype}; observed per-image min/max range {int(images.min())}/{int(images.max())}; all 800 images independently span 0..255.")
                # Independently inspect persisted template parameters and the first two
                # PCG64 draws; this checks the documented bounds without relying on
                # the generator's rendering helper.
                templates = json.loads((data_dir / "fingerprint_identity_templates.json").read_text(encoding="utf-8"))
                identity_cfg = fingerprint_config.get("identity_template", {})
                identity_seed_cfg = fingerprint_config.get("identity_seed", {})
                impression_seed_cfg = fingerprint_config.get("impression_seed", {})
                perturb_cfg = fingerprint_config.get("impression_perturbations", {})
                template_ok = len(templates) == 100
                for identity, template in enumerate(templates):
                    expected_seed = (identity_seed_cfg.get("base", -1)
                                     + identity_seed_cfg.get("identity_stride", -1) * identity)
                    template_ok &= (template.get("finger_id") == identity
                                    and template.get("identity_seed") == expected_seed
                                    and identity_cfg.get("base_orientation_radians", [0, 0])[0] <= template.get("base_orientation", float("inf")) <= identity_cfg.get("base_orientation_radians", [0, 0])[1]
                                    and identity_cfg.get("ridge_frequency", [0, 0])[0] <= template.get("ridge_frequency", float("inf")) <= identity_cfg.get("ridge_frequency", [0, 0])[1])
                draw_ok = True
                for identity in range(100):
                    for impression in range(8):
                        seed = (impression_seed_cfg.get("base", -1)
                                + impression_seed_cfg.get("identity_stride", -1) * identity
                                + impression_seed_cfg.get("impression_stride", -1) * impression)
                        rng = np.random.Generator(np.random.PCG64(seed))
                        orientation_delta = rng.uniform(*perturb_cfg["orientation_delta_radians"])
                        frequency_delta = rng.uniform(*perturb_cfg["frequency_fractional_delta"])
                        draw_ok &= (abs(orientation_delta) <= np.deg2rad(1.5)
                                    and abs(frequency_delta) <= 0.01
                                    and -np.pi / 2 <= templates[identity]["base_orientation"] + orientation_delta <= np.pi / 2)
                _check(checks, "Persistent template and impression perturbation bounds",
                       bool(template_ok and draw_ok),
                       "All 100 templates retain in-range identity base orientation/frequency; independently reconstructed seeded impression draws stay within ±1.5° and ±1%.")
                _check(checks, "Fingerprint identity-to-label mapping",
                       np.array_equal(np.unique(regenerated_labels), np.arange(100))
                       and manifest.get("metadata", {}).get("fingerprint", {}).get("identity_to_label") == [
                           {"identity_id": identity, "label": identity} for identity in range(100)
                       ],
                       "Persisted synthetic identity IDs map one-to-one to integer labels 0–99.")
            except (ValueError, IndexError) as error:
                _check(checks, "Fingerprint generation reproducibility", False, str(error))
            else:
                _check(checks, "Fingerprint impression variation",
                       len(np.unique(images[0:8], axis=0)) > 1 and not np.array_equal(images[0], images[8]),
                       "Impressions vary within identity 0 and the first two identities are not duplicate samples.")

    try:
        def expected_groups(train_labels: np.ndarray) -> list[np.ndarray]:
            classes = np.unique(train_labels)
            groups = np.array_split(np.random.RandomState(42).permutation(classes), 5)
            return [np.flatnonzero(np.isin(train_labels, identities)) for identities in groups]

        face_groups = expected_groups(face_labels[face["train_indices"]])
        finger_groups = expected_groups(finger_labels[finger["train_indices"]])
        assignments = {"face": [], "fingerprint": []}
        client_rows = []
        for client_id in range(5):
            client = _load(data_dir / "clients" / f"client_{client_id}.npz")
            fi, pi = face_groups[client_id], finger_groups[client_id]
            face_ids = np.unique(client["face_y"])
            finger_ids = np.unique(client["finger_y"])
            valid = (np.array_equal(client["face_source_indices"], face["train_indices"][fi])
                     and np.array_equal(client["finger_source_indices"], finger["train_indices"][pi])
                     and np.array_equal(client["face_y"], face_labels[client["face_source_indices"]])
                     and np.array_equal(client["finger_y"], finger_labels[client["finger_source_indices"]])
                     and np.array_equal(client["face_X"], face["X_train"][fi])
                     and np.array_equal(client["finger_X"], finger["X_train"][pi]))
            _check(checks, f"Client {client_id} persistence", bool(valid),
                   f"face={len(fi)} samples/{len(face_ids)} identities; fingerprint={len(pi)} samples/{len(finger_ids)} identities")
            assignments["face"].append(set(face_ids.tolist()))
            assignments["fingerprint"].append(set(finger_ids.tolist()))
            client_rows.append({"Client": client_id, "Face samples": len(fi), "Face identities": len(face_ids),
                                "Fingerprint samples": len(pi), "Fingerprint identities": len(finger_ids)})
        disjoint = all(not (a & b) for modality in assignments.values()
                       for i, a in enumerate(modality) for b in modality[i + 1:])
        counts_ok = [row["Face samples"] for row in client_rows] == EXPECTED_FACE_CLIENT_SAMPLES
        counts_ok &= [row["Fingerprint samples"] for row in client_rows] == EXPECTED_FINGER_CLIENT_SAMPLES
        identity_counts_ok = [row["Face identities"] for row in client_rows] == [13, 13, 12, 12, 12]
        identity_counts_ok &= [row["Fingerprint identities"] for row in client_rows] == [20] * 5
        coverage_ok = (sum(row["Face samples"] for row in client_rows) == 2418
                       and sum(row["Fingerprint samples"] for row in client_rows) == 640)
        _check(checks, "Identity-exclusive persistent clients", bool(disjoint and counts_ok and identity_counts_ok and coverage_ok),
               f"face sample counts={[r['Face samples'] for r in client_rows]}; fingerprint sample counts={[r['Fingerprint samples'] for r in client_rows]}")
        partition = json.loads((data_dir / "client_partition.json").read_text(encoding="utf-8"))
        _check(checks, "Client partition metadata", partition.get("clients") == manifest.get("metadata", {}).get("clients"),
               "Persisted client summary matches the machine-readable Phase 1 manifest.")
    except (OSError, KeyError, ValueError, IndexError, TypeError) as error:
        client_rows = []
        _check(checks, "Persistent client coverage", False, str(error))

    return {"status": "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "FAIL",
            "checks": checks, "clients": client_rows, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--record-verification", action="store_true",
                        help="Record successful/failed independent checks in the Phase 1 manifest")
    args = parser.parse_args()
    result = inspect_phase1(args.data_dir)
    if args.record_verification:
        manifest_path = args.data_dir / "phase1_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verified_utc = datetime.now(timezone.utc).isoformat()
        report = {"replay_id": manifest.get("replay_id"), "status": result["status"],
                  "verified_utc": verified_utc, "checks": result["checks"]}
        report_path = args.data_dir / "phase1_verification_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["artifacts"] = [item for item in manifest.get("artifacts", [])
                                 if item.get("path") != "phase1_verification_report.json"]
        manifest["status"] = ("PHASE1-ARTIFACTS-INDEPENDENTLY-VERIFIED"
                              if result["status"] == "PASS" else "PHASE1-VERIFICATION-FAILED")
        manifest["verification"] = {"status": result["status"],
                                     "verified_utc": verified_utc,
                                     "checks": result["checks"]}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

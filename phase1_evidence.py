"""Read-only inspection of persisted Phase 1 datasets and client artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from do_fmbs_core import (
    FACE_SPLIT_SEED,
    fingerprint_impression_split,
    non_iid_identity_partitions,
    stratified_three_way_split,
)


CHAPTER_FOUR_EXPECTED = {
    "face": {"total": 3023, "identities": 62, "image_shape": (50, 37), "features": 1850,
             "splits": {"train": 2115, "calibration": 454, "test": 454}},
    "fingerprint": {"total": 800, "identities": 100, "image_shape": (96, 96), "features": 9216,
                    "splits": {"train": 500, "calibration": 100, "test": 200}},
}


def _record(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _verify_scaler(data: dict[str, np.ndarray], path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "Scaler archive is missing."
    try:
        saved = _load_npz(path)
        splits = ("train", "calibration", "test")
        indices = [data[f"{split}_indices"] for split in splits]
        images = [data[f"images_{split}"] for split in splits]
        raw = np.empty((sum(map(len, indices)), data["X_train"].shape[1]), dtype=np.float64)
        for source_indices, image_array in zip(indices, images):
            raw[source_indices] = image_array.reshape(len(image_array), -1).astype(np.float64)
        fitted = StandardScaler().fit(raw[data["train_indices"]])
        parameters_match = all(np.allclose(saved[name], getattr(fitted, attr), rtol=1e-10, atol=1e-10)
                               for name, attr in (("mean", "mean_"), ("scale", "scale_"), ("var", "var_")))
        fit_indices_match = np.array_equal(saved["fitted_train_indices"], data["train_indices"])
        transformed_match = all(np.allclose(
            data[f"X_{split}"], fitted.transform(raw[data[f"{split}_indices"]]).astype(np.float32),
            rtol=2e-5, atol=2e-5) for split in splits)
        passed = parameters_match and fit_indices_match and transformed_match
        return passed, ("Saved scaler parameters, fit indices and transformed arrays match a training-only refit."
                        if passed else "Scaler parameters, fit indices or transformed arrays disagree with a training-only refit.")
    except (KeyError, OSError, ValueError, IndexError) as error:
        return False, f"Could not verify scaler: {error}"


def _verify_modality(data_dir: Path, modality: str, checks: list[dict]) -> tuple[dict | None, dict]:
    expected = CHAPTER_FOUR_EXPECTED[modality]
    path = data_dir / f"{modality}_data.npz"
    if not path.is_file():
        _record(checks, f"{modality} dataset", False, "Dataset archive is missing.")
        return None, {}
    try:
        data = _load_npz(path)
        splits = ("train", "calibration", "test")
        sizes = {split: len(data[f"y_{split}"]) for split in splits}
        indices = [data[f"{split}_indices"] for split in splits]
        images = [data[f"images_{split}"] for split in splits]
        features = [data[f"X_{split}"] for split in splits]
        all_indices = np.concatenate(indices)
        coverage = (len(np.unique(all_indices)) == expected["total"]
                    and set(all_indices.tolist()) == set(range(expected["total"])))
        labels_by_index = np.empty(expected["total"], dtype=np.int64)
        for split in splits:
            labels_by_index[data[f"{split}_indices"]] = data[f"y_{split}"]
        disjoint = all(not (set(indices[i].tolist()) & set(indices[j].tolist()))
                       for i in range(3) for j in range(i + 1, 3))
        structure_ok = (sum(sizes.values()) == expected["total"]
                        and all(sizes[s] == expected["splits"][s] for s in splits)
                        and all(tuple(image.shape[1:]) == expected["image_shape"] for image in images)
                        and all(feature.ndim == 2 and feature.shape[1] == expected["features"] for feature in features)
                        and len(np.unique(labels_by_index)) == expected["identities"])
        _record(checks, f"{modality} structure", structure_ok,
                f"{sizes}; {len(np.unique(labels_by_index))} identities; image {expected['image_shape']}; {expected['features']} features.")
        _record(checks, f"{modality} split coverage", bool(coverage and disjoint),
                "Indices are mutually disjoint and cover each source observation exactly once."
                if coverage and disjoint else "Indices overlap or do not cover the expected source observations.")

        if modality == "face":
            replayed = stratified_three_way_split(labels_by_index, seed=FACE_SPLIT_SEED)
            split_seed = FACE_SPLIT_SEED
        else:
            impressions = np.empty(expected["total"], dtype=np.int64)
            for split in splits:
                impressions[data[f"{split}_indices"]] = data[f"impression_{split}"]
            replayed = fingerprint_impression_split(impressions)
            split_seed = None
        reproducible = all(np.array_equal(left, right) for left, right in zip(replayed, indices))
        _record(checks, f"{modality} split reproducibility", reproducible,
                f"Existing split function reproduces saved indices{f' with seed {split_seed}' if split_seed is not None else ' from impression IDs 0–4 / 5 / 6–7'}.")

        scaler_ok, scaler_detail = _verify_scaler(data, data_dir / f"{modality}_scaler.npz")
        _record(checks, f"{modality} scaler", scaler_ok, scaler_detail)
        summary = {
            "source": "Labeled Faces in the Wild via scikit-learn" if modality == "face"
                      else "Locally generated synthetic fingerprints (not FVC)",
            "acquisition": "fetch_lfw_people(min_faces_per_person=20, resize=0.4, color=False)"
                          if modality == "face" else "Deterministic local identity-template/impression generator",
            "samples": sum(sizes.values()), "identities": len(np.unique(labels_by_index)),
            "image_shape": list(expected["image_shape"]), "features": expected["features"],
            "representation": "grayscale float images" if modality == "face" else "grayscale uint8 ridge images",
            "splits": sizes, "split_seed": split_seed,
        }
        if modality == "fingerprint":
            summary["impressions_per_identity"] = 8
        return data, summary
    except (KeyError, OSError, ValueError, IndexError) as error:
        _record(checks, f"{modality} dataset", False, f"Could not inspect archive: {error}")
        return None, {}


def inspect_phase1_artifacts(data_dir: Path | str) -> dict:
    """Read and verify persisted Phase 1 artifacts without changing them."""
    data_dir = Path(data_dir)
    checks: list[dict] = []
    metadata_path = data_dir / "phase1_metadata.json"
    partition_path = data_dir / "client_partition.json"
    metadata = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            _record(checks, "Phase 1 metadata", metadata.get("phase") == 1,
                    "Readable Phase 1 metadata." if metadata.get("phase") == 1 else "Unexpected phase number in metadata.")
        except (OSError, json.JSONDecodeError) as error:
            _record(checks, "Phase 1 metadata", False, f"Metadata cannot be read: {error}")
    else:
        _record(checks, "Phase 1 metadata", False, "Metadata file is missing.")

    loaded: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    for modality in ("face", "fingerprint"):
        dataset, summary = _verify_modality(data_dir, modality, checks)
        if dataset is None:
            continue
        loaded[modality], summaries[modality] = dataset, summary
        recorded = metadata.get(modality, {})
        expected_fields = {
            "total": summary["samples"], "identities": summary["identities"],
            "train": summary["splits"]["train"], "calibration": summary["splits"]["calibration"],
            "test": summary["splits"]["test"],
        }
        # Older accepted metadata records the face dimensions but omits the
        # fingerprint dimensions; the persisted arrays are checked directly above.
        if "image_shape" in recorded:
            expected_fields["image_shape"] = summary["image_shape"]
        matches = all(recorded.get(key) == value for key, value in expected_fields.items())
        _record(checks, f"{modality} metadata consistency", matches,
                "Metadata agrees with persisted arrays." if matches else "Metadata differs from persisted arrays.")

    client_rows: list[dict] = []
    client_check = False
    try:
        if not partition_path.is_file():
            raise FileNotFoundError("client_partition.json is missing")
        if set(loaded) != {"face", "fingerprint"}:
            raise ValueError("Both modality datasets are required to verify clients")
        partition = json.loads(partition_path.read_text(encoding="utf-8"))
        entries = partition["clients"]
        client_check = len(entries) == 5
        assignments = {"face": [], "fingerprint": []}
        identity_sets = {"face": [], "fingerprint": []}
        for modality, source in loaded.items():
            seed = int(partition["seed"]) + (1 if modality == "fingerprint" else 0)
            groups = non_iid_identity_partitions(source["y_train"], len(entries), seed)
            train_positions = {int(index): position for position, index in enumerate(source["train_indices"])}
            for group_index, row in enumerate(entries):
                client_path = data_dir / "clients" / f"client_{row['client_id']}.npz"
                if not client_path.is_file():
                    client_check = False
                    continue
                client = _load_npz(client_path)
                prefix = "face" if modality == "face" else "finger"
                saved_indices = client[f"{prefix}_source_indices"]
                expected_indices = source["train_indices"][groups[group_index]]
                positions = np.asarray([train_positions.get(int(index), -1) for index in saved_indices])
                ids = set(client[f"{prefix}_y"].astype(int).tolist())
                match = (np.array_equal(saved_indices, expected_indices) and np.all(positions >= 0)
                         and np.array_equal(client[f"{prefix}_y"], source["y_train"][positions])
                         and np.array_equal(client[f"{prefix}_X"], source["X_train"][positions])
                         and len(saved_indices) == int(row[f"{modality}_samples"])
                         and ids == set(map(int, row[f"{modality}_identities"])))
                client_check &= bool(match)
                assignments[modality].extend(saved_indices.astype(int).tolist())
                identity_sets[modality].append(ids)
                key = int(row["client_id"])
                record = next((item for item in client_rows if item["Client"] == key), None)
                if record is None:
                    record = {"Client": key}
                    client_rows.append(record)
                record[f"{modality.title()} samples"] = len(saved_indices)
                record[f"{modality.title()} identities"] = len(ids)
            client_check &= (len(assignments[modality]) == len(set(assignments[modality]))
                             and set(assignments[modality]) == set(source["train_indices"].astype(int).tolist()))
            client_check &= all(not (left & right) for i, left in enumerate(identity_sets[modality])
                                for right in identity_sets[modality][i + 1:])
        client_rows.sort(key=lambda item: item["Client"])
        _record(checks, "Identity-exclusive client artifacts", bool(client_check),
                "Five clients reproduce the seeded assignments; indices, labels, features and full training coverage match."
                if client_check else "Client assignment, count, identity exclusivity or persisted data consistency failed.")
    except (KeyError, OSError, json.JSONDecodeError, ValueError, IndexError) as error:
        client_check = False
        _record(checks, "Identity-exclusive client artifacts", False, f"Could not verify clients: {error}")

    previews: list[dict] = []
    if "face" in loaded:
        face = loaded["face"]
        labels = face["y_train"]
        selected = np.unique(labels)
        selected = selected[np.linspace(0, len(selected) - 1, min(8, len(selected)), dtype=int)]
        names = face.get("target_names")
        for label in selected:
            index = int(np.flatnonzero(labels == label)[0])
            caption = str(names[int(label)]) if names is not None and int(label) < len(names) else f"Identity {int(label)}"
            previews.append({"image": face["images_train"][index], "caption": caption})

    return {
        "status": "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "CHECKS-INCOMPLETE",
        "checks": checks, "datasets": summaries, "clients": client_rows,
        "clients_verified": bool(client_check), "face_previews": previews,
    }

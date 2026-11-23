"""Shared scientific utilities for the DO-FMBS experimental pipeline.

The project contains two independent biometric identity spaces: public LFW face
identities and locally generated synthetic fingerprint identities.  Utilities
in this module keep those tasks separate while allowing one two-input model to
train both modality-specific heads.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


GLOBAL_SEED = 42
FACE_SPLIT_SEED = 4201
FINGERPRINT_IDENTITY_SEED = 73001
FINGERPRINT_IMPRESSION_SEED = 91009
CLIENT_PARTITION_SEED = 5201
BOOTSTRAP_SEED = 6201

FINGERPRINT_GENERATION_CONFIG: dict[str, object] = {
    "identities": 100, "impressions_per_identity": 8, "image_shape": [96, 96],
    "random_generator": "numpy.random.Generator(numpy.random.PCG64(seed))",
    "identity_seed": {"base": FINGERPRINT_IDENTITY_SEED, "identity_stride": 1009,
                       "formula": "base + identity_stride * identity_id"},
    "impression_seed": {"base": FINGERPRINT_IMPRESSION_SEED,
                         "identity_stride": 10007, "impression_stride": 101,
                         "formula": "base + identity_stride * identity_id + impression_stride * impression_id",
                         "draw_order": ["orientation_delta_radians", "frequency_fractional_delta", "remaining acquisition variations"]},
    "normalization": {"scope": "each completed impression independently",
                      "formula": "(pixel - impression_min) / (impression_max - impression_min) * 255",
                      "quantization": "numpy.rint: nearest integer, ties to even, then uint8",
                      "constant_impression": "all-zero uint8 image",
                      "final_clipping": "none; min-max scaling bounds finite nonconstant values"},
    "identity_template": {
        "minutiae_count": 5, "minutiae_x": [-0.62, 0.62], "minutiae_y": [-0.72, 0.72],
        "minutiae_radius": [0.10, 0.24], "minutiae_phase_shift": [-0.42, 0.42],
        "base_orientation_radians": [-1.10, 1.10], "ridge_frequency": [5.3, 8.1],
        "core_x": [-0.25, 0.25], "core_y": [-0.30, 0.20],
        "curvature": [-0.34, 0.34], "swirl": [-0.22, 0.22],
        "phase_offset_radians": [0.0, 6.283185307179586], "warp_x": [0.10, 0.32],
        "warp_y": [0.10, 0.32], "warp_phase_radians": [0.0, 6.283185307179586],
        "ellipticity": [0.82, 1.12],
    },
    "identity_field": {"coordinate_warp_amplitude": 0.055,
                       "warp_x_angular_frequency_pi": 2.4, "warp_y_angular_frequency_pi": 2.0,
                       "minutiae_amplitude": 0.055, "ridge_tanh_scale": 3.2,
                       "minutiae_gaussian_denominator_factor": 2.0,
                       "ridge_phase_frequency_factor": 2.0,
                       "contact_ellipse_outer": 1.12, "contact_ellipse_width": 0.22,
                       "background_level": 0.93, "background_gradient": 0.025, "radial_epsilon": 1e-6},
    "impression_perturbations": {
        "orientation_delta_radians": [-0.02617993877991494, 0.02617993877991494],
        "orientation_delta_degrees": [-1.5, 1.5],
        "orientation_formula": "base_orientation + first impression-RNG uniform draw",
        "frequency_fractional_delta": [-0.01, 0.01],
        "frequency_formula": "base_frequency * (1 + second impression-RNG uniform draw)",
        "assumption": "Conservative researcher-approved bounds, not fitted or performance-optimized parameters.",
    },
    "acquisition_variation": {
        "rotation_degrees": [-1.15, 1.15], "shift_pixels_yx": [-1.1, 1.1],
        "rigid_interpolation_order": 1, "boundary_mode": "nearest",
        "deformation_grid_shape": [7, 7], "deformation_noise_mean": 0.0,
        "deformation_noise_std": 1.0, "deformation_zoom_order": 3,
        "deformation_zoom_grid_divisor": 7.0,
        "deformation_smoothing_sigma": 4.0, "deformation_displacement_std_pixels": 0.65,
        "deformation_std_epsilon": 1e-8, "contrast": [0.90, 1.10],
        "brightness": [-0.025, 0.025], "gamma": [0.92, 1.08],
        "pre_blur_clip": [0.0, 1.0], "blur_sigma": [0.45, 0.82],
        "sensor_noise_mean": 0.0, "sensor_noise_std": [0.012, 0.030],
        "contact_center_x_y": [-0.035, 0.035], "contact_scale_x": 1.02,
        "contact_scale_y": 1.08, "contact_mask_outer": 1.16,
        "contact_mask_width": 0.18, "contact_background": 0.94,
    },
}


def set_global_determinism(seed: int = GLOBAL_SEED) -> None:
    """Set reproducible Python/NumPy/TensorFlow seeds where available."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass


def json_dump(path: Path | str, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_seed(finger_id: int) -> int:
    seed = FINGERPRINT_GENERATION_CONFIG["identity_seed"]
    return int(seed["base"]) + int(seed["identity_stride"]) * int(finger_id)


def impression_seed(finger_id: int, impression: int) -> int:
    seed = FINGERPRINT_GENERATION_CONFIG["impression_seed"]
    return (int(seed["base"]) + int(seed["identity_stride"]) * int(finger_id)
            + int(seed["impression_stride"]) * int(impression))


def create_fingerprint_identity_template(
    finger_id: int, height: int = 96, width: int = 96
) -> dict[str, object]:
    """Create persistent ridge parameters for exactly one synthetic finger."""
    cfg = FINGERPRINT_GENERATION_CONFIG["identity_template"]
    rng = np.random.Generator(np.random.PCG64(identity_seed(finger_id)))
    minutiae = []
    for _ in range(int(cfg["minutiae_count"])):
        minutiae.append(
            {
                "x": float(rng.uniform(*cfg["minutiae_x"])),
                "y": float(rng.uniform(*cfg["minutiae_y"])),
                "radius": float(rng.uniform(*cfg["minutiae_radius"])),
                "phase_shift": float(rng.uniform(*cfg["minutiae_phase_shift"])),
            }
        )
    return {
        "finger_id": int(finger_id),
        "identity_seed": identity_seed(finger_id),
        "height": int(height),
        "width": int(width),
        "base_orientation": float(rng.uniform(*cfg["base_orientation_radians"])),
        "ridge_frequency": float(rng.uniform(*cfg["ridge_frequency"])),
        "core_x": float(rng.uniform(*cfg["core_x"])),
        "core_y": float(rng.uniform(*cfg["core_y"])),
        "curvature": float(rng.uniform(*cfg["curvature"])),
        "swirl": float(rng.uniform(*cfg["swirl"])),
        "phase_offset": float(rng.uniform(*cfg["phase_offset_radians"])),
        "warp_x": float(rng.uniform(*cfg["warp_x"])),
        "warp_y": float(rng.uniform(*cfg["warp_y"])),
        "warp_phase": float(rng.uniform(*cfg["warp_phase_radians"])),
        "ellipticity": float(rng.uniform(*cfg["ellipticity"])),
        "minutiae": minutiae,
    }


def _render_identity_field(
    template: dict[str, object], orientation_delta: float = 0.0,
    frequency_fractional_delta: float = 0.0,
) -> np.ndarray:
    field_cfg = FINGERPRINT_GENERATION_CONFIG["identity_field"]
    height = int(template["height"])
    width = int(template["width"])
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    x = xx - float(template["core_x"])
    y = yy - float(template["core_y"])
    angle = float(template["base_orientation"]) + orientation_delta
    u = x * np.cos(angle) + y * np.sin(angle)
    v = -x * np.sin(angle) + y * np.cos(angle)
    radial = np.sqrt(x * x + y * y) + field_cfg["radial_epsilon"]
    polar = np.arctan2(v, u)

    coordinate = (
        u
        + float(template["curvature"]) * v * v
        + float(template["swirl"]) * radial * polar / np.pi
        + float(template["warp_x"]) * field_cfg["coordinate_warp_amplitude"] * np.sin(field_cfg["warp_x_angular_frequency_pi"] * np.pi * x + float(template["warp_phase"]))
        + float(template["warp_y"]) * field_cfg["coordinate_warp_amplitude"] * np.sin(field_cfg["warp_y_angular_frequency_pi"] * np.pi * y - float(template["warp_phase"]))
    )
    for point in template["minutiae"]:
        distance2 = (xx - point["x"]) ** 2 + (yy - point["y"]) ** 2
        coordinate += point["phase_shift"] * field_cfg["minutiae_amplitude"] * np.exp(-distance2 / (field_cfg["minutiae_gaussian_denominator_factor"] * point["radius"] ** 2))

    frequency = float(template["ridge_frequency"]) * (1.0 + frequency_fractional_delta)
    phase = field_cfg["ridge_phase_frequency_factor"] * np.pi * frequency * coordinate + float(template["phase_offset"])
    ridge = 0.5 + 0.5 * np.cos(phase)
    ridge = 0.5 + 0.5 * np.tanh((ridge - 0.5) * field_cfg["ridge_tanh_scale"])
    ellipse = (xx / float(template["ellipticity"])) ** 2 + (yy * float(template["ellipticity"])) ** 2
    contact = np.clip((field_cfg["contact_ellipse_outer"] - ellipse) / field_cfg["contact_ellipse_width"], 0.0, 1.0)
    background = field_cfg["background_level"] + field_cfg["background_gradient"] * (xx + yy)
    return ridge * contact + background * (1.0 - contact)


def render_fingerprint_impression(
    template: dict[str, object], impression: int
) -> np.ndarray:
    """Render a controlled acquisition variation of a persistent identity."""
    finger_id = int(template["finger_id"])
    rng = np.random.Generator(np.random.PCG64(impression_seed(finger_id, impression)))
    perturb_cfg = FINGERPRINT_GENERATION_CONFIG["impression_perturbations"]
    orientation_delta = float(rng.uniform(*perturb_cfg["orientation_delta_radians"]))
    frequency_delta = float(rng.uniform(*perturb_cfg["frequency_fractional_delta"]))
    base = _render_identity_field(template, orientation_delta, frequency_delta)
    height, width = base.shape
    acq = FINGERPRINT_GENERATION_CONFIG["acquisition_variation"]

    # Modest rigid acquisition variation.
    rotated = ndimage.rotate(
        base,
        angle=float(rng.uniform(*acq["rotation_degrees"])),
        reshape=False,
        order=int(acq["rigid_interpolation_order"]),
        mode=acq["boundary_mode"],
    )
    shifted = ndimage.shift(
        rotated,
        shift=tuple(float(x) for x in rng.uniform(*acq["shift_pixels_yx"], size=2)),
        order=int(acq["rigid_interpolation_order"]),
        mode=acq["boundary_mode"],
    )

    # Smooth sensor/contact deformation; displacement stays below one pixel.
    grid_shape = tuple(int(v) for v in acq["deformation_grid_shape"])
    coarse_y = rng.normal(acq["deformation_noise_mean"], acq["deformation_noise_std"], grid_shape)
    coarse_x = rng.normal(acq["deformation_noise_mean"], acq["deformation_noise_std"], grid_shape)
    zoom = (height / acq["deformation_zoom_grid_divisor"], width / acq["deformation_zoom_grid_divisor"])
    dy = ndimage.zoom(coarse_y, zoom, order=int(acq["deformation_zoom_order"]))[:height, :width]
    dx = ndimage.zoom(coarse_x, zoom, order=int(acq["deformation_zoom_order"]))[:height, :width]
    dy = ndimage.gaussian_filter(dy, acq["deformation_smoothing_sigma"])
    dx = ndimage.gaussian_filter(dx, acq["deformation_smoothing_sigma"])
    dy = acq["deformation_displacement_std_pixels"] * dy / (np.std(dy) + acq["deformation_std_epsilon"])
    dx = acq["deformation_displacement_std_pixels"] * dx / (np.std(dx) + acq["deformation_std_epsilon"])
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    image = ndimage.map_coordinates(shifted, [grid_y + dy, grid_x + dx], order=1, mode="nearest")

    # Pressure, illumination, blur, and sensor noise vary by impression.
    contrast = float(rng.uniform(*acq["contrast"]))
    brightness = float(rng.uniform(*acq["brightness"]))
    gamma = float(rng.uniform(*acq["gamma"]))
    image = np.clip((image - 0.5) * contrast + 0.5 + brightness, *acq["pre_blur_clip"]) ** gamma
    image = ndimage.gaussian_filter(image, sigma=float(rng.uniform(*acq["blur_sigma"])))
    image += rng.normal(acq["sensor_noise_mean"], float(rng.uniform(*acq["sensor_noise_std"])), image.shape)

    # A weak shifted contact mask permits partial edge contact without erasing
    # the core ridge field.
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    cx, cy = rng.uniform(*acq["contact_center_x_y"], size=2)
    mask = np.clip((acq["contact_mask_outer"] - ((xx - cx) / acq["contact_scale_x"]) ** 2 - ((yy - cy) / acq["contact_scale_y"]) ** 2) / acq["contact_mask_width"], 0.0, 1.0)
    image = image * mask + acq["contact_background"] * (1.0 - mask)
    return minmax_to_uint8(image)


def minmax_to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale one impression's observed range linearly to the 8-bit range."""
    values = np.asarray(image, dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    normalization = FINGERPRINT_GENERATION_CONFIG["normalization"]
    if maximum == minimum and normalization["constant_impression"] == "all-zero uint8 image":
        return np.zeros(values.shape, dtype=np.uint8)
    if maximum == minimum:
        raise ValueError("Unsupported constant-impression normalization policy")
    scaled = (values - minimum) / (maximum - minimum)
    return np.rint(scaled * 255.0).astype(np.uint8)


def generate_fingerprint_dataset(
    identities: int = 100, impressions: int = 8, height: int = 96, width: int = 96
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    impression_ids: list[int] = []
    templates: list[dict[str, object]] = []
    for finger_id in range(identities):
        template = create_fingerprint_identity_template(finger_id, height, width)
        templates.append(template)
        for impression in range(impressions):
            images.append(render_fingerprint_impression(template, impression))
            labels.append(finger_id)
            impression_ids.append(impression)
    return (
        np.asarray(images, dtype=np.uint8),
        np.asarray(labels, dtype=np.int64),
        np.asarray(impression_ids, dtype=np.int64),
        templates,
    )


def fingerprint_quality_features(images: np.ndarray) -> np.ndarray:
    """Extract acquisition-tolerant ridge descriptors for a pre-training gate."""
    descriptors = []
    for raw in images:
        image = raw.astype(np.float64) / 255.0
        image = ndimage.gaussian_filter(image, 1.0)
        gx = ndimage.sobel(image, axis=1)
        gy = ndimage.sobel(image, axis=0)
        magnitude = np.hypot(gx, gy)
        orientation = np.mod(np.arctan2(gy, gx), np.pi)
        cells = []
        for row in np.array_split(np.arange(image.shape[0]), 4):
            for col in np.array_split(np.arange(image.shape[1]), 4):
                hist, _ = np.histogram(
                    orientation[np.ix_(row, col)],
                    bins=12,
                    range=(0.0, np.pi),
                    weights=magnitude[np.ix_(row, col)],
                )
                cells.extend(hist.tolist())
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))
        center = spectrum[36:60, 36:60]
        descriptor = np.concatenate([np.asarray(cells), center.ravel()])
        descriptor -= descriptor.mean()
        descriptor /= np.linalg.norm(descriptor) + 1e-12
        descriptors.append(descriptor)
    return np.asarray(descriptors)


def fingerprint_separability_diagnostics(
    images: np.ndarray, labels: np.ndarray, *, different_pair_limit: int = 20_000
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    features = fingerprint_quality_features(images)
    similarity = features @ features.T
    same_mask = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(same_mask, dtype=bool), k=1)
    same_scores = similarity[same_mask & upper]
    different_scores_all = similarity[(~same_mask) & upper]
    rng = np.random.default_rng(GLOBAL_SEED)
    if len(different_scores_all) > different_pair_limit:
        different_scores = rng.choice(different_scores_all, different_pair_limit, replace=False)
    else:
        different_scores = different_scores_all

    np.fill_diagonal(similarity, -np.inf)
    nearest = labels[np.argmax(similarity, axis=1)]
    nn_accuracy = float(np.mean(nearest == labels))
    same_mean = float(np.mean(same_scores))
    different_mean = float(np.mean(different_scores))
    similarity_margin = same_mean - different_mean
    intra_distance = float(np.mean(1.0 - same_scores))
    inter_distance = float(np.mean(1.0 - different_scores))
    separation_ratio = inter_distance / max(intra_distance, 1e-12)
    passed = bool(similarity_margin >= 0.08 and separation_ratio >= 1.15 and nn_accuracy >= 0.80)
    diagnostics = {
        "descriptor": "4x4-cell gradient-orientation histograms plus central log Fourier magnitude",
        "same_identity_pairs": int(len(same_scores)),
        "different_identity_pairs_sampled": int(len(different_scores)),
        "same_identity_similarity_mean": same_mean,
        "same_identity_similarity_std": float(np.std(same_scores)),
        "different_identity_similarity_mean": different_mean,
        "different_identity_similarity_std": float(np.std(different_scores)),
        "similarity_margin": similarity_margin,
        "intra_class_distance_mean": intra_distance,
        "inter_class_distance_mean": inter_distance,
        "inter_to_intra_distance_ratio": separation_ratio,
        "nearest_neighbour_identity_accuracy": nn_accuracy,
        "acceptance_rules": {
            "minimum_similarity_margin": 0.08,
            "minimum_inter_to_intra_ratio": 1.15,
            "minimum_nearest_neighbour_accuracy": 0.80,
        },
        "passed": passed,
    }
    return diagnostics, same_scores, different_scores


def stratified_three_way_split(
    labels: np.ndarray, *, calibration_size: float = 0.15, test_size: float = 0.15,
    seed: int = FACE_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_cal, test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    relative_cal = calibration_size / (1.0 - test_size)
    train, calibration = train_test_split(
        train_cal,
        test_size=relative_cal,
        random_state=seed + 1,
        stratify=labels[train_cal],
    )
    return np.sort(train), np.sort(calibration), np.sort(test)


def fingerprint_impression_split(
    impression_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.flatnonzero(np.isin(impression_ids, [0, 1, 2, 3, 4]))
    calibration = np.flatnonzero(impression_ids == 5)
    test = np.flatnonzero(np.isin(impression_ids, [6, 7]))
    return train, calibration, test


def fit_training_scaler(
    flat_images: np.ndarray, train_indices: np.ndarray
) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(flat_images[train_indices])
    transformed = scaler.transform(flat_images).astype(np.float32)
    return scaler, transformed


def save_scaler(path: Path | str, scaler: StandardScaler, train_indices: np.ndarray) -> None:
    np.savez_compressed(
        path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        n_samples_seen=np.asarray([scaler.n_samples_seen_]),
        fitted_train_indices=np.asarray(train_indices, dtype=np.int64),
    )


def validate_disjoint_splits(*splits: Sequence[int]) -> None:
    sets = [set(np.asarray(split, dtype=int).tolist()) for split in splits]
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            if left.intersection(right):
                raise ValueError("Training, calibration and test partitions overlap")


def non_iid_identity_partitions(labels: np.ndarray, n_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(labels))
    class_groups = np.array_split(classes, n_clients)
    return [np.flatnonzero(np.isin(labels, group)) for group in class_groups]


def create_dual_branch_model(
    face_input_dim: int,
    finger_input_dim: int,
    n_face_classes: int,
    n_finger_classes: int,
):
    """Create independent modality branches under one serializable Keras model."""
    import tensorflow as tf

    face_input = tf.keras.Input((face_input_dim,), name="face_input")
    face = tf.keras.layers.Dense(192, activation="relu", name="face_dense_1")(face_input)
    face = tf.keras.layers.BatchNormalization(name="face_bn_1")(face)
    face = tf.keras.layers.Dropout(0.20, name="face_dropout")(face)
    face_embedding = tf.keras.layers.Dense(96, activation="relu", name="face_embedding")(face)
    face_output = tf.keras.layers.Dense(
        n_face_classes, activation="softmax", name="face_output"
    )(face_embedding)

    finger_input = tf.keras.Input((finger_input_dim,), name="finger_input")
    finger = tf.keras.layers.Dense(192, activation="relu", name="finger_dense_1")(finger_input)
    finger = tf.keras.layers.BatchNormalization(name="finger_bn_1")(finger)
    finger = tf.keras.layers.Dropout(0.20, name="finger_dropout")(finger)
    finger_embedding = tf.keras.layers.Dense(96, activation="relu", name="finger_embedding")(finger)
    finger_output = tf.keras.layers.Dense(
        n_finger_classes, activation="softmax", name="finger_output"
    )(finger_embedding)

    return tf.keras.Model(
        inputs={"face_input": face_input, "finger_input": finger_input},
        outputs={"face_output": face_output, "finger_output": finger_output},
        name="do_fmbs_independent_dual_branch",
    )


def compile_dual_branch_model(model, learning_rate: float = 1e-3):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={
            "face_output": "sparse_categorical_crossentropy",
            "finger_output": "sparse_categorical_crossentropy",
        },
        loss_weights={"face_output": 0.5, "finger_output": 0.5},
        metrics={"face_output": ["accuracy"], "finger_output": ["accuracy"]},
    )
    return model


def modality_probabilities(model, modality: str, values: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Run one independent branch while supplying zeros to the unused input."""
    face_dim = int(model.get_layer("face_input").input.shape[-1])
    finger_dim = int(model.get_layer("finger_input").input.shape[-1])
    values = np.asarray(values, dtype=np.float32)
    if modality == "face":
        output = model.predict(
            {"face_input": values, "finger_input": np.zeros((len(values), finger_dim), np.float32)},
            batch_size=batch_size,
            verbose=0,
        )
        key = "face_output"
    elif modality == "fingerprint":
        output = model.predict(
            {"face_input": np.zeros((len(values), face_dim), np.float32), "finger_input": values},
            batch_size=batch_size,
            verbose=0,
        )
        key = "finger_output"
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return np.asarray(output[key] if isinstance(output, dict) else output[0 if modality == "face" else 1])


def claimed_identity_scores(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    genuine = probabilities[np.arange(len(labels)), labels]
    mask = np.ones_like(probabilities, dtype=bool)
    mask[np.arange(len(labels)), labels] = False
    impostor = probabilities[mask]
    probe_ids = np.repeat(np.arange(len(labels)), probabilities.shape[1] - 1)
    return genuine.astype(float), impostor.astype(float), probe_ids


def threshold_curve(genuine: np.ndarray, impostor: np.ndarray) -> list[dict[str, float]]:
    candidates = np.unique(np.concatenate(([0.0], genuine, impostor, [1.0])))
    if len(candidates) > 2500:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 2500))
    rows = []
    for threshold in candidates:
        rows.append(
            {
                "threshold": float(threshold),
                "FAR": float(np.mean(impostor >= threshold)),
                "FRR": float(np.mean(genuine < threshold)),
            }
        )
    return rows


def select_eer_threshold(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, dict[str, float]]:
    rows = threshold_curve(genuine, impostor)
    selected = min(rows, key=lambda row: (abs(row["FAR"] - row["FRR"]), row["FAR"] + row["FRR"]))
    return float(selected["threshold"]), selected


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("A confidence interval requires a positive denominator")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def identity_bootstrap_intervals(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    repetitions: int = 2000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float]]:
    identities = np.unique(labels)
    rng = np.random.default_rng(seed)
    far_values = np.empty(repetitions)
    frr_values = np.empty(repetitions)
    for repetition in range(repetitions):
        sampled = rng.choice(identities, size=len(identities), replace=True)
        chunks = [np.flatnonzero(labels == identity) for identity in sampled]
        selected = np.concatenate(chunks)
        genuine, impostor, _ = claimed_identity_scores(probabilities[selected], labels[selected])
        far_values[repetition] = np.mean(impostor >= threshold)
        frr_values[repetition] = np.mean(genuine < threshold)
    return {
        "FAR_95": [float(value) for value in np.quantile(far_values, [0.025, 0.975])],
        "FRR_95": [float(value) for value in np.quantile(frr_values, [0.025, 0.975])],
    }


def verification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    bootstrap_repetitions: int = 2000,
) -> dict[str, object]:
    genuine, impostor, _ = claimed_identity_scores(probabilities, labels)
    false_accepts = int(np.count_nonzero(impostor >= threshold))
    false_rejects = int(np.count_nonzero(genuine < threshold))
    far = false_accepts / len(impostor)
    frr = false_rejects / len(genuine)
    truth = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    scores = np.concatenate([genuine, impostor])
    fpr, tpr, roc_thresholds = roc_curve(truth, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "threshold": float(threshold),
        "genuine_attempts": int(len(genuine)),
        "impostor_attempts": int(len(impostor)),
        "genuine_accepts": int(len(genuine) - false_rejects),
        "genuine_rejects": false_rejects,
        "impostor_accepts": false_accepts,
        "impostor_rejects": int(len(impostor) - false_accepts),
        "FAR": float(far),
        "FRR": float(frr),
        "TAR": float(1.0 - frr),
        "FAR_wilson_95": list(wilson_interval(false_accepts, len(impostor))),
        "FRR_wilson_95": list(wilson_interval(false_rejects, len(genuine))),
        "identity_cluster_bootstrap": identity_bootstrap_intervals(
            probabilities, labels, threshold, bootstrap_repetitions
        ),
        "ROC_AUC": float(roc_auc_score(truth, scores)),
        "EER": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "EER_threshold": float(roc_thresholds[eer_index]),
    }


def load_modality_data(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def csv_write(path: Path | str, rows: Iterable[dict[str, object]]) -> None:
    import csv

    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

"""Shared scientific utilities for the DO-FMBS experimental pipeline.

The project contains two independent biometric identity spaces: public LFW face
identities and locally generated synthetic fingerprint identities.  Utilities
in this module keep those tasks separate while allowing one two-input model to
train both modality-specific heads.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
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
    "configuration_schema_version": 1,
    "generator_version": "1.1.0",
    "dataset": {"identities": 100, "impressions_per_identity": 8,
                "image_shape_pixels_yx": [96, 96], "output_dtype": "uint8"},
    "randomness": {
        "bit_generator": "PCG64",
        "uniform_interval": "low inclusive, high exclusive",
        "identity_seed": {"base": FINGERPRINT_IDENTITY_SEED, "identity_stride": 1009,
                           "formula": "base + identity_stride * identity_id"},
        "impression_seed": {"base": FINGERPRINT_IMPRESSION_SEED,
                             "identity_stride": 10007, "impression_stride": 101,
                             "formula": "base + identity_stride * identity_id + impression_stride * impression_id",
                             "draw_order": ["orientation_perturbation", "frequency_perturbation", "rotation_angle", "translation_y", "translation_x", "deformation_noise_y", "deformation_noise_x", "contrast", "brightness", "gamma", "blur_sigma", "sensor_noise_sigma", "sensor_noise_array", "contact_center_x", "contact_center_y"]},
        "deformation_noise_distribution": "normal(mean, standard_deviation)",
        "sensor_noise_distribution": "normal(mean, sampled_standard_deviation)",
    },
    "identity_template": {
        "minutiae_count": 5, "minutiae_x_normalized": [-0.62, 0.62],
        "minutiae_y_normalized": [-0.72, 0.72], "minutiae_radius_normalized": [0.10, 0.24],
        "minutiae_phase_shift": [-0.42, 0.42],
        "base_orientation_radians": [-1.10, 1.10], "ridge_frequency_cycles_per_normalized_unit": [5.3, 8.1],
        "core_x_normalized": [-0.25, 0.25], "core_y_normalized": [-0.30, 0.20],
        "curvature": [-0.34, 0.34], "swirl": [-0.22, 0.22],
        "phase_offset_radians": [0.0, 2.0 * math.pi], "warp_x": [0.10, 0.32],
        "warp_y": [0.10, 0.32], "warp_phase_radians": [0.0, 2.0 * math.pi],
        "ellipticity": [0.82, 1.12],
    },
    "coordinate_grid": {"normalized_bounds_xy": [-1.0, 1.0],
                         "complex_step_real_part": 0.0,
                         "sampling": "numpy.mgrid complex-step sample-count semantics; both endpoints included",
                         "axis_order": "yy, xx = mgrid[y, x]; x is columns and y is rows"},
    "ridge_pattern": {
        "pi_value": math.pi,
        "coordinate_warp_amplitude": 0.055,
        "warp_x_angular_frequency_pi": 2.4, "warp_y_angular_frequency_pi": 2.0,
        "minutiae_amplitude": 0.055, "minutiae_gaussian_denominator_factor": 2.0,
        "ridge_phase_frequency_factor": 2.0, "ridge_tanh_scale": 3.2,
        "radial_epsilon": 1e-6,
        "ridge_contrast_center": 0.5,
        "contact_ellipse_outer": 1.12, "contact_ellipse_width": 0.22,
        "background_level": 0.93, "background_gradient": 0.025,
        "ridge_contact_clip": [0.0, 1.0],
    },
    "impression_variation": {
        "orientation_delta_degrees": [-1.5, 1.5],
        "orientation_delta_radians": [-math.radians(1.5), math.radians(1.5)],
        "orientation_formula": "base_orientation + first impression RNG uniform draw",
        "frequency_fractional_delta": [-0.01, 0.01],
        "frequency_fractional_base": 1.0,
        "frequency_formula": "base_frequency * (1 + second impression RNG uniform draw)",
        "assumption": "Conservative researcher-approved bounds, not fitted or performance-optimized parameters.",
        "rotation_angle_degrees": [-1.15, 1.15], "translation_yx_pixels": [-1.1, 1.1],
        "rotation_reshape": False, "rotation_axes_yx": [1, 0],
        "rigid_interpolation_order": 1, "rigid_boundary_mode": "nearest",
        "rigid_boundary_cval": 0.0, "rigid_prefilter": True,
    },
    "deformation": {
        "coarse_grid_shape_yx": [7, 7], "zoom_factor_divisor_yx": [7.0, 7.0],
        "zoom_interpolation_order": 3, "zoom_mode": "constant", "zoom_cval": 0.0,
        "zoom_prefilter": True, "zoom_grid_mode": False,
        "zoom_crop_origin_yx": [0, 0],
        "smoothing_sigma_pixels": 4.0, "smoothing_order": 0,
        "smoothing_mode": "reflect", "smoothing_cval": 0.0,
        "smoothing_truncate": 4.0, "smoothing_radius": None,
        "displacement_standard_deviation_pixels": 0.65,
        "standard_deviation_epsilon": 1e-8, "standard_deviation_ddof": 0,
        "standard_deviation_axis": None, "standard_deviation_keepdims": False,
        "displacement_grid_index_origin": 0,
        "map_coordinates_order": 1, "map_coordinates_mode": "nearest",
        "map_coordinates_cval": 0.0, "map_coordinates_prefilter": True,
    },
    "contact": {
        "ellipse_outer": 1.16, "ellipse_width": 0.18,
        "scale_x": 1.02, "scale_y": 1.08,
        "center_x_y_normalized": [-0.035, 0.035], "background_intensity": 0.94,
        "mask_clip": [0.0, 1.0], "mask_complement_max": 1.0,
    },
    "illumination": {
        "contrast": [0.90, 1.10], "contrast_center": 0.5,
        "brightness": [-0.025, 0.025], "gamma": [0.92, 1.08],
        "pre_blur_clip": [0.0, 1.0],
    },
    "blur": {
        "sigma_pixels": [0.45, 0.82], "gaussian_filter_order": 0,
        "mode": "reflect", "cval": 0.0, "truncate": 4.0, "radius": None,
    },
    "noise": {
        "deformation_mean": 0.0, "deformation_standard_deviation": 1.0,
        "sensor_mean": 0.0, "sensor_standard_deviation": [0.012, 0.030],
    },
    "normalization": {
        "scope": "each completed impression independently",
        "formula": "(pixel - impression_min) / (impression_max - impression_min) * (output_max - output_min) + output_min",
        "output_min": 0, "output_max": 255,
        "working_dtype": "float64",
        "constant_impression": "all-zero uint8 image",
        "final_clipping": "none; min-max scaling bounds finite nonconstant values",
    },
    "quantization": {"method": "numpy.rint", "rounding": "nearest, ties to even",
                     "output_dtype": "uint8", "cast": "after rounding"},
    "parameter_documentation": {
        "units": "Field names declare units (pixels, normalized coordinates, radians, degrees, cycles, or intensity); RNG uniform intervals are [lower, upper).",
        "sections": {
            "dataset": "Identity count, impression count, output raster shape and stored image dtype.",
            "randomness": "Bit generator, seed formulas, random distributions and deterministic draw order.",
            "identity_template": "Identity-stable random ridge geometry and base orientation/frequency ranges.",
            "coordinate_grid": "Normalized sampling bounds, mgrid sample-count convention and axis orientation.",
            "ridge_pattern": "Analytic ridge field, minutiae influence, contrast, and identity-field contact-mask parameters.",
            "impression_variation": "Bounded per-impression orientation/frequency perturbations and rigid transform settings.",
            "deformation": "Coarse displacement generation, zoom/smoothing/interpolation settings and displacement normalization.",
            "contact": "Per-impression contact-mask geometry, translation bounds, clipping, and background intensity.",
            "illumination": "Contrast, brightness, gamma and pre-blur intensity clipping.",
            "blur": "Gaussian blur sigma range and all effective scipy gaussian_filter settings.",
            "noise": "Deformation-field and sensor-noise Gaussian distribution parameters.",
            "normalization": "Per-image min-max range, floating computation dtype and constant-image policy.",
            "quantization": "Rounding algorithm and final integer output dtype.",
        },
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


def _fingerprint_rng(seed: int, config: dict[str, object]) -> np.random.Generator:
    bit_generator_name = config["randomness"]["bit_generator"]
    bit_generator = getattr(np.random, bit_generator_name)
    return np.random.Generator(bit_generator(seed))


def _sample_configured_noise(
    rng: np.random.Generator, distribution: str, mean: float,
    standard_deviation: float, size: tuple[int, ...] | None = None,
) -> np.ndarray | float:
    if distribution not in ("normal(mean, standard_deviation)",
                             "normal(mean, sampled_standard_deviation)"):
        raise ValueError(f"Unsupported configured noise distribution: {distribution}")
    return rng.normal(mean, standard_deviation, size=size)


def identity_seed(finger_id: int, config: dict[str, object] | None = None) -> int:
    cfg = FINGERPRINT_GENERATION_CONFIG if config is None else config
    seed = cfg["randomness"]["identity_seed"]
    return int(seed["base"]) + int(seed["identity_stride"]) * int(finger_id)


def impression_seed(finger_id: int, impression: int,
                    config: dict[str, object] | None = None) -> int:
    cfg = FINGERPRINT_GENERATION_CONFIG if config is None else config
    seed = cfg["randomness"]["impression_seed"]
    return (int(seed["base"]) + int(seed["identity_stride"]) * int(finger_id)
            + int(seed["impression_stride"]) * int(impression))


def create_fingerprint_identity_template(
    finger_id: int, height: int | None = None, width: int | None = None,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create persistent ridge parameters for exactly one synthetic finger."""
    generation = FINGERPRINT_GENERATION_CONFIG if config is None else config
    cfg = generation["identity_template"]
    shape_yx = generation["dataset"]["image_shape_pixels_yx"]
    height = int(shape_yx[0] if height is None else height)
    width = int(shape_yx[1] if width is None else width)
    rng = _fingerprint_rng(identity_seed(finger_id, generation), generation)
    minutiae = []
    for _ in range(int(cfg["minutiae_count"])):
        minutiae.append(
            {
                "x": float(rng.uniform(*cfg["minutiae_x_normalized"])),
                "y": float(rng.uniform(*cfg["minutiae_y_normalized"])),
                "radius": float(rng.uniform(*cfg["minutiae_radius_normalized"])),
                "phase_shift": float(rng.uniform(*cfg["minutiae_phase_shift"])),
            }
        )
    return {
        "finger_id": int(finger_id),
        "identity_seed": identity_seed(finger_id, generation),
        "height": int(height),
        "width": int(width),
        "base_orientation": float(rng.uniform(*cfg["base_orientation_radians"])),
        "ridge_frequency": float(rng.uniform(*cfg["ridge_frequency_cycles_per_normalized_unit"])),
        "core_x": float(rng.uniform(*cfg["core_x_normalized"])),
        "core_y": float(rng.uniform(*cfg["core_y_normalized"])),
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
    config: dict[str, object] | None = None,
) -> np.ndarray:
    generation = FINGERPRINT_GENERATION_CONFIG if config is None else config
    field_cfg = generation["ridge_pattern"]
    grid_cfg = generation["coordinate_grid"]
    height = int(template["height"])
    width = int(template["width"])
    low, high = grid_cfg["normalized_bounds_xy"]
    step_count = grid_cfg["complex_step_real_part"]
    yy, xx = np.mgrid[low:high:complex(step_count, height), low:high:complex(step_count, width)]
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
        + float(template["swirl"]) * radial * polar / field_cfg["pi_value"]
        + float(template["warp_x"]) * field_cfg["coordinate_warp_amplitude"] * np.sin(field_cfg["warp_x_angular_frequency_pi"] * field_cfg["pi_value"] * x + float(template["warp_phase"]))
        + float(template["warp_y"]) * field_cfg["coordinate_warp_amplitude"] * np.sin(field_cfg["warp_y_angular_frequency_pi"] * field_cfg["pi_value"] * y - float(template["warp_phase"]))
    )
    for point in template["minutiae"]:
        distance2 = (xx - point["x"]) ** 2 + (yy - point["y"]) ** 2
        coordinate += point["phase_shift"] * field_cfg["minutiae_amplitude"] * np.exp(-distance2 / (field_cfg["minutiae_gaussian_denominator_factor"] * point["radius"] ** 2))

    frequency = float(template["ridge_frequency"]) * (generation["impression_variation"]["frequency_fractional_base"] + frequency_fractional_delta)
    phase = field_cfg["ridge_phase_frequency_factor"] * field_cfg["pi_value"] * frequency * coordinate + float(template["phase_offset"])
    center = field_cfg["ridge_contrast_center"]
    ridge = center + center * np.cos(phase)
    ridge = center + center * np.tanh((ridge - center) * field_cfg["ridge_tanh_scale"])
    ellipse = (xx / float(template["ellipticity"])) ** 2 + (yy * float(template["ellipticity"])) ** 2
    contact = np.clip((field_cfg["contact_ellipse_outer"] - ellipse) / field_cfg["contact_ellipse_width"], *field_cfg["ridge_contact_clip"])
    background = field_cfg["background_level"] + field_cfg["background_gradient"] * (xx + yy)
    return ridge * contact + background * (1.0 - contact)


def render_fingerprint_impression(
    template: dict[str, object], impression: int,
    config: dict[str, object] | None = None,
) -> np.ndarray:
    """Render a controlled acquisition variation of a persistent identity."""
    generation = FINGERPRINT_GENERATION_CONFIG if config is None else config
    finger_id = int(template["finger_id"])
    rng = _fingerprint_rng(impression_seed(finger_id, impression, generation), generation)
    variation = generation["impression_variation"]
    orientation_delta = float(rng.uniform(*variation["orientation_delta_radians"]))
    frequency_delta = float(rng.uniform(*variation["frequency_fractional_delta"]))
    base = _render_identity_field(template, orientation_delta, frequency_delta, generation)
    height, width = base.shape
    deformation = generation["deformation"]
    contact_cfg = generation["contact"]
    illumination = generation["illumination"]
    blur_cfg = generation["blur"]
    noise_cfg = generation["noise"]

    # Modest rigid acquisition variation.
    rotated = ndimage.rotate(
        base,
        angle=float(rng.uniform(*variation["rotation_angle_degrees"])),
        axes=tuple(variation["rotation_axes_yx"]),
        reshape=variation["rotation_reshape"],
        order=int(variation["rigid_interpolation_order"]),
        mode=variation["rigid_boundary_mode"],
        cval=variation["rigid_boundary_cval"],
        prefilter=variation["rigid_prefilter"],
    )
    shifted = ndimage.shift(
        rotated,
        shift=tuple(float(x) for x in rng.uniform(*variation["translation_yx_pixels"], size=2)),
        order=int(variation["rigid_interpolation_order"]),
        mode=variation["rigid_boundary_mode"],
        cval=variation["rigid_boundary_cval"],
        prefilter=variation["rigid_prefilter"],
    )

    # Smooth sensor/contact deformation; displacement stays below one pixel.
    grid_shape = tuple(int(v) for v in deformation["coarse_grid_shape_yx"])
    deformation_distribution = generation["randomness"]["deformation_noise_distribution"]
    coarse_y = _sample_configured_noise(rng, deformation_distribution,
                                        noise_cfg["deformation_mean"],
                                        noise_cfg["deformation_standard_deviation"], grid_shape)
    coarse_x = _sample_configured_noise(rng, deformation_distribution,
                                        noise_cfg["deformation_mean"],
                                        noise_cfg["deformation_standard_deviation"], grid_shape)
    zoom = (height / deformation["zoom_factor_divisor_yx"][0], width / deformation["zoom_factor_divisor_yx"][1])
    zoom_args = {"order": int(deformation["zoom_interpolation_order"]), "mode": deformation["zoom_mode"],
                 "cval": deformation["zoom_cval"], "prefilter": deformation["zoom_prefilter"],
                 "grid_mode": deformation["zoom_grid_mode"]}
    crop_y, crop_x = deformation["zoom_crop_origin_yx"]
    dy = ndimage.zoom(coarse_y, zoom, **zoom_args)[crop_y:crop_y + height, crop_x:crop_x + width]
    dx = ndimage.zoom(coarse_x, zoom, **zoom_args)[crop_y:crop_y + height, crop_x:crop_x + width]
    smoothing_args = {"sigma": deformation["smoothing_sigma_pixels"],
                      "order": deformation["smoothing_order"], "mode": deformation["smoothing_mode"],
                      "cval": deformation["smoothing_cval"], "truncate": deformation["smoothing_truncate"],
                      "radius": deformation["smoothing_radius"]}
    dy = ndimage.gaussian_filter(dy, **smoothing_args)
    dx = ndimage.gaussian_filter(dx, **smoothing_args)
    std_args = {"axis": deformation["standard_deviation_axis"],
                "ddof": deformation["standard_deviation_ddof"],
                "keepdims": deformation["standard_deviation_keepdims"]}
    dy = deformation["displacement_standard_deviation_pixels"] * dy / (np.std(dy, **std_args) + deformation["standard_deviation_epsilon"])
    dx = deformation["displacement_standard_deviation_pixels"] * dx / (np.std(dx, **std_args) + deformation["standard_deviation_epsilon"])
    grid_origin = deformation["displacement_grid_index_origin"]
    grid_y, grid_x = np.mgrid[grid_origin:grid_origin + height, grid_origin:grid_origin + width]
    image = ndimage.map_coordinates(shifted, [grid_y + dy, grid_x + dx],
                                    order=deformation["map_coordinates_order"],
                                    mode=deformation["map_coordinates_mode"],
                                    cval=deformation["map_coordinates_cval"],
                                    prefilter=deformation["map_coordinates_prefilter"])

    # Pressure, illumination, blur, and sensor noise vary by impression.
    contrast = float(rng.uniform(*illumination["contrast"]))
    brightness = float(rng.uniform(*illumination["brightness"]))
    gamma = float(rng.uniform(*illumination["gamma"]))
    center = illumination["contrast_center"]
    image = np.clip((image - center) * contrast + center + brightness, *illumination["pre_blur_clip"]) ** gamma
    blur_args = {"sigma": float(rng.uniform(*blur_cfg["sigma_pixels"])),
                 "order": blur_cfg["gaussian_filter_order"], "mode": blur_cfg["mode"],
                 "cval": blur_cfg["cval"], "truncate": blur_cfg["truncate"], "radius": blur_cfg["radius"]}
    image = ndimage.gaussian_filter(image, **blur_args)
    sensor_sigma = float(rng.uniform(*noise_cfg["sensor_standard_deviation"]))
    image += _sample_configured_noise(
        rng, generation["randomness"]["sensor_noise_distribution"],
        noise_cfg["sensor_mean"], sensor_sigma, image.shape,
    )

    # A weak shifted contact mask permits partial edge contact without erasing
    # the core ridge field.
    low, high = generation["coordinate_grid"]["normalized_bounds_xy"]
    step_count = generation["coordinate_grid"]["complex_step_real_part"]
    yy, xx = np.mgrid[low:high:complex(step_count, height), low:high:complex(step_count, width)]
    cx, cy = rng.uniform(*contact_cfg["center_x_y_normalized"], size=2)
    mask = np.clip((contact_cfg["ellipse_outer"] - ((xx - cx) / contact_cfg["scale_x"]) ** 2 - ((yy - cy) / contact_cfg["scale_y"]) ** 2) / contact_cfg["ellipse_width"], *contact_cfg["mask_clip"])
    image = image * mask + contact_cfg["background_intensity"] * (contact_cfg["mask_complement_max"] - mask)
    return minmax_to_uint8(image, generation)


def minmax_to_uint8(image: np.ndarray, config: dict[str, object] | None = None) -> np.ndarray:
    """Scale one impression's observed range linearly to the 8-bit range."""
    generation = FINGERPRINT_GENERATION_CONFIG if config is None else config
    normalization = generation["normalization"]
    quantization = generation["quantization"]
    values = np.asarray(image, dtype=np.dtype(normalization["working_dtype"]))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum == minimum and normalization["constant_impression"] == "all-zero uint8 image":
        return np.zeros(values.shape, dtype=np.dtype(quantization["output_dtype"]))
    if maximum == minimum:
        raise ValueError("Unsupported constant-impression normalization policy")
    scaled = ((values - minimum) / (maximum - minimum)
              * (normalization["output_max"] - normalization["output_min"])
              + normalization["output_min"])
    if quantization["method"] != "numpy.rint":
        raise ValueError(f"Unsupported quantization method: {quantization['method']}")
    rounded = np.rint(scaled)
    return rounded.astype(np.dtype(quantization["output_dtype"]))


def generate_fingerprint_dataset(
    identities: int | None = None, impressions: int | None = None,
    height: int | None = None, width: int | None = None,
    config: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    generation = FINGERPRINT_GENERATION_CONFIG if config is None else config
    dataset_cfg = generation["dataset"]
    identities = int(dataset_cfg["identities"] if identities is None else identities)
    impressions = int(dataset_cfg["impressions_per_identity"] if impressions is None else impressions)
    height = int(dataset_cfg["image_shape_pixels_yx"][0] if height is None else height)
    width = int(dataset_cfg["image_shape_pixels_yx"][1] if width is None else width)
    images: list[np.ndarray] = []
    labels: list[int] = []
    impression_ids: list[int] = []
    templates: list[dict[str, object]] = []
    for finger_id in range(identities):
        template = create_fingerprint_identity_template(finger_id, height, width, generation)
        templates.append(template)
        for impression in range(impressions):
            images.append(render_fingerprint_impression(template, impression, generation))
            labels.append(finger_id)
            impression_ids.append(impression)
    return (
        np.asarray(images, dtype=np.dtype(dataset_cfg["output_dtype"])),
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

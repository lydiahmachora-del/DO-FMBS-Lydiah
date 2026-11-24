"""Versioned configuration schemas and validation for interactive phase runs."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


CONFIG_VERSION = "1.0.0"

# Only controls with a concrete backend consumer are exposed here. Ranges are
# also used by the Streamlit form so UI constraints and validation stay aligned.
PHASE_SCHEMAS: dict[int, dict[str, dict[str, Any]]] = {
    1: {
        "face_min_faces_per_person": {"default": 20, "type": "int", "min": 2, "max": 100, "unit": "images/identity", "label": "LFW minimum images per identity", "description": "Controls which LFW identities are included."},
        "face_resize": {"default": 0.4, "type": "float", "min": 0.1, "max": 1.0, "unit": "scale", "label": "LFW image resize", "description": "Image scale passed to scikit-learn's LFW loader."},
        "face_test_fraction": {"default": 0.15, "type": "float", "min": 0.05, "max": 0.4, "unit": "fraction", "label": "Face test fraction", "description": "Held-out face test share."},
        "face_calibration_fraction": {"default": 0.15, "type": "float", "min": 0.05, "max": 0.4, "unit": "fraction", "label": "Face calibration fraction", "description": "Calibration share, taken from the complete face dataset."},
        "fingerprint_identities": {"default": 100, "type": "int", "min": 10, "max": 100, "unit": "identities", "label": "Synthetic fingerprint identities", "description": "Number of identity-persistent generated templates."},
        "fingerprint_impressions": {"default": 8, "type": "int", "min": 3, "max": 12, "unit": "impressions/identity", "label": "Impressions per identity", "description": "Number of generated samples per fingerprint identity."},
        "fingerprint_train_impressions": {"default": 5, "type": "int", "min": 1, "max": 10, "unit": "impressions/identity", "label": "Fingerprint training impressions", "description": "First consecutive impressions assigned to training."},
        "fingerprint_calibration_impressions": {"default": 1, "type": "int", "min": 1, "max": 5, "unit": "impressions/identity", "label": "Fingerprint calibration impressions", "description": "Next consecutive impressions assigned to calibration."},
        "clients": {"default": 5, "type": "int", "min": 2, "max": 10, "unit": "clients", "label": "Federated clients", "description": "Identity-exclusive client partitions prepared for Phase 2."},
        "split_seed": {"default": 4201, "type": "int", "min": 0, "max": 2147483647, "unit": "seed", "label": "Face split seed", "description": "Seed for face stratification, matching the phase runner's research default."},
        "client_seed": {"default": 5201, "type": "int", "min": 0, "max": 2147483647, "unit": "seed", "label": "Client partition seed", "description": "Seed for identity-exclusive client allocation."},
    },
    2: {
        "rounds": {"default": 12, "type": "int", "min": 1, "max": 50, "unit": "rounds", "label": "Communication rounds", "description": "Global aggregation rounds."},
        "local_epochs": {"default": 2, "type": "int", "min": 1, "max": 20, "unit": "epochs/round", "label": "Local epochs", "description": "Training epochs performed by each selected client per round."},
        "batch_size": {"default": 64, "type": "int", "min": 8, "max": 512, "unit": "samples", "label": "Batch size", "description": "Local optimizer mini-batch size."},
        "learning_rate": {"default": 0.001, "type": "float", "min": 0.00001, "max": 0.1, "unit": "Adam step size", "label": "Learning rate", "description": "Adam learning rate used by local client models."},
        "seed": {"default": 42, "type": "int", "min": 0, "max": 2147483647, "unit": "seed", "label": "Training seed", "description": "Global and per-client sampling seed."},
    },
    3: {
        "epsilons": {"default": [0.01, 0.05, 0.10, 0.20, 0.30], "type": "float_list", "min": 0.0, "max": 1.0, "unit": "standardized input units", "label": "Attack epsilon values", "description": "Shared FGSM/PGD L-infinity perturbation bounds."},
        "pgd_iterations": {"default": 10, "type": "int", "min": 1, "max": 100, "unit": "steps", "label": "PGD iterations", "description": "Projected-gradient updates per attack."},
        "pgd_step_size": {"default": 0.01, "type": "float", "min": 0.0001, "max": 1.0, "unit": "standardized input units/step", "label": "PGD step size", "description": "Step length used before projection onto the epsilon ball."},
        "poison_rate": {"default": 0.10, "type": "float", "min": 0.01, "max": 0.5, "unit": "fraction", "label": "Label-flipping rate", "description": "Share of synchronized training records whose labels are flipped."},
        "poison_epochs": {"default": 3, "type": "int", "min": 1, "max": 20, "unit": "epochs", "label": "Poisoning fine-tune epochs", "description": "Epochs used to measure the label-flipping scenario."},
        "max_test_samples": {"default": 0, "type": "int", "min": 0, "max": 10000, "unit": "samples (0 = all)", "label": "Attack evaluation sample cap", "description": "Resource cap for evasion tests; zero evaluates the full test set."},
        "seed": {"default": 42, "type": "int", "min": 0, "max": 2147483647, "unit": "seed", "label": "Attack seed", "description": "Controls membership sampling, poisoning selection, and model determinism."},
    },
    4: {
        "epochs": {"default": 10, "type": "int", "min": 1, "max": 50, "unit": "epochs", "label": "Defense training epochs", "description": "Fine-tuning epochs per defense condition."},
        "batch_size": {"default": 64, "type": "int", "min": 8, "max": 512, "unit": "samples", "label": "Batch size", "description": "Mini-batch size for defense fine-tuning."},
        "noise_stddev": {"default": 0.0003, "type": "float", "min": 0.0, "max": 0.1, "unit": "gradient units", "label": "Gaussian gradient-noise standard deviation", "description": "Noise added to gradients; this is DP-inspired and has no formal privacy guarantee."},
        "adversarial_epsilon": {"default": 0.05, "type": "float", "min": 0.0, "max": 1.0, "unit": "standardized input units", "label": "Adversarial training epsilon", "description": "FGSM perturbation magnitude for adversarial training and evaluation."},
        "learning_rate": {"default": 0.0005, "type": "float", "min": 0.00001, "max": 0.1, "unit": "Adam step size", "label": "Fine-tuning learning rate", "description": "Adam learning rate for defense model updates."},
        "seed": {"default": 42, "type": "int", "min": 0, "max": 2147483647, "unit": "seed", "label": "Defense seed", "description": "Controls deterministic model and defense randomness."},
    },
    5: {
        "bootstrap_repetitions": {"default": 2000, "type": "int", "min": 100, "max": 20000, "unit": "identity resamples", "label": "Bootstrap repetitions", "description": "Identity-cluster bootstrap repetitions for verification intervals."},
        "robustness_epsilons": {"default": [0.0, 0.05, 0.10, 0.15, 0.20], "type": "float_list", "min": 0.0, "max": 1.0, "unit": "standardized input units", "label": "Robustness epsilon values", "description": "FGSM evaluation levels on the face branch."},
        "evaluation_sample_count": {"default": 0, "type": "int", "min": 0, "max": 10000, "unit": "samples per partition (0 = all)", "label": "Evaluation sample cap", "description": "Deterministic cap on calibration and test samples; zero uses all available samples."},
        "seed": {"default": 6201, "type": "int", "min": 0, "max": 2147483647, "unit": "seed", "label": "Evaluation/bootstrap seed", "description": "Controls deterministic evaluation subsampling and identity-bootstrap randomness; the bootstrap default matches the research runner."},
    },
    6: {
        "rounds": {"default": 12, "type": "int", "min": 1, "max": 100, "unit": "controller rounds", "label": "Optimization rounds", "description": "Number of sequential threat states evaluated by the dynamic controller."},
        "candidates_per_round": {"default": 63, "type": "int", "min": 3, "max": 300, "unit": "candidate policies/round", "label": "Candidate policies per round", "description": "Number of defense-intensity/client-count combinations evaluated per controller round."},
        "target_y": {"default": 0.60, "type": "float", "min": 0.0, "max": 1.0, "unit": "response score", "label": "Minimum response target", "description": "Minimum acceptable predicted response."},
        "preferred_target_y": {"default": 0.70, "type": "float", "min": 0.0, "max": 1.0, "unit": "response score", "label": "Preferred response target", "description": "Preferred predicted response target."},
        "defense_cost_weight": {"default": 0.60, "type": "float", "min": 0.0, "max": 10.0, "unit": "cost coefficient", "label": "Defense cost weight", "description": "Cost coefficient for defense intensity."},
        "participation_cost_weight": {"default": 0.40, "type": "float", "min": 0.0, "max": 10.0, "unit": "cost coefficient", "label": "Participation cost weight", "description": "Cost coefficient for client participation."},
        "defense_switch_cost_weight": {"default": 0.15, "type": "float", "min": 0.0, "max": 10.0, "unit": "cost coefficient", "label": "Defense switching cost weight", "description": "Cost coefficient for change in defense between rounds."},
        "participation_switch_cost_weight": {"default": 0.10, "type": "float", "min": 0.0, "max": 10.0, "unit": "cost coefficient", "label": "Participation switching cost weight", "description": "Cost coefficient for change in participating clients."},
        "seed": {"default": 42, "type": "int", "min": 0, "max": 2147483647, "unit": "seed", "label": "Simulation seed", "description": "Seed for the synthetic response observations."},
    },
}


def phase_defaults(phase: int) -> dict[str, Any]:
    return {name: deepcopy(spec["default"]) for name, spec in PHASE_SCHEMAS[phase].items()}


def defaults_for_all_phases() -> dict[str, dict[str, Any]]:
    return {f"phase{phase}": phase_defaults(phase) for phase in PHASE_SCHEMAS}


def validate_phase_config(phase: int, values: dict[str, Any]) -> dict[str, Any]:
    schema = PHASE_SCHEMAS[phase]
    unknown = set(values) - set(schema)
    if unknown:
        raise ValueError(f"Phase {phase} has unsupported parameter(s): {', '.join(sorted(unknown))}")
    checked = phase_defaults(phase)
    checked.update(values)
    for name, spec in schema.items():
        value = checked[name]
        kind = spec["type"]
        if kind == "int" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name} must be an integer")
        if kind == "float" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{name} must be numeric")
        if kind == "float_list":
            if not isinstance(value, list) or not value or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
                raise ValueError(f"{name} must be a non-empty list of numbers")
            if value != sorted(set(value)):
                raise ValueError(f"{name} values must be unique and in ascending order")
            if any(not spec["min"] <= float(v) <= spec["max"] for v in value):
                raise ValueError(f"{name} values must be between {spec['min']} and {spec['max']}")
            checked[name] = [float(v) for v in value]
            continue
        if not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"{name} must be between {spec['min']} and {spec['max']} {spec['unit']}")
        if kind == "float":
            checked[name] = float(value)
    if phase == 1:
        if checked["face_test_fraction"] + checked["face_calibration_fraction"] >= 0.8:
            raise ValueError("Face test plus calibration fractions must leave at least 20% for training")
        if checked["fingerprint_train_impressions"] + checked["fingerprint_calibration_impressions"] >= checked["fingerprint_impressions"]:
            raise ValueError("Fingerprint train and calibration counts must leave at least one test impression")
        if checked["fingerprint_identities"] < checked["clients"]:
            raise ValueError("Fingerprint identity count must be at least the client count")
    if phase == 3 and checked["pgd_step_size"] > max(checked["epsilons"]):
        raise ValueError("PGD step size cannot exceed the largest configured epsilon")
    if phase == 6 and checked["preferred_target_y"] < checked["target_y"]:
        raise ValueError("Preferred response target must be at least the minimum target")
    return checked


def changed_parameters(phase: int, values: dict[str, Any]) -> list[str]:
    defaults = phase_defaults(phase)
    return [key for key, value in values.items() if value != defaults[key]]


def cli_arguments(phase: int, config: dict[str, Any]) -> list[str]:
    values = validate_phase_config(phase, config)
    mappings = {
        1: {"face_min_faces_per_person": "--face-min-faces-per-person", "face_resize": "--face-resize", "face_test_fraction": "--face-test-fraction", "face_calibration_fraction": "--face-calibration-fraction", "fingerprint_identities": "--fingerprint-identities", "fingerprint_impressions": "--fingerprint-impressions", "fingerprint_train_impressions": "--fingerprint-train-impressions", "fingerprint_calibration_impressions": "--fingerprint-calibration-impressions", "clients": "--clients", "split_seed": "--split-seed", "client_seed": "--client-seed"},
        2: {"rounds": "--rounds", "local_epochs": "--local-epochs", "batch_size": "--batch-size", "learning_rate": "--learning-rate", "seed": "--seed"},
        3: {"epsilons": "--epsilons", "pgd_iterations": "--pgd-iterations", "pgd_step_size": "--pgd-step-size", "poison_rate": "--poison-rate", "poison_epochs": "--poison-epochs", "max_test_samples": "--max-test-samples", "seed": "--seed"},
        4: {"epochs": "--epochs", "batch_size": "--batch-size", "noise_stddev": "--noise-stddev", "adversarial_epsilon": "--adversarial-epsilon", "learning_rate": "--learning-rate", "seed": "--seed"},
        5: {"bootstrap_repetitions": "--bootstrap-repetitions", "robustness_epsilons": "--robustness-epsilons", "evaluation_sample_count": "--evaluation-sample-count", "seed": "--seed"},
        6: {"rounds": "--rounds", "candidates_per_round": "--candidates-per-round", "target_y": "--target-y", "preferred_target_y": "--preferred-target-y", "defense_cost_weight": "--defense-cost-weight", "participation_cost_weight": "--participation-cost-weight", "defense_switch_cost_weight": "--defense-switch-cost-weight", "participation_switch_cost_weight": "--participation-switch-cost-weight", "seed": "--seed"},
    }[phase]
    args: list[str] = []
    for key, flag in mappings.items():
        value = values[key]
        if isinstance(value, list):
            value = ",".join(f"{float(item):g}" for item in value)
        args.extend([flag, str(value)])
    return args

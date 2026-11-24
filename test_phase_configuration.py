from __future__ import annotations

import unittest
import numpy as np

from phase_configuration import (
    CONFIG_VERSION, cli_arguments, defaults_for_all_phases, phase_defaults,
    validate_phase_config,
)
from phase3_adversarial_attacks import exact_linf_projection
from do_fmbs_core import fingerprint_impression_split
from scripts.constrained_dynamic_model import dynamic_schedule, fit_equation, generate_observations


class PhaseConfigurationTests(unittest.TestCase):
    def test_all_research_defaults_validate(self):
        for phase in range(1, 7):
            with self.subTest(phase=phase):
                self.assertEqual(validate_phase_config(phase, phase_defaults(phase)), phase_defaults(phase))
        self.assertEqual(CONFIG_VERSION, "1.0.0")

    def test_phase_one_partition_counts_must_leave_test_impressions(self):
        config = phase_defaults(1)
        config["fingerprint_train_impressions"] = 7
        config["fingerprint_calibration_impressions"] = 1
        with self.assertRaisesRegex(ValueError, "leave at least one test impression"):
            validate_phase_config(1, config)

    def test_phase_three_step_size_must_fit_epsilon_budget(self):
        config = phase_defaults(3)
        config["pgd_step_size"] = 0.4
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            validate_phase_config(3, config)

    def test_phase_six_response_targets_are_ordered(self):
        config = phase_defaults(6)
        config["target_y"] = 0.8
        with self.assertRaisesRegex(ValueError, "at least the minimum"):
            validate_phase_config(6, config)

    def test_backend_cli_mapping_contains_modified_values(self):
        config = phase_defaults(2)
        config["rounds"] = 3
        config["learning_rate"] = 0.002
        args = cli_arguments(2, config)
        self.assertEqual(args[args.index("--rounds") + 1], "3")
        self.assertEqual(args[args.index("--learning-rate") + 1], "0.002")

    def test_unknown_backend_parameters_are_rejected(self):
        config = phase_defaults(4)
        config["pretend_privacy_epsilon"] = 1.0
        with self.assertRaisesRegex(ValueError, "unsupported parameter"):
            validate_phase_config(4, config)

    def test_default_bundle_contains_all_six_phases(self):
        self.assertEqual(set(defaults_for_all_phases()), {f"phase{i}" for i in range(1, 7)})

    def test_attack_projection_enforces_exact_linf_bound_after_float32_rounding(self):
        origin = np.asarray([[0.1234567, -0.654321]], dtype=np.float32)
        candidate = origin + np.asarray([[0.01001, -0.01001]], dtype=np.float32)
        projected = exact_linf_projection(candidate, origin, 0.01)
        self.assertLessEqual(float(np.max(np.abs(projected.astype(np.float64) - origin.astype(np.float64)))), 0.01)

    def test_fingerprint_impression_parameters_change_actual_partition_counts(self):
        ids = np.tile(np.arange(7), 2)
        train, calibration, test = fingerprint_impression_split(ids, 4, 1)
        self.assertEqual((len(train), len(calibration), len(test)), (8, 2, 4))

    def test_dynamic_round_and_candidate_parameters_change_schedule_output(self):
        fit = fit_equation(generate_observations(seed=42))
        schedule = dynamic_schedule(fit, rounds=2, candidates_per_round=6)
        self.assertEqual(len(schedule), 2)
        self.assertEqual([row["candidate_policies_evaluated"] for row in schedule], [6, 6])
        changed_costs = dynamic_schedule(
            fit, rounds=1, candidates_per_round=6,
            cost_weights={"defense": 0.7, "participation": 0.3,
                          "defense_switch": 0.15, "participation_switch": 0.1},
        )
        self.assertNotEqual(schedule[0]["control_cost"], changed_costs[0]["control_cost"])


if __name__ == "__main__":
    unittest.main()

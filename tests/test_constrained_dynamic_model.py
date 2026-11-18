import json
import unittest
from pathlib import Path

import numpy as np

from scripts.constrained_dynamic_model import (
    CLIENTS_TOTAL, MIN_PARTICIPATING_CLIENTS, TARGET_Y, acceptance_ranges,
    constraint_checks, cost_weight_sensitivity, defense_configuration, dynamic_schedule, fit_equation,
    generate_observations, gs_from_clients, membership_advantage, normalize_ai,
    optimize_controls, predict, stress_scenarios,
)


class ConstrainedDynamicModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate_observations()
        cls.fit = fit_equation(cls.rows)

    def test_balanced_design_count_and_domains(self):
        self.assertEqual(len(self.rows), 700)
        for name in ("AI", "PA", "DM", "GS", "Y"):
            values = np.array([row[name] for row in self.rows])
            self.assertGreaterEqual(values.min(), 0)
            self.assertLessEqual(values.max(), 1)

    def test_ai_normalization(self):
        self.assertAlmostEqual(normalize_ai(0.10), 1/3)
        self.assertEqual(normalize_ai(0.30), 1.0)
        with self.assertRaises(ValueError):
            normalize_ai(0.31)

    def test_pa_and_gs_definitions(self):
        self.assertAlmostEqual(membership_advantage(0.51), 0.02)
        self.assertEqual(gs_from_clients(1), 0.0)
        self.assertEqual(gs_from_clients(3), 0.5)
        self.assertEqual(gs_from_clients(5), 1.0)

    def test_coefficients_obey_constraints(self):
        b = dict(zip(self.fit.names, self.fit.coefficients))
        self.assertLess(b["B_AI"], 0)
        self.assertLess(b["B_PA"], 0)
        self.assertGreater(b["B_DM"], 0)
        self.assertGreater(b["B_GS"], 0)
        self.assertTrue(all(row["satisfied"] for row in constraint_checks(self.fit)))
        self.assertTrue(all(row["slack"] >= 0 for row in constraint_checks(self.fit)))
        self.assertTrue(self.fit.optimization_success)
        self.assertEqual(self.fit.active_constraints, [])
        self.assertLessEqual(self.fit.maximum_constrained_ols_difference, 1e-8)

    def test_fit_quality(self):
        self.assertGreater(self.fit.r_squared, 0.90)
        self.assertGreater(self.fit.cross_validation_r2_mean, 0.89)
        self.assertLess(self.fit.rmse, 0.06)
        self.assertTrue(all(value < 2.0 for value in self.fit.vifs.values()))

    def test_prediction_arithmetic(self):
        b = self.fit.coefficients
        expected = b[0] + b[1]*0.2 + b[2]*0.1 + b[3]*0.8 + b[4]*1.0
        self.assertAlmostEqual(predict(self.fit, 0.2, 0.1, 0.8, 1.0), expected)

    def test_optimizer_bounds_and_target(self):
        result = optimize_controls(self.fit, 0.2, 0.1, 0.3, 0.8)
        self.assertGreaterEqual(result["DM"], 0)
        self.assertLessEqual(result["DM"], 1)
        self.assertIn(result["participating_clients"], range(MIN_PARTICIPATING_CLIENTS, CLIENTS_TOTAL+1))
        if result["status"] in {"preferred_target_met", "minimum_target_met"}:
            self.assertGreaterEqual(result["predicted_Y"], TARGET_Y)

    def test_defense_intensity_mapping(self):
        maximum = defense_configuration(1.0)
        self.assertAlmostEqual(maximum["adversarial_fraction"], 0.50)
        self.assertAlmostEqual(maximum["training_attack_epsilon"], 0.05)
        self.assertAlmostEqual(maximum["gradient_noise_std"], 0.0003)

    def test_cost_weight_sensitivity_is_reported(self):
        rows = cost_weight_sensitivity(self.fit)
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["profile"] for row in rows}, {
            "balanced", "defense_sparing", "participation_sparing"
        })
        self.assertTrue(all(row["predicted_Y"] >= TARGET_Y for row in rows))

    def test_dynamic_schedule_has_twelve_rounds(self):
        rows = dynamic_schedule(self.fit)
        self.assertEqual(len(rows), 12)
        self.assertEqual([row["round"] for row in rows], list(range(1,13)))
        self.assertGreater(len({row["DM"] for row in rows}), 1)
        self.assertEqual(rows[-1]["epsilon"], 0.30)
        self.assertEqual(rows[-1]["AI"], 1.0)
        self.assertEqual(rows[-1]["deployment_decision"], "accept")
        self.assertGreaterEqual(rows[-1]["predicted_Y"], TARGET_Y)

    def test_ranges_cover_all_variables(self):
        self.assertEqual({row["variable"] for row in acceptance_ranges()}, {"AI","PA","DM","GS","Y"})

    def test_outside_range_scenario_is_rejected(self):
        rows = stress_scenarios(self.fit)
        critical = next(row for row in rows if row["scenario"] == "modeled_critical")
        outside = next(row for row in rows if row["scenario"] == "out_of_domain")
        self.assertEqual(critical["deployment_decision"], "reject_or_defer")
        self.assertLess(critical["predicted_Y"], TARGET_Y)
        self.assertEqual(outside["status"], "out_of_domain")
        self.assertIsNone(outside["predicted_Y"])

    def test_predict_refuses_extrapolation(self):
        with self.assertRaises(ValueError):
            predict(self.fit, 1.01, 0.1, 0.5, 0.5)
        with self.assertRaises(ValueError):
            predict(self.fit, 0.2, 0.51, 0.5, 0.5)


if __name__ == "__main__":
    unittest.main()

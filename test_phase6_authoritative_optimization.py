"""Focused controller and safety tests for the final Phase 6 workflow."""
from dataclasses import replace
import unittest

import numpy as np

from phase6_authoritative_optimization import _candidate_rows, _choose, _safety_evidence
from scripts.constrained_dynamic_model import (
    COST_WEIGHTS, PREFERRED_Y, TARGET_Y, fit_equation, generate_observations,
    gs_from_clients, normalize_ai, optimize_controls, predict,
)


class Phase6AuthoritativeControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit = fit_equation(generate_observations(42))

    def test_exact_700_observation_design_and_coefficients(self):
        self.assertEqual(len(generate_observations(42)), 700)
        self.assertAlmostEqual(self.fit.r_squared, .9680638787210141, places=12)
        self.assertTrue(self.fit.optimization_success)

    def test_client_mapping_and_deployment_grid_yield_63_candidates(self):
        self.assertEqual([gs_from_clients(k) for k in (3, 4, 5)], [.5, .75, 1.0])
        rows = _candidate_rows(self.fit, 1/3, .3, .3, .75)
        self.assertEqual(len(rows), 63)
        self.assertEqual(len({row["DM"] for row in rows}), 21)
        self.assertEqual({row["K"] for row in rows}, {3, 4, 5})

    def test_cost_components_recompose_total_cost(self):
        item = _candidate_rows(self.fit, .333333333333, .3, .3, .75)[0]
        total = (COST_WEIGHTS["defense"]*item["DM"]**2
                 + COST_WEIGHTS["participation"]*item["GS"]**2
                 + COST_WEIGHTS["defense_switch"]*(item["DM"]-.3)**2
                 + COST_WEIGHTS["participation_switch"]*(item["GS"]-.75)**2)
        self.assertAlmostEqual(item["total_cost"], total)

    def test_preferred_policy_selects_lowest_cost_eligible_candidate(self):
        rows = _candidate_rows(self.fit, .1666666666667, .1, .3, .75)
        selected, status = _choose(rows)
        best = min((r for r in rows if r["predicted_Y"] >= PREFERRED_Y),
                   key=lambda r: (r["total_cost"], -r["predicted_Y"], r["DM"], r["GS"], r["K"]))
        self.assertEqual(status, "preferred_target_met")
        self.assertEqual((selected["DM"], selected["K"]), (best["DM"], best["K"]))

    def test_managed_policy_selected_when_no_preferred_candidate(self):
        managed_fit = replace(self.fit, coefficients=[.50, 0.0, 0.0, .05, .10])
        rows = _candidate_rows(managed_fit, 0.0, 0.0, .3, .75)
        self.assertFalse(any(r["predicted_Y"] >= PREFERRED_Y for r in rows))
        selected, status = _choose(rows)
        best = min((r for r in rows if r["predicted_Y"] >= TARGET_Y),
                   key=lambda r: (r["total_cost"], -r["predicted_Y"], r["DM"], r["GS"], r["K"]))
        self.assertEqual(status, "minimum_target_met")
        self.assertEqual((selected["DM"], selected["K"]), (best["DM"], best["K"]))

    def test_valid_in_domain_critical_scenario_is_rejected(self):
        result = optimize_controls(self.fit, 1.0, .5, .3, .75)
        self.assertEqual(result["status"], "maximum_controls_target_unmet")
        self.assertLess(result["predicted_Y"], TARGET_Y)

    def test_domain_boundaries_are_accepted_and_outside_inputs_rejected(self):
        self.assertEqual(normalize_ai(0.0), 0.0)
        self.assertEqual(normalize_ai(.30), 1.0)
        self.assertTrue(np.isfinite(predict(self.fit, 0.0, 0.0, 0.0, .5)))
        self.assertTrue(np.isfinite(predict(self.fit, 1.0, .5, 1.0, 1.0)))
        for epsilon in (-.001, .300001, float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                normalize_ai(epsilon)
        for pa in (-.001, .500001, float("nan"), float("inf")):
            with self.subTest(pa=pa), self.assertRaises(ValueError):
                predict(self.fit, .2, pa, .5, .75)
        for controls in ((float("nan"), .1, .5, .75), (.2, .1, .5, float("inf"))):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                predict(self.fit, *controls)

    def test_all_required_safety_branches_have_real_decision_evidence(self):
        evidence = _safety_evidence(self.fit)
        self.assertEqual(evidence["status"], "PASS")
        self.assertTrue(all(evidence["branch_results"].values()))
        self.assertEqual(len(evidence["invalid_inputs_rejected"]), 10)


if __name__ == "__main__":
    unittest.main()

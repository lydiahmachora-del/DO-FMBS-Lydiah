import csv
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "simulation_evidence" / "constrained_dynamic_model"
REPORTS = ROOT / "docs" / "constrained_dynamic_model"


def read_csv(name):
    with (EVIDENCE / name).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Phase6ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit = json.loads((EVIDENCE / "model_fit.json").read_text(encoding="utf-8"))

    def test_coefficient_artifact_matches_model_fit(self):
        rows = read_csv("coefficient_estimates.csv")
        self.assertEqual([row["parameter"] for row in rows], self.fit["names"])
        estimates = np.array([float(row["estimate"]) for row in rows])
        np.testing.assert_allclose(estimates, self.fit["coefficients"], atol=1e-12)
        self.assertLess(self.fit["maximum_coefficient_recovery_error"], 0.01)
        self.assertIn("synthetic", self.fit["scope_statement"].lower())

    def test_constraint_artifact_distinguishes_fitted_and_evaluated_checks(self):
        rows = read_csv("constraint_checks.csv")
        self.assertEqual({row["constraint_type"] for row in rows}, {
            "fitted sign restriction", "evaluated hypothesis"
        })
        self.assertTrue(all(row["satisfied"].lower() == "true" for row in rows))

    def test_schedule_enforces_deployment_gates(self):
        rows = read_csv("dynamic_optimization_schedule.csv")
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertGreaterEqual(int(row["participating_clients"]), 3)
            if row["deployment_decision"] == "accept":
                self.assertGreaterEqual(float(row["predicted_Y"]), self.fit["target_Y"])

    def test_out_of_domain_stress_has_no_prediction(self):
        rows = read_csv("stress_test_scenarios.csv")
        outside = next(row for row in rows if row["scenario"] == "out_of_domain")
        self.assertEqual(outside["status"], "out_of_domain")
        self.assertEqual(outside["predicted_Y"], "")
        self.assertEqual(outside["deployment_decision"], "reject_or_defer")

    def test_cost_sensitivity_records_three_policies(self):
        rows = read_csv("cost_weight_sensitivity.csv")
        self.assertEqual({row["profile"] for row in rows}, {
            "balanced", "defense_sparing", "participation_sparing"
        })
        self.assertTrue(all(float(row["predicted_Y"]) >= self.fit["target_Y"] for row in rows))

    def test_app_has_no_metric_to_coefficient_substitution(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("-abs(results.get", source)
        self.assertNotIn("0.281,", source)
        self.assertIn("load_dynamic_model_results", source)

    def test_professional_reports_exist(self):
        markdown = REPORTS / "DO_FMBS_CONSTRAINED_DYNAMIC_MODEL_REPORT.md"
        docx = REPORTS / "DO_FMBS_CONSTRAINED_DYNAMIC_MODEL_REPORT.docx"
        pdf = REPORTS / "DO_FMBS_CONSTRAINED_DYNAMIC_MODEL_REPORT.pdf"
        for path in (markdown, docx, pdf):
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 10_000)
        self.assertTrue(docx.read_bytes().startswith(b"PK"))
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
        report_text = markdown.read_text(encoding="utf-8")
        self.assertIn(f"{self.fit['coefficients'][0]:.6f}", report_text)
        self.assertIn("GS=(K−1)/4", report_text)
        self.assertIn("Out of domain", report_text)
        self.assertIn("simulation estimates", report_text)


if __name__ == "__main__":
    unittest.main()

import csv
import json
import math
import unittest
from pathlib import Path

import numpy as np

from scripts.measure_operational_biometric_metrics import (
    DEPLOYMENT_FAR_LIMIT,
    DEPLOYMENT_FRR_LIMIT,
    EPSILONS,
    identity_split,
    select_threshold,
    wilson,
)


ROOT = Path(__file__).resolve().parents[1]
ISOLATED_RUN = ROOT / "runtime_recording" / "full_pipeline_20260809"
DATA_DIR = ISOLATED_RUN / "do_fmbs_data"
EVIDENCE_DIR = (
    ROOT
    / "runtime_metrics_evidence"
    / "recorded_full_pipeline_biometric_evaluation"
)


def read_csv(name: str) -> list[dict[str, str]]:
    with (EVIDENCE_DIR / name).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class OperationalBiometricHelperTests(unittest.TestCase):
    def test_identity_split_is_deterministic_complete_and_identity_stratified(self):
        with np.load(DATA_DIR / "face_data.npz", allow_pickle=False) as data:
            labels = data["y_test"].astype(np.int64)

        calibration, evaluation = identity_split(labels)
        calibration_again, evaluation_again = identity_split(labels)

        np.testing.assert_array_equal(calibration, calibration_again)
        np.testing.assert_array_equal(evaluation, evaluation_again)
        self.assertEqual(len(calibration), 291)
        self.assertEqual(len(evaluation), 314)
        self.assertEqual(len(np.unique(labels[calibration])), 62)
        self.assertEqual(len(np.unique(labels[evaluation])), 62)
        self.assertEqual(set(calibration).intersection(evaluation), set())
        self.assertEqual(
            set(calibration).union(evaluation),
            set(range(len(labels))),
        )

    def test_wilson_interval_and_threshold_rule(self):
        low, high = wilson(15, 100)
        self.assertLess(low, 0.15)
        self.assertGreater(high, 0.15)
        self.assertTrue(0.0 <= low <= high <= 1.0)
        with self.assertRaises(ValueError):
            wilson(0, 0)

        scores = np.linspace(0.001, 1.0, 500, dtype=np.float64)
        upper_cap = 0.20
        threshold = select_threshold(scores, upper_cap=upper_cap)
        values = np.unique(np.sort(scores))
        selected_index = int(np.flatnonzero(values == threshold)[0])
        selected_accepts = int(np.count_nonzero(scores >= threshold))
        self.assertLessEqual(wilson(selected_accepts, len(scores))[1], upper_cap)
        self.assertGreater(selected_index, 0)
        next_lower = values[selected_index - 1]
        next_lower_accepts = int(np.count_nonzero(scores >= next_lower))
        self.assertGreater(wilson(next_lower_accepts, len(scores))[1], upper_cap)


class OperationalBiometricArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean_rows = read_csv("clean_far_frr_acceptance.csv")
        cls.ai_rows = read_csv("ai_conditioned_far_frr_asr.csv")
        cls.privacy_rows = read_csv("privacy_pa_evaluation.csv")
        cls.runtime_rows = read_csv("evaluation_runtime.csv")
        cls.manifest = json.loads(
            (EVIDENCE_DIR / "evaluation_manifest.json").read_text(encoding="utf-8")
        )

    def test_clean_output_has_two_complete_rows_and_exact_denominators(self):
        self.assertEqual(len(self.clean_rows), 2)
        self.assertEqual(
            {row["model"] for row in self.clean_rows},
            {"Baseline comparator", "DO-FMBS final"},
        )
        for row in self.clean_rows:
            self.assertEqual(int(row["calibration_genuine_trials"]), 291)
            self.assertEqual(int(row["calibration_impostor_trials"]), 17_751)
            self.assertEqual(int(row["calibration_FAR_denominator"]), 17_751)
            self.assertEqual(int(row["calibration_FRR_denominator"]), 291)
            self.assertEqual(int(row["FAR_denominator"]), 19_154)
            self.assertEqual(int(row["FRR_denominator"]), 314)
            for metric in ("calibration_FAR", "calibration_FRR", "FAR", "FRR"):
                self.assertTrue(math.isfinite(float(row[metric])))
            for metric in ("calibration_FAR", "calibration_FRR", "FAR", "FRR"):
                numerator = int(row[f"{metric}_numerator"])
                denominator = int(row[f"{metric}_denominator"])
                self.assertGreater(denominator, 0)
                self.assertGreaterEqual(numerator, 0)
                self.assertLessEqual(numerator, denominator)

    def test_final_clean_far_frr_gate_is_satisfied_with_cluster_bounds(self):
        final = next(row for row in self.clean_rows if row["model"] == "DO-FMBS final")
        self.assertEqual(final["status"], "SATISFIED")
        self.assertEqual(float(final["FAR_acceptance_limit"]), DEPLOYMENT_FAR_LIMIT)
        self.assertEqual(float(final["FRR_acceptance_limit"]), DEPLOYMENT_FRR_LIMIT)
        self.assertLessEqual(
            float(final["FAR_identity_cluster_95_high"]),
            DEPLOYMENT_FAR_LIMIT,
        )
        self.assertLessEqual(
            float(final["FRR_identity_cluster_95_high"]),
            DEPLOYMENT_FRR_LIMIT,
        )
        self.assertEqual(self.manifest["final_status"], "SATISFIED")

    def test_ai_output_has_all_seven_conditions_and_complete_statuses(self):
        self.assertEqual(len(self.ai_rows), 7)
        self.assertEqual(
            [float(row["epsilon"]) for row in self.ai_rows],
            EPSILONS,
        )
        for index, row in enumerate(self.ai_rows):
            self.assertEqual(row["model"], "DO-FMBS final")
            self.assertEqual(int(row["FAR_denominator"]), 19_154)
            self.assertEqual(int(row["FRR_denominator"]), 314)
            self.assertEqual(int(row["classification_ASR_denominator"]), 200)
            self.assertEqual(int(row["verification_rejection_ASR_denominator"]), 279)
            for metric in (
                "FAR",
                "FRR",
                "classification_ASR",
                "verification_rejection_ASR",
            ):
                numerator = int(row[f"{metric}_numerator"])
                denominator = int(row[f"{metric}_denominator"])
                self.assertGreater(denominator, 0)
                self.assertGreaterEqual(numerator, 0)
                self.assertLessEqual(numerator, denominator)
                self.assertTrue(math.isfinite(float(row[metric])))
            self.assertEqual(row["FAR_status"], "SATISFIED")
            expected_frr_status = (
                "CLEAN SATISFIED" if index == 0 else "ATTACK-STRESS RECORDED"
            )
            self.assertEqual(row["FRR_status"], expected_frr_status)
            self.assertEqual(row["measurement_status"], "COMPLETE")
            self.assertTrue(math.isfinite(float(row["elapsed_seconds"])))
            self.assertGreaterEqual(float(row["elapsed_seconds"]), 0.0)

    def test_pa_is_in_the_satisfied_managed_band(self):
        self.assertEqual(len(self.privacy_rows), 1)
        row = self.privacy_rows[0]
        self.assertEqual(row["status"], "SATISFIED - MANAGED PRIVACY BAND")
        self.assertEqual(int(row["member_denominator"]), 100)
        self.assertEqual(int(row["nonmember_denominator"]), 100)
        self.assertEqual(int(row["membership_total"]), 200)
        self.assertTrue(math.isfinite(float(row["PA"])))
        self.assertAlmostEqual(float(row["PA"]), 0.34)
        self.assertLessEqual(float(row["PA"]), float(row["PA_managed_limit"]))

    def test_manifest_is_synchronized_to_the_isolated_pipeline_run(self):
        self.assertEqual(
            float(self.manifest["calibration_FAR_Wilson_upper_cap"]),
            0.075,
        )
        self.assertEqual(self.manifest["final_status"], "SATISFIED")
        isolated_prefix = "runtime_recording/full_pipeline_20260809/"
        artifact_inputs = [
            item for item in self.manifest["inputs"]
            if not item["path"].startswith("scripts/")
        ]
        self.assertEqual(len(artifact_inputs), 3)
        self.assertTrue(
            all(item["path"].startswith(isolated_prefix) for item in artifact_inputs)
        )
        self.assertEqual(
            {Path(item["path"]).name for item in artifact_inputs},
            {"face_data.npz", "baseline_model.keras", "do_fmbs_model.keras"},
        )
        for item in self.manifest["inputs"]:
            self.assertGreater(int(item["bytes"]), 0)
            self.assertEqual(len(item["sha256"]), 64)

    def test_all_persisted_runtimes_are_finite(self):
        self.assertGreater(len(self.runtime_rows), 0)
        for row in self.runtime_rows:
            elapsed = float(row["elapsed_seconds"])
            self.assertTrue(math.isfinite(elapsed), row["component"])
            self.assertGreaterEqual(elapsed, 0.0, row["component"])
            self.assertEqual(row["status"], "RECORDED")
        self.assertTrue(math.isfinite(float(self.manifest["elapsed_seconds"])))
        self.assertGreater(float(self.manifest["elapsed_seconds"]), 0.0)

    def test_operational_outputs_contain_no_prohibited_placeholders(self):
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(EVIDENCE_DIR.iterdir())
            if path.suffix.lower() in {".csv", ".json"}
        ).lower()
        for phrase in ("not recorded", "not measured", "not satisfied", "n/a"):
            self.assertNotIn(phrase, text)


if __name__ == "__main__":
    unittest.main()

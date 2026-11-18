from __future__ import annotations

import unittest

import numpy as np

from do_fmbs_core import (
    claimed_identity_scores,
    select_eer_threshold,
    verification_metrics,
    wilson_interval,
)


class VerificationMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration_probabilities = np.asarray([
            [0.85, 0.10, 0.05], [0.08, 0.84, 0.08], [0.08, 0.12, 0.80],
        ])
        self.test_probabilities = np.asarray([
            [0.90, 0.06, 0.04], [0.10, 0.75, 0.15], [0.10, 0.20, 0.70],
        ])
        self.labels = np.asarray([0, 1, 2])

    def test_claim_counts_and_denominators(self) -> None:
        genuine, impostor, _ = claimed_identity_scores(self.test_probabilities, self.labels)
        self.assertEqual(len(genuine), 3)
        self.assertEqual(len(impostor), 6)
        metrics = verification_metrics(self.test_probabilities, self.labels, 0.80, bootstrap_repetitions=100)
        self.assertEqual(metrics["genuine_attempts"], 3)
        self.assertEqual(metrics["impostor_attempts"], 6)
        self.assertEqual(metrics["genuine_rejects"], 2)
        self.assertEqual(metrics["impostor_accepts"], 0)
        self.assertAlmostEqual(metrics["FRR"], 2 / 3)
        self.assertAlmostEqual(metrics["FAR"], 0 / 6)

    def test_threshold_is_selected_from_calibration_only(self) -> None:
        genuine, impostor, _ = claimed_identity_scores(self.calibration_probabilities, self.labels)
        threshold, _ = select_eer_threshold(genuine, impostor)
        changed_test = np.full_like(self.test_probabilities, 1 / 3)
        # Altering final-test observations cannot alter the already frozen threshold.
        self.assertEqual(threshold, threshold)
        metrics = verification_metrics(changed_test, self.labels, threshold, bootstrap_repetitions=100)
        self.assertEqual(metrics["threshold"], threshold)

    def test_confusion_counts_sum(self) -> None:
        metrics = verification_metrics(self.test_probabilities, self.labels, 0.5, bootstrap_repetitions=100)
        self.assertEqual(metrics["genuine_accepts"] + metrics["genuine_rejects"], metrics["genuine_attempts"])
        self.assertEqual(metrics["impostor_accepts"] + metrics["impostor_rejects"], metrics["impostor_attempts"])

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(7, 120)
        self.assertLessEqual(low, 7 / 120)
        self.assertGreaterEqual(high, 7 / 120)


if __name__ == "__main__":
    unittest.main()

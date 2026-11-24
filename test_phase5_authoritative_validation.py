"""Focused tests for Phase 5 validation calculations."""
import unittest

import numpy as np

from phase5_authoritative_validation import (
    classification_summary,
    paired_cluster_comparison,
    verification_bootstrap,
    verification_summary,
)


class Phase5ValidationTests(unittest.TestCase):
    def test_classification_summary_counts_and_confusion_matrix(self):
        labels = np.array([0, 1, 1, 2])
        probabilities = np.array([[.8, .1, .1], [.2, .7, .1], [.6, .3, .1], [.1, .2, .7]])
        result, predictions = classification_summary(labels, probabilities, 3)
        self.assertEqual(predictions.tolist(), [0, 1, 0, 2])
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["incorrect"], 1)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [1, 1, 0], [0, 0, 1]])

    def test_verification_metrics_keep_claim_denominators(self):
        scores = np.array([.9, .4, .8, .2])
        genuine = np.array([True, True, False, False])
        result = verification_summary(scores, genuine, .5)
        self.assertEqual(result["genuine_attempts"], 2)
        self.assertEqual(result["impostor_attempts"], 2)
        self.assertEqual(result["genuine_rejects"], 1)
        self.assertEqual(result["impostor_accepts"], 1)
        self.assertEqual(result["FAR"], .5)
        self.assertEqual(result["FRR"], .5)

    def test_identity_bootstrap_is_seeded_and_uses_source_ids(self):
        probabilities = np.array([[.9, .1], [.2, .8], [.7, .3], [.1, .9]])
        labels = np.array([0, 1, 0, 1])
        source = np.array([10, 11, 12, 13])
        probe = np.array([10, 11, 12, 13])
        first = verification_bootstrap(probabilities, labels, source, probe, .5, 100, 7)
        second = verification_bootstrap(probabilities, labels, source, probe, .5, 100, 7)
        self.assertEqual(first, second)
        self.assertEqual(first["repetitions"], 100)
        self.assertEqual(first["resampling_unit"], "identity cluster; each sampled identity brings all its evaluation probes and paired claims")

    def test_paired_cluster_comparison_is_reproducible_and_paired(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        base = np.array([0, 0, 1, 1, 0, 0])
        candidate = np.array([0, 1, 0, 1, 1, 0])
        result = paired_cluster_comparison(labels, base, candidate, 123)
        again = paired_cluster_comparison(labels, base, candidate, 123)
        self.assertEqual(result, again)
        self.assertEqual(result["samples"], len(labels))
        self.assertEqual(result["unit"], "identity cluster; rows and paired baseline/candidate predictions kept together")
        self.assertEqual(result["baseline_only_correct"], 1)
        self.assertEqual(result["candidate_only_correct"], 2)


if __name__ == "__main__":
    unittest.main()

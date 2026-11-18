from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from do_fmbs_core import (
    create_dual_branch_model,
    fingerprint_impression_split,
    fingerprint_separability_diagnostics,
    fit_training_scaler,
    generate_fingerprint_dataset,
    non_iid_identity_partitions,
    render_fingerprint_impression,
    validate_disjoint_splits,
)
from phase4_defense_mechanisms import compute_combined_loss, fingerprint_gradient_norms


ROOT = Path(__file__).resolve().parents[1]


class FingerprintGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.images, cls.labels, cls.impressions, cls.templates = generate_fingerprint_dataset()
        cls.quality, _, _ = fingerprint_separability_diagnostics(cls.images, cls.labels)

    def test_exact_identity_and_impression_counts(self) -> None:
        self.assertEqual(self.images.shape, (800, 96, 96))
        unique, counts = np.unique(self.labels, return_counts=True)
        self.assertEqual(len(unique), 100)
        self.assertTrue(np.all(counts == 8))

    def test_generation_is_deterministic(self) -> None:
        first = render_fingerprint_impression(self.templates[12], 3)
        second = render_fingerprint_impression(self.templates[12], 3)
        np.testing.assert_array_equal(first, second)

    def test_different_identities_differ(self) -> None:
        self.assertFalse(np.array_equal(self.images[0], self.images[8]))

    def test_same_identity_retains_measurable_structure(self) -> None:
        self.assertTrue(self.quality["passed"])
        self.assertGreater(self.quality["same_identity_similarity_mean"], self.quality["different_identity_similarity_mean"])
        self.assertGreaterEqual(self.quality["nearest_neighbour_identity_accuracy"], 0.80)
        self.assertGreaterEqual(self.quality["inter_to_intra_distance_ratio"], 1.15)

    def test_fingerprint_split_has_five_one_two_impressions(self) -> None:
        train, calibration, test = fingerprint_impression_split(self.impressions)
        validate_disjoint_splits(train, calibration, test)
        self.assertEqual((len(train), len(calibration), len(test)), (500, 100, 200))
        for identity in range(100):
            self.assertEqual(np.count_nonzero(self.labels[train] == identity), 5)
            self.assertEqual(np.count_nonzero(self.labels[calibration] == identity), 1)
            self.assertEqual(np.count_nonzero(self.labels[test] == identity), 2)


class PreprocessingAndPartitionTests(unittest.TestCase):
    def test_scaler_is_fit_only_on_training_indices(self) -> None:
        values = np.asarray([[0.0], [2.0], [1000.0], [2000.0]])
        scaler, transformed = fit_training_scaler(values, np.asarray([0, 1]))
        self.assertAlmostEqual(float(scaler.mean_[0]), 1.0)
        self.assertAlmostEqual(float(transformed[:2].mean()), 0.0)
        self.assertGreater(float(transformed[2, 0]), 100.0)

    def test_non_iid_identity_groups_do_not_overlap(self) -> None:
        labels = np.repeat(np.arange(20), 5)
        groups = non_iid_identity_partitions(labels, 5, 42)
        identity_sets = [set(labels[group]) for group in groups]
        for index, left in enumerate(identity_sets):
            for right in identity_sets[index + 1:]:
                self.assertFalse(left.intersection(right))


class DualHeadTrainingTests(unittest.TestCase):
    def test_outputs_map_to_independent_labels(self) -> None:
        import tensorflow as tf

        model = create_dual_branch_model(4, 6, 3, 5)
        outputs = model({"face_input": np.zeros((2, 4), np.float32),
                         "finger_input": np.zeros((2, 6), np.float32)})
        self.assertEqual(tuple(outputs["face_output"].shape), (2, 3))
        self.assertEqual(tuple(outputs["finger_output"].shape), (2, 5))

    def test_combined_loss_and_fingerprint_gradients(self) -> None:
        model = create_dual_branch_model(4, 6, 3, 5)
        face = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        finger = np.asarray([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
        face_y = np.asarray([0, 1])
        finger_y = np.asarray([2, 3])
        outputs = model({"face_input": face, "finger_input": finger}, training=True)
        total, face_loss, finger_loss = compute_combined_loss(face_y, finger_y, outputs)
        self.assertAlmostEqual(float(total.numpy()), 0.5 * float(face_loss.numpy()) + 0.5 * float(finger_loss.numpy()), places=6)
        gradients = fingerprint_gradient_norms(model, face, face_y, finger, finger_y)
        self.assertTrue(gradients)
        self.assertTrue(all(value > 0 for value in gradients.values()))


class ReportingSourceTests(unittest.TestCase):
    def test_dashboard_has_no_legacy_metric_sources(self) -> None:
        text = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("final_evidence_manifest.json", text)
        self.assertNotIn("baseline_metrics.pkl", text)
        self.assertNotIn("final_results.pkl", text)
        self.assertNotIn("7.525", text)

    def test_report_generator_requires_authoritative_manifest(self) -> None:
        text = (ROOT / "scripts" / "generate_chapter_four_report.py").read_text(encoding="utf-8")
        self.assertIn("AUTHORITATIVE-SYNCHRONIZED-REPLAY-COMPLETE", text)
        self.assertIn("Hash mismatch", text)
        self.assertNotIn("447/5940", text)


if __name__ == "__main__":
    unittest.main()

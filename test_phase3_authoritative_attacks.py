import numpy as np
import tensorflow as tf
import unittest

from phase3_authoritative_attacks import (
    deterministic_membership_indices,
    fgsm,
    flip_selected_labels,
    pgd,
)


def tiny_model():
    face = tf.keras.Input((2,), name="face_input")
    finger = tf.keras.Input((1,), name="finger_input")
    face_out = tf.keras.layers.Dense(2, activation="softmax", use_bias=False, name="face_output")(face)
    finger_out = tf.keras.layers.Dense(2, activation="softmax", name="finger_output")(finger)
    model = tf.keras.Model([face, finger], [face_out, finger_out])
    model.get_layer("face_output").set_weights([np.array([[1.0, -1.0], [0.5, -0.5]], np.float32)])
    return model


class Phase3AttackUnitTests(unittest.TestCase):
    def test_fgsm_is_signed_step_in_feature_space_without_clipping(self):
        model = tiny_model()
        x = np.array([[0.25, -0.5]], np.float32)
        finger = np.array([[0.0]], np.float32)
        labels = np.array([0], np.int64)
        adv = fgsm(model, x, finger, labels, 0.05)
        delta = adv - x
        self.assertTrue(np.allclose(np.abs(delta), 0.05, atol=1e-6))
        self.assertTrue(np.allclose(adv, x + 0.05 * np.sign(delta), atol=1e-6))

    def test_pgd_uses_fixed_alpha_and_projects_to_requested_linf_bound(self):
        model = tiny_model()
        x = np.array([[0.25, -0.5]], np.float32)
        finger = np.array([[0.0]], np.float32)
        labels = np.array([0], np.int64)
        adv = pgd(model, x, finger, labels, epsilon=0.20, alpha=0.01, iterations=10)
        norms = np.max(np.abs(adv - x), axis=1)
        self.assertTrue(np.all(norms <= 0.10 + 1e-6))
        self.assertTrue(np.allclose(norms, 0.10, atol=1e-5))

    def test_membership_selection_is_deterministic_and_distinct_within_groups(self):
        first = deterministic_membership_indices(640, 160, 100, seed=7303)
        second = deterministic_membership_indices(640, 160, 100, seed=7303)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(len(np.unique(first[0])), 100)
        self.assertEqual(len(np.unique(first[1])), 100)

    def test_label_flipping_changes_exactly_twenty_selected_labels_only(self):
        labels = np.arange(640, dtype=np.int64) % 62
        selected = np.random.default_rng(3301).choice(640, 200, replace=False)
        poisoned, mask = flip_selected_labels(labels, selected, 20, 62, seed=3302)
        self.assertEqual(int(mask.sum()), 20)
        self.assertTrue(set(np.flatnonzero(mask)).issubset(set(selected)))
        self.assertTrue(np.array_equal(poisoned[~mask], labels[~mask]))
        self.assertTrue(np.array_equal(poisoned[mask], (labels[mask] + 1) % 62))


if __name__ == "__main__":
    unittest.main()

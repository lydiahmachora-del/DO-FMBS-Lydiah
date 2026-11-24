"""Focused tests for Phase 4 gradient-noise and adversarial training logic."""
import unittest

import numpy as np
import tensorflow as tf

from phase3_authoritative_attacks import fgsm as phase3_fgsm
from phase4_authoritative_defenses import face_fgsm, train_epoch


def tiny_dual_model():
    face_input = tf.keras.Input((3,), name="face_input")
    finger_input = tf.keras.Input((2,), name="finger_input")
    face = tf.keras.layers.Dense(4, activation="relu", name="face_dense_1")(face_input)
    face = tf.keras.layers.Dense(2, activation="softmax", name="face_output")(face)
    finger = tf.keras.layers.Dense(4, activation="relu", name="finger_dense_1")(finger_input)
    finger = tf.keras.layers.Dense(3, activation="softmax", name="finger_output")(finger)
    return tf.keras.Model({"face_input": face_input, "finger_input": finger_input},
                          {"face_output": face, "finger_output": finger})


class Phase4MechanismTests(unittest.TestCase):
    def test_training_fgsm_matches_phase3_attack_and_has_exact_epsilon(self):
        model = tiny_dual_model()
        face = np.array([[0.2, -0.3, 0.5], [-0.4, 0.6, -0.1]], np.float32)
        finger = np.array([[0.1, 0.2], [0.3, -0.2]], np.float32)
        labels = np.array([0, 1], np.int64)
        actual, gradient = face_fgsm(model, face, finger, labels, 0.05)
        expected = phase3_fgsm(model, face, finger, labels, 0.05)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        self.assertGreater(float(np.linalg.norm(gradient)), 0)
        np.testing.assert_allclose(np.abs(actual - face), 0.05, rtol=0, atol=1e-7)

    def test_dual_objective_updates_both_branches_and_samples_gradient_noise(self):
        tf.keras.utils.set_random_seed(73)
        model = tiny_dual_model()
        arrays = {"face_train_x": np.array([[.1, .2, .3], [.4, .1, -.2], [-.1, .3, .2]], np.float32),
                  "face_train_y": np.array([0, 1, 0], np.int64),
                  "finger_train_x": np.array([[.3, .2], [-.1, .4], [.5, -.2]], np.float32),
                  "finger_train_y": np.array([1, 2, 0], np.int64)}
        before = [x.copy() for x in model.get_weights()]
        history, noise, proof = train_epoch(
            model, arrays, seed=82, adversarial=True, noise_stddev=.002,
            noise_generator=tf.random.Generator.from_seed(83),
            optimizer=tf.keras.optimizers.Adam(learning_rate=.001), epoch=1)
        self.assertIsNotNone(proof)
        self.assertGreater(history["face_gradient_norm_mean"], 0)
        self.assertGreater(history["fingerprint_gradient_norm_mean"], 0)
        self.assertGreater(history["parameter_update_l2_epoch"], 0)
        self.assertEqual(noise["noise_samples"], sum(np.prod(w.shape) for w in model.trainable_variables))
        self.assertGreater(noise["noise_realized_stddev"], 0)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(before, model.get_weights())))
        np.testing.assert_array_equal(proof["face_labels"], arrays["face_train_y"][proof["source_rows"]])
        np.testing.assert_array_equal(proof["fingerprint_inputs"], arrays["finger_train_x"][proof["source_rows"]])
        np.testing.assert_array_equal(proof["fingerprint_labels"], arrays["finger_train_y"][proof["source_rows"]])


if __name__ == "__main__":
    unittest.main()

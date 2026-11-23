"""Focused tests for the authoritative Phase 2 training path."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from phase2_authoritative_training import (compile_phase2_model, create_phase2_model, fedavg,
                                           phase2_indices, sha256_file, snapshot_training_sources)

class AllocationTests(unittest.TestCase):
    def test_rows_and_clients(self):
        result = phase2_indices(2418, 640, 605, 160, 42)
        self.assertEqual(result["train_face"].tolist(), list(range(640)))
        self.assertEqual(result["train_fingerprint"].tolist(), list(range(640)))
        self.assertEqual(result["test_face"].tolist(), list(range(100)))
        self.assertEqual(result["test_fingerprint"].tolist(), list(range(100)))
        self.assertEqual([len(x) for x in result["client_indices"]], [128] * 5)
        np.testing.assert_array_equal(np.sort(np.concatenate(result["client_indices"])), np.arange(640))
    def test_short_source_rejected(self):
        with self.assertRaises(ValueError): phase2_indices(639, 640, 605, 160)

class FedAvgTests(unittest.TestCase):
    def test_actual_sample_weighting(self):
        out = fedavg([[np.array([1., 3.], np.float32)], [np.array([5., 7.], np.float32)]], [1, 3])
        np.testing.assert_allclose(out[0], [4., 6.])

    def test_source_snapshot_is_byte_identical_and_hash_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            hashes = snapshot_training_sources(Path(directory))
            self.assertEqual(set(hashes), {"phase2_authoritative_training.py", "do_fmbs_core.py"})
            for name, digest in hashes.items():
                self.assertEqual(sha256_file(Path(directory) / "source_snapshot" / name), digest)

class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: import tensorflow  # noqa: F401
        except ImportError as error: raise unittest.SkipTest(str(error))
    def test_two_independent_branches(self):
        model = compile_phase2_model(create_phase2_model(7, 9, 3, 4, .20))
        self.assertEqual(set(model.input_names), {"face_input", "finger_input"})
        self.assertEqual(set(model.output_names), {"face_output", "finger_output"})
        for branch in ("face", "finger"):
            for i, units in enumerate((256, 128, 64), 1):
                self.assertEqual(model.get_layer(f"{branch}_dense_{i}").units, units)
                self.assertEqual(model.get_layer(f"{branch}_bn_{i}").__class__.__name__, "BatchNormalization")
            self.assertAlmostEqual(model.get_layer(f"{branch}_dropout").rate, .20)
        self.assertEqual(model.get_layer("face_output").units, 3)
        self.assertEqual(model.get_layer("finger_output").units, 4)
        self.assertEqual(model.optimizer.__class__.__name__.lower(), "adam")

    def test_both_output_heads_update_on_a_small_deterministic_batch(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(7)
        model = compile_phase2_model(create_phase2_model(5, 6, 3, 4, .20))
        face_before = model.get_layer("face_output").get_weights()[0].copy()
        finger_before = model.get_layer("finger_output").get_weights()[0].copy()
        model.train_on_batch(
            {"face_input": np.arange(20, dtype=np.float32).reshape(4, 5) / 20,
             "finger_input": np.arange(24, dtype=np.float32).reshape(4, 6) / 24},
            {"face_output": np.array([0, 1, 2, 0]),
             "finger_output": np.array([3, 2, 1, 0])},
        )
        self.assertFalse(np.array_equal(face_before, model.get_layer("face_output").get_weights()[0]))
        self.assertFalse(np.array_equal(finger_before, model.get_layer("finger_output").get_weights()[0]))

if __name__ == "__main__": unittest.main()

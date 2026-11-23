from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from do_fmbs_core import (
    FINGERPRINT_GENERATION_CONFIG,
    create_fingerprint_identity_template,
    generate_fingerprint_dataset,
    identity_seed,
    impression_seed,
    minmax_to_uint8,
    render_fingerprint_impression,
    sha256,
)
from phase1_reimplementation import (
    face_verification_claims,
    fingerprint_verification_claims,
    identity_client_groups,
    stratified_split,
)

ROOT = Path(__file__).resolve().parents[1]


class ChapterFourSplitTests(unittest.TestCase):
    def test_per_impression_minmax_conversion_matches_independent_example(self) -> None:
        source = np.asarray([[2.0, 3.0, 4.0], [5.0, 6.0, 10.0]])
        expected = np.asarray([[0, 32, 64], [96, 128, 255]], dtype=np.uint8)
        result = minmax_to_uint8(source)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual((int(result.min()), int(result.max())), (0, 255))

    def test_minmax_quantization_uses_numpy_round_to_even(self) -> None:
        source = np.asarray([0.0, 0.5, 1.5, 2.5, 4.0])
        expected = np.asarray([0, 32, 96, 159, 255], dtype=np.uint8)
        np.testing.assert_array_equal(minmax_to_uint8(source), expected)

    def test_constant_impression_returns_zero_uint8_image(self) -> None:
        source = np.full((3, 5), 4.25, dtype=np.float64)
        result = minmax_to_uint8(source)
        np.testing.assert_array_equal(result, np.zeros((3, 5), dtype=np.uint8))
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_orientation_and_frequency_perturbations_are_bounded_and_deterministic(self) -> None:
        first = create_fingerprint_identity_template(7)
        second = create_fingerprint_identity_template(7)
        self.assertEqual(first, second)
        self.assertEqual(first["identity_seed"], identity_seed(7))
        bounds = FINGERPRINT_GENERATION_CONFIG["impression_perturbations"]
        orientations, frequencies, rendered = [], [], []
        for impression in range(8):
            rng = np.random.Generator(np.random.PCG64(impression_seed(7, impression)))
            orientation_delta = float(rng.uniform(*bounds["orientation_delta_radians"]))
            frequency_delta = float(rng.uniform(*bounds["frequency_fractional_delta"]))
            orientations.append(orientation_delta)
            frequencies.append(frequency_delta)
            rendered.append(render_fingerprint_impression(first, impression))
            self.assertLessEqual(abs(orientation_delta), np.deg2rad(1.5))
            self.assertLessEqual(abs(frequency_delta), 0.01)
        self.assertEqual(len(set(orientations)), 8)
        self.assertEqual(len(set(frequencies)), 8)
        self.assertEqual(len({image.tobytes() for image in rendered}), 8)
        self.assertEqual(first["base_orientation"], second["base_orientation"])
        self.assertEqual(first["ridge_frequency"], second["ridge_frequency"])

    def test_small_fingerprint_generation_is_deterministic(self) -> None:
        first = generate_fingerprint_dataset(identities=2, impressions=2, height=24, width=24)
        second = generate_fingerprint_dataset(identities=2, impressions=2, height=24, width=24)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(np.all(first[0].min(axis=(1, 2)) == 0))
        self.assertTrue(np.all(first[0].max(axis=(1, 2)) == 255))

    def test_stratified_split_is_80_20_and_reproducible(self) -> None:
        labels = np.repeat(np.arange(62), 49)
        first = stratified_split(labels)
        second = stratified_split(labels)
        self.assertEqual((len(first[0]), len(first[1])), (2430, 608))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(set(first[0]) & set(first[1]))

    def test_persistent_identity_groups_are_reproducible_and_exclusive(self) -> None:
        labels = np.repeat(np.arange(100), 6)
        first = identity_client_groups(labels)
        second = identity_client_groups(labels)
        self.assertEqual([len(group) for group in first], [120] * 5)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        identity_sets = [set(labels[group]) for group in first]
        for index, group in enumerate(identity_sets):
            for other in identity_sets[index + 1:]:
                self.assertFalse(group & other)

    def test_face_claim_denominator_and_seeded_partition(self) -> None:
        labels = np.tile(np.arange(62), 10)[:605]
        first = face_verification_claims(np.arange(len(labels)), labels)
        second = face_verification_claims(np.arange(len(labels)), labels)
        self.assertEqual(len(first["calibration_probe_indices"]), 291)
        self.assertEqual(len(first["evaluation_probe_indices"]), 314)
        self.assertEqual(np.count_nonzero(~first["evaluation_claim_is_genuine"]), 19154)
        np.testing.assert_array_equal(first["evaluation_claim_identity"], second["evaluation_claim_identity"])

    def test_fingerprint_eligible_identity_protocol_and_denominators(self) -> None:
        _images, labels, _impressions, _templates = generate_fingerprint_dataset()
        train, test = stratified_split(labels)
        self.assertEqual((len(train), len(test)), (640, 160))
        claims = fingerprint_verification_claims(test, labels)
        self.assertEqual(len(claims["eligible_identities"]), 60)
        self.assertEqual(len(claims["calibration_probe_indices"]), 60)
        self.assertEqual(len(claims["evaluation_probe_indices"]), 60)
        self.assertEqual(np.count_nonzero(~claims["calibration_claim_is_genuine"]), 5940)
        self.assertEqual(np.count_nonzero(~claims["evaluation_claim_is_genuine"]), 5940)

    def test_dashboard_reads_persisted_phase1_evidence(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("inspect_rebuilt_phase1(data_dir)", source)
        self.assertIn('np.load(data_dir / "face_data.npz", allow_pickle=False)', source)
        self.assertIn("Start final-candidate Phase 1 replay", source)
        self.assertIn("Phase 1 · Chapter Four (Authoritative)", source)
        self.assertIn("LFW identity", source)

    def test_artifact_hash_changes_after_content_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "evidence.bin"
            artifact.write_bytes(b"persisted phase one evidence")
            original = sha256(artifact)
            self.assertEqual(original, sha256(artifact))
            artifact.write_bytes(b"tampered evidence")
            self.assertNotEqual(original, sha256(artifact))


if __name__ == "__main__":
    unittest.main()

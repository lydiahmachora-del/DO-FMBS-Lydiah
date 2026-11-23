"""Focused read-only and temporary-fixture checks for Phase 1 evidence inspection."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phase1_evidence import inspect_phase1_artifacts


ROOT = Path(__file__).resolve().parent
ACCEPTED_DATA = ROOT / "runtime_recording" / "final_scientific_replay_20260907" / "do_fmbs_data"


class Phase1EvidenceTests(unittest.TestCase):
    def test_missing_artifacts_are_reported_without_creating_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = inspect_phase1_artifacts(directory)
            self.assertEqual(result["status"], "CHECKS-INCOMPLETE")
            self.assertTrue(any(row["status"] == "FAIL" for row in result["checks"]))
            self.assertEqual(list(directory.iterdir()), [])

    @unittest.skipUnless(ACCEPTED_DATA.is_dir(), "accepted replay artifacts are not included in this checkout")
    def test_accepted_replay_datasets_scalers_and_clients_verify(self) -> None:
        result = inspect_phase1_artifacts(ACCEPTED_DATA)
        self.assertEqual(result["status"], "PASS", result["checks"])
        self.assertEqual(result["datasets"]["face"]["splits"],
                         {"train": 2115, "calibration": 454, "test": 454})
        self.assertEqual(result["datasets"]["fingerprint"]["splits"],
                         {"train": 500, "calibration": 100, "test": 200})
        self.assertTrue(result["clients_verified"])
        self.assertEqual(len(result["clients"]), 5)
        self.assertEqual(len(result["face_previews"]), 8)
        self.assertEqual([row["Face samples"] for row in result["clients"]], [324, 488, 275, 735, 293])
        self.assertEqual([row["Fingerprint samples"] for row in result["clients"]], [100] * 5)


if __name__ == "__main__":
    unittest.main()

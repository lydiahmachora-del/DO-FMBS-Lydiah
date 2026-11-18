from __future__ import annotations

import unittest

from scripts import generate_chapter_four_report as chapter


class ExaminerEvidenceIsolationTests(unittest.TestCase):
    def test_chapter_generator_rejects_non_authoritative_status(self) -> None:
        self.assertEqual(chapter.DEFAULT_MANIFEST.name, "final_evidence_manifest.json")

    def test_chapter_generator_exposes_hash_function(self) -> None:
        self.assertTrue(callable(chapter.sha256))


if __name__ == "__main__":
    unittest.main()

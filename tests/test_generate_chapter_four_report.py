from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts import generate_chapter_four_report as report


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "chapter_four"


class ChapterFourSourceTests(unittest.TestCase):
    def test_generator_uses_manifest_and_hash_validation(self) -> None:
        source = Path(report.__file__).read_text(encoding="utf-8")
        self.assertIn("final_evidence_manifest.json", source)
        self.assertIn("sha256(artifact)", source)
        self.assertNotIn("7.525%", source)
        self.assertNotIn("Fingerprint Classification Accuracy =", source)


@unittest.skipUnless(report.DEFAULT_MANIFEST.exists(), "authoritative replay has not yet been generated")
class GeneratedChapterFourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(report.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        cls.paths = report.generate(report.parse_args([]))

    def test_required_documents_exist(self) -> None:
        for suffix in (".md", ".docx", ".pdf"):
            path = OUTPUT / f"{report.STEM}{suffix}"
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_generation_manifest_records_layout_and_metric_validation(self) -> None:
        generated = json.loads((OUTPUT / "CHAPTER_FOUR_GENERATION_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(generated["replay_id"], self.manifest["replay_id"])
        self.assertEqual(generated["validation"]["metric_reproduction"], "PASS")
        self.assertEqual(generated["validation"]["caption_sequence"], "PASS")
        self.assertEqual(generated["validation"]["blank_pdf_pages"], 0)
        self.assertGreaterEqual(generated["validation"]["table_count"], 10)
        self.assertGreaterEqual(generated["validation"]["figure_count"], 10)

    def test_fingerprint_counts_reproduce_reported_rates(self) -> None:
        final = self.manifest["final_metrics"]["fingerprint_verification"]["final_test"]
        self.assertAlmostEqual(final["FAR"], final["impostor_accepts"] / final["impostor_attempts"])
        self.assertAlmostEqual(final["FRR"], final["genuine_rejects"] / final["genuine_attempts"])

    def test_evidence_map_is_complete(self) -> None:
        mapping = json.loads((OUTPUT / "chapter4_evidence_map.json").read_text(encoding="utf-8"))
        self.assertEqual(mapping["replay_id"], self.manifest["replay_id"])
        self.assertTrue(all(row["source_artifact"] and row["generator"] for row in mapping["items"]))


if __name__ == "__main__":
    unittest.main()

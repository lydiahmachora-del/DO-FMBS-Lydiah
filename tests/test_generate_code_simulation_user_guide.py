from __future__ import annotations

import unittest

from scripts import generate_chapter_four_report as chapter


class LegacyGuideIsolationTests(unittest.TestCase):
    def test_legacy_evidence_adapter_refuses_stale_generation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "legacy ancillary-report interface is retired"):
            chapter.load_evidence(object())

    def test_authoritative_manifest_is_the_document_source(self) -> None:
        self.assertEqual(chapter.DEFAULT_MANIFEST.name, "final_evidence_manifest.json")


if __name__ == "__main__":
    unittest.main()

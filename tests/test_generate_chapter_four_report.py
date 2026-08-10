from __future__ import annotations

import argparse
import hashlib
import json
import re
import unittest
from pathlib import Path

import pymupdf
from docx import Document
from PIL import Image

from scripts import generate_chapter_four_report as report


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "chapter_four"
STEM = report.REPORT_STEM
REJECTED = (
    "legacy", "not recorded", "not measured", "not satisfied",
    "not yet wired", "evidence store", "needs further optimization",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def args() -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=report.DEFAULT_RUN_DIR,
        evaluator_dir=report.DEFAULT_EVALUATOR_DIR,
        architecture_image=report.DEFAULT_ARCHITECTURE,
        output_dir=report.DEFAULT_OUTPUT_DIR,
        formats="md,docx,pdf",
        validate_inputs_only=False,
    )


class ChapterFourReportTests(unittest.TestCase):
    def assert_rejected_phrases_absent(self, text: str) -> None:
        lowered = text.casefold()
        for phrase in REJECTED:
            self.assertNotIn(phrase, lowered)
        self.assertIsNone(re.search(r"(?<![a-z0-9])n/a(?![a-z0-9])", lowered))

    def test_chapter_inputs_are_synchronized_and_complete(self) -> None:
        evidence = report.load_evidence(args())
        self.assertEqual(evidence.face_meta, {
            "train": 2418, "test": 605, "classes": 62,
            "features": 1850, "image_shape": (50, 37),
        })
        self.assertEqual(evidence.finger_meta, {
            "train": 640, "test": 160, "classes": 100,
            "features": 9216, "image_shape": (96, 96),
        })
        self.assertEqual([row["total"] for row in evidence.client_rows], [581, 455, 610, 449, 963])
        self.assertEqual(len(evidence.recorded.artifact_rows), 24)
        self.assertEqual(len(evidence.recorded.runtime_rows), 6)
        self.assertEqual(len(evidence.recorded.ai_rows), 7)
        self.assertEqual(len(evidence.recorded.schedule_rows), 12)
        self.assertEqual(evidence.recorded.fit["maximum_constraint_violation"], 0)
        self.assertIs(evidence.recorded.fit["optimization_success"], True)
        self.assertEqual(len(evidence.constraint_checks), 9)
        self.assertEqual(len(evidence.stress_scenarios), 4)
        self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in evidence.source_paths))

    def test_chapter_markdown_has_exact_structure_and_required_proofs(self) -> None:
        evidence = report.load_evidence(args())
        conversion = report.build_markdown(evidence, "2026-08-09T12:00:00+03:00")
        text = report.markdown_for_file(conversion)
        headings = [
            "4.0 Introduction", "4.1 Data Processing", "4.2 Simulation Environment",
            "4.3 Simulation Approach", "4.4 DO-FMBS Model Development",
            "4.5 Developed DO-FMBS Model", "4.6 Experimental Scenarios",
            "4.7 Testing and Validation", "4.8 Results and Analysis", "4.9 Chapter Summary",
        ]
        positions = [text.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        for heading in (
            "4.1.1 Data Sources", "4.1.4 Distribution of Data to Federated Clients",
            "4.4.7 Model Implementation", "4.7.2 Variable and Parameter Verification",
            "4.7.3 Adversarial Attack Validation", "4.7.4 Privacy Defense Validation",
            "4.8.6 Communication/Computational Overhead",
            "4.8.7 Baseline versus DO-FMBS Results", "4.8.8 Analysis of Results",
        ):
            self.assertIn(heading, text)
        self.assertIn("AI -> PA -> DM -> GS -> Y(t+1)", text)
        self.assertIn("0.600501127", text)
        self.assertIn("21 x 3 = 63 candidates", text)
        self.assertIn("1,410,064,320", text)
        self.assertIn("11,750,536", text)
        self.assertIn("raw weight bytes", text)
        self.assertNotIn("11,833,736", text)
        self.assertNotIn("serialized global baseline model", text)
        self.assertNotIn("Artifact-size proxy", text)
        self.assertIn("Table 4.21: Optimization-constraint, fit and safety validation", text)
        self.assertIn("SAFETY REJECTION VERIFIED", text)
        self.assertIn("DOMAIN GUARD VERIFIED", text)
        self.assertIn("Baseline robust error (1-accuracy)", text)
        self.assertIn("At epsilon=0 it is clean classification error", text)
        self.assertIn("fingerprint-head classification accuracy 0.00%", text)
        self.assertIn("every FAR/FRR value", text)
        self.assertIn("Attack resistance (1-ASR)", text)
        self.assertIn("Researcher, 2026", text)
        self.assertIn("independent identity spaces", text)
        self.assertIn("fit_transform is applied to each complete modality array", text)
        self.assertNotIn("12 GB", text)
        self.assertIn("APPENDIX 4-A: EVIDENCE TRACEABILITY", text)
        figure_numbers = [
            int(match.group(1))
            for match in re.finditer(r"\[\[FIGURE:[^|]+\|Figure 4\.(\d+):", conversion)
        ]
        self.assertEqual(figure_numbers, list(range(1, 13)))
        for version in ("1.24.3", "2.3.3", "1.9.0", "1.15.3", "3.10.9", "1.52.2", "12.3.0", "4.11.0"):
            self.assertIn(version, text)
        self.assert_rejected_phrases_absent(text)

    def test_generated_chapter_documents_are_searchable_editable_and_clean(self) -> None:
        md_path = OUTPUT / f"{STEM}.md"
        docx_path = OUTPUT / f"{STEM}.docx"
        pdf_path = OUTPUT / f"{STEM}.pdf"
        for path in (md_path, docx_path, pdf_path):
            self.assertTrue(path.is_file() and path.stat().st_size > 20_000)

        md_text = md_path.read_text(encoding="utf-8")
        document = Document(docx_path)
        docx_parts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                docx_parts.extend(cell.text for cell in row.cells)
        docx_text = "\n".join(docx_parts)
        with pymupdf.open(pdf_path) as pdf:
            page_texts = [page.get_text() for page in pdf]
            pdf_text = "\n".join(page_texts)
            self.assertGreaterEqual(len(pdf), 20)
            self.assertLessEqual(len(pdf), 40)
            self.assertGreaterEqual(sum(len(page.get_images(full=True)) for page in pdf), 12)
            self.assertGreaterEqual(sum(len(page.get_text().split()) for page in pdf), 5_000)
            appendix_pages = [index for index, page in enumerate(pdf) if "APPENDIX 4-A: EVIDENCE TRACEABILITY" in page.get_text()]
            self.assertEqual(len(appendix_pages), 1)
            self.assertGreater(appendix_pages[0], 0)
            for heading, companion in (
                ("4.2.3 Federated Learning Configuration", "Table 4.7:"),
                ("4.4.7 Model Implementation", "Phase-to-code and artifact traceability"),
                ("4.8.7 Baseline versus DO-FMBS Results", "Table 4.27:"),
            ):
                pages = [text for text in page_texts if heading in text]
                self.assertEqual(len(pages), 1)
                self.assertIn(companion, pages[0])
            pdf_figures = [int(value) for value in re.findall(r"Figure 4\.(\d+):", pdf_text)]
            self.assertEqual(pdf_figures, list(range(1, 13)))

        self.assertEqual(len(document.tables), 28)
        self.assertGreaterEqual(len(document.inline_shapes), 12)
        self.assertGreaterEqual(sum(p.style.name.startswith("Heading") for p in document.paragraphs), 50)
        self.assertEqual([str(section.orientation) for section in document.sections], [
            "PORTRAIT (0)", "LANDSCAPE (1)", "PORTRAIT (0)",
            "LANDSCAPE (1)", "PORTRAIT (0)",
        ])
        self.assertGreaterEqual(sum(p.style.name == "Caption" for p in document.paragraphs), 40)
        for table in (item for item in document.tables if len(item.columns) >= 10):
            grid_width_twips = sum(
                int(column.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}w"))
                for column in table._tbl.tblGrid.gridCol_lst
            )
            self.assertGreater(grid_width_twips / 1440, 9.4)
        for text in (md_text, docx_text, pdf_text):
            self.assertIn("4.8 Results and Analysis", text)
            self.assertIn("0.600501", text)
            self.assertIn("63 candidate", text)
            self.assert_rejected_phrases_absent(text)

    def test_chapter_figures_and_generation_manifest_are_verified(self) -> None:
        manifest_path = OUTPUT / "CHAPTER_FOUR_GENERATION_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "CHAPTER-FOUR-GENERATED-FROM-SYNCHRONIZED-EVIDENCE")
        self.assertEqual(manifest["replay"], "full_pipeline_20260809")
        self.assertEqual(len(manifest["figures"]), 12)
        self.assertEqual(len(manifest["documents"]), 3)
        for row in manifest["figures"]:
            path = OUTPUT / row["path"]
            self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertEqual(digest(path), row["sha256"])
            with Image.open(path) as image:
                self.assertGreaterEqual(image.width, 1200)
                self.assertGreaterEqual(image.height, 700)
        expected_chapter_figures = {
            filename for filename, _ in report.FIGURE_SPECS if filename.startswith("CHAPTER4_FIGURE_")
        }
        self.assertEqual(
            {path.name for path in OUTPUT.glob("CHAPTER4_FIGURE_*.png")},
            expected_chapter_figures,
        )
        for row in manifest["documents"]:
            path = OUTPUT / row["path"]
            self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertEqual(digest(path), row["sha256"])


if __name__ == "__main__":
    unittest.main()

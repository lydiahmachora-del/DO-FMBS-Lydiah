from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import generate_fully_recorded_metrics_report as report


RUN_DIR = ROOT / "runtime_recording" / "full_pipeline_20260809"
EVALUATOR_DIR = (
    ROOT / "runtime_metrics_evidence" / "recorded_full_pipeline_biometric_evaluation"
)
ARCHITECTURE = (
    ROOT / "docs" / "runtime_metrics_architecture"
    / "DO_FMBS_PHASE_1_TO_6_SIMULATION_ARCHITECTURE.png"
)


def args_for(output_dir: Path, *, formats: str = "md,docx,pdf",
             validate_only: bool = False):
    values = [
        "--run-dir", str(RUN_DIR),
        "--evaluator-dir", str(EVALUATOR_DIR),
        "--output-dir", str(output_dir),
        "--architecture-image", str(ARCHITECTURE),
        "--formats", formats,
    ]
    if validate_only:
        values.append("--validate-inputs-only")
    return report.parse_args(values)


class FullyRecordedMetricsReportTests(unittest.TestCase):
    def test_finalized_inputs_are_synchronized(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = report.load_evidence(
                args_for(Path(directory), validate_only=True)
            )

        self.assertEqual(len(evidence.runtime_rows), 6)
        self.assertEqual(len(evidence.artifact_rows), 24)
        self.assertEqual(len(evidence.ai_rows), 7)
        self.assertEqual(len(evidence.schedule_rows), 12)
        self.assertEqual(len(evidence.evaluator_runtime_rows), 14)
        self.assertEqual(evidence.evaluator_manifest["final_status"], "SATISFIED")
        self.assertEqual(
            evidence.pa_row["status"], "SATISFIED - MANAGED PRIVACY BAND"
        )
        self.assertTrue(
            all(row["status"] == "EXECUTION-COMPLETED" for row in evidence.runtime_rows)
        )
        self.assertTrue(
            all(
                row["FRR_status"] == "ATTACK-STRESS RECORDED"
                for row in evidence.ai_rows[1:]
            )
        )
        recorded_total = sum(float(row["elapsed_seconds"]) for row in evidence.runtime_rows)
        self.assertGreater(recorded_total, 0)

    def test_markdown_contains_dense_exact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = report.load_evidence(
                args_for(Path(directory), validate_only=True)
            )
        markdown = report.build_markdown(evidence, "2026-08-09T01:00:00+03:00")

        for expected in (
            "1504/19154",
            "35/314",
            "COMPARATOR RECORDED",
            "ATTACK-STRESS RECORDED",
            "24 replay artifacts",
            f"{sum(float(row['elapsed_seconds']) for row in evidence.runtime_rows):.6f} seconds",
            "10% and 20%",
            "project protocol limits",
            "Declared project rule",
            "7.5% Wilson FAR upper cap as a safety margin below the final 10% project ceiling",
            "not universal biometric standards",
            "12 of 12 accepted",
            "### 7.1 Operational metric-evaluation runtimes",
            "Face data load and identity split",
            "Baseline comparator complete protocol",
            "DO-FMBS final complete protocol",
            "DO-FMBS membership-inference and PA evaluation",
            "DO-FMBS seven-level AI/FGSM sweep",
            "Complete operational biometric evaluation",
            "3.844164 seconds",
        ):
            self.assertIn(expected, markdown)
        report.assert_prohibited_free(markdown, "test Markdown")

    def test_prohibited_wording_guard_rejects_every_phrase(self):
        for phrase in report.PROHIBITED_REPORT_PHRASES:
            with self.subTest(phrase=phrase):
                with self.assertRaisesRegex(ValueError, "prohibited report wording"):
                    report.assert_prohibited_free(
                        f"cell value: {phrase}", "synthetic report"
                    )

    def test_all_report_formats_are_prohibited_wording_free(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = report.generate(args_for(Path(directory)))
            by_suffix = {path.suffix: path for path in outputs}

            self.assertEqual(set(by_suffix), {".md", ".docx", ".pdf"})
            self.assertTrue(
                all(path.is_file() and path.stat().st_size > 1000 for path in outputs)
            )
            report.assert_prohibited_free(
                by_suffix[".md"].read_text(encoding="utf-8"), "Markdown"
            )
            report.assert_prohibited_free(
                report.extract_docx_text(by_suffix[".docx"]), "DOCX"
            )
            report.assert_prohibited_free(
                report.extract_pdf_text(by_suffix[".pdf"]), "PDF"
            )

            from docx import Document
            import pymupdf

            document = Document(by_suffix[".docx"])
            self.assertGreaterEqual(len(document.inline_shapes), 1)
            with pymupdf.open(by_suffix[".pdf"]) as pdf:
                self.assertGreaterEqual(len(pdf), 2)
                self.assertTrue(any(page.get_images(full=True) for page in pdf))


if __name__ == "__main__":
    unittest.main()

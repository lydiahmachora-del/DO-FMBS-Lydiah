import csv
import json
import math
import unittest
from pathlib import Path

from PIL import Image

from scripts.generate_runtime_metrics_architecture_supplement import (
    ARCHITECTURE_IMAGE,
    END_TO_END_ARCHITECTURE_IMAGE,
    REPORT_DIR,
    REPORT_STEM,
    derive_tables,
    load_source_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "runtime_metrics_evidence"


class RuntimeMetricsEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = load_source_evidence()
        cls.tables = derive_tables(
            cls.source,
            {
                "median_seconds": 0.05,
                "repetitions": 5,
                "scope": "test benchmark scope",
            },
        )

    def test_phase2_exact_proxy_counts(self):
        metrics = {row["metric"]: row for row in self.tables["phase2_metrics"]}
        self.assertEqual(metrics["Face classification accuracy"]["numerator"], 25)
        self.assertEqual(metrics["Face classification accuracy"]["denominator"], 100)
        self.assertEqual(metrics["FAR proxy"]["numerator"], 577)
        self.assertEqual(metrics["FAR proxy"]["denominator"], 9412)
        self.assertAlmostEqual(metrics["FAR proxy"]["decimal_value"], 577 / 9412)
        self.assertEqual(metrics["FRR proxy"]["numerator"], 75)

    def test_phase3_ai_asr_curve_and_denominator(self):
        rows = self.tables["phase3_ai_asr"]
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["originally_correct_denominator"] == 25 for row in rows))
        fgsm = [row for row in rows if row["attack"] == "FGSM"]
        pgd = [row for row in rows if row["attack"] == "PGD"]
        self.assertEqual([row["successful_attacks"] for row in fgsm], [3, 15, 19, 25, 25])
        self.assertEqual([row["successful_attacks"] for row in pgd], [4, 17, 21, 21, 21])
        selected = next(row for row in fgsm if math.isclose(row["epsilon"], 0.10))
        self.assertAlmostEqual(selected["AI"], 1 / 3)
        self.assertAlmostEqual(selected["ASR"], 0.76)

    def test_pa_is_computed_from_membership_accuracy(self):
        pa = next(row for row in self.tables["pa_measurements"] if row["measure"].startswith("Privacy-attack"))
        self.assertAlmostEqual(pa["result"], 0.02)
        attack_accuracy = next(row for row in self.tables["pa_measurements"] if row["measure"] == "Membership attack accuracy")
        self.assertEqual(attack_accuracy["raw_or_count"], "102/200")
        false_member = next(row for row in self.tables["pa_measurements"] if row["measure"] == "False-member rate")
        self.assertIn("Not biometric FAR", false_member["interpretation"])

    def test_empirical_and_simulated_gs_are_separate(self):
        mapping = {row["participating_clients_K"]: row["GS"] for row in self.tables["gs_mapping"]}
        self.assertEqual(mapping, {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0})
        schedule = self.tables["gs_dynamic_schedule"]
        self.assertEqual(len(schedule), 12)
        self.assertEqual([row["participating_clients_K"] for row in schedule], [4, 4, 4, 5, 4, 5, 4, 4, 4, 4, 4, 5])
        self.assertTrue(all("simulated controller" in row["evidence_type"] for row in schedule))

    def test_unmeasured_values_are_not_filled(self):
        defense = self.tables["defense_comparison"]
        self.assertTrue(all(row["FAR"] == "Not measured" for row in defense))
        self.assertTrue(all(row["FRR"] == "Not measured" for row in defense))
        matrix = {row["variable"]: row for row in self.tables["metric_linkage_matrix"]}
        self.assertEqual(matrix["AI"]["FAR"], "Not stratified")
        self.assertEqual(matrix["GS"]["FRR"], "Not stratified")

    def test_historical_elapsed_runtime_is_not_invented(self):
        historical = [row for row in self.tables["execution_timing_registry"] if row["timing_type"] == "Historical artifact run"]
        self.assertEqual(len(historical), 6)
        self.assertTrue(all(row["elapsed_seconds"] == "Not recorded" for row in historical))

    def test_phase_architecture_matrix_matches_actual_dependencies(self):
        rows = self.tables["phase_architecture_matrix"]
        self.assertEqual(len(rows), 6)
        phase4 = next(row for row in rows if row["phase"].startswith("Phase 4"))
        self.assertNotIn("attack_results.pkl", phase4["actual_inputs"])
        self.assertIn("no Phase-3 artifact", phase4["dependency_qualification"])
        phase5 = next(row for row in rows if row["phase"].startswith("Phase 5"))
        self.assertIn("baseline_model.keras", phase5["actual_inputs"])
        self.assertIn("do_fmbs_model.keras", phase5["actual_inputs"])
        phase6 = next(row for row in rows if row["phase"].startswith("Phase 6"))
        self.assertEqual(phase6["evidence_type"], "Synthetic policy simulation")
        self.assertIn("does not automatically retrain", phase6["dependency_qualification"])


class RuntimeMetricsDocumentTests(unittest.TestCase):
    def test_required_outputs_exist(self):
        expected = [
            REPORT_DIR / f"{REPORT_STEM}.md",
            REPORT_DIR / f"{REPORT_STEM}.docx",
            REPORT_DIR / f"{REPORT_STEM}.pdf",
            ARCHITECTURE_IMAGE,
            END_TO_END_ARCHITECTURE_IMAGE,
            REPORT_DIR / "ARCHITECTURE_IMAGE_PROVENANCE.md",
            EVIDENCE / "evidence_manifest.json",
            EVIDENCE / "document_manifest.json",
        ]
        for path in expected:
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 100, path)

    def test_markdown_contains_key_values_and_qualifications(self):
        path = REPORT_DIR / f"{REPORT_STEM}.md"
        text = path.read_text(encoding="utf-8")
        for required in (
            "577/9412", "75/100", "AI=0.333333", "PA=0.020000", "GS=1.000000",
            "Not measured", "not a universal privacy guarantee", "63 defense-and-participation",
            "Controlled experiment required for per-variable FAR/FRR/ASR",
            "Figure 2: End-to-end DO-FMBS simulation workflow",
            "Complete simulation process from Phase 1 to Phase 6",
            "standalone synthetic policy simulation",
        ):
            self.assertIn(required, text)
        self.assertNotIn("legacy", text.lower())

    def test_machine_tables_have_expected_rows(self):
        expected_rows = {
            "phase2_metrics.csv": 4,
            "phase3_ai_asr.csv": 10,
            "phase5_ai_asr.csv": 10,
            "pa_measurements.csv": 7,
            "gs_mapping.csv": 5,
            "gs_dynamic_schedule.csv": 12,
            "defense_comparison.csv": 4,
            "runtime_output_snapshot.csv": 4,
            "future_measurement_grid.csv": 3,
            "phase_architecture_matrix.csv": 6,
        }
        for name, count in expected_rows.items():
            with (EVIDENCE / name).open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), count, name)

    def test_manifests_preserve_document_separation(self):
        evidence_manifest = json.loads((EVIDENCE / "evidence_manifest.json").read_text(encoding="utf-8"))
        document_manifest = json.loads((EVIDENCE / "document_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence_manifest["historical_elapsed_runtime"], "Not recorded for Phases 1-6")
        self.assertEqual(document_manifest["previous_report_files_modified_by_this_generator"], [])
        self.assertTrue(all("docs/constrained_dynamic_model" not in row["path"] for row in document_manifest["documents"]))
        document_paths = {row["path"] for row in document_manifest["documents"]}
        self.assertIn(
            "docs/runtime_metrics_architecture/DO_FMBS_SYSTEM_ARCHITECTURE.png",
            document_paths,
        )
        self.assertIn(
            "docs/runtime_metrics_architecture/DO_FMBS_PHASE_1_TO_6_SIMULATION_ARCHITECTURE.png",
            document_paths,
        )

    def test_architecture_image_is_high_resolution(self):
        for path in (ARCHITECTURE_IMAGE, END_TO_END_ARCHITECTURE_IMAGE):
            with Image.open(path) as image:
                self.assertGreaterEqual(image.width, 1500)
                self.assertGreaterEqual(image.height, 1000)

    def test_docx_contains_tables_equations_image_and_correct_footer(self):
        from docx import Document

        document = Document(REPORT_DIR / f"{REPORT_STEM}.docx")
        self.assertGreaterEqual(len(document.tables), 13)
        self.assertGreaterEqual(len(document.inline_shapes), 10)
        footer_text = " ".join(
            paragraph.text
            for section in document.sections
            for paragraph in section.footer.paragraphs
        )
        self.assertIn("DO-FMBS Runtime Metrics and Architecture Supplement", footer_text)
        self.assertNotIn("Constrained Dynamic Optimization Model", footer_text)

    def test_pdf_is_searchable_and_has_architecture_page(self):
        import pymupdf

        document = pymupdf.open(REPORT_DIR / f"{REPORT_STEM}.pdf")
        self.assertGreaterEqual(document.page_count, 9)
        combined_text = "\n".join(page.get_text("text") for page in document)
        self.assertIn("DO-FMBS Runtime Metrics", combined_text)
        self.assertIn("Short oral explanation", combined_text)
        self.assertIn("Figure 2: End-to-end DO-FMBS simulation workflow", combined_text)
        self.assertNotIn("DO-FMBS Phase 6", combined_text)
        self.assertGreaterEqual(len(document[-2].get_images(full=True)), 1)
        self.assertGreaterEqual(len(document[-1].get_images(full=True)), 1)


if __name__ == "__main__":
    unittest.main()

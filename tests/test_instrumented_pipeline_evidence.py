import argparse
import csv
import io
import json
import math
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import finalize_instrumented_pipeline_evidence as finalizer
from scripts import run_isolated_instrumented_pipeline as runner


ROOT = Path(__file__).resolve().parents[1]
LIVE_RUN = ROOT / "runtime_recording" / "full_pipeline_20260809"

REGISTRY_FIELDS = {
    "phase",
    "script",
    "scope",
    "command",
    "start_timestamp",
    "end_timestamp",
    "elapsed_seconds",
    "exit_code",
    "status",
    "log_path",
}
REGISTRY_METADATA = {
    "run_directory",
    "clock",
    "timezone",
    "python",
    "platform",
    "processor",
    "records",
}
FINAL_REGISTRY_METADATA = REGISTRY_METADATA | {
    "execution_status",
    "artifact_status",
    "total_elapsed_seconds",
}
ARTIFACT_FIELDS = {"phase", "artifact", "bytes", "sha256", "status"}
PROHIBITED_WORDING = (
    "not recorded",
    "not measured",
    "not satisfied",
    "undecided",
    "recorded satisfied",
    "recorded failed",
)


def normalized_status_text(text: str) -> str:
    return re.sub(r"[-_]+", " ", text.lower())


def make_records(status: str = "EXECUTION-COMPLETED") -> list[dict]:
    records = []
    for phase, spec in enumerate(runner.PHASE_SPECS, start=1):
        records.append(
            {
                "phase": phase,
                "script": spec["script"],
                "scope": spec["scope"],
                "command": f"python -u {spec['script']}",
                "start_timestamp": f"2026-08-09T00:{phase:02d}:00.000+03:00",
                "end_timestamp": f"2026-08-09T00:{phase:02d}:01.000+03:00",
                "elapsed_seconds": phase + 0.25,
                "exit_code": 0,
                "status": status,
                "log_path": f"phase{phase}_console.log",
            }
        )
    return records


def create_expected_artifacts(run_dir: Path) -> None:
    for phase, relatives in finalizer.EXPECTED.items():
        for relative in relatives:
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"phase={phase}; artifact={relative}\n".encode("utf-8"))


class InstrumentedPipelineUnitTests(unittest.TestCase):
    def test_console_fallback_replaces_unencodable_unicode(self):
        class Cp1252Console:
            encoding = "cp1252"

            def __init__(self):
                self.fragments = []

            def write(self, text):
                text.encode(self.encoding)
                self.fragments.append(text)

            def flush(self):
                return None

        console = Cp1252Console()
        with patch.object(runner.sys, "stdout", console):
            runner.write_console("Attack strength ε=0.10\n")

        rendered = "".join(console.fragments)
        self.assertEqual(rendered, "Attack strength ?=0.10\n")

    def test_phase_specs_are_complete_unique_and_configuration_locked(self):
        phases = [int(spec["phase"]) for spec in runner.PHASE_SPECS]
        scripts = [spec["script"] for spec in runner.PHASE_SPECS]
        self.assertEqual(phases, list(range(1, 7)))
        self.assertEqual(len(scripts), len(set(scripts)))

        by_phase = {int(spec["phase"]): spec for spec in runner.PHASE_SPECS}
        self.assertEqual(
            by_phase[2]["arguments"],
            [
                "--rounds", "12", "--local-epochs", "2", "--batch-size", "64",
            ],
        )
        self.assertEqual(
            by_phase[3]["arguments"],
            [
                "--epsilons", "0.01,0.05,0.10,0.20,0.30",
                "--pgd-iterations", "10", "--poison-rate", "0.10",
                "--poison-epochs", "3", "--max-test-samples", "0",
            ],
        )
        self.assertEqual(
            by_phase[4]["arguments"],
            [
                "--epochs", "10", "--batch-size", "64",
                "--noise-stddev", "0.0003", "--adversarial-epsilon", "0.05",
            ],
        )

    def test_runner_registry_schema_without_running_a_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            records = make_records()
            runner.write_registry(run_dir, records)

            payload = json.loads(
                (run_dir / "phase_runtime_registry.json").read_text(encoding="utf-8")
            )
            self.assertTrue(REGISTRY_METADATA.issubset(payload))
            self.assertEqual(payload["clock"], "time.perf_counter")
            self.assertEqual(payload["timezone"], "Africa/Nairobi")
            self.assertEqual([row["phase"] for row in payload["records"]], list(range(1, 7)))
            self.assertTrue(
                all(REGISTRY_FIELDS.issubset(row) for row in payload["records"])
            )
            self.assertTrue(all(float(row["elapsed_seconds"]) > 0 for row in payload["records"]))

            with (run_dir / "phase_runtime_registry.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertTrue(REGISTRY_FIELDS.issubset(csv_rows[0]))
            self.assertEqual([int(row["phase"]) for row in csv_rows], list(range(1, 7)))

    def test_finalizer_builds_verified_evidence_without_running_phases(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            # Exercise normalization of registries created before the execution-status rename.
            records = make_records(status="RECORDED-SATISFIED")
            runner.write_registry(run_dir, records)
            create_expected_artifacts(run_dir)

            arguments = argparse.Namespace(run_dir=run_dir)
            with patch.object(finalizer, "parse_args", return_value=arguments):
                with redirect_stdout(io.StringIO()):
                    finalizer.main()

            registry = json.loads(
                (run_dir / "phase_runtime_registry.json").read_text(encoding="utf-8")
            )
            self.assertTrue(FINAL_REGISTRY_METADATA.issubset(registry))
            self.assertEqual(registry["execution_status"], "ALL-SIX-PHASES-COMPLETED")
            self.assertEqual(
                registry["artifact_status"], "ALL-REQUIRED-ARTIFACTS-VERIFIED"
            )
            self.assertTrue(
                all(row["status"] == "EXECUTION-COMPLETED" for row in registry["records"])
            )
            expected_total = sum(float(row["elapsed_seconds"]) for row in registry["records"])
            self.assertAlmostEqual(registry["total_elapsed_seconds"], expected_total, places=12)

            manifest = json.loads(
                (run_dir / "phase_artifact_manifest.json").read_text(encoding="utf-8")
            )
            expected_pairs = {
                (phase, relative)
                for phase, relatives in finalizer.EXPECTED.items()
                for relative in relatives
            }
            actual_pairs = {
                (int(row["phase"]), row["artifact"]) for row in manifest["artifacts"]
            }
            self.assertEqual(actual_pairs, expected_pairs)
            self.assertTrue(
                all(row["status"] == "ARTIFACT-VERIFIED" for row in manifest["artifacts"])
            )
            self.assertTrue(
                all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in manifest["artifacts"])
            )

            evidence_text = normalized_status_text("\n".join(
                path.read_text(encoding="utf-8")
                for path in (
                    run_dir / "phase_runtime_registry.json",
                    run_dir / "phase_runtime_registry.csv",
                    run_dir / "phase_artifact_manifest.json",
                    run_dir / "phase_artifact_hashes.csv",
                )
            ))
            for wording in PROHIBITED_WORDING:
                self.assertNotIn(wording, evidence_text)


class FinalizedLivePipelineEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = (
            LIVE_RUN / "phase_runtime_registry.json",
            LIVE_RUN / "phase_runtime_registry.csv",
            LIVE_RUN / "phase_artifact_manifest.json",
            LIVE_RUN / "phase_artifact_hashes.csv",
        )
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest("The isolated six-phase replay has not been finalized yet")

        cls.registry_path, cls.registry_csv_path, cls.manifest_path, cls.hash_csv_path = required
        cls.registry = json.loads(cls.registry_path.read_text(encoding="utf-8"))
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        with cls.registry_csv_path.open("r", encoding="utf-8", newline="") as stream:
            cls.registry_csv = list(csv.DictReader(stream))
        with cls.hash_csv_path.open("r", encoding="utf-8", newline="") as stream:
            cls.hash_csv = list(csv.DictReader(stream))

    def test_final_registry_schema_positive_timings_and_unique_phases(self):
        self.assertTrue(FINAL_REGISTRY_METADATA.issubset(self.registry))
        records = self.registry["records"]
        self.assertEqual(len(records), 6)
        self.assertTrue(all(REGISTRY_FIELDS.issubset(row) for row in records))
        phases = [int(row["phase"]) for row in records]
        self.assertEqual(sorted(phases), list(range(1, 7)))
        self.assertEqual(len(phases), len(set(phases)))
        self.assertTrue(
            all(math.isfinite(float(row["elapsed_seconds"])) for row in records)
        )
        self.assertTrue(all(float(row["elapsed_seconds"]) > 0 for row in records))
        self.assertTrue(all(int(row["exit_code"]) == 0 for row in records))
        self.assertTrue(all(row["status"] == "EXECUTION-COMPLETED" for row in records))

    def test_csv_and_json_runtime_registries_are_synchronized(self):
        json_records = {int(row["phase"]): row for row in self.registry["records"]}
        csv_records = {int(row["phase"]): row for row in self.registry_csv}
        self.assertEqual(set(csv_records), set(json_records))
        for phase, json_row in json_records.items():
            csv_row = csv_records[phase]
            for field in REGISTRY_FIELDS - {"phase", "elapsed_seconds", "exit_code"}:
                self.assertEqual(csv_row[field], str(json_row[field]))
            self.assertAlmostEqual(
                float(csv_row["elapsed_seconds"]), float(json_row["elapsed_seconds"]), places=12
            )
            self.assertEqual(int(csv_row["exit_code"]), int(json_row["exit_code"]))

    def test_final_statuses_and_total_timing_arithmetic(self):
        self.assertEqual(self.registry["execution_status"], "ALL-SIX-PHASES-COMPLETED")
        self.assertEqual(
            self.registry["artifact_status"], "ALL-REQUIRED-ARTIFACTS-VERIFIED"
        )
        expected = sum(float(row["elapsed_seconds"]) for row in self.registry["records"])
        self.assertAlmostEqual(float(self.registry["total_elapsed_seconds"]), expected, places=9)

    def test_artifact_manifest_and_hash_csv_have_exact_verified_coverage(self):
        expected_pairs = {
            (phase, relative)
            for phase, relatives in finalizer.EXPECTED.items()
            for relative in relatives
        }
        manifest_by_pair = {
            (int(row["phase"]), row["artifact"]): row
            for row in self.manifest["artifacts"]
        }
        csv_by_pair = {
            (int(row["phase"]), row["artifact"]): row for row in self.hash_csv
        }
        self.assertEqual(set(manifest_by_pair), expected_pairs)
        self.assertEqual(set(csv_by_pair), expected_pairs)
        self.assertEqual(self.manifest["status"], "ALL-REQUIRED-ARTIFACTS-VERIFIED")

        for pair, row in manifest_by_pair.items():
            phase, relative = pair
            self.assertTrue(ARTIFACT_FIELDS.issubset(row))
            self.assertEqual(row["status"], "ARTIFACT-VERIFIED")
            self.assertGreater(int(row["bytes"]), 0)
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")
            artifact = LIVE_RUN / relative
            self.assertTrue(artifact.is_file(), f"Phase {phase}: {artifact}")
            self.assertEqual(artifact.stat().st_size, int(row["bytes"]))
            self.assertEqual(finalizer.sha256(artifact), row["sha256"])

            csv_row = csv_by_pair[pair]
            self.assertEqual(int(csv_row["bytes"]), int(row["bytes"]))
            self.assertEqual(csv_row["sha256"], row["sha256"])
            self.assertEqual(csv_row["status"], row["status"])

    def test_final_evidence_contains_no_prohibited_status_wording(self):
        evidence_text = normalized_status_text("\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                self.registry_path,
                self.registry_csv_path,
                self.manifest_path,
                self.hash_csv_path,
            )
        ))
        for wording in PROHIBITED_WORDING:
            self.assertNotIn(wording, evidence_text)


if __name__ == "__main__":
    unittest.main()

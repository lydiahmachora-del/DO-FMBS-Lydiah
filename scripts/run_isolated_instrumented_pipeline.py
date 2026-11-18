"""Run DO-FMBS Phases 1-6 in isolation and record real elapsed durations.

The runner copies the six phase entry points into a new run directory so their
hard-coded relative output paths cannot overwrite the project's existing model
or evidence artifacts.  Every phase receives a start timestamp, end timestamp,
``perf_counter`` duration, exit status, command and complete console log.

Example::

    python scripts/run_isolated_instrumented_pipeline.py \
        --run-dir runtime_recording/full_pipeline_20260809
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runtime_recording" / "final_scientific_replay_20260907"
PHASE_SPECS = [
    {
        "phase": 1,
        "script": "phase1_data_preparation.py",
        "arguments": [],
        "scope": "LFW loading, fingerprint generation, preparation and saved client partitions",
    },
    {
        "phase": 2,
        "script": "phase2_federated_learning.py",
        "arguments": [
            "--rounds", "12", "--local-epochs", "2", "--batch-size", "64",
        ],
        "scope": "Five-client federated baseline with the presentation configuration",
    },
    {
        "phase": 3,
        "script": "phase3_adversarial_attacks.py",
        "arguments": [
            "--epsilons", "0.01,0.05,0.10,0.20,0.30",
            "--pgd-iterations", "10", "--poison-rate", "0.10",
            "--poison-epochs", "3", "--max-test-samples", "0",
        ],
        "scope": "FGSM, PGD, membership inference and label-flipping poisoning",
    },
    {
        "phase": 4,
        "script": "phase4_defense_mechanisms.py",
        "arguments": [
            "--epochs", "10", "--batch-size", "64",
            "--noise-stddev", "0.0003", "--adversarial-epsilon", "0.05",
        ],
        "scope": "Baseline, DP-inspired gradient-noise, adversarial-training and combined-model comparison",
    },
    {
        "phase": 5,
        "script": "phase5_model_validation.py",
        "arguments": ["--bootstrap-repetitions", "2000"],
        "scope": "Independent classification, calibrated verification, robustness and statistical validation",
    },
    {
        "phase": 6,
        "script": "phase6_constrained_dynamic_optimization.py",
        "arguments": [],
        "scope": "Synthetic factorial generation, constrained fit and dynamic control",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated, fully timed DO-FMBS Phase 1-6 reproduction."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--start-phase", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--end-phase", type=int, choices=range(1, 7), default=6)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip phases already recorded with exit code zero in the timing JSON.",
    )
    parser.add_argument(
        "--rerun-completed", action="store_true",
        help="With --resume, rerun the selected completed phase(s) and replace only their timing records.",
    )
    return parser.parse_args()


def copy_runtime_sources(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scripts").mkdir(parents=True, exist_ok=True)
    for name in [f"phase{phase}_" for phase in range(1, 7)]:
        matches = list(ROOT.glob(name + "*.py"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one source matching {name}*.py; found {matches}")
        shutil.copy2(matches[0], run_dir / matches[0].name)
    shutil.copy2(ROOT / "do_fmbs_core.py", run_dir / "do_fmbs_core.py")
    shutil.copy2(
        ROOT / "scripts" / "constrained_dynamic_model.py",
        run_dir / "scripts" / "constrained_dynamic_model.py",
    )


def write_registry(run_dir: Path, records: list[dict]) -> None:
    json_path = run_dir / "phase_runtime_registry.json"
    json_path.write_text(
        json.dumps(
            {
                "run_directory": str(run_dir),
                "clock": "time.perf_counter",
                "timezone": "Africa/Nairobi",
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor() or "unreported",
                "records": records,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    csv_path = run_dir / "phase_runtime_registry.csv"
    fields = [
        "phase", "script", "scope", "command", "start_timestamp",
        "end_timestamp", "elapsed_seconds", "exit_code", "status", "log_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record.get(field, "") for field in fields} for record in records)


def existing_records(run_dir: Path) -> list[dict]:
    path = run_dir / "phase_runtime_registry.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def write_console(text: str) -> None:
    """Echo child output without failing on a narrow Windows code page."""
    try:
        print(text, end="", flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
        print(safe_text, end="", flush=True)


def run_phase(run_dir: Path, spec: dict) -> dict:
    command = [sys.executable, "-u", spec["script"], *spec["arguments"]]
    log_path = run_dir / f"phase{spec['phase']}_console.log"
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
    environment["TF_NUM_INTRAOP_THREADS"] = "2"
    environment["TF_NUM_INTEROP_THREADS"] = "2"
    environment["DO_FMBS_REPLAY_ID"] = run_dir.name
    started_at = datetime.now(ZoneInfo("Africa/Nairobi"))
    start_clock = time.perf_counter()
    print(f"\n[TIMING] Phase {spec['phase']} started at {started_at.isoformat(timespec='seconds')}", flush=True)
    print("[TIMING] Command: " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            write_console(line)
        exit_code = process.wait()
    elapsed = time.perf_counter() - start_clock
    ended_at = datetime.now(ZoneInfo("Africa/Nairobi"))
    status = "EXECUTION-COMPLETED" if exit_code == 0 else "EXECUTION-FAILED"
    record = {
        "phase": spec["phase"],
        "script": spec["script"],
        "scope": spec["scope"],
        "command": " ".join(command),
        "start_timestamp": started_at.isoformat(timespec="milliseconds"),
        "end_timestamp": ended_at.isoformat(timespec="milliseconds"),
        "elapsed_seconds": elapsed,
        "exit_code": exit_code,
        "status": status,
        "log_path": str(log_path.relative_to(run_dir)).replace("\\", "/"),
    }
    print(
        f"[TIMING] Phase {spec['phase']} ended with {status}; elapsed={elapsed:.3f} seconds",
        flush=True,
    )
    return record


def main() -> None:
    args = parse_args()
    if args.start_phase > args.end_phase:
        raise ValueError("start phase must not exceed end phase")
    if args.rerun_completed and not args.resume:
        raise ValueError("--rerun-completed requires --resume to retain existing phase records")
    run_dir = args.run_dir.resolve()
    copy_runtime_sources(run_dir)
    records = existing_records(run_dir) if args.resume else []
    completed = {
        int(record["phase"])
        for record in records
        if int(record.get("exit_code", 1)) == 0
    }
    for spec in PHASE_SPECS:
        phase = int(spec["phase"])
        if not args.start_phase <= phase <= args.end_phase:
            continue
        if args.resume and phase in completed and not args.rerun_completed:
            print(f"[TIMING] Phase {phase} already completed; skipping under --resume", flush=True)
            continue
        record = run_phase(run_dir, spec)
        records = [row for row in records if int(row["phase"]) != phase]
        records.append(record)
        records.sort(key=lambda row: int(row["phase"]))
        write_registry(run_dir, records)
        if record["exit_code"] != 0:
            raise SystemExit(record["exit_code"])
    write_registry(run_dir, records)
    print(f"\n[TIMING] Complete registry: {run_dir / 'phase_runtime_registry.csv'}", flush=True)


if __name__ == "__main__":
    main()

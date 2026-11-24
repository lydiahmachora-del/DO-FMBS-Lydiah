"""Isolated, configuration-aware runner for interactive DO-FMBS replays."""
from __future__ import annotations

import argparse
import csv
import hashlib
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

from phase_configuration import CONFIG_VERSION, cli_arguments, defaults_for_all_phases, validate_phase_config

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runtime_recording"
PHASE_SPECS = [
    (1, "phase1_data_preparation.py", "LFW loading, fingerprint generation, preparation and saved client partitions"),
    (2, "phase2_federated_learning.py", "Federated two-modality baseline training across persisted clients"),
    (3, "phase3_adversarial_attacks.py", "FGSM, PGD, membership inference and label-flipping poisoning"),
    (4, "phase4_defense_mechanisms.py", "Gradient-noise, adversarial-training and combined-defense comparison"),
    (5, "phase5_model_validation.py", "Classification, verification, robustness and statistical validation"),
    (6, "phase6_constrained_dynamic_optimization.py", "Synthetic constrained response model and dynamic control schedule"),
]


def copy_runtime_sources(run_dir: Path) -> None:
    (run_dir / "scripts").mkdir(parents=True, exist_ok=True)
    for _, script, _ in PHASE_SPECS:
        shutil.copy2(ROOT / script, run_dir / script)
    shutil.copy2(ROOT / "do_fmbs_core.py", run_dir / "do_fmbs_core.py")
    shutil.copy2(ROOT / "phase_configuration.py", run_dir / "phase_configuration.py")
    shutil.copy2(ROOT / "scripts" / "constrained_dynamic_model.py",
                 run_dir / "scripts" / "constrained_dynamic_model.py")


def load_configuration(path: Path | None, run_dir: Path) -> dict:
    if path:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
        if payload.get("configuration_version") != CONFIG_VERSION:
            raise ValueError("Unsupported phase configuration version")
        phases = payload.get("phases", {})
    else:
        phases = defaults_for_all_phases()
    validated = {f"phase{number}": validate_phase_config(number, phases.get(f"phase{number}", {}))
                 for number in range(1, 7)}
    return {"run_id": run_dir.name, "configuration_version": CONFIG_VERSION,
            "created_at": datetime.now(ZoneInfo("Africa/Nairobi")).isoformat(timespec="seconds"),
            "default_preset": "research_defaults", "phases": validated}


def write_registry(run_dir: Path, records: list[dict]) -> None:
    payload = {"run_directory": str(run_dir), "clock": "time.perf_counter",
               "timezone": "Africa/Nairobi", "python": sys.version,
               "platform": platform.platform(), "processor": platform.processor() or "unreported",
               "records": records}
    (run_dir / "phase_runtime_registry.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = ["phase", "script", "scope", "command", "start_timestamp", "end_timestamp",
              "elapsed_seconds", "exit_code", "status", "log_path", "configuration_version", "configuration"]
    with (run_dir / "phase_runtime_registry.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: json.dumps(record[field]) if field in {"configuration"} and field in record
                          else record.get(field, "") for field in fields} for record in records)


def write_experiment_manifest(run_dir: Path, records: list[dict], configurations: dict) -> None:
    complete = {int(row.get("phase", 0)) for row in records
                if row.get("status") == "EXECUTION-COMPLETED" and int(row.get("exit_code", 1)) == 0}
    if complete != set(range(1, 7)):
        return
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "interactive_experiment_manifest.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append({"path": path.relative_to(run_dir).as_posix(),
                          "bytes": path.stat().st_size, "sha256": digest})
    payload = {"status": "COMPLETED-EXPLORATORY-NOT-AUTHORITATIVE",
               "run_id": run_dir.name,
               "created_at": datetime.now(ZoneInfo("Africa/Nairobi")).isoformat(timespec="seconds"),
               "configuration_version": CONFIG_VERSION,
               "scientific_status": "Exploratory configurable execution; not an accepted thesis run or independent scientific audit.",
               "configuration": configurations, "phase_records": records,
               "artifacts": artifacts,
               "validation": {"all_six_phases_completed": True,
                              "artifact_inventory_with_sha256": True,
                              "independent_authoritative_audit": False}}
    (run_dir / "interactive_experiment_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_phase(run_dir: Path, phase: int, script: str, scope: str, config: dict) -> dict:
    command = [sys.executable, "-u", script, *cli_arguments(phase, config)]
    started_at = datetime.now(ZoneInfo("Africa/Nairobi"))
    started = time.perf_counter()
    log_path = run_dir / f"phase{phase}_console.log"
    phase_metadata = {"phase": phase, "configuration_version": CONFIG_VERSION,
                      "default_preset": "research_defaults",
                      "configuration": config,
                      "changed_parameters": [key for key, value in config.items()
                                             if value != defaults_for_all_phases()[f"phase{phase}"][key]],
                      "status": "RUNNING", "started_at": started_at.isoformat(timespec="seconds"),
                      "command": command}
    phase_config_path = run_dir / f"phase{phase}_config.json"
    phase_config_path.write_text(json.dumps(phase_metadata, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2",
                "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2",
                "DO_FMBS_REPLAY_ID": run_dir.name})
    print(f"\n[TIMING] Phase {phase} started at {started_at.isoformat(timespec='seconds')}", flush=True)
    print("[TIMING] Command: " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(command, cwd=run_dir, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        code = process.wait()
    ended_at = datetime.now(ZoneInfo("Africa/Nairobi"))
    status = "EXECUTION-COMPLETED" if code == 0 else "EXECUTION-FAILED"
    elapsed = time.perf_counter() - started
    phase_metadata.update({"status": status, "finished_at": ended_at.isoformat(timespec="seconds"),
                           "exit_code": code, "elapsed_seconds": elapsed})
    phase_config_path.write_text(json.dumps(phase_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[TIMING] Phase {phase} ended with {status}; elapsed={elapsed:.3f} seconds", flush=True)
    return {"phase": phase, "script": script, "scope": scope,
            "command": " ".join(command), "start_timestamp": started_at.isoformat(timespec="milliseconds"),
            "end_timestamp": ended_at.isoformat(timespec="milliseconds"), "elapsed_seconds": elapsed,
            "exit_code": code, "status": status, "log_path": f"phase{phase}_console.log",
            "configuration_version": CONFIG_VERSION, "configuration": config}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start-phase", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--end-phase", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--config-file", type=Path)
    args = parser.parse_args()
    if args.start_phase > args.end_phase:
        raise ValueError("start phase must not exceed end phase")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    configurations = load_configuration(args.config_file, run_dir)
    (run_dir / "run_config.json").write_text(json.dumps(configurations, indent=2) + "\n", encoding="utf-8")
    copy_runtime_sources(run_dir)
    registry = run_dir / "phase_runtime_registry.json"
    records = json.loads(registry.read_text(encoding="utf-8")).get("records", []) if args.resume and registry.exists() else []
    completed = {int(row["phase"]) for row in records
                 if row.get("status") == "EXECUTION-COMPLETED" and int(row.get("exit_code", 1)) == 0}
    for phase, script, scope in PHASE_SPECS:
        if not args.start_phase <= phase <= args.end_phase or (phase in completed and not args.rerun_completed):
            continue
        record = run_phase(run_dir, phase, script, scope, configurations["phases"][f"phase{phase}"])
        records = [row for row in records if int(row["phase"]) != phase]
        records.append(record)
        records.sort(key=lambda row: int(row["phase"]))
        write_registry(run_dir, records)
        if record["exit_code"] != 0:
            raise SystemExit(record["exit_code"])
    write_registry(run_dir, records)
    write_experiment_manifest(run_dir, records, configurations)
    print(f"\n[TIMING] Complete registry: {run_dir / 'phase_runtime_registry.csv'}", flush=True)


if __name__ == "__main__":
    main()

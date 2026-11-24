"""Interactive entry point for an isolated, timed six-phase demonstration."""
from __future__ import annotations

import shutil
from pathlib import Path

from scripts import run_isolated_instrumented_pipeline as pipeline


def copy_runtime_sources(run_dir: Path) -> None:
    """Copy only the phase scripts named by the runner's execution plan."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scripts").mkdir(parents=True, exist_ok=True)
    for spec in pipeline.PHASE_SPECS:
        source = pipeline.ROOT / spec["script"]
        if not source.is_file():
            raise FileNotFoundError(f"Phase {spec['phase']} source is missing: {source}")
        shutil.copy2(source, run_dir / source.name)
    shutil.copy2(pipeline.ROOT / "do_fmbs_core.py", run_dir / "do_fmbs_core.py")
    helper = pipeline.ROOT / "scripts" / "constrained_dynamic_model.py"
    shutil.copy2(helper, run_dir / "scripts" / helper.name)


pipeline.copy_runtime_sources = copy_runtime_sources


if __name__ == "__main__":
    pipeline.main()

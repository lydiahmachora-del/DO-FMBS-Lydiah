"""Safely archive legacy generated artifacts without touching source or caches.

The command is a dry run unless ``--execute`` is supplied.  It uses an explicit
allowlist and moves files into a timestamped archive so recovery remains easy.
New scientific replays normally run in isolated ``runtime_recording`` folders,
so archiving is optional rather than a prerequisite for contamination control.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = [
    ROOT / "do_fmbs_data",
    *ROOT.glob("do_fmbs_*.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Perform recoverable moves; default is preview only")
    parser.add_argument("--archive-root", type=Path, default=ROOT / "artifact_archive")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_root = args.archive_root.resolve()
    workspace = ROOT.resolve()
    try:
        archive_root.relative_to(workspace)
    except ValueError as error:
        raise ValueError("Archive root must remain inside the project workspace") from error
    targets = [path.resolve() for path in ALLOWLIST if path.exists()]
    for target in targets:
        try:
            target.relative_to(workspace)
        except ValueError as error:
            raise RuntimeError(f"Refusing out-of-workspace target: {target}") from error
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = archive_root / f"legacy_generated_{stamp}"
    for target in targets:
        print(f"{'MOVE' if args.execute else 'WOULD MOVE'} {target.relative_to(workspace)} -> {destination.relative_to(workspace)}")
    if not args.execute:
        print("Dry run only. Re-run with --execute to create the recoverable archive.")
        return 0
    destination.mkdir(parents=True, exist_ok=False)
    for target in targets:
        relative = target.relative_to(workspace)
        final = destination / relative
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(final))
    print(f"Archived {len(targets)} allowlisted targets to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

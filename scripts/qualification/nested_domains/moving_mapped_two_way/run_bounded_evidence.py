#!/usr/bin/env python3
"""Moving mapped public profiles: native ownership, closure, threading and restart."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("evidence directory must be empty")
    binary = args.binary.resolve()
    identity = sha(binary)
    inputs = sorted((ROOT / "tests/fixtures/configs/qualification").glob("moving_mapped_nested_*_ready.yaml"))
    inputs += [ROOT / "tests/fixtures/configs/qualification/soundings/moving_two_way_thermal.txt",
               ROOT / "scripts/tests/science/test_moving_two_way_nested_ready.py",
               ROOT / "scripts/tests/science/test_native_nested_motion_ready.py", Path(__file__).resolve()]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    results = {}
    full_profiles = [("lin", "schar3", "weno"), ("kessler", "bell2", "tvd")]
    with tempfile.TemporaryDirectory(prefix=".moving-mapped-matrix-", dir=binary.parent) as temporary:
        snapshot = Path(temporary) / "nimbus_sim"
        shutil.copy2(binary, snapshot)

        def execute(physics: str, terrain: str, advection: str) -> tuple[str, dict]:
            label = f"{physics}-{terrain}-{advection}"
            coverage = "full" if (physics, terrain, advection) in full_profiles else "synchronized"
            command = [sys.executable, str(ROOT / "scripts/tests/science/test_native_nested_motion_ready.py"),
                       "--binary", str(snapshot), "--family", "moving", "--terrain", terrain,
                       "--microphysics", physics, "--advection", advection, "--restart-coverage", coverage,
                       "--output", str(output / label)]
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            with (output / f"{label}.log").open("w") as log:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                return label, {"passed": False, "returncode": completed.returncode}
            summary = json.loads((output / label / "summary.json").read_text())
            if summary["binary_sha256"] != identity:
                raise RuntimeError(f"binary identity changed in {label}")
            return label, {"passed": True, "summary": summary}

        with ThreadPoolExecutor(max_workers=args.jobs) as workers:
            profiles = [(physics, terrain, advection) for physics in ("kessler", "lin")
                        for terrain in ("bell2", "bell3", "schar2", "schar3")
                        for advection in ("tvd", "weno")]
            profiles = full_profiles + [profile for profile in profiles if profile not in full_profiles]
            pending = [workers.submit(execute, *profile) for profile in profiles]
            try:
                for future in as_completed(pending):
                    label, result = future.result()
                    results[label] = result
                    (output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
                    print(f"{'PASS' if result['passed'] else 'FAIL'} {label}", flush=True)
            except KeyboardInterrupt:
                for future in pending:
                    future.cancel()
                raise
    if sha(binary) != identity or any(sha(ROOT / name) != expected for name, expected in hashes.items()):
        raise RuntimeError("qualification inputs changed")
    passed = all(result["passed"] for result in results.values())
    (output / "summary.json").write_text(json.dumps({
        "bounded_matrix_passed": passed, "binary_sha256": identity, "input_sha256": hashes,
        "groups": results, "backend_cases": 64, "campaign_validated": False, "externally_reviewed": False,
    }, indent=2) + "\n")
    if not passed:
        raise SystemExit("moving mapped matrix has failed groups")


if __name__ == "__main__":
    main()

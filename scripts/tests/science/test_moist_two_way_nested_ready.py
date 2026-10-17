#!/usr/bin/env python3
"""Bounded stationary CPU moist feedback evidence on one immutable executable."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
PROFILES = {
    "humid-flat": "stationary_nested_moist_cpu_two_way_ready.yaml",
    "condensing-flat": "stationary_nested_condensing_cpu_two_way_ready.yaml",
    **{terrain: f"stationary_nested_condensing_{terrain}_cpu_two_way_ready.yaml"
       for terrain in ("bell", "schar", "bell3", "schar3")},
}
PROFILES.update({f"lin-{label}": config.replace("_cpu_", "_lin_cpu_")
                 for label, config in list(PROFILES.items()) if label != "humid-flat"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("moist feedback evidence directory must be empty")
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix="nimbus-moist-feedback-") as temporary:
        snapshot = Path(temporary) / "nimbus_sim"
        shutil.copy2(binary, snapshot)
        identity = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        results = {}
        for label, config in PROFILES.items():
            command = [sys.executable, str(ROOT / "scripts/tests/science/test_two_way_nested_ready.py"),
                       "--binary", str(snapshot), "--config",
                       str(ROOT / "tests/fixtures/configs/qualification" / config),
                       "--output", str(output / label)]
            result = subprocess.run(command, cwd=ROOT, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (output / f"{label}.log").write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(f"{label} moist feedback failed:\n{result.stdout[-6000:]}")
            evidence = json.loads((output / label / "summary.json").read_text())
            if evidence["binary_sha256"] != identity:
                raise RuntimeError("moist feedback binary identity changed")
            results[label] = evidence
            print(f"passed {label}: closure, bounded winds, exact threading and restart", flush=True)
        (output / "summary.json").write_text(json.dumps({
            "binary_sha256": identity, "results": results,
            "campaign_validated": False, "externally_reviewed": False,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()

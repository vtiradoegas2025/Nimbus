#!/usr/bin/env python3
"""Existing stationary feedback families sharing the repaired reflux audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("regression evidence directory must be empty")
    summaries = {}
    for label, terrain, family, advection in (
        ("flat-tvd", "flat", "all", "tvd"),
        ("flat-weno", "flat", "all", "weno"),
        ("mapped-schar3-lin-weno", "schar3", "lin", "weno"),
    ):
        command = [sys.executable, str(ROOT / "scripts/tests/science/test_native_two_way_nested_ready.py"),
                   "--binary", str(args.binary.resolve()), "--terrain", terrain, "--family", family,
                   "--advection", advection, "--output", str(output / label)]
        with (output / f"{label}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        summaries[label] = json.loads((output / label / "summary.json").read_text())
        print(f"PASS {label}", flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()

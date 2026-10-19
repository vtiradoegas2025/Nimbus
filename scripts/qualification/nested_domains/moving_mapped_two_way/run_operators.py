#!/usr/bin/env python3
"""Bounded geometry, relocation, audit, checkpoint and public-selection gates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
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
        raise RuntimeError("operator evidence directory must be empty")
    results = {}
    for name, executable, filters in (
        ("configuration", "test_init_config_v1", []),
        ("geometry", "test_core_nested_terrain_geometry", []),
        ("feedback", "test_core_nested_grid", []),
        ("movement", "test_core_nested_production_step", ["[movement]"]),
        ("checkpoint", "test_core_simulation_checkpoint", []),
        ("output", "test_core_output", []),
        ("pressure-projection", "test_diagnostics_pressure_projection", []),
        ("resolved-diagnostics", "test_diagnostics_resolved", []),
        ("cartesian-dynamics", "test_dynamics_cartesian", []),
        ("trajectories", "test_diagnostics_trajectories", []),
        ("potential-vorticity", "test_diagnostics_potential_vorticity", []),
    ):
        binary = ROOT / "bin" / executable
        identity = hashlib.sha256(binary.read_bytes()).hexdigest()
        with (output / f"{name}.log").open("w") as log:
            subprocess.run([str(binary), *filters], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if hashlib.sha256(binary.read_bytes()).hexdigest() != identity:
            raise RuntimeError(f"{name} executable changed")
        counts = re.search(r"All tests passed \((\d+) assertions? in (\d+) test cases?\)",
                           (output / f"{name}.log").read_text())
        if not counts or min(map(int, counts.groups())) <= 0:
            raise RuntimeError(f"{name} did not execute its bounded checks")
        results[name] = {"passed": True, "executable_sha256": identity,
                         "assertions": int(counts[1]), "test_cases": int(counts[2])}
        print(f"PASS {name}", flush=True)
    command = [sys.executable, str(ROOT / "scripts/tests/science/test_nested_terrain_checkpoint_compatibility.py"),
               "--binary", str(args.binary.resolve()), "--output", str(output / "historical-checkpoint")]
    with (output / "historical-checkpoint.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    results["historical-checkpoint"] = json.loads((output / "historical-checkpoint/summary.json").read_text())
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

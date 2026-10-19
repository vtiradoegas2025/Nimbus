#!/usr/bin/env python3
"""Bounded public CPU/Metal/Vulkan Earth-fixed runtime and parity evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microphysics", choices=("kessler", "lin"), default="kessler")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("native Earth-fixed evidence directory must be empty")
    binary = args.binary.resolve()
    identity = hashlib.sha256(binary.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix=".earth-ready-", dir=binary.parent) as temporary:
        snapshot = Path(temporary) / "nimbus_sim"
        shutil.copy2(binary, snapshot)
        fixtures = ROOT / "tests/fixtures/configs/qualification"
        results = {}
        prefix = "earth_fixed_terrain_motion_" + ("lin_" if args.microphysics == "lin" else "")
        cpu_suffix = "cpu" if args.microphysics == "lin" else "diagnostics"
        for label, config in (
            ("cpu", f"{prefix}{cpu_suffix}_ready.yaml"),
            ("metal", f"{prefix}metal_ready.yaml"),
            ("vulkan", f"{prefix}vulkan_ready.yaml"),
            ("vulkan-staged", f"{prefix}vulkan_ready.yaml"),
        ):
            command = [sys.executable, str(ROOT / "scripts/tests/science/test_earth_fixed_terrain_motion_ready.py"),
                       "--binary", str(snapshot), "--config", str(fixtures / config), "--output", str(output / label)]
            if label != "cpu":
                command += ["--cpu-reference", str(output / "cpu/one")]
            environment = os.environ.copy()
            environment.pop("NIMBUS_VULKAN_MEMORY_POLICY", None)
            if label == "vulkan-staged":
                environment["NIMBUS_VULKAN_MEMORY_POLICY"] = "staged"
            result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (output / f"{label}.log").write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(f"{label} Earth-fixed gate failed:\n{result.stdout[-6000:]}")
            evidence = json.loads((output / label / "result.json").read_text())
            if evidence["binary_sha256"] != identity:
                raise RuntimeError("Earth-fixed gate changed its binary identity")
            if label == "vulkan-staged" and "memory plan=staged_forced" not in (output / label / "one.log").read_text():
                raise RuntimeError("staged Earth-fixed gate did not use staged device memory")
            results[label] = evidence
            print(f"passed {label}: public runtime, conservation, exact threading/restart and parity", flush=True)
        (output / "summary.json").write_text(json.dumps({"binary_sha256": identity, "results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()

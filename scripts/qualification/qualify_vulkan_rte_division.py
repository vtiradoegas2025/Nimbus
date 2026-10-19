#!/usr/bin/env python3
"""Bounded GPU arithmetic check; this is not a long campaign or external science review."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dev/shaders"))
import division_workaround


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nimbus-rte-division-") as directory:
        temp = Path(directory)
        library = division_workaround.build_library(ROOT / "vulkan/shaders/support/exact_division.glsl", temp)
        original, lowered = temp / "original.spv", temp / "lowered.spv"
        subprocess.run(["glslangValidator", "-V", "-Os", str(ROOT / "tests/fixtures/vulkan/rte_division.comp"), "-o", str(original)], check=True)
        if division_workaround.lower(original, library, lowered, temp) != 1:
            raise RuntimeError("arithmetic fixture must exercise exactly one lowered division")
        executable = temp / "probe"
        subprocess.run([args.cxx, "-O2", "-std=c++17", str(ROOT / "tests/vulkan/rte_division_probe.cpp"), "-lvulkan", "-o", str(executable)], check=True)
        result = subprocess.run([str(executable), str(lowered)], text=True, capture_output=True, timeout=120)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        evidence = {"strategy": division_workaround.STRATEGY, "exit_code": result.returncode, "stderr": result.stderr, "stdout": result.stdout}
        if result.returncode == 0:
            evidence["result"] = json.loads(result.stdout)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence))
        return 0 if result.returncode == 0 and evidence["result"]["mismatches"] == 0 and evidence["result"]["pairs"] == 8388608 else 1


if __name__ == "__main__":
    raise SystemExit(main())

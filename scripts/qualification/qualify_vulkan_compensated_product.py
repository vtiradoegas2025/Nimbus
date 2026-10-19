#!/usr/bin/env python3
"""Bounded native product-residual check, not campaign or scientific qualification."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dev/shaders"))
import verify_spirv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    fixture = ROOT / "tests/fixtures/vulkan/compensated_product.comp"
    probe = ROOT / "tests/vulkan/rte_division_probe.cpp"
    helper = ROOT / "vulkan/shaders/compute/microphysics/compensated_fma.glsl"
    with tempfile.TemporaryDirectory(prefix="nimbus-compensated-product-") as directory:
        temp = Path(directory)
        artifact = temp / "product.spv"
        verify_spirv.compile_shader(fixture, artifact)
        verify_spirv.validate(artifact)
        executable = temp / "probe"
        command = [args.cxx, "-O2", "-std=c++17", "-ffp-contract=off",
                   str(probe), "-lvulkan", "-o", str(executable)]
        subprocess.run(command, check=True)
        result = subprocess.run([str(executable), str(artifact), "--product-error"],
                                text=True, capture_output=True, timeout=120)
        evidence = {
            "scope": "bounded-float32-product-residual",
            "reference": "std::fma(a,b,-volatile_float_product), round-to-nearest-even",
            "comparison": "exact Float32 bits including signed zero; NaN payloads equivalent",
            "exit_code": result.returncode,
            "stderr": result.stderr,
            "stdout": result.stdout,
            "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (fixture, probe, helper)},
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        if result.returncode == 0:
            evidence["result"] = json.loads(result.stdout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence))
        return 0 if result.returncode == 0 and evidence["result"]["mismatches"] == 0 and evidence["result"]["product_mismatches"] == 0 and evidence["result"]["pairs"] == 1048576 else 1


if __name__ == "__main__":
    raise SystemExit(main())

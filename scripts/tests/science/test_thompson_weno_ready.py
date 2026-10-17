#!/usr/bin/env python3
"""Maintained WENO/Thompson/YSU family with optional implicit diffusion."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for implicit in (False, True):
        command = [sys.executable, str(ROOT / "scripts/tests/science/test_thompson_accelerator_ready.py"),
                   "--binary", str(args.binary.resolve()), "--weno"]
        if implicit:
            command.append("--implicit")
        if args.output:
            command.extend(["--output", str(args.output.resolve() / ("implicit" if implicit else "explicit"))])
        subprocess.run(command, cwd=ROOT, check=True)
    print("[thompson-weno-ready] PASS: CPU/Metal/Vulkan WENO/Thompson/YSU, with and without implicit diffusion")


if __name__ == "__main__":
    main()

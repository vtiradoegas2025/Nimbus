#!/usr/bin/env python3
"""Maintained strict WENO5 accelerator continuation class."""
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
    for backend in ("metal", "vulkan"):
        command = [sys.executable, str(ROOT / "scripts/tests/science/test_native_weno_ready.py"),
                   "--backend", backend, "--binary", str(args.binary.resolve())]
        if args.output:
            command.extend(["--output", str(args.output.resolve()/backend)])
        if backend == "vulkan":
            command.insert(0, str(ROOT / "scripts/run_with_vulkan_env.sh"))
        subprocess.run(command, cwd=ROOT, check=True)
    print("[weno-native-family-ready] PASS: strict native transport, CPU parity, conservation, threading and restart")


if __name__ == "__main__":
    main()

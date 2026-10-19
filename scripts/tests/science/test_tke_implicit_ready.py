#!/usr/bin/env python3
"""Maintained restart class for the coupled CPU/Metal/Vulkan reference."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    args = parser.parse_args()
    for backend in ("metal", "vulkan"):
        command = [sys.executable, str(ROOT / "scripts/tests/science/test_native_tke_ready.py"),
                   "--implicit", "--backend", backend, "--binary", str(args.binary.resolve())]
        if backend == "vulkan":
            command.insert(0, str(ROOT / "scripts/run_with_vulkan_env.sh"))
        subprocess.run(command, cwd=ROOT, check=True)
    print("[tke-implicit-ready] PASS: coupled CPU/Metal/Vulkan parity, conservation, threading and restart")


if __name__ == "__main__":
    main()

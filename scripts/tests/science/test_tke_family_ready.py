#!/usr/bin/env python3
"""Shared runner for maintained TKE families with optional Thompson and WENO."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main(*, thompson=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weno", action="store_true")
    args = parser.parse_args()
    for implicit in (False, True):
        for backend in ("metal", "vulkan"):
            command = [sys.executable, str(ROOT / "scripts/tests/science/test_native_tke_ready.py"),
                       "--backend", backend, "--binary", str(args.binary.resolve())]
            if thompson:
                command.append("--thompson")
            if args.weno:
                command.append("--weno")
            if implicit:
                command.append("--implicit")
            if args.output:
                command.extend(["--output", str(args.output.resolve() / f"{backend}-{'implicit' if implicit else 'explicit'}")])
            if backend == "vulkan":
                command.insert(0, str(ROOT / "scripts/run_with_vulkan_env.sh"))
            subprocess.run(command, cwd=ROOT, check=True)
    print(f"[tke-family-ready] PASS: CPU/Metal/Vulkan coupling with and without implicit diffusion; Thompson={thompson}, WENO={args.weno}")


if __name__ == "__main__":
    main()

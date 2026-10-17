#!/usr/bin/env python3
"""Compile and execute the maintained source-level extension fixture."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    fixture = (
        root
        / "tests"
        / "fixtures"
        / "external_scheme"
        / "minimal_chaos_scheme.cpp"
    )

    with tempfile.TemporaryDirectory(prefix="nimbus-external-scheme-") as temp_dir:
        binary = Path(temp_dir) / "minimal_chaos_scheme"
        compile_result = subprocess.run(
            shlex.split(args.compiler)
            + [
                "-std=c++17",
                "-I",
                str(root / "include"),
                str(fixture),
                "-o",
                str(binary),
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            print("[external-scheme] FAIL: public-only fixture did not compile")
            print(compile_result.stdout, end="")
            print(compile_result.stderr, end="")
            return compile_result.returncode

        run_result = subprocess.run(binary, cwd=root, check=False)
        if run_result.returncode != 0:
            print("[external-scheme] FAIL: fixture contract check failed")
            return run_result.returncode

    print("[external-scheme] PASS: public-only fixture compiled and executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

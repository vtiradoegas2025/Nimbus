#!/usr/bin/env python3
"""Compile each public factory header in isolation."""

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
    include_dir = root / "include"
    headers = sorted(include_dir.rglob("factory.hpp"))
    compiler = shlex.split(args.compiler)

    with tempfile.TemporaryDirectory(prefix="nimbus-factory-headers-") as temp_dir:
        temp_path = Path(temp_dir)
        for index, header in enumerate(headers):
            relative = header.relative_to(include_dir)
            source = temp_path / f"header_{index}.cpp"
            source.write_text(f'#include "{relative.as_posix()}"\n', encoding="utf-8")
            result = subprocess.run(
                compiler
                + [
                    "-std=c++17",
                    "-I",
                    str(include_dir),
                    "-fsyntax-only",
                    str(source),
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"[public-factory-headers] FAIL: {relative}")
                print(result.stdout, end="")
                print(result.stderr, end="")
                return result.returncode

    print(f"[public-factory-headers] PASS: {len(headers)} standalone headers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

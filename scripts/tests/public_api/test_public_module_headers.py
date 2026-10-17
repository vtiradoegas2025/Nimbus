#!/usr/bin/env python3
"""Compile conventional public module contracts in isolation."""

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
    contract_names = (
        "config.hpp",
        "factory.hpp",
        "runtime.hpp",
        "scheme.hpp",
        "state.hpp",
    )
    headers = sorted(
        path for name in contract_names for path in include_dir.rglob(name)
    )
    extra_headers = (
        include_dir / "core" / "io" / "npy_reader.hpp",
        include_dir / "core" / "runtime" / "process_state.hpp",
        include_dir / "core" / "runtime" / "simulation.hpp",
    )
    headers.extend(path for path in extra_headers if path not in headers)
    compiler = shlex.split(args.compiler)

    with tempfile.TemporaryDirectory(prefix="nimbus-public-headers-") as temp_dir:
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
                print(f"[public-module-headers] FAIL: {relative}")
                print(result.stdout, end="")
                print(result.stderr, end="")
                return result.returncode

    print(f"[public-module-headers] PASS: {len(headers)} standalone headers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

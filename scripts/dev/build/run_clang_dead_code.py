#!/usr/bin/env python3
"""Run Nimbus's strict Clang dead-code pass over backend translation units."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


CHECKS = ",".join(
    (
        "-*",
        "clang-analyzer-core.*",
        "clang-analyzer-deadcode.DeadStores",
        "bugprone-unused-return-value",
        "misc-unused-alias-decls",
        "misc-unused-parameters",
        "misc-unused-using-decls",
        "readability-redundant-control-flow",
        "readability-redundant-declaration",
        "readability-redundant-function-ptr-dereference",
        "readability-redundant-inline-specifier",
        "readability-redundant-member-init",
        "readability-redundant-preprocessor",
        "readability-redundant-smartptr-get",
        "readability-redundant-string-cstr",
        "readability-redundant-string-init",
        "readability-redundant-typename",
    )
)


def executable(name: str, llvm_bin: Path | None) -> str:
    override = os.environ.get(name.upper().replace("-", "_"))
    candidates = (override, shutil.which(name), llvm_bin / name if llvm_bin else None)
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(
        f"{name} not found; install LLVM or set "
        f"{name.upper().replace('-', '_')}"
    )


def homebrew_llvm_bin() -> Path | None:
    brew = shutil.which("brew")
    if not brew:
        return None
    result = subprocess.run(
        (brew, "--prefix", "llvm"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()) / "bin"


def xcode_arguments() -> list[str]:
    if sys.platform != "darwin":
        return []
    xcrun = shutil.which("xcrun")
    if not xcrun:
        return []
    result = subprocess.run(
        (xcrun, "--show-sdk-path"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    sdk = Path(result.stdout.strip())
    libcxx = sdk / "usr/include/c++/v1"
    arguments = ["-extra-arg=-isysroot", f"-extra-arg={sdk}"]
    if libcxx.is_dir():
        arguments.extend(("-extra-arg=-isystem", f"-extra-arg={libcxx}"))
    return arguments


def validate_database(root: Path) -> None:
    database_path = root / "compile_commands.json"
    with database_path.open(encoding="utf-8") as stream:
        database = json.load(stream)
    missing = [entry["file"] for entry in database if not Path(entry["file"]).is_file()]
    if missing:
        raise RuntimeError(
            f"compile_commands.json contains {len(missing)} missing source paths"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    root = args.root.resolve()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    validate_database(root)

    llvm_bin = homebrew_llvm_bin()
    clang_tidy = executable("clang-tidy", llvm_bin)
    run_clang_tidy = executable("run-clang-tidy", llvm_bin)
    source_filter = (
        rf"^(?:{re.escape(str(root / 'src'))}/.*|"
        rf"{re.escape(str(root / 'vulkan/src/compute'))}/.*)\.(?:cpp|cc|cxx)$"
    )
    command = [
        run_clang_tidy,
        "-p",
        str(root),
        "-j",
        str(args.jobs),
        "-clang-tidy-binary",
        clang_tidy,
        f"-checks={CHECKS}",
        "-warnings-as-errors=*",
        "-quiet",
        "-hide-progress",
        *xcode_arguments(),
        source_filter,
    ]
    print(f"clang-tidy: {clang_tidy}", flush=True)
    print(
        "scope: src/ and vulkan/src/compute/ (Metal and rendering excluded)",
        flush=True,
    )
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        sys.stderr.write(f"clang dead-code setup failed: {error}\n")
        raise SystemExit(2)

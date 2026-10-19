#!/usr/bin/env python3
"""Discover and run Nimbus repository scripts from one stable entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
GROUPS = {
    "benchmark": SCRIPT_ROOT / "benchmarks",
    "dev": SCRIPT_ROOT / "dev",
    "qualify": SCRIPT_ROOT / "qualification",
    "test": SCRIPT_ROOT / "tests",
}
STATIC_TESTS = (
    "test_public_factory_headers",
    "test_public_module_headers",
    "test_scheme_descriptors",
    "test_external_scheme_fixture",
    "test_module_ownership",
    "test_build_manifest",
    "test_domain_motion_inventory",
    "test_qualification_fixture_paths",
    "test_studio_distribution",
    "test_checkout_endings",
    "test_windows_release_safeguards",
    "test_shader_assets",
)
COMPILER_TESTS = frozenset(STATIC_TESTS[:4])


def scripts_in(group: str) -> dict[str, Path]:
    directory = GROUPS[group]
    return {
        path.relative_to(directory).with_suffix("").as_posix(): path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh"}
    }


def resolve(group: str, name: str) -> Path:
    candidates = scripts_in(group)
    normalized_path = Path(name).with_suffix("")
    normalized = normalized_path.as_posix().removeprefix("./")
    if normalized in candidates:
        return candidates[normalized]

    prefix = {
        "benchmark": "benchmark_",
        "dev": "",
        "qualify": "qualify_",
        "test": "test_",
    }[group]
    prefixed_path = normalized_path.with_name(prefix + normalized_path.name)
    prefixed = prefixed_path.as_posix().removeprefix("./")
    if prefixed in candidates:
        return candidates[prefixed]

    requested_stems = {normalized_path.name, prefix + normalized_path.name}
    matches = [
        path for path in candidates.values() if path.stem in requested_stems
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        qualified = ", ".join(
            path.relative_to(GROUPS[group]).with_suffix("").as_posix()
            for path in matches
        )
        raise SystemExit(f"ambiguous {group} script {name!r}; use: {qualified}")

    choices = ", ".join(candidates)
    raise SystemExit(f"unknown {group} script {name!r}; available: {choices}")


def command_for(path: Path, arguments: list[str]) -> list[str]:
    launcher = [sys.executable] if path.suffix == ".py" else ["bash"]
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    return launcher + [str(path)] + arguments


def run(path: Path, arguments: list[str]) -> int:
    display = path.relative_to(ROOT)
    print(f"[scripts] running {display}", flush=True)
    return subprocess.run(command_for(path, arguments), cwd=ROOT).returncode


def run_static_tests(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="scripts/run.py test static")
    parser.add_argument("--compiler", default="c++")
    options = parser.parse_args(arguments)

    for name in STATIC_TESTS:
        test_arguments = (
            ["--compiler", options.compiler] if name in COMPILER_TESTS else []
        )
        result = run(resolve("test", name), test_arguments)
        if result:
            return result
    return 0


def list_scripts(group: str | None) -> int:
    selected = [group] if group else list(GROUPS)
    for group_name in selected:
        print(f"{group_name}:")
        for relative_name in scripts_in(group_name):
            print(f"  {relative_name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=["list", *GROUPS])
    parser.add_argument("name", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()

    if options.group == "list":
        if options.name and options.name not in GROUPS:
            parser.error(f"unknown group: {options.name}")
        return list_scripts(options.name)
    if options.name is None:
        parser.error(f"{options.group} requires a script name")
    if options.group == "test" and options.name == "static":
        return run_static_tests(options.arguments)
    return run(resolve(options.group, options.name), options.arguments)


if __name__ == "__main__":
    raise SystemExit(main())

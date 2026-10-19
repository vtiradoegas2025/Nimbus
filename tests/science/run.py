#!/usr/bin/env python3
"""Discover and run Nimbus science and smoke-test tools."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = Path(__file__).resolve().parent
GROUPS = {
    "agrid": TOOL_ROOT / "agrid",
    "analysis": TOOL_ROOT / "analysis",
    "cgrid": TOOL_ROOT / "cgrid",
    "common": TOOL_ROOT / "common",
    "generate": TOOL_ROOT / "generators",
    "hydrostatic": TOOL_ROOT / "hydrostatic",
    "smoke": TOOL_ROOT / "smoke",
}
AGRID_SUITE = (
    ("agrid", "qualify_agrid_pair"),
    ("agrid", "qualify_agrid_acoustic_sweep"),
    ("agrid", "qualify_agrid_hydrometeor_transport"),
    ("agrid", "qualify_agrid_production_acoustic_wave"),
    ("agrid", "qualify_agrid_production_acoustic_spatial"),
    ("agrid", "qualify_agrid_acoustic_energy_exchange"),
    ("agrid", "qualify_agrid_production_acoustic_cfl"),
    ("agrid", "qualify_agrid_warm_bubble"),
    ("agrid", "qualify_agrid_boundary_matrix"),
    ("agrid", "qualify_agrid_boundary_pulse"),
    ("agrid", "qualify_agrid_execution_consistency"),
    ("agrid", "qualify_agrid_radiation_coupling"),
    ("agrid", "qualify_agrid_kessler_coupling"),
    ("cgrid", "qualify_cgrid_coupled_physics"),
)


def tools_in(group: str) -> dict[str, Path]:
    directory = GROUPS[group]
    return {
        path.relative_to(directory).with_suffix("").as_posix(): path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh"}
    }


def resolve(group: str, name: str) -> Path:
    candidates = tools_in(group)
    normalized_path = Path(name).with_suffix("")
    normalized = normalized_path.as_posix().removeprefix("./")
    if normalized in candidates:
        return candidates[normalized]

    matches = [
        path for path in candidates.values() if path.stem == normalized_path.name
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        qualified = ", ".join(
            path.relative_to(GROUPS[group]).with_suffix("").as_posix()
            for path in matches
        )
        raise SystemExit(f"ambiguous {group} tool {name!r}; use: {qualified}")
    choices = ", ".join(candidates)
    raise SystemExit(f"unknown {group} tool {name!r}; available: {choices}")


def run(path: Path, arguments: list[str] | None = None) -> int:
    arguments = list(arguments or ())
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    launcher = [sys.executable] if path.suffix == ".py" else ["bash"]
    print(f"[tools] running {path.relative_to(ROOT)}", flush=True)
    return subprocess.run(launcher + [str(path), *arguments], cwd=ROOT).returncode


def list_tools(group: str | None) -> int:
    selected = [group] if group else list(GROUPS)
    for group_name in selected:
        print(f"{group_name}:")
        for relative_name in tools_in(group_name):
            print(f"  {relative_name}")
    print("suites:\n  agrid")
    return 0


def run_suite(name: str) -> int:
    if name != "agrid":
        raise SystemExit("unknown suite; available: agrid")
    for group, tool_name in AGRID_SUITE:
        result = run(resolve(group, tool_name))
        if result:
            return result
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=["list", "suite", *GROUPS])
    parser.add_argument("name", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()

    if options.group == "list":
        if options.name and options.name not in GROUPS:
            parser.error(f"unknown group: {options.name}")
        return list_tools(options.name)
    if options.name is None:
        parser.error(f"{options.group} requires a name")
    if options.group == "suite":
        if options.arguments:
            parser.error("suites do not accept additional arguments")
        return run_suite(options.name)
    return run(resolve(options.group, options.name), options.arguments)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the accepted A-grid boundary-pair matrix on the production integrator."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping, Sequence


COMMON_TOOLS = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

from check_hydrostatic_hold import QualificationError, qualify_run, require


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SIMULATOR = ROOT / "bin" / "nimbus_sim"
QUALIFICATION_FIXTURES = ROOT / "tests" / "fixtures" / "configs" / "qualification"
BASE_CONFIG = QUALIFICATION_FIXTURES / "agrid_production_integrator_calm.yaml"
SOUNDING = QUALIFICATION_FIXTURES / "soundings" / "moist_isothermal_profile.txt"
NONPERIODIC = ("zero_gradient", "rigid", "rigid_slip")


def render_config(
    template: str,
    lateral: Mapping[str, str],
    bottom: str,
    top: str,
) -> str:
    rendered = template
    for face in ("west", "east", "south", "north"):
        source = f"    {face}: rigid\n"
        require(rendered.count(source) == 1, f"template does not uniquely define {face}")
        rendered = rendered.replace(source, f"    {face}: {lateral[face]}\n")
    require(rendered.count("  bottom: rigid_slip\n") == 1, "template bottom is ambiguous")
    require(rendered.count("  top: rigid\n") == 1, "template top is ambiguous")
    rendered = rendered.replace("  bottom: rigid_slip\n", f"  bottom: {bottom}\n")
    rendered = rendered.replace("  top: rigid\n", f"  top: {top}\n")
    return rendered


def accepted_cases() -> dict[str, tuple[dict[str, str], str, str]]:
    rigid = {face: "rigid" for face in ("west", "east", "south", "north")}
    cases: dict[str, tuple[dict[str, str], str, str]] = {}
    for west, east in itertools.product(NONPERIODIC, repeat=2):
        lateral = dict(rigid, west=west, east=east)
        cases[f"x_{west}_{east}"] = (lateral, "rigid_slip", "rigid")
    cases["x_periodic"] = (
        dict(rigid, west="periodic", east="periodic"),
        "rigid_slip",
        "rigid",
    )
    for south, north in itertools.product(NONPERIODIC, repeat=2):
        lateral = dict(rigid, south=south, north=north)
        key = f"y_{south}_{north}"
        if lateral != rigid:
            cases[key] = (lateral, "rigid_slip", "rigid")
    cases["y_periodic"] = (
        dict(rigid, south="periodic", north="periodic"),
        "rigid_slip",
        "rigid",
    )
    for bottom, top in itertools.product(
        ("rigid_noslip", "rigid_slip"),
        ("rigid", "rigid_with_sponge", "open"),
    ):
        key = f"vertical_{bottom}_{top}"
        if (bottom, top) != ("rigid_slip", "rigid"):
            cases[key] = (dict(rigid), bottom, top)

    cases.update({
        "all_zero_gradient": (
            {face: "zero_gradient" for face in rigid}, "rigid_slip", "rigid"
        ),
        "all_periodic": (
            {face: "periodic" for face in rigid}, "rigid_slip", "rigid"
        ),
        "periodic_x_zero_gradient_y": (
            {
                "west": "periodic", "east": "periodic",
                "south": "zero_gradient", "north": "zero_gradient",
            },
            "rigid_slip",
            "rigid",
        ),
        "rigid_slip_x_periodic_y_open_top": (
            {
                "west": "rigid_slip", "east": "rigid_slip",
                "south": "periodic", "north": "periodic",
            },
            "rigid_noslip",
            "open",
        ),
        "legacy_open_alias": (
            {face: "open" for face in rigid}, "rigid_slip", "rigid"
        ),
    })
    return cases


def constant_velocity_template(template: str) -> str:
    require("domain_motion:" not in template, "boundary template already owns motion")
    template = template.replace(
        "dynamics:\n",
        "domain_motion:\n"
        "  mode: constant_velocity\n"
        "  frame_velocity:\n"
        "    u_ms: 7\n"
        "    v_ms: -3\n\n"
        "dynamics:\n",
        1,
    )
    for old, new in (
        ("    u_sfc_ms: 0\n", "    u_sfc_ms: 7\n"),
        ("    v_sfc_ms: 0\n", "    v_sfc_ms: -3\n"),
        ("    u_1km_ms: 0\n", "    u_1km_ms: 7\n"),
        ("    v_1km_ms: 0\n", "    v_1km_ms: -3\n"),
        ("    u_6km_ms: 0\n", "    u_6km_ms: 7\n"),
        ("    v_6km_ms: 0\n", "    v_6km_ms: -3\n"),
    ):
        require(template.count(old) == 1, f"boundary template drifted at {old!r}")
        template = template.replace(old, new, 1)
    return template


def run_case(
    binary: Path, name: str, config_path: Path, output_path: Path
) -> dict[str, object]:
    command = [
        str(binary),
        "--headless",
        f"--config={config_path}",
        "--duration=2",
        "--write-every=2",
        f"--outdir={output_path}",
        "--compute-backend=cpu",
        "--guard-mode=strict",
        "--log-profile=quiet",
    ]
    if "  mode: constant_velocity\n" not in config_path.read_text(
        encoding="utf-8"
    ):
        command.append(f"--sounding={SOUNDING}")
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    require(
        completed.returncode == 0,
        f"{name}: simulator failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
    )
    return qualify_run(output_path, minimum_duration_s=2.0)


def require_rejection(
    binary: Path,
    name: str,
    config_path: Path,
    expected_message: str,
) -> None:
    completed = subprocess.run(
        [str(binary), "--validate-only", f"--config={config_path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    combined = completed.stdout + completed.stderr
    require(completed.returncode != 0, f"{name}: invalid boundary config was accepted")
    require(
        expected_message in combined,
        f"{name}: rejection did not contain {expected_message!r}\n{combined}",
    )


def qualify_matrix(binary: Path) -> dict[str, object]:
    template = BASE_CONFIG.read_text(encoding="utf-8")
    cases = accepted_cases()
    case_metrics: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="nimbus-agrid-boundaries-") as temporary:
        root = Path(temporary)
        for motion, motion_template in (
            ("stationary", template),
            ("constant_velocity", constant_velocity_template(template)),
        ):
            for name, (lateral, bottom, top) in cases.items():
                case_name = f"{motion}_{name}"
                config_path = root / f"{case_name}.yaml"
                config_path.write_text(
                    render_config(motion_template, lateral, bottom, top),
                    encoding="utf-8",
                )
                result = run_case(
                    binary, case_name, config_path,
                    root / f"{case_name}-output",
                )
                case_metrics[case_name] = result["metrics"]

        radiative = root / "reject_radiative.yaml"
        radiative.write_text(
            render_config(
                template,
                {"west": "radiative", "east": "zero_gradient",
                 "south": "rigid", "north": "rigid"},
                "rigid_slip",
                "rigid",
            ),
            encoding="utf-8",
        )
        require_rejection(
            binary,
            "radiative",
            radiative,
            "radiative lateral boundaries require grid.staggering=c_grid",
        )

        half_periodic = root / "reject_half_periodic.yaml"
        half_periodic.write_text(
            render_config(
                template,
                {"west": "periodic", "east": "rigid",
                 "south": "rigid", "north": "rigid"},
                "rigid_slip",
                "rigid",
            ),
            encoding="utf-8",
        )
        require_rejection(
            binary,
            "half_periodic",
            half_periodic,
            "periodic lateral boundaries must be configured as opposing pairs",
        )

    max_wind = max(float(metrics["max_wind_ms"]) for metrics in case_metrics.values())
    max_mass_drift = max(
        float(metrics["mass_relative_drift"])
        for metrics in case_metrics.values()
    )
    max_energy_drift = max(
        float(metrics["total_energy_relative_drift"])
        for metrics in case_metrics.values()
    )
    return {
        "status": "PASS",
        "scope": "production_agrid_stationary_and_constant_velocity_supported_boundary_pairs",
        "accepted_case_count": len(case_metrics),
        "negative_case_count": 2,
        "aggregate_metrics": {
            "maximum_peak_wind_ms": max_wind,
            "maximum_mass_relative_drift": max_mass_drift,
            "maximum_total_energy_relative_drift": max_energy_drift,
        },
        "accepted_cases": sorted(case_metrics),
        "negative_cases": ["radiative", "half_periodic"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binary", type=Path, default=DEFAULT_SIMULATOR)
    args = parser.parse_args(argv)
    try:
        binary = args.binary.resolve()
        require(binary.is_file(), f"missing simulator: {binary}")
        result = qualify_matrix(binary)
    except (OSError, QualificationError) as error:
        parser.exit(1, f"[agrid-boundary-matrix] FAIL: {error}\n")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

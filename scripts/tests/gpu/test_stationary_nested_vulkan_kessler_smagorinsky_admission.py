#!/usr/bin/env python3
"""Exercise the exact Kessler plus active-Smagorinsky nesting boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification" /
    "stationary_nested_vulkan_kessler_smagorinsky.yaml"
)
PROFILE_ERROR = "domain_motion.nesting is limited to"
SCIENCE_ERROR = "Science profile rejected before initialization"
COMMON_ERROR = (
    "enabled domain_motion.nesting requires stationary flat Cartesian C-grid"
)
TRACKING_ERROR = (
    "domain_motion.tracking diagnostic evidence is not reference-qualified"
)
KESSLER_ERROR = (
    "microphysics.scheme=kessler is qualified only in an exact public reference profile"
)


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"fixture mutation expected one occurrence of {old!r}, "
            f"found {source.count(old)}"
        )
    return source.replace(old, new, 1)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--require-vulkan-positive",
        action="store_true",
        help=(
            "require the exact active fallback-off profile to reach live "
            "Vulkan startup"
        ),
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    for required in (binary, CONFIG):
        if not required.is_file():
            raise RuntimeError(f"required admission input is missing: {required}")

    base_text = CONFIG.read_text(encoding="utf-8")
    profile_only = (PROFILE_ERROR,)
    profile_or_science = (
        PROFILE_ERROR, SCIENCE_ERROR, COMMON_ERROR, KESSLER_ERROR
    )
    mutations: dict[str, tuple[str, tuple[str, ...]]] = {
        "noncanonical cloud-water threshold": (
            replace_once(
                base_text,
                "    cloud_water_threshold_kgkg: 0.001\n",
                "    cloud_water_threshold_kgkg: 0.0009\n",
            ),
            profile_only,
        ),
        "noncanonical autoconversion rate": (
            replace_once(
                base_text,
                "    autoconversion_rate_s_inv: 0.001\n",
                "    autoconversion_rate_s_inv: 0.0009\n",
            ),
            profile_only,
        ),
        "noncanonical accretion coefficient": (
            replace_once(
                base_text,
                "    accretion_coefficient: 2.2\n",
                "    accretion_coefficient: 2.1\n",
            ),
            profile_only,
        ),
        "nonreference evaporation efficiency": (
            replace_once(
                base_text,
                "    evaporation_efficiency: 1.0\n",
                "    evaporation_efficiency: 0.9\n",
            ),
            profile_only,
        ),
        "host microphysics commit": (
            replace_once(
                base_text,
                "    gpu_apply_microphysics: true\n",
                "    gpu_apply_microphysics: false\n",
            ),
            profile_only,
        ),
        "alternate microphysics": (
            replace_once(
                base_text,
                "microphysics:\n"
                "  scheme: kessler\n"
                "  shared:\n"
                "    cloud_water_threshold_kgkg: 0.001\n"
                "    autoconversion_rate_s_inv: 0.001\n"
                "  kessler:\n"
                "    accretion_coefficient: 2.2\n"
                "    evaporation_efficiency: 1.0\n",
                "microphysics:\n"
                "  scheme: lin\n"
                "  shared:\n"
                "    cloud_water_threshold_kgkg: 0.001\n"
                "    autoconversion_rate_s_inv: 0.001\n"
                "  lin:\n"
                "    accretion_coefficient: 2.2\n"
                "    evaporation_rate_s_inv: 0.001\n",
            ),
            profile_or_science,
        ),
        "horizontal SGS": (
            replace_once(base_text, "  mode: 3d\n", "  mode: 2d\n"),
            profile_only,
        ),
        "active radiation": (
            replace_once(
                replace_once(
                    base_text,
                    "surface:\n  scheme: none\n  apply_fluxes: false\n",
                    "surface:\n  scheme: none\n  apply_fluxes: false\n"
                    "  tsk_k: 290\n",
                ),
                "radiation:\n  scheme: none\n",
                "radiation:\n"
                "  scheme: simple_grey\n"
                "  solar_forcing: idealized_diurnal\n"
                "  latitude_deg: 35\n"
                "  declination_deg: 15\n"
                "  local_solar_time_at_start_h: 0\n"
                "  do_lw: true\n"
                "  do_sw: false\n"
                "  cadence_s: 10\n"
                "  clear_sky_only: true\n"
                "  tau_lw_ref: 2.5\n"
                "  tau_sw_ref: 0.22\n"
                "  n_lw: 1\n"
                "  n_sw: 2\n",
            ),
            profile_or_science,
        ),
        "active boundary layer": (
            replace_once(
                base_text,
                "boundary_layer:\n  scheme: none\n",
                "boundary_layer:\n  scheme: slab\n",
            ),
            profile_or_science,
        ),
        "active diffusion": (
            replace_once(base_text, "  K_h: 0\n", "  K_h: 1\n"),
            profile_only,
        ),
        "active terrain": (
            replace_once(
                base_text,
                "terrain:\n  scheme: none\n",
                "terrain:\n  scheme: bell\n",
            ),
            profile_or_science,
        ),
        "active stochastic physics": (
            replace_once(
                base_text,
                "  chaos:\n    scheme: none\n",
                "  chaos:\n    scheme: initial_conditions\n",
            ),
            profile_or_science,
        ),
        "active tracking": (
            replace_once(
                base_text,
                "  tracking:\n    enabled: false\n    recycle_domain: false\n",
                "  tracking:\n    enabled: true\n    recycle_domain: false\n"
                "    inner_margin_m: 1000\n",
            ),
            (PROFILE_ERROR, SCIENCE_ERROR, COMMON_ERROR, TRACKING_ERROR),
        ),
        "nonqualified execution": (
            replace_once(
                base_text,
                "    mode: qualified\n",
                "    mode: exploratory\n",
            ),
            profile_only,
        ),
    }
    with tempfile.TemporaryDirectory(
        prefix="nimbus_stationary_nested_kessler_smag_admission_"
    ) as temporary:
        temporary_root = Path(temporary)
        for index, (label, (contents, expected_errors)) in enumerate(
            mutations.items()
        ):
            candidate = temporary_root / f"negative_{index:02d}.yaml"
            candidate.write_text(contents, encoding="utf-8")
            result = run(
                [
                    str(binary), "--config", str(candidate), "--headless",
                    "--threads", "1", "--duration", "0",
                    "--write-every=0",
                    f"--outdir={temporary_root / ('negative_' + str(index))}",
                    "--log-profile=quiet",
                ]
            )
            generic_microphysics_scope = (
                "microphysics.scheme=" in result.stdout
                and ("qualified only" in result.stdout
                     or "requires a maintained" in result.stdout)
            )
            if result.returncode == 0 or (not generic_microphysics_scope and not any(
                marker in result.stdout
                for marker in (*expected_errors, KESSLER_ERROR)
            )):
                raise RuntimeError(
                    f"{label} did not fail at the Kessler Vulkan nesting "
                    f"profile boundary:\n{result.stdout}"
                )

        if args.require_vulkan_positive:
            positive = run(
                [
                    str(binary), "--config", str(CONFIG), "--headless",
                    "--threads", "1", "--duration", "0",
                    "--write-every=0",
                    f"--outdir={temporary_root / 'vulkan'}",
                    "--log-profile=normal",
                ]
            )
            if positive.returncode != 0:
                raise RuntimeError(
                    "exact Kessler Vulkan stationary-nesting profile failed "
                    f"({positive.returncode}):\n{positive.stdout}"
                )
            required_log = (
                "[COMPUTE] backend=vulkan, requested=vulkan, "
                "device_index=-1, fallback=off"
            )
            if required_log not in positive.stdout:
                raise RuntimeError(
                    "positive Kessler admission did not prove an active "
                    f"fallback-off device:\n{positive.stdout}"
                )

    print(
        "[stationary-nested-kessler-smagorinsky-admission] PASS: "
        f"{len(mutations)} active-profile neighbors rejected"
        + (
            "; exact active fallback-off profile reached live startup"
            if args.require_vulkan_positive else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

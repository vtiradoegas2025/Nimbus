#!/usr/bin/env python3
"""Exercise the exact active-Smagorinsky stationary-nesting boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification" /
    "stationary_nested_vulkan_smagorinsky.yaml"
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
LIN_ERROR = (
    "microphysics.scheme=lin is qualified only for an exact flat periodic C-grid "
    "Lin/YSU/RRTMG reference profile with declared backend ownership"
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
        help="also require the exact active profile to reach live startup",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    for required in (binary, CONFIG):
        if not required.is_file():
            raise RuntimeError(f"required admission input is missing: {required}")

    base_text = CONFIG.read_text(encoding="utf-8")
    mutations = {
        "horizontal SGS": replace_once(
            base_text, "  mode: 3d\n", "  mode: 2d\n"
        ),
        "vertical-only SGS": replace_once(
            base_text, "  mode: 3d\n", "  mode: vertical_only\n"
        ),
        "noncanonical Smagorinsky coefficient": replace_once(
            base_text, "  Cs: 0.18\n", "  Cs: 0.17\n"
        ),
        "noncanonical turbulent Prandtl number": replace_once(
            base_text,
            "  Pr_t: 0.3333333333333333\n",
            "  Pr_t: 0.4\n",
        ),
        "noncanonical turbulent Schmidt number": replace_once(
            base_text,
            "  Sc_t: 0.3333333333333333\n",
            "  Sc_t: 0.4\n",
        ),
        "host SGS commit": replace_once(
            base_text, "    gpu_apply_sgs: true\n",
            "    gpu_apply_sgs: false\n",
        ),
        "unsupported microphysics": replace_once(
            base_text, "microphysics:\n  scheme: none\n",
            "microphysics:\n  scheme: lin\n",
        ),
        "active radiation": replace_once(
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
        "active boundary layer": replace_once(
            base_text, "boundary_layer:\n  scheme: none\n",
            "boundary_layer:\n  scheme: slab\n",
        ),
        "active diffusion": replace_once(
            base_text, "  K_h: 0\n", "  K_h: 1\n"
        ),
        "active terrain": replace_once(
            base_text, "terrain:\n  scheme: none\n",
            "terrain:\n  scheme: bell\n",
        ),
        "active stochastic physics": replace_once(
            base_text, "  chaos:\n    scheme: none\n",
            "  chaos:\n    scheme: initial_conditions\n",
        ),
        "active tracking": replace_once(
            base_text,
            "  tracking:\n    enabled: false\n    recycle_domain: false\n",
            "  tracking:\n    enabled: true\n    recycle_domain: false\n"
            "    inner_margin_m: 1000\n",
        ),
        "nonqualified execution": replace_once(
            base_text, "    mode: qualified\n", "    mode: exploratory\n"
        ),
    }

    with tempfile.TemporaryDirectory(
        prefix="nimbus_stationary_nested_smag_admission_"
    ) as temporary:
        temporary_root = Path(temporary)
        for index, (label, contents) in enumerate(mutations.items()):
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
            if (
                result.returncode == 0 or
                (PROFILE_ERROR not in result.stdout and
                 SCIENCE_ERROR not in result.stdout and
                 COMMON_ERROR not in result.stdout and
                 TRACKING_ERROR not in result.stdout and
                 KESSLER_ERROR not in result.stdout and
                 LIN_ERROR not in result.stdout)
            ):
                raise RuntimeError(
                    f"{label} did not fail at the active Vulkan nesting "
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
                    "exact active Vulkan stationary-nesting profile failed "
                    f"({positive.returncode}):\n{positive.stdout}"
                )
            required_log = (
                "[COMPUTE] backend=vulkan, requested=vulkan, "
                "device_index=-1, fallback=off"
            )
            if required_log not in positive.stdout:
                raise RuntimeError(
                    "positive active admission did not prove an active "
                    f"fallback-off device:\n{positive.stdout}"
                )

    print(
        "[stationary-nested-smagorinsky-admission] PASS: "
        f"{len(mutations)} active-profile neighbors rejected"
        + ("; exact active profile reached live startup"
           if args.require_vulkan_positive else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

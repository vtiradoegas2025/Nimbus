#!/usr/bin/env python3
"""Exercise the exact stationary-nesting runtime admission boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
VULKAN_CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification" /
    "stationary_nested_vulkan_dry.yaml"
)
CPU_CONFIG = ROOT / "tests" / "fixtures" / "configs" / "test" / "cart_cgrid_stationary_nest.yaml"
PROFILE_ERROR = "domain_motion.nesting is limited to"
COMMON_ERROR = (
    "enabled domain_motion.nesting requires stationary flat Cartesian C-grid"
)
TRACKING_ERROR = (
    "domain_motion.tracking diagnostic evidence is not reference-qualified"
)
SURFACE_ERROR = (
    "cannot be none when surface.scheme=slab"
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


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed ({result.returncode}):\n{result.stdout}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--require-vulkan-positive",
        action="store_true",
        help="also require the exact Vulkan fixture to reach live startup",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    for required in (binary, VULKAN_CONFIG, CPU_CONFIG):
        if not required.is_file():
            raise RuntimeError(f"required admission input is missing: {required}")

    base_text = VULKAN_CONFIG.read_text(encoding="utf-8")
    mutations = {
        "spatial ratio": replace_once(
            replace_once(
                replace_once(
                    replace_once(
                        replace_once(
                            base_text,
                            "    parent_ratio_x: 2\n",
                            "    parent_ratio_x: 3\n",
                        ),
                        "    parent_ratio_y: 2\n",
                        "    parent_ratio_y: 3\n",
                    ),
                    "    parent_time_step_ratio: 2\n",
                    "    parent_time_step_ratio: 3\n",
                ),
                "    child_cells_x: 8\n",
                "    child_cells_x: 9\n",
            ),
            "    child_cells_y: 8\n",
            "    child_cells_y: 9\n",
        ),
        "time ratio": replace_once(
            base_text,
            "    parent_time_step_ratio: 2\n",
            "    parent_time_step_ratio: 3\n",
        ),
        "lateral boundary": replace_once(
            base_text, "    west: zero_gradient\n", "    west: radiative\n"
        ),
        "active damping": replace_once(
            base_text, "  enabled: false\n  z_d: 12000\n",
            "  enabled: true\n  z_d: 12000\n",
        ),
        "active lateral sponge": replace_once(
            base_text,
            "  lateral_sponge_enabled: false\n",
            "  lateral_sponge_enabled: true\n",
        ),
        "active w-only sponge": replace_once(
            base_text,
            "  w_only_top_sponge: false\n",
            "  w_only_top_sponge: true\n",
        ),
        "active Coriolis": replace_once(
            base_text,
            "coriolis:\n  enabled: false\n",
            "coriolis:\n  enabled: true\n",
        ),
        "active microphysics": replace_once(
            base_text, "microphysics:\n  scheme: none\n",
            "microphysics:\n  scheme: kessler\n",
        ),
        "unsupported turbulence mode": replace_once(
            replace_once(
                base_text, "turbulence:\n  scheme: none\n",
                "turbulence:\n  scheme: smagorinsky\n",
            ),
            "  mode: 3d\n", "  mode: 2d\n",
        ),
        "active boundary layer": replace_once(
            replace_once(
                base_text,
                "surface:\n  scheme: none\n",
                "surface:\n  scheme: slab\n",
            ),
            "boundary_layer:\n  scheme: none\n",
            "boundary_layer:\n  scheme: slab\n",
        ),
        "active surface selection": replace_once(
            base_text,
            "surface:\n  scheme: none\n",
            "surface:\n  scheme: slab\n",
        ),
        "active radiation": replace_once(
            replace_once(
                base_text,
                "surface:\n  scheme: none\n  apply_fluxes: false\n",
                "surface:\n  scheme: none\n  apply_fluxes: false\n  tsk_k: 290\n",
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
        "active diffusion": replace_once(
            base_text, "  K_h: 0\n", "  K_h: 1\n"
        ),
        "active terrain": replace_once(
            base_text, "terrain:\n  scheme: none\n",
            "terrain:\n  scheme: bell\n",
        ),
        "active stochastic physics": replace_once(
            base_text,
            "  chaos:\n    scheme: none\n",
            "  chaos:\n    scheme: initial_conditions\n",
        ),
        "alternate top boundary": replace_once(
            base_text, "  top: rigid\n", "  top: rigid_with_sponge\n"
        ),
        "active tracking": replace_once(
            base_text,
            "  tracking:\n    enabled: false\n    recycle_domain: false\n",
            "  tracking:\n    enabled: true\n    recycle_domain: false\n"
            "    inner_margin_m: 1000\n",
        ),
        "active recycling": replace_once(
            base_text,
            "  tracking:\n    enabled: false\n    recycle_domain: false\n",
            "  tracking:\n    enabled: true\n    recycle_domain: true\n"
            "    inner_margin_m: 1000\n"
            "    maximum_shift_cells: 1\n",
        ),
        "nonqualified execution": replace_once(
            base_text, "    mode: qualified\n", "    mode: exploratory\n"
        ),
    }

    with tempfile.TemporaryDirectory(
        prefix="nimbus_stationary_nested_admission_"
    ) as temporary:
        temporary_root = Path(temporary)
        cpu_result = run(
            [
                str(binary), "--config", str(CPU_CONFIG), "--headless",
                "--compute-backend", "cpu", "--threads", "1",
                "--duration", "0", "--write-every=0",
                f"--outdir={temporary_root / 'cpu'}", "--log-profile=quiet",
            ]
        )
        require_success(cpu_result, "existing CPU stationary-nesting profile")

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
            admission_error = (
                PROFILE_ERROR in result.stdout or
                COMMON_ERROR in result.stdout or
                TRACKING_ERROR in result.stdout or
                SURFACE_ERROR in result.stdout or
                KESSLER_ERROR in result.stdout
            )
            if result.returncode == 0 or not admission_error:
                raise RuntimeError(
                    f"{label} did not fail at the exact Vulkan nesting "
                    f"profile boundary:\n{result.stdout}"
                )

        if args.require_vulkan_positive:
            positive = run(
                [
                    str(binary), "--config", str(VULKAN_CONFIG), "--headless",
                    "--threads", "1", "--duration", "0",
                    "--write-every=0",
                    f"--outdir={temporary_root / 'vulkan'}",
                    "--log-profile=normal",
                ]
            )
            require_success(positive, "exact Vulkan stationary-nesting profile")
            required_log = (
                "[COMPUTE] backend=vulkan, requested=vulkan, "
                "device_index=-1, fallback=off"
            )
            if required_log not in positive.stdout:
                raise RuntimeError(
                    "positive Vulkan admission did not prove an active "
                    f"fallback-off device:\n{positive.stdout}"
                )

    print(
        "[stationary-nested-admission] PASS: existing CPU selection retained; "
        f"{len(mutations)} Vulkan neighbor profiles rejected"
        + ("; exact Vulkan profile reached live startup" if args.require_vulkan_positive else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

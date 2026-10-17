#!/usr/bin/env python3
"""Exercise the exact longwave stationary-nesting admission boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification" /
    "stationary_nested_vulkan_radiation.yaml"
)
PROFILE_ERROR = "domain_motion.nesting is limited to"
COMMON_ERROR = (
    "enabled domain_motion.nesting requires stationary flat Cartesian C-grid"
)
SCIENCE_ERROR = "Science profile rejected before initialization"
CONFIG_ERROR = "v1 config load failed"


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
    parser.add_argument("--require-vulkan-positive", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()
    for required in (binary, CONFIG):
        if not required.is_file():
            raise RuntimeError(f"required admission input is missing: {required}")

    base = CONFIG.read_text(encoding="utf-8")
    mutations = {
        "shortwave enabled": replace_once(
            base, "  do_sw: false\n", "  do_sw: true\n"
        ),
        "longwave disabled": replace_once(
            base, "  do_lw: true\n", "  do_lw: false\n"
        ),
        "non-clear-sky selection": replace_once(
            base, "  clear_sky_only: true\n", "  clear_sky_only: false\n"
        ),
        "alternate cadence": replace_once(
            base, "  cadence_s: 2\n", "  cadence_s: 3\n"
        ),
        "alternate longwave optical depth": replace_once(
            base, "  tau_lw_ref: 2.5\n", "  tau_lw_ref: 2.4\n"
        ),
        "alternate longwave exponent": replace_once(
            base, "  n_lw: 1\n", "  n_lw: 1.1\n"
        ),
        "alternate surface emissivity": replace_once(
            base, "  emissivity: 0.96\n", "  emissivity: 0.95\n"
        ),
        "alternate surface temperature": replace_once(
            base, "  tsk_k: 290\n", "  tsk_k: 291\n"
        ),
        "active microphysics": replace_once(
            base,
            "microphysics:\n  scheme: none\n",
            "microphysics:\n  scheme: kessler\n",
        ),
        "active turbulence": replace_once(
            base,
            "turbulence:\n  scheme: none\n",
            "turbulence:\n  scheme: smagorinsky\n",
        ),
        "active surface and boundary layer": replace_once(
            replace_once(
                base,
                "surface:\n  scheme: none\n  apply_fluxes: false\n",
                "surface:\n  scheme: slab\n  apply_fluxes: true\n",
            ),
            "boundary_layer:\n  scheme: none\n",
            "boundary_layer:\n  scheme: slab\n",
        ),
        "active terrain": replace_once(
            base, "terrain:\n  scheme: none\n", "terrain:\n  scheme: bell\n"
        ),
        "active stochastic physics": replace_once(
            base,
            "  chaos:\n    scheme: none\n",
            "  chaos:\n    scheme: initial_conditions\n",
        ),
        "active diffusion": replace_once(base, "  K_h: 0\n", "  K_h: 1\n"),
        "active Coriolis": replace_once(
            base,
            "coriolis:\n  enabled: false\n",
            "coriolis:\n  enabled: true\n",
        ),
        "nonreference time ratio": replace_once(
            base,
            "    parent_time_step_ratio: 2\n",
            "    parent_time_step_ratio: 3\n",
        ),
        "nonreference boundary": replace_once(
            base, "    west: zero_gradient\n", "    west: radiative\n"
        ),
        "qualified CPU Milbrandt is not implicitly admitted": replace_once(
            replace_once(
                base,
                "microphysics:\n  scheme: none\n",
                "microphysics:\n  scheme: milbrandt\n",
            ),
            "radiation:\n  scheme: simple_grey\n",
            "radiation:\n  scheme: none\n",
        ),
    }

    with tempfile.TemporaryDirectory(
        prefix="nimbus_stationary_nested_radiation_admission_"
    ) as temporary:
        root = Path(temporary)
        for index, (label, contents) in enumerate(mutations.items()):
            candidate = root / f"negative_{index:02d}.yaml"
            candidate.write_text(contents, encoding="utf-8")
            result = run(
                [
                    str(binary), "--config", str(candidate), "--headless",
                    "--compute-backend", "cpu", "--threads", "1",
                    "--duration", "0", "--write-every=0",
                    f"--outdir={root / ('negative_' + str(index))}",
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
                for marker in (
                    PROFILE_ERROR, COMMON_ERROR, SCIENCE_ERROR, CONFIG_ERROR
                )
            )):
                raise RuntimeError(
                    f"{label} did not fail at the qualified stationary-"
                    f"nesting boundary:\n{result.stdout}"
                )

        cpu = run(
            [
                str(binary), "--config", str(CONFIG), "--headless",
                "--compute-backend", "cpu", "--threads", "1",
                "--duration", "0", "--write-every=0",
                f"--outdir={root / 'cpu'}", "--log-profile=normal",
            ]
        )
        if cpu.returncode != 0 or "[COMPUTE] backend=cpu" not in cpu.stdout:
            raise RuntimeError(
                "exact CPU longwave nesting profile failed:\n" + cpu.stdout
            )

        if args.require_vulkan_positive:
            vulkan = run(
                [
                    str(binary), "--config", str(CONFIG), "--headless",
                    "--threads", "1", "--duration", "0", "--write-every=0",
                    f"--outdir={root / 'vulkan'}", "--log-profile=normal",
                ]
            )
            device = (
                "[COMPUTE] backend=vulkan, requested=vulkan, "
                "device_index=-1, fallback=off"
            )
            if vulkan.returncode != 0 or device not in vulkan.stdout:
                raise RuntimeError(
                    "exact Vulkan longwave nesting profile failed:\n" +
                    vulkan.stdout
                )

    print(
        "[stationary-nested-radiation-admission] PASS: "
        f"{len(mutations)} negative neighbors rejected; exact CPU admitted"
        + ("; exact fallback-off Vulkan admitted"
           if args.require_vulkan_positive else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

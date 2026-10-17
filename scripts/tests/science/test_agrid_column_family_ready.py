#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
FIELDS = (
    "u", "v", "w", "rho", "p", "theta", "qv", "tracer",
    "qc", "qr", "qi", "qs", "qg", "qh", "tke",
)
LEDGERS = (
    "budget.csv", "momentum_budget.csv", "energy_budget.csv",
    "radiation_budget.csv", "moisture_budget.csv",
    "thermodynamic_source_budget.csv",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    if "[SCIENTIFIC INVALIDITY]" in result.stdout:
        raise RuntimeError(f"{label} reported scientific invalidity")


def field_payload(output: Path, second: int, field: str) -> bytes:
    path = output / f"step_{second:06d}" / f"{field}.npy"
    shape, raw, values = read_float32_npy(path)
    if shape != (18, 18, 18) or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"invalid A-grid column output {path}")
    if field in ("rho", "p", "theta") and any(value <= 0.0 for value in values):
        raise RuntimeError(f"A-grid column {field} is not positive")
    if field in ("qv", "tracer", "qc", "qr", "qi", "qs", "qg", "qh", "tke") and any(
        value < 0.0 for value in values
    ):
        raise RuntimeError(f"A-grid column {field} is negative")
    return raw


def render(source: str, owner: str) -> str:
    rendered = source.replace(
        "output:\n  preset: minimal\n",
        "output:\n  preset: dynamics_full\n",
        1,
    )
    if owner == "none":
        return rendered
    surface = (
        "surface:\n"
        "  scheme: slab\n"
        "  apply_fluxes: true\n"
        "  z0m_m: 0.1\n"
        "  z0h_m: 0.01\n"
        "  albedo: 0.2\n"
        "  emissivity: 0.96\n"
        "  tsk_k: 305\n"
    )
    rendered = rendered.replace(
        "surface:\n  scheme: none\n  apply_fluxes: false\n",
        surface,
        1,
    )
    top = "tke" if owner == "mynn" else "ri"
    rendered = rendered.replace(
        "boundary_layer:\n  scheme: none\n",
        "boundary_layer:\n"
        f"  scheme: {owner}\n"
        f"  pbl_top_method: {top}\n",
        1,
    )
    if owner == "slab":
        radiation = (
            "radiation:\n"
            "  scheme: simple_grey\n"
            "  solar_forcing: idealized_diurnal\n"
            "  latitude_deg: 35\n"
            "  declination_deg: 15\n"
            "  local_solar_time_at_start_h: 12\n"
            "  do_lw: true\n"
            "  do_sw: false\n"
            "  cadence_s: 1\n"
            "  clear_sky_only: true\n"
            "  tau_lw_ref: 2.5\n"
            "  n_lw: 1\n"
        )
    else:
        radiation = (
            "radiation:\n"
            "  scheme: rrtmg\n"
            "  solar_forcing: idealized_diurnal\n"
            "  latitude_deg: 35\n"
            "  declination_deg: 15\n"
            "  local_solar_time_at_start_h: 12\n"
            "  do_lw: true\n"
            "  do_sw: true\n"
            "  cadence_s: 1\n"
            "  clear_sky_only: true\n"
            "  rrtmg_ozone_vmr: 0.000004\n"
        )
    return rendered.replace("radiation:\n  scheme: none\n", radiation, 1)


def execute(binary: Path, config: Path, output: Path, threads: int) -> str:
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"A-grid column run ({threads} workers)")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    source = CONFIG.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nimbus-agrid-column-") as temporary:
        root = Path(temporary)
        finals: dict[str, dict[str, bytes]] = {}
        for owner in ("none", "slab", "ysu", "mynn"):
            config = root / f"agrid-{owner}.yaml"
            config.write_text(render(source, owner), encoding="utf-8")
            preflight = run([str(binary), "--validate-only", "--config", str(config)])
            require_success(preflight, f"A-grid {owner} column preflight")

            serial = root / f"{owner}-serial"
            parallel = root / f"{owner}-parallel"
            resumed = root / f"{owner}-resumed"
            serial_log = execute(binary, config, serial, 1)
            parallel_log = execute(binary, config, parallel, 4)
            if owner != "none":
                for marker in (
                    f"Initialized boundary layer scheme: {owner}",
                    "Initialized radiation scheme: " +
                        ("simple_grey" if owner == "slab" else "rrtmg"),
                ):
                    if marker not in serial_log or marker not in parallel_log:
                        raise RuntimeError(f"A-grid {owner} omitted owner {marker}")

            finals[owner] = {}
            for field in FIELDS:
                serial_final = field_payload(serial, 2, field)
                parallel_final = field_payload(parallel, 2, field)
                if serial_final != parallel_final:
                    raise RuntimeError(f"A-grid {owner} {field} differs across workers")
                finals[owner][field] = parallel_final
            for ledger in LEDGERS:
                if (serial / ledger).read_bytes() != (parallel / ledger).read_bytes():
                    raise RuntimeError(f"A-grid {owner} {ledger} differs across workers")

            serial_checkpoint = serial / "checkpoints/checkpoint_step_000000010.nimbus"
            checkpoint = parallel / "checkpoints/checkpoint_step_000000010.nimbus"
            if serial_checkpoint.read_bytes() != checkpoint.read_bytes():
                raise RuntimeError(f"A-grid {owner} checkpoint differs across workers")
            continuation = run([
                str(binary), "--headless", "--config", str(config),
                f"--resume={checkpoint}", "--duration=2", "--write-every=1",
                "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
                "--log-profile=normal",
            ])
            require_success(continuation, f"A-grid {owner} continuation")
            for field in FIELDS:
                if field_payload(resumed, 1, field) != finals[owner][field]:
                    raise RuntimeError(f"A-grid {owner} {field} restart mismatch")

        for owner in ("slab", "ysu", "mynn"):
            if all(
                finals[owner][field] == finals["none"][field]
                for field in ("u", "v", "theta", "qv", "tke")
            ):
                raise RuntimeError(f"A-grid {owner} column owner was inert")

        invalid = root / "invalid-surface.yaml"
        invalid.write_text(
            source.replace(
                "surface:\n  scheme: none\n  apply_fluxes: false\n",
                "surface:\n  scheme: none\n  apply_fluxes: true\n",
                1,
            ),
            encoding="utf-8",
        )
        rejected = run([str(binary), "--validate-only", "--config", str(invalid)])
        if rejected.returncode == 0:
            raise RuntimeError("A-grid invalid surface-ownership neighbor was admitted")

    print(
        "[agrid-column-family-ready] PASS: none/slab/YSU/MYNN column owners "
        "evolved distinctly, matched 1/4-worker fields, ledgers, checkpoints, "
        "resumed bitwise, and rejected invalid surface ownership"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

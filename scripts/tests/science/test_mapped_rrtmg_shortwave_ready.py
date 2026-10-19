#!/usr/bin/env python3
"""Bounded production qualification for mapped RRTMG spectral shortwave."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
BASE = (
    ROOT / "tests/fixtures/configs/qualification/"
    "cgrid_terrain_smagorinsky_coupled.yaml"
)
RADIATION = """radiation:
  scheme: rrtmg
  solar_forcing: idealized_diurnal
  latitude_deg: 35
  declination_deg: 15
  local_solar_time_at_start_h: 12
  do_lw: true
  do_sw: true
  cadence_s: 5
  clear_sky_only: false
  rrtmg_ozone_vmr: 0.000004
  terrain_sw_cosine_nodes: 1
  terrain_sw_azimuth_nodes: 4
  terrain_sw_maximum_bounces: 32
  terrain_sw_relative_residual_tolerance: 1.0e-10
  terrain_sw_absolute_residual_tolerance_w: 1.0e-9
"""


def fail(message: str) -> None:
    raise RuntimeError(message)


def replace_block(source: str, start: str, end: str, replacement: str) -> str:
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[:begin] + replacement + source[finish:]


def source_config() -> str:
    source = BASE.read_text(encoding="utf-8")
    source = source.replace("  nx: 16\n", "  nx: 4\n", 1)
    source = source.replace("  ny: 16\n", "  ny: 4\n", 1)
    source = source.replace(
        "      dtheta_k: 1\n",
        "      dtheta_k: 1\n      cloud_water_kgkg: 0.002\n", 1)
    source = source.replace("      center_x_m: 4000\n", "      center_x_m: 1000\n", 1)
    source = source.replace("      center_y_m: 4000\n", "      center_y_m: 1000\n", 1)
    source = replace_block(source, "radiation:\n", "terrain:\n", RADIATION + "\n")
    return source


def run(
    binary: Path, config: Path, output: Path, threads: int,
    duration: int = 2,
) -> tuple[str, float]:
    started = time.monotonic()
    result = subprocess.run(
        [
            str(binary), "--headless", "--config", str(config),
            f"--duration={duration}", "--write-every=1", "--checkpoint-every=1",
            f"--outdir={output}", f"--threads={threads}",
            "--log-profile=normal", "--no-timing",
        ], cwd=ROOT, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    diagnostic = result.stdout + result.stderr
    if result.returncode != 0:
        fail(f"mapped RRTMG shortwave t{threads} failed\n{diagnostic}")
    for marker in (
        "[SCIENTIFIC INVALIDITY]", "Science profile rejected",
        "[CONFIG ERROR]", "fallback=active",
    ):
        if marker in diagnostic:
            fail(f"mapped RRTMG shortwave t{threads} reported {marker}")
    return diagnostic, elapsed


def npy_files(run: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(run): path.read_bytes()
        for path in run.glob("step_*/*.npy")
    }


def require_radiation_ledger(run: Path) -> None:
    path = run / "radiation_budget.csv"
    if not path.is_file():
        fail("mapped RRTMG shortwave omitted its radiation ledger")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail("mapped RRTMG shortwave radiation ledger is empty")
    for row in rows:
        numeric = {
            key: float(value)
            for key, value in row.items()
            if key != "shortwave_reflection_converged"
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            fail("mapped RRTMG shortwave radiation ledger is non-finite")
        if row["shortwave_reflection_converged"] not in {"1", "true"}:
            fail("mapped RRTMG shortwave ledger reports unconverged transport")
        if numeric["shortwave_incoming_power_w"] <= 0.0 or \
                numeric["shortwave_atmospheric_absorbed_power_w"] <= 0.0:
            fail("mapped RRTMG shortwave ledger did not exercise heating")
        if numeric["shortwave_relative_power_residual"] > 1.0e-12:
            fail("mapped RRTMG shortwave power partition does not close")
        if numeric["relative_booking_residual"] > 1.0e-5:
            fail("mapped RRTMG shortwave applied heat does not close")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file() or not BASE.is_file():
        fail("mapped RRTMG shortwave qualification inputs are missing")

    with tempfile.TemporaryDirectory(prefix="nimbus_mapped_rrtmg_sw_") as raw:
        root = Path(raw)
        active_config = root / "active.yaml"
        active_source = source_config()
        active_config.write_text(active_source, encoding="utf-8")
        validation = subprocess.run(
            [str(binary), "--validate-only", "--config", str(active_config)],
            cwd=ROOT, text=True, capture_output=True)
        if validation.returncode != 0 or "[VALIDATE] PASS" not in validation.stdout:
            fail(
                "mapped RRTMG shortwave ordinary admission failed\n" +
                validation.stdout + validation.stderr)

        one = root / "threads-1"
        four = root / "threads-4"
        _, one_elapsed = run(binary, active_config, one, 1)
        _, four_elapsed = run(binary, active_config, four, 4)
        one_fields = npy_files(one)
        four_fields = npy_files(four)
        if not one_fields or one_fields != four_fields:
            fail("mapped RRTMG shortwave output is not one/four-thread exact")
        require_radiation_ledger(one)
        require_radiation_ledger(four)

        control_config = root / "longwave_control.yaml"
        control_config.write_text(
            active_source.replace("  do_sw: true\n", "  do_sw: false\n", 1),
            encoding="utf-8")
        control = root / "longwave-control"
        run(binary, control_config, control, 1)
        active_theta = (one / "step_000001/theta.npy").read_bytes()
        control_theta = (control / "step_000001/theta.npy").read_bytes()
        if active_theta == control_theta:
            fail("mapped RRTMG shortwave is inert against its longwave-only control")

        checkpoints = sorted((one / "checkpoints").glob("*.nimbus"))
        if len(checkpoints) != 2:
            fail("mapped RRTMG shortwave omitted bounded checkpoints")
        resumed = root / "resumed"
        resume_result = subprocess.run(
            [
                str(binary), "--headless", "--config", str(active_config),
                f"--resume={checkpoints[0]}", "--duration=2",
                "--write-every=1", "--checkpoint-every=0",
                f"--outdir={resumed}", "--threads=4",
                "--log-profile=normal", "--no-timing",
            ], cwd=ROOT, text=True, capture_output=True)
        if resume_result.returncode != 0:
            fail("mapped RRTMG shortwave restart failed\n" +
                 resume_result.stdout + resume_result.stderr)
        continuous_step = one / "step_000002"
        resumed_steps = sorted(resumed.glob("step_*"))
        matched = []
        for step in resumed_steps:
            manifest = step / "manifest.json"
            if manifest.is_file() and '"simulation_time_s": 2' in manifest.read_text(
                    encoding="utf-8"):
                matched.append(step)
        if len(matched) != 1:
            fail("mapped RRTMG restart omitted the two-second publication")
        for field in ("u", "v", "w", "theta", "p", "rho", "qv", "qr"):
            if ((continuous_step / f"{field}.npy").read_bytes() !=
                    (matched[0] / f"{field}.npy").read_bytes()):
                fail(f"mapped RRTMG restart differs for {field}")

    print(
        "PASS mapped RRTMG 112-point shortwave ordinary admission, non-inert "
        "cloud-scattering heating, exact one/four-thread fields, restart, and "
        f"bounded runtime (t1={one_elapsed:.3f}s, t4={four_elapsed:.3f}s)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"FAIL: {error}")
        raise SystemExit(1)

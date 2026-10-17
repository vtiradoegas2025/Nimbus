#!/usr/bin/env python3
"""Mapped scientific output identity across workers and continuation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/agrid_mapped_advection_none_chaos_none_bell_none_ready.yaml"


def run(binary: Path, output: Path, threads: int, checkpoint: Path | None = None,
        config: Path = CONFIG, duration: int = 2) -> None:
    command = [str(binary), "--headless", f"--config={config}", f"--duration={duration}",
               "--write-every=1", "--checkpoint-every=1", f"--threads={threads}",
               f"--outdir={output}", "--no-timing"]
    if checkpoint:
        command.append(f"--resume={checkpoint}")
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stdout)


def frame(output: Path, time: float) -> Path:
    for manifest in output.glob("step_*/manifest.json"):
        if json.loads(manifest.read_text())["simulation_time_s"] == time:
            return manifest.parent
    raise RuntimeError(f"{output}: missing frame at {time}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nimbus_diagnostic_science_") as temporary:
        root = Path(temporary)
        mapped, mapped_parallel = root / "mapped", root / "mapped_parallel"
        run(args.binary.resolve(), mapped, 1, duration=0)
        run(args.binary.resolve(), mapped_parallel, 4, duration=0)
        initial = frame(mapped, 0.0)
        for field in ("p_prime", "theta_prime", "buoyancy"):
            shape, _, values = read_float32_npy(initial / f"{field}.npy")
            index = (9 * shape[1] + 9) * shape[2] + 15
            if abs(values[index]) > 4.0e-5:
                raise RuntimeError(f"unperturbed terrain background leaks into {field}: {values[index]}")
        _, _, skin = read_float32_npy(initial / "skin_temperature.npy")
        if any(abs(value - 305.0) > 1.0e-5 for value in skin):
            raise RuntimeError("skin_temperature does not publish prescribed 305 K forcing")
        mapped_manifest = json.loads((initial / "manifest.json").read_text())
        for field in ("lifted_index", "showalter_index", "k_index", "total_totals"):
            shape, _, values = read_float32_npy(initial / f"{field}.npy")
            if not math.isnan(values[9 * shape[1] + 9]):
                raise RuntimeError(f"{field} fabricates a 500 hPa level above the modeled column")
            metadata = next(value for value in mapped_manifest["fields"] if value["field_id"] == field)
            if metadata.get("missing_value") != "NaN":
                raise RuntimeError(f"{field} does not declare its missing-value representation")
        mapped_fields = mapped_manifest["published_fields"]
        for field in mapped_fields:
            if (initial / f"{field}.npy").read_bytes() != (frame(mapped_parallel, 0.0) / f"{field}.npy").read_bytes():
                raise RuntimeError(f"mapped {field} changes with thread count")
        flat_source = (ROOT / "tests/fixtures/configs/qualification/cgrid_mynn_rrtmg_ready.yaml").read_text()
        flat_config = root / "flat_full_output.yaml"
        flat_config.write_text(flat_source.replace("  preset: minimal", "  preset: dynamics_full"))
        serial, parallel, resumed = root / "serial", root / "parallel", root / "resumed"
        run(args.binary.resolve(), serial, 1, config=flat_config)
        run(args.binary.resolve(), parallel, 4, config=flat_config)
        checkpoint = next(
            path.with_suffix("") for path in (parallel / "checkpoints").glob("*.nimbus.json")
            if abs(json.loads(path.read_text())["simulation_time_s"] - 1.0) < 1.0e-9
        )
        async_config = root / "flat_async_output.yaml"
        async_config.write_text(flat_config.read_text().replace("  async_io: false", "  async_io: true"))
        run(args.binary.resolve(), resumed, 4, checkpoint, config=async_config)
        left, right, continuation = frame(serial, 2.0), frame(parallel, 2.0), frame(resumed, 2.0)
        manifest = json.loads((left / "manifest.json").read_text())
        fields = manifest["published_fields"]
        for required in ("buoyancy", "p_prime", "theta_prime", "pressure_gradient_force_x",
                         "divergence", "srh_0_1km", "srh_0_3km", "updraft_helicity", "cape", "cin"):
            if required not in fields:
                raise RuntimeError(f"scientific output coverage missing {required}")
        for field in fields:
            expected = (left / f"{field}.npy").read_bytes()
            if expected != (right / f"{field}.npy").read_bytes():
                raise RuntimeError(f"{field} changes with thread count")
            if expected != (continuation / f"{field}.npy").read_bytes():
                raise RuntimeError(f"{field} changes across continuation")
        print(f"[diagnostic-science] PASS: {len(mapped_fields)} mapped outputs and physical references; {len(fields)} coupled outputs, 1/4 workers and async restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

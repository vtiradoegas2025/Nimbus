#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_simple_grey_absorbing_water_cloud_cpu_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
SCOPE_ERROR = (
    "absorbing water/cloud-path simple_grey is qualified only for the exact "
    "flat periodic C-grid CPU profile"
)


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
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected"):
        if marker in result.stdout:
            raise RuntimeError(f"{label} reported {marker}:\n{result.stdout}")


def require_scope_rejection(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode == 0 or SCOPE_ERROR not in result.stdout:
        raise RuntimeError(f"{label} did not fail with the exact scope guard:\n{result.stdout}")


def field_payload(output: Path, step: int, field: str):
    path = output / f"step_{step:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing {field} output: {path}")
    return read_float32_npy(path)


def coupled_run(binary: Path, config: Path, output: Path, threads: int, label: str) -> str:
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, label)
    for marker in (
        "Execution mode: qualified",
        "Initialized radiation scheme: simple_grey",
        "Initialized microphysics scheme: kessler",
        "simple_grey_absorbing_water_cloud_path_v1",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"{label} omitted provenance marker {marker!r}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (binary, CONFIG):
        if not path.is_file():
            raise RuntimeError(f"required ready-state input is missing: {path}")

    preflight = run([str(binary), "--validate-only", "--config", str(CONFIG)])
    require_success(preflight, "simple-grey qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("simple-grey qualified preflight did not publish PASS")

    with tempfile.TemporaryDirectory(prefix="nimbus-simple-grey-water-cloud-ready-") as temporary:
        root = Path(temporary)
        source = CONFIG.read_text()
        mutations = (
            ("coefficient", "  simple_grey_lw_liquid_absorption_m2kg: 85\n",
             "  simple_grey_lw_liquid_absorption_m2kg: 84\n"),
            ("diagnostics", "  enable_diagnostics: true\n",
             "  enable_diagnostics: false\n"),
            ("subcycles", "  subcycle_count: 2\n", "  subcycle_count: 3\n"),
            ("microphysics", "    accretion_coefficient: 2.2\n",
             "    accretion_coefficient: 2.1\n"),
            ("sounding",
             "    file_path: tests/fixtures/configs/qualification/soundings/simple_grey_condensing_profile.txt\n",
             "    file_path: tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt\n"),
            ("backend", "    backend: cpu\n", "    backend: metal\n"),
        )
        for name, old, new in mutations:
            mutated = source.replace(old, new, 1)
            if mutated == source:
                raise RuntimeError(f"{name} scope mutation did not alter the fixture")
            path = root / f"invalid-{name}.yaml"
            path.write_text(mutated)
            require_scope_rejection(
                run([str(binary), "--validate-only", "--config", str(path)]),
                f"{name} validate-only mutation",
            )
            require_scope_rejection(
                run([
                    str(binary), "--headless", "--config", str(path),
                    "--duration=0", "--write-every=0",
                    f"--outdir={root / f'invalid-{name}-runtime'}",
                ]),
                f"{name} runtime mutation",
            )

        serial = root / "serial"
        parallel = root / "parallel"
        resumed = root / "resumed"
        coupled_run(binary, CONFIG, serial, 1, "serial qualified run")
        coupled_run(binary, CONFIG, parallel, 4, "parallel qualified run")
        initial_shape, initial_raw, _ = field_payload(parallel, 0, "theta")
        final_shape, parallel_raw, final_values = field_payload(parallel, 2, "theta")
        serial_shape, serial_raw, _ = field_payload(serial, 2, "theta")
        if initial_shape != (12, 12, 16) or final_shape != initial_shape:
            raise RuntimeError("simple-grey theta output shape disagrees with the grid")
        if serial_shape != final_shape or serial_raw != parallel_raw:
            raise RuntimeError("simple-grey theta differs between one and four workers")
        if initial_raw == parallel_raw:
            raise RuntimeError("simple-grey radiation did not evolve theta")
        if any(not math.isfinite(value) for value in final_values):
            raise RuntimeError("simple-grey theta contains non-finite values")

        qc_shape, parallel_qc_raw, parallel_qc = field_payload(parallel, 2, "qc")
        serial_qc_shape, serial_qc_raw, _ = field_payload(serial, 2, "qc")
        if qc_shape != final_shape or serial_qc_shape != qc_shape:
            raise RuntimeError("simple-grey cloud-water output shape disagrees with the grid")
        if serial_qc_raw != parallel_qc_raw:
            raise RuntimeError("cloud-water evolution differs between one and four workers")
        if max(parallel_qc) <= 1.0e-5 or sum(value > 0.0 for value in parallel_qc) < 100:
            raise RuntimeError("ordinary Kessler coupling did not produce a resolved cloud-water path")
        compared_fields = (
            "u", "v", "w", "theta", "p", "rho", "qv", "qc", "qr",
            "cold_pool_boundary", "precip_rate", "accumulated_rainfall",
        )
        for field in compared_fields:
            serial_field_shape, serial_field_raw, _ = field_payload(serial, 2, field)
            parallel_field_shape, parallel_field_raw, _ = field_payload(parallel, 2, field)
            if serial_field_shape != parallel_field_shape or serial_field_raw != parallel_field_raw:
                raise RuntimeError(f"{field} differs between one and four workers")

        exploratory_source = source.replace("    mode: qualified\n", "    mode: exploratory\n", 1)
        exploratory_source = exploratory_source.replace(
            "    mode: qualified\n",
            "    mode: qualified\n    acknowledge_maturity_limits: false\n", 1,
        )
        low_water_source = exploratory_source.replace(
            "  simple_grey_lw_vapor_absorption_m2kg: 0.02\n",
            "  simple_grey_lw_vapor_absorption_m2kg: 0\n", 1,
        ).replace(
            "  simple_grey_sw_vapor_absorption_m2kg: 0.001\n",
            "  simple_grey_sw_vapor_absorption_m2kg: 0\n", 1,
        )
        low_water_config = root / "low-water-exploratory.yaml"
        low_water_config.write_text(low_water_source)
        low_water_output = root / "low-water"
        low_water_result = run([
            str(binary), "--headless", "--config", str(low_water_config),
            f"--outdir={low_water_output}", "--write-every=1", "--threads=1",
            "--log-profile=normal",
        ])
        require_success(low_water_result, "low-water sensitivity run")
        _, low_water_raw, _ = field_payload(low_water_output, 2, "theta")
        if low_water_raw == parallel_raw:
            raise RuntimeError("nonzero water vapor did not alter coupled simple-grey theta")

        no_cloud_source = exploratory_source.replace(
            "  simple_grey_lw_liquid_absorption_m2kg: 85\n",
            "  simple_grey_lw_liquid_absorption_m2kg: 0\n", 1,
        ).replace(
            "  simple_grey_sw_liquid_absorption_m2kg: 20\n",
            "  simple_grey_sw_liquid_absorption_m2kg: 0\n", 1,
        )
        no_cloud_config = root / "no-cloud-absorption-exploratory.yaml"
        no_cloud_config.write_text(no_cloud_source)
        no_cloud_output = root / "no-cloud-absorption"
        no_cloud_result = run([
            str(binary), "--headless", "--config", str(no_cloud_config),
            f"--outdir={no_cloud_output}", "--write-every=1", "--threads=1",
            "--log-profile=normal",
        ])
        require_success(no_cloud_result, "no-cloud-absorption sensitivity run")
        _, no_cloud_raw, _ = field_payload(no_cloud_output, 2, "theta")
        if no_cloud_raw == parallel_raw:
            raise RuntimeError("nonzero cloud water did not alter coupled simple-grey theta")

        checkpoint = parallel / "checkpoints/checkpoint_step_000000005.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("simple-grey run did not publish its midpoint restart pair")
        continuation = run([
            str(binary), "--headless", "--config", str(CONFIG),
            f"--resume={checkpoint}", "--duration=2", "--write-every=1",
            "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
            "--log-profile=normal",
        ])
        require_success(continuation, "simple-grey checkpoint continuation")
        if "[CHECKPOINT] resumed step=5, time_s=1" not in continuation.stdout:
            raise RuntimeError("simple-grey continuation did not restore its midpoint")
        for field in compared_fields:
            resumed_shape, resumed_raw, _ = field_payload(resumed, 1, field)
            continuous_shape, continuous_raw, _ = field_payload(parallel, 2, field)
            if resumed_shape != continuous_shape or resumed_raw != continuous_raw:
                raise RuntimeError(
                    f"simple-grey checkpoint continuation is not bitwise exact for {field}"
                )

    print(
        "[simple-grey-absorbing-water-cloud-ready] PASS: exact CPU profile "
        "closed admission, retained its equation provenance, formed cloud water, "
        "responded to vapor and cloud absorption, matched 1/4 workers bitwise, "
        "and resumed bitwise"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

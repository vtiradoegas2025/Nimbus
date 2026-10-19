#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_dynamic_smagorinsky_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
FIELDS = ("u", "v", "w", "rho", "p", "theta", "qv")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected"):
        if marker in result.stdout:
            raise RuntimeError(f"{label} reported {marker}:\n{result.stdout}")


def require_scope_rejection(
    result: subprocess.CompletedProcess[str], label: str
) -> None:
    if result.returncode == 0:
        raise RuntimeError(f"{label} unexpectedly succeeded:\n{result.stdout}")
    expected = (
        "dynamic Smagorinsky is qualified only for the exact flat periodic "
        "C-grid CPU reference profile"
    )
    if expected not in result.stdout:
        raise RuntimeError(f"{label} omitted the exact scope error:\n{result.stdout}")


def field_payload(output: Path, second: int, field: str) -> bytes:
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing output field: {path}")
    shape, raw, values = read_float32_npy(path)
    if shape != (12, 12, 16):
        raise RuntimeError(f"{field} has unexpected shape {shape}")
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"{field} contains non-finite values")
    return raw


def coupled_run(
    binary: Path, config: Path, output: Path, threads: int
) -> subprocess.CompletedProcess[str]:
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"dynamic Smagorinsky run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        "Initialized Smagorinsky Turbulence",
        "Dynamic coefficient = Favre Germano-Lilly",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"dynamic run omitted {marker!r}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (
        binary,
        CONFIG,
        ROOT / "data/rrtmg/rrtmg_lw.nc",
        ROOT / "data/rrtmg/rrtmg_sw.nc",
    ):
        if not path.is_file():
            raise RuntimeError(f"required Ready-state input is missing: {path}")

    preflight = run([str(binary), "--validate-only", "--config", str(CONFIG)])
    require_success(preflight, "dynamic Smagorinsky qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("dynamic preflight did not publish PASS")

    with tempfile.TemporaryDirectory(prefix="nimbus-dynamic-smag-ready-") as temporary:
        root = Path(temporary)
        source = CONFIG.read_text()
        for label, old, new in (
            ("backend", "    backend: cpu\n", "    backend: metal\n"),
            ("mode", "  mode: 3d\n", "  mode: 2d\n"),
            ("filter", "  filter_width: cubic_root\n", "  filter_width: dx\n"),
            ("coefficient cap", "  Cs: 0.23\n", "  Cs: 0.18\n"),
            (
                "boundary",
                "    south: periodic\n    north: periodic\n",
                "    south: zero_gradient\n    north: zero_gradient\n",
            ),
        ):
            mutated = source.replace(old, new, 1)
            if mutated == source:
                raise RuntimeError(f"{label} mutation did not alter fixture")
            path = root / f"invalid-{label.replace(' ', '-')}.yaml"
            path.write_text(mutated)
            require_scope_rejection(
                run([str(binary), "--validate-only", "--config", str(path)]),
                f"dynamic {label} scope guard",
            )

        serial = root / "serial"
        parallel = root / "parallel"
        resumed = root / "resumed"
        coupled_run(binary, CONFIG, serial, 1)
        coupled_run(binary, CONFIG, parallel, 4)

        evolved = False
        for field in FIELDS:
            initial = field_payload(parallel, 0, field)
            serial_final = field_payload(serial, 2, field)
            parallel_final = field_payload(parallel, 2, field)
            if serial_final != parallel_final:
                raise RuntimeError(f"{field} differs between one and four workers")
            evolved = evolved or initial != parallel_final
        if not evolved:
            raise RuntimeError("qualified dynamic profile did not evolve")

        checkpoint = parallel / "checkpoints/checkpoint_step_000000005.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("dynamic run did not publish midpoint checkpoint")
        continuation = run([
            str(binary), "--headless", "--config", str(CONFIG),
            f"--resume={checkpoint}", "--duration=2", "--write-every=1",
            "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
            "--log-profile=normal",
        ])
        require_success(continuation, "dynamic checkpoint continuation")
        if "[CHECKPOINT] resumed step=5, time_s=1" not in continuation.stdout:
            raise RuntimeError("dynamic continuation did not restore midpoint")
        for field in FIELDS:
            if field_payload(resumed, 1, field) != field_payload(parallel, 2, field):
                raise RuntimeError(f"{field} restart continuation is not bitwise exact")

        control = root / "fixed-control.yaml"
        control.write_text(
            source.replace("    mode: qualified\n", "    mode: exploratory\n", 1)
                  .replace("    mode: qualified\n",
                           "    mode: qualified\n    acknowledge_maturity_limits: false\n", 1)
                  .replace("  dynamic_coefficient: true\n",
                           "  dynamic_coefficient: false\n", 1)
        )
        control_output = root / "fixed-control"
        control_result = run([
            str(binary), "--headless", "--config", str(control),
            f"--outdir={control_output}", "--write-every=1", "--threads=1",
            "--log-profile=normal",
        ])
        require_success(control_result, "fixed-coefficient exploratory control")
        if all(
            field_payload(control_output, 2, field)
            == field_payload(serial, 2, field)
            for field in ("u", "v", "w")
        ):
            raise RuntimeError("dynamic selection did not change momentum evolution")

    print(
        "[dynamic-smagorinsky-ready] PASS: exact CPU profile selected the "
        "Favre Germano-Lilly procedure, evolved distinctly from fixed Cs, "
        "matched 1/4 workers, resumed bitwise, and rejected near profiles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_user_filter_width_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
FIELDS = ("u", "v", "w", "theta", "rho")


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
        "turbulence.filter_width=user is qualified only for the exact flat "
        "periodic C-grid CPU Smagorinsky profile"
    )
    if expected not in result.stdout:
        raise RuntimeError(f"{label} omitted the exact scope error:\n{result.stdout}")


def field_payload(output: Path, second: int, field: str) -> tuple[tuple[int, ...], bytes]:
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing output field: {path}")
    shape, raw, values = read_float32_npy(path)
    if shape != (12, 12, 16):
        raise RuntimeError(f"{field} has unexpected shape {shape}")
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"{field} contains non-finite values")
    return shape, raw


def coupled_run(
    binary: Path, config: Path, output: Path, threads: int
) -> subprocess.CompletedProcess[str]:
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"user-filter run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        "Initialized Smagorinsky Turbulence",
        "Filter width = user (500 m)",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"user-filter run omitted {marker!r}")
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
    require_success(preflight, "user-filter qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("user-filter preflight did not publish PASS")

    with tempfile.TemporaryDirectory(prefix="nimbus-user-filter-ready-") as temporary:
        root = Path(temporary)
        source = CONFIG.read_text()
        for label, old, new in (
            ("width", "  user_filter_width_m: 500\n", "  user_filter_width_m: 250\n"),
            ("backend", "    backend: cpu\n", "    backend: metal\n"),
            ("scheme", "  scheme: smagorinsky\n", "  scheme: tke\n"),
        ):
            mutated = source.replace(old, new, 1)
            if mutated == source:
                raise RuntimeError(f"{label} mutation did not alter fixture")
            path = root / f"invalid-{label}.yaml"
            path.write_text(mutated)
            require_scope_rejection(
                run([str(binary), "--validate-only", "--config", str(path)]),
                f"user-filter {label} scope guard",
            )

        missing = root / "missing-width.yaml"
        missing.write_text(source.replace("  user_filter_width_m: 500\n", "", 1))
        result = run([str(binary), "--validate-only", "--config", str(missing)])
        if result.returncode == 0 or "must be positive and finite" not in result.stdout:
            raise RuntimeError(f"missing user width did not fail closed:\n{result.stdout}")

        serial = root / "serial"
        parallel = root / "parallel"
        resumed = root / "resumed"
        coupled_run(binary, CONFIG, serial, 1)
        coupled_run(binary, CONFIG, parallel, 4)

        evolved = False
        for field in FIELDS:
            _, initial = field_payload(parallel, 0, field)
            _, serial_final = field_payload(serial, 2, field)
            _, parallel_final = field_payload(parallel, 2, field)
            if serial_final != parallel_final:
                raise RuntimeError(f"{field} differs between one and four workers")
            evolved = evolved or initial != parallel_final
        if not evolved:
            raise RuntimeError("qualified user-filter profile did not evolve")

        checkpoint = parallel / "checkpoints/checkpoint_step_000000005.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("user-filter run did not publish midpoint checkpoint")
        continuation = run([
            str(binary), "--headless", "--config", str(CONFIG),
            f"--resume={checkpoint}", "--duration=2", "--write-every=1",
            "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
            "--log-profile=normal",
        ])
        require_success(continuation, "user-filter checkpoint continuation")
        if "[CHECKPOINT] resumed step=5, time_s=1" not in continuation.stdout:
            raise RuntimeError("user-filter continuation did not restore midpoint")
        for field in FIELDS:
            _, expected = field_payload(parallel, 2, field)
            _, actual = field_payload(resumed, 1, field)
            if actual != expected:
                raise RuntimeError(f"{field} restart continuation is not bitwise exact")

        control = root / "control.yaml"
        control.write_text(
            source.replace("    mode: qualified\n", "    mode: exploratory\n", 1)
                  .replace("    mode: qualified\n",
                           "    mode: qualified\n    acknowledge_maturity_limits: false\n", 1)
                  .replace("  user_filter_width_m: 500\n", "  user_filter_width_m: 250\n", 1)
        )
        control_output = root / "control"
        control_run = run([
            str(binary), "--headless", "--config", str(control),
            f"--outdir={control_output}", "--write-every=1", "--threads=1",
            "--log-profile=normal",
        ])
        require_success(control_run, "half-width exploratory control")
        if all(
            field_payload(control_output, 2, field)[1] ==
            field_payload(serial, 2, field)[1]
            for field in ("u", "v", "w")
        ):
            raise RuntimeError("changing the explicit filter width had no momentum response")

    print(
        "[user-filter-width-ready] PASS: exact CPU Smagorinsky profile used "
        "the explicit physical width, evolved, responded to width scaling, "
        "matched 1/4 workers, resumed bitwise, and rejected near profiles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

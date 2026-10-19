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
SCOPE_ERROR = (
    "qualified Cartesian A-grid execution is limited to the exact "
    "flat/native complete or mapped CPU reference profiles"
)
FIELDS = ("u", "v", "w", "theta", "p", "rho", "qv", "qr")
LEDGERS = (
    "budget.csv",
    "momentum_budget.csv",
    "energy_budget.csv",
    "radiation_budget.csv",
    "moisture_budget.csv",
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


def field_payload(output: Path, second: int, field: str):
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing {field} output: {path}")
    return read_float32_npy(path)


def coupled_run(binary: Path, output: Path, threads: int) -> None:
    result = run([
        str(binary),
        "--headless",
        "--config",
        str(CONFIG),
        f"--outdir={output}",
        f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"A-grid core run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        "Dynamics: klemp_wilhelmson_split_cartesian_agrid",
        "Trigger source: warm_bubble",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"A-grid core run omitted {marker!r}")


def require_scope_rejection(
    result: subprocess.CompletedProcess[str], label: str
) -> None:
    expected = (
        SCOPE_ERROR,
        "does not support staggering=collocated",
        "allow_fallback=false",
        "qualified coupled RK integration requires a ready flat CPU",
    )
    if result.returncode == 0 or not any(
        marker in result.stdout for marker in expected
    ):
        raise RuntimeError(f"{label} did not fail with the scope guard:\n{result.stdout}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (binary, CONFIG):
        if not path.is_file():
            raise RuntimeError(f"required A-grid ready-state input is missing: {path}")

    preflight = run([str(binary), "--validate-only", "--config", str(CONFIG)])
    require_success(preflight, "A-grid qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("A-grid qualified preflight did not publish PASS")

    with tempfile.TemporaryDirectory(prefix="nimbus-agrid-core-ready-") as temporary:
        root = Path(temporary)
        source = CONFIG.read_text()
        admitted_mutations = (
            ("geometry", "  nx: 18\n", "  nx: 19\n"),
            ("initialization", "      center_x_m: 4500\n", "      center_x_m: 4000\n"),
            ("no-advection", "advection:\n  scheme: tvd\n", "advection:\n  scheme: none\n"),
        )
        for name, old, new in admitted_mutations:
            admitted_source = source.replace(old, new, 1)
            if admitted_source == source:
                raise RuntimeError(f"A-grid {name} mutation did not alter the fixture")
            admitted_config = root / f"admitted-{name}.yaml"
            admitted_config.write_text(admitted_source)
            result = run([
                str(binary), "--headless", "--config", str(admitted_config),
                "--duration=1", "--write-every=1", "--checkpoint-every=0",
                f"--outdir={root / ('admitted-' + name)}", "--threads=1",
                "--log-profile=normal",
            ])
            require_success(result, f"A-grid {name} family member")
            if name == "no-advection":
                if "Numerics-Advection: none" not in result.stdout:
                    raise RuntimeError(
                        f"A-grid did not select no-advection:\n{result.stdout}"
                    )
                output = root / "admitted-no-advection"
                _, initial_w, _ = field_payload(output, 0, "w")
                _, final_w, _ = field_payload(output, 1, "w")
                if initial_w == final_w:
                    raise RuntimeError("A-grid no-advection disabled dynamics")

        mutations = (
            ("ownership", "    allow_fallback: false\n", "    allow_fallback: true\n"),
            ("physics", "microphysics:\n  scheme: none\n", "microphysics:\n  scheme: lin\n"),
            ("integrator", "time:\n", "time:\n  integrator: ssprk3\n"),
        )
        for name, old, new in mutations:
            invalid_source = source.replace(old, new, 1)
            if invalid_source == source:
                raise RuntimeError(f"A-grid {name} mutation did not alter the fixture")
            invalid_config = root / f"invalid-{name}.yaml"
            invalid_config.write_text(invalid_source)
            require_scope_rejection(
                run([str(binary), "--validate-only", "--config", str(invalid_config)]),
                f"A-grid {name} validate-only neighbor",
            )
            require_scope_rejection(
                run([
                    str(binary),
                    "--headless",
                    "--config",
                    str(invalid_config),
                    "--duration=0",
                    f"--outdir={root / ('invalid-' + name)}",
                ]),
                f"A-grid {name} runtime neighbor",
            )

        serial = root / "serial"
        parallel = root / "parallel"
        resumed = root / "resumed"
        coupled_run(binary, serial, 1)
        coupled_run(binary, parallel, 4)

        evolved = set()
        for field in FIELDS:
            initial_shape, initial_raw, _ = field_payload(parallel, 0, field)
            final_shape, parallel_raw, values = field_payload(parallel, 2, field)
            serial_shape, serial_raw, _ = field_payload(serial, 2, field)
            if initial_shape != (18, 18, 18) or final_shape != initial_shape:
                raise RuntimeError(f"A-grid {field} output shape disagrees with the grid")
            if serial_shape != final_shape or serial_raw != parallel_raw:
                raise RuntimeError(
                    f"A-grid {field} differs between one and four workers"
                )
            if initial_raw != parallel_raw:
                evolved.add(field)
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError(f"A-grid {field} output is non-finite")
            if field in ("rho", "p") and any(value <= 0.0 for value in values):
                raise RuntimeError(f"A-grid {field} output is not strictly positive")
            if field in ("qv", "qr") and any(value < 0.0 for value in values):
                raise RuntimeError(f"A-grid {field} output is negative")
        for field in ("w", "theta", "p"):
            if field not in evolved:
                raise RuntimeError(f"A-grid {field} did not evolve")

        for ledger in LEDGERS:
            serial_path = serial / ledger
            parallel_path = parallel / ledger
            if not serial_path.is_file() or not parallel_path.is_file():
                raise RuntimeError(f"A-grid run omitted {ledger}")
            if serial_path.read_bytes() != parallel_path.read_bytes():
                raise RuntimeError(
                    f"A-grid {ledger} differs between one and four workers"
                )

        checkpoint = parallel / "checkpoints/checkpoint_step_000000010.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("A-grid run did not publish its midpoint restart pair")
        continuation = run([
            str(binary),
            "--headless",
            "--config",
            str(CONFIG),
            f"--resume={checkpoint}",
            f"--outdir={resumed}",
            "--threads=4",
            "--log-profile=normal",
        ])
        require_success(continuation, "A-grid checkpoint continuation")
        if "[CHECKPOINT] resumed step=10, time_s=1" not in continuation.stdout:
            raise RuntimeError("A-grid continuation did not restore the midpoint")
        for field in FIELDS:
            final_shape, final_raw, _ = field_payload(parallel, 2, field)
            resumed_shape, resumed_raw, _ = field_payload(resumed, 1, field)
            if resumed_shape != final_shape or resumed_raw != final_raw:
                raise RuntimeError(
                    f"A-grid {field} continuation is not bitwise exact"
                )

    print(
        "[agrid-core-ready] PASS: qualified flat CPU family admitted flexible "
        "geometry/initialization, evolved the "
        "retained core, matched 1/4-worker fields and ledgers bitwise, and "
        "resumed bitwise"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
CPU_CONFIG = ROOT / "tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml"
CONFIGS = {
    "vulkan": ROOT / "tests/fixtures/configs/qualification/agrid_flat_vulkan_complete_ready.yaml",
    "metal": ROOT / "tests/fixtures/configs/qualification/agrid_flat_metal_complete_ready.yaml",
}
FIELDS = ("u", "v", "w", "theta", "p", "rho", "qv", "qr")
LEDGERS = (
    "budget.csv",
    "momentum_budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "thermodynamic_source_budget.csv",
)
SCOPE_ERROR = (
    "qualified Cartesian A-grid execution is limited to the exact "
    "flat/native complete or mapped CPU reference profiles"
)
KERNELS = (
    "dynamics/cartesian_tendencies.comp",
    "dynamics/slow_apply.comp",
    "acoustic/acoustic_continuity_massflux_cartesian_agrid.comp",
    "acoustic/acoustic_pressure_eos_cartesian_agrid.comp",
    "acoustic/acoustic_momentum_cartesian_agrid.comp",
    "acoustic/acoustic_null_mode_constraint_cartesian_agrid.comp",
    "acoustic/acoustic_boundary_cartesian_agrid.comp",
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


def require_rejection(result: subprocess.CompletedProcess[str], label: str) -> None:
    scope_markers = (
        SCOPE_ERROR,
        "qualified Metal execution is limited",
        "qualified Vulkan execution is limited",
        "qualified mapped terrain is limited",
        "strict nimbus.v1 load failed",
        "v1 config load failed",
    )
    if result.returncode == 0 or not any(
        marker in result.stdout for marker in scope_markers
    ):
        raise RuntimeError(f"{label} did not fail with the exact-scope guard:\n{result.stdout}")


def field_payload(output: Path, second: int, field: str):
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing {field} output: {path}")
    return read_float32_npy(path)


def execute(binary: Path, config: Path, output: Path, threads: int, label: str):
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, label)
    return result


def compare_with_cpu(reference: list[float], actual: list[float], field: str) -> float:
    if len(reference) != len(actual):
        raise RuntimeError(f"{field} CPU/native value counts differ")
    maximum = 0.0
    for expected, observed in zip(reference, actual):
        if not math.isfinite(expected) or not math.isfinite(observed):
            raise RuntimeError(f"{field} CPU/native comparison is non-finite")
        maximum = max(maximum, abs(expected - observed))
    tolerance = {
        "p": 0.08,
        "theta": 3.0e-3,
        "rho": 3.0e-5,
        "u": 1.0e-4,
        "v": 1.0e-4,
        "w": 1.0e-4,
    }.get(field, 5.0e-5)
    if maximum > tolerance:
        raise RuntimeError(
            f"{field} CPU/native maximum absolute error {maximum} exceeds {tolerance}"
        )
    return maximum


def require_native_provenance(log: str, backend: str, resumed: bool = False) -> None:
    for marker in (
        f"[COMPUTE] backend={backend}",
        "fallback=off",
        "[HARDWARE] GPU:",
    ):
        if marker not in log:
            raise RuntimeError(f"{backend} run omitted {marker!r}")
    line = next(
        (entry for entry in log.splitlines()
         if f"[COMPUTE DISPATCH] backend={backend}," in entry),
        "",
    )
    if not line:
        raise RuntimeError(f"{backend} run omitted dispatch provenance")
    if re.search(r"completed_submissions=[1-9][0-9]*", line) is None:
        raise RuntimeError(f"{backend} run omitted completed submissions: {line}")
    if re.search(r"kernel_dispatches=[1-9][0-9]*", line) is None:
        raise RuntimeError(f"{backend} run omitted completed dispatches: {line}")
    for kernel in KERNELS:
        minimum = 20 if resumed and kernel == KERNELS[0] else 1
        match = re.search(re.escape(kernel) + r":([0-9]+)", line)
        if match is None or int(match.group(1)) < minimum:
            raise RuntimeError(f"{backend} run omitted native {kernel}: {line}")
        if kernel == "dynamics/slow_apply.comp" and int(match.group(1)) != (20 if resumed else 40):
            raise RuntimeError(f"{backend} run omitted a native Heun slow stage: {line}")


def require_ledgers(output: Path) -> None:
    for name in LEDGERS:
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"native A-grid run omitted {name}")
        content = path.read_text().lower()
        if re.search(r"(^|,)(nan|inf|-inf)(,|$)", content, re.MULTILINE):
            raise RuntimeError(f"native A-grid {name} contains a non-finite value")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--backend", choices=tuple(CONFIGS), required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    backend = args.backend
    config = CONFIGS[backend]
    for path in (binary, CPU_CONFIG, config):
        if not path.is_file():
            raise RuntimeError(f"required native A-grid input is missing: {path}")

    preflight = run([str(binary), "--validate-only", "--config", str(config)])
    require_success(preflight, f"{backend} A-grid preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError(f"{backend} A-grid preflight omitted PASS")

    with tempfile.TemporaryDirectory(prefix=f"nimbus-agrid-{backend}-ready-") as temporary:
        root = Path(temporary)
        source = config.read_text()
        mutations = (
            ("fallback", "    allow_fallback: false\n", "    allow_fallback: true\n"),
            ("host slow owner", "    gpu_apply_slow: true\n", "    gpu_apply_slow: false\n"),
            ("terrain", "  scheme: none\n\ninitial_conditions:\n", "  scheme: bell\n\ninitial_conditions:\n"),
            ("open boundary", "    west: rigid\n", "    west: radiative\n"),
        )
        for name, old, new in mutations:
            changed = source.replace(old, new, 1)
            if changed == source:
                raise RuntimeError(f"{backend} {name} mutation did not alter the fixture")
            invalid = root / f"invalid-{name.replace(' ', '-')}.yaml"
            invalid.write_text(changed)
            require_rejection(
                run([str(binary), "--validate-only", "--config", str(invalid)]),
                f"{backend} {name} validate-only neighbor",
            )
            require_rejection(
                run([
                    str(binary), "--headless", "--config", str(invalid),
                    "--duration=0", f"--outdir={root / ('invalid-' + name)}",
                ]),
                f"{backend} {name} runtime neighbor",
            )

        cpu = root / "cpu"
        native_one = root / "native-one"
        native_four = root / "native-four"
        resumed = root / "resumed"
        execute(binary, CPU_CONFIG, cpu, 4, "A-grid CPU oracle run")
        one_result = execute(binary, config, native_one, 1, f"{backend} one-worker run")
        four_result = execute(binary, config, native_four, 4, f"{backend} four-worker run")
        require_native_provenance(one_result.stdout, backend)
        require_native_provenance(four_result.stdout, backend)

        maxima = {}
        evolved = set()
        for field in FIELDS:
            initial_shape, initial_raw, _ = field_payload(native_four, 0, field)
            one_shape, one_raw, _ = field_payload(native_one, 2, field)
            four_shape, four_raw, values = field_payload(native_four, 2, field)
            cpu_shape, _, cpu_values = field_payload(cpu, 2, field)
            if initial_shape != (18, 18, 18) or one_shape != initial_shape:
                raise RuntimeError(f"{backend} {field} output shape is invalid")
            if four_shape != one_shape or cpu_shape != one_shape:
                raise RuntimeError(f"{backend} {field} output shapes disagree")
            if one_raw != four_raw:
                raise RuntimeError(f"{backend} {field} differs between one/four workers")
            maxima[field] = compare_with_cpu(cpu_values, values, field)
            if initial_raw != four_raw:
                evolved.add(field)
            if field in ("rho", "p") and any(value <= 0.0 for value in values):
                raise RuntimeError(f"{backend} {field} is not strictly positive")
            if field in ("qv", "qr") and any(value < 0.0 for value in values):
                raise RuntimeError(f"{backend} {field} is negative")
        for field in ("w", "theta", "p", "rho"):
            if field not in evolved:
                raise RuntimeError(f"{backend} {field} did not evolve")

        require_ledgers(native_one)
        require_ledgers(native_four)
        for ledger in LEDGERS:
            if (native_one / ledger).read_bytes() != (native_four / ledger).read_bytes():
                raise RuntimeError(f"{backend} {ledger} differs between one/four workers")

        checkpoint = native_four / "checkpoints/checkpoint_step_000000010.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError(f"{backend} run omitted its midpoint checkpoint pair")
        continuation = run([
            str(binary), "--headless", "--config", str(config),
            f"--resume={checkpoint}", f"--outdir={resumed}",
            "--threads=4", "--log-profile=normal",
        ])
        require_success(continuation, f"{backend} midpoint continuation")
        if "[CHECKPOINT] resumed step=10, time_s=1" not in continuation.stdout:
            raise RuntimeError(f"{backend} continuation did not restore the midpoint")
        require_native_provenance(continuation.stdout, backend, resumed=True)
        for field in FIELDS:
            _, final_raw, _ = field_payload(native_four, 2, field)
            _, resumed_raw, _ = field_payload(resumed, 1, field)
            if resumed_raw != final_raw:
                raise RuntimeError(f"{backend} {field} restart is not bitwise exact")

    summary = ", ".join(f"{name}={value:.7g}" for name, value in maxima.items())
    print(
        f"[agrid-native-ready] PASS backend={backend}: fallback-off native complete "
        f"A-grid matched the CPU oracle ({summary}), was bitwise deterministic at "
        "one/four host workers, and resumed bitwise from the midpoint"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

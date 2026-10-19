#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
CONFIGS = {
    "vulkan": ROOT / "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_vulkan_ready.yaml",
    "metal": ROOT / "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_metal_ready.yaml",
}
FIELDS = ("p", "rho", "theta", "u", "v", "w", "qv", "qr")
LEDGERS = (
    "budget.csv", "energy_budget.csv", "moisture_budget.csv",
    "momentum_budget.csv", "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
SCOPE_ERROR = "microphysics.scheme=lin is qualified only"


def command(binary: Path, backend: str, arguments: list[str]) -> list[str]:
    result = [str(binary), *arguments]
    if backend == "vulkan":
        return [str(ROOT / "scripts/run_with_vulkan_env.sh"), *result]
    return result


def run(binary: Path, backend: str, arguments: list[str], threads: int = 1):
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    return subprocess.run(
        command(binary, backend, arguments), cwd=ROOT, env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected"):
        if marker in result.stdout:
            raise RuntimeError(f"{label} reported {marker}:\n{result.stdout}")


def final_frame(output: Path) -> Path:
    frames = sorted(path for path in output.glob("step_*") if path.is_dir())
    if not frames:
        raise RuntimeError(f"{output}: no published frames")
    return frames[-1]


def require_state_identity(reference: Path, candidate: Path, label: str) -> None:
    lhs_frame = final_frame(reference)
    rhs_frame = final_frame(candidate)
    for field in FIELDS:
        if (lhs_frame / f"{field}.npy").read_bytes() != (
                rhs_frame / f"{field}.npy").read_bytes():
            raise RuntimeError(f"{label}: final {field} bytes differ")


def require_ledger_identity(reference: Path, candidate: Path, label: str) -> None:
    for filename in LEDGERS:
        if (reference / filename).read_bytes() != (candidate / filename).read_bytes():
            raise RuntimeError(f"{label}: {filename} bytes differ")


def require_cpu_parity(reference: Path, candidate: Path) -> float:
    expected_frame, actual_frame = final_frame(reference), final_frame(candidate)
    worst = 0.0
    for name in FIELDS:
        shape, _, expected = read_float32_npy(expected_frame / f"{name}.npy")
        actual_shape, _, actual = read_float32_npy(actual_frame / f"{name}.npy")
        if shape != actual_shape or any(not math.isfinite(value) for values in (expected, actual) for value in values):
            raise RuntimeError(f"{candidate.name}: invalid parity field {name}")
        scale = max(map(abs, expected))
        ulp = math.ldexp(1, math.frexp(scale)[1] - 24) if scale else math.ldexp(1, -149)
        limit = 32 * 20 * ulp
        if name in ("u", "v", "w"):
            limit = max(limit, 20 * 5e-5)
        error = max(abs(a - b) for a, b in zip(expected, actual))
        if error > limit:
            raise RuntimeError(f"{candidate.name}: CPU parity failed for {name}: {error} > {limit}")
        worst = max(worst, error / limit)
    return worst


def require_restart_suffix(continuous: Path, resumed: Path) -> None:
    for filename in LEDGERS:
        reference = (continuous / filename).read_text().splitlines()
        candidate = (resumed / filename).read_text().splitlines()
        if not reference or len(candidate) < 2 or reference[0] != candidate[0]:
            raise RuntimeError(f"restart: malformed {filename}")
        if reference[-(len(candidate) - 1):] != candidate[1:]:
            raise RuntimeError(f"restart: {filename} is not an exact suffix")


def require_process_activity(output: Path) -> None:
    with (output / "thermodynamic_source_budget.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Lin native run published no source ledger")
    activity = max(
        abs(float(row[f"microphysics_{name}_requested_kg"]))
        for row in rows for name in ("qv", "qc")
    )
    if activity <= 0.0:
        raise RuntimeError("Lin native run did not exercise cloud processes")
    if any(int(row["microphysics_conservation_active"]) != 1 for row in rows):
        raise RuntimeError("Lin native run omitted conservation admission")
    for row in rows:
        residual = abs(float(row["microphysics_conservation_residual_kg"]))
        baseline = max(abs(float(row["microphysics_baseline_water_kg"])), 1.0e-30)
        if residual / baseline > 1.0e-6:
            raise RuntimeError("Lin native run exceeded its water closure bound")


def require_native_dispatch(log: str, backend: str, resumed: bool = False) -> None:
    lines = [line for line in log.splitlines() if "[COMPUTE DISPATCH]" in line]
    if not lines:
        raise RuntimeError(f"{backend} run omitted compute dispatch telemetry")
    line = lines[-1]
    expected = {
        "acoustic/acoustic_pressure_cartesian_cgrid.comp": 60 if resumed else 120,
        "acoustic/acoustic_momentum_cartesian_cgrid.comp": 60 if resumed else 120,
        "microphysics/lin_pointwise.comp": 10 if resumed else 20,
        "microphysics/lin_sedimentation.comp": 10 if resumed else 20,
        "radiation/rrtmg_longwave_transfer_batch.comp": 4 if resumed else 8,
        "radiation/rrtmg_shortwave_transfer_batch.comp": 4 if resumed else 8,
    }
    for kernel, count in expected.items():
        match = re.search(re.escape(kernel) + r":(\d+)", line)
        if match is None or int(match.group(1)) != count:
            raise RuntimeError(
                f"{backend} telemetry did not report {count} {kernel} dispatches: {line}"
            )
    if "microphysics/microphysics_apply.comp" in line:
        raise RuntimeError(f"{backend} strict profile unexpectedly used a device state commit")


def require_cloud_response(binary: Path, backend: str, source: str,
                           cloudy: Path, root: Path) -> None:
    clear_source = source.replace(
        "    mode: qualified\n", "    mode: exploratory\n", 1
    ).replace(
        "    mode: qualified\n",
        "    mode: qualified\n    acknowledge_maturity_limits: false\n", 1
    ).replace("  clear_sky_only: false\n", "  clear_sky_only: true\n", 1)
    clear_config = root / f"{backend}-clear.yaml"
    clear_config.write_text(clear_source)
    clear_output = root / f"{backend}-clear"
    result = run(binary, backend, [
        "--headless", "--config", str(clear_config),
        f"--outdir={clear_output}", "--write-every=4", "--checkpoint-every=0",
        "--threads=1", "--log-profile=quiet",
    ])
    require_success(result, f"{backend} clear-sky control")
    with (cloudy / "radiation_budget.csv").open(newline="") as stream:
        cloudy_rows = list(csv.DictReader(stream))
    with (clear_output / "radiation_budget.csv").open(newline="") as stream:
        clear_rows = list(csv.DictReader(stream))
    response = max(
        abs(float(lhs["held_atmospheric_power_w"]) -
            float(rhs["held_atmospheric_power_w"]))
        for lhs, rhs in zip(cloudy_rows, clear_rows)
    )
    if not cloudy_rows or len(cloudy_rows) != len(clear_rows) or response <= 1.0:
        raise RuntimeError(f"{backend} cloudy RRTMG produced no resolved response")


def require_rejection(binary: Path, backend: str, source: str,
                      old: str, new: str, root: Path, label: str) -> None:
    changed = source.replace(old, new, 1)
    if changed == source:
        raise RuntimeError(f"{backend} {label}: mutation anchor is missing")
    config = root / f"invalid-{backend}-{label}.yaml"
    config.write_text(changed)
    for mode, arguments in (
        ("validate", ["--validate-only"]),
        ("startup", ["--headless", "--duration=0", "--write-every=0",
                     f"--outdir={root / ('invalid-' + backend + '-' + label)}"]),
    ):
        result = run(binary, backend, [*arguments, "--config", str(config)])
        if result.returncode == 0 or SCOPE_ERROR not in result.stdout:
            raise RuntimeError(
                f"{backend} {label} {mode} did not fail closed:\n{result.stdout}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--backend", choices=("vulkan", "metal", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    backends = tuple(CONFIGS) if args.backend == "all" else (args.backend,)
    for path in (binary, *(CONFIGS[name] for name in backends)):
        if not path.is_file():
            raise RuntimeError(f"required Lin accelerator input is missing: {path}")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise RuntimeError("Lin native evidence directory must be empty")
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix="nimbus-lin-native-ready-")
    with context as temporary:
        root = Path(temporary)
        cpu_config = ROOT / "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_ready.yaml"
        cpu_output = root / "cpu-reference"
        cpu_result = run(binary, "cpu", [
            "--headless", "--config", str(cpu_config), f"--outdir={cpu_output}",
            "--write-every=4", "--checkpoint-every=0", "--threads=1", "--log-profile=normal",
        ])
        (root / "cpu-reference.log").write_text(cpu_result.stdout)
        require_success(cpu_result, "CPU Lin/RRTMG reference")
        parity = {}
        for backend in backends:
            config = CONFIGS[backend]
            source = config.read_text()
            (root / f"{backend}.yaml").write_text(source)
            preflight = run(binary, backend, ["--validate-only", "--config", str(config)])
            (root / f"{backend}-preflight.log").write_text(preflight.stdout)
            require_success(preflight, f"{backend} preflight")
            if "[VALIDATE] PASS" not in preflight.stdout:
                raise RuntimeError(f"{backend} preflight omitted PASS")

            require_rejection(
                binary, backend, source, "    allow_fallback: false\n",
                "    allow_fallback: true\n", root, "fallback")
            require_rejection(
                binary, backend, source, "    gpu_apply_microphysics: true\n",
                "    gpu_apply_microphysics: false\n", root, "ownership")

            outputs = {}
            logs = {}
            for threads in (1, 4):
                output = root / f"{backend}-{threads}"
                result = run(binary, backend, [
                    "--headless", "--config", str(config), f"--outdir={output}",
                    "--write-every=4", "--checkpoint-every=2",
                    f"--threads={threads}", "--log-profile=normal",
                ], threads)
                (root / f"{backend}-{threads}.log").write_text(result.stdout)
                require_success(result, f"{backend} {threads}-worker run")
                for marker in (f"[COMPUTE] backend={backend}", "fallback=off"):
                    if marker not in result.stdout:
                        raise RuntimeError(f"{backend} run omitted {marker!r}")
                require_native_dispatch(result.stdout, backend)
                outputs[threads] = output
                logs[threads] = result.stdout

            require_state_identity(outputs[1], outputs[4], f"{backend} threading")
            require_ledger_identity(outputs[1], outputs[4], f"{backend} threading")
            require_process_activity(outputs[4])
            parity[backend] = require_cpu_parity(cpu_output, outputs[1])
            require_cloud_response(binary, backend, source, outputs[1], root)

            checkpoint = outputs[4] / "checkpoints/checkpoint_step_000000010.nimbus"
            resumed = root / f"{backend}-resumed"
            result = run(binary, backend, [
                "--headless", "--config", str(config), f"--resume={checkpoint}",
                "--duration=4", "--write-every=4", "--checkpoint-every=0",
                f"--outdir={resumed}", "--threads=4", "--log-profile=normal",
            ], 4)
            (root / f"{backend}-resumed.log").write_text(result.stdout)
            require_success(result, f"{backend} restart")
            if "[CHECKPOINT] resumed step=10, time_s=2" not in result.stdout:
                raise RuntimeError(f"{backend} restart omitted restore marker")
            require_native_dispatch(result.stdout, backend, resumed=True)
            require_state_identity(outputs[4], resumed, f"{backend} restart")
            require_restart_suffix(outputs[4], resumed)
        (root / "summary.json").write_text(json.dumps({
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "backends": list(backends), "public_admission": True,
            "native_dispatch": True, "process_activity": True,
            "cloud_response": True, "threading": "exact", "restart": "exact", "conservation": True,
            "cpu_parity_maximum_error_to_limit": parity,
        }, indent=2)+"\n")

    print(
        "[lin-rrtmg-accelerator-ready] PASS: ordinary fallback-off native Lin "
        "process/sedimentation and cloudy RRTMG execution, one/four-worker "
        "identity, exact restart, conservation, and scope gates passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

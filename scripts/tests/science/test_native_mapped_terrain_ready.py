#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.qualification.checkpointing.qualify_vulkan_restart import (  # noqa: E402
    require_report_owner_identities,
)
from scripts.tests.science.test_earth_fixed_terrain_motion_ready import read_npy

DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
CONFIGS = {
    backend: ROOT / "tests/fixtures/configs/qualification/"
    / f"cgrid_stationary_bell_{backend}_complete_ready.yaml"
    for backend in ("vulkan", "metal")
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
    "qualified mapped terrain is limited to the exact stationary-bell "
    "and constant-velocity Schar CPU reference profiles"
)
KERNEL_COUNTS_PER_STEP = {
    "advection/advect_conservative_scalar_flux_cartesian_cgrid.comp": 4,
    "terrain/mapped_mass_continuity.comp": 16,
    "terrain/mapped_fast_momentum.comp": 6,
    "terrain/mapped_slow_tendencies.comp": 2,
    "terrain/mapped_scalar_update.comp": 4,
    "terrain/mapped_scalar_limiter.comp": 3,
    "terrain/mapped_boundary.comp": 14,
    "terrain/mapped_state_update.comp": 14,
    "terrain/mapped_flux_accumulation.comp": 12,
}
SUBMISSIONS_PER_STEP = {"vulkan": 75, "metal": 60}


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
    markers = (
        SCOPE_ERROR,
        "qualified Vulkan execution is limited",
        "qualified Metal execution is limited",
        "strict nimbus.v1 load failed",
        "v1 config load failed",
    )
    if result.returncode == 0 or not any(marker in result.stdout for marker in markers):
        raise RuntimeError(
            f"{label} did not fail with the exact-scope guard:\n{result.stdout}"
        )


def execute(
    binary: Path, config: Path, output: Path, threads: int, label: str
):
    result = run([
        str(binary),
        "--headless",
        "--config",
        str(config),
        f"--outdir={output}",
        f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, label)
    return result


def field_payload(output: Path, second: int, field: str):
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing {field} output: {path}")
    return read_float32_npy(path)


def require_native_provenance(log: str, backend: str, steps: int, terrain: str = "bell") -> None:
    for marker in (
        "Execution mode: qualified",
        f"Terrain: {terrain}",
        f"[COMPUTE] backend={backend}",
        "fallback=off",
        "[HARDWARE] GPU:",
    ):
        if marker not in log:
            raise RuntimeError(f"native mapped run omitted {marker!r}")

    transaction = (
        f"[TERRAIN_NATIVE] backend={backend}, fallback=off, spatial_owner=device, "
        "final_eos_owner=host_fp64, complete=1, acoustic_substeps=6, "
        f"backend_dispatches={SUBMISSIONS_PER_STEP[backend]}, mass=8, slow=2, slow_average=6, "
        "fast=6, state=14, "
        "flux_accum=6, scalar_flux=4, scalar_limiter=3, scalar_update=4, "
        "boundary=14"
    )
    if log.count(transaction) != steps:
        raise RuntimeError(
            f"native mapped transaction provenance count was {log.count(transaction)}, "
            f"expected {steps}"
        )

    line = next(
        (entry for entry in log.splitlines()
         if f"[COMPUTE DISPATCH] backend={backend}," in entry),
        "",
    )
    expected_submissions = SUBMISSIONS_PER_STEP[backend] * steps
    if f"completed_submissions={expected_submissions}" not in line:
        raise RuntimeError(
            f"native mapped run omitted exact completed-submission provenance: {line}"
        )
    expected_kernel_dispatches = 75 * steps
    if f"kernel_dispatches={expected_kernel_dispatches}" not in line:
        raise RuntimeError(
            f"native mapped run omitted exact kernel-dispatch provenance: {line}"
        )
    for kernel, per_step in KERNEL_COUNTS_PER_STEP.items():
        match = re.search(re.escape(kernel) + r":([0-9]+)", line)
        expected = per_step * steps
        if match is None or int(match.group(1)) != expected:
            raise RuntimeError(
                f"native mapped run expected {expected} dispatches of {kernel}: {line}"
            )
    kernel_block = re.search(r"kernels=\{(.*)\}$", line)
    if kernel_block is None:
        raise RuntimeError(f"native mapped run omitted its kernel set: {line}")
    observed_kernels = {
        item.rsplit(":", 1)[0]
        for item in kernel_block.group(1).split(",")
        if item
    }
    if observed_kernels != set(KERNEL_COUNTS_PER_STEP):
        raise RuntimeError(
            "native mapped run reported unexpected or missing kernels: "
            f"{observed_kernels ^ set(KERNEL_COUNTS_PER_STEP)}"
        )


def require_ledgers(output: Path) -> int:
    for name in LEDGERS:
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"native mapped run omitted {name}")
        content = path.read_text().lower()
        if re.search(r"(^|,)(nan|inf|-inf)(,|$)", content, re.MULTILINE):
            raise RuntimeError(f"native mapped {name} contains a non-finite value")
    return require_report_owner_identities(output)


def require_restart_suffix(continuous: Path, resumed: Path) -> None:
    for filename in LEDGERS:
        continuous_lines = (continuous / filename).read_text().splitlines()
        resumed_lines = (resumed / filename).read_text().splitlines()
        if not continuous_lines or not resumed_lines or (
            continuous_lines[0] != resumed_lines[0]
        ):
            raise RuntimeError(f"{filename}: restart changed the ledger schema")
        if len(resumed_lines) <= 1 or (
            continuous_lines[-(len(resumed_lines) - 1):] != resumed_lines[1:]
        ):
            raise RuntimeError(
                f"{filename}: resumed ledger is not an exact continuous suffix"
            )

    continuous_radiation = continuous / "radiation_budget.csv"
    resumed_radiation = resumed / "radiation_budget.csv"
    if continuous_radiation.read_bytes() != resumed_radiation.read_bytes():
        raise RuntimeError("inactive radiation ledger changed across restart")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--backend", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--terrain", choices=("bell", "schar"), default="bell")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    backend = args.backend
    terrain = args.terrain
    config = ROOT / "tests/fixtures/configs/qualification" / f"cgrid_stationary_{terrain}_{backend}_complete_ready.yaml"
    for path in (binary, config):
        if not path.is_file():
            raise RuntimeError(f"required native mapped input is missing: {path}")

    preflight = run([str(binary), "--validate-only", "--config", str(config)])
    require_success(preflight, "native mapped preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("native mapped preflight omitted PASS")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise RuntimeError("native mapped evidence directory must be empty")
    output_context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(
        prefix=f"nimbus-native-mapped-{backend}-ready-")
    with output_context as temporary:
        root = Path(temporary)
        source = config.read_text()
        height_key = "h0" if terrain == "bell" else "H"
        mutations = (
            ("fallback", "    allow_fallback: false\n", "    allow_fallback: true\n"),
            ("host slow owner", "    gpu_apply_slow: true\n", "    gpu_apply_slow: false\n"),
            ("terrain geometry", f"    {height_key}: 220\n", f"    {height_key}: 221\n"),
            ("open boundary", "    west: periodic\n", "    west: radiative\n"),
        )
        for name, old, new in mutations:
            changed = source.replace(old, new, 1)
            if changed == source:
                raise RuntimeError(f"{name} mutation did not alter the fixture")
            invalid = root / f"invalid-{name.replace(' ', '-')}.yaml"
            invalid.write_text(changed)
            require_rejection(
                run([str(binary), "--validate-only", "--config", str(invalid)]),
                f"{name} validate-only neighbor",
            )
            require_rejection(
                run([
                    str(binary),
                    "--headless",
                    "--config",
                    str(invalid),
                    "--duration=0",
                    f"--outdir={root / ('invalid-' + name.replace(' ', '-'))}",
                ]),
                f"{name} runtime neighbor",
            )

        one = root / "one-worker"
        four = root / "four-worker"
        resumed = root / "resumed"
        one_result = execute(
            binary, config, one, 1, f"{backend} mapped one-worker run"
        )
        four_result = execute(
            binary, config, four, 4, f"{backend} mapped four-worker run"
        )
        require_native_provenance(one_result.stdout, backend, 20, terrain)
        require_native_provenance(four_result.stdout, backend, 20, terrain)

        cpu_config = root / "cpu-reference.yaml"
        cpu_config.write_text(source.replace(f"    backend: {backend}", "    backend: cpu", 1)
                              .replace("    gpu_apply_slow: true", "    gpu_apply_slow: false", 1))
        cpu_output = root / "cpu-reference"
        execute(binary, cpu_config, cpu_output, 1, "mapped CPU reference")
        errors = {}
        for second in (1, 2):
            steps = 10 * second
            for field in FIELDS:
                expected_shape, _, expected = field_payload(cpu_output, second, field)
                actual_shape, _, actual = field_payload(one, second, field)
                if expected_shape != (12, 12, 18) or actual_shape != expected_shape:
                    raise RuntimeError(f"CPU/native parity shape mismatch for {field}")
                if any(not math.isfinite(value) for values in (expected, actual) for value in values):
                    raise RuntimeError(f"CPU/native parity has non-finite {field}")
                if field in ("u", "v", "w"):
                    # Same linear horizon rule and stencil absolute band as
                    # src/compute/parity_comparison.cpp; no relative allowance.
                    tolerance = steps * 5e-5
                elif field in ("qv", "qr"):
                    tolerance = 0.0
                else:
                    # Four stored-float ULPs per outer step cover the different
                    # FP32/FP64 EOS and conservative scalar evaluations.
                    exponent = math.frexp(max(abs(x) for x in expected))[1]
                    tolerance = steps * 4 * math.ldexp(1.0, exponent - 24)
                error = max(abs(a-b) for a,b in zip(actual,expected))
                errors[f"{second}s/{field}"] = {"maximum_absolute_error": error, "absolute_limit": tolerance}
                if terrain == "schar" and field in ("u", "v", "w"):
                    for label, values in (("cpu", expected), (backend, actual)):
                        spread = max(
                            max(values[(i*12+j)*18+k] for j in range(1,11)) -
                            min(values[(i*12+j)*18+k] for j in range(1,11))
                            for i in range(1,11) for k in range(18))
                        errors[f"{second}s/{label}/{field}/ridge_symmetry"] = {
                            "maximum_absolute_error": spread, "absolute_limit": 5e-5}
        (root / "cpu-parity.json").write_text(json.dumps(errors, indent=2) + "\n")
        failures = {field: values for field, values in errors.items()
                    if values["maximum_absolute_error"] > values["absolute_limit"]}
        if failures:
            raise RuntimeError(f"{terrain}/{backend} CPU reference parity failures: {failures}")
        if terrain == "schar":
            step = one / "step_000000"
            manifest = json.loads((step / "manifest.json").read_text())
            geometry = manifest["grid"]["vertical_geometry"]
            shape, heights = read_npy((step / geometry["interface_path"]).resolve())
            if shape != (12,12,19):
                raise RuntimeError("Schar geometry shape is invalid")
            for i in range(12):
                x = (i+0.5)*500.0-3000.0
                expected = 220.0*math.exp(-(x/1600.0)**2)*math.cos(math.pi*x/2000.0)**2
                for j in range(12):
                    if abs(heights[(i*12+j)*19]-expected) > 1e-10:
                        raise RuntimeError("native Schar output does not match the analytic surface")

        evolved = set()
        for field in FIELDS:
            initial_shape, initial_raw, _ = field_payload(four, 0, field)
            one_shape, one_raw, _ = field_payload(one, 2, field)
            four_shape, four_raw, values = field_payload(four, 2, field)
            if initial_shape != (12, 12, 18) or one_shape != initial_shape:
                raise RuntimeError(f"native mapped {field} output shape is invalid")
            if four_shape != one_shape or one_raw != four_raw:
                raise RuntimeError(
                    f"native mapped {field} differs between one/four workers"
                )
            if initial_raw != four_raw:
                evolved.add(field)
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError(f"native mapped {field} is non-finite")
            if field in ("rho", "p", "theta") and any(
                value <= 0.0 for value in values
            ):
                raise RuntimeError(f"native mapped {field} is not strictly positive")
            if field in ("qv", "qr") and any(value < 0.0 for value in values):
                raise RuntimeError(f"native mapped {field} is negative")
        for field in ("u", "v", "w", "theta", "p", "rho"):
            if field not in evolved:
                raise RuntimeError(f"native mapped {field} did not evolve")

        one_checks = require_ledgers(one)
        four_checks = require_ledgers(four)
        if one_checks == 0 or four_checks != one_checks:
            raise RuntimeError("native mapped owner-identity coverage is incomplete")
        for ledger in LEDGERS:
            if (one / ledger).read_bytes() != (four / ledger).read_bytes():
                raise RuntimeError(
                    f"native mapped {ledger} differs between one/four workers"
                )

        checkpoint = four / "checkpoints/checkpoint_step_000000010.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("native mapped run omitted its midpoint checkpoint pair")
        for step in (10,20):
            name=f"checkpoints/checkpoint_step_{step:09d}.nimbus"
            if (one/name).read_bytes() != (four/name).read_bytes():
                raise RuntimeError("native mapped checkpoint differs across workers")
        continuation = run([
            str(binary),
            "--headless",
            "--config",
            str(config),
            f"--resume={checkpoint}",
            f"--outdir={resumed}",
            "--threads=4",
            "--log-profile=normal",
        ])
        require_success(continuation, "native mapped midpoint continuation")
        if "[CHECKPOINT] resumed step=10, time_s=1" not in continuation.stdout:
            raise RuntimeError("native mapped continuation did not restore the midpoint")
        require_native_provenance(continuation.stdout, backend, 10, terrain)
        require_ledgers(resumed)
        require_restart_suffix(four, resumed)
        for field in FIELDS:
            _, final_raw, _ = field_payload(four, 2, field)
            _, resumed_raw, _ = field_payload(resumed, 1, field)
            if resumed_raw != final_raw:
                raise RuntimeError(f"native mapped {field} restart is not bitwise exact")

        (root / "result.json").write_text(json.dumps({
            "backend": backend, "terrain": terrain, "status": "passed",
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cpu_parity": errors, "owner_identity_checks": one_checks,
            "threading": "bitwise", "restart": "bitwise",
        }, indent=2) + "\n")

    print(
        f"[native-mapped-terrain-ready] PASS backend={backend}: the exact dry "
        f"stationary {terrain} fallback-off profile dispatched every native mapped stage, "
        "closed its bounded owner identities, matched one/four workers, rejected "
        "near profiles, and resumed bitwise from the midpoint"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

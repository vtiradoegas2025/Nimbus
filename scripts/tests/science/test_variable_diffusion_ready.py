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
CONFIGS = {
    "vulkan": ROOT / "tests/fixtures/configs/qualification/cgrid_variable_diffusion_vulkan_ready.yaml",
    "metal": ROOT / "tests/fixtures/configs/qualification/cgrid_variable_diffusion_metal_ready.yaml",
}
FIELDS = ("u", "v", "w", "theta", "p", "rho", "qv")
SCOPE_ERROR = (
    "qualified linear-height variable diffusion is limited to the exact flat "
    "native reference profiles"
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


def require_rejection(
    result: subprocess.CompletedProcess[str], label: str, marker: str
) -> None:
    if result.returncode == 0 or marker not in result.stdout:
        raise RuntimeError(f"{label} did not fail closed:\n{result.stdout}")


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


def cpu_reference_source(source: str) -> str:
    result = source.replace("    mode: qualified\n", "    mode: exploratory\n", 1)
    result = result.replace(
        "    mode: qualified\n",
        "    mode: qualified\n    acknowledge_maturity_limits: false\n",
        1,
    )
    result = result.replace("    backend: vulkan\n", "    backend: cpu\n", 1)
    if result == source:
        raise RuntimeError("CPU reference mutation did not alter the Vulkan fixture")
    return result


def constant_control_source(source: str) -> str:
    result = cpu_reference_source(source)
    result = result.replace(
        "  coefficient_source: linear_height\n",
        "  coefficient_source: constant\n",
        1,
    )
    result = result.replace("  K_v: 0\n", "  K_v: 12\n", 1)
    result = result.replace("  K_v_surface: 4\n", "", 1)
    result = result.replace("  K_v_top: 20\n", "", 1)
    if "linear_height" in result or "K_v_surface" in result or "K_v_top" in result:
        raise RuntimeError("constant control retained prescribed-profile keys")
    return result


def compare_with_tolerance(reference: list[float], actual: list[float], field: str) -> None:
    if len(reference) != len(actual):
        raise RuntimeError(f"{field} CPU/native value counts differ")
    maximum = 0.0
    for expected, observed in zip(reference, actual):
        if not math.isfinite(expected) or not math.isfinite(observed):
            raise RuntimeError(f"{field} contains a non-finite value")
        maximum = max(maximum, abs(expected - observed))
    if field == "p":
        tolerance = 2.0e-2
    elif field == "theta":
        tolerance = 2.0e-3
    else:
        tolerance = 3.0e-5
    if maximum > tolerance:
        raise RuntimeError(
            f"{field} CPU/native maximum absolute error {maximum} exceeds {tolerance}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--backend", choices=("vulkan", "metal", "all"), default="all")
    args = parser.parse_args()
    binary = args.binary.resolve()
    backends = tuple(CONFIGS) if args.backend == "all" else (args.backend,)
    for path in (binary, *(CONFIGS[name] for name in backends)):
        if not path.is_file():
            raise RuntimeError(f"required variable-diffusion input is missing: {path}")

    with tempfile.TemporaryDirectory(prefix="nimbus-variable-diffusion-ready-") as temporary:
        root = Path(temporary)
        vulkan_source = CONFIGS["vulkan"].read_text()
        cpu_config = root / "cpu-reference.yaml"
        cpu_config.write_text(cpu_reference_source(vulkan_source))
        cpu_serial = root / "cpu-serial"
        cpu_parallel = root / "cpu-parallel"
        constant_control = root / "cpu-constant-control"
        execute(binary, cpu_config, cpu_serial, 1, "one-worker CPU reference")
        execute(binary, cpu_config, cpu_parallel, 4, "four-worker CPU reference")
        for field in FIELDS:
            serial_shape, serial_raw, _ = field_payload(cpu_serial, 2, field)
            parallel_shape, parallel_raw, _ = field_payload(cpu_parallel, 2, field)
            if serial_shape != (12, 12, 12) or parallel_shape != serial_shape:
                raise RuntimeError(f"{field} CPU reference shape is invalid")
            if serial_raw != parallel_raw:
                raise RuntimeError(f"{field} differs between one and four CPU workers")
        control_config = root / "cpu-constant-control.yaml"
        control_config.write_text(constant_control_source(vulkan_source))
        execute(binary, control_config, constant_control, 4, "constant-K CPU control")
        profile_changed_solution = False
        for field in ("w", "theta"):
            _, variable_raw, _ = field_payload(cpu_parallel, 2, field)
            _, control_raw, _ = field_payload(constant_control, 2, field)
            profile_changed_solution = (
                profile_changed_solution or variable_raw != control_raw
            )
        if not profile_changed_solution:
            raise RuntimeError(
                "linear-height profile did not differ from the constant-K control"
            )

        for backend in backends:
            config = CONFIGS[backend]
            source = config.read_text()
            preflight = run([str(binary), "--validate-only", "--config", str(config)])
            require_success(preflight, f"{backend} qualified preflight")
            if "[VALIDATE] PASS" not in preflight.stdout:
                raise RuntimeError(f"{backend} preflight omitted PASS")

            mutations = (
                ("surface coefficient", "  K_v_surface: 4\n", "  K_v_surface: 5\n", SCOPE_ERROR),
                ("constant coefficient ambiguity", "  K_v: 0\n", "  K_v: 1\n", "must be zero when coefficient_source is linear_height"),
                ("turbulence double owner", "  scheme: none\n  mode: 3d\n", "  scheme: smagorinsky\n  mode: 3d\n", SCOPE_ERROR),
                ("fallback", "    allow_fallback: false\n", "    allow_fallback: true\n", SCOPE_ERROR),
            )
            for name, old, new, marker in mutations:
                changed = source.replace(old, new, 1)
                if changed == source:
                    raise RuntimeError(f"{backend} {name} mutation did not alter the fixture")
                invalid = root / f"invalid-{backend}-{name.replace(' ', '-')}.yaml"
                invalid.write_text(changed)
                require_rejection(
                    run([str(binary), "--validate-only", "--config", str(invalid)]),
                    f"{backend} {name} validate-only neighbor",
                    marker,
                )
                require_rejection(
                    run([str(binary), "--headless", "--config", str(invalid),
                         "--duration=0", f"--outdir={root / ('invalid-' + backend)}"]),
                    f"{backend} {name} runtime neighbor",
                    marker,
                )

            native = root / backend
            resumed = root / f"{backend}-resumed"
            result = execute(binary, config, native, 4, f"physical {backend} run")
            for marker in (f"[COMPUTE] backend={backend}", "fallback=off",
                           "coefficient_source=linear_height",
                           "K_v_surface=4", "K_v_top=20", "[HARDWARE] GPU:"):
                if marker not in result.stdout:
                    raise RuntimeError(f"{backend} run omitted {marker!r}")
            dispatch_line = next(
                (
                    line
                    for line in result.stdout.splitlines()
                    if f"[COMPUTE DISPATCH] backend={backend}," in line
                ),
                "",
            )
            dispatch_patterns = (
                r"completed_submissions=[1-9][0-9]*",
                r"kernel_dispatches=[1-9][0-9]*",
                r"diffusion/diffusion\.comp:[1-9][0-9]*",
            )
            if not dispatch_line or any(
                re.search(pattern, dispatch_line) is None
                for pattern in dispatch_patterns
            ):
                raise RuntimeError(
                    f"{backend} run omitted completed native diffusion provenance: "
                    f"{dispatch_line!r}"
                )
            for field in FIELDS:
                initial_shape, initial_raw, _ = field_payload(native, 0, field)
                final_shape, final_raw, native_values = field_payload(native, 2, field)
                cpu_shape, _, cpu_values = field_payload(cpu_parallel, 2, field)
                if initial_shape != final_shape or final_shape != cpu_shape:
                    raise RuntimeError(f"{backend} {field} output shape changed")
                compare_with_tolerance(cpu_values, native_values, field)
                if field in ("w", "theta") and initial_raw == final_raw:
                    raise RuntimeError(f"{backend} {field} did not evolve")

            checkpoint = native / "checkpoints/checkpoint_step_000000010.nimbus"
            if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
                raise RuntimeError(f"{backend} run omitted its midpoint checkpoint pair")
            continuation = run([
                str(binary), "--headless", "--config", str(config),
                f"--resume={checkpoint}", f"--outdir={resumed}",
                "--threads=4", "--log-profile=normal",
            ])
            require_success(continuation, f"{backend} checkpoint continuation")
            if "[CHECKPOINT] resumed step=10, time_s=1" not in continuation.stdout:
                raise RuntimeError(f"{backend} continuation did not restore the midpoint")
            for field in FIELDS:
                _, final_raw, _ = field_payload(native, 2, field)
                _, resumed_raw, _ = field_payload(resumed, 1, field)
                if resumed_raw != final_raw:
                    raise RuntimeError(f"{backend} {field} restart is not bitwise exact")

    print(
        "[variable-diffusion-ready] PASS: physical native profiles were "
        "fallback-off, matched the CPU reference, and resumed bitwise; CPU "
        "reference fields were bitwise deterministic at one/four workers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

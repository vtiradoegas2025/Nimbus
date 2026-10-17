#!/usr/bin/env python3
"""Run and inspect a same-binary oversized physical-accelerator reference.

This closes the missing backend half of the moving-nest comparison campaign.
It does not itself declare the comparison qualified; the extracted moving and
reference series still have to pass a separately bound policy/comparator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import extract_moving_nested_reference_series as extractor
import qualify_stationary_nested_vulkan_runtime as stationary


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim_internal_qualification"
DEFAULT_VALIDATOR = ROOT / "bin/field_validator"
DEFAULT_CONFIG = (
    ROOT / "tests/fixtures/configs/qualification/"
    "moving_nested_supercell_reference_vulkan_v1.yaml"
)
DEFAULT_METAL_CONFIG = (
    ROOT / "tests/fixtures/configs/qualification/"
    "moving_nested_supercell_reference_metal_v1.yaml"
)
CPU_REFERENCE_CONFIG = (
    ROOT / "tests/fixtures/configs/qualification/moving_nested_supercell_reference_v5.yaml"
)
MOVING_CHAIN = (
    ROOT / "tests/artifacts/"
    "moving_nested_vulkan_lifetime_v2_20260819_chain_report.json"
)
METAL_MOVING_CHAIN = (
    ROOT / "tests/artifacts/"
    "moving_nested_metal_lifetime_v1_chain_report.json"
)
DEFAULT_OUTPUT = (
    ROOT / "tests/artifacts/"
    "moving_nested_vulkan_reference_v1_20260819"
)
DEFAULT_METAL_OUTPUT = (
    ROOT / "tests/artifacts/"
    "moving_nested_metal_reference_v1"
)
REPORT_SCHEMA = "nimbus.diagnostic.moving_nested_vulkan_reference_runtime"
SUPPORTED_BACKENDS = ("vulkan", "metal")
ALLOWED_CONFIG_DELTAS = {
    "extensions.compute.backend",
    "extensions.compute.allow_fallback",
    "extensions.compute.gpu_apply_microphysics",
    "extensions.compute.gpu_apply_slow",
    "extensions.compute.gpu_apply_sgs",
    "extensions.checkpoint.every_s",
}
EXPECTED_VULKAN_OWNERSHIP = {
    "extensions.compute.backend": "vulkan",
    "extensions.compute.allow_fallback": "false",
    "extensions.compute.gpu_apply_microphysics": "false",
    "extensions.compute.gpu_apply_slow": "false",
    "extensions.compute.gpu_apply_sgs": "true",
    "extensions.checkpoint.every_s": "450",
}


def expected_ownership(compute_backend: str) -> dict[str, str]:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    return {
        **EXPECTED_VULKAN_OWNERSHIP,
        "extensions.compute.backend": compute_backend,
    }


def default_config(compute_backend: str) -> Path:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    return DEFAULT_CONFIG if compute_backend == "vulkan" else DEFAULT_METAL_CONFIG


def default_chain(compute_backend: str) -> Path:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    return MOVING_CHAIN if compute_backend == "vulkan" else METAL_MOVING_CHAIN


def default_output(compute_backend: str) -> Path:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    return DEFAULT_OUTPUT if compute_backend == "vulkan" else DEFAULT_METAL_OUTPUT


def reference_scope(compute_backend: str, duration_s: int) -> str:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    if duration_s == 4500:
        return f"same_binary_oversized_physical_{compute_backend}_reference_runtime"
    return f"oversized_physical_{compute_backend}_reference_smoke"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def validate_config_delta(config: Path, compute_backend: str = "vulkan") -> None:
    cpu = extractor.flat_yaml_scalars(CPU_REFERENCE_CONFIG)
    accelerator = extractor.flat_yaml_scalars(config)
    cpu_common = {
        key: value for key, value in cpu.items()
        if key not in ALLOWED_CONFIG_DELTAS
    }
    accelerator_common = {
        key: value for key, value in accelerator.items()
        if key not in ALLOWED_CONFIG_DELTAS
    }
    require(cpu_common == accelerator_common,
            f"{compute_backend.capitalize()} reference differs from the sealed "
            "CPU reference outside "
            "compute ownership/checkpoint cadence")
    expected = expected_ownership(compute_backend)
    observed = {key: accelerator.get(key) for key in expected}
    require(observed == expected,
            f"{compute_backend.capitalize()} reference ownership tuple changed: "
            f"{observed}")


def validate_binary_identity(binary: Path, moving_chain: Path = MOVING_CHAIN,
                             compute_backend: str = "vulkan") -> str:
    chain = load_json(moving_chain)
    expected = chain.get("inputs", {}).get("binary_sha256")
    actual = sha256(binary)
    require(chain.get("schema") ==
            f"nimbus.diagnostic.moving_nested_{compute_backend}_lifetime_chain" and
            chain.get("status") == "complete" and
            chain.get("claims", {}).get(
                f"fallback_off_physical_{compute_backend}") is True and
            chain.get("claims", {}).get("uninterrupted_single_process") is True,
            f"moving {compute_backend} lifetime is not the completed "
            "uninterrupted campaign")
    require(actual == expected,
            f"reference binary differs from the completed moving "
            f"{compute_backend} binary")
    return actual


def require_device_log(log: str, compute_backend: str) -> None:
    required = (
        f"[COMPUTE] backend={compute_backend}, requested={compute_backend}, "
        "device_index=-1, fallback=off"
    )
    require(required in log,
            f"run did not prove fallback-off physical {compute_backend} execution")


def timed_publications(root: Path, pattern: str, manifest_name: str | None) -> list[float]:
    times: list[float] = []
    for path in sorted(root.glob(pattern)):
        document = path / manifest_name if manifest_name is not None else path
        if not document.is_file():
            continue
        value = load_json(document).get("simulation_time_s")
        require(type(value) in (int, float) and math.isfinite(float(value)),
                f"publication lacks finite simulation time: {document}")
        times.append(float(value))
    return times


def parse_gpu_telemetry(log: str, compute_backend: str = "vulkan") -> dict:
    if compute_backend == "metal":
        rows = list(re.finditer(
            r"\[COMPUTE DISPATCH\] backend=metal, "
            r"completed_submissions=([0-9]+), kernel_dispatches=([0-9]+), "
            r"resident_uploads=([0-9]+), resident_reuses=([0-9]+), "
            r"kernels=\{([^}]*)\}", log))
        require(rows, "runtime log lacks physical Metal dispatch telemetry")
        command_buffers, dispatches, uploads, reuses, stage_text = rows[-1].groups()
        stages: dict[str, int] = {}
        for item in stage_text.split(","):
            if not item:
                continue
            name, separator, count = item.rpartition(":")
            require(bool(separator) and bool(name) and count.isdigit(),
                    "Metal stage dispatch telemetry is malformed")
            stages[name] = int(count)
        performance = re.search(
            r"\[PERF SUMMARY\] measured_steps=([0-9]+)", log)
        require(performance is not None,
                "runtime log lacks Metal model-step telemetry")
        require(int(command_buffers) > 0 and int(dispatches) > 0,
                "physical Metal dispatch telemetry is vacuous")
        require(any(count > 0 for name, count in stages.items()
                    if "smagorinsky" in name or "sgs" in name),
                "runtime log lacks native Metal Smagorinsky/SGS dispatch evidence")
        require(any(count > 0 for name, count in stages.items()
                    if "kessler" in name),
                "runtime log lacks native Metal Kessler process dispatch evidence")
        return {
            "model_steps": int(performance.group(1)),
            "command_buffers": int(command_buffers),
            "dispatches": int(dispatches),
            "resident_hits": int(reuses),
            "resident_misses": int(uploads),
            "stage_dispatches": dict(sorted(stages.items())),
        }
    require(compute_backend == "vulkan",
            f"unsupported telemetry backend: {compute_backend}")
    rows = list(re.finditer(
        r"\[GPU TELEMETRY\] model_steps=([0-9]+) command_buffers=([0-9]+) "
        r"dispatches=([0-9]+).*?resident_hits=([0-9]+) "
        r"resident_misses=([0-9]+)", log))
    require(rows, "runtime log lacks physical GPU telemetry")
    # Strict pre-allocation validation constructs and destroys an empty probe
    # backend before the actual runtime backend. Bind the final telemetry row,
    # which belongs to the completed simulation process.
    row = rows[-1]
    model_steps, command_buffers, dispatches, hits, misses = (
        int(value) for value in row.groups())
    require(model_steps > 0 and command_buffers > 0 and dispatches > 0,
            "physical GPU telemetry is vacuous")
    stages = {
        match.group(1): int(match.group(2))
        for match in re.finditer(
            r"\[GPU STAGE TELEMETRY\] ([^ ]+) command_buffers=[0-9]+ "
            r"dispatches=([0-9]+)", log)
    }
    require(any(count > 0 for name, count in stages.items()
                if "smagorinsky" in name or "sgs" in name),
            "runtime log lacks native Smagorinsky/SGS dispatch evidence")
    require(any(count > 0 for name, count in stages.items()
                if "kessler" in name),
            "runtime log lacks native Kessler process dispatch evidence")
    return {
        "model_steps": model_steps,
        "command_buffers": command_buffers,
        "dispatches": dispatches,
        "resident_hits": hits,
        "resident_misses": misses,
        "stage_dispatches": dict(sorted(stages.items())),
    }


def inspect(output: Path, binary: Path, validator: Path, config: Path,
            duration_s: int, wall_seconds: float | None,
            compute_backend: str = "vulkan",
            moving_chain: Path = MOVING_CHAIN) -> dict:
    validate_config_delta(config, compute_backend)
    complete_reference = duration_s == 4500
    binary_digest = (
        validate_binary_identity(binary, moving_chain, compute_backend)
        if complete_reference else sha256(binary))
    require(sha256(output / "qualification_input.yaml") == sha256(config),
            "runtime config snapshot changed")
    recorded_binary = (output / "qualification_binary.sha256").read_text(
        encoding="ascii").strip()
    require(recorded_binary == binary_digest,
            "runtime binary snapshot changed")
    log_path = output / "qualification_run.log"
    log = log_path.read_text(encoding="utf-8")
    require("[ERROR]" not in log, "runtime log contains an error")
    require_device_log(log, compute_backend)
    telemetry = parse_gpu_telemetry(log, compute_backend)

    performance = re.search(
        r"\[PERF SUMMARY\] measured_steps=([0-9]+).*?"
        r"step_wall_s=([0-9.eE+-]+).*?"
        r"sim_s_per_wall_s=([0-9.eE+-]+).*?"
        r"measured_simulated_s=([0-9.eE+-]+)", log)
    require(performance is not None, "runtime log lacks performance summary")

    manifest = load_json(output / "nimbus_output.json")
    require(manifest.get("publication_state") == "ready" and
            "qualification_diagnostics" in manifest.get("data_products", []),
            "output contract is incomplete")
    expected_sample_count = duration_s // 30
    require(expected_sample_count > 0 and duration_s % 30 == 0,
            "reference duration is not aligned to the 30-second cadence")
    expected_times = [30.0 * (index + 1)
                      for index in range(expected_sample_count)]
    tracker_times = timed_publications(output / "storm_track",
                                       "track_step_*.json", None)
    diagnostic_times = timed_publications(
        output / "qualification_diagnostics", "step_*", "manifest.json")
    require(tracker_times == expected_times,
            "reference tracker cadence is incomplete")
    require(diagnostic_times == expected_times,
            "reference diagnostic cadence is incomplete")

    stationary.validate_history(validator, output)
    stationary.validate_history(validator, output / "qualification_diagnostics")

    checkpoints = sorted((output / "checkpoints").glob(
        "checkpoint_step_*.nimbus"))
    sidecars = [load_json(Path(str(path) + ".json")) for path in checkpoints]
    checkpoint_times = [float(item.get("simulation_time_s", math.nan))
                        for item in sidecars]
    expected_checkpoints: list[float] = []
    checkpoint_time = 450.0
    while checkpoint_time <= duration_s:
        expected_checkpoints.append(checkpoint_time)
        checkpoint_time += 450.0
    if not expected_checkpoints or expected_checkpoints[-1] != duration_s:
        expected_checkpoints.append(float(duration_s))
    require(checkpoint_times == expected_checkpoints,
            f"checkpoint cadence changed: {checkpoint_times}")
    for item in sidecars:
        require(item.get("boundary") == "completed_timestep" and
                item.get("portable_across_compute_memory_topology") is True,
                "reference checkpoint is not a portable completed boundary")

    return {
        "schema": f"nimbus.diagnostic.moving_nested_{compute_backend}_reference_runtime",
        "schema_version": 1,
        "status": "pass",
        "qualification_eligible": False,
        "scope": reference_scope(compute_backend, duration_s),
        "claims": {
            f"same_binary_as_moving_{compute_backend}_lifetime":
                complete_reference,
            f"fallback_off_physical_{compute_backend}": True,
            "native_smagorinsky_sgs_dispatch": True,
            "native_kessler_process_dispatch": True,
            "host_authoritative_kessler_state_commit": True,
            "strict_finite_bounded_outputs": True,
            "formal_comparison_complete": False,
        },
        "inputs": {
            "binary": relative(binary),
            "binary_sha256": binary_digest,
            "config": relative(config),
            "config_sha256": sha256(config),
            "duration_s": duration_s,
            **({
                "moving_lifetime_chain": relative(moving_chain),
                "moving_lifetime_chain_sha256": sha256(moving_chain),
            } if complete_reference else {
                "moving_lifetime_chain": None,
                "moving_lifetime_chain_sha256": None,
            }),
        },
        "runtime": {
            **({"process_wall_seconds": wall_seconds}
               if wall_seconds is not None else {}),
            "measured_steps": int(performance.group(1)),
            "solver_step_wall_seconds": float(performance.group(2)),
            "simulated_seconds_per_solver_wall_second": float(performance.group(3)),
            "measured_simulated_seconds": float(performance.group(4)),
            "tracker_samples": len(tracker_times),
            "diagnostic_samples": len(diagnostic_times),
            "gpu_telemetry": telemetry,
        },
        "artifacts": {
            "run_directory": relative(output),
            "runtime_log": relative(log_path),
            "runtime_log_sha256": sha256(log_path),
            "checkpoint_count": len(checkpoints),
            "checkpoint_times_s": checkpoint_times,
            "final_checkpoint": relative(checkpoints[-1]),
            "final_checkpoint_sha256": sha256(checkpoints[-1]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--validator", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument(
        "--compute-backend", choices=SUPPORTED_BACKENDS, default="vulkan")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--moving-chain", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=int, default=4500)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--allow-short-smoke", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()
    validator = args.validator.resolve()
    config = (args.config or default_config(args.compute_backend)).resolve()
    moving_chain = (args.moving_chain or default_chain(args.compute_backend)).resolve()
    output = (args.output or default_output(args.compute_backend)).resolve()
    require(args.duration == 4500 or
            (args.allow_short_smoke and args.duration > 0 and
             args.duration % 30 == 0),
            "formal reference runtime is fixed at 4500 seconds; bounded "
            "smokes require --allow-short-smoke and 30-second alignment")
    require(args.threads > 0, "thread count must be positive")
    required_paths = [binary, validator, config, CPU_REFERENCE_CONFIG]
    if args.duration == 4500:
        required_paths.append(moving_chain)
    for path in required_paths:
        require(path.is_file(), f"required input is missing: {path}")
    validate_config_delta(config, args.compute_backend)
    binary_digest = (
        validate_binary_identity(binary, moving_chain, args.compute_backend)
        if args.duration == 4500 else sha256(binary))

    wall_seconds: float | None = None
    if not args.inspect_only:
        require(not output.exists() or not any(output.iterdir()),
                f"output directory is not fresh: {output}")
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config, output / "qualification_input.yaml")
        (output / "qualification_binary.sha256").write_text(
            binary_digest + "\n", encoding="ascii")
        command = [
            str(binary), "--headless", "--config", str(config),
            "--duration", str(args.duration), "--write-every=450",
            "--checkpoint-every=450",
            f"--checkpoint-dir={output / 'checkpoints'}",
            "--threads", str(args.threads), "--guard-mode", "strict",
            f"--outdir={output}", "--log-profile", "normal", "--timing",
            f"--compute-backend={args.compute_backend}",
        ]
        environment = os.environ.copy()
        environment["NIMBUS_GPU_KERNEL_TIMING"] = "1"
        started = time.monotonic()
        with (output / "qualification_run.log").open(
            "w", encoding="utf-8", buffering=1) as log_stream:
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1)
            require(process.stdout is not None, "failed to capture runtime output")
            for line in process.stdout:
                log_stream.write(line)
                sys.stdout.write(line)
                sys.stdout.flush()
            return_code = process.wait()
        wall_seconds = time.monotonic() - started
        require(return_code == 0,
                f"{args.compute_backend.capitalize()} reference runtime exited "
                f"{return_code}")

    report = inspect(
        output, binary, validator, config, args.duration, wall_seconds,
        args.compute_backend, moving_chain)
    report_path = output / (
        f"moving_nested_{args.compute_backend}_reference_runtime_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": relative(report_path),
        "tracker_samples": report["runtime"]["tracker_samples"],
        "checkpoint_count": report["artifacts"]["checkpoint_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[moving-nested-accelerator-reference] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)

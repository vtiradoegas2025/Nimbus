#!/usr/bin/env python3
"""Prove live moving-nest physics across a physical accelerator window.

The default mode retains the exploratory internal diagnostic. ``--public-runtime``
uses the exact qualified public profile and production binary. A resumed Vulkan
branch crosses the known late-storm relocation; the bounded Metal smoke starts
fresh and may end before relocation. Both levels must retain fallback-off native
Smagorinsky and Kessler process ownership. Relocation remains host-owned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

import qualify_moving_nested_supercell as moving
import qualify_stationary_nested_vulkan_runtime as stationary


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim_internal_qualification"
DEFAULT_VALIDATOR = ROOT / "bin" / "field_validator"
DEFAULT_CONFIG = (
    ROOT / "tests" / "qualification" / "fixtures" /
    "moving_nested_supercell_frontier_vulkan_v1.yaml"
)
DEFAULT_METAL_CONFIG = (
    ROOT / "tests" / "qualification" / "fixtures" /
    "moving_nested_supercell_frontier_metal_v1.yaml"
)
DEFAULT_PUBLIC_CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification" /
    "moving_nested_supercell_vulkan_v1.yaml"
)
DEFAULT_PUBLIC_METAL_CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification" /
    "moving_nested_supercell_metal_v1.yaml"
)
DEFAULT_CHECKPOINT = (
    ROOT / "tests" / "artifacts" /
    "moving_nested_reference_frontier_v12_20260818" / "moving_child" /
    "checkpoints" / "checkpoint_step_000002700.nimbus"
)
DEFAULT_OUTPUT = (
    ROOT / "tests" / "artifacts" /
    "moving_nested_vulkan_live_storm_v1_20260819"
)
DEFAULT_METAL_OUTPUT = (
    ROOT / "tests" / "artifacts" /
    "moving_nested_metal_live_storm_v1"
)
REPORT_SCHEMA = "nimbus.diagnostic.moving_nested_vulkan_live_storm"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_device_log(output: str, backend: str) -> None:
    required = (
        f"[COMPUTE] backend={backend}, requested={backend}, "
        "device_index=-1, fallback=off"
    )
    require(required in output,
            f"run did not prove fallback-off physical {backend} execution")


def complete_cadence_window(duration_s: float, resume_time_s: float) -> bool:
    return duration_s - resume_time_s >= 450.0


def acceptable_final_validity(status: object, require_relocation: bool,
                              duration_s: float, resume_time_s: float) -> bool:
    if require_relocation and complete_cadence_window(duration_s, resume_time_s):
        return status == "valid"
    return status != "invalid"


def sha256(path: Path) -> str:
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


def require_public_binary_identity(binary: Path, public_source_binary: Path) -> None:
    require(
        sha256(binary) == sha256(public_source_binary),
        "--public-runtime requires bytes identical to the production simulator",
    )


def capture_level_counters(log: str) -> dict[str, dict[str, int]]:
    smag = re.search(
        r"\[NESTING DEVICE LEVELS\] "
        r"parent_coefficients=([0-9]+), parent_tendencies=([0-9]+), "
        r"parent_sgs_apply=([0-9]+), child_coefficients=([0-9]+), "
        r"child_tendencies=([0-9]+), child_sgs_apply=([0-9]+)",
        log,
    )
    kessler = re.search(
        r"\[NESTING DEVICE MICROPHYSICS LEVELS\] "
        r"parent_pointwise=([0-9]+), parent_sedimentation=([0-9]+), "
        r"parent_apply=([0-9]+), child_pointwise=([0-9]+), "
        r"child_sedimentation=([0-9]+), child_apply=([0-9]+)",
        log,
    )
    require(smag is not None and kessler is not None,
            "runtime log lacks per-level native physics counters")
    return {
        "smagorinsky": dict(zip(
            ("parent_coefficients", "parent_tendencies", "parent_sgs_apply",
             "child_coefficients", "child_tendencies", "child_sgs_apply"),
            (int(value) for value in smag.groups()),
        )),
        "kessler": dict(zip(
            ("parent_pointwise", "parent_sedimentation", "parent_apply",
             "child_pointwise", "child_sedimentation", "child_apply"),
            (int(value) for value in kessler.groups()),
        )),
    }


def require_moving_kessler_log(log: str, require_water_activity: bool = True) -> None:
    aggregate = re.search(
        r"\[NESTING DEVICE MICROPHYSICS\] "
        r"kessler_pointwise_dispatches=([0-9]+), "
        r"kessler_sedimentation_dispatches=([0-9]+), "
        r"microphysics_apply_dispatches=([0-9]+)",
        log,
    )
    require(
        aggregate is not None and int(aggregate.group(1)) > 0 and
        int(aggregate.group(2)) > 0 and int(aggregate.group(3)) == 0,
        "moving branch lacks native Kessler process work or performed a "
        "device state commit",
    )
    counters = capture_level_counters(log)["kessler"]
    require(
        all(counters[name] > 0 for name in (
            "parent_pointwise", "parent_sedimentation",
            "child_pointwise", "child_sedimentation")) and
        counters["parent_apply"] == 0 and counters["child_apply"] == 0,
        "moving branch lacks parent/child Kessler process work or changed "
        "the host-authoritative state commit",
    )
    rows = [line for line in log.splitlines()
            if "[NESTING MICROPHYSICS] " in line]
    require(len(rows) == 1, "moving branch lacks one Kessler science row")
    values = {
        key: float(value)
        for key, value in re.findall(
            r"([a-z_]+)=([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
            r"(?:[eE][+-]?[0-9]+)?)",
            rows[0],
        )
    }
    required = {
        f"{level}_{name}"
        for level in ("parent", "child")
        for name in (
            "active", "relative_imbalance", "baseline_water_kg",
            "water_activity_kg", "material_water_repair_kg",
            "theta_repair_j", "theta_repair_count",
            "moist_energy_complete", "moist_energy_residual_j",
            "energy_activity_j",
        )
    }
    require(required <= values.keys() and
            all(math.isfinite(values[name]) for name in required),
            "moving Kessler science row is incomplete or non-finite")
    for level in ("parent", "child"):
        baseline = values[f"{level}_baseline_water_kg"]
        activity = values[f"{level}_water_activity_kg"]
        energy_activity = values[f"{level}_energy_activity_j"]
        require(
            values[f"{level}_active"] == 1 and baseline > 0.0 and
            (activity > 0.0 or not require_water_activity) and
            energy_activity > 0.0 and
            abs(values[f"{level}_relative_imbalance"]) <= 1.0e-8 and
            values[f"{level}_material_water_repair_kg"] <=
                1.0e-12 * baseline and
            values[f"{level}_theta_repair_j"] == 0.0 and
            values[f"{level}_theta_repair_count"] == 0.0 and
            values[f"{level}_moist_energy_complete"] == 1.0 and
            abs(values[f"{level}_moist_energy_residual_j"]) <=
                1.0e-12 * energy_activity,
            f"moving Kessler {level} conservation or roundoff bound failed",
        )


def inspect(output: Path, binary: Path, validator: Path, config: Path,
            checkpoint: Path | None, duration_s: int,
            wall_seconds: float | None,
            require_relocation: bool = True,
            compute_backend: str = "vulkan",
            public_runtime: bool = False) -> dict:
    log_path = output / "qualification_run.log"
    require(log_path.is_file(), "runtime log is missing")
    log = log_path.read_text(encoding="utf-8")
    require("[ERROR]" not in log, "runtime log contains an error")
    require_device_log(log, compute_backend)
    label = f"moving {compute_backend} branch"
    stationary.require_hierarchy_memory_plan(log, label)
    stationary.require_smagorinsky_log(log, label)
    require_moving_kessler_log(log, require_relocation)
    counters = capture_level_counters(log)
    performance = re.search(
        r"\[PERF SUMMARY\] measured_steps=([0-9]+).*?"
        r"step_wall_s=([0-9.eE+-]+).*?"
        r"sim_s_per_wall_s=([0-9.eE+-]+).*?"
        r"measured_simulated_s=([0-9.eE+-]+)",
        log,
    )
    require(performance is not None, "runtime log lacks its performance summary")

    source_sidecar: dict | None = None
    if checkpoint is None:
        resume_time_s = 0.0
        resumed_steps = 0
        require(not (output / "nimbus_branch.json").exists(),
                "fresh Vulkan run unexpectedly published resume provenance")
    else:
        source_sidecar = load_json(Path(str(checkpoint) + ".json"))
        resume_time_s = float(source_sidecar.get("simulation_time_s", math.nan))
        resumed_steps = int(source_sidecar.get("completed_steps", -1))
        require(source_sidecar.get("portable_across_compute_memory_topology") is True,
                "source checkpoint is not compute-topology portable")
        require(math.isfinite(resume_time_s) and 0.0 < resume_time_s < duration_s,
                "source checkpoint time is outside the diagnostic window")
        resume_marker = (
            f"[CHECKPOINT] resumed step={resumed_steps}, time_s={resume_time_s:g}"
        )
        require(resume_marker in log, "runtime did not publish the expected resume")
        branch = load_json(output / "nimbus_branch.json")
        require(branch.get("resumed_step") == resumed_steps and
                float(branch.get("resumed_simulation_time_s", math.nan)) == resume_time_s,
                "branch provenance disagrees with the portable source checkpoint")
    contract = load_json(output / "nimbus_output.json")
    require(contract.get("publication_state") == "ready",
            "output contract is not ready")
    require("nested_child_qualification_diagnostics" in
            contract.get("data_products", []),
            "output contract lacks child qualification diagnostics")
    require(not (output / "moving_nest_scientific_validity.json").exists(),
            "runtime published moving-nest scientific invalidity")

    stationary.validate_history(validator, output)
    child_history = output / "nests" / "child_0000"
    if any(child_history.glob("step_*")):
        stationary.validate_history(validator, child_history)
    else:
        require(not complete_cadence_window(duration_s, resume_time_s),
                "child history is missing after the first output cadence")
    stationary.validate_history(
        validator,
        output / "nests" / "child_0000" / "qualification_diagnostics",
    )

    tracker_paths = sorted((output / "storm_track").glob("track_step_*.json"))
    expected_samples = int((duration_s - resume_time_s) // 30)
    require(len(tracker_paths) == expected_samples,
            f"expected {expected_samples} post-resume tracker samples, "
            f"found {len(tracker_paths)}")
    records = [load_json(path) for path in tracker_paths]
    expected_times = [resume_time_s + 30.0 * (index + 1)
                      for index in range(expected_samples)]
    actual_times = [float(record.get("simulation_time_s", math.nan))
                    for record in records]
    require(actual_times == expected_times,
            "post-resume tracker cadence is incomplete or non-causal")

    commits: list[dict] = []
    reasons: dict[str, int] = {}
    for record in records:
        require(record.get("schema_version") == 12,
                "tracker schema changed in the Vulkan branch")
        require(record.get("tracking_source") == "synchronized_nested_child",
                "tracker lost synchronized child ownership")
        authority = record.get("movement_authority", {})
        moving.require_authority_contract(authority)
        moving.require_runtime_scientific_validity(record)
        moving.require_precipitation_relocation(
            authority, float(record["simulation_time_s"]))
        reason = str(authority.get("reason", "missing"))
        reasons[reason] = reasons.get(reason, 0) + 1
        if authority.get("commit_performed") is True:
            shift = authority.get("committed_shift_cells", {})
            require(type(shift.get("i")) is int and type(shift.get("j")) is int and
                    (shift["i"] != 0 or shift["j"] != 0),
                f"{compute_backend} relocation lacks a non-zero integer shift")
            require(authority.get("prognostic_state_mutation") is True,
                    "relocation denied its host-owned prognostic mutation")
            commits.append({
                "simulation_time_s": record["simulation_time_s"],
                "completed_step": record["completed_step"],
                "shift_cells": shift,
                "cumulative_shift_cells":
                    authority.get("cumulative_committed_shift_cells"),
                "reason": reason,
            })
    if require_relocation:
        require(commits,
                f"bounded {compute_backend} window did not cross a live relocation")
    final_validity = records[-1].get("moving_nest_scientific_validity", {}).get(
        "status")
    require(acceptable_final_validity(
                final_validity, require_relocation, duration_s, resume_time_s),
            f"final {compute_backend} tracker record has the wrong scientific validity")

    checkpoints = sorted((output / "checkpoints").glob(
        "checkpoint_step_*.nimbus"))
    checkpoint_sidecars = [
        load_json(Path(str(path) + ".json")) for path in checkpoints
    ]
    expected_checkpoint_times: list[float] = []
    candidate_time = (math.floor(resume_time_s / 450.0) + 1.0) * 450.0
    while candidate_time <= duration_s:
        expected_checkpoint_times.append(candidate_time)
        candidate_time += 450.0
    if not expected_checkpoint_times or expected_checkpoint_times[-1] != duration_s:
        expected_checkpoint_times.append(float(duration_s))
    actual_checkpoint_times = [
        float(sidecar.get("simulation_time_s", math.nan))
        for sidecar in checkpoint_sidecars
    ]
    require(actual_checkpoint_times == expected_checkpoint_times,
            "checkpoint cadence changed: expected "
            f"{expected_checkpoint_times}, got {actual_checkpoint_times}")
    for sidecar in checkpoint_sidecars:
        require(sidecar.get("boundary") == "completed_timestep" and
                sidecar.get("portable_across_compute_memory_topology") is True,
                f"{compute_backend} checkpoint is not a portable completed boundary")
        if source_sidecar is not None:
            require(sidecar.get("parent_run_id") == source_sidecar.get("run_id") and
                    Path(str(sidecar.get("parent_checkpoint"))).resolve() ==
                        checkpoint.resolve(),
                    "continued Vulkan checkpoint lineage changed")
    final_sidecar = checkpoint_sidecars[-1]
    require(float(final_sidecar.get("simulation_time_s", math.nan)) == duration_s,
            "final checkpoint did not seal the requested endpoint")

    report = {
        "schema": (REPORT_SCHEMA if compute_backend == "vulkan" else
                   "nimbus.diagnostic.moving_nested_metal_live_storm"),
        "schema_version": 1,
        "status": "pass",
        "qualification_eligible": public_runtime,
        "scope": (
            f"qualified_public_fresh_start_physical_{compute_backend}"
            if public_runtime and checkpoint is None else
            f"qualified_public_resume_physical_{compute_backend}"
            if public_runtime else
            f"exploratory_fresh_start_physical_{compute_backend}"
            if checkpoint is None else
            f"exploratory_cross_backend_resume_physical_{compute_backend}"
        ),
        "claims": {
            "public_moving_nest_admitted": public_runtime,
            "device_resident_relocation": False,
            f"fallback_off_physical_{compute_backend}": True,
            "native_parent_child_smagorinsky": True,
            "native_parent_child_kessler_processes": True,
            "host_authoritative_kessler_state_commit": True,
            "host_owned_live_relocation_crossed": bool(commits),
            "strict_finite_bounded_outputs": True,
        },
        "inputs": {
            "binary": relative(binary),
            "binary_sha256": sha256(binary),
            "config": relative(config),
            "config_sha256": sha256(config),
            **({
                "source_checkpoint": relative(checkpoint),
                "source_checkpoint_sha256": sha256(checkpoint),
                "source_checkpoint_time_s": resume_time_s,
            } if checkpoint is not None else {
                "initialization": "fresh_configured_state",
                "source_checkpoint_time_s": 0.0,
            }),
            "duration_s": duration_s,
        },
        "runtime": {
            **({"process_wall_seconds": wall_seconds}
               if wall_seconds is not None else {}),
            "measured_steps": int(performance.group(1)),
            "solver_step_wall_seconds": float(performance.group(2)),
            "simulated_seconds_per_solver_wall_second":
                float(performance.group(3)),
            "measured_simulated_seconds": float(performance.group(4)),
            "simulated_window_s": duration_s - resume_time_s,
            "tracker_samples": len(records),
            "first_tracker_time_s": actual_times[0],
            "final_tracker_time_s": actual_times[-1],
            "relocations": commits,
            "movement_reasons": dict(sorted(reasons.items())),
            "device_level_counters": counters,
        },
        "artifacts": {
            "run_directory": relative(output),
            "runtime_log": relative(log_path),
            "runtime_log_sha256": sha256(log_path),
            **({"branch_provenance": relative(output / "nimbus_branch.json")}
               if checkpoint is not None else {}),
            "checkpoint_count": len(checkpoints),
            "checkpoint_times_s": actual_checkpoint_times,
            "final_checkpoint_sidecar": relative(
                Path(str(checkpoints[-1]) + ".json")),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--validator", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compute-backend", choices=("vulkan", "metal"), default="vulkan")
    parser.add_argument("--duration", type=int, default=4500)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--public-runtime", action="store_true")
    parser.add_argument(
        "--public-source-binary", type=Path,
        default=ROOT / "bin" / "nimbus_sim",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-no-relocation", action="store_true",
        help="permit a short startup smoke that ends before the first move",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    validator = args.validator.resolve()
    config = (args.config or
              (DEFAULT_PUBLIC_CONFIG if args.public_runtime and
               args.compute_backend == "vulkan" else
               DEFAULT_PUBLIC_METAL_CONFIG if args.public_runtime else
               DEFAULT_CONFIG if args.compute_backend == "vulkan" else
               DEFAULT_METAL_CONFIG)).resolve()
    checkpoint = None if args.fresh_start else args.checkpoint.resolve()
    output = (args.output if args.output is not None else
              (DEFAULT_OUTPUT if args.compute_backend == "vulkan" else
               DEFAULT_METAL_OUTPUT)).resolve()
    required_paths = [binary, validator, config]
    if checkpoint is not None:
        required_paths += [checkpoint, Path(str(checkpoint) + ".json")]
    public_source_binary = args.public_source_binary.resolve()
    if args.public_runtime:
        required_paths.append(public_source_binary)
    for path in required_paths:
        require(path.is_file(), f"required input is missing: {path}")
    if args.public_runtime:
        require_public_binary_identity(binary, public_source_binary)
    require(args.duration > 0 and args.threads > 0,
            "duration and thread count must be positive")

    wall_seconds: float | None = None
    if not args.inspect_only:
        require(not output.exists() or not any(output.iterdir()),
                f"output directory is not fresh: {output}")
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config, output / "qualification_input.yaml")
        (output / "qualification_binary.sha256").write_text(
            sha256(binary) + "\n", encoding="ascii")
        command = [
            str(binary), "--headless",
            "--config", str(config), "--duration", str(args.duration),
            "--write-every=450", "--checkpoint-every=450",
            f"--checkpoint-dir={output / 'checkpoints'}",
            "--threads", str(args.threads), "--guard-mode", "strict",
            f"--outdir={output}", "--log-profile", "normal", "--timing",
            f"--compute-backend={args.compute_backend}",
        ]
        if not args.public_runtime:
            command.insert(1, "--qualification-moving-nest")
        if checkpoint is not None:
            command.append(f"--resume={checkpoint}")
        started = time.monotonic()
        result = subprocess.run(
            command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        wall_seconds = time.monotonic() - started
        (output / "qualification_run.log").write_text(
            result.stdout, encoding="utf-8")
        require(result.returncode == 0,
                f"moving {args.compute_backend} branch exited {result.returncode}:\n"
                f"{result.stdout[-16000:]}")

    report = inspect(
        output, binary, validator, config, checkpoint,
        args.duration, wall_seconds, not args.allow_no_relocation,
        args.compute_backend, args.public_runtime)
    report_path = output / f"moving_nested_{args.compute_backend}_runtime_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if args.report is not None:
        external_report = args.report.resolve()
        external_report.parent.mkdir(parents=True, exist_ok=True)
        external_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[moving-nested-accelerator-runtime] FAIL: {exc}")
        raise SystemExit(1)

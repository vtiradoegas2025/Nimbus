#!/usr/bin/env python3
"""Qualify one live flat Cartesian C-grid D5 recycling transaction.

The gate verifies committed transaction provenance, durable retiring-swath
publication, continued integration and restart after commit, and Earth-frame
trajectory continuity through the grid-coordinate reset for the selected
configuration and compute backend.
"""

from __future__ import annotations

import argparse
import array
import ast
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
DEFAULT_CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "test" /
    "domain_recycling_smoke.yaml"
)
EXPECTED_FIELDS = {"p", "qr", "qv", "rho", "theta", "u", "v", "w"}
EXPECTED_SHIFT = (4, 0)
EXPECTED_POST_ORIGIN_T0_M = (4000.0, 0.0)
# CPU and accelerator backends are scientifically equivalent rather than
# bitwise-identical time integrators. The threshold-crossing sample may
# therefore differ while the qualified two-sample request, exact shift, and
# transaction remain the same. The checkpoint branch requires the commit no
# later than 24 s.
EARLIEST_COMMIT_TIME_S = 2.0
LATEST_COMMIT_TIME_S = 24.0
# The fixture's meteorological motion remains below 25 m/s.  This deliberately
# broad physical ceiling is still two orders of magnitude below the 4-km /
# 0.5-s discontinuity that an uncorrected grid-coordinate reset would create.
MAXIMUM_TRAJECTORY_HORIZONTAL_SPEED_MS = 100.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def runtime_environment(compute_backend: str) -> dict[str, str]:
    environment = os.environ.copy()
    if compute_backend != "vulkan" or platform.system() != "Darwin":
        return environment
    icd = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    if "VK_ICD_FILENAMES" not in environment and icd.is_file():
        environment["VK_ICD_FILENAMES"] = str(icd)
    library_dirs = [
        Path("/opt/homebrew/opt/vulkan-loader/lib"),
        Path("/opt/homebrew/opt/molten-vk/lib"),
    ]
    discovered = [str(path) for path in library_dirs if path.is_dir()]
    inherited = environment.get("DYLD_LIBRARY_PATH")
    if inherited:
        discovered.append(inherited)
    if discovered:
        environment["DYLD_LIBRARY_PATH"] = ":".join(discovered)
    return environment


def require_selected_backend(
    output: str,
    compute_backend: str,
    required_backend: str | None,
    context: str,
) -> None:
    if required_backend is None:
        return
    require(
        compute_backend == required_backend,
        f"--require-{required_backend} requires "
        f"--compute-backend={required_backend}",
    )
    if required_backend == "vulkan":
        require("[COMPUTE][VULKAN] selected device[" in output,
                f"{context} did not select a physical Vulkan device\n{output[-6000:]}")
    elif required_backend == "metal":
        require("[HARDWARE] GPU: " in output and
                "[HARDWARE] GPU: not available" not in output,
                f"{context} did not identify a physical Metal device\n{output[-6000:]}")
    require(f"[COMPUTE] backend={required_backend}" in output,
            f"{context} did not retain {required_backend} authority\n{output[-6000:]}")
    require("Falling back to CPU" not in output and "fallback to CPU" not in output,
            f"{context} used CPU fallback\n{output[-6000:]}")


def require_native_chaos_dispatch(output: str, context: str) -> int:
    matches = re.findall(
        r"chaos/earth_control_evolution\.comp:([0-9]+)", output)
    require(matches, f"{context} did not report native Earth-control evolution\n{output[-6000:]}")
    dispatches = int(matches[-1])
    require(dispatches > 0,
            f"{context} reported zero native Earth-control dispatches\n{output[-6000:]}")
    return dispatches


def load_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON evidence is not an object: {path}")
    return value


def require_swath_brick_hierarchy(
    hierarchy: dict, expected_field: str, context: str
) -> None:
    require(
        hierarchy.get("schema") == "nimbus.field-brick-hierarchy"
        and hierarchy.get("schemaVersion") == 2,
        f"wrong {context} brick hierarchy schema",
    )
    require(hierarchy.get("field") == expected_field,
            f"wrong {context} brick field")
    require(hierarchy.get("sourceDtype") == "float32",
            f"wrong {context} brick dtype")
    require(hierarchy.get("sampleLocation") == "cell_center",
            f"wrong {context} brick sample location")
    require(hierarchy.get("cullingSafe") is True,
            f"unsafe {context} brick extrema")
    generation = hierarchy.get("publicationGeneration")
    require(isinstance(generation, str) and len(generation) == 32,
            f"invalid {context} brick generation")
    shape = hierarchy.get("sourceShape")
    offset = hierarchy.get("sourceIndexOffset")
    require(
        isinstance(shape, list) and len(shape) == 3
        and all(isinstance(value, int) and value > 0 for value in shape),
        f"invalid {context} brick source shape",
    )
    require(
        isinstance(offset, list) and len(offset) == 3
        and all(isinstance(value, int) and value >= 0 for value in offset),
        f"invalid {context} brick source offset",
    )
    levels = hierarchy.get("levels")
    require(isinstance(levels, list) and len(levels) == 1,
            f"unexpected {context} brick levels")
    level = levels[0]
    require(
        isinstance(level, dict)
        and level.get("level") == 0
        and level.get("stride") == [1, 1, 1]
        and level.get("shape") == shape
        and level.get("source") == {"kind": "primary_field"},
        f"invalid {context} primary brick level",
    )
    bricks = level.get("bricks")
    require(isinstance(bricks, list) and bricks,
            f"missing {context} primary bricks")
    for brick in bricks:
        require(isinstance(brick, dict), f"invalid {context} brick record")
        content_sha = brick.get("contentSha256")
        scalar_range = brick.get("scalarRange")
        physical_bounds = brick.get("physicalBoundsM")
        require(isinstance(content_sha, str) and len(content_sha) == 64,
                f"invalid {context} brick content identity")
        require(
            isinstance(scalar_range, dict)
            and isinstance(scalar_range.get("finiteCount"), int)
            and scalar_range["finiteCount"] > 0
            and scalar_range.get("nonFiniteCount") == 0,
            f"invalid {context} brick scalar range",
        )
        require(
            isinstance(physical_bounds, dict)
            and isinstance(physical_bounds.get("lower"), list)
            and len(physical_bounds["lower"]) == 3
            and isinstance(physical_bounds.get("upper"), list)
            and len(physical_bounds["upper"]) == 3,
            f"invalid {context} brick physical bounds",
        )


def pair(value: object, first: str, second: str, context: str) -> tuple[float, float]:
    require(isinstance(value, dict), f"{context} must be an object")
    require(first in value and second in value, f"{context} is incomplete")
    return float(value[first]), float(value[second])


def fnv1a64(path: Path) -> str:
    value = 0xCBF29CE484222325
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            for byte in chunk:
                value ^= byte
                value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def load_trajectory_npy(path: Path) -> tuple[tuple[int, int, int], array.array]:
    require(path.is_file(), f"missing trajectory evidence: {path}")
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", f"invalid NPY magic: {path}")
        version = stream.read(2)
        require(len(version) == 2 and version[0] in {1, 2, 3}, f"unsupported NPY version: {path}")
        length_size = 2 if version[0] == 1 else 4
        raw_length = stream.read(length_size)
        require(len(raw_length) == length_size, f"truncated NPY header length: {path}")
        header_length = struct.unpack("<H" if length_size == 2 else "<I", raw_length)[0]
        header = ast.literal_eval(stream.read(header_length).decode("latin1").strip())
        require(header.get("descr") in {"<f4", "=f4", "|f4"}, f"unsupported NPY dtype: {path}")
        require(header.get("fortran_order") is False, f"Fortran-order NPY is unsupported: {path}")
        shape = tuple(int(item) for item in header.get("shape", ()))
        require(len(shape) == 3 and shape[2] == 9, f"wrong trajectory shape: {shape}")
        payload = stream.read()
    require(len(payload) == math.prod(shape) * 4, f"wrong NPY payload length: {path}")
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little" and header.get("descr") == "<f4":
        values.byteswap()
    return (shape[0], shape[1], shape[2]), values


def qualify_trajectories(path: Path, commit_time_s: float) -> dict[str, float | int]:
    (parcel_count, sample_capacity, column_count), values = load_trajectory_npy(path)
    require(parcel_count > 0 and sample_capacity > 0, "trajectory output is empty")
    surviving_crossings = 0
    total_samples = 0
    maximum_crossing_speed = 0.0
    maximum_all_step_speed = 0.0

    for parcel in range(parcel_count):
        rows: list[array.array] = []
        padding_started = False
        for sample in range(sample_capacity):
            base = (parcel * sample_capacity + sample) * column_count
            row = values[base : base + column_count]
            if math.isnan(row[0]):
                padding_started = True
                require(all(math.isnan(value) for value in row), "partial trajectory padding row")
                continue
            require(not padding_started, "finite trajectory sample follows NaN padding")
            require(all(math.isfinite(value) for value in row), "non-finite populated trajectory sample")
            rows.append(row)
        total_samples += len(rows)

        for previous, current in zip(rows, rows[1:]):
            dt = float(current[0] - previous[0])
            require(dt > 0.0, "trajectory time is not strictly increasing")
            speed = math.hypot(current[1] - previous[1], current[2] - previous[2]) / dt
            maximum_all_step_speed = max(maximum_all_step_speed, speed)
            if previous[0] <= commit_time_s < current[0]:
                surviving_crossings += 1
                maximum_crossing_speed = max(maximum_crossing_speed, speed)

    require(total_samples > parcel_count, "trajectory paths did not advance")
    require(surviving_crossings > 0, "no parcel survived across the recycling transaction")
    require(
        maximum_crossing_speed <= MAXIMUM_TRAJECTORY_HORIZONTAL_SPEED_MS,
        "Earth-frame trajectory discontinuity at recycling: "
        f"{maximum_crossing_speed:.6g} m/s exceeds "
        f"{MAXIMUM_TRAJECTORY_HORIZONTAL_SPEED_MS:.6g} m/s",
    )
    return {
        "parcel_count": parcel_count,
        "total_samples": total_samples,
        "surviving_crossings": surviving_crossings,
        "maximum_crossing_speed_ms": maximum_crossing_speed,
        "maximum_all_step_speed_ms": maximum_all_step_speed,
    }


def qualify_swath(output: Path, transaction_id: str, commit_time_s: float) -> dict[str, int]:
    event_directory = output / "domain_recycling_swath" / f"transaction_{transaction_id}"
    event = load_json(event_directory / "event.json")
    require(event.get("schema") == "nimbus.domain_recycling_swath", "wrong swath schema")
    require(event.get("schema_version") == 1, "wrong swath schema version")
    require(event.get("publication_phase") == "captured_precommit", "wrong swath capture phase")
    require(event.get("transaction_id") == transaction_id, "swath transaction linkage mismatch")
    require(float(event.get("simulation_time_s", -1.0)) == commit_time_s,
            "swath simulation time does not match its committed transaction")
    require(tuple(event.get("shift_cells", ())) == EXPECTED_SHIFT, "wrong swath shift")

    artifacts = event.get("artifacts")
    expected_paths = {"tile_00/manifest.json"}
    expected_paths.update(
        f"tile_00/{field}{suffix}"
        for field in EXPECTED_FIELDS
        for suffix in (".npy", ".bricks.json")
    )
    require(isinstance(artifacts, list), "swath artifacts are not a list")
    actual_paths = {
        artifact.get("path") for artifact in artifacts
        if isinstance(artifact, dict)
    }
    require(actual_paths == expected_paths, "wrong swath artifact set")
    field_names: set[str] = set()
    brick_fields: set[str] = set()
    total_bytes = 0
    for artifact in artifacts:
        require(isinstance(artifact, dict), "swath artifact entry is not an object")
        relative = artifact.get("path")
        require(isinstance(relative, str) and relative.startswith("tile_00/"), "invalid swath artifact path")
        artifact_path = event_directory / relative
        require(artifact_path.is_file(), f"missing swath artifact: {relative}")
        expected_bytes = int(artifact.get("bytes", -1))
        require(artifact_path.stat().st_size == expected_bytes, f"swath artifact size mismatch: {relative}")
        require(fnv1a64(artifact_path) == artifact.get("fnv1a64"), f"swath artifact hash mismatch: {relative}")
        total_bytes += expected_bytes
        if artifact_path.suffix == ".npy":
            field_names.add(artifact_path.stem)
        elif artifact_path.name.endswith(".bricks.json"):
            hierarchy = load_json(artifact_path)
            field = artifact_path.name.removesuffix(".bricks.json")
            require(field in EXPECTED_FIELDS, f"wrong swath brick field: {field}")
            require_swath_brick_hierarchy(hierarchy, field, relative)
            brick_fields.add(field)
    require(field_names == EXPECTED_FIELDS, f"wrong swath fields: {sorted(field_names)}")
    require(
        brick_fields == EXPECTED_FIELDS,
        f"wrong swath brick fields: {sorted(brick_fields)}",
    )
    return {"artifact_count": len(artifacts), "artifact_bytes": total_bytes}


def qualify_tracking(output: Path) -> tuple[str, dict[str, float | int]]:
    paths = sorted((output / "storm_track").glob("track_step_*.json"))
    require(paths, "no storm-tracking records were published")
    records = [load_json(path) for path in paths]
    require(all(record.get("schema") == "nimbus.storm_feature" for record in records), "wrong tracker schema")
    require(all(record.get("schema_version") == 12 for record in records), "wrong tracker schema version")
    require(all(record.get("tracking_source") == "root_grid" for record in records), "wrong tracker source")
    require(all(record.get("movement_authority", {}).get("recycling_enabled") is True for record in records),
            "live recycling authority is missing")
    require(all(record.get("movement_authority", {}).get("nested_movement_enabled") is False for record in records),
            "flat recycling records claimed nested movement authority")

    committed = [record for record in records if record.get("state_mutation") is True]
    require(len(committed) == 1, f"expected exactly one committed transaction, found {len(committed)}")
    record = committed[0]
    commit_time_s = float(record.get("simulation_time_s", -1.0))
    require(EARLIEST_COMMIT_TIME_S <= commit_time_s <= LATEST_COMMIT_TIME_S,
            f"commit time {commit_time_s} s is outside the qualified branch window")
    authority = record.get("movement_authority", {})
    require(authority.get("commit_performed") is True, "commit provenance is missing")
    transaction_id = authority.get("transaction_id")
    require(isinstance(transaction_id, str) and transaction_id.startswith("d5_"), "invalid transaction id")
    require(pair(authority.get("committed_shift_cells"), "i", "j", "committed shift") == EXPECTED_SHIFT,
            "wrong committed shift")
    require(pair(authority.get("post_shift_grid_origin_t0_earth_m"), "x", "y", "post-shift origin")
            == EXPECTED_POST_ORIGIN_T0_M, "wrong post-shift origin")

    final = records[-1]
    final_frame = final.get("horizontal_frame", {})
    require(pair(final_frame.get("grid_origin_t0_earth_m"), "x", "y", "final frame origin")
            == EXPECTED_POST_ORIGIN_T0_M, "integration did not retain the committed frame")
    final_authority = final.get("movement_authority", {})
    require(pair(final_authority.get("cumulative_committed_shift_cells"), "i", "j", "cumulative shift")
            == EXPECTED_SHIFT, "final cumulative shift is wrong")
    require(float(final.get("simulation_time_s", -1.0)) >= 30.0, "run did not continue after commit")
    return transaction_id, {
        "tracking_records": len(records),
        "commit_step": int(record["completed_step"]),
        "commit_time_s": commit_time_s,
    }


def run_direct(
    binary: Path,
    config: Path,
    output: Path,
    compute_backend: str,
    required_backend: str | None,
    threads: int,
    require_native_chaos: bool,
) -> None:
    command = [
        str(binary),
        "--headless",
        f"--config={config}",
        f"--outdir={output}",
        f"--compute-backend={compute_backend}",
        f"--threads={threads}",
        "--duration=30",
        "--write-every=0",
        f"--log-profile={'normal' if required_backend else 'quiet'}",
        "--no-timing",
        "--trajectories",
        "--trajectory-seed-time=0",
        "--trajectory-count=3",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=runtime_environment(compute_backend),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(completed.returncode == 0, f"live D5 run exited {completed.returncode}\n{completed.stdout[-12000:]}")
    require_selected_backend(completed.stdout, compute_backend, required_backend, "live D5 run")
    if require_native_chaos:
        require_native_chaos_dispatch(completed.stdout, "live D5 run")


def run_command(
    command: list[str],
    context: str,
    compute_backend: str,
    required_backend: str | None,
    require_native_chaos: bool = False,
) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=runtime_environment(compute_backend),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(completed.returncode == 0, f"{context} exited {completed.returncode}\n{completed.stdout[-12000:]}")
    require_selected_backend(completed.stdout, compute_backend, required_backend, context)
    if require_native_chaos:
        require_native_chaos_dispatch(completed.stdout, context)


def qualify_restart(
    binary: Path,
    config: Path,
    root: Path,
    compute_backend: str,
    required_backend: str | None,
    threads: int,
    require_native_chaos: bool,
) -> dict[str, int]:
    first_output = root / "first_output"
    checkpoint_directory = root / "checkpoints"
    common = [
        str(binary),
        "--headless",
        f"--config={config}",
        f"--compute-backend={compute_backend}",
        f"--threads={threads}",
        "--write-every=0",
        f"--log-profile={'normal' if required_backend else 'quiet'}",
        "--no-timing",
    ]
    run_command(
        common
        + [
            "--duration=24",
            "--checkpoint-every=24",
            f"--checkpoint-dir={checkpoint_directory}",
            f"--outdir={first_output}",
        ],
        "pre-restart live D5 run", compute_backend, required_backend,
        require_native_chaos,
    )
    checkpoints = sorted(checkpoint_directory.glob("checkpoint_step_*.nimbus"))
    require(len(checkpoints) == 1, f"expected one D5 checkpoint, found {len(checkpoints)}")
    sidecar = load_json(Path(f"{checkpoints[0]}.json"))
    require(sidecar.get("completed_steps") == 48, "wrong D5 checkpoint step")
    require(float(sidecar.get("simulation_time_s", -1.0)) == 24.0, "wrong D5 checkpoint time")
    provenance = sidecar.get("domain_recycling")
    require(isinstance(provenance, dict), "checkpoint is missing D5 provenance")
    require(provenance.get("control_volume_exchange_complete") is True,
            "checkpoint D5 exchange is incomplete")
    require(provenance.get("control_volume_exchange_basis")
            == "instantaneous_fixed_earth_control_volume_exchange_v1",
            "checkpoint D5 exchange basis is wrong")

    first_record_paths = sorted((first_output / "storm_track").glob("track_step_*.json"))
    preserved_record_bytes = {
        path.name: path.read_bytes() for path in first_record_paths
    }
    first_records = [load_json(path) for path in first_record_paths]
    first_commits = [record for record in first_records if record.get("state_mutation") is True]
    require(len(first_commits) == 1, "checkpoint branch did not commit exactly one D5 transaction")
    transaction_id = first_commits[0].get("movement_authority", {}).get("transaction_id")
    require(transaction_id == provenance.get("transaction_id"), "checkpoint transaction id mismatch")
    commit_time_s = float(provenance.get("transaction_simulation_time_s", -1.0))
    require(EARLIEST_COMMIT_TIME_S <= commit_time_s <= LATEST_COMMIT_TIME_S,
            "checkpoint transaction time is outside the qualified branch window")
    qualify_swath(first_output, transaction_id, commit_time_s)

    resumed_output = first_output
    run_command(
        common
        + [
            "--duration=30",
            "--checkpoint-every=0",
            f"--outdir={resumed_output}",
            f"--resume={checkpoints[0]}",
        ],
        "resumed live D5 run", compute_backend, required_backend,
    )
    branch = load_json(resumed_output / "nimbus_branch.json")
    require(branch.get("parent_run_id") == sidecar.get("run_id"), "D5 branch parent mismatch")
    require(branch.get("resumed_step") == 48, "wrong D5 resumed step")
    require(float(branch.get("resumed_simulation_time_s", -1.0)) == 24.0,
            "wrong D5 resumed time")
    for filename, before_bytes in preserved_record_bytes.items():
        preserved_path = resumed_output / "storm_track" / filename
        require(preserved_path.is_file(), f"resume removed storm-track history record {filename}")
        require(preserved_path.read_bytes() == before_bytes,
                f"resume rewrote storm-track history record {filename}")
    resume_markers = sorted((resumed_output / "storm_track").glob("resume_step_*.json"))
    require(len(resume_markers) == 1, "resume did not publish exactly one storm-track lineage marker")
    storm_branch = load_json(resume_markers[0])
    require(storm_branch.get("schema") == "nimbus.storm_track.resume",
            "wrong storm-track resume schema")
    require(storm_branch.get("parent_run_id") == sidecar.get("run_id"),
            "storm-track resume parent mismatch")
    require(storm_branch.get("branch_run_id") == branch.get("run_id"),
            "storm-track resume branch mismatch")
    require(storm_branch.get("history_mode") == "preserved_prefix",
            "same-directory resume did not retain its tracker prefix")
    require(storm_branch.get("preserved_history_records") == len(preserved_record_bytes),
            "storm-track resume marker has the wrong preserved-record count")
    require(storm_branch.get("resumed_step") == 48,
            "storm-track resume marker has the wrong checkpoint step")
    latest = load_json(resumed_output / "storm_track_latest.json")
    require(latest.get("schema_version") == 12, "wrong resumed tracker schema")
    require(float(latest.get("simulation_time_s", -1.0)) == 30.0, "resumed D5 run did not reach 30 s")
    require(pair(latest.get("horizontal_frame", {}).get("grid_origin_t0_earth_m"),
                 "x", "y", "resumed frame origin") == EXPECTED_POST_ORIGIN_T0_M,
            "resumed D5 frame origin is wrong")
    require(pair(latest.get("movement_authority", {}).get("cumulative_committed_shift_cells"),
                 "i", "j", "resumed cumulative shift") == EXPECTED_SHIFT,
            "resumed D5 cumulative shift is wrong")
    preserved_after_resume = {
        path.relative_to(resumed_output).as_posix(): path.read_bytes()
        for path in sorted((resumed_output / "storm_track").iterdir())
        if path.is_file()
    }
    latest_after_resume = (resumed_output / "storm_track_latest.json").read_bytes()
    rejected = subprocess.run(
        common
        + [
            "--duration=30",
            "--checkpoint-every=0",
            f"--outdir={resumed_output}",
            f"--resume={checkpoints[0]}",
        ],
        cwd=ROOT,
        env=runtime_environment(compute_backend),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(rejected.returncode != 0, "resume accepted tracker history newer than its checkpoint")
    require("storm-track history extends beyond the resume checkpoint" in rejected.stdout,
            "stale-future tracker rejection did not report its reason")
    require((resumed_output / "storm_track_latest.json").read_bytes() == latest_after_resume,
            "rejected resume rewrote storm_track_latest.json")
    for relative_path, before_bytes in preserved_after_resume.items():
        artifact = resumed_output / relative_path
        require(artifact.is_file(), f"rejected resume removed {relative_path}")
        require(artifact.read_bytes() == before_bytes,
                f"rejected resume rewrote {relative_path}")
    return {"checkpoint_step": 48, "resumed_step": int(latest["completed_step"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="retain evidence in this directory")
    parser.add_argument(
        "--compute-backend", choices=("cpu", "vulkan", "metal"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    required = parser.add_mutually_exclusive_group()
    required.add_argument("--require-vulkan", action="store_true",
                          help="fail unless every branch retains a physical Vulkan device")
    required.add_argument("--require-metal", action="store_true",
                          help="fail unless every branch retains a physical Metal device")
    parser.add_argument(
        "--require-native-chaos", action="store_true",
        help="require native Earth-control dispatch in branches that evolve chaos controls",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    config = args.config.resolve()
    require(binary.is_file(), f"missing production binary: {binary}")
    require(config.is_file(), f"missing qualification config: {config}")
    required_backend = (
        "vulkan" if args.require_vulkan else "metal" if args.require_metal else None
    )
    require(required_backend is None or args.compute_backend == required_backend,
            f"--require-{required_backend} requires "
            f"--compute-backend={required_backend}")
    require(args.threads > 0, "--threads must be positive")
    require(not args.require_native_chaos or required_backend is not None,
            "--require-native-chaos requires --require-vulkan or --require-metal")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.output is None:
        temporary = tempfile.TemporaryDirectory(prefix="nimbus_domain_recycling_")
        output = Path(temporary.name)
    else:
        output = args.output.resolve()
        require(not output.exists() or not any(output.iterdir()), f"output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)

    try:
        direct_output = output / "direct"
        run_direct(
            binary, config, direct_output,
            args.compute_backend, required_backend, args.threads,
            args.require_native_chaos,
        )
        transaction_id, tracking = qualify_tracking(direct_output)
        commit_time_s = float(tracking["commit_time_s"])
        swath = qualify_swath(direct_output, transaction_id, commit_time_s)
        trajectories = qualify_trajectories(direct_output / "trajectories.npy", commit_time_s)
        restart = qualify_restart(
            binary, config, output / "restart",
            args.compute_backend, required_backend, args.threads,
            args.require_native_chaos,
        )
        print(
            f"[domain-recycling] PASS ({args.compute_backend}, {args.threads} thread(s)): committed "
            f"{transaction_id} at step {tracking['commit_step']} "
            f"({commit_time_s:g} s); "
            f"{tracking['tracking_records']} tracker records; "
            f"{swath['artifact_count']} hash-verified swath artifacts; "
            f"{trajectories['surviving_crossings']} trajectory crossings; "
            f"maximum crossing speed {trajectories['maximum_crossing_speed_ms']:.3f} m/s; "
            f"checkpoint/resume {restart['checkpoint_step']} -> {restart['resumed_step']}"
            f"; native Earth-control dispatch required={args.require_native_chaos}"
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

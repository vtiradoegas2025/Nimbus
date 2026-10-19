#!/usr/bin/env python3
"""Validate and seal a segmented fresh-start accelerator moving-nest campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qualify_moving_nested_supercell as moving


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "tests" / "artifacts"
DEFAULT_GLOB = "moving_nested_vulkan_lifetime_v2_20260819_segment_*"
DEFAULT_OUTPUT = ARTIFACTS / "moving_nested_vulkan_lifetime_v2_20260819_chain_report.json"
SCHEMA = "nimbus.diagnostic.moving_nested_vulkan_lifetime_chain"
SEGMENT_NAME = re.compile(r".*_segment_([0-9]{4})_([0-9]{4})$")
SUPPORTED_BACKENDS = ("vulkan", "metal")


def default_segment_glob(compute_backend: str) -> str:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    if compute_backend == "vulkan":
        return DEFAULT_GLOB
    return "moving_nested_metal_lifetime_v1_segment_*"


def default_output(compute_backend: str) -> Path:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    if compute_backend == "vulkan":
        return DEFAULT_OUTPUT
    return ARTIFACTS / "moving_nested_metal_lifetime_v1_chain_report.json"


class ChainError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ChainError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChainError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def segment_bounds(path: Path) -> tuple[float, float]:
    match = SEGMENT_NAME.fullmatch(path.name)
    require(match is not None, f"segment name lacks exact bounds: {path.name}")
    return float(match.group(1)), float(match.group(2))


def discover_segments(pattern: str) -> list[Path]:
    result = sorted(
        (path.resolve() for path in ARTIFACTS.glob(pattern) if path.is_dir()),
        key=segment_bounds,
    )
    require(result, f"no Vulkan lifetime segments match {pattern}")
    return result


def checkpoint_times(start_s: float, end_s: float) -> list[float]:
    duration_s = end_s - start_s
    intervals = duration_s / 450.0
    require(duration_s > 0.0 and intervals.is_integer(),
            "segment duration must be a positive multiple of 450 seconds")
    return [start_s + 450.0 * interval
            for interval in range(1, int(intervals) + 1)]


def tracker_records(segment: Path, start_s: float, end_s: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (segment / "storm_track").glob("track_step_*.json"):
        record = load_json(path)
        record["_path"] = path
        rows.append(record)
    rows.sort(key=lambda item: float(item.get("simulation_time_s", math.nan)))
    expected = [start_s + 30.0 * index
                for index in range(1, int((end_s - start_s) / 30.0) + 1)]
    actual = [float(item.get("simulation_time_s", math.nan)) for item in rows]
    require(actual == expected,
            f"tracker cadence changed in {segment.name}: expected {expected}, got {actual}")
    return rows


def inspect_chain(segments: list[Path], expected_duration_s: float,
                  compute_backend: str = "vulkan") -> dict[str, Any]:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    require(expected_duration_s > 0.0 and expected_duration_s % 450.0 == 0.0,
            "expected duration must be a positive 450-second boundary")
    expected_start = 0.0
    binary_sha: str | None = None
    config_sha: str | None = None
    prior_checkpoint: Path | None = None
    prior_checkpoint_sha: str | None = None
    prior_run_id: str | None = None
    segment_records: list[dict[str, Any]] = []
    all_trackers: list[dict[str, Any]] = []
    relocations: list[dict[str, Any]] = []
    checkpoint_count = 0
    aggregate_counters: dict[str, dict[str, int]] = {
        "smagorinsky": {}, "kessler": {}}

    for index, segment in enumerate(segments):
        start_s, end_s = segment_bounds(segment)
        segment_duration_s = end_s - start_s
        require(start_s == expected_start,
                f"segment chain is not contiguous at {segment.name}")
        expected_checkpoint_times = checkpoint_times(start_s, end_s)
        report_path = segment / f"moving_nested_{compute_backend}_runtime_report.json"
        report = load_json(report_path)
        require(report.get("schema") ==
                f"nimbus.diagnostic.moving_nested_{compute_backend}_live_storm" and
                report.get("schema_version") == 1 and report.get("status") == "pass" and
                report.get("qualification_eligible") is False,
                f"segment report is not a diagnostic pass: {segment.name}")
        claims = report.get("claims", {})
        require(claims.get(f"fallback_off_physical_{compute_backend}") is True and
                claims.get("native_parent_child_smagorinsky") is True and
                claims.get("native_parent_child_kessler_processes") is True and
                claims.get("host_authoritative_kessler_state_commit") is True and
                claims.get("strict_finite_bounded_outputs") is True and
                claims.get("public_moving_nest_admitted") is False and
                claims.get("device_resident_relocation") is False,
                f"segment backend/science claims changed: {segment.name}")
        inputs = report.get("inputs", {})
        current_binary_sha = inputs.get("binary_sha256")
        current_config_sha = inputs.get("config_sha256")
        require(isinstance(current_binary_sha, str) and len(current_binary_sha) == 64 and
                isinstance(current_config_sha, str) and len(current_config_sha) == 64,
                f"segment input hashes are invalid: {segment.name}")
        binary_sha = current_binary_sha if binary_sha is None else binary_sha
        config_sha = current_config_sha if config_sha is None else config_sha
        require(current_binary_sha == binary_sha and current_config_sha == config_sha,
                f"binary/config identity changed at {segment.name}")
        require(float(inputs.get("source_checkpoint_time_s", math.nan)) == start_s and
                float(inputs.get("duration_s", math.nan)) == end_s,
                f"segment input time bounds changed: {segment.name}")
        if index == 0:
            require(report.get("scope") ==
                    f"exploratory_fresh_start_physical_{compute_backend}" and
                    inputs.get("initialization") == "fresh_configured_state",
                    "first lifetime segment is not a fresh start")
        else:
            require(report.get("scope") ==
                    f"exploratory_cross_backend_resume_physical_{compute_backend}",
                    f"continued segment scope changed: {segment.name}")
            source = (ROOT / str(inputs.get("source_checkpoint"))).resolve()
            require(prior_checkpoint is not None and source == prior_checkpoint and
                    inputs.get("source_checkpoint_sha256") == prior_checkpoint_sha and
                    sha256(source) == prior_checkpoint_sha,
                    f"checkpoint hash/lineage changed at {segment.name}")

        artifact_block = report.get("artifacts", {})
        require(artifact_block.get("checkpoint_count") == len(expected_checkpoint_times) and
                artifact_block.get("checkpoint_times_s") == expected_checkpoint_times,
                f"segment checkpoint cadence changed: {segment.name}")
        sidecar_paths = sorted(
            (segment / "checkpoints").glob("checkpoint_step_*.nimbus.json"),
            key=lambda path: float(load_json(path).get("simulation_time_s", math.nan)),
        )
        require(len(sidecar_paths) == len(expected_checkpoint_times),
                f"segment checkpoint inventory changed: {segment.name}")
        checkpoint_records: list[dict[str, Any]] = []
        checkpoint_path: Path | None = None
        sidecar_path: Path | None = None
        for expected_checkpoint_time, current_sidecar_path in zip(
                expected_checkpoint_times, sidecar_paths, strict=True):
            sidecar = load_json(current_sidecar_path)
            current_checkpoint_path = current_sidecar_path.with_suffix("")
            require(current_checkpoint_path.is_file() and
                    float(sidecar.get("simulation_time_s", math.nan)) ==
                    expected_checkpoint_time and
                    sidecar.get("boundary") == "completed_timestep" and
                    sidecar.get("portable_across_compute_memory_topology") is True,
                    f"segment checkpoint is not a portable completed boundary: "
                    f"{segment.name}@{expected_checkpoint_time:g}")
            if prior_checkpoint is not None:
                require(Path(str(sidecar.get("parent_checkpoint"))).resolve() ==
                        prior_checkpoint and
                        sidecar.get("parent_run_id") == prior_run_id,
                        f"checkpoint sidecar parent changed at {segment.name}")
            checkpoint_records.append({
                "simulation_time_s": expected_checkpoint_time,
                "checkpoint": artifact(current_checkpoint_path),
                "sidecar": artifact(current_sidecar_path),
            })
            checkpoint_path = current_checkpoint_path
            sidecar_path = current_sidecar_path
        require(checkpoint_path is not None and sidecar_path is not None and
                (ROOT / str(artifact_block.get("final_checkpoint_sidecar"))).resolve() ==
                sidecar_path,
                f"segment final checkpoint pointer changed: {segment.name}")
        checkpoint_count += len(checkpoint_records)

        rows = tracker_records(segment, start_s, end_s)
        runtime = report.get("runtime", {})
        expected_tracker_count = int(segment_duration_s / 30.0)
        require(runtime.get("tracker_samples") == len(rows) == expected_tracker_count and
                float(runtime.get("first_tracker_time_s", math.nan)) == start_s + 30.0 and
                float(runtime.get("final_tracker_time_s", math.nan)) == end_s,
                f"segment tracker summary changed: {segment.name}")
        segment_relocations: list[dict[str, Any]] = []
        for record in rows:
            require(record.get("schema_version") == 12 and
                    record.get("tracking_source") == "synchronized_nested_child",
                    f"tracker ownership changed in {segment.name}")
            authority = record.get("movement_authority", {})
            moving.require_authority_contract(authority)
            moving.require_runtime_scientific_validity(record)
            moving.require_precipitation_relocation(
                authority, float(record["simulation_time_s"]))
            if authority.get("commit_performed") is True:
                relocation = {
                    "simulation_time_s": record["simulation_time_s"],
                    "shift_cells": authority.get("committed_shift_cells"),
                    "cumulative_shift_cells":
                        authority.get("cumulative_committed_shift_cells"),
                    "reason": authority.get("reason"),
                }
                segment_relocations.append(relocation)
                relocations.append(relocation)
            clean = dict(record)
            clean.pop("_path", None)
            all_trackers.append(clean)
        require(runtime.get("relocations") and segment_relocations
                if claims.get("host_owned_live_relocation_crossed") is True
                else not runtime.get("relocations") and not segment_relocations,
                f"segment relocation summary disagrees with trackers: {segment.name}")

        counters = runtime.get("device_level_counters", {})
        for family in ("smagorinsky", "kessler"):
            values = counters.get(family, {})
            require(isinstance(values, dict) and values,
                    f"segment lacks {family} counters: {segment.name}")
            for name, value in values.items():
                require(type(value) is int and value >= 0,
                        f"invalid {family}.{name} counter in {segment.name}")
                aggregate_counters[family][name] = \
                    aggregate_counters[family].get(name, 0) + value
        require(counters["kessler"].get("parent_apply") == 0 and
                counters["kessler"].get("child_apply") == 0,
                f"segment changed host-authoritative Kessler commit: {segment.name}")

        runtime_log_path = segment / "qualification_run.log"
        require((ROOT / str(artifact_block.get("runtime_log"))).resolve() ==
                runtime_log_path.resolve() and
                artifact_block.get("runtime_log_sha256") == sha256(runtime_log_path),
                f"segment runtime log binding changed: {segment.name}")

        segment_records.append({
            "start_s": start_s,
            "end_s": end_s,
            "directory": relative(segment),
            "report": artifact(report_path),
            "runtime_log": artifact(segment / "qualification_run.log"),
            "checkpoints": checkpoint_records,
            "final_checkpoint": artifact(checkpoint_path),
            "final_checkpoint_sidecar": artifact(sidecar_path),
            "tracker_samples": len(rows),
            "relocations": segment_relocations,
            "simulated_seconds_per_solver_wall_second":
                runtime.get("simulated_seconds_per_solver_wall_second"),
        })
        prior_checkpoint = checkpoint_path
        prior_checkpoint_sha = sha256(checkpoint_path)
        prior_run_id = load_json(sidecar_path).get("run_id")
        require(isinstance(prior_run_id, str) and prior_run_id,
                f"segment final checkpoint lacks a run id: {segment.name}")
        expected_start = end_s

    require(expected_start <= expected_duration_s,
            "segment chain exceeds the expected lifetime")
    expected_tracker_times = [30.0 * index
                              for index in range(1, int(expected_start / 30.0) + 1)]
    require([float(row["simulation_time_s"]) for row in all_trackers] ==
            expected_tracker_times,
            "combined tracker cadence is incomplete")
    final_authority = all_trackers[-1]["movement_authority"]
    uninterrupted = (len(segment_records) == 1 and
                     segment_records[0]["start_s"] == 0.0 and
                     segment_records[0]["end_s"] == expected_duration_s)
    return {
        "schema": f"nimbus.diagnostic.moving_nested_{compute_backend}_lifetime_chain",
        "schema_version": 1,
        "status": "complete" if expected_start == expected_duration_s else "in_progress",
        "qualification_eligible": False,
        "scope": (f"uninterrupted_fresh_start_physical_{compute_backend}_lifetime"
                  if uninterrupted else
                  f"segmented_fresh_start_physical_{compute_backend}_lifetime"),
        "claims": {
            f"fallback_off_physical_{compute_backend}": True,
            "single_binary_config_identity": True,
            "uninterrupted_single_process": uninterrupted,
            "checkpoint_lineage_exact": True,
            "native_parent_child_smagorinsky": True,
            "native_parent_child_kessler_processes": True,
            "host_authoritative_kessler_state_commit": True,
            "host_owned_relocations": bool(relocations),
            "device_resident_relocation": False,
            "public_moving_nest_admitted": False,
            "formal_reference_comparison_complete": False,
        },
        "progress": {
            "completed_through_s": expected_start,
            "expected_duration_s": expected_duration_s,
            "fraction_complete": expected_start / expected_duration_s,
            "segment_count": len(segment_records),
            "tracker_samples": len(all_trackers),
            "checkpoint_count": checkpoint_count,
            "relocation_count": len(relocations),
            "relocations": relocations,
            "final_cumulative_shift_cells":
                final_authority.get("cumulative_committed_shift_cells"),
        },
        "inputs": {
            "binary_sha256": binary_sha,
            "config_sha256": config_sha,
        },
        "aggregate_device_level_counters": aggregate_counters,
        "segments": segment_records,
        "next_resume": {
            "checkpoint": relative(prior_checkpoint),
            "checkpoint_sha256": prior_checkpoint_sha,
            "time_s": expected_start,
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compute-backend", choices=SUPPORTED_BACKENDS, default="vulkan")
    parser.add_argument("--segment-glob")
    parser.add_argument("--expected-duration", type=float, default=4500.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    segment_glob = args.segment_glob or default_segment_glob(args.compute_backend)
    output = (args.output or default_output(args.compute_backend)).resolve()
    report = inspect_chain(
        discover_segments(segment_glob), args.expected_duration,
        args.compute_backend)
    write_json(output, report)
    print(json.dumps({
        "status": report["status"],
        "qualification_eligible": report["qualification_eligible"],
        **report["progress"],
        "output": relative(output),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] in {"in_progress", "complete"} else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ChainError, RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[moving-nested-accelerator-lifetime-chain] FAIL: {exc}",
              file=sys.stderr)
        raise SystemExit(2)

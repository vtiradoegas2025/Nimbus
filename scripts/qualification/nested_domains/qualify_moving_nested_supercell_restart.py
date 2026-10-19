#!/usr/bin/env python3
"""Qualify restart-exact live-storm movement on the internal nested campaign."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import qualify_moving_nested_supercell as campaign


CHECKPOINT_INTERVAL_S = 450
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
U64_MASK = 0xFFFFFFFFFFFFFFFF


class CheckpointLifecycle:
    __slots__ = ("path", "completed_steps", "simulation_time_s",
                 "checkpoint_bytes", "checkpoint_fnv1a64", "run_id",
                 "parent_run_id", "parent_checkpoint")

    def __init__(self, path: Path, completed_steps: int,
                 simulation_time_s: float, checkpoint_bytes: int,
                 checkpoint_fnv1a64: str, run_id: str,
                 parent_run_id: str, parent_checkpoint: str) -> None:
        self.path = path
        self.completed_steps = completed_steps
        self.simulation_time_s = simulation_time_s
        self.checkpoint_bytes = checkpoint_bytes
        self.checkpoint_fnv1a64 = checkpoint_fnv1a64
        self.run_id = run_id
        self.parent_run_id = parent_run_id
        self.parent_checkpoint = parent_checkpoint


class ContinuousAuthority:
    __slots__ = ("output", "report", "checkpoints", "records",
                 "split_checkpoint", "expected_trace",
                 "first_movement_time_s", "first_face_motion_sample_time_s")

    def __init__(self, output: Path, report: dict,
                 checkpoint_paths: list[Path], records: list[dict],
                 split_checkpoint: CheckpointLifecycle,
                 expected_trace: list[dict], first_movement_time_s: float,
                 first_face_motion_sample_time_s: float) -> None:
        self.output = output
        self.report = report
        self.checkpoints = checkpoint_paths
        self.records = records
        self.split_checkpoint = split_checkpoint
        self.expected_trace = expected_trace
        self.first_movement_time_s = first_movement_time_s
        self.first_face_motion_sample_time_s = first_face_motion_sample_time_s


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_branch(binary: Path, output: Path, checkpoints: Path,
               duration_s: int, threads: int,
               checkpoint_every_s: int,
               resume: Path | None = None) -> tuple[float, str]:
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(campaign.CONFIG, output / "qualification_input.yaml")
    (output / "qualification_binary.sha256").write_text(
        campaign.digest(binary) + "\n", encoding="utf-8")
    command = [
        str(binary),
        "--qualification-moving-nest",
        "--headless",
        "--config", str(campaign.CONFIG),
        "--duration", str(duration_s),
        f"--write-every={checkpoint_every_s}",
        f"--checkpoint-every={checkpoint_every_s}",
        f"--checkpoint-dir={checkpoints}",
        "--threads", str(threads),
        "--guard-mode", "strict",
        f"--outdir={output}",
        "--log-profile", "normal",
        "--timing",
    ]
    if resume is not None:
        command.append(f"--resume={resume}")
    started = time.monotonic()
    result = subprocess.run(
        command, cwd=campaign.ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    wall = time.monotonic() - started
    (output / "qualification_run.log").write_text(
        result.stdout, encoding="utf-8")
    require(result.returncode == 0,
            f"moving-nest branch exited {result.returncode}\n{result.stdout[-16000:]}")
    return wall, result.stdout


def checkpoints(path: Path) -> list[Path]:
    return sorted(path.glob("checkpoint_step_*.nimbus"))


def fnv1a64(path: Path) -> str:
    value = FNV_OFFSET
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            for byte in chunk:
                value ^= byte
                value = (value * FNV_PRIME) & U64_MASK
    return f"{value:016x}"


def checkpoint_step_identity(path: Path) -> int:
    try:
        step = int(path.stem.removeprefix("checkpoint_step_"))
    except ValueError as exc:
        raise RuntimeError(f"invalid checkpoint step identity: {path}") from exc
    require(step >= 0, f"negative checkpoint step identity: {path}")
    return step


def load_checkpoint_lifecycle(path: Path) -> CheckpointLifecycle:
    require(path.is_file(), f"missing checkpoint: {path}")
    sidecar_path = Path(f"{path}.json")
    require(sidecar_path.is_file(), f"missing checkpoint lifecycle sidecar: {sidecar_path}")
    sidecar = campaign.load_json(sidecar_path)
    require(sidecar.get("schema") == "nimbus.checkpoint.lifecycle" and
            sidecar.get("schema_version") == 1,
            f"checkpoint lifecycle schema mismatch: {sidecar_path}")
    require(sidecar.get("checkpoint_file") == path.name,
            f"checkpoint lifecycle file identity mismatch: {sidecar_path}")
    completed_steps = sidecar.get("completed_steps")
    require(type(completed_steps) is int and completed_steps >= 0 and
            completed_steps == checkpoint_step_identity(path),
            f"checkpoint lifecycle step identity mismatch: {sidecar_path}")
    simulation_time_s = sidecar.get("simulation_time_s")
    require(type(simulation_time_s) in (int, float) and
            math.isfinite(float(simulation_time_s)) and
            float(simulation_time_s) >= 0.0,
            f"checkpoint lifecycle simulation time is invalid: {sidecar_path}")
    checkpoint_bytes = sidecar.get("checkpoint_bytes")
    require(type(checkpoint_bytes) is int and checkpoint_bytes >= 0 and
            checkpoint_bytes == path.stat().st_size,
            f"checkpoint lifecycle byte count mismatch: {sidecar_path}")
    checkpoint_digest = sidecar.get("checkpoint_fnv1a64")
    require(isinstance(checkpoint_digest, str) and
            re.fullmatch(r"[0-9a-f]{16}", checkpoint_digest) is not None and
            checkpoint_digest == fnv1a64(path),
            f"checkpoint lifecycle digest mismatch: {sidecar_path}")
    run_id = sidecar.get("run_id")
    require(isinstance(run_id, str) and run_id,
            f"checkpoint lifecycle run identity is invalid: {sidecar_path}")
    parent_run_id = sidecar.get("parent_run_id")
    parent_checkpoint = sidecar.get("parent_checkpoint")
    require(isinstance(parent_run_id, str) and
            isinstance(parent_checkpoint, str),
            f"checkpoint lifecycle parent identity is invalid: {sidecar_path}")
    require(sidecar.get("boundary") == "completed_timestep",
            f"checkpoint lifecycle boundary is not restart-safe: {sidecar_path}")
    require(sidecar.get("portable_across_compute_memory_topology") is True,
            f"checkpoint lifecycle portability contract changed: {sidecar_path}")
    return CheckpointLifecycle(
        path=path,
        completed_steps=completed_steps,
        simulation_time_s=float(simulation_time_s),
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_fnv1a64=checkpoint_digest,
        run_id=run_id,
        parent_run_id=parent_run_id,
        parent_checkpoint=parent_checkpoint)


def checkpoint_simulation_time_s(path: Path) -> float:
    return load_checkpoint_lifecycle(path).simulation_time_s


def select_restart_checkpoint(paths: list[Path],
                              first_movement_time_s: float) -> CheckpointLifecycle:
    require(first_movement_time_s > 0.0,
            "continuous movement has an invalid sample time")
    lifecycles = [load_checkpoint_lifecycle(item) for item in paths]
    eligible = [
        item for item in lifecycles
        if item.simulation_time_s < first_movement_time_s
    ]
    require(eligible,
            "no committed checkpoint precedes the first movement; use a shorter checkpoint cadence")
    return max(eligible, key=lambda item: item.simulation_time_s)


def require_final_checkpoint_exact(continuous_paths: list[Path],
                                   resumed_paths: list[Path],
                                   final_time_s: int) -> CheckpointLifecycle:
    require(continuous_paths, "continuous branch produced no checkpoint")
    require(resumed_paths, "resumed branch produced no checkpoint")
    continuous = load_checkpoint_lifecycle(continuous_paths[-1])
    resumed = load_checkpoint_lifecycle(resumed_paths[-1])
    require(continuous.simulation_time_s == final_time_s and
            resumed.simulation_time_s == final_time_s,
            "continuous/resumed final checkpoint sidecar is not at the requested endpoint")
    require(continuous.completed_steps == resumed.completed_steps and
            continuous.path.name == resumed.path.name,
            "continuous/resumed final checkpoint step identity differs")
    require(continuous.checkpoint_bytes == resumed.checkpoint_bytes and
            continuous.checkpoint_fnv1a64 == resumed.checkpoint_fnv1a64,
            "continuous/resumed final checkpoint lifecycle identity differs")
    require(continuous.path.read_bytes() == resumed.path.read_bytes(),
            "continuous and resumed moving-nest checkpoints differ")
    return continuous


def require_exact_parent_checkpoint(candidate: object,
                                    split: CheckpointLifecycle,
                                    context: str) -> None:
    require(isinstance(candidate, str) and candidate,
            f"{context} lacks its parent checkpoint")
    candidate_path = Path(candidate)
    require(candidate_path.is_absolute(),
            f"{context} parent checkpoint is not absolute")
    try:
        same_path = candidate_path.resolve(strict=True) == split.path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"{context} parent checkpoint cannot be resolved: {exc}") from exc
    require(same_path,
            f"{context} is not bound to the selected split checkpoint")


def require_split_identity(current: CheckpointLifecycle,
                           selected: CheckpointLifecycle) -> None:
    require(
        current.completed_steps == selected.completed_steps and
        current.simulation_time_s == selected.simulation_time_s and
        current.checkpoint_bytes == selected.checkpoint_bytes and
        current.checkpoint_fnv1a64 == selected.checkpoint_fnv1a64 and
        current.run_id == selected.run_id and
        current.parent_run_id == selected.parent_run_id and
        current.parent_checkpoint == selected.parent_checkpoint,
        "selected split checkpoint lifecycle changed before lineage validation")


def validate_resumed_lineage(
        output: Path, split: CheckpointLifecycle,
        resumed_paths: list[Path]) -> list[CheckpointLifecycle]:
    """Bind a fresh resumed branch to one exact uninterrupted split state."""
    current_split = load_checkpoint_lifecycle(split.path)
    require_split_identity(current_split, split)
    require(not current_split.parent_run_id and not current_split.parent_checkpoint,
            "selected split checkpoint is not from an uninterrupted authority")

    branch = campaign.load_json(output / "nimbus_branch.json")
    require(branch.get("schema") == "nimbus.checkpoint.branch" and
            branch.get("schema_version") == 1,
            "resumed branch provenance schema mismatch")
    branch_run_id = branch.get("run_id")
    require(isinstance(branch_run_id, str) and branch_run_id and
            branch_run_id != split.run_id,
            "resumed branch run identity is invalid")
    require(branch.get("parent_run_id") == split.run_id,
            "resumed branch parent run identity differs from the split authority")
    require_exact_parent_checkpoint(
        branch.get("parent_checkpoint"), split, "resumed branch provenance")
    require(branch.get("resumed_step") == split.completed_steps and
            branch.get("resumed_simulation_time_s") == split.simulation_time_s,
            "resumed branch provenance differs from the split boundary")

    marker_paths = sorted((output / "storm_track").glob("resume_step_*.json"))
    require(len(marker_paths) == 1,
            "resumed branch must publish exactly one storm-track resume marker")
    expected_marker_name = (
        f"resume_step_{split.completed_steps:09d}_{branch_run_id}.json")
    marker_path = marker_paths[0]
    require(marker_path.name == expected_marker_name,
            "storm-track resume marker filename differs from branch provenance")
    marker = campaign.load_json(marker_path)
    require(marker.get("schema") == "nimbus.storm_track.resume" and
            marker.get("schema_version") == 1,
            "storm-track resume marker schema mismatch")
    require(marker.get("branch_run_id") == branch_run_id and
            marker.get("parent_run_id") == split.run_id,
            "storm-track resume marker run lineage differs from branch provenance")
    require_exact_parent_checkpoint(
        marker.get("parent_checkpoint"), split, "storm-track resume marker")
    require(marker.get("parent_checkpoint_fnv1a64") ==
            split.checkpoint_fnv1a64,
            "storm-track resume marker digest differs from the split checkpoint")
    require(marker.get("resumed_step") == split.completed_steps and
            marker.get("resumed_simulation_time_s") == split.simulation_time_s,
            "storm-track resume marker differs from the split boundary")
    require(marker.get("history_mode") == "new_branch" and
            marker.get("preserved_history_records") == 0 and
            marker.get("preserved_last_completed_step") is None,
            "fresh resumed output unexpectedly preserved storm-track history")

    require(resumed_paths, "resumed branch produced no checkpoint")
    lifecycles = [load_checkpoint_lifecycle(path) for path in resumed_paths]
    require(all(item.run_id == branch_run_id for item in lifecycles),
            "resumed checkpoint run identity differs from branch provenance")
    require(all(item.parent_run_id == split.run_id for item in lifecycles),
            "resumed checkpoint parent run identity differs from the split authority")
    for item in lifecycles:
        require_exact_parent_checkpoint(
            item.parent_checkpoint, split, "resumed checkpoint lifecycle")
    require(all(lhs.completed_steps < rhs.completed_steps and
                lhs.simulation_time_s < rhs.simulation_time_s
                for lhs, rhs in zip(lifecycles, lifecycles[1:])),
            "resumed checkpoint lifecycle is not strictly ordered")
    require(lifecycles[0].completed_steps > split.completed_steps and
            lifecycles[0].simulation_time_s > split.simulation_time_s,
            "resumed checkpoint lifecycle does not begin after the split boundary")
    return lifecycles


def require_post_split_face_motion_before_move(records: list[dict]) -> float:
    first_commit_index = next(
        (index for index, record in enumerate(records)
         if record.get("movement_authority", {}).get("commit_performed") is True),
        None)
    require(first_commit_index is not None,
            "selected restart suffix does not contain a movement")
    selected_before_move = [
        record for record in records[:first_commit_index]
        if record.get("selected_feature_available") is True
    ]
    require(selected_before_move,
            "restart suffix has no selected storm sample before its first movement")
    first_selected = selected_before_move[0]
    motion = first_selected.get("envelope_face_motion")
    require(isinstance(motion, dict) and motion.get("available") is True,
            "first selected post-split sample lacks restored envelope-face motion")
    sample_time_s = first_selected.get("simulation_time_s")
    require(type(sample_time_s) in (int, float) and
            math.isfinite(float(sample_time_s)),
            "post-split envelope-face motion sample time is invalid")
    return float(sample_time_s)


def require_no_scientific_invalidity_artifact(output: Path) -> None:
    require(not (output / "moving_nest_scientific_validity.json").exists(),
            "branch published a moving-nest scientific-invalidity artifact")


def validate_continuous_authority(
        output: Path, binary: Path, final_time_s: int,
        wall_seconds: float | None = None) -> ContinuousAuthority:
    output = output.resolve()
    require(output.is_dir(),
            f"continuous authority directory is missing: {output}")
    _, recorded_binary_digest, _ = campaign.campaign_artifact_binding(output)
    selected_binary_digest = campaign.digest(binary)
    require(recorded_binary_digest == selected_binary_digest,
            "continuous authority binary digest does not match the selected "
            "qualification binary")

    contract_path = output / "nimbus_output.json"
    require(contract_path.is_file(),
            "continuous authority lacks its run-readiness artifact")
    contract = campaign.load_json(contract_path)
    require(contract.get("publication_state") == "ready",
            "continuous authority run publication is incomplete")
    require(contract.get("data_products") ==
            ["history", "nested_child_history"],
            "continuous authority nested data-product contract changed")
    require_no_scientific_invalidity_artifact(output)

    checkpoint_paths = checkpoints(output / "checkpoints")
    require(checkpoint_paths,
            "continuous authority produced no checkpoint")
    lifecycles = [load_checkpoint_lifecycle(path) for path in checkpoint_paths]
    run_ids = {item.run_id for item in lifecycles}
    require(len(run_ids) == 1,
            "continuous authority checkpoint run identity is not contiguous")
    require(all(not item.parent_run_id and not item.parent_checkpoint
                for item in lifecycles),
            "continuous authority checkpoint lineage is resumed, not uninterrupted")
    require(all(lhs.completed_steps < rhs.completed_steps and
                lhs.simulation_time_s < rhs.simulation_time_s
                for lhs, rhs in zip(lifecycles, lifecycles[1:])),
            "continuous authority checkpoint lifecycle is not strictly ordered")
    final_checkpoint = lifecycles[-1]
    require(final_checkpoint.simulation_time_s == final_time_s,
            "continuous authority is stale or does not end at the requested endpoint")

    records = tracker_records(output)
    require(records,
            "continuous authority published no tracker evidence")
    for record in records:
        campaign.require_runtime_scientific_validity(record)
    commits = [
        record for record in records
        if record.get("movement_authority", {}).get("commit_performed") is True
    ]
    require(commits,
            "continuous authority never moved; extend --final-time for a movement restart gate")
    first_movement_time_s = float(commits[0].get(
        "simulation_time_s", float("nan")))
    require(math.isfinite(first_movement_time_s) and
            first_movement_time_s > 0.0,
            "continuous movement has an invalid sample time")
    split_checkpoint = select_restart_checkpoint(
        checkpoint_paths, first_movement_time_s)
    split_time_s = split_checkpoint.simulation_time_s
    expected_suffix_records = [
        record for record in records
        if float(record.get("simulation_time_s", -1.0)) > split_time_s
    ]
    expected_trace = campaign.movement_trace(expected_suffix_records)
    first_face_motion_sample_time_s = (
        require_post_split_face_motion_before_move(expected_suffix_records))

    report = campaign.inspect(
        output, final_time_s, wall_seconds, True)
    require(report.get("status") == "pass" and
            report.get("binary_sha256") == selected_binary_digest and
            report.get("config_sha256") == campaign.digest(campaign.CONFIG),
            "continuous authority inspection is not bound to the selected binary and config")
    return ContinuousAuthority(
        output=output,
        report=report,
        checkpoint_paths=checkpoint_paths,
        records=records,
        split_checkpoint=split_checkpoint,
        expected_trace=expected_trace,
        first_movement_time_s=first_movement_time_s,
        first_face_motion_sample_time_s=first_face_motion_sample_time_s)


def final_step(path: Path) -> Path:
    steps = sorted(item for item in path.glob("step_*") if item.is_dir())
    require(steps, f"no history steps in {path}")
    return steps[-1]


def require_endpoint_exact(continuous: Path, resumed: Path, label: str,
                           final_time_s: int) -> None:
    continuous_step = final_step(continuous)
    resumed_step = final_step(resumed)
    continuous_manifest = campaign.load_json(continuous_step / "manifest.json")
    resumed_manifest = campaign.load_json(resumed_step / "manifest.json")
    require(continuous_manifest.get("simulation_time_s") == final_time_s,
            f"continuous {label} did not reach {final_time_s} s")
    require(resumed_manifest.get("simulation_time_s") == final_time_s,
            f"resumed {label} did not reach {final_time_s} s")
    for name in sorted(campaign.FIELDS):
        lhs = continuous_step / f"{name}.npy"
        rhs = resumed_step / f"{name}.npy"
        require(lhs.read_bytes() == rhs.read_bytes(),
                f"continuous/resumed {label} {name} endpoint differs")


def tracker_records(path: Path) -> list[dict]:
    records: list[dict] = []
    previous_step = -1
    previous_time_s = -math.inf
    for item in sorted((path / "storm_track").glob("track_step_*.json")):
        match = re.fullmatch(r"track_step_([0-9]{9})\.json", item.name)
        require(match is not None,
                f"malformed storm-track history filename: {item}")
        record = campaign.load_json(item)
        completed_step = record.get("completed_step")
        simulation_time_s = record.get("simulation_time_s")
        require(type(completed_step) is int and
                completed_step == int(match.group(1)) and
                completed_step > previous_step,
                f"storm-track filename/step identity is invalid: {item}")
        require(type(simulation_time_s) in (int, float) and
                math.isfinite(float(simulation_time_s)) and
                float(simulation_time_s) > previous_time_s,
                f"storm-track samples are not causally ordered: {item}")
        records.append(record)
        previous_step = completed_step
        previous_time_s = float(simulation_time_s)
    return records


def require_resumed_tracker_contract(records: list[dict]) -> None:
    previous_time_s: float | None = None
    for record in records:
        require(record.get("schema_version") == 12,
                "resumed tracker schema version changed")
        require(record.get("tracking_source") == "synchronized_nested_child",
                "resumed tracker did not use the synchronized refined child")
        require(record.get("detector_coordinate_frame") == "earth_absolute",
                "resumed tracker did not preserve Earth-fixed coordinates")
        require(record.get("algorithm") ==
                "updraft_core_watershed_condensate_rotation_branches_v5",
                "resumed connected-storm detector algorithm changed")
        require(record.get("continuity_algorithm") ==
                "fixed_window_temporal_branch_association_v5",
                "resumed storm-continuity algorithm changed")
        sample_time_s = float(record["simulation_time_s"])
        if previous_time_s is not None:
            require(math.isclose(
                        sample_time_s - previous_time_s, 30.0,
                        rel_tol=0.0, abs_tol=1.0e-9),
                    "resumed tracker lost its 30-second cadence")
        authority = record.get("movement_authority", {})
        campaign.require_authority_contract(authority)
        campaign.require_runtime_scientific_validity(record)
        campaign.require_precipitation_relocation(authority, sample_time_s)
        previous_time_s = sample_time_s


def inspect_resumed(output: Path, final_time_s: int,
                    expected_trace: list[dict]) -> dict:
    campaign.campaign_artifact_binding(output)
    require_no_scientific_invalidity_artifact(output)
    root_manifests, root_extrema = campaign.inspect_history(
        output, "history", (135, 108, 50))
    child_manifests, child_extrema = campaign.inspect_history(
        output / "nests" / "child_0000", "nested_child_history",
        (158, 168, 50))
    require(root_manifests[-1].get("simulation_time_s") == final_time_s,
            f"resumed root output did not reach {final_time_s} s")
    require(child_manifests[-1].get("simulation_time_s") == final_time_s,
            f"resumed child output did not reach {final_time_s} s")
    records = tracker_records(output)
    require(records, "resumed branch published no tracker suffix")
    require_resumed_tracker_contract(records)
    face_motion_sample_time_s = require_post_split_face_motion_before_move(records)
    resumed_trace = campaign.movement_trace(records)
    require(resumed_trace == expected_trace,
            "continuous/resumed tracker and movement-authority suffix differs")
    commits = [
        item for item in records
        if item.get("movement_authority", {}).get("commit_performed") is True
    ]
    require(commits, "resumed suffix did not exercise a child relocation")
    ledgers = {
        name: campaign.finite_ledger(output / name)
        for name in ("budget.csv", "energy_budget.csv", "moisture_budget.csv",
                     "momentum_budget.csv", "thermodynamic_source_budget.csv")
    }
    return {
        "tracker_suffix_samples": len(records),
        "first_restored_face_motion_sample_time_s": face_motion_sample_time_s,
        "movement_times_s": [item.get("simulation_time_s") for item in commits],
        "movement_shifts_parent_cells": [
            item["movement_authority"].get("committed_shift_cells")
            for item in commits
        ],
        "movement_sequences": [
            item["movement_authority"].get("nested_movement_sequence")
            for item in commits
        ],
        "root_field_extrema": root_extrema,
        "child_field_extrema": child_extrema,
        "ledger_rows": ledgers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=campaign.INTERNAL_BINARY)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--continuous-threads", type=int,
                        help=("override the uninterrupted branch worker count; "
                              "unused with --continuous-authority"))
    parser.add_argument("--resume-threads", type=int,
                        help="override the restored branch worker count")
    parser.add_argument(
        "--continuous-authority", type=Path,
        help=("reuse a completed hash-bound uninterrupted campaign directory "
              "and run only its resumed suffix"))
    parser.add_argument("--output", type=Path,
                        help="persistent campaign root; omit for a temporary run")
    parser.add_argument("--final-time", type=int, default=1350,
                        help="continuous/resumed endpoint in seconds")
    args = parser.parse_args()
    continuous_threads = (
        args.continuous_threads
        if args.continuous_threads is not None else args.threads)
    resume_threads = (
        args.resume_threads
        if args.resume_threads is not None else args.threads)
    require(continuous_threads > 0 and resume_threads > 0,
            "branch thread counts must be positive")
    require(args.final_time > 0 and args.final_time % 30 == 0,
            "final time must be a positive tracker-cadence multiple")
    require(args.final_time % CHECKPOINT_INTERVAL_S == 0,
            "final time must be a 450-second publication multiple")
    require(CHECKPOINT_INTERVAL_S < args.final_time,
            "final time must extend beyond the first restart checkpoint")
    binary = args.binary.resolve()
    require(binary.is_file(), f"missing internal qualification binary: {binary}")
    campaign.assert_public_admission_closed()
    campaign.assert_internal_admission_exact(binary)

    temporary = None
    if args.output is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="nimbus_moving_nested_restart_")
        root = Path(temporary.name)
    else:
        root = args.output.resolve()
        require(not root.exists() or not any(root.iterdir()),
                f"output directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)

    continuous_authority_path = (
        args.continuous_authority.resolve()
        if args.continuous_authority is not None else None)
    if continuous_authority_path is not None:
        require(root != continuous_authority_path and
                continuous_authority_path not in root.parents,
                "output directory must not modify the reused continuous authority")

    try:
        if continuous_authority_path is None:
            continuous_output = root / "continuous_output"
            continuous_checkpoints = continuous_output / "checkpoints"
            continuous_wall, _ = run_branch(
                binary, continuous_output, continuous_checkpoints,
                args.final_time, continuous_threads, CHECKPOINT_INTERVAL_S)
            continuous_authority = validate_continuous_authority(
                continuous_output, binary, args.final_time, continuous_wall)
        else:
            continuous_output = continuous_authority_path
            continuous_wall = None
            continuous_authority = validate_continuous_authority(
                continuous_output, binary, args.final_time)

        continuous_report = continuous_authority.report
        continuous = continuous_authority.checkpoints
        split_checkpoint = continuous_authority.split_checkpoint
        split_time_s = split_checkpoint.simulation_time_s
        expected_trace = continuous_authority.expected_trace
        first_movement_time_s = continuous_authority.first_movement_time_s
        first_face_motion_sample_time_s = (
            continuous_authority.first_face_motion_sample_time_s)

        resumed_output = root / "resumed_output"
        resumed_checkpoints = resumed_output / "checkpoints"
        resumed_wall, _ = run_branch(
            binary, resumed_output, resumed_checkpoints, args.final_time,
            resume_threads, CHECKPOINT_INTERVAL_S, split_checkpoint.path)
        resumed = checkpoints(resumed_checkpoints)
        validate_resumed_lineage(
            resumed_output, split_checkpoint, resumed)
        final_checkpoint = require_final_checkpoint_exact(
            continuous, resumed, args.final_time)
        require_endpoint_exact(
            continuous_output, resumed_output, "root", args.final_time)
        require_endpoint_exact(
            continuous_output / "nests" / "child_0000",
            resumed_output / "nests" / "child_0000", "child",
            args.final_time)
        resumed_report = inspect_resumed(
            resumed_output, args.final_time, expected_trace)

        report = {
            "schema": "nimbus.qualification.moving_nested_supercell.restart",
            "schema_version": 2,
            "status": "pass",
            "scope": (
                "internal_cpu_live_storm_reused_continuous_authority_restart_not_selectable"
                if continuous_authority_path is not None else
                "internal_cpu_live_storm_changed_thread_restart_not_selectable"
                if continuous_threads != resume_threads else
                "internal_cpu_live_storm_restart_before_movement_not_selectable"),
            "binary_sha256": campaign.digest(binary),
            "config_sha256": campaign.digest(campaign.CONFIG),
            "split_time_s": split_time_s,
            "final_time_s": args.final_time,
            "first_continuous_movement_time_s": first_movement_time_s,
            "first_restored_face_motion_sample_time_s":
                first_face_motion_sample_time_s,
            "continuous_authority_reused":
                continuous_authority_path is not None,
            "continuous_authority_path": (
                str(continuous_authority_path)
                if continuous_authority_path is not None else None),
            "continuous_threads": (
                None if continuous_authority_path is not None
                else continuous_threads),
            "resume_threads": resume_threads,
            "changed_thread_count": (
                None if continuous_authority_path is not None
                else continuous_threads != resume_threads),
            "continuous_wall_seconds": continuous_wall,
            "resumed_wall_seconds": resumed_wall,
            "checkpoint_exact": True,
            "checkpoint_sidecars_validated": True,
            "resumed_lineage_validated": True,
            "final_checkpoint_completed_steps": final_checkpoint.completed_steps,
            "final_checkpoint_simulation_time_s":
                final_checkpoint.simulation_time_s,
            "final_checkpoint_fnv1a64": final_checkpoint.checkpoint_fnv1a64,
            "root_endpoint_fields_exact": True,
            "child_endpoint_fields_exact": True,
            "external_ledgers": "finite_only",
            "continuous": continuous_report,
            "resumed": resumed_report,
        }
        if args.output is not None:
            (root / "qualification_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[moving-nested-supercell-restart] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)

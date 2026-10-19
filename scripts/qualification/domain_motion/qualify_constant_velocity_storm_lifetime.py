#!/usr/bin/env python3
"""Run or inspect the D2 constant-velocity 4,500 s storm-lifetime candidate."""

from __future__ import annotations

import argparse
import array
import ast
import csv
from collections import Counter
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/constant_velocity_supercell_lifetime.yaml"
CONFIG_SNAPSHOT_NAME = "qualification_input.yaml"
BINARY_DIGEST_NAME = "qualification_binary.sha256"
FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
PUBLICATIONS = tuple(range(11))
FRAME_U_MS = 26.3
FRAME_V_MS = 17.4
REQUIRED_COMPLETE_COLUMNS = (
    "completed_step_energy_step_complete",
    "completed_step_energy_cumulative_complete",
)
SCIENTIFIC_EVERY_STEPS = 10
REFERENCE = {
    "coherent_samples": 148,
    "motion_samples": 146,
    "maximum_output_updraft_ms": 45.8696,
    "maximum_tracker_updraft_ms": 46.4365,
    "maximum_absolute_updraft_helicity_m2s2": 716.938,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON evidence is not an object: {path}")
    return value


def run(binary: Path, output: Path, threads: int) -> float:
    output.mkdir(parents=True, exist_ok=True)
    (output / CONFIG_SNAPSHOT_NAME).write_bytes(CONFIG.read_bytes())
    (output / BINARY_DIGEST_NAME).write_text(digest(binary) + "\n", encoding="ascii")
    started = time.monotonic()
    result = subprocess.run(
        [
            str(binary),
            "--headless",
            f"--config={CONFIG}",
            f"--outdir={output}",
            "--compute-backend=cpu",
            f"--threads={threads}",
            "--log-profile=quiet",
            "--no-timing",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode == 0, f"long moving run exited {result.returncode}\n{result.stdout[-12000:]}")
    return time.monotonic() - started


def npy_extrema(path: Path) -> tuple[tuple[int, ...], float, float]:
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", f"invalid NPY magic: {path}")
        version = stream.read(2)
        require(len(version) == 2, f"truncated NPY version: {path}")
        length_bytes = 2 if version[0] == 1 else 4
        raw_length = stream.read(length_bytes)
        require(len(raw_length) == length_bytes, f"truncated NPY header length: {path}")
        header_length = struct.unpack("<H" if length_bytes == 2 else "<I", raw_length)[0]
        header = ast.literal_eval(stream.read(header_length).decode("latin1").strip())
        require(header.get("descr") in {"<f4", "=f4", "|f4"}, f"unsupported NPY dtype: {path}")
        require(header.get("fortran_order") is False, f"Fortran-order NPY is unsupported: {path}")
        shape = tuple(int(item) for item in header.get("shape", ()))
        payload = stream.read()
    require(len(payload) == math.prod(shape) * 4, f"wrong NPY payload length: {path}")
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little" and header.get("descr") == "<f4":
        values.byteswap()
    require(values and all(math.isfinite(value) for value in values), f"non-finite NPY values: {path}")
    return shape, float(min(values)), float(max(values))


def finite_ledger(path: Path) -> int:
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty ledger: {path}")
    rows = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"missing ledger header: {path}")
        complete_columns = [name for name in REQUIRED_COMPLETE_COLUMNS if name in reader.fieldnames]
        has_completed_energy = bool(complete_columns)
        if has_completed_energy:
            require(
                set(REQUIRED_COMPLETE_COLUMNS).issubset(reader.fieldnames)
                and "completed_step_energy_count" in reader.fieldnames,
                f"incomplete completed-step energy cadence schema in {path}",
            )
        previous_completed_energy_count = 0
        for row in reader:
            rows += 1
            for name, token in row.items():
                if name == "stage" or token is None or token == "":
                    continue
                value = float(token)
                require(math.isfinite(value), f"non-finite {name} in {path}")
            if has_completed_energy:
                completed_energy_count_value = float(row["completed_step_energy_count"])
                completed_energy_count = int(completed_energy_count_value)
                step_complete = float(row["completed_step_energy_step_complete"])
                cumulative_complete = float(row["completed_step_energy_cumulative_complete"])
                require(
                    completed_energy_count_value == completed_energy_count
                    and completed_energy_count == rows // SCIENTIFIC_EVERY_STEPS,
                    f"wrong completed-step energy cadence/count at row {rows} in {path}",
                )
                emitted = completed_energy_count == previous_completed_energy_count + 1
                require(
                    step_complete == (1.0 if emitted else 0.0),
                    f"completed-step energy availability disagrees with its count at row {rows} in {path}",
                )
                require(
                    cumulative_complete == (1.0 if completed_energy_count > 0 else 0.0),
                    f"completed-step cumulative energy availability is inconsistent at row {rows} in {path}",
                )
                previous_completed_energy_count = completed_energy_count
    require(rows > 0, f"ledger has no rows: {path}")
    if complete_columns:
        require(
            rows % SCIENTIFIC_EVERY_STEPS == 0
            and previous_completed_energy_count == rows // SCIENTIFIC_EVERY_STEPS,
            f"completed-step energy cadence did not close at the final row in {path}",
        )
    return rows


def inactive_radiation_ledger(path: Path) -> int:
    zero_columns = {
        "held_atmospheric_power_w",
        "expected_atmospheric_energy_j",
        "booked_radiation_heat_j",
        "booking_residual_j",
        "relative_booking_residual",
        "cumulative_expected_energy_j",
        "cumulative_booked_heat_j",
        "cumulative_booking_residual_j",
    }
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty ledger: {path}")
    rows = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"missing ledger header: {path}")
        require(zero_columns.issubset(reader.fieldnames), "radiation ledger schema is incomplete")
        for row in reader:
            rows += 1
            for name, token in row.items():
                require(token is not None and token != "", f"empty {name} in {path}")
                value = float(token)
                require(math.isfinite(value), f"non-finite {name} in {path}")
                if name in zero_columns:
                    require(value == 0.0, f"inactive radiation booked nonzero {name} in {path}")
    require(rows > 0, f"ledger has no rows: {path}")
    return rows


def close(left: float, right: float, tolerance: float = 1.0e-6) -> bool:
    return abs(left - right) <= tolerance


def inspect(output: Path, binary: Path | None, wall_seconds: float | None) -> dict:
    config_snapshot = output / CONFIG_SNAPSHOT_NAME
    require(config_snapshot.is_file(), "missing qualification input snapshot")
    require(digest(config_snapshot) == digest(CONFIG), "qualification input differs from current config")
    binary_digest_snapshot = output / BINARY_DIGEST_NAME
    require(binary_digest_snapshot.is_file(), "missing qualification binary digest")
    recorded_binary_digest = binary_digest_snapshot.read_text(encoding="ascii").strip()
    require(binary is not None, "qualification inspection requires the executed binary")
    require(recorded_binary_digest == digest(binary), "qualification binary differs from launch digest")
    root_manifest = load_json(output / "nimbus_output.json")
    require(root_manifest.get("publication_state") == "ready", "root publication is not ready")
    require(root_manifest.get("manifest_version") in {6, 7}, "unexpected root manifest version")

    step_dirs = sorted(path for path in output.glob("step_*") if path.is_dir())
    indices = tuple(int(path.name.removeprefix("step_")) for path in step_dirs)
    require(indices == PUBLICATIONS, f"wrong publication sequence: {indices}")
    extrema: dict[str, dict[str, float]] = {}
    maximum_output_updraft_ms = -math.inf
    for publication, step_dir in zip(indices, step_dirs):
        simulation_time_s = publication * 450.0
        manifest = load_json(step_dir / "manifest.json")
        require(manifest.get("manifest_version") in {6, 7}, f"wrong manifest version at {publication}")
        require(manifest.get("publication_state") == "complete", f"incomplete publication {publication}")
        require(manifest.get("simulation_time_s") == simulation_time_s, f"wrong time at {publication}")
        require(set(manifest.get("published_fields", [])) == set(FIELDS), f"wrong fields at {publication}")
        grid = manifest.get("grid", {})
        require(grid.get("dimensions") == {"x": 270, "y": 248, "z": 50}, "wrong moving grid")
        frame = grid.get("horizontal_frame", {})
        require(frame.get("motion_mode") == "constant_velocity", "moving frame mode missing")
        require(frame.get("frame_velocity_earth_ms") == [FRAME_U_MS, FRAME_V_MS], "wrong frame velocity")
        origin = frame.get("grid_origin_earth_m", [])
        require(len(origin) == 2, "missing live Earth origin")
        require(close(float(origin[0]), FRAME_U_MS * simulation_time_s), "wrong live x origin")
        require(close(float(origin[1]), FRAME_V_MS * simulation_time_s), "wrong live y origin")
        validation = load_json(step_dir / "validation_report.json")
        require(validation.get("failed") is False, f"strict validation failed at {publication}")
        for field in FIELDS:
            shape, low, high = npy_extrema(step_dir / f"{field}.npy")
            require(shape == (270, 248, 50), f"wrong {field} shape at {publication}")
            aggregate = extrema.setdefault(field, {"min": low, "max": high})
            aggregate["min"] = min(aggregate["min"], low)
            aggregate["max"] = max(aggregate["max"], high)
            if field == "w":
                maximum_output_updraft_ms = max(maximum_output_updraft_ms, high)

    records = [load_json(path) for path in sorted((output / "storm_track").glob("track_step_*.json"))]
    require(len(records) == 150, f"expected 150 tracker records, found {len(records)}")
    require(records[-1].get("completed_step") == 3000, "tracker does not reach step 3000")
    require(records[-1].get("simulation_time_s") == 4500, "tracker does not reach 4500 s")
    coherent = []
    motion = []
    minimum_boundary_m = math.inf
    minimum_boundary_by_side_m = {
        "west": math.inf,
        "east": math.inf,
        "south": math.inf,
        "north": math.inf,
    }
    continuity_reasons: Counter[str] = Counter()
    branch_overflow_samples = 0
    dominant_overflow_samples = 0
    maximum_tracker_updraft_ms = 0.0
    maximum_absolute_uh_m2s2 = 0.0
    topology_reorganization_samples = 0
    topology_reorganization_indices: list[int] = []
    established_branch_id: int | None = None
    for record_index, record in enumerate(records):
        require(record.get("schema_version") == 12, "wrong tracker schema")
        require(record.get("tracking_source") == "root_grid", "wrong tracker source")
        require(record.get("detector_coordinate_frame") == "grid_relative",
                "wrong tracker coordinate frame")
        require(
            record.get("algorithm") == "updraft_core_watershed_condensate_rotation_branches_v5",
            "wrong core-within-envelope detector algorithm",
        )
        require(record.get("prognostic_state_mutation") is False, "tracker mutated state")
        require(
            record.get("continuity_algorithm") == "fixed_window_temporal_branch_association_v5",
            "wrong tracker continuity algorithm",
        )
        continuity = record.get("continuity", {})
        continuity_reason = str(continuity.get("reason", "missing"))
        continuity_reasons[continuity_reason] += 1
        if established_branch_id is not None:
            require(continuity.get("status") != "lost", "mature storm continuity became lost")
        if record.get("branch_candidate_overflow") is True:
            branch_overflow_samples += 1
            if record.get("status") == "detected":
                dominant_overflow_samples += 1
                require(
                    continuity_reason != "branch_candidate_overflow",
                    "resolved dominant storm was discarded because lower-score branch reporting overflowed",
                )
        frame = record.get("horizontal_frame", {})
        require(frame.get("motion_mode") == "constant_velocity", "tracker lost moving-frame identity")
        simulation_time_s = float(record.get("simulation_time_s", -1.0))
        origin = frame.get("grid_origin_earth_m", {})
        require(close(float(origin.get("x", math.nan)), FRAME_U_MS * simulation_time_s), "tracker x origin mismatch")
        require(close(float(origin.get("y", math.nan)), FRAME_V_MS * simulation_time_s), "tracker y origin mismatch")
        if record.get("selected_feature_available") is True:
            coherent.append(record)
            require(
                record.get("selected_feature_role") == "continuity_selected_candidate",
                "selected storm fields have ambiguous provenance",
            )
            branch_id = int(continuity.get("branch_id", 0))
            if established_branch_id is None:
                established_branch_id = branch_id
                require(established_branch_id > 0, "selected storm lacks a branch identity")
            require(branch_id == established_branch_id, "selected storm changed branch identity")
            if continuity.get("topology_reorganization") is True:
                topology_reorganization_samples += 1
                topology_reorganization_indices.append(record_index)
                require(
                    continuity.get("reorganization_overlap_basis") == "exact_watershed_partition_cells",
                    "authority reorganization used summary-box rather than exact partition overlap",
                )
                require(continuity.get("status") == "tentative", "reorganization claimed mature motion")
                require(record.get("motion_available") is False, "reorganization invented storm translation")
                require(continuity.get("history_samples") == 1, "reorganization did not reset motion history")
                require(
                    float(continuity.get("reorganization_overlap_fraction", 0.0)) >= 0.5,
                    "reorganization lacks majority prior-partition ownership",
                )
            require(record.get("boundary_contact") is not True, "storm contacted the moving boundary")
            require(record.get("inner_margin_breached") is not True, "storm breached the moving margin")
            distance = record.get("distance_to_physical_boundary_m", {}).get("nearest")
            require(distance is not None, "coherent storm lacks boundary distance")
            minimum_boundary_m = min(minimum_boundary_m, float(distance))
            for side in minimum_boundary_by_side_m:
                side_distance = record.get("distance_to_physical_boundary_m", {}).get(side)
                require(side_distance is not None, f"coherent storm lacks {side} boundary distance")
                minimum_boundary_by_side_m[side] = min(
                    minimum_boundary_by_side_m[side], float(side_distance)
                )
            earth = record.get("core_centroid_m", {})
            grid = record.get("grid_core_centroid_m", {})
            require(close(float(earth.get("x", math.nan)), float(grid.get("x", math.nan)) + float(origin["x"])),
                    "tracker core x frame transform is inconsistent")
            require(close(float(earth.get("y", math.nan)), float(grid.get("y", math.nan)) + float(origin["y"])),
                    "tracker core y frame transform is inconsistent")
            maximum_tracker_updraft_ms = max(
                maximum_tracker_updraft_ms, float(record.get("maximum_updraft_ms") or 0.0)
            )
            maximum_absolute_uh_m2s2 = max(
                maximum_absolute_uh_m2s2,
                float(record.get("rotation", {}).get("maximum_absolute_updraft_helicity_m2s2") or 0.0),
            )
        else:
            require(record.get("selected_feature_role") is None, "unavailable storm has a selected role")
            require(
                record.get("distance_to_physical_boundary_m", {}).get("nearest") is None,
                "unassociated snapshot substituted a boundary envelope",
            )
            require(
                record.get("boundary_contact") is None and record.get("inner_margin_breached") is None,
                "unassociated snapshot invented selected-storm boundary state",
            )
        if record.get("motion_available") is True:
            motion.append(record)

    first_selected_index = next(
        (index for index, record in enumerate(records) if record.get("selected_feature_available") is True),
        None,
    )
    require(first_selected_index is not None, "moving storm never established a selected feature")
    mature_records = records[first_selected_index:]
    mature_selected_fraction = len(coherent) / len(mature_records)
    selected_motion_fraction = len(motion) / len(coherent)
    require(
        mature_selected_fraction >= 0.95,
        "moving storm selection was unavailable for more than 5% of its mature lifetime",
    )
    require(
        selected_motion_fraction >= 0.85,
        "moving storm lacked a mature motion estimate for more than 15% of selected samples",
    )

    maximum_consecutive_unselected_samples = 0
    consecutive_unselected_samples = 0
    for record in mature_records:
        if record.get("selected_feature_available") is True:
            consecutive_unselected_samples = 0
        else:
            consecutive_unselected_samples += 1
            maximum_consecutive_unselected_samples = max(
                maximum_consecutive_unselected_samples,
                consecutive_unselected_samples,
            )
    require(
        maximum_consecutive_unselected_samples <= 2,
        "mature storm association was unavailable for more than two consecutive samples",
    )

    maximum_reorganization_recovery_samples = 0
    for event_index in topology_reorganization_indices:
        recovery_offset = next(
            (
                offset
                for offset, record in enumerate(records[event_index + 1:event_index + 5], start=1)
                if record.get("motion_available") is True
            ),
            None,
        )
        require(
            recovery_offset is not None,
            "storm motion did not recover within four samples of a topology reorganization",
        )
        maximum_reorganization_recovery_samples = max(
            maximum_reorganization_recovery_samples,
            recovery_offset,
        )
    require(established_branch_id == 1, "authority did not retain its original mature storm branch")
    require(branch_overflow_samples == 0, "64-candidate storm history snapshot was still incomplete")
    require(records[-1].get("status") in {"detected", "ambiguous"}, "storm is absent at 4500 s")
    require(records[-1].get("selected_feature_available") is True,
            "storm lacks a continuity-selected feature at 4500 s")
    require(records[-1].get("motion_available") is True,
            "storm lacks a mature motion estimate at 4500 s")
    require(minimum_boundary_m >= 10000.0, "storm violated the moving-domain 10 km margin")
    require(0.5 * REFERENCE["maximum_output_updraft_ms"] <= maximum_output_updraft_ms
            <= 1.5 * REFERENCE["maximum_output_updraft_ms"], "output updraft left D1 intensity envelope")
    require(0.5 * REFERENCE["maximum_tracker_updraft_ms"] <= maximum_tracker_updraft_ms
            <= 1.5 * REFERENCE["maximum_tracker_updraft_ms"], "tracker updraft left D1 intensity envelope")
    require(0.35 * REFERENCE["maximum_absolute_updraft_helicity_m2s2"] <= maximum_absolute_uh_m2s2
            <= 2.0 * REFERENCE["maximum_absolute_updraft_helicity_m2s2"], "UH left D1 intensity envelope")

    checkpoints = sorted((output / "checkpoints").glob("checkpoint_step_*.nimbus"))
    checkpoint_steps = tuple(int(path.stem.removeprefix("checkpoint_step_")) for path in checkpoints)
    require(checkpoint_steps == (1500, 3000), f"wrong checkpoint cadence: {checkpoint_steps}")
    ledger_rows = {
        name: finite_ledger(output / name)
        for name in (
            "budget.csv",
            "energy_budget.csv",
            "moisture_budget.csv",
            "momentum_budget.csv",
            "thermodynamic_source_budget.csv",
        )
    }
    ledger_rows["radiation_budget.csv"] = inactive_radiation_ledger(output / "radiation_budget.csv")

    return {
        "schema": "nimbus.qualification.constant_velocity_storm_lifetime",
        "schema_version": 2,
        "status": "pass",
        "scope": "d2_long_lifetime_against_committed_d1_summary_envelopes",
        "inspector_sha256": digest(Path(__file__)),
        "config_sha256": digest(config_snapshot),
        "binary_sha256": recorded_binary_digest,
        "wall_seconds": wall_seconds,
        "grid": {"nx": 270, "ny": 248, "nz": 50, "horizontal_cells": 270 * 248},
        "stationary_reference_horizontal_cells": 267 * 208,
        "horizontal_cell_ratio_to_stationary_reference": (270 * 248) / (267 * 208),
        "horizontal_cell_delta_fraction": (270 * 248) / (267 * 208) - 1.0,
        "storm_samples": len(records),
        "coherent_samples": len(coherent),
        "motion_available_samples": len(motion),
        "mature_selected_fraction": mature_selected_fraction,
        "selected_motion_fraction": selected_motion_fraction,
        "maximum_consecutive_unselected_samples": maximum_consecutive_unselected_samples,
        "maximum_reorganization_recovery_samples": maximum_reorganization_recovery_samples,
        "minimum_storm_boundary_distance_m": minimum_boundary_m,
        "minimum_storm_boundary_distance_by_side_m": minimum_boundary_by_side_m,
        "branch_candidate_overflow_samples": branch_overflow_samples,
        "dominant_branch_overflow_samples": dominant_overflow_samples,
        "topology_reorganization_samples": topology_reorganization_samples,
        "continuity_reason_counts": dict(sorted(continuity_reasons.items())),
        "maximum_output_updraft_ms": maximum_output_updraft_ms,
        "maximum_tracker_updraft_ms": maximum_tracker_updraft_ms,
        "maximum_absolute_updraft_helicity_m2s2": maximum_absolute_uh_m2s2,
        "d1_reference": REFERENCE,
        "field_extrema": extrema,
        "checkpoint_steps": list(checkpoint_steps),
        "ledger_rows": ledger_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin" / "nimbus_sim_cpu")
    parser.add_argument("--output", type=Path, help="inspect an existing completed output")
    parser.add_argument(
        "--wall-seconds",
        type=float,
        help="measured launch-to-completion wall time for an existing output",
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    require(binary.is_file(), f"missing CPU binary: {binary}")
    if args.output:
        require(
            args.wall_seconds is None or (math.isfinite(args.wall_seconds) and args.wall_seconds > 0.0),
            "--wall-seconds must be finite and positive",
        )
        report = inspect(args.output.resolve(), binary, args.wall_seconds)
    else:
        require(args.wall_seconds is None, "--wall-seconds is only valid with --output")
        with tempfile.TemporaryDirectory(prefix="nimbus_constant_velocity_lifetime_") as temporary:
            output = Path(temporary)
            elapsed = run(binary, output, args.threads)
            report = inspect(output, binary, elapsed)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[constant-velocity-storm-lifetime] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)

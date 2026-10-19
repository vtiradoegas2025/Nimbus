#!/usr/bin/env python3
"""Run or inspect the internal moving-nest supercell campaign."""

from __future__ import annotations

import argparse
import array
import ast
import csv
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests" / "qualification" / "fixtures" / "moving_nested_supercell_lifetime.yaml"
INTERNAL_BINARY = ROOT / "bin" / "nimbus_sim_internal_qualification"
PUBLIC_BINARY = ROOT / "bin" / "nimbus_sim"
REFERENCE_EVIDENCE = ROOT / "tests/fixtures/reference/domain_history/constant_velocity_storm_lifetime.json"
REFERENCE_CONFIG = ROOT / "tests/fixtures/configs/qualification/constant_velocity_supercell_lifetime.yaml"
VOLUME_FIELDS = {"p", "qc", "qr", "qv", "rho", "theta", "u", "v", "w"}
PLANAR_FIELDS = {"cold_pool_boundary", "precip_rate", "accumulated_rainfall"}
FIELDS = VOLUME_FIELDS | PLANAR_FIELDS
REFERENCE = {
    "maximum_output_updraft_ms": 45.8696,
    "maximum_tracker_updraft_ms": 46.4365,
    "maximum_absolute_updraft_helicity_m2s2": 716.938,
}
CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M = (3000.0, 1800.0)
CHILD_PHYSICAL_LOWER_OFFSET_M = (600.0, 600.0)
CHILD_PHYSICAL_EXTENT_M = (93600.0, 99600.0)
PARENT_SPACING_M = (1200.0, 1200.0)
INNER_MARGIN_M = 15600.0
RELEASE_MARGIN_M = 16800.0
RELAXATION_WIDTH_M = 4800.0
MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v14"
TRACKING_INTERVAL_S = 30.0
OBSERVED_MAX_ENVELOPE_SPAN_M = (58200.0, 64200.0)
OBSERVED_MAX_ONE_SAMPLE_ADVANCE_M = 10200.0
CHILD_CELL_RESERVE_M = 600.0
PARENT_LATTICE_PHASE_ALLOWANCE_M = 1200.0
MAXIMUM_SHIFT_CELLS = 14


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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON evidence {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON evidence is not an object: {path}")
    return value


def reference_binding() -> dict:
    evidence = load_json(REFERENCE_EVIDENCE)
    require(evidence.get("schema") ==
            "nimbus.qualification.constant_velocity_storm_lifetime",
            "moving-nest intensity reference has the wrong schema")
    require(evidence.get("schema_version") == 2,
            "moving-nest intensity reference has the wrong schema version")
    require(evidence.get("status") == "pass",
            "moving-nest intensity reference is not a passing record")
    require(evidence.get("config_sha256") == digest(REFERENCE_CONFIG),
            "moving-nest intensity reference is detached from its committed config")
    recorded_summary = evidence.get("d1_reference", {})
    require(isinstance(recorded_summary, dict) and all(
                recorded_summary.get(name) == value
                for name, value in REFERENCE.items()),
            "moving-nest intensity bounds changed without a bound reference record")
    binary_sha256 = str(evidence.get("binary_sha256", ""))
    require(re.fullmatch(r"[0-9a-f]{64}", binary_sha256) is not None,
            "moving-nest intensity reference lacks a valid binary digest")
    return {
        "evidence": str(REFERENCE_EVIDENCE.relative_to(ROOT)),
        "evidence_sha256": digest(REFERENCE_EVIDENCE),
        "config": str(REFERENCE_CONFIG.relative_to(ROOT)),
        "config_sha256": digest(REFERENCE_CONFIG),
        "binary_sha256": binary_sha256,
        "summary": REFERENCE,
    }


def child_envelope_distances(envelope: dict, storage_origin: tuple[float, float]) -> dict[str, float]:
    lower_x = storage_origin[0] + CHILD_PHYSICAL_LOWER_OFFSET_M[0]
    lower_y = storage_origin[1] + CHILD_PHYSICAL_LOWER_OFFSET_M[1]
    upper_x = lower_x + CHILD_PHYSICAL_EXTENT_M[0]
    upper_y = lower_y + CHILD_PHYSICAL_EXTENT_M[1]
    distances = {
        "west": float(envelope.get("west", math.nan)) - lower_x,
        "east": upper_x - float(envelope.get("east", math.nan)),
        "south": float(envelope.get("south", math.nan)) - lower_y,
        "north": upper_y - float(envelope.get("north", math.nan)),
    }
    require(all(math.isfinite(value) for value in distances.values()),
            "selected storm lacks a finite child-relative envelope")
    return distances


def attainable_release_margin(first_m: float, second_m: float, spacing_m: float) -> float:
    centered_error = abs(math.remainder(first_m - second_m, 2.0 * spacing_m))
    best_symmetric_margin = 0.5 * (first_m + second_m - centered_error)
    return max(INNER_MARGIN_M, min(RELEASE_MARGIN_M, best_symmetric_margin))


def release_box_recovered(distances: dict[str, float]) -> bool:
    x_release = attainable_release_margin(
        distances["west"], distances["east"], PARENT_SPACING_M[0])
    y_release = attainable_release_margin(
        distances["south"], distances["north"], PARENT_SPACING_M[1])
    return (distances["west"] >= x_release and
            distances["east"] >= x_release and
            distances["south"] >= y_release and
            distances["north"] >= y_release)


def require_distance_match(published: dict, expected: dict[str, float]) -> None:
    for side, value in expected.items():
        candidate = float(published.get(side, math.nan))
        require(math.isfinite(candidate) and math.isclose(candidate, value, abs_tol=1.0e-6),
                f"published {side} child clearance disagrees with Earth-frame geometry")


def require_authority_contract(authority: dict) -> None:
    require(authority.get("algorithm") == MOVEMENT_ALGORITHM,
            "tracker lost the v14 parent-stencil-aware containment authority")
    require(authority.get("mode") == "live_one_way_nested_child",
            "tracker lost nested movement ownership")
    require(authority.get("enabled") is True,
            "nested movement authority was not enabled")
    require(authority.get("nested_movement_enabled") is True,
            "nested movement authority was disabled")
    require(authority.get("recycling_enabled") is False,
            "whole-domain recycling was enabled")
    require(float(authority.get("inner_margin_m", math.nan)) == INNER_MARGIN_M,
            "movement inner margin changed")
    require(float(authority.get("release_margin_m", math.nan)) == RELEASE_MARGIN_M,
            "movement release margin changed")
    require(authority.get("maximum_shift_cells") == MAXIMUM_SHIFT_CELLS,
            "movement cap changed")
    require(float(authority.get("cooldown_s", math.nan)) == 300.0,
            "movement cooldown changed")
    require(authority.get("confirmation_samples_required") == 2,
            "movement confirmation contract changed")
    bounds = authority.get("legal_parent_shift_bounds_cells")
    require(isinstance(bounds, dict) and bounds.get("available") is True,
            "movement authority lacks live parent-stencil bounds")
    values = [bounds.get(name) for name in (
        "minimum_i", "maximum_i", "minimum_j", "maximum_j")]
    require(all(type(value) is int for value in values) and
            values[0] <= 0 <= values[1] and
            values[2] <= 0 <= values[3],
            "movement authority published malformed parent-stencil bounds")


def require_runtime_scientific_validity(record: dict) -> None:
    validity = record.get("moving_nest_scientific_validity")
    require(isinstance(validity, dict),
            "tracker lacks runtime moving-nest scientific-validity ownership")
    require(validity.get("owner") == "physical_child_trusted_interior_v1",
            "tracker scientific-validity owner changed")
    require(validity.get("status") != "invalid",
            "runtime marked the moving child scientifically invalid")
    require(validity.get("emergency_diagnostic_only") is False,
            "an emergency diagnostic was treated as successful science")
    widths = validity.get("relaxation_width_m", {})
    require(float(widths.get("x", math.nan)) == RELAXATION_WIDTH_M and
            float(widths.get("y", math.nan)) == RELAXATION_WIDTH_M,
            "runtime trusted-interior width differs from physical relaxation")
    selected = record.get("selected_feature_available") is True
    if not selected:
        require(validity.get("status") == "not_assessed" and
                validity.get("reason") == "no_selected_feature" and
                validity.get("trusted_interior") is None,
                "unselected tracker sample invented trusted-interior evidence")
        return
    require(validity.get("trusted_interior") is True,
            "selected storm was not inside the trusted child interior")
    authority = record.get("movement_authority", {})
    if authority.get("status") == "pending_confirmation":
        require(validity.get("status") == "valid_pending_relocation" and
                validity.get("reason") ==
                    "confirmation_pending_outside_relaxation",
                "ordinary pending relocation lacks bounded scientific status")
    else:
        require(validity.get("status") == "valid" and
                validity.get("reason") == "none",
                "selected tracker sample lacks valid runtime ownership")


def require_uninterrupted_authoritative_track(record: dict,
                                              established: bool) -> bool:
    continuity = record.get("continuity", {})
    current_authoritative = (
        record.get("selected_feature_available") is True and
        continuity.get("status") == "tracking")
    if established:
        require(current_authoritative,
                "mature tracker authority was interrupted")
    return established or current_authoritative


def require_precipitation_relocation(authority: dict,
                                     sample_time_s: float) -> None:
    report = authority.get("precipitation_relocation")
    committed = authority.get("commit_performed") is True
    if not committed:
        require(report is None,
                "uncommitted movement invented precipitation relocation")
        return
    require(isinstance(report, dict),
            "committed movement lacks precipitation relocation provenance")
    require(report.get("owner") == "earth_fixed_surface_history_v1" and
            report.get("schema_version") == 1 and
            report.get("active") is True,
            "committed movement lost active Earth-fixed precipitation ownership")
    require(float(report.get("sample_time_s", math.nan)) == sample_time_s,
            "precipitation relocation sample time differs from movement")
    require(int(report.get("newly_admitted_columns", 0)) > 0,
            "precipitation relocation admitted no new columns")
    water_names = (
        "previous_child_accumulated_water_m3",
        "discarded_child_accumulated_water_m3",
        "retained_child_accumulated_water_m3",
        "newly_admitted_parent_accumulated_water_m3",
        "committed_child_accumulated_water_m3",
    )
    rate_names = (
        "previous_child_precipitation_volume_rate_m3s",
        "discarded_child_precipitation_volume_rate_m3s",
        "retained_child_precipitation_volume_rate_m3s",
        "newly_admitted_parent_precipitation_volume_rate_m3s",
        "committed_child_precipitation_volume_rate_m3s",
    )
    water = [float(report.get(name, math.nan)) for name in water_names]
    rates = [float(report.get(name, math.nan)) for name in rate_names]
    require(all(math.isfinite(value) and value >= 0.0
                for value in water + rates),
            "precipitation relocation contains invalid inventory")
    for values, label in ((water, "accumulated water"),
                          (rates, "precipitation rate")):
        previous, discarded, retained, newly_admitted, committed_value = values
        require(math.isclose(previous, discarded + retained,
                             rel_tol=1.0e-12, abs_tol=1.0e-12),
                f"precipitation relocation does not partition prior {label}")
        require(math.isclose(committed_value, retained + newly_admitted,
                             rel_tol=1.0e-12, abs_tol=1.0e-12),
                f"precipitation relocation does not close committed {label}")


def committed_origin_and_cumulative(authority: dict,
                                    pre_shift_origin: tuple[float, float],
                                    cumulative_i: int,
                                    cumulative_j: int) -> tuple[tuple[float, float], int, int]:
    shift = authority.get("committed_shift_cells", {})
    shift_i = int(shift.get("i", 0))
    shift_j = int(shift.get("j", 0))
    require(abs(shift_i) <= MAXIMUM_SHIFT_CELLS and
            abs(shift_j) <= MAXIMUM_SHIFT_CELLS,
            "committed movement exceeded its cap")
    require(shift_i != 0 or shift_j != 0,
            "movement transaction committed a zero shift")
    require(authority.get("requested_shift_cells") == shift,
            "committed movement differs from its authority request")
    post_origin = authority.get("post_shift_child_grid_origin_t0_earth_m", {})
    post_x = float(post_origin.get("x", math.nan))
    post_y = float(post_origin.get("y", math.nan))
    require(math.isfinite(post_x) and math.isfinite(post_y),
            "committed movement lacks a finite child Earth origin")
    require(math.isclose(
                post_x, pre_shift_origin[0] + shift_i * PARENT_SPACING_M[0],
                abs_tol=1.0e-6) and
            math.isclose(
                post_y, pre_shift_origin[1] + shift_j * PARENT_SPACING_M[1],
                abs_tol=1.0e-6),
            "committed child origin disagrees with its integer-parent shift")
    next_i = cumulative_i + shift_i
    next_j = cumulative_j + shift_j
    require(authority.get("cumulative_committed_shift_cells") == {
                "i": next_i, "j": next_j},
            "movement cumulative displacement is not causal")
    return (post_x, post_y), next_i, next_j


def require_safe_movement_reasons(reasons) -> None:
    hard_geometry_failures = {
        "shift_limit_insufficient",
        "opposite_release_margin_insufficient",
        "opposite_faces_breached",
        "emergency_containment",
    }
    require(not hard_geometry_failures.intersection(reasons),
            "movement authority encountered boundary-sensitive child-envelope geometry")


def selected_envelope_geometry(record: dict,
                               pre_shift_origin: tuple[float, float],
                               post_shift_origin: tuple[float, float]) -> tuple[dict[str, float], dict[str, float], float, float]:
    envelope = record.get("envelope_faces_m", {})
    envelope_west = float(envelope.get("west", math.nan))
    envelope_east = float(envelope.get("east", math.nan))
    envelope_south = float(envelope.get("south", math.nan))
    envelope_north = float(envelope.get("north", math.nan))
    require(all(math.isfinite(value) for value in (
                envelope_west, envelope_east,
                envelope_south, envelope_north)) and
            envelope_east >= envelope_west and
            envelope_north >= envelope_south,
            "selected storm has an invalid Earth-frame envelope")
    pre_shift_distances = child_envelope_distances(
        envelope, pre_shift_origin)
    require_distance_match(
        record.get("distance_to_physical_boundary_m", {}),
        pre_shift_distances)
    require(min(pre_shift_distances.values()) >= RELAXATION_WIDTH_M,
            "selected storm entered the child boundary-relaxation zone")
    post_shift_distances = child_envelope_distances(
        envelope, post_shift_origin)
    require(min(post_shift_distances.values()) >= RELAXATION_WIDTH_M,
            "selected storm remained in the child boundary-relaxation zone after movement")
    if record.get("movement_authority", {}).get("commit_performed") is True:
        require(release_box_recovered(post_shift_distances),
                "committed movement did not restore its grid-attainable release box")
    return (
        pre_shift_distances,
        post_shift_distances,
        envelope_east - envelope_west,
        envelope_north - envelope_south,
    )


def envelope_face_coordinate_velocities(previous_record: dict,
                                        current_record: dict) -> dict[str, float]:
    previous_time = float(previous_record.get("simulation_time_s", math.nan))
    current_time = float(current_record.get("simulation_time_s", math.nan))
    elapsed_s = current_time - previous_time
    require(math.isfinite(elapsed_s) and elapsed_s > 0.0,
            "selected envelope samples are not causally ordered")
    previous = previous_record.get("envelope_faces_m", {})
    current = current_record.get("envelope_faces_m", {})
    values = {
        "west": (float(current.get("west", math.nan)) -
                 float(previous.get("west", math.nan))) / elapsed_s,
        "east": (float(current.get("east", math.nan)) -
                 float(previous.get("east", math.nan))) / elapsed_s,
        "south": (float(current.get("south", math.nan)) -
                  float(previous.get("south", math.nan))) / elapsed_s,
        "north": (float(current.get("north", math.nan)) -
                  float(previous.get("north", math.nan))) / elapsed_s,
    }
    require(all(math.isfinite(value) for value in values.values()),
            "selected envelope face propagation is nonfinite")
    return values


def outward_envelope_face_speeds(previous_record: dict,
                                 current_record: dict) -> dict[str, float]:
    values = envelope_face_coordinate_velocities(
        previous_record, current_record)
    return {
        "west": max(0.0, -values["west"]),
        "east": max(0.0, values["east"]),
        "south": max(0.0, -values["south"]),
        "north": max(0.0, values["north"]),
    }


def reported_envelope_face_motion(record: dict) -> dict[str, float] | None:
    motion = record.get("envelope_face_motion")
    require(isinstance(motion, dict),
            "tracker record lacks envelope-face motion provenance")
    available = motion.get("available")
    require(isinstance(available, bool),
            "envelope-face motion availability is not boolean")
    names = {
        "west": "west_coordinate_velocity_ms",
        "east": "east_coordinate_velocity_ms",
        "south": "south_coordinate_velocity_ms",
        "north": "north_coordinate_velocity_ms",
    }
    if not available:
        require(motion.get("interval_s") is None and all(
                    motion.get(name) is None for name in names.values()),
                "unavailable envelope-face motion contains invented values")
        return None
    interval_s = float(motion.get("interval_s", math.nan))
    require(math.isfinite(interval_s) and interval_s > 0.0,
            "envelope-face motion interval is not finite and positive")
    require(math.isclose(interval_s, TRACKING_INTERVAL_S, abs_tol=1.0e-9),
            "envelope-face motion interval differs from tracker cadence")
    coordinate_velocity = {
        side: float(motion.get(name, math.nan))
        for side, name in names.items()
    }
    require(all(math.isfinite(value) for value in coordinate_velocity.values()),
            "envelope-face coordinate velocity is nonfinite")
    return coordinate_velocity


def require_confirmation_headroom(maximum_outward_speed_ms: float) -> float:
    maximum_advance_m = maximum_outward_speed_ms * TRACKING_INTERVAL_S
    require(maximum_advance_m <= OBSERVED_MAX_ONE_SAMPLE_ADVANCE_M,
            "observed envelope-face propagation exceeded the footprint sizing authority")
    require(maximum_advance_m <= INNER_MARGIN_M - RELAXATION_WIDTH_M,
            "observed envelope-face propagation can cross the relaxation headroom during confirmation")
    return maximum_advance_m


def movement_geometry_contract() -> dict:
    expected_inner = (
        RELAXATION_WIDTH_M + OBSERVED_MAX_ONE_SAMPLE_ADVANCE_M +
        CHILD_CELL_RESERVE_M)
    require(INNER_MARGIN_M == expected_inner,
            "movement inner margin is detached from relaxation and morphology evidence")
    require(RELEASE_MARGIN_M == INNER_MARGIN_M + max(PARENT_SPACING_M),
            "movement release margin is not one parent cell beyond the trigger")
    unrounded_x_m = (
        OBSERVED_MAX_ENVELOPE_SPAN_M[0] + 2.0 * RELEASE_MARGIN_M +
        PARENT_LATTICE_PHASE_ALLOWANCE_M)
    required_x_m = (
        math.ceil(unrounded_x_m / PARENT_SPACING_M[0]) *
        PARENT_SPACING_M[0])
    unrounded_y_m = (
        OBSERVED_MAX_ENVELOPE_SPAN_M[1] + 2.0 * RELEASE_MARGIN_M +
        PARENT_LATTICE_PHASE_ALLOWANCE_M)
    required_y_m = (
        math.ceil(unrounded_y_m / PARENT_SPACING_M[1]) *
        PARENT_SPACING_M[1])
    require(CHILD_PHYSICAL_EXTENT_M == (required_x_m, required_y_m),
            "child footprint is detached from observed envelope and release geometry")
    minimum_clean_clearance_m = RELAXATION_WIDTH_M + CHILD_CELL_RESERVE_M
    required_clean_shift_cells = max(
        math.ceil(
            (RELEASE_MARGIN_M - minimum_clean_clearance_m) /
            PARENT_SPACING_M[0]),
        math.ceil(
            (RELEASE_MARGIN_M - minimum_clean_clearance_m) /
            PARENT_SPACING_M[1]))
    required_emergency_shift_cells = max(
        math.ceil(RELEASE_MARGIN_M / PARENT_SPACING_M[0]),
        math.ceil(RELEASE_MARGIN_M / PARENT_SPACING_M[1]))
    require(MAXIMUM_SHIFT_CELLS >= required_emergency_shift_cells,
            "movement cap cannot recover a sub-cell emergency approach")
    require(MAXIMUM_SHIFT_CELLS >= required_clean_shift_cells,
            "movement cap cannot recover the worst confirmed clean envelope")
    return {
        "observed_maximum_envelope_span_m": {
            "x": OBSERVED_MAX_ENVELOPE_SPAN_M[0],
            "y": OBSERVED_MAX_ENVELOPE_SPAN_M[1],
        },
        "observed_maximum_one_sample_advance_m":
            OBSERVED_MAX_ONE_SAMPLE_ADVANCE_M,
        "child_cell_reserve_m": CHILD_CELL_RESERVE_M,
        "parent_lattice_phase_allowance_m":
            PARENT_LATTICE_PHASE_ALLOWANCE_M,
        "inner_margin_m": INNER_MARGIN_M,
        "release_margin_m": RELEASE_MARGIN_M,
        "required_clean_shift_cells": required_clean_shift_cells,
        "required_maximum_shift_cells": required_emergency_shift_cells,
        "physical_child_extent_m": {
            "x": CHILD_PHYSICAL_EXTENT_M[0],
            "y": CHILD_PHYSICAL_EXTENT_M[1],
        },
    }


def movement_trace(records: list[dict]) -> list[dict]:
    """Return deterministic tracker/movement state for restart comparison."""
    trace: list[dict] = []
    for record in records:
        authority = record.get("movement_authority", {})
        continuity = record.get("continuity", {})
        trace.append({
            "simulation_time_s": record.get("simulation_time_s"),
            "completed_step": record.get("completed_step"),
            "selected_feature_available": record.get("selected_feature_available"),
            "motion_available": record.get("motion_available"),
            "centroid_m": record.get("centroid_m"),
            "core_centroid_m": record.get("core_centroid_m"),
            "envelope_faces_m": record.get("envelope_faces_m"),
            "envelope_face_motion": record.get("envelope_face_motion"),
            "distance_to_physical_boundary_m":
                record.get("distance_to_physical_boundary_m"),
            "boundary_contact": record.get("boundary_contact"),
            "inner_margin_breached": record.get("inner_margin_breached"),
            "moving_nest_scientific_validity":
                record.get("moving_nest_scientific_validity"),
            "continuity_status": continuity.get("status"),
            "continuity_reason": continuity.get("reason"),
            "continuity_branch_id": continuity.get("branch_id"),
            "continuity_selected_candidate_available":
                continuity.get("selected_candidate_available"),
            "continuity_selected_candidate_rank":
                continuity.get("selected_candidate_rank"),
            "continuity_selected_component_label":
                continuity.get("selected_component_label"),
            "continuity_topology": continuity.get("topology"),
            "continuity_pending_topology": continuity.get("pending_topology"),
            "continuity_pending_topology_samples":
                continuity.get("pending_topology_samples"),
            "continuity_topology_event": continuity.get("topology_event"),
            "continuity_topology_event_id": continuity.get("topology_event_id"),
            "continuity_topology_reorganization":
                continuity.get("topology_reorganization"),
            "continuity_history_samples": continuity.get("history_samples"),
            "continuity_missed_samples": continuity.get("missed_samples"),
            "movement_algorithm": authority.get("algorithm"),
            "movement_status": authority.get("status"),
            "movement_reason": authority.get("reason"),
            "request_ready": authority.get("request_ready"),
            "requested_shift_cells": authority.get("requested_shift_cells"),
            "confirmation_count": authority.get("confirmation_count"),
            "commit_performed": authority.get("commit_performed"),
            "committed_shift_cells": authority.get("committed_shift_cells"),
            "nested_movement_sequence": authority.get("nested_movement_sequence"),
            "precipitation_relocation":
                authority.get("precipitation_relocation"),
            "post_shift_child_grid_origin_t0_earth_m":
                authority.get("post_shift_child_grid_origin_t0_earth_m"),
            "armed": authority.get("armed"),
            "committed_move_available": authority.get("committed_move_available"),
            "cumulative_committed_shift_cells":
                authority.get("cumulative_committed_shift_cells"),
        })
    return trace


def campaign_artifact_binding(output: Path) -> tuple[Path, str, dict]:
    movement_geometry_contract()
    config_snapshot = output / "qualification_input.yaml"
    require(config_snapshot.is_file(), "campaign lacks its immutable config snapshot")
    require(digest(config_snapshot) == digest(CONFIG),
            "campaign config snapshot does not match the admitted moving-nest fixture")
    binary_digest_path = output / "qualification_binary.sha256"
    require(binary_digest_path.is_file(), "campaign lacks its binary digest")
    recorded_binary_digest = binary_digest_path.read_text(encoding="utf-8").strip()
    require(re.fullmatch(r"[0-9a-f]{64}", recorded_binary_digest) is not None,
            "campaign binary digest is malformed")
    return config_snapshot, recorded_binary_digest, reference_binding()


def npy_float32_extrema(path: Path) -> tuple[tuple[int, ...], float, float]:
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
        shape = tuple(int(value) for value in header.get("shape", ()))
        payload = stream.read()
    require(len(payload) == math.prod(shape) * 4, f"wrong NPY payload length: {path}")
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little" and header.get("descr") == "<f4":
        values.byteswap()
    require(values and all(math.isfinite(value) for value in values), f"non-finite NPY payload: {path}")
    return shape, float(min(values)), float(max(values))


def finite_ledger(path: Path) -> int:
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty ledger: {path}")
    rows = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"missing ledger header: {path}")
        for row in reader:
            rows += 1
            for key, token in row.items():
                if token in {None, ""}:
                    continue
                try:
                    value = float(token)
                except ValueError:
                    continue
                require(math.isfinite(value), f"non-finite ledger value {key} in {path}")
    require(rows > 0, f"ledger has no rows: {path}")
    return rows


def assert_public_admission_closed() -> None:
    rejected_config = subprocess.run(
        [str(PUBLIC_BINARY), "--validate-only", "--config", str(CONFIG)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(rejected_config.returncode != 0, "public binary admitted moving_nest YAML")
    require("adaptive motion mode is not implemented" in rejected_config.stdout,
            "public YAML rejection did not come from the motion-mode boundary")
    rejected_flag = subprocess.run(
        [str(PUBLIC_BINARY), "--qualification-moving-nest", "--validate-only", "--config", str(CONFIG)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(rejected_flag.returncode != 0, "public binary accepted the internal campaign flag")
    require("internal qualification binary" in rejected_flag.stdout,
            "public internal-flag rejection was ambiguous")


def assert_internal_admission_exact(binary: Path) -> None:
    source = CONFIG.read_text(encoding="utf-8")
    root_x_token = "  nx: 135\n"
    root_y_token = "  ny: 108\n"
    geometry_x_token = "    child_cells_x: 156\n"
    geometry_y_token = "    child_cells_y: 166\n"
    shift_token = "    maximum_shift_cells: 14\n"
    margin_token = "    inner_margin_m: 15600\n"
    require(source.count(root_x_token) == 1, "campaign fixture lost its exact root-width token")
    require(source.count(root_y_token) == 1, "campaign fixture lost its exact root-height token")
    require(source.count(geometry_x_token) == 1, "campaign fixture lost its exact child-width token")
    require(source.count(geometry_y_token) == 1, "campaign fixture lost its exact child-height token")
    require(source.count(shift_token) == 1,
            "campaign fixture lost its emergency-release-restoring movement cap")
    require(source.count(margin_token) == 1, "campaign fixture lost its morphology-aware inner margin")
    with tempfile.TemporaryDirectory(prefix="nimbus_moving_nested_admission_") as temporary:
        neighbors = {
            "root_geometry_x": source.replace(root_x_token, "  nx: 134\n"),
            "root_geometry_y": source.replace(root_y_token, "  ny: 107\n"),
            "child_geometry_x": source.replace(geometry_x_token, "    child_cells_x: 154\n"),
            "child_geometry_y": source.replace(geometry_y_token, "    child_cells_y: 164\n"),
            "unattainable_shift_cap": source.replace(shift_token, "    maximum_shift_cells: 13\n"),
            "insufficient_inner_margin": source.replace(margin_token, "    inner_margin_m: 15000\n"),
        }
        for name, contents in neighbors.items():
            neighbor = Path(temporary) / f"{name}.yaml"
            neighbor.write_text(contents, encoding="utf-8")
            result = subprocess.run(
                [str(binary), "--qualification-moving-nest", "--validate-only", "--config", str(neighbor)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            require(result.returncode != 0, f"internal campaign admitted neighbor {name}")
            require("exact versioned CPU exploratory supercell profile" in result.stdout,
                    f"internal neighbor {name} did not fail at exact admission")


def run_campaign(binary: Path, output: Path, duration_s: int, write_every_s: int, threads: int) -> float:
    assert_internal_admission_exact(binary)
    validate = subprocess.run(
        [str(binary), "--qualification-moving-nest", "--validate-only", "--config", str(CONFIG)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(validate.returncode == 0, f"internal config validation failed:\n{validate.stdout}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, output / "qualification_input.yaml")
    (output / "qualification_binary.sha256").write_text(
        digest(binary) + "\n", encoding="utf-8")
    command = [
        str(binary),
        "--qualification-moving-nest",
        "--headless",
        "--config", str(CONFIG),
        "--duration", str(duration_s),
        f"--write-every={write_every_s}",
        "--threads", str(threads),
        "--guard-mode", "strict",
        f"--outdir={output}",
        "--log-profile", "normal",
        "--timing",
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_seconds = time.monotonic() - started
    (output / "qualification_run.log").write_text(result.stdout, encoding="utf-8")
    require(result.returncode == 0, f"moving-nest campaign exited {result.returncode}\n{result.stdout[-16000:]}")
    return wall_seconds


def inspect_history(history_root: Path, expected_kind: str,
                    expected_shape: tuple[int, int, int]) -> tuple[list[dict], dict[str, dict[str, float]]]:
    steps = sorted(path for path in history_root.glob("step_*") if path.is_dir())
    require(steps, f"history has no publications: {history_root}")
    manifests: list[dict] = []
    extrema: dict[str, dict[str, float]] = {}
    for step in steps:
        manifest = load_json(step / "manifest.json")
        require(manifest.get("publication_state") == "complete", f"incomplete publication: {step}")
        require(manifest.get("data_product", {}).get("kind") == expected_kind,
                f"wrong data product at {step}")
        require(set(manifest.get("published_fields", [])) == FIELDS, f"wrong field inventory at {step}")
        dimensions = manifest.get("grid", {}).get("dimensions", {})
        require(dimensions == dict(zip(("x", "y", "z"), expected_shape)), f"wrong dimensions at {step}")
        frame = manifest.get("grid", {}).get("horizontal_frame", {})
        require(frame.get("motion_mode") == "moving_nest", f"wrong motion provenance at {step}")
        validation = load_json(step / "validation_report.json")
        require(validation.get("failed") is False, f"strict validation failed at {step}")
        for name in sorted(FIELDS):
            shape, low, high = npy_float32_extrema(step / f"{name}.npy")
            field_shape = expected_shape if name in VOLUME_FIELDS else expected_shape[:2]
            require(shape == field_shape, f"wrong {name} shape at {step}")
            item = extrema.setdefault(name, {"min": low, "max": high})
            item["min"] = min(item["min"], low)
            item["max"] = max(item["max"], high)
        manifests.append(manifest)
    return manifests, extrema


def inspect(output: Path, duration_s: int, wall_seconds: float | None,
            require_movement: bool) -> dict:
    config_snapshot, recorded_binary_digest, bound_reference = (
        campaign_artifact_binding(output))
    contract = load_json(output / "nimbus_output.json")
    require(contract.get("publication_state") == "ready", "run publication is not ready")
    require(contract.get("data_products") == ["history", "nested_child_history"], "nested product contract changed")
    require(not (output / "moving_nest_scientific_validity.json").exists(),
            "run published a moving-nest scientific-invalidity artifact")

    root_manifests, root_extrema = inspect_history(output, "history", (135, 108, 50))
    child_root = output / "nests" / "child_0000"
    child_manifests, child_extrema = inspect_history(child_root, "nested_child_history", (158, 168, 50))
    require(root_manifests[0].get("simulation_time_s") == 0, "root history lacks t=0")
    require(child_manifests[0].get("simulation_time_s", 0) > 0, "child history published before a causal parent interval")
    expected_root_times = list(range(0, duration_s + 1, min(450, duration_s)))
    expected_child_times = expected_root_times[1:]
    require([item.get("simulation_time_s") for item in root_manifests] == expected_root_times,
            "root history lost the declared publication cadence")
    require([item.get("simulation_time_s") for item in child_manifests] == expected_child_times,
            "child history lost the causal publication cadence")

    root_bounds = root_manifests[-1]["grid"]["physical_bounds_m"]
    child_origins: list[list[float]] = []
    for manifest in child_manifests:
        bounds = manifest["grid"]["physical_bounds_m"]
        require(bounds["lower"][0] >= root_bounds["lower"][0] and
                bounds["lower"][1] >= root_bounds["lower"][1] and
                bounds["upper"][0] <= root_bounds["upper"][0] and
                bounds["upper"][1] <= root_bounds["upper"][1],
                "child publication escaped the parent physical domain")
        child_origins.append(manifest["grid"]["horizontal_frame"]["grid_origin_t0_earth_m"])

    records = [load_json(path) for path in sorted((output / "storm_track").glob("track_step_*.json"))]
    require(len(records) == duration_s // 30, f"wrong 30-second tracker cadence: {len(records)} records")
    coherent = [record for record in records if record.get("selected_feature_available") is True]
    motion = [record for record in records if record.get("motion_available") is True]
    commits: list[dict] = []
    commit_times_s: list[float] = []
    continuity_reasons: Counter[str] = Counter()
    movement_reasons: Counter[str] = Counter()
    topology_reorganization_indices: list[int] = []
    established_branch_id: int | None = None
    child_storage_origin_x_m, child_storage_origin_y_m = CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M
    minimum_pre_shift_child_boundary_m = math.inf
    minimum_pre_shift_child_boundary_by_side_m = {
        "west": math.inf,
        "east": math.inf,
        "south": math.inf,
        "north": math.inf,
    }
    minimum_post_shift_child_boundary_m = math.inf
    minimum_post_shift_child_boundary_by_side_m = {
        "west": math.inf,
        "east": math.inf,
        "south": math.inf,
        "north": math.inf,
    }
    maximum_envelope_span_x_m = 0.0
    maximum_envelope_span_y_m = 0.0
    maximum_outward_envelope_face_speed_ms = 0.0
    envelope_face_motion_samples = 0
    previous_selected_record: dict | None = None
    authoritative_track_established = False
    cumulative_shift_i = 0
    cumulative_shift_j = 0
    for record_index, record in enumerate(records):
        require(record.get("schema_version") == 12, "wrong tracker schema version")
        require(record.get("tracking_source") == "synchronized_nested_child",
                "moving tracker did not use the synchronized refined child")
        require(record.get("detector_coordinate_frame") == "earth_absolute",
                "moving tracker did not preserve Earth-fixed continuity coordinates")
        require(record.get("algorithm") == "updraft_core_watershed_condensate_rotation_branches_v5",
                "wrong connected-storm detector algorithm")
        require(record.get("continuity_algorithm") == "fixed_window_temporal_branch_association_v5",
                "wrong storm-continuity algorithm")
        continuity = record.get("continuity", {})
        continuity_reasons[str(continuity.get("reason", "missing"))] += 1
        authority = record.get("movement_authority", {})
        movement_reasons[str(authority.get("reason", "missing"))] += 1
        require_authority_contract(authority)
        authoritative_track_established = (
            require_uninterrupted_authoritative_track(
                record, authoritative_track_established))
        require_runtime_scientific_validity(record)
        require_precipitation_relocation(
            authority, float(record.get("simulation_time_s", math.nan)))
        reported_face_velocities = reported_envelope_face_motion(record)
        if reported_face_velocities is not None:
            require(record.get("selected_feature_available") is True,
                    "envelope-face motion exists without a selected storm")
            require(previous_selected_record is not None,
                    "envelope-face motion exists without prior selected evidence")
            independently_derived = envelope_face_coordinate_velocities(
                previous_selected_record, record)
            require(all(math.isclose(
                        reported_face_velocities[side],
                        independently_derived[side],
                        rel_tol=1.0e-12, abs_tol=1.0e-9)
                        for side in independently_derived),
                    "published envelope-face motion disagrees with tracker envelopes")
            reported_outward_speeds = {
                "west": max(0.0, -reported_face_velocities["west"]),
                "east": max(0.0, reported_face_velocities["east"]),
                "south": max(0.0, -reported_face_velocities["south"]),
                "north": max(0.0, reported_face_velocities["north"]),
            }
            envelope_face_motion_samples += 1
            maximum_outward_envelope_face_speed_ms = max(
                maximum_outward_envelope_face_speed_ms,
                max(reported_outward_speeds.values()))
        pre_shift_origin = (
            child_storage_origin_x_m, child_storage_origin_y_m)
        if authority.get("commit_performed") is True:
            require(authority.get("status") == "requested" and
                    authority.get("request_ready") is True,
                    "movement commit lacks a ready request")
            require(authority.get("prognostic_state_mutation") is True,
                    "committed movement denied prognostic mutation")
            require(authority.get("armed") is True,
                    "committed movement did not recover a grid-attainable release box")
            (post_origin, cumulative_shift_i,
             cumulative_shift_j) = committed_origin_and_cumulative(
                authority, pre_shift_origin,
                cumulative_shift_i, cumulative_shift_j)
            child_storage_origin_x_m, child_storage_origin_y_m = post_origin
            commit_time_s = float(record.get("simulation_time_s", math.nan))
            require(math.isfinite(commit_time_s),
                    "committed movement lacks a finite sample time")
            commits.append(authority)
            commit_times_s.append(commit_time_s)
        else:
            require(authority.get("prognostic_state_mutation") is False,
                    "uncommitted tracker decision claimed prognostic mutation")
        if record.get("selected_feature_available") is True:
            require(record.get("selected_feature_role") == "continuity_selected_candidate",
                    "selected storm fields have ambiguous provenance")
            branch_id = int(continuity.get("branch_id", 0))
            if established_branch_id is None:
                established_branch_id = branch_id
                require(established_branch_id > 0, "selected storm lacks a branch identity")
            require(branch_id == established_branch_id, "selected storm changed branch identity")
            if continuity.get("topology_reorganization") is True:
                topology_reorganization_indices.append(record_index)
                require(continuity.get("reorganization_overlap_basis") ==
                        "exact_watershed_partition_cells",
                        "storm reorganization used a summary box instead of exact ownership")
                require(continuity.get("status") == "tentative",
                        "storm reorganization claimed mature motion")
                require(record.get("motion_available") is False,
                        "storm reorganization invented translation")

            (pre_shift_distances, child_distances,
             envelope_span_x_m, envelope_span_y_m) = selected_envelope_geometry(
                record, pre_shift_origin,
                (child_storage_origin_x_m, child_storage_origin_y_m))
            maximum_envelope_span_x_m = max(
                maximum_envelope_span_x_m, envelope_span_x_m)
            maximum_envelope_span_y_m = max(
                maximum_envelope_span_y_m, envelope_span_y_m)
            previous_selected_record = record
            minimum_pre_shift_child_boundary_m = min(
                minimum_pre_shift_child_boundary_m,
                min(pre_shift_distances.values()))
            minimum_post_shift_child_boundary_m = min(
                minimum_post_shift_child_boundary_m,
                min(child_distances.values()))
            for side, distance in pre_shift_distances.items():
                minimum_pre_shift_child_boundary_by_side_m[side] = min(
                    minimum_pre_shift_child_boundary_by_side_m[side], distance)
            for side, distance in child_distances.items():
                minimum_post_shift_child_boundary_by_side_m[side] = min(
                    minimum_post_shift_child_boundary_by_side_m[side], distance)
        else:
            require(record.get("selected_feature_role") is None,
                    "unavailable storm has a selected role")
            require(authority.get("commit_performed") is not True,
                    "movement committed without a selected storm envelope")
    sequences = [int(item["nested_movement_sequence"]) for item in commits]
    require(sequences == list(range(1, len(sequences) + 1)), "movement sequence is not contiguous")
    if require_movement:
        require(commits, "campaign never committed a tracked child relocation")
        require(envelope_face_motion_samples > 0,
                "moving campaign published no accepted transient envelope-face motion")
    require_safe_movement_reasons(movement_reasons)
    maximum_confirmation_advance_m = require_confirmation_headroom(
        maximum_outward_envelope_face_speed_ms)

    maximum_tracker_updraft = max((float(record.get("maximum_updraft_ms") or 0.0) for record in coherent), default=0.0)
    maximum_uh = max((float(record.get("rotation", {}).get("maximum_absolute_updraft_helicity_m2s2") or 0.0)
                      for record in coherent), default=0.0)
    if not coherent:
        minimum_pre_shift_child_boundary_m = None
        minimum_pre_shift_child_boundary_by_side_m = {
            "west": None,
            "east": None,
            "south": None,
            "north": None,
        }
        minimum_post_shift_child_boundary_m = None
        minimum_post_shift_child_boundary_by_side_m = {
            "west": None,
            "east": None,
            "south": None,
            "north": None,
        }
    if duration_s >= 4500:
        first_selected_index = next(
            (index for index, record in enumerate(records)
             if record.get("selected_feature_available") is True), None)
        require(first_selected_index is not None, "lifetime campaign never selected a storm")
        mature_records = records[first_selected_index:]
        mature_selected_fraction = len(coherent) / len(mature_records)
        selected_motion_fraction = len(motion) / len(coherent)
        require(mature_selected_fraction >= 0.95,
                "storm selection was unavailable for more than 5% of mature lifetime")
        require(selected_motion_fraction >= 0.85,
                "storm motion was unavailable for more than 15% of selected samples")

        maximum_consecutive_unselected_samples = 0
        consecutive_unselected_samples = 0
        for record in mature_records:
            if record.get("selected_feature_available") is True:
                consecutive_unselected_samples = 0
            else:
                consecutive_unselected_samples += 1
                maximum_consecutive_unselected_samples = max(
                    maximum_consecutive_unselected_samples,
                    consecutive_unselected_samples)
        require(maximum_consecutive_unselected_samples <= 2,
                "mature storm association was unavailable for more than two samples")

        maximum_reorganization_recovery_samples = 0
        for event_index in topology_reorganization_indices:
            recovery_offset = next(
                (offset for offset, record in enumerate(
                    records[event_index + 1:event_index + 5], start=1)
                 if record.get("motion_available") is True), None)
            require(recovery_offset is not None,
                    "storm motion did not recover within four samples of reorganization")
            maximum_reorganization_recovery_samples = max(
                maximum_reorganization_recovery_samples, recovery_offset)

        require(records[-1].get("status") in {"detected", "ambiguous"},
                "storm is absent at 4500 s")
        require(records[-1].get("selected_feature_available") is True,
                "storm lacks a continuity-selected feature at 4500 s")
        require(records[-1].get("motion_available") is True,
                "storm lacks a mature motion estimate at 4500 s")
        require(0.5 * REFERENCE["maximum_output_updraft_ms"] <= child_extrema["w"]["max"] <=
                1.5 * REFERENCE["maximum_output_updraft_ms"], "child updraft left the reference envelope")
        require(0.5 * REFERENCE["maximum_tracker_updraft_ms"] <= maximum_tracker_updraft <=
                1.5 * REFERENCE["maximum_tracker_updraft_ms"], "tracker updraft left the reference envelope")
        require(0.35 * REFERENCE["maximum_absolute_updraft_helicity_m2s2"] <= maximum_uh <=
                2.0 * REFERENCE["maximum_absolute_updraft_helicity_m2s2"], "UH left the reference envelope")
    else:
        mature_selected_fraction = None
        selected_motion_fraction = None
        maximum_consecutive_unselected_samples = None
        maximum_reorganization_recovery_samples = None

    ledgers = {
        name: finite_ledger(output / name)
        for name in ("budget.csv", "energy_budget.csv", "moisture_budget.csv",
                     "momentum_budget.csv", "thermodynamic_source_budget.csv")
    }
    checkpoints = sorted((output / "checkpoints").glob("checkpoint_step_*.nimbus"))
    require(checkpoints, "campaign produced no synchronized checkpoint")
    if duration_s >= 4500:
        checkpoint_steps = [int(path.stem.removeprefix("checkpoint_step_")) for path in checkpoints]
        require(checkpoint_steps == [1500, 3000],
                f"lifetime campaign lost checkpoint cadence: {checkpoint_steps}")
    log = (output / "qualification_run.log").read_text(encoding="utf-8") if (output / "qualification_run.log").is_file() else ""
    memory_milestones = {
        phase: int(value)
        for phase, value in re.findall(
            r"\[MEMORY MILESTONE\] phase=([^,]+), peak_rss_bytes=(\d+)", log)
    }
    rss_values = list(memory_milestones.values())
    predicted = re.findall(r"\[MEMORY PLAN\].*predicted_bytes=(\d+)", log)
    perf = re.findall(r"\[PERF SUMMARY\].*sim_s_per_wall_s=([0-9.eE+-]+)", log)
    output_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    parent_cells = 135 * 108
    child_cells = 158 * 168
    reference_cells = 269 * 215
    return {
        "schema": "nimbus.qualification.moving_nested_supercell",
        "schema_version": 2,
        "status": "pass",
        "scope": "internal_cpu_moving_nest_campaign_not_selectable",
        "duration_s": duration_s,
        "wall_seconds": wall_seconds,
        "inspector_sha256": digest(Path(__file__)),
        "config_sha256": digest(config_snapshot),
        "binary_sha256": recorded_binary_digest,
        "reference_binding": bound_reference,
        "movement_geometry_contract": movement_geometry_contract(),
        "root_storage_shape": [135, 108, 50],
        "child_storage_shape": [158, 168, 50],
        "horizontal_storage_cell_ratio_to_reference": (parent_cells + child_cells) / reference_cells,
        "storm_samples": len(records),
        "coherent_samples": len(coherent),
        "motion_available_samples": len(motion),
        "mature_selected_fraction": mature_selected_fraction,
        "selected_motion_fraction": selected_motion_fraction,
        "maximum_consecutive_unselected_samples": maximum_consecutive_unselected_samples,
        "maximum_reorganization_recovery_samples": maximum_reorganization_recovery_samples,
        "continuity_reason_counts": dict(sorted(continuity_reasons.items())),
        "movement_reason_counts": dict(sorted(movement_reasons.items())),
        "topology_reorganization_samples": len(topology_reorganization_indices),
        "committed_movements": len(commits),
        "movement_sequences": sequences,
        "committed_movement_times_s": commit_times_s,
        "committed_movement_shifts_parent_cells": [
            item["committed_shift_cells"] for item in commits
        ],
        "committed_movement_post_child_origins_earth_m": [
            item["post_shift_child_grid_origin_t0_earth_m"] for item in commits
        ],
        "child_published_origins_earth_m": child_origins,
        "minimum_selected_envelope_distance_to_child_boundary_m":
            minimum_pre_shift_child_boundary_m,
        "minimum_selected_envelope_distance_to_child_boundary_by_side_m":
            minimum_pre_shift_child_boundary_by_side_m,
        "minimum_post_shift_selected_envelope_distance_to_child_boundary_m":
            minimum_post_shift_child_boundary_m,
        "minimum_post_shift_selected_envelope_distance_to_child_boundary_by_side_m":
            minimum_post_shift_child_boundary_by_side_m,
        "relaxation_zone_width_m": RELAXATION_WIDTH_M,
        "maximum_selected_envelope_span_x_m": maximum_envelope_span_x_m,
        "maximum_selected_envelope_span_y_m": maximum_envelope_span_y_m,
        "maximum_observed_outward_envelope_face_speed_ms":
            maximum_outward_envelope_face_speed_ms,
        "envelope_face_motion_samples": envelope_face_motion_samples,
        "maximum_one_cadence_outward_envelope_advance_m":
            maximum_confirmation_advance_m,
        "confirmation_headroom_to_relaxation_zone_m":
            INNER_MARGIN_M - RELAXATION_WIDTH_M - maximum_confirmation_advance_m,
        "maximum_tracker_updraft_ms": maximum_tracker_updraft,
        "maximum_absolute_updraft_helicity_m2s2": maximum_uh,
        "root_field_extrema": root_extrema,
        "child_field_extrema": child_extrema,
        "reference_summary": REFERENCE,
        "ledger_rows": ledgers,
        "checkpoint_count": len(checkpoints),
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoints),
        "output_bytes": output_bytes,
        "predicted_memory_bytes": int(predicted[-1]) if predicted else None,
        "measured_peak_rss_bytes": max(rss_values) if rss_values else None,
        "memory_milestones_peak_rss_bytes": memory_milestones,
        "simulated_seconds_per_wall_second": float(perf[-1]) if perf else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=INTERNAL_BINARY)
    parser.add_argument("--output", type=Path, help="persistent run directory; omit for a temporary campaign")
    parser.add_argument("--inspect", type=Path, help="inspect an existing campaign directory")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--write-every", type=int)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--require-movement", action="store_true")
    args = parser.parse_args()
    require(args.duration > 0 and args.duration % 30 == 0, "duration must be a positive multiple of 30 s")
    require(args.threads > 0, "threads must be positive")
    assert_public_admission_closed()
    if args.inspect is not None:
        report = inspect(args.inspect.resolve(), args.duration, None, args.require_movement)
    elif args.output is not None:
        binary = args.binary.resolve()
        require(binary.is_file(), f"missing internal qualification binary: {binary}")
        output = args.output.resolve()
        require(not output.exists() or not any(output.iterdir()), f"output directory is not empty: {output}")
        write_every = args.write_every or min(450, args.duration)
        wall = run_campaign(binary, output, args.duration, write_every, args.threads)
        report = inspect(output, args.duration, wall, args.require_movement)
        (output / "qualification_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        with tempfile.TemporaryDirectory(prefix="nimbus_moving_nested_") as temporary:
            output = Path(temporary)
            write_every = args.write_every or min(450, args.duration)
            wall = run_campaign(args.binary.resolve(), output, args.duration, write_every, args.threads)
            report = inspect(output, args.duration, wall, args.require_movement)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[moving-nested-supercell] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)

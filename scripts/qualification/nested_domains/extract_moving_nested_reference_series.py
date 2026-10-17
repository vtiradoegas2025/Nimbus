#!/usr/bin/env python3
"""Extract a hash-bound moving-nest/reference comparison series.

This is deliberately a strict extractor, not a best-effort plotting reader.
It consumes complete NPY publications and storm-tracker records, restricts
every diagnostic to the same Earth-fixed trusted-interior window, and emits a
versioned source manifest alongside the scalar series used by the frontier
comparator.
"""

from __future__ import annotations

import argparse
import array
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import compare_moving_nested_supercell_reference as comparator


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = (
    ROOT / "tests" / "qualification" / "fixtures" /
    "moving_nested_supercell_reference_frontier_v12.json"
)
SERIES_SCHEMA = "nimbus.qualification.moving_nested_reference_series"
SOURCE_SCHEMA = "nimbus.qualification.moving_nested_reference_source_manifest"
COLD_POOL_ALGORITHM = "surface_density_potential_temperature_deficit_v1"
PRECIPITATION_ALGORITHM = "earth_aligned_accumulated_rainfall_v1"
DETECTOR_ALGORITHM = "updraft_core_watershed_condensate_rotation_branches_v5"
CONTINUITY_ALGORITHM = "fixed_window_temporal_branch_association_v5"
MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v5"
PROJECTED_CONTAINMENT_MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v6"
MAXIMUM_SAFE_PROJECTED_MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v7"
TRUSTED_INTERIOR_CONTAINMENT_MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v8"
BALANCED_TRUSTED_CONTAINMENT_MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v9"
ORTHOGONAL_CONTAINMENT_RESERVE_MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v10"
RECEDING_TRUSTED_FACE_MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v11"
QUANTIZED_TRUSTED_BALANCE_MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v12"
ORDINARY_TRUSTED_IMPROVEMENT_MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v13"
PARENT_STENCIL_LIMIT_MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v14"
GEOMETRY_PLAN_SCHEMA = "nimbus.qualification.moving_nested_geometry_plan"
APERTURE_SCHEMA = "nimbus.qualification.moving_nested_fixed_aperture"
APERTURE_REGION_ID = "moving_child_qualification_aperture_v1"
HEX64 = re.compile(r"[0-9a-f]{64}")
SOURCE_FILES = (
    "include/diagnostics/derived_field_math.hpp",
    "src/diagnostics/derived/tranche_product_registry.cpp",
)
NUMERICAL_PREFIXES = (
    "dynamics.", "coriolis.", "microphysics.",
    "turbulence.", "diffusion.", "surface.", "boundary_layer.",
    "radiation.", "terrain.", "initial_conditions.sounding.",
    "initial_conditions.hodograph.",
)
NUMERICAL_EXACT_KEYS = {
    "schema", "grid.coordinate_system", "grid.staggering", "grid.nz",
    "grid.dz", "grid.ztop", "time.acoustic_substeps",
    "time.adaptive_dt", "time.cfl_safety", "boundary_conditions.bottom",
    "boundary_conditions.top", "initial_conditions.trigger.type",
    "initial_conditions.trigger.bubble.dtheta_k",
    "initial_conditions.trigger.bubble.radius_km",
    "initial_conditions.trigger.bubble.center_z_m",
    "extensions.compute.backend",
    "extensions.diagnostics.deterministic_reductions",
}
COMPARABLE_DAMPING_KEYS = {
    "damping.enabled", "damping.z_d", "damping.alpha", "damping.mode",
    "damping.w_only_top_sponge", "damping.w_only_depth_cells",
    "damping.w_only_alpha_max",
}


class ExtractionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtractionError(message)


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing artifact: {path}")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relocated_qualification_fixture(path: Path) -> Path:
    """Resolve the retired qualification-config prefix without changing seals."""
    resolved = path.resolve()
    if resolved.is_file():
        return resolved
    try:
        fixture_name = resolved.relative_to(ROOT / "configs/qualification")
    except ValueError:
        return resolved
    return ROOT / "tests/fixtures/configs/qualification" / fixture_name


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def strip_yaml_comment(line: str) -> str:
    single = False
    double = False
    for index, character in enumerate(line):
        if character == "'" and not double:
            single = not single
        elif character == '"' and not single and (index == 0 or line[index - 1] != "\\"):
            double = not double
        elif character == "#" and not single and not double:
            return line[:index]
    return line


def flat_yaml_scalars(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    stack: list[tuple[int, str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require("\t" not in raw, f"tabs are not supported in YAML: {path}:{number}")
        content = strip_yaml_comment(raw).rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        token = content.strip()
        require(not token.startswith("-"),
                f"sequence-style YAML is not supported by the signature extractor: {path}:{number}")
        require(":" in token, f"invalid YAML mapping line: {path}:{number}")
        key, value = token.split(":", 1)
        key = key.strip()
        require(key, f"empty YAML key: {path}:{number}")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([entry[1] for entry in stack] + [key])
        value = value.strip()
        if value:
            require(dotted not in result, f"duplicate YAML scalar {dotted}: {path}:{number}")
            result[dotted] = value
        else:
            stack.append((indent, key))
    return result


def yaml_float(flat: dict[str, str], key: str) -> float:
    require(key in flat, f"configuration lacks {key}")
    try:
        value = float(flat[key])
    except ValueError as exc:
        raise ExtractionError(f"configuration {key} is not numeric") from exc
    require(math.isfinite(value), f"configuration {key} is not finite")
    return value


def yaml_positive_integer(flat: dict[str, str], key: str) -> int:
    value = yaml_float(flat, key)
    result = int(value)
    require(value == result and result > 0,
            f"configuration {key} must be a positive integer")
    return result


def numerical_signature(
    config: Path, role: str, observed_spacing: dict[str, float]
) -> tuple[dict[str, object], str, dict[str, object]]:
    flat = flat_yaml_scalars(config)
    selected = {
        key: value for key, value in flat.items()
        if key in NUMERICAL_EXACT_KEYS or
        any(key.startswith(prefix) for prefix in NUMERICAL_PREFIXES)
    }
    selected.update({key: flat[key] for key in COMPARABLE_DAMPING_KEYS if key in flat})
    missing = sorted(key for key in (
        "schema", "grid.coordinate_system", "grid.staggering", "grid.nz",
        "grid.dz", "time.acoustic_substeps",
        "dynamics.scheme", "microphysics.scheme", "turbulence.scheme",
        "initial_conditions.sounding.type", "initial_conditions.hodograph.type",
        "initial_conditions.trigger.type", "extensions.compute.backend",
        "extensions.diagnostics.deterministic_reductions",
    ) if key not in selected)
    require(not missing, "configuration lacks numerical-signature keys: " + ", ".join(missing))
    root_dt = yaml_float(flat, "time.dt_large")
    root_dx = yaml_float(flat, "grid.dx")
    root_dy = yaml_float(flat, "grid.dy")
    require(root_dt > 0.0 and root_dx > 0.0 and root_dy > 0.0,
            "root timestep and horizontal spacing must be positive")
    if role == "moving_child":
        time_ratio = yaml_positive_integer(
            flat, "domain_motion.nesting.parent_time_step_ratio")
        ratio_x = yaml_positive_integer(flat, "domain_motion.nesting.parent_ratio_x")
        ratio_y = yaml_positive_integer(flat, "domain_motion.nesting.parent_ratio_y")
        level = "nested_child"
    else:
        time_ratio = ratio_x = ratio_y = 1
        level = "root_reference"
    effective_dt = root_dt / time_ratio
    effective_dx = root_dx / ratio_x
    effective_dy = root_dy / ratio_y
    require(math.isclose(effective_dx, observed_spacing["x"],
                         rel_tol=0.0, abs_tol=1.0e-9) and
            math.isclose(effective_dy, observed_spacing["y"],
                         rel_tol=0.0, abs_tol=1.0e-9),
            "derived effective grid spacing differs from the published comparison level")
    selected["comparison_level.effective_dt_large_s"] = effective_dt
    selected["comparison_level.effective_dx_m"] = effective_dx
    selected["comparison_level.effective_dy_m"] = effective_dy
    derivation = {
        "role": role,
        "effective_level": level,
        "root_dt_large_s": root_dt,
        "parent_time_step_ratio": time_ratio,
        "effective_dt_large_s": effective_dt,
        "root_dx_m": root_dx,
        "root_dy_m": root_dy,
        "parent_ratio_x": ratio_x,
        "parent_ratio_y": ratio_y,
        "effective_dx_m": effective_dx,
        "effective_dy_m": effective_dy,
        "published_comparison_spacing_m": observed_spacing,
    }
    return selected, canonical_sha256(selected), derivation


def read_npy_float32(path: Path) -> tuple[tuple[int, ...], array.array]:
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", f"invalid NPY magic: {path}")
        version = stream.read(2)
        require(len(version) == 2 and version[0] in {1, 2, 3},
                f"unsupported NPY version: {path}")
        length_size = 2 if version[0] == 1 else 4
        raw_length = stream.read(length_size)
        require(len(raw_length) == length_size, f"truncated NPY header: {path}")
        header_size = struct.unpack("<H" if length_size == 2 else "<I", raw_length)[0]
        header = ast.literal_eval(stream.read(header_size).decode("latin1").strip())
        require(header.get("descr") in {"<f4", "=f4", "|f4"},
                f"only float32 NPY is supported: {path}")
        require(header.get("fortran_order") is False,
                f"Fortran-order NPY is unsupported: {path}")
        shape = tuple(int(item) for item in header.get("shape", ()))
        payload = stream.read()
    require(shape and all(item > 0 for item in shape), f"invalid NPY shape: {path}")
    require(len(payload) == math.prod(shape) * 4, f"wrong NPY payload length: {path}")
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little" and header.get("descr") == "<f4":
        values.byteswap()
    require(all(math.isfinite(value) for value in values), f"non-finite NPY payload: {path}")
    return shape, values


def finite_number(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def step_publications(history: Path, expected_kind: str) -> dict[float, tuple[Path, dict[str, Any]]]:
    result: dict[float, tuple[Path, dict[str, Any]]] = {}
    for directory in sorted(history.glob("step_*")):
        if not directory.is_dir():
            continue
        manifest_path = directory / "manifest.json"
        manifest = load_json(manifest_path)
        require(manifest.get("schema") == "nimbus.step" and
                manifest.get("manifest_version") in {6, 7} and
                manifest.get("publication_state") == "complete" and
                manifest.get("data_product", {}).get("kind") == expected_kind,
                f"incomplete or incompatible step manifest: {manifest_path}")
        time_s = finite_number(manifest.get("simulation_time_s"),
                               f"{manifest_path}: simulation_time_s")
        require(time_s not in result, f"duplicate publication time {time_s}")
        result[time_s] = (directory, manifest)
    require(result, f"no complete publications in {history}")
    return result


def tracker_publications(run: Path) -> dict[float, tuple[Path, dict[str, Any]]]:
    result: dict[float, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((run / "storm_track").glob("track_step_*.json")):
        record = load_json(path)
        require(record.get("schema") == "nimbus.storm_feature" and
                record.get("schema_version") == 12,
                f"incompatible storm record: {path}")
        require(record.get("algorithm") == DETECTOR_ALGORITHM and
                record.get("continuity_algorithm") == CONTINUITY_ALGORITHM,
                f"storm algorithms changed: {path}")
        require(record.get("detector_coordinate_frame") in
                {"earth_absolute", "grid_relative"},
                f"storm detector coordinate frame is unsupported: {path}")
        time_s = finite_number(record.get("simulation_time_s"),
                               f"{path}: simulation_time_s")
        require(time_s not in result, f"duplicate tracker time {time_s}")
        result[time_s] = (path, record)
    require(result, f"no storm records in {run / 'storm_track'}")
    return result


def expected_times(policy: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    require(policy.get("schema") ==
            "nimbus.qualification.moving_nested_reference_frontier.policy" and
            policy.get("schema_version") in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
            "wrong reference-frontier policy schema")
    if policy.get("schema_version") == 1:
        require(policy.get("policy_status") == "measurement_only",
                "v1 extraction policy is permanently measurement-only")
    else:
        require(policy.get("policy_status") == "ratified",
                "ratified extraction policy must be explicitly ratified")
    expected = policy.get("expected_contract")
    require(isinstance(expected, dict), "policy lacks expected_contract")
    start = finite_number(expected.get("sample_start_s"), "policy sample start")
    interval = finite_number(expected.get("sample_interval_s"), "policy sample interval")
    duration = finite_number(expected.get("duration_s"), "policy duration")
    require(start >= 0.0 and interval > 0.0 and duration >= start,
            "invalid policy sample cadence")
    count = int(round((duration - start) / interval)) + 1
    require(count > 0 and math.isclose(start + (count - 1) * interval, duration,
                                        rel_tol=0.0, abs_tol=1.0e-9),
            "policy duration is not exactly reachable at its cadence")
    return [start + index * interval for index in range(count)], expected


def bound_policy_document(policy_path: Path, record: object, label: str,
                          schema: str, identity_key: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(record, dict), f"policy lacks {label} binding")
    require(set(record) == {"path", "sha256", "schema_version", identity_key},
            f"policy {label} binding inventory changed")
    raw_path = record.get("path")
    digest = record.get("sha256")
    require(isinstance(raw_path, str) and raw_path and
            isinstance(digest, str) and HEX64.fullmatch(digest) is not None,
            f"policy {label} binding is invalid")
    path = Path(raw_path)
    if not path.is_absolute():
        path = policy_path.parent / path
    path = path.resolve()
    require(sha256(path) == digest, f"policy {label} hash mismatch")
    document = load_json(path)
    require(document.get("schema") == schema and
            document.get("schema_version") == record.get("schema_version") and
            document.get(identity_key) == record.get(identity_key),
            f"policy {label} identity mismatch")
    return path, document


def validate_geometry_plan(plan: dict[str, Any]) -> None:
    require(plan.get("schema") == GEOMETRY_PLAN_SCHEMA and
            plan.get("schema_version") == 1 and
            plan.get("plan_id") == "moving_nested_reference_geometry_v1" and
            plan.get("status") == "reference_derived_pre_rerun",
            "moving-nest geometry plan identity changed")
    source = plan.get("source_reference")
    require(isinstance(source, dict) and source.get("run_role") ==
            "oversized_stationary_reference" and
            HEX64.fullmatch(str(source.get("binary_sha256", ""))) is not None and
            HEX64.fullmatch(str(source.get("configuration_snapshot_sha256", ""))) is not None,
            "moving-nest geometry plan source binding is invalid")
    inventory = source.get("tracker_inventory")
    require(isinstance(inventory, dict) and inventory.get("record_count") == 150 and
            inventory.get("first_time_s") == 30.0 and
            inventory.get("last_time_s") == 4500.0 and
            inventory.get("canonical_record_set_sha256") ==
                "df7ca2b4a5c63ec387375ecd15096c81872cdc51899508ee3bb3064651c0ecdf",
            "moving-nest geometry tracker inventory changed")
    inputs = plan.get("scientific_inputs")
    child = plan.get("planned_child")
    parent = plan.get("planned_parent")
    replay = plan.get("reference_driven_movement_replay")
    requirement = plan.get("qualification_aperture_requirement")
    require(all(isinstance(value, dict) for value in
                (inputs, child, parent, replay, requirement)),
            "moving-nest geometry plan is incomplete")
    require(inputs.get("maximum_authoritative_envelope_span_m") ==
                {"x": 58200.0, "y": 64200.0} and
            inputs.get("inner_margin_m") == 15600.0 and
            inputs.get("nominal_release_margin_m") == 16800.0 and
            inputs.get("parent_lattice_phase_allowance_m") ==
                {"x": 1200.0, "y": 1200.0},
            "moving-nest geometry scientific inputs changed")
    require(child.get("physical_cells") == {"x": 156, "y": 166, "z": 50} and
            child.get("storage_cells") == {"x": 158, "y": 168, "z": 50} and
            child.get("parent_cell_span") == {"x": 78, "y": 83} and
            child.get("operational_interior_shape_cells") == {"x": 140, "y": 150},
            "moving-nest planned child geometry changed")
    spans = inputs["maximum_authoritative_envelope_span_m"]
    release = inputs["nominal_release_margin_m"]
    lattice = inputs["parent_lattice_phase_allowance_m"]
    spacing = inputs.get("parent_shift_spacing_m")
    extent = child.get("physical_extent_m")
    require(spacing == {"x": 1200.0, "y": 1200.0} and
            extent == {"x": 93600.0, "y": 99600.0},
            "moving-nest geometry spacing or extent changed")
    for axis in ("x", "y"):
        minimum = spans[axis] + 2.0 * release + lattice[axis]
        realized = math.ceil(minimum / spacing[axis]) * spacing[axis]
        require(realized == extent[axis],
                f"moving-nest {axis} extent does not satisfy its sizing rule")
    require(parent.get("boundary_stencil_parent_cells") == 1 and
            parent.get("minimum_physical_cells") == {"x": 133, "y": 106, "z": 50} and
            parent.get("minimum_storage_cells") == {"x": 135, "y": 108, "z": 50},
            "moving-nest planned parent geometry changed")
    expected_moves = [
        {"simulation_time_s": 2010.0, "i": 14, "j": 0},
        {"simulation_time_s": 3150.0, "i": 14, "j": 0},
        {"simulation_time_s": 3420.0, "i": 0, "j": 10},
        {"simulation_time_s": 3480.0, "i": 8, "j": 0},
        {"simulation_time_s": 3900.0, "i": 5, "j": 6},
        {"simulation_time_s": 3990.0, "i": 4, "j": 0},
        {"simulation_time_s": 4140.0, "i": 3, "j": 3},
        {"simulation_time_s": 4380.0, "i": 4, "j": 2},
    ]
    require(replay.get("algorithm") == MOVEMENT_ALGORITHM and
            replay.get("initial_child_origin_parent_cell") == {"i": 2, "j": 1} and
            replay.get("maximum_child_origin_parent_cell") == {"i": 54, "j": 22} and
            replay.get("final_child_origin_parent_cell") == {"i": 54, "j": 22} and
            replay.get("ineligible_geometry_decisions") == 0 and
            replay.get("expected_moves") == expected_moves,
            "moving-nest reference movement replay changed")
    maximum_origin = replay["maximum_child_origin_parent_cell"]
    parent_span = child["parent_cell_span"]
    stencil = parent["boundary_stencil_parent_cells"]
    require(parent["minimum_physical_cells"]["x"] ==
                maximum_origin["i"] + parent_span["x"] + stencil and
            parent["minimum_physical_cells"]["y"] ==
                maximum_origin["j"] + parent_span["y"] + stencil,
            "moving-nest parent minimum does not contain the replayed child")
    require(requirement.get("aperture_id") == "moving_nested_fixed_100x96_v1" and
            requirement.get("shape_cells") == {"x": 100, "y": 96} and
            requirement.get("cell_count") == 9600,
            "moving-nest geometry plan aperture requirement changed")


def validate_geometry_plan_sources(plan_path: Path,
                                   plan: dict[str, Any]) -> None:
    configurations = plan.get("planned_configurations")
    require(isinstance(configurations, dict) and set(configurations) == {
                "moving_lifetime", "moving_frontier", "oversized_reference"},
            "moving-nest planned configuration inventory changed")
    for name, record in configurations.items():
        require(isinstance(record, dict) and set(record) == {"path", "sha256"},
                f"moving-nest planned configuration {name} binding changed")
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = plan_path.parent / path
        path = relocated_qualification_fixture(path)
        require(sha256(path) == record.get("sha256"),
                f"moving-nest planned configuration {name} hash mismatch")


def validate_aperture(aperture: dict[str, Any]) -> None:
    require(aperture.get("schema") == APERTURE_SCHEMA and
            aperture.get("schema_version") == 1 and
            aperture.get("aperture_id") == "moving_nested_fixed_100x96_v1" and
            aperture.get("owner") == "fixed_centered_operational_trusted_interior_v1" and
            aperture.get("placement") ==
                "fixed_source_index_offset_within_operational_interior",
            "qualification aperture identity changed")
    require(aperture.get("horizontal_grid_spacing_m") == {"x": 600.0, "y": 600.0} and
            aperture.get("grid_phase_origin_m") == {"x": 0.0, "y": 0.0} and
            aperture.get("shape_cells") == {"x": 100, "y": 96} and
            aperture.get("cell_count") == 9600,
            "qualification aperture numerical geometry changed")
    interior = aperture.get("operational_interior")
    require(isinstance(interior, dict) and
            interior.get("owner") == "physical_child_trusted_interior_v1" and
            interior.get("relaxation_exclusion_cells") == 8 and
            interior.get("required_shape_cells") == {"x": 140, "y": 150} and
            interior.get("aperture_offset_cells") == {"x": 20, "y": 27},
            "qualification aperture operational placement changed")
    crop = aperture.get("reference_crop")
    require(isinstance(crop, dict) and
            crop.get("rule") == "exact_moving_aperture_earth_bounds_v1" and
            crop.get("require_identical_spacing") is True and
            crop.get("require_identical_grid_phase") is True and
            crop.get("require_complete_physical_coverage") is True,
            "qualification aperture reference crop changed")


def validate_configured_geometry(config: Path, role: str,
                                 plan: dict[str, Any]) -> None:
    flat = flat_yaml_scalars(config)
    child = plan["planned_child"]
    parent = plan["planned_parent"]
    if role == "moving_child":
        expected = {
            "grid.nx": parent["minimum_storage_cells"]["x"],
            "grid.ny": parent["minimum_storage_cells"]["y"],
            "grid.nz": parent["minimum_storage_cells"]["z"],
            "domain_motion.nesting.child_cells_x": child["physical_cells"]["x"],
            "domain_motion.nesting.child_cells_y": child["physical_cells"]["y"],
            "domain_motion.nesting.child_origin_parent_cell_i": 2,
            "domain_motion.nesting.child_origin_parent_cell_j": 1,
            "domain_motion.nesting.relaxation_width_child_cells": 8,
        }
    else:
        expected = {"grid.nx": 269, "grid.ny": 215, "grid.nz": 50}
    for key, value in expected.items():
        require(yaml_positive_integer(flat, key) == value,
                f"configuration {key} differs from the bound geometry plan")


def field_descriptor(manifest: dict[str, Any], field_id: str) -> dict[str, Any]:
    fields = manifest.get("fields")
    require(isinstance(fields, list), "step manifest lacks field descriptors")
    matches = [value for value in fields
               if isinstance(value, dict) and value.get("field_id") == field_id]
    require(len(matches) == 1, f"step manifest lacks unique {field_id} descriptor")
    return matches[0]


def grid_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    grid = manifest.get("grid")
    require(isinstance(grid, dict) and grid.get("coordinate_system") == "cartesian",
            "reference extraction requires a Cartesian grid")
    dims = grid.get("dimensions")
    spacing = grid.get("spacing_m")
    valid = grid.get("valid_index_range")
    frame = grid.get("horizontal_frame")
    require(all(isinstance(item, dict) for item in (dims, spacing, valid, frame)),
            "step manifest has incomplete grid geometry")
    nx = dims.get("x")
    ny = dims.get("y")
    require(type(nx) is int and type(ny) is int and nx > 0 and ny > 0,
            "invalid horizontal dimensions")
    dx = finite_number(spacing.get("x"), "grid dx")
    dy = finite_number(spacing.get("y"), "grid dy")
    require(dx > 0.0 and dy > 0.0, "grid spacing must be positive")
    grid_origin = frame.get("grid_origin_earth_m")
    scalar_origin = grid.get("grid_scalar_origin_m")
    require(isinstance(grid_origin, list) and len(grid_origin) >= 2 and
            isinstance(scalar_origin, list) and len(scalar_origin) >= 2,
            "manifest lacks Earth/grid scalar origins")
    origin_x = finite_number(grid_origin[0], "Earth grid origin x")
    origin_y = finite_number(grid_origin[1], "Earth grid origin y")
    scalar_x = origin_x + finite_number(scalar_origin[0], "grid scalar origin x")
    scalar_y = origin_y + finite_number(scalar_origin[1], "grid scalar origin y")
    require(math.isclose(finite_number(scalar_origin[0], "grid scalar origin x"),
                         0.5 * dx, rel_tol=0.0, abs_tol=1.0e-9) and
            math.isclose(finite_number(scalar_origin[1], "grid scalar origin y"),
                         0.5 * dy, rel_tol=0.0, abs_tol=1.0e-9),
            "publication scalar centers are not on the canonical half-cell phase")
    lower = valid.get("lower")
    upper = valid.get("upper")
    require(isinstance(lower, list) and isinstance(upper, list) and
            len(lower) >= 2 and len(upper) >= 2,
            "manifest lacks physical index ownership")
    lo_i, lo_j = int(lower[0]), int(lower[1])
    hi_i, hi_j = int(upper[0]), int(upper[1])
    require(0 <= lo_i <= hi_i < nx and 0 <= lo_j <= hi_j < ny,
            "invalid physical index ownership")
    return {
        "nx": nx, "ny": ny, "dx": dx, "dy": dy,
        "scalar_x": scalar_x, "scalar_y": scalar_y,
        "lo_i": lo_i, "lo_j": lo_j, "hi_i": hi_i, "hi_j": hi_j,
        "phase_x": math.remainder(origin_x, dx),
        "phase_y": math.remainder(origin_y, dy),
    }


def moving_region(manifest: dict[str, Any], exclusion: int,
                  aperture: dict[str, Any] | None = None) -> dict[str, Any]:
    grid = grid_contract(manifest)
    lo_i = grid["lo_i"] + exclusion
    lo_j = grid["lo_j"] + exclusion
    hi_i = grid["hi_i"] - exclusion
    hi_j = grid["hi_j"] - exclusion
    require(lo_i <= hi_i and lo_j <= hi_j,
            "relaxation exclusion empties the moving-child trusted interior")
    if aperture is None:
        return region_record(grid, lo_i, lo_j, hi_i, hi_j,
                             "moving_child_trusted_interior_v1")
    validate_aperture(aperture)
    interior = aperture["operational_interior"]
    shape = {"x": hi_i - lo_i + 1, "y": hi_j - lo_j + 1}
    require(exclusion == interior["relaxation_exclusion_cells"] and
            shape == interior["required_shape_cells"],
            "operational child interior differs from the bound qualification aperture")
    offset = interior["aperture_offset_cells"]
    aperture_shape = aperture["shape_cells"]
    aperture_lo_i = lo_i + offset["x"]
    aperture_lo_j = lo_j + offset["y"]
    aperture_hi_i = aperture_lo_i + aperture_shape["x"] - 1
    aperture_hi_j = aperture_lo_j + aperture_shape["y"] - 1
    require(aperture_hi_i <= hi_i and aperture_hi_j <= hi_j,
            "qualification aperture exceeds the operational trusted interior")
    result = region_record(
        grid, aperture_lo_i, aperture_lo_j, aperture_hi_i, aperture_hi_j,
        APERTURE_REGION_ID)
    require(result["cell_count"] == aperture["cell_count"] and
            result["horizontal_grid_spacing_m"] ==
                aperture["horizontal_grid_spacing_m"] and
            result["grid_phase_origin_m"] == aperture["grid_phase_origin_m"],
            "realized qualification aperture differs from its bound geometry")
    result["aperture_id"] = aperture["aperture_id"]
    result["operational_interior_source_index_range"] = {
        "lower": {"i": lo_i, "j": lo_j},
        "upper": {"i": hi_i, "j": hi_j},
    }
    return result


def region_record(grid: dict[str, Any], lo_i: int, lo_j: int,
                  hi_i: int, hi_j: int,
                  region_id: str = "moving_child_trusted_interior_v1") -> dict[str, Any]:
    dx, dy = grid["dx"], grid["dy"]
    return {
        "id": region_id,
        "earth_bounds_m": {
            "west": grid["scalar_x"] + (lo_i - 0.5) * dx,
            "east": grid["scalar_x"] + (hi_i + 0.5) * dx,
            "south": grid["scalar_y"] + (lo_j - 0.5) * dy,
            "north": grid["scalar_y"] + (hi_j + 0.5) * dy,
        },
        "source_index_range": {
            "lower": {"i": lo_i, "j": lo_j},
            "upper": {"i": hi_i, "j": hi_j},
        },
        "horizontal_grid_spacing_m": {"x": dx, "y": dy},
        "grid_phase_origin_m": {"x": grid["phase_x"], "y": grid["phase_y"]},
        "cell_count": (hi_i - lo_i + 1) * (hi_j - lo_j + 1),
    }


def reference_region(manifest: dict[str, Any], requested: dict[str, Any]) -> dict[str, Any]:
    grid = grid_contract(manifest)
    spacing = requested.get("horizontal_grid_spacing_m", {})
    phase = requested.get("grid_phase_origin_m", {})
    require(grid["dx"] == finite_number(spacing.get("x"), "region dx") and
            grid["dy"] == finite_number(spacing.get("y"), "region dy"),
            "reference grid spacing differs from the moving trusted region")
    require(math.isclose(grid["phase_x"], finite_number(phase.get("x"), "region phase x"),
                         rel_tol=0.0, abs_tol=1.0e-9) and
            math.isclose(grid["phase_y"], finite_number(phase.get("y"), "region phase y"),
                         rel_tol=0.0, abs_tol=1.0e-9),
            "reference grid phase differs from the moving trusted region")
    bounds = requested.get("earth_bounds_m")
    require(isinstance(bounds, dict), "moving trusted region lacks Earth bounds")
    west = finite_number(bounds.get("west"), "region west")
    east = finite_number(bounds.get("east"), "region east")
    south = finite_number(bounds.get("south"), "region south")
    north = finite_number(bounds.get("north"), "region north")
    lo_i = round((west - grid["scalar_x"]) / grid["dx"] + 0.5)
    hi_i = round((east - grid["scalar_x"]) / grid["dx"] - 0.5)
    lo_j = round((south - grid["scalar_y"]) / grid["dy"] + 0.5)
    hi_j = round((north - grid["scalar_y"]) / grid["dy"] - 0.5)
    region = region_record(
        grid, lo_i, lo_j, hi_i, hi_j,
        str(requested.get("id", APERTURE_REGION_ID)))
    require(region["earth_bounds_m"] == requested.get("earth_bounds_m") and
            region["horizontal_grid_spacing_m"] == spacing and
            region["grid_phase_origin_m"] == phase,
            "moving trusted region is not exactly representable on the reference grid")
    require(grid["lo_i"] <= lo_i <= hi_i <= grid["hi_i"] and
            grid["lo_j"] <= lo_j <= hi_j <= grid["hi_j"],
            "moving trusted region is outside the reference physical domain")
    if "aperture_id" in requested:
        region["aperture_id"] = requested["aperture_id"]
    return region


def selected_values(values: array.array, shape: tuple[int, ...],
                    region: dict[str, Any]) -> list[tuple[int, int, float]]:
    require(len(shape) == 2, "surface diagnostic is not two-dimensional")
    lower = region["source_index_range"]["lower"]
    upper = region["source_index_range"]["upper"]
    result: list[tuple[int, int, float]] = []
    for i in range(lower["i"], upper["i"] + 1):
        for j in range(lower["j"], upper["j"] + 1):
            require(0 <= i < shape[0] and 0 <= j < shape[1],
                    "trusted-region index is outside a diagnostic array")
            result.append((i, j, float(values[i * shape[1] + j])))
    require(len(result) == region["cell_count"], "trusted-region cell count mismatch")
    return result


def diagnostic_records(
    step: Path, manifest: dict[str, Any], region: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    published = manifest.get("published_fields")
    require(isinstance(published, list) and
            set(published) == {"cold_pool_boundary", "accumulated_rainfall"},
            f"publication does not contain exactly the moving-reference diagnostics: {step}")
    cold_descriptor = field_descriptor(manifest, "cold_pool_boundary")
    require(cold_descriptor.get("units") == "K" and
            COLD_POOL_ALGORITHM in str(cold_descriptor.get("description", "")),
            "cold-pool publication lacks the all-species algorithm identity")
    rain_descriptor = field_descriptor(manifest, "accumulated_rainfall")
    require(rain_descriptor.get("units") == "mm" and
            rain_descriptor.get("derived") is True and
            rain_descriptor.get("grid_location") == "surface_cell_center" and
            rain_descriptor.get("state_semantic") == "diagnostic",
            "accumulated rainfall publication has an incompatible field descriptor")
    manifest_precipitation_algorithm = rain_descriptor.get("algorithm_id")
    if manifest_precipitation_algorithm is None:
        manifest_precipitation_algorithm = "unavailable_in_manifest_v6"
    else:
        require(manifest_precipitation_algorithm == PRECIPITATION_ALGORITHM,
                "accumulated rainfall manifest algorithm identity changed")
    cold_shape, cold_values = read_npy_float32(step / "cold_pool_boundary.npy")
    rain_shape, rain_values = read_npy_float32(step / "accumulated_rainfall.npy")
    require(cold_shape == rain_shape, "cold-pool/rainfall surface shapes differ")
    cold = selected_values(cold_values, cold_shape, region)
    rain = selected_values(rain_values, rain_shape, region)
    spacing = region["horizontal_grid_spacing_m"]
    area = spacing["x"] * spacing["y"]
    grid = grid_contract(manifest)
    cold_active = [(i, j, value) for i, j, value in cold if value >= 2.0]
    total_weight = sum(value for _, _, value in cold if value > 0.0)
    centroid = None
    if total_weight > 0.0:
        centroid = {
            "x": sum((grid["scalar_x"] + i * grid["dx"]) * value
                     for i, _, value in cold if value > 0.0) / total_weight,
            "y": sum((grid["scalar_y"] + j * grid["dy"]) * value
                     for _, j, value in cold if value > 0.0) / total_weight,
        }
    depths = sorted(value for _, _, value in rain)
    require(all(value >= 0.0 for value in depths),
            "effective accumulated rainfall contains a negative depth")
    rank95 = max(1, math.ceil(0.95 * len(depths))) - 1
    return (
        {
            "definition": COLD_POOL_ALGORITHM,
            "area_m2": len(cold_active) * area,
            "integrated_deficit_k_m2": sum(value for _, _, value in cold) * area,
            "minimum_theta_rho_perturbation_k": -max(value for _, _, value in cold),
            "centroid_m": centroid,
        },
        {
            "definition": PRECIPITATION_ALGORITHM,
            "effective_depth_semantics":
                "residual_inclusive_then_float32_published_v1",
            "accumulated_mass_kg": sum(depths) * area,
            "footprint_area_m2": sum(value >= 1.0 for value in depths) * area,
            "p95_depth_mm": depths[rank95],
            "maximum_depth_mm": depths[-1],
        },
        manifest_precipitation_algorithm,
    )


def artifact_record(path: Path, relative_to: Path) -> dict[str, str]:
    try:
        rendered = path.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        rendered = str(path.resolve())
    return {"path": rendered, "sha256": sha256(path)}


def extract(run: Path, role: str, config: Path, binary: Path, policy_path: Path,
            output: Path, source_manifest_path: Path,
            region_series_path: Path | None = None) -> dict[str, Any]:
    require(role in {"moving_child", "oversized_reference"}, "invalid series role")
    run = run.resolve()
    config = config.resolve()
    binary = binary.resolve()
    output = output.resolve()
    source_manifest_path = source_manifest_path.resolve()
    policy_path = policy_path.resolve()
    policy = load_json(policy_path)
    if policy.get("schema_version") in {6, 7, 8, 9, 10, 11, 12, 13, 14}:
        try:
            policy = comparator.validate_policy(policy, policy_path)
        except comparator.ContractError as exc:
            raise ExtractionError(str(exc)) from exc
    policy_schema_version = policy.get("schema_version")
    movement_algorithm = (
        PARENT_STENCIL_LIMIT_MOVEMENT_ALGORITHM
        if policy_schema_version in {12, 13, 14} else
        ORDINARY_TRUSTED_IMPROVEMENT_MOVEMENT_ALGORITHM
        if policy_schema_version == 11 else
        QUANTIZED_TRUSTED_BALANCE_MOVEMENT_ALGORITHM
        if policy_schema_version == 10 else
        RECEDING_TRUSTED_FACE_MOVEMENT_ALGORITHM
        if policy_schema_version == 9 else
        ORTHOGONAL_CONTAINMENT_RESERVE_MOVEMENT_ALGORITHM
        if policy_schema_version == 8 else
        BALANCED_TRUSTED_CONTAINMENT_MOVEMENT_ALGORITHM
        if policy_schema_version == 7 else
        TRUSTED_INTERIOR_CONTAINMENT_MOVEMENT_ALGORITHM
        if policy_schema_version == 6 else
        MAXIMUM_SAFE_PROJECTED_MOVEMENT_ALGORITHM
        if policy_schema_version == 5 else
        PROJECTED_CONTAINMENT_MOVEMENT_ALGORITHM
        if policy_schema_version == 4 else MOVEMENT_ALGORITHM)
    times, expected = expected_times(policy)
    geometry_plan_path: Path | None = None
    aperture_path: Path | None = None
    geometry_plan: dict[str, Any] | None = None
    aperture: dict[str, Any] | None = None
    if policy.get("schema_version") in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
        geometry_plan_path, geometry_plan = bound_policy_document(
            policy_path, expected.get("geometry_plan"), "geometry_plan",
            GEOMETRY_PLAN_SCHEMA, "plan_id")
        aperture_path, aperture = bound_policy_document(
            policy_path, expected.get("qualification_aperture"),
            "qualification_aperture", APERTURE_SCHEMA, "aperture_id")
        validate_geometry_plan(geometry_plan)
        validate_geometry_plan_sources(geometry_plan_path, geometry_plan)
        validate_aperture(aperture)
        validate_configured_geometry(config, role, geometry_plan)
        requirement = geometry_plan["qualification_aperture_requirement"]
        require(requirement["aperture_id"] == aperture["aperture_id"] and
                requirement["shape_cells"] == aperture["shape_cells"] and
                requirement["cell_count"] == aperture["cell_count"] and
                requirement["horizontal_grid_spacing_m"] ==
                    aperture["horizontal_grid_spacing_m"] and
                requirement["grid_phase_origin_m"] ==
                    aperture["grid_phase_origin_m"],
                "geometry plan and qualification aperture disagree")

    run_manifest_path = run / "nimbus_output.json"
    run_manifest = load_json(run_manifest_path)
    require(run_manifest.get("schema") == "nimbus.output" and
            run_manifest.get("manifest_version") in {6, 7} and
            run_manifest.get("publication_state") == "ready",
            "run publication is not complete")
    config_snapshot = run / "qualification_input.yaml"
    require(sha256(config_snapshot) == sha256(config),
            "run config snapshot differs from the requested configuration")
    binary_digest = sha256(binary)
    recorded_binary = (run / "qualification_binary.sha256").read_text(
        encoding="ascii").strip()
    require(HEX64.fullmatch(recorded_binary) is not None and
            recorded_binary == binary_digest,
            "run binary digest differs from the requested binary")

    history = (run / "nests" / "child_0000" / "qualification_diagnostics"
               if role == "moving_child"
               else run / "qualification_diagnostics")
    expected_data_product = (
        "nested_child_qualification_diagnostics"
        if role == "moving_child" else "qualification_diagnostics")
    products = run_manifest.get("data_products")
    require(isinstance(products, list) and
            expected_data_product in products,
            "run manifest does not advertise the sparse qualification stream")
    steps = step_publications(history, expected_data_product)
    trackers = tracker_publications(run)
    extra_steps = set(steps) - set(times)
    extra_trackers = set(trackers) - set(times)
    require(all(value < times[0] for value in extra_steps),
            "run has diagnostic publications outside policy cadence")
    require(all(value < times[0] for value in extra_trackers),
            "run has tracker records outside policy cadence")
    missing_steps = [time for time in times if time not in steps]
    missing_trackers = [time for time in times if time not in trackers]
    require(not missing_steps, f"run lacks {len(missing_steps)} required diagnostic publications")
    require(not missing_trackers, f"run lacks {len(missing_trackers)} required tracker records")

    exclusion = expected.get("relaxation_exclusion_cells")
    require(type(exclusion) is int and exclusion >= 0,
            "policy relaxation exclusion must be a nonnegative integer")
    requested_regions: dict[float, dict[str, Any]] = {}
    if role == "oversized_reference":
        require(region_series_path is not None,
                "oversized reference extraction requires --region-series")
        region_document = load_json(region_series_path.resolve())
        require(region_document.get("schema") == SERIES_SCHEMA and
                region_document.get("schema_version") == 2 and
                region_document.get("role") == "moving_child",
                "region series is not a v2 moving-child extraction")
        for sample in region_document.get("samples", []):
            require(isinstance(sample, dict), "invalid moving region sample")
            requested_regions[finite_number(sample.get("simulation_time_s"),
                                            "moving region time")] = sample.get("region")
        require(set(requested_regions) == set(times),
                "moving region series does not exactly cover policy cadence")

    samples: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    tracker_records: list[dict[str, Any]] = []
    diagnostic_field_records: list[dict[str, Any]] = []
    precipitation_manifest_algorithms: set[str] = set()
    phase: dict[str, float] | None = None
    spacing: dict[str, float] | None = None
    runtime_tracking_source = "synchronized_nested_child" if role == "moving_child" else "root_grid"
    detector_frames: set[str] = set()
    for time_s in times:
        step_path, manifest = steps[time_s]
        tracker_path, tracker = trackers[time_s]
        require(tracker.get("tracking_source") == runtime_tracking_source,
                f"wrong runtime tracking source at {time_s}s")
        detector_frame = str(tracker.get("detector_coordinate_frame"))
        detector_frames.add(detector_frame)
        if role == "moving_child":
            require(detector_frame == "earth_absolute",
                    f"moving tracker is not Earth-fixed at {time_s}s")
            authority = tracker.get("movement_authority")
            require(isinstance(authority, dict) and
                    authority.get("algorithm") == movement_algorithm,
                    f"moving sample lacks the policy movement authority at {time_s}s")
            if policy_schema_version >= 12:
                bounds = authority.get("legal_parent_shift_bounds_cells")
                require(isinstance(bounds, dict) and
                        bounds.get("available") is True,
                        f"moving sample lacks live parent-stencil bounds at {time_s}s")
                bound_values = [bounds.get(name) for name in (
                    "minimum_i", "maximum_i", "minimum_j", "maximum_j")]
                require(all(type(value) is int for value in bound_values) and
                        bound_values[0] <= 0 <= bound_values[1] and
                        bound_values[2] <= 0 <= bound_values[3],
                        f"moving sample has malformed parent-stencil bounds at {time_s}s")
            validity = tracker.get("moving_nest_scientific_validity")
            require(isinstance(validity, dict) and validity.get("status") != "invalid",
                    f"moving sample is scientifically invalid at {time_s}s")
            region = moving_region(manifest, exclusion, aperture)
        else:
            region = reference_region(manifest, requested_regions[time_s])
        (cold_pool, precipitation,
         precipitation_manifest_algorithm) = diagnostic_records(
            step_path, manifest, region)
        precipitation_manifest_algorithms.add(precipitation_manifest_algorithm)
        current_phase = region["grid_phase_origin_m"]
        current_spacing = region["horizontal_grid_spacing_m"]
        phase = current_phase if phase is None else phase
        spacing = current_spacing if spacing is None else spacing
        require(current_phase == phase and current_spacing == spacing,
                "sample grid phase or spacing changed during extraction")

        selected = tracker.get("selected_feature_available")
        require(type(selected) is bool, f"tracker selection is invalid at {time_s}s")
        sample: dict[str, Any] = {
            "simulation_time_s": time_s,
            "selected": selected,
            "region": region,
            "core_centroid_m": None,
            "maximum_updraft_ms": None,
            "maximum_absolute_uh_m2s2": None,
            "cold_pool": cold_pool,
            "precipitation": precipitation,
        }
        if selected:
            centroid = tracker.get("core_centroid_m")
            rotation = tracker.get("rotation")
            require(isinstance(centroid, dict) and isinstance(rotation, dict),
                    f"selected tracker lacks intensity/path data at {time_s}s")
            centroid_x = finite_number(centroid.get("x"), "core centroid x")
            centroid_y = finite_number(centroid.get("y"), "core centroid y")
            if detector_frame == "grid_relative":
                require(role == "oversized_reference",
                        "only the stationary reference may require coordinate conversion")
                frame = manifest["grid"]["horizontal_frame"]
                velocity = frame.get("frame_velocity_earth_ms")
                origin = frame.get("grid_origin_earth_m")
                require(frame.get("motion_mode") == "stationary" and
                        isinstance(velocity, list) and len(velocity) >= 2 and
                        finite_number(velocity[0], "reference frame velocity x") == 0.0 and
                        finite_number(velocity[1], "reference frame velocity y") == 0.0 and
                        isinstance(origin, list) and len(origin) >= 2,
                        "grid-relative reference tracker is not on a stationary Earth-fixed grid")
                centroid_x += finite_number(origin[0], "reference Earth origin x")
                centroid_y += finite_number(origin[1], "reference Earth origin y")
            sample["core_centroid_m"] = {"x": centroid_x, "y": centroid_y}
            sample["maximum_updraft_ms"] = finite_number(
                tracker.get("maximum_updraft_ms"), "maximum updraft")
            sample["maximum_absolute_uh_m2s2"] = abs(finite_number(
                rotation.get("maximum_absolute_updraft_helicity_m2s2"),
                "maximum absolute UH"))
        samples.append(sample)
        manifest_records.append({
            "simulation_time_s": time_s,
            **artifact_record(step_path / "manifest.json", source_manifest_path.parent),
        })
        tracker_records.append({
            "simulation_time_s": time_s,
            **artifact_record(tracker_path, source_manifest_path.parent),
        })
        diagnostic_field_records.append({
            "simulation_time_s": time_s,
            "cold_pool_boundary": artifact_record(
                step_path / "cold_pool_boundary.npy", source_manifest_path.parent),
            "accumulated_rainfall": artifact_record(
                step_path / "accumulated_rainfall.npy", source_manifest_path.parent),
        })

    require(spacing is not None, "series lacks an effective comparison spacing")
    (signature_values, signature_digest,
     signature_derivation) = numerical_signature(config, role, spacing)
    if policy.get("schema_version") in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
        require(signature_digest == expected.get("physics_signature_sha256"),
                "comparison-level numerical signature differs from ratified policy")
        expected_phase = expected.get("grid_phase_origin_m")
        require(isinstance(expected_phase, dict) and phase == expected_phase,
                "comparison grid phase differs from ratified policy")
        trusted_count = expected.get("trusted_region_cell_count")
        trusted_shape = expected.get("trusted_region_shape_cells")
        require(type(trusted_count) is int and isinstance(trusted_shape, dict),
                "ratified policy lacks a trusted-region shape")
        for sample in samples:
            region = sample["region"]
            lower = region["source_index_range"]["lower"]
            upper = region["source_index_range"]["upper"]
            shape = {
                "x": upper["i"] - lower["i"] + 1,
                "y": upper["j"] - lower["j"] + 1,
            }
            require(region["cell_count"] == trusted_count and
                    shape == trusted_shape,
                    "trusted-region geometry differs from ratified policy")
    require(len(precipitation_manifest_algorithms) == 1,
            "accumulated rainfall manifest algorithm provenance changed across samples")
    source_files = [artifact_record(ROOT / value, source_manifest_path.parent)
                    | {"repository_path": value} for value in SOURCE_FILES]
    source_manifest: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "schema_version": 2 if policy_schema_version in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} else 1,
        "role": role,
        "policy_status": policy["policy_status"],
        "policy": artifact_record(policy_path.resolve(), source_manifest_path.parent),
        "binary": artifact_record(binary, source_manifest_path.parent),
        "config": artifact_record(config_snapshot, source_manifest_path.parent),
        "run_manifest": artifact_record(run_manifest_path, source_manifest_path.parent),
        "numerical_signature": {
            "schema": "nimbus.numerical_signature",
            "schema_version": 1,
            "sha256": signature_digest,
            "values": signature_values,
            "derivation": signature_derivation,
        },
        "diagnostics": {
            "cold_pool": {
                "algorithm": COLD_POOL_ALGORITHM,
                "water_species": ["qv", "qc", "qr", "qi", "qs", "qg", "qh"],
                "formula": "theta*(1+qv/epsilon)/(1+qv+qc+qr+qi+qs+qg+qh)",
                "threshold_k": 2.0,
            },
            "precipitation": {
                "algorithm": PRECIPITATION_ALGORITHM,
                "effective_depth_semantics":
                    "residual_inclusive_then_float32_published_v1",
                "manifest_algorithm_identity":
                    next(iter(precipitation_manifest_algorithms)),
                "footprint_threshold_mm": 1.0,
            },
        },
        "diagnostic_stream": {
            "preset": "moving_reference_v1",
            "data_product_kind": expected_data_product,
            "storage_format": "npy_3d",
            "published_fields": [
                "cold_pool_boundary", "accumulated_rainfall"],
            "interval_s":
                finite_number(expected.get("sample_interval_s"),
                              "policy sample interval"),
        },
        "coordinate_conversion": {
            "output_frame": "earth_absolute",
            "runtime_detector_frames": sorted(detector_frames),
            "grid_relative_reference_rule":
                "stationary_zero_velocity_grid_origin_translation_v1",
        },
        "implementation_sources": source_files,
        "step_manifests": manifest_records,
        "storm_records": tracker_records,
        "diagnostic_fields": diagnostic_field_records,
        "region_series": (
            artifact_record(region_series_path.resolve(), source_manifest_path.parent)
            if region_series_path is not None else None
        ),
    }
    if policy_schema_version in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
        source_manifest.update({
            "policy": {
                **artifact_record(policy_path, source_manifest_path.parent),
                "policy_id": policy.get("policy_id"),
                "schema_version": policy_schema_version,
                "policy_status": policy.get("policy_status"),
            },
            "extraction_disposition": "measurement_only",
        })
    if policy_schema_version in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
        assert geometry_plan_path is not None and aperture_path is not None
        assert geometry_plan is not None and aperture is not None
        source_manifest.update({
            "geometry_plan": {
                **artifact_record(geometry_plan_path, source_manifest_path.parent),
                "schema_version": 1,
                "plan_id": geometry_plan["plan_id"],
            },
            "qualification_aperture": {
                **artifact_record(aperture_path, source_manifest_path.parent),
                "schema_version": 1,
                "aperture_id": aperture["aperture_id"],
            },
        })
    source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    contract: dict[str, Any] = {
        "duration_s": times[-1],
        "sample_start_s": times[0],
        "sample_interval_s": times[1] - times[0] if len(times) > 1 else
            finite_number(expected.get("sample_interval_s"), "sample interval"),
        "coordinate_frame": "earth_absolute",
        "detector_algorithm": DETECTOR_ALGORITHM,
        "continuity_algorithm": CONTINUITY_ALGORITHM,
        "physics_signature_sha256": signature_digest,
        "tracking_source": ("synchronized_nested_child" if role == "moving_child"
                            else "oversized_stationary_reference"),
        "comparison_region": (
            "moving_child_qualification_aperture"
            if policy_schema_version in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} else "moving_child_trusted_interior"),
        "relaxation_exclusion_cells": exclusion,
        "horizontal_grid_spacing_m": spacing,
        "grid_phase_origin_m": phase,
    }
    if policy_schema_version in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
        contract["geometry_plan"] = expected["geometry_plan"]
        contract["qualification_aperture"] = expected["qualification_aperture"]
    if role == "moving_child":
        contract["movement_algorithm"] = movement_algorithm
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": SERIES_SCHEMA,
        "schema_version": 2,
        "role": role,
        "artifacts": {
            "binary": artifact_record(binary, output.parent),
            "config": artifact_record(config_snapshot, output.parent),
            "run_manifest": artifact_record(run_manifest_path, output.parent),
            "source_manifest": artifact_record(source_manifest_path, output.parent),
        },
        "contract": contract,
        "samples": samples,
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--role", choices=("moving_child", "oversized_reference"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--region-series", type=Path)
    args = parser.parse_args()
    source_manifest = args.source_manifest
    if source_manifest is None:
        source_manifest = args.output.with_suffix(".source_manifest.json")
    extract(args.run, args.role, args.config, args.binary, args.policy,
            args.output, source_manifest, args.region_series)
    print(f"[moving-reference-extractor] PASS: {args.role} -> {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExtractionError, OSError, ValueError) as exc:
        print(f"[moving-reference-extractor] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)

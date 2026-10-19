#!/usr/bin/env python3
"""Compare a moving nested storm against an oversized stationary reference.

The comparator consumes prepared, versioned scalar time-series artifacts. It
does not infer unavailable science products from minimal history output. Every
source artifact is hash-bound, and absent required metric families produce an
explicit incomplete result rather than a qualification pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = (
    ROOT
    / "tests"
    / "qualification"
    / "fixtures"
    / "moving_nested_supercell_reference_frontier_v12.json"
)
INPUT_SCHEMA = "nimbus.qualification.moving_nested_reference_series"
POLICY_SCHEMA = "nimbus.qualification.moving_nested_reference_frontier.policy"
REPORT_SCHEMA = "nimbus.qualification.moving_nested_reference_frontier"
REQUIRED_ARTIFACTS = {"binary", "config", "run_manifest", "source_manifest"}
HEX_DIGITS = frozenset("0123456789abcdef")
SOURCE_SCHEMA = "nimbus.qualification.moving_nested_reference_source_manifest"
COLD_POOL_ALGORITHM = "surface_density_potential_temperature_deficit_v1"
PRECIPITATION_ALGORITHM = "earth_aligned_accumulated_rainfall_v1"
GEOMETRY_PLAN_SCHEMA = "nimbus.qualification.moving_nested_geometry_plan"
APERTURE_SCHEMA = "nimbus.qualification.moving_nested_fixed_aperture"
APERTURE_REGION_ID = "moving_child_qualification_aperture_v1"
BLINDED_V2_POLICY_SHA256 = (
    "130f39702e3c2d05823c98b6ba40dab5efe42d745e3766ffe55099fe32e76229"
)
GEOMETRY_PLAN_SHA256 = (
    "59ba6ddbb2f17e27f0287d565b28d814420dce87c9152a904e555a8cb0fd559c"
)
QUALIFICATION_APERTURE_SHA256 = (
    "969ae21f9369d8d414fe9eea18208c3586a31faf21596125f7288a0c0779f6a9"
)
GEOMETRY_FAILURE_SHA256 = (
    "dd020dd0d1c83850a3074d52e4928bd040a7d10f7ac597dffb9d62e6dfaf9cb4"
)
V3_POLICY_SHA256 = (
    "c39190880d0ad278b8fab063b9cf4d236b0cbef425af7ad5cb10716960305da1"
)
PROJECTED_CONTAINMENT_FAILURE_SHA256 = (
    "8cd67e341affcad8a72bd030e15ded49b31e32290591a224ed873aad972c0ca5"
)
V4_POLICY_SHA256 = (
    "9d059cebf54b49c024f6c78ea68026a734ac29e318e7901ef5366769525ea777"
)
POST_COMMIT_CONTAINMENT_FAILURE_SHA256 = (
    "54f2b0f3c5f1c0bb2dbfb3a9df7c4d32e421c7cfe78299bc4a6c517d76b4fef3"
)
V5_POLICY_SHA256 = (
    "1f12775281252f3e82792ef6400d277bd1bbd182c173c54f2dfe5c2779bc58a1"
)
TRUSTED_CONTAINMENT_FAILURE_SHA256 = (
    "907bc9df5f3707dab0bb445e1867857e577d7f42944160f6db37a6fb819832e6"
)
V6_POLICY_SHA256 = (
    "6fb2a44972efba53f5a1549bf5f822a7c1ff68f9d8a3f7ea0aa0c020ee7390e3"
)
BALANCED_CONTAINMENT_PROBE_FAILURE_SHA256 = (
    "0c104fb56c3c88059b3f131f7dbe14dfd3baf23c7252a372f5806c913e574fa8"
)
V7_POLICY_SHA256 = (
    "3a058e68e1b6d04e660ab3eefae9d1d19a99e9efc1fbdef88ad720d722270fa1"
)
ORTHOGONAL_CONTAINMENT_RESERVE_FAILURE_SHA256 = (
    "f65bfc701a9db5464052fd5f1d42ee6a9a1bf7d2724988f2cf8fb333af8bf0c9"
)
V8_POLICY_SHA256 = (
    "71dc4c453eb5fdf9dd359b47294d9f31f7099436999536b6fdd567ce92dadd2a"
)
RECEDING_TRUSTED_FACE_FAILURE_SHA256 = (
    "7cd978a0669fc103468907ba9eac3544352dd38b86bb75845ba330f67c2f5eed"
)
V9_POLICY_SHA256 = (
    "aaa4aa0d6e07a81fc180e510d2a6bf6506051f52ecf82a96afb7bfe476edee2f"
)
QUANTIZED_TRUSTED_BALANCE_FAILURE_SHA256 = (
    "32e1540915400bfee3291de6e2a01adb2f587cb64890a130d49bdd3ae0130486"
)
V10_POLICY_SHA256 = (
    "2b42d72fe746859715781a0b22a807660e3cf413bd78d911a12d12a2be156ec7"
)
ORDINARY_TRUSTED_IMPROVEMENT_FAILURE_SHA256 = (
    "d5421120a8eba5648469f82595d00d0db7585cd437594b44ebcb357cb287cdb8"
)
V11_POLICY_SHA256 = (
    "9fbd879e6f3ccfd3b4968df102c5c9a6a3e2a0a26a83abc35f08d2886564e66a"
)
PARENT_STENCIL_LIMIT_FAILURE_SHA256 = (
    "1bd0b4bde252f5815e3378fe2811558693b20bbd9daa47b8c682f716db87cc17"
)
V12_POLICY_SHA256 = (
    "ba4a8f76f196be809f5b036f77a27ec49880ccaa476bd151119252110890c9cc"
)
VULKAN_REFERENCE_RUNTIME_EVIDENCE_SHA256 = (
    "542cd3404e223064aa45e1f19d5b27a6d309c7728c30f33d96656dd5b6ff3c85"
)
VULKAN_PHYSICS_SIGNATURE_SHA256 = (
    "7e5c77671bb7f58bee72416efc54f6753fd2a3ce67593b5ebf4c3cea0ca14ada"
)
V13_POLICY_SHA256 = (
    "663a930995e2fa0a81ca89c172db35b6822e2aeafb23bf8f203b10846ed49b3f"
)
METAL_PHYSICS_SIGNATURE_SHA256 = (
    "bfb35530da9a4ed20984f43f69d1b96436cb9b803d79ddb7da4a470a81568581"
)


class ContractError(ValueError):
    """The supplied artifacts cannot participate in this comparison."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_evidence_path(raw_path: object) -> Path:
    """Relocate sealed repository references without changing their byte hashes."""
    require(isinstance(raw_path, str) and raw_path,
            "evidence-bound artifact path must be nonempty")
    relative = Path(raw_path)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "evidence-bound repository path must not be absolute or traverse parents")
    root = ROOT.resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root),
            "evidence-bound repository path leaves the repository")
    if path.is_file():
        return path
    archived_reports = {
        "tests/artifacts/moving_nested_vulkan_lifetime_v2_20260819_chain_report.json":
            "moving_nested_vulkan_lifetime_v2_20260819_chain_report.json",
        "tests/artifacts/moving_nested_vulkan_reference_v1_20260819/moving_nested_vulkan_reference_runtime_report.json":
            "moving_nested_vulkan_reference_runtime_report.json",
    }
    if relative.as_posix() in archived_reports:
        destination = root / "tests/fixtures/reference/domain_history"
        relocated = (destination / archived_reports[relative.as_posix()]).resolve()
        require(relocated.is_relative_to(destination),
                "relocated evidence-bound artifact leaves its fixture directory")
        return relocated
    for retired, retained in (
        ("configs/qualification", "tests/fixtures/configs/qualification"),
        ("docs/evidence", "tests/fixtures/reference/domain_history"),
    ):
        try:
            fixture_name = relative.relative_to(retired)
        except ValueError:
            continue
        destination = root / retained
        relocated = (destination / fixture_name).resolve()
        require(relocated.is_relative_to(destination),
                "relocated evidence-bound artifact leaves its fixture directory")
        return relocated
    return path


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_DIGITS)
    )


def finite_number(value: object, label: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)),
            f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def nonnegative_number(value: object, label: str) -> float:
    result = finite_number(value, label)
    require(result >= 0.0, f"{label} must be nonnegative")
    return result


def positive_number(value: object, label: str) -> float:
    result = finite_number(value, label)
    require(result > 0.0, f"{label} must be positive")
    return result


def object_value(value: object, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def coordinate(value: object, label: str) -> tuple[float, float]:
    item = object_value(value, label)
    return (
        finite_number(item.get("x"), f"{label}.x"),
        finite_number(item.get("y"), f"{label}.y"),
    )


def resolve_artifacts(document: dict[str, Any], source: Path) -> dict[str, dict[str, str]]:
    artifacts = object_value(document.get("artifacts"), f"{source}: artifacts")
    require(set(artifacts) == REQUIRED_ARTIFACTS,
            f"{source}: artifacts must be exactly {sorted(REQUIRED_ARTIFACTS)}")
    verified: dict[str, dict[str, str]] = {}
    for name in sorted(REQUIRED_ARTIFACTS):
        record = object_value(artifacts[name], f"{source}: artifacts.{name}")
        require(set(record) == {"path", "sha256"},
                f"{source}: artifacts.{name} must contain only path and sha256")
        raw_path = record.get("path")
        expected = record.get("sha256")
        require(isinstance(raw_path, str) and raw_path,
                f"{source}: artifacts.{name}.path must be nonempty")
        require(valid_sha256(expected),
                f"{source}: artifacts.{name}.sha256 is invalid")
        path = Path(raw_path)
        if not path.is_absolute():
            path = source.parent / path
        path = path.resolve()
        require(path.is_file(), f"{source}: missing {name} artifact {path}")
        actual = sha256(path)
        require(actual == expected,
                f"{source}: {name} artifact hash mismatch: expected {expected}, got {actual}")
        verified[name] = {"path": str(path), "sha256": actual}
    return verified


def resolve_bound_record(record: object, owner: Path, label: str) -> dict[str, str]:
    item = object_value(record, label)
    require(set(item) >= {"path", "sha256"},
            f"{label} must contain path and sha256")
    raw_path = item.get("path")
    expected = item.get("sha256")
    require(isinstance(raw_path, str) and raw_path, f"{label}.path must be nonempty")
    require(valid_sha256(expected), f"{label}.sha256 is invalid")
    path = Path(raw_path)
    if not path.is_absolute():
        path = owner.parent / path
    for index in range(len(path.parts) - 1):
        if path.parts[index:index + 2] in {
                ("docs", "evidence"), ("configs", "qualification")}:
            require(".." not in path.parts[index + 2:],
                    f"{label}: retired evidence namespace must not traverse parents")
    path = Path(os.path.abspath(path))
    try:
        repository_relative = path.relative_to(ROOT.resolve())
    except ValueError:
        path = path.resolve()
    else:
        path = repository_evidence_path(str(repository_relative))
    require(path.is_file(), f"{label} is missing: {path}")
    actual = sha256(path)
    require(actual == expected,
            f"{label} hash mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}


def validate_source_manifest(path: Path, role: str,
                             artifacts: dict[str, dict[str, str]]) -> dict[str, Any]:
    source = load_json(path)
    schema_version = source.get("schema_version")
    require(source.get("schema") == SOURCE_SCHEMA and schema_version in {1, 2},
            f"{path}: wrong source-manifest schema")
    require(source.get("role") == role, f"{path}: source-manifest role mismatch")
    if schema_version == 1:
        require(source.get("policy_status") == "measurement_only",
                f"{path}: v1 source manifest is not measurement-only")
    else:
        require(source.get("policy_status") == "ratified" and
                source.get("extraction_disposition") == "measurement_only",
                f"{path}: v2 source policy/disposition is invalid")
        policy_record = object_value(source.get("policy"), f"{path}: policy")
        policy_schema_version = policy_record.get("schema_version")
        require(policy_schema_version in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} and
                policy_record.get("policy_status") == "ratified" and
                isinstance(policy_record.get("policy_id"), str) and
                policy_record["policy_id"],
                f"{path}: v2 policy binding is incomplete")
        source["_verified_policy"] = resolve_bound_record(
            policy_record, path, f"{path}: policy")
        if policy_schema_version in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            geometry_record = object_value(
                source.get("geometry_plan"), f"{path}: geometry_plan")
            aperture_record = object_value(
                source.get("qualification_aperture"),
                f"{path}: qualification_aperture")
            require(geometry_record.get("schema_version") == 1 and
                geometry_record.get("plan_id") ==
                    "moving_nested_reference_geometry_v1",
                f"{path}: geometry-plan binding is incomplete")
            require(aperture_record.get("schema_version") == 1 and
                aperture_record.get("aperture_id") ==
                    "moving_nested_fixed_100x96_v1",
                f"{path}: qualification-aperture binding is incomplete")
            source["_verified_geometry_plan"] = resolve_bound_record(
                geometry_record, path, f"{path}: geometry_plan")
            source["_verified_qualification_aperture"] = resolve_bound_record(
                aperture_record, path, f"{path}: qualification_aperture")
            geometry_document = load_json(
                Path(source["_verified_geometry_plan"]["path"]))
            aperture_document = load_json(
                Path(source["_verified_qualification_aperture"]["path"]))
            require(geometry_document.get("schema") == GEOMETRY_PLAN_SCHEMA and
                geometry_document.get("schema_version") == 1 and
                geometry_document.get("plan_id") == geometry_record["plan_id"],
                f"{path}: bound geometry-plan identity changed")
            require(aperture_document.get("schema") == APERTURE_SCHEMA and
                aperture_document.get("schema_version") == 1 and
                aperture_document.get("aperture_id") ==
                    aperture_record["aperture_id"],
                f"{path}: bound qualification-aperture identity changed")
    for name in ("binary", "config", "run_manifest"):
        resolved = resolve_bound_record(source.get(name), path, f"{path}: {name}")
        require(resolved["sha256"] == artifacts[name]["sha256"] and
                resolved["path"] == artifacts[name]["path"],
                f"{path}: {name} does not match the series artifact")
    signature = object_value(source.get("numerical_signature"),
                             f"{path}: numerical_signature")
    require(signature.get("schema") == "nimbus.numerical_signature" and
            signature.get("schema_version") == 1,
            f"{path}: wrong numerical-signature schema")
    values = object_value(signature.get("values"), f"{path}: numerical signature values")
    signature_sha = signature.get("sha256")
    require(valid_sha256(signature_sha) and canonical_sha256(values) == signature_sha,
            f"{path}: numerical signature digest mismatch")
    derivation = object_value(signature.get("derivation"),
                              f"{path}: numerical signature derivation")
    require(derivation.get("role") == role and
            derivation.get("effective_level") ==
                ("nested_child" if role == "moving_child" else "root_reference"),
            f"{path}: numerical signature level ownership changed")
    root_dt = finite_number(derivation.get("root_dt_large_s"),
                            f"{path}: root dt")
    time_ratio = derivation.get("parent_time_step_ratio")
    ratio_x = derivation.get("parent_ratio_x")
    ratio_y = derivation.get("parent_ratio_y")
    require(all(type(value) is int and value > 0
                for value in (time_ratio, ratio_x, ratio_y)),
            f"{path}: numerical signature refinement ratios are invalid")
    root_dx = finite_number(derivation.get("root_dx_m"), f"{path}: root dx")
    root_dy = finite_number(derivation.get("root_dy_m"), f"{path}: root dy")
    effective_dt = finite_number(derivation.get("effective_dt_large_s"),
                                 f"{path}: effective dt")
    effective_dx = finite_number(derivation.get("effective_dx_m"),
                                 f"{path}: effective dx")
    effective_dy = finite_number(derivation.get("effective_dy_m"),
                                 f"{path}: effective dy")
    require(math.isclose(effective_dt, root_dt / time_ratio,
                         rel_tol=0.0, abs_tol=1.0e-12) and
            math.isclose(effective_dx, root_dx / ratio_x,
                         rel_tol=0.0, abs_tol=1.0e-12) and
            math.isclose(effective_dy, root_dy / ratio_y,
                         rel_tol=0.0, abs_tol=1.0e-12),
            f"{path}: effective-level numerical derivation is inconsistent")
    published_spacing = object_value(
        derivation.get("published_comparison_spacing_m"),
        f"{path}: published comparison spacing")
    require(finite_number(published_spacing.get("x"), f"{path}: published dx") ==
                effective_dx and
            finite_number(published_spacing.get("y"), f"{path}: published dy") ==
                effective_dy and
            values.get("comparison_level.effective_dt_large_s") == effective_dt and
            values.get("comparison_level.effective_dx_m") == effective_dx and
            values.get("comparison_level.effective_dy_m") == effective_dy,
            f"{path}: effective-level signature values disagree with derivation")
    diagnostics = object_value(source.get("diagnostics"), f"{path}: diagnostics")
    stream = object_value(source.get("diagnostic_stream"),
                          f"{path}: diagnostic_stream")
    require(stream.get("preset") == "moving_reference_v1" and
            stream.get("data_product_kind") ==
                ("nested_child_qualification_diagnostics"
                 if role == "moving_child" else "qualification_diagnostics") and
            stream.get("storage_format") == "npy_3d" and
            stream.get("published_fields") ==
                ["cold_pool_boundary", "accumulated_rainfall"] and
            finite_number(stream.get("interval_s"),
                          f"{path}: diagnostic interval") > 0.0,
            f"{path}: sparse diagnostic stream contract changed")
    cold = object_value(diagnostics.get("cold_pool"), f"{path}: cold_pool diagnostic")
    require(cold.get("algorithm") == COLD_POOL_ALGORITHM and
            cold.get("water_species") == ["qv", "qc", "qr", "qi", "qs", "qg", "qh"] and
            cold.get("formula") ==
                "theta*(1+qv/epsilon)/(1+qv+qc+qr+qi+qs+qg+qh)",
            f"{path}: cold-pool diagnostic is not the all-species contract")
    precipitation = object_value(
        diagnostics.get("precipitation"), f"{path}: precipitation diagnostic")
    require(precipitation.get("algorithm") == PRECIPITATION_ALGORITHM and
            precipitation.get("effective_depth_semantics") ==
                "residual_inclusive_then_float32_published_v1" and
            precipitation.get("manifest_algorithm_identity") in
                {PRECIPITATION_ALGORITHM, "unavailable_in_manifest_v6"},
            f"{path}: precipitation residual semantics changed")
    conversion = object_value(source.get("coordinate_conversion"),
                              f"{path}: coordinate_conversion")
    frames = conversion.get("runtime_detector_frames")
    require(conversion.get("output_frame") == "earth_absolute" and
            conversion.get("grid_relative_reference_rule") ==
                "stationary_zero_velocity_grid_origin_translation_v1" and
            isinstance(frames, list) and len(frames) == 1 and
            frames[0] in {"earth_absolute", "grid_relative"},
            f"{path}: coordinate-conversion contract changed")
    if role == "moving_child":
        require(frames == ["earth_absolute"],
                f"{path}: moving-child detector was not Earth-fixed")
        require(source.get("region_series") is None,
                f"{path}: moving-child source invented an upstream region series")
    else:
        resolve_bound_record(source.get("region_series"), path,
                             f"{path}: region_series")
    implementation = source.get("implementation_sources")
    require(isinstance(implementation, list) and implementation,
            f"{path}: implementation source binding is empty")
    for index, record in enumerate(implementation):
        resolve_bound_record(record, path, f"{path}: implementation_sources[{index}]")
    for collection in ("step_manifests", "storm_records"):
        records = source.get(collection)
        require(isinstance(records, list) and records,
                f"{path}: {collection} binding is empty")
        times: list[float] = []
        for index, record in enumerate(records):
            item = object_value(record, f"{path}: {collection}[{index}]")
            times.append(finite_number(item.get("simulation_time_s"),
                                       f"{path}: {collection}[{index}].simulation_time_s"))
            resolve_bound_record(item, path, f"{path}: {collection}[{index}]")
        require(times == sorted(times) and len(times) == len(set(times)),
                f"{path}: {collection} times are not unique and increasing")
    diagnostic_fields = source.get("diagnostic_fields")
    require(isinstance(diagnostic_fields, list) and diagnostic_fields,
            f"{path}: diagnostic field binding is empty")
    diagnostic_times: list[float] = []
    for index, record in enumerate(diagnostic_fields):
        item = object_value(record, f"{path}: diagnostic_fields[{index}]")
        require(set(item) == {
                    "simulation_time_s", "cold_pool_boundary",
                    "accumulated_rainfall"},
                f"{path}: diagnostic_fields[{index}] inventory changed")
        diagnostic_times.append(finite_number(
            item.get("simulation_time_s"),
            f"{path}: diagnostic_fields[{index}].simulation_time_s"))
        for field in ("cold_pool_boundary", "accumulated_rainfall"):
            resolve_bound_record(
                item.get(field), path,
                f"{path}: diagnostic_fields[{index}].{field}")
    require(diagnostic_times == sorted(diagnostic_times) and
            len(diagnostic_times) == len(set(diagnostic_times)),
            f"{path}: diagnostic field times are not unique and increasing")
    return source


def validate_contract(contract: dict[str, Any], source: Path) -> dict[str, Any]:
    numeric = (
        "duration_s",
        "sample_start_s",
        "sample_interval_s",
    )
    for key in numeric:
        contract[key] = nonnegative_number(contract.get(key), f"{source}: contract.{key}")
    require(contract["duration_s"] >= contract["sample_start_s"],
            f"{source}: duration precedes sample start")
    require(contract["sample_interval_s"] > 0.0,
            f"{source}: sample interval must be positive")
    for key in (
        "coordinate_frame",
        "detector_algorithm",
        "continuity_algorithm",
        "physics_signature_sha256",
        "tracking_source",
        "comparison_region",
    ):
        require(isinstance(contract.get(key), str) and contract[key],
                f"{source}: contract.{key} must be nonempty")
    require(valid_sha256(contract["physics_signature_sha256"]),
            f"{source}: invalid physics signature")
    spacing = object_value(contract.get("horizontal_grid_spacing_m"),
                           f"{source}: contract.horizontal_grid_spacing_m")
    phase = object_value(contract.get("grid_phase_origin_m"),
                         f"{source}: contract.grid_phase_origin_m")
    contract["horizontal_grid_spacing_m"] = {
        "x": nonnegative_number(spacing.get("x"), f"{source}: grid spacing x"),
        "y": nonnegative_number(spacing.get("y"), f"{source}: grid spacing y"),
    }
    require(contract["horizontal_grid_spacing_m"]["x"] > 0.0 and
            contract["horizontal_grid_spacing_m"]["y"] > 0.0,
            f"{source}: grid spacing must be positive")
    contract["grid_phase_origin_m"] = {
        "x": finite_number(phase.get("x"), f"{source}: grid phase x"),
        "y": finite_number(phase.get("y"), f"{source}: grid phase y"),
    }
    exclusion = contract.get("relaxation_exclusion_cells")
    require(isinstance(exclusion, int) and not isinstance(exclusion, bool) and exclusion >= 0,
            f"{source}: relaxation_exclusion_cells must be a nonnegative integer")
    return contract


def validate_region(value: object, source: Path, index: int) -> dict[str, Any]:
    label = f"{source}: samples[{index}].region"
    region = object_value(value, label)
    require(region.get("id") in {
                "moving_child_trusted_interior_v1", APERTURE_REGION_ID},
            f"{label}.id is incompatible")
    bounds = object_value(region.get("earth_bounds_m"), f"{label}.earth_bounds_m")
    parsed_bounds = {
        name: finite_number(bounds.get(name), f"{label}.earth_bounds_m.{name}")
        for name in ("west", "east", "south", "north")
    }
    require(parsed_bounds["west"] < parsed_bounds["east"] and
            parsed_bounds["south"] < parsed_bounds["north"],
            f"{label} has empty Earth bounds")
    spacing = object_value(region.get("horizontal_grid_spacing_m"),
                           f"{label}.horizontal_grid_spacing_m")
    parsed_spacing = {
        name: nonnegative_number(spacing.get(name), f"{label} spacing {name}")
        for name in ("x", "y")
    }
    require(parsed_spacing["x"] > 0.0 and parsed_spacing["y"] > 0.0,
            f"{label} spacing must be positive")
    phase = object_value(region.get("grid_phase_origin_m"),
                         f"{label}.grid_phase_origin_m")
    parsed_phase = {
        name: finite_number(phase.get(name), f"{label} phase {name}")
        for name in ("x", "y")
    }
    indices = object_value(region.get("source_index_range"),
                           f"{label}.source_index_range")
    lower = object_value(indices.get("lower"), f"{label}.source_index_range.lower")
    upper = object_value(indices.get("upper"), f"{label}.source_index_range.upper")
    parsed_indices: dict[str, dict[str, int]] = {"lower": {}, "upper": {}}
    for side, record in (("lower", lower), ("upper", upper)):
        for axis in ("i", "j"):
            candidate = record.get(axis)
            require(type(candidate) is int and candidate >= 0,
                    f"{label} {side}.{axis} must be a nonnegative integer")
            parsed_indices[side][axis] = candidate
    require(parsed_indices["lower"]["i"] <= parsed_indices["upper"]["i"] and
            parsed_indices["lower"]["j"] <= parsed_indices["upper"]["j"],
            f"{label} has empty source indices")
    count = region.get("cell_count")
    expected_count = (
        parsed_indices["upper"]["i"] - parsed_indices["lower"]["i"] + 1
    ) * (parsed_indices["upper"]["j"] - parsed_indices["lower"]["j"] + 1)
    require(type(count) is int and count == expected_count,
            f"{label}.cell_count disagrees with its indices")
    result = {
        "id": region["id"], "earth_bounds_m": parsed_bounds,
        "source_index_range": parsed_indices,
        "horizontal_grid_spacing_m": parsed_spacing,
        "grid_phase_origin_m": parsed_phase, "cell_count": count,
    }
    if region["id"] == APERTURE_REGION_ID:
        require(region.get("aperture_id") == "moving_nested_fixed_100x96_v1",
                f"{label}.aperture_id is incompatible")
        result["aperture_id"] = region["aperture_id"]
        operational = region.get("operational_interior_source_index_range")
        if operational is not None:
            operational = object_value(
                operational,
                f"{label}.operational_interior_source_index_range")
            op_lower = object_value(
                operational.get("lower"), f"{label} operational lower")
            op_upper = object_value(
                operational.get("upper"), f"{label} operational upper")
            require(all(type(value) is int and value >= 0 for value in (
                        op_lower.get("i"), op_lower.get("j"),
                        op_upper.get("i"), op_upper.get("j"))) and
                    op_lower["i"] <= parsed_indices["lower"]["i"] and
                    op_lower["j"] <= parsed_indices["lower"]["j"] and
                    op_upper["i"] >= parsed_indices["upper"]["i"] and
                    op_upper["j"] >= parsed_indices["upper"]["j"],
                    f"{label} aperture is outside its operational interior")
            result["operational_interior_source_index_range"] = operational
    return result


def validate_sample(sample: object, source: Path, index: int,
                    schema_version: int) -> dict[str, Any]:
    item = object_value(sample, f"{source}: samples[{index}]")
    result = dict(item)
    result["simulation_time_s"] = finite_number(
        item.get("simulation_time_s"), f"{source}: samples[{index}].simulation_time_s")
    require(isinstance(item.get("selected"), bool),
            f"{source}: samples[{index}].selected must be boolean")
    if schema_version == 2:
        result["region"] = validate_region(item.get("region"), source, index)
    if item["selected"]:
        result["core_centroid_m"] = (
            None if item.get("core_centroid_m") is None else
            coordinate(item["core_centroid_m"],
                       f"{source}: samples[{index}].core_centroid_m")
        )
        result["maximum_updraft_ms"] = (
            None if item.get("maximum_updraft_ms") is None else
            nonnegative_number(item["maximum_updraft_ms"],
                               f"{source}: samples[{index}].maximum_updraft_ms")
        )
        result["maximum_absolute_uh_m2s2"] = (
            None if item.get("maximum_absolute_uh_m2s2") is None else
            nonnegative_number(
                item["maximum_absolute_uh_m2s2"],
                f"{source}: samples[{index}].maximum_absolute_uh_m2s2")
        )
    return result


def validate_series(document: dict[str, Any], source: Path, expected_role: str) -> dict[str, Any]:
    require(document.get("schema") == INPUT_SCHEMA, f"{source}: wrong input schema")
    schema_version = document.get("schema_version")
    require(schema_version in {1, 2}, f"{source}: wrong input schema version")
    require(document.get("role") == expected_role,
            f"{source}: expected role {expected_role}")
    verified = resolve_artifacts(document, source)
    contract = validate_contract(
        object_value(document.get("contract"), f"{source}: contract"), source)
    samples_value = document.get("samples")
    require(isinstance(samples_value, list) and samples_value,
            f"{source}: samples must be a nonempty array")
    samples = [validate_sample(value, source, index, schema_version)
               for index, value in enumerate(samples_value)]
    times = [sample["simulation_time_s"] for sample in samples]
    require(len(set(times)) == len(times), f"{source}: duplicate sample times")
    require(times == sorted(times), f"{source}: sample times are not increasing")
    source_manifest = None
    if schema_version == 2:
        source_manifest = validate_source_manifest(
            Path(verified["source_manifest"]["path"]), expected_role, verified)
        manifest_times = [finite_number(value.get("simulation_time_s"), "manifest time")
                          for value in source_manifest["step_manifests"]]
        tracker_times = [finite_number(value.get("simulation_time_s"), "tracker time")
                         for value in source_manifest["storm_records"]]
        diagnostic_times = [finite_number(value.get("simulation_time_s"), "diagnostic time")
                            for value in source_manifest["diagnostic_fields"]]
        require(manifest_times == times and tracker_times == times and
                diagnostic_times == times,
                f"{source}: source manifest does not bind every series sample")
        require(source_manifest["numerical_signature"]["sha256"] ==
                contract["physics_signature_sha256"],
                f"{source}: numerical signature differs from its contract")
        require(finite_number(
                    source_manifest["diagnostic_stream"].get("interval_s"),
                    f"{source}: diagnostic stream interval") ==
                contract["sample_interval_s"],
                f"{source}: diagnostic stream cadence differs from its contract")
        if (source_manifest.get("schema_version") == 2 and
                source_manifest.get("policy", {}).get("schema_version") in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}):
            for key, verified_key in (
                ("geometry_plan", "_verified_geometry_plan"),
                ("qualification_aperture",
                 "_verified_qualification_aperture"),
            ):
                contract_record = object_value(
                    contract.get(key), f"{source}: contract.{key}")
                verified_record = source_manifest.get(verified_key)
                require(isinstance(verified_record, dict) and
                        contract_record.get("sha256") ==
                            verified_record.get("sha256"),
                        f"{source}: {key} differs from its source manifest")
    return {
        "role": expected_role,
        "schema_version": schema_version,
        "artifacts": verified,
        "contract": contract,
        "samples": samples,
        "source_manifest": source_manifest,
        "input_sha256": sha256(source),
        "input_path": str(source.resolve()),
    }


def validate_v6_policy(policy: dict[str, Any], source: Path) -> dict[str, Any]:
    require(set(policy) == {
                "schema", "schema_version", "policy_id", "policy_status",
                "scope", "prior_policy", "failure_evidence",
                "ratification", "limitations"},
            f"{source}: v6 amendment inventory changed")
    require(policy.get("policy_id") ==
                "moving-nested-reference-frontier-v6-trusted-interior-containment-amendment" and
            policy.get("policy_status") == "ratified" and
            isinstance(policy.get("scope"), str) and policy["scope"],
            f"{source}: v6 identity is invalid")
    prior_record = object_value(policy.get("prior_policy"),
                                f"{source}: prior_policy")
    require(set(prior_record) ==
                {"path", "sha256", "schema_version", "policy_id"} and
            prior_record.get("schema_version") == 5 and
            prior_record.get("sha256") == V5_POLICY_SHA256 and
            prior_record.get("policy_id") ==
                "moving-nested-reference-frontier-v5-maximum-safe-projected-shift-amendment",
            f"{source}: v6 prior-policy binding changed")
    verified_prior = resolve_bound_record(
        prior_record, source, f"{source}: prior_policy")
    prior_policy = validate_policy(
        load_json(Path(verified_prior["path"])),
        Path(verified_prior["path"]))

    failure_record = object_value(policy.get("failure_evidence"),
                                  f"{source}: failure_evidence")
    require(set(failure_record) ==
                {"path", "sha256", "schema_version", "reason"} and
            failure_record.get("schema_version") == 1 and
            failure_record.get("sha256") ==
                TRUSTED_CONTAINMENT_FAILURE_SHA256 and
            failure_record.get("reason") ==
                "trusted_envelope_rejected_when_release_box_unattainable",
            f"{source}: v6 failure-evidence binding changed")
    verified_failure = resolve_bound_record(
        failure_record, source, f"{source}: failure_evidence")
    failure = load_json(Path(verified_failure["path"]))
    require(failure.get("schema") ==
                "nimbus.qualification.moving_nested_reference_frontier.movement_failure" and
            failure.get("schema_version") == 1 and
            failure.get("status") == "invalid" and
            failure.get("qualification_disposition") == "incomplete" and
            failure.get("qualification_eligible") is False and
            failure.get("reason") ==
                "trusted_envelope_rejected_when_release_box_unattainable" and
            failure.get("comparison_performed") is False and
            failure.get("candidate_metric_values_inspected") is False,
            f"{source}: bound trusted-containment failure identity changed")

    ratification = object_value(policy.get("ratification"),
                                f"{source}: ratification")
    require(set(ratification) == {
                "decision", "review_date", "reviewer_role",
                "failed_v5_trusted_containment_sequence_observed",
                "candidate_metric_values_inspected",
                "minimum_input_series_schema_version",
                "trusted_interior_containment_amendment"} and
            ratification.get("decision") ==
                "post_failure_trusted_interior_containment_amendment_of_v5_and_blinded_v2_thresholds" and
            ratification.get(
                "failed_v5_trusted_containment_sequence_observed") is True and
            ratification.get("candidate_metric_values_inspected") is False and
            ratification.get("minimum_input_series_schema_version") == 2 and
            isinstance(ratification.get("review_date"), str) and
            isinstance(ratification.get("reviewer_role"), str),
            f"{source}: v6 amendment provenance is incomplete")
    amendment = object_value(
        ratification.get("trusted_interior_containment_amendment"),
        f"{source}: trusted_interior_containment_amendment")
    require(amendment == {
                "reason":
                    "fail_closed_trusted_envelope_rejected_when_release_box_unattainable",
                "algorithm": "qualified_envelope_hysteresis_v8",
                "eligibility":
                    "authoritative selected track with finite signed per-face motion, current trusted-interior membership, and projected relaxation risk",
                "objective":
                    "maximize the minimum current opposing-face clearance on the parent-cell lattice",
                "current_trusted_interior_required": True,
                "full_release_box_preferred": True,
                "full_release_box_required_for_fallback": False,
                "fallback_shift_must_strictly_improve_minimum_clearance": True,
                "fallback_shift_must_preserve_all_relaxation_faces": True,
                "maximum_shift_cap_parent_cells": 14,
                "parent_containment_required": True,
                "ambiguity_topology_or_missing_face_motion_bypass_allowed": False,
                "current_relaxation_entry_recoverable": False,
                "numerical_thresholds_changed": False,
                "geometry_or_aperture_changed": False,
                "measurement_cadence_changed": False,
                "failed_candidate_disposition": "invalid_not_comparable"},
            f"{source}: trusted-interior containment amendment changed")
    limitations = policy.get("limitations")
    require(isinstance(limitations, list) and limitations and
            all(isinstance(value, str) and value for value in limitations),
            f"{source}: v6 limitations are incomplete")

    effective = copy.deepcopy(prior_policy)
    for key in tuple(effective):
        if key.startswith("_verified_"):
            del effective[key]
    effective.update({
        "schema_version": 6,
        "policy_id": policy["policy_id"],
        "policy_status": "ratified",
        "scope": policy["scope"],
        "prior_policy": copy.deepcopy(prior_record),
        "failure_evidence": copy.deepcopy(failure_record),
        "ratification": copy.deepcopy(ratification),
        "limitations": copy.deepcopy(limitations),
    })
    effective["expected_contract"] = copy.deepcopy(
        prior_policy["expected_contract"])
    effective["expected_contract"]["moving_movement_algorithm"] = \
        "qualified_envelope_hysteresis_v8"
    effective["_verified_prior_policy"] = verified_prior
    effective["_verified_failure_evidence"] = verified_failure
    effective["_verified_threshold_policy"] = \
        prior_policy.get("_verified_threshold_policy")
    effective["_verified_geometry_plan"] = \
        prior_policy.get("_verified_geometry_plan")
    effective["_verified_qualification_aperture"] = \
        prior_policy.get("_verified_qualification_aperture")
    return effective


def validate_v7_policy(policy: dict[str, Any], source: Path) -> dict[str, Any]:
    require(set(policy) == {
                "schema", "schema_version", "policy_id", "policy_status",
                "scope", "prior_policy", "failure_evidence",
                "ratification", "limitations"},
            f"{source}: v7 amendment inventory changed")
    require(policy.get("policy_id") ==
                "moving-nested-reference-frontier-v7-balanced-trusted-containment-amendment" and
            policy.get("policy_status") == "ratified" and
            isinstance(policy.get("scope"), str) and policy["scope"],
            f"{source}: v7 identity is invalid")
    prior_record = object_value(policy.get("prior_policy"),
                                f"{source}: prior_policy")
    require(set(prior_record) ==
                {"path", "sha256", "schema_version", "policy_id"} and
            prior_record.get("schema_version") == 6 and
            prior_record.get("sha256") == V6_POLICY_SHA256 and
            prior_record.get("policy_id") ==
                "moving-nested-reference-frontier-v6-trusted-interior-containment-amendment",
            f"{source}: v7 prior-policy binding changed")
    verified_prior = resolve_bound_record(
        prior_record, source, f"{source}: prior_policy")
    prior_policy = validate_policy(
        load_json(Path(verified_prior["path"])),
        Path(verified_prior["path"]))
    failure_record = object_value(policy.get("failure_evidence"),
                                  f"{source}: failure_evidence")
    require(set(failure_record) ==
                {"path", "sha256", "schema_version", "reason"} and
            failure_record.get("schema_version") == 1 and
            failure_record.get("sha256") ==
                BALANCED_CONTAINMENT_PROBE_FAILURE_SHA256 and
            failure_record.get("reason") ==
                "balanced_trusted_envelope_rejected_as_opposite_faces_breached",
            f"{source}: v7 failure-evidence binding changed")
    verified_failure = resolve_bound_record(
        failure_record, source, f"{source}: failure_evidence")
    failure = load_json(Path(verified_failure["path"]))
    require(failure.get("schema") ==
                "nimbus.qualification.moving_nested_reference_frontier.diagnostic_probe_failure" and
            failure.get("schema_version") == 1 and
            failure.get("status") == "invalid" and
            failure.get("qualification_disposition") ==
                "diagnostic_only_not_comparable" and
            failure.get("qualification_eligible") is False and
            failure.get("reason") ==
                "balanced_trusted_envelope_rejected_as_opposite_faces_breached" and
            failure.get("comparison_performed") is False and
            failure.get("candidate_metric_values_inspected") is False,
            f"{source}: bound balanced-containment probe identity changed")
    ratification = object_value(policy.get("ratification"),
                                f"{source}: ratification")
    require(set(ratification) == {
                "decision", "review_date", "reviewer_role",
                "failed_v6_balanced_containment_probe_observed",
                "candidate_metric_values_inspected",
                "minimum_input_series_schema_version",
                "balanced_trusted_containment_amendment"} and
            ratification.get("decision") ==
                "post_probe_balanced_trusted_containment_amendment_of_v6_and_blinded_v2_thresholds" and
            ratification.get(
                "failed_v6_balanced_containment_probe_observed") is True and
            ratification.get("candidate_metric_values_inspected") is False and
            ratification.get("minimum_input_series_schema_version") == 2 and
            isinstance(ratification.get("review_date"), str) and
            isinstance(ratification.get("reviewer_role"), str),
            f"{source}: v7 amendment provenance is incomplete")
    amendment = object_value(
        ratification.get("balanced_trusted_containment_amendment"),
        f"{source}: balanced_trusted_containment_amendment")
    require(amendment == {
                "reason":
                    "fail_closed_balanced_trusted_envelope_rejected_as_opposite_faces_breached",
                "algorithm": "qualified_envelope_hysteresis_v9",
                "eligibility":
                    "every inner-margin-breached axis is an opposing pair whose two faces remain outside the exact relaxation width",
                "movement_status": "ineligible",
                "movement_reason": "trusted_interior_contained",
                "prognostic_state_mutation": False,
                "diagnostic_history_mutation": True,
                "current_trusted_interior_required": True,
                "current_relaxation_entry_recoverable": False,
                "ordinary_or_projected_movement_changed": False,
                "numerical_thresholds_changed": False,
                "geometry_or_aperture_changed": False,
                "measurement_cadence_changed": False,
                "failed_candidate_disposition":
                    "diagnostic_probe_invalid_not_comparable"},
            f"{source}: balanced trusted-containment amendment changed")
    limitations = policy.get("limitations")
    require(isinstance(limitations, list) and limitations and
            all(isinstance(value, str) and value for value in limitations),
            f"{source}: v7 limitations are incomplete")
    effective = copy.deepcopy(prior_policy)
    for key in tuple(effective):
        if key.startswith("_verified_"):
            del effective[key]
    effective.update({
        "schema_version": 7,
        "policy_id": policy["policy_id"],
        "policy_status": "ratified",
        "scope": policy["scope"],
        "prior_policy": copy.deepcopy(prior_record),
        "failure_evidence": copy.deepcopy(failure_record),
        "ratification": copy.deepcopy(ratification),
        "limitations": copy.deepcopy(limitations),
    })
    effective["expected_contract"] = copy.deepcopy(
        prior_policy["expected_contract"])
    effective["expected_contract"]["moving_movement_algorithm"] = \
        "qualified_envelope_hysteresis_v9"
    effective["_verified_prior_policy"] = verified_prior
    effective["_verified_failure_evidence"] = verified_failure
    for key in ("_verified_threshold_policy", "_verified_geometry_plan",
                "_verified_qualification_aperture"):
        effective[key] = prior_policy.get(key)
    return effective


def validate_v8_policy(policy: dict[str, Any], source: Path) -> dict[str, Any]:
    require(set(policy) == {
                "schema", "schema_version", "policy_id", "policy_status",
                "scope", "prior_policy", "failure_evidence",
                "ratification", "limitations"},
            f"{source}: v8 amendment inventory changed")
    require(policy.get("policy_id") ==
                "moving-nested-reference-frontier-v8-orthogonal-containment-reserve-amendment" and
            policy.get("policy_status") == "ratified" and
            isinstance(policy.get("scope"), str) and policy["scope"],
            f"{source}: v8 identity is invalid")
    prior_record = object_value(policy.get("prior_policy"),
                                f"{source}: prior_policy")
    require(set(prior_record) ==
                {"path", "sha256", "schema_version", "policy_id"} and
            prior_record.get("schema_version") == 7 and
            prior_record.get("sha256") == V7_POLICY_SHA256 and
            prior_record.get("policy_id") ==
                "moving-nested-reference-frontier-v7-balanced-trusted-containment-amendment",
            f"{source}: v8 prior-policy binding changed")
    verified_prior = resolve_bound_record(
        prior_record, source, f"{source}: prior_policy")
    prior_policy = validate_policy(
        load_json(Path(verified_prior["path"])),
        Path(verified_prior["path"]))

    failure_record = object_value(policy.get("failure_evidence"),
                                  f"{source}: failure_evidence")
    require(set(failure_record) ==
                {"path", "sha256", "schema_version", "reason"} and
            failure_record.get("schema_version") == 1 and
            failure_record.get("sha256") ==
                ORTHOGONAL_CONTAINMENT_RESERVE_FAILURE_SHA256 and
            failure_record.get("reason") ==
                "post_containment_orthogonal_reserve_absent",
            f"{source}: v8 failure-evidence binding changed")
    verified_failure = resolve_bound_record(
        failure_record, source, f"{source}: failure_evidence")
    failure = load_json(Path(verified_failure["path"]))
    require(failure.get("schema") ==
                "nimbus.qualification.moving_nested_reference_frontier.movement_failure" and
            failure.get("schema_version") == 1 and
            failure.get("status") == "invalid" and
            failure.get("qualification_disposition") == "incomplete" and
            failure.get("qualification_eligible") is False and
            failure.get("reason") ==
                "post_containment_orthogonal_reserve_absent" and
            failure.get("comparison_performed") is False and
            failure.get("candidate_metric_values_inspected") is False,
            f"{source}: bound orthogonal-reserve failure identity changed")

    ratification = object_value(policy.get("ratification"),
                                f"{source}: ratification")
    require(set(ratification) == {
                "decision", "review_date", "reviewer_role",
                "failed_v7_orthogonal_reserve_sequence_observed",
                "candidate_metric_values_inspected",
                "minimum_input_series_schema_version",
                "orthogonal_containment_reserve_amendment"} and
            ratification.get("decision") ==
                "post_failure_orthogonal_containment_reserve_amendment_of_v7_and_blinded_v2_thresholds" and
            ratification.get(
                "failed_v7_orthogonal_reserve_sequence_observed") is True and
            ratification.get("candidate_metric_values_inspected") is False and
            ratification.get("minimum_input_series_schema_version") == 2 and
            isinstance(ratification.get("review_date"), str) and
            isinstance(ratification.get("reviewer_role"), str),
            f"{source}: v8 amendment provenance is incomplete")
    amendment = object_value(
        ratification.get("orthogonal_containment_reserve_amendment"),
        f"{source}: orthogonal_containment_reserve_amendment")
    require(amendment == {
                "reason":
                    "fail_closed_post_containment_orthogonal_reserve_absent",
                "algorithm": "qualified_envelope_hysteresis_v10",
                "eligibility":
                    "an admitted trusted-interior containment transaction with an orthogonal authoritative approach direction and whole-parent-cell slack above both immutable release faces",
                "transaction_scope":
                    "same_immediate_trusted_interior_containment_transaction",
                "orthogonal_shift_objective":
                    "maximize directional reserve using only whole-parent-cell slack above the opposite release margin",
                "both_orthogonal_release_faces_required_after_shift": True,
                "all_current_relaxation_faces_required_after_shift": True,
                "maximum_shift_cap_parent_cells": 14,
                "current_relaxation_entry_recoverable": False,
                "standalone_reserve_movement_allowed": False,
                "numerical_thresholds_changed": False,
                "geometry_or_aperture_changed": False,
                "measurement_cadence_changed": False,
                "failed_candidate_disposition": "invalid_not_comparable"},
            f"{source}: orthogonal containment reserve amendment changed")
    limitations = policy.get("limitations")
    require(isinstance(limitations, list) and limitations and
            all(isinstance(value, str) and value for value in limitations),
            f"{source}: v8 limitations are incomplete")

    effective = copy.deepcopy(prior_policy)
    for key in tuple(effective):
        if key.startswith("_verified_"):
            del effective[key]
    effective.update({
        "schema_version": 8,
        "policy_id": policy["policy_id"],
        "policy_status": "ratified",
        "scope": policy["scope"],
        "prior_policy": copy.deepcopy(prior_record),
        "failure_evidence": copy.deepcopy(failure_record),
        "ratification": copy.deepcopy(ratification),
        "limitations": copy.deepcopy(limitations),
    })
    effective["expected_contract"] = copy.deepcopy(
        prior_policy["expected_contract"])
    effective["expected_contract"]["moving_movement_algorithm"] = \
        "qualified_envelope_hysteresis_v10"
    effective["_verified_prior_policy"] = verified_prior
    effective["_verified_failure_evidence"] = verified_failure
    for key in ("_verified_threshold_policy", "_verified_geometry_plan",
                "_verified_qualification_aperture"):
        effective[key] = prior_policy.get(key)
    return effective


def validate_v9_policy(policy: dict[str, Any], source: Path) -> dict[str, Any]:
    require(set(policy) == {
                "schema", "schema_version", "policy_id", "policy_status",
                "scope", "prior_policy", "failure_evidence",
                "ratification", "limitations"},
            f"{source}: v9 amendment inventory changed")
    require(policy.get("policy_id") ==
                "moving-nested-reference-frontier-v9-receding-trusted-face-amendment" and
            policy.get("policy_status") == "ratified" and
            isinstance(policy.get("scope"), str) and policy["scope"],
            f"{source}: v9 identity is invalid")
    prior_record = object_value(policy.get("prior_policy"),
                                f"{source}: prior_policy")
    require(set(prior_record) ==
                {"path", "sha256", "schema_version", "policy_id"} and
            prior_record.get("schema_version") == 8 and
            prior_record.get("sha256") == V8_POLICY_SHA256 and
            prior_record.get("policy_id") ==
                "moving-nested-reference-frontier-v8-orthogonal-containment-reserve-amendment",
            f"{source}: v9 prior-policy binding changed")
    verified_prior = resolve_bound_record(
        prior_record, source, f"{source}: prior_policy")
    prior_policy = validate_policy(
        load_json(Path(verified_prior["path"])),
        Path(verified_prior["path"]))

    failure_record = object_value(policy.get("failure_evidence"),
                                  f"{source}: failure_evidence")
    require(set(failure_record) ==
                {"path", "sha256", "schema_version", "reason"} and
            failure_record.get("schema_version") == 1 and
            failure_record.get("sha256") ==
                RECEDING_TRUSTED_FACE_FAILURE_SHA256 and
            failure_record.get("reason") ==
                "trusted_receding_face_unrecognized",
            f"{source}: v9 failure-evidence binding changed")
    verified_failure = resolve_bound_record(
        failure_record, source, f"{source}: failure_evidence")
    failure = load_json(Path(verified_failure["path"]))
    require(failure.get("schema") ==
                "nimbus.qualification.moving_nested_reference_frontier.movement_failure" and
            failure.get("schema_version") == 1 and
            failure.get("status") == "invalid" and
            failure.get("qualification_disposition") == "incomplete" and
            failure.get("qualification_eligible") is False and
            failure.get("reason") ==
                "trusted_receding_face_unrecognized" and
            failure.get("comparison_performed") is False and
            failure.get("candidate_metric_values_inspected") is False,
            f"{source}: bound receding trusted-face failure identity changed")

    ratification = object_value(policy.get("ratification"),
                                f"{source}: ratification")
    require(set(ratification) == {
                "decision", "review_date", "reviewer_role",
                "failed_v8_receding_trusted_face_sequence_observed",
                "candidate_metric_values_inspected",
                "minimum_input_series_schema_version",
                "receding_trusted_face_amendment"} and
            ratification.get("decision") ==
                "post_failure_receding_trusted_face_amendment_of_v8_and_blinded_v2_thresholds" and
            ratification.get(
                "failed_v8_receding_trusted_face_sequence_observed") is True and
            ratification.get("candidate_metric_values_inspected") is False and
            ratification.get("minimum_input_series_schema_version") == 2 and
            isinstance(ratification.get("review_date"), str) and
            isinstance(ratification.get("reviewer_role"), str),
            f"{source}: v9 amendment provenance is incomplete")
    amendment = object_value(
        ratification.get("receding_trusted_face_amendment"),
        f"{source}: receding_trusted_face_amendment")
    require(amendment == {
                "reason":
                    "fail_closed_single_receding_trusted_face_unrecognized",
                "algorithm": "qualified_envelope_hysteresis_v11",
                "eligibility":
                    "one or more inner-margin faces remain outside the immutable relaxation zone and no breached face is approaching by either selected motion or signed face motion",
                "disposition":
                    "trusted_interior_contained_without_mutation",
                "all_breached_faces_outside_relaxation_required": True,
                "no_breached_face_approaching_required": True,
                "selected_motion_and_face_motion_both_authoritative": True,
                "movement_performed": False,
                "current_relaxation_entry_recoverable": False,
                "ordinary_or_projected_movement_changed": False,
                "orthogonal_containment_reserve_changed": False,
                "maximum_shift_cap_parent_cells": 14,
                "numerical_thresholds_changed": False,
                "geometry_or_aperture_changed": False,
                "measurement_cadence_changed": False,
                "failed_candidate_disposition": "invalid_not_comparable"},
            f"{source}: receding trusted-face amendment changed")
    limitations = policy.get("limitations")
    require(isinstance(limitations, list) and limitations and
            all(isinstance(value, str) and value for value in limitations),
            f"{source}: v9 limitations are incomplete")

    effective = copy.deepcopy(prior_policy)
    for key in tuple(effective):
        if key.startswith("_verified_"):
            del effective[key]
    effective.update({
        "schema_version": 9,
        "policy_id": policy["policy_id"],
        "policy_status": "ratified",
        "scope": policy["scope"],
        "prior_policy": copy.deepcopy(prior_record),
        "failure_evidence": copy.deepcopy(failure_record),
        "ratification": copy.deepcopy(ratification),
        "limitations": copy.deepcopy(limitations),
    })
    effective["expected_contract"] = copy.deepcopy(
        prior_policy["expected_contract"])
    effective["expected_contract"]["moving_movement_algorithm"] = \
        "qualified_envelope_hysteresis_v11"
    effective["_verified_prior_policy"] = verified_prior
    effective["_verified_failure_evidence"] = verified_failure
    for key in ("_verified_threshold_policy", "_verified_geometry_plan",
                "_verified_qualification_aperture"):
        effective[key] = prior_policy.get(key)
    return effective


def validate_v10_policy(policy: dict[str, Any], source: Path) -> dict[str, Any]:
    require(set(policy) == {
                "schema", "schema_version", "policy_id", "policy_status",
                "scope", "prior_policy", "failure_evidence",
                "ratification", "limitations"},
            f"{source}: v10 amendment inventory changed")
    require(policy.get("policy_id") ==
                "moving-nested-reference-frontier-v10-quantized-trusted-balance-amendment" and
            policy.get("policy_status") == "ratified" and
            isinstance(policy.get("scope"), str) and policy["scope"],
            f"{source}: v10 identity is invalid")
    prior_record = object_value(policy.get("prior_policy"),
                                f"{source}: prior_policy")
    require(set(prior_record) ==
                {"path", "sha256", "schema_version", "policy_id"} and
            prior_record.get("schema_version") == 9 and
            prior_record.get("sha256") == V9_POLICY_SHA256 and
            prior_record.get("policy_id") ==
                "moving-nested-reference-frontier-v9-receding-trusted-face-amendment",
            f"{source}: v10 prior-policy binding changed")
    verified_prior = resolve_bound_record(
        prior_record, source, f"{source}: prior_policy")
    prior_policy = validate_policy(
        load_json(Path(verified_prior["path"])),
        Path(verified_prior["path"]))

    failure_record = object_value(policy.get("failure_evidence"),
                                  f"{source}: failure_evidence")
    require(set(failure_record) ==
                {"path", "sha256", "schema_version", "reason"} and
            failure_record.get("schema_version") == 1 and
            failure_record.get("sha256") ==
                QUANTIZED_TRUSTED_BALANCE_FAILURE_SHA256 and
            failure_record.get("reason") ==
                "quantized_balanced_trusted_face_unrecognized",
            f"{source}: v10 failure-evidence binding changed")
    verified_failure = resolve_bound_record(
        failure_record, source, f"{source}: failure_evidence")
    failure = load_json(Path(verified_failure["path"]))
    require(failure.get("schema") ==
                "nimbus.qualification.moving_nested_reference_frontier.movement_failure" and
            failure.get("schema_version") == 1 and
            failure.get("status") == "invalid" and
            failure.get("qualification_disposition") == "incomplete" and
            failure.get("qualification_eligible") is False and
            failure.get("reason") ==
                "quantized_balanced_trusted_face_unrecognized" and
            failure.get("comparison_performed") is False and
            failure.get("candidate_metric_values_inspected") is False,
            f"{source}: bound quantized trusted-balance failure identity changed")

    ratification = object_value(policy.get("ratification"),
                                f"{source}: ratification")
    require(set(ratification) == {
                "decision", "review_date", "reviewer_role",
                "failed_v9_quantization_tie_sequence_observed",
                "candidate_metric_values_inspected",
                "minimum_input_series_schema_version",
                "quantized_trusted_balance_amendment"} and
            ratification.get("decision") ==
                "post_failure_quantized_trusted_balance_amendment_of_v9_and_blinded_v2_thresholds" and
            ratification.get(
                "failed_v9_quantization_tie_sequence_observed") is True and
            ratification.get("candidate_metric_values_inspected") is False and
            ratification.get("minimum_input_series_schema_version") == 2 and
            isinstance(ratification.get("review_date"), str) and
            isinstance(ratification.get("reviewer_role"), str),
            f"{source}: v10 amendment provenance is incomplete")
    amendment = object_value(
        ratification.get("quantized_trusted_balance_amendment"),
        f"{source}: quantized_trusted_balance_amendment")
    require(amendment == {
                "reason":
                    "fail_closed_parent_cell_quantization_tie_unrecognized",
                "algorithm": "qualified_envelope_hysteresis_v12",
                "eligibility":
                    "a single approaching inner-margin face and its opposite remain outside the immutable relaxation zone and no bounded whole-parent-cell translation in the approaching direction strictly improves the axis minimum clearance",
                "disposition":
                    "trusted_interior_contained_without_mutation",
                "single_breached_face_per_axis_required": True,
                "both_axis_faces_outside_relaxation_required": True,
                "approaching_face_required": True,
                "selected_motion_or_signed_face_motion_may_establish_approach": True,
                "bounded_shift_search_starts_parent_cells": 1,
                "bounded_shift_search_ends_parent_cells": 14,
                "strict_axis_minimum_improvement_available": False,
                "movement_performed": False,
                "current_relaxation_entry_recoverable": False,
                "ordinary_or_projected_movement_changed_when_improvement_exists": False,
                "receding_trusted_face_state_changed": False,
                "orthogonal_containment_reserve_changed": False,
                "maximum_shift_cap_parent_cells": 14,
                "numerical_thresholds_changed": False,
                "geometry_or_aperture_changed": False,
                "measurement_cadence_changed": False,
                "failed_candidate_disposition": "invalid_not_comparable"},
            f"{source}: quantized trusted-balance amendment changed")
    limitations = policy.get("limitations")
    require(isinstance(limitations, list) and limitations and
            all(isinstance(value, str) and value for value in limitations),
            f"{source}: v10 limitations are incomplete")

    effective = copy.deepcopy(prior_policy)
    for key in tuple(effective):
        if key.startswith("_verified_"):
            del effective[key]
    effective.update({
        "schema_version": 10,
        "policy_id": policy["policy_id"],
        "policy_status": "ratified",
        "scope": policy["scope"],
        "prior_policy": copy.deepcopy(prior_record),
        "failure_evidence": copy.deepcopy(failure_record),
        "ratification": copy.deepcopy(ratification),
        "limitations": copy.deepcopy(limitations),
    })
    effective["expected_contract"] = copy.deepcopy(
        prior_policy["expected_contract"])
    effective["expected_contract"]["moving_movement_algorithm"] = \
        "qualified_envelope_hysteresis_v12"
    effective["_verified_prior_policy"] = verified_prior
    effective["_verified_failure_evidence"] = verified_failure
    for key in ("_verified_threshold_policy", "_verified_geometry_plan",
                "_verified_qualification_aperture"):
        effective[key] = prior_policy.get(key)
    return effective


def validate_v11_policy(policy: dict[str, Any], source: Path) -> dict[str, Any]:
    require(set(policy) == {
                "schema", "schema_version", "policy_id", "policy_status",
                "scope", "prior_policy", "failure_evidence",
                "ratification", "limitations"},
            f"{source}: v11 amendment inventory changed")
    require(policy.get("policy_id") ==
                "moving-nested-reference-frontier-v11-ordinary-trusted-improvement-amendment" and
            policy.get("policy_status") == "ratified" and
            isinstance(policy.get("scope"), str) and policy["scope"],
            f"{source}: v11 identity is invalid")
    prior_record = object_value(policy.get("prior_policy"),
                                f"{source}: prior_policy")
    require(set(prior_record) ==
                {"path", "sha256", "schema_version", "policy_id"} and
            prior_record.get("schema_version") == 10 and
            prior_record.get("sha256") == V10_POLICY_SHA256 and
            prior_record.get("policy_id") ==
                "moving-nested-reference-frontier-v10-quantized-trusted-balance-amendment",
            f"{source}: v11 prior-policy binding changed")
    verified_prior = resolve_bound_record(
        prior_record, source, f"{source}: prior_policy")
    prior_policy = validate_policy(
        load_json(Path(verified_prior["path"])),
        Path(verified_prior["path"]))

    failure_record = object_value(policy.get("failure_evidence"),
                                  f"{source}: failure_evidence")
    require(set(failure_record) ==
                {"path", "sha256", "schema_version", "reason"} and
            failure_record.get("schema_version") == 1 and
            failure_record.get("sha256") ==
                ORDINARY_TRUSTED_IMPROVEMENT_FAILURE_SHA256 and
            failure_record.get("reason") ==
                "ordinary_trusted_improvement_unrecognized",
            f"{source}: v11 failure-evidence binding changed")
    verified_failure = resolve_bound_record(
        failure_record, source, f"{source}: failure_evidence")
    failure = load_json(Path(verified_failure["path"]))
    require(failure.get("schema") ==
                "nimbus.qualification.moving_nested_reference_frontier.movement_failure" and
            failure.get("schema_version") == 1 and
            failure.get("status") == "invalid" and
            failure.get("qualification_disposition") == "incomplete" and
            failure.get("qualification_eligible") is False and
            failure.get("reason") ==
                "ordinary_trusted_improvement_unrecognized" and
            failure.get("comparison_performed") is False and
            failure.get("candidate_metric_values_inspected") is False,
            f"{source}: bound ordinary trusted-improvement failure identity changed")

    ratification = object_value(policy.get("ratification"),
                                f"{source}: ratification")
    require(set(ratification) == {
                "decision", "review_date", "reviewer_role",
                "failed_v10_ordinary_improvement_sequence_observed",
                "candidate_metric_values_inspected",
                "minimum_input_series_schema_version",
                "ordinary_trusted_improvement_amendment"} and
            ratification.get("decision") ==
                "post_failure_ordinary_trusted_improvement_amendment_of_v10_and_blinded_v2_thresholds" and
            ratification.get(
                "failed_v10_ordinary_improvement_sequence_observed") is True and
            ratification.get("candidate_metric_values_inspected") is False and
            ratification.get("minimum_input_series_schema_version") == 2 and
            isinstance(ratification.get("review_date"), str) and
            isinstance(ratification.get("reviewer_role"), str),
            f"{source}: v11 amendment provenance is incomplete")
    amendment = object_value(
        ratification.get("ordinary_trusted_improvement_amendment"),
        f"{source}: ordinary_trusted_improvement_amendment")
    require(amendment == {
                "reason":
                    "fail_closed_strictly_improving_trusted_shift_unrecognized",
                "algorithm": "qualified_envelope_hysteresis_v13",
                "eligibility":
                    "an ordinary approaching inner-margin breach cannot recover the grid-attainable release margin but a bounded whole-parent-cell translation in the approaching direction strictly improves the axis minimum clearance while every face remains outside relaxation",
                "disposition":
                    "trusted_interior_containment_with_ordinary_confirmation",
                "ordinary_confirmation_samples_required": 2,
                "strict_axis_minimum_improvement_required": True,
                "all_faces_outside_relaxation_required": True,
                "bounded_shift_optimizer":
                    "existing_best_trusted_containment_shift",
                "projected_relaxation_immediacy_changed": False,
                "quantized_no_improvement_state_changed": False,
                "receding_trusted_face_state_changed": False,
                "current_relaxation_entry_recoverable": False,
                "orthogonal_containment_reserve_changed": False,
                "maximum_shift_cap_parent_cells": 14,
                "numerical_thresholds_changed": False,
                "geometry_or_aperture_changed": False,
                "measurement_cadence_changed": False,
                "failed_candidate_disposition": "invalid_not_comparable"},
            f"{source}: ordinary trusted-improvement amendment changed")
    limitations = policy.get("limitations")
    require(isinstance(limitations, list) and limitations and
            all(isinstance(value, str) and value for value in limitations),
            f"{source}: v11 limitations are incomplete")

    effective = copy.deepcopy(prior_policy)
    for key in tuple(effective):
        if key.startswith("_verified_"):
            del effective[key]
    effective.update({
        "schema_version": 11,
        "policy_id": policy["policy_id"],
        "policy_status": "ratified",
        "scope": policy["scope"],
        "prior_policy": copy.deepcopy(prior_record),
        "failure_evidence": copy.deepcopy(failure_record),
        "ratification": copy.deepcopy(ratification),
        "limitations": copy.deepcopy(limitations),
    })
    effective["expected_contract"] = copy.deepcopy(
        prior_policy["expected_contract"])
    effective["expected_contract"]["moving_movement_algorithm"] = \
        "qualified_envelope_hysteresis_v13"
    effective["_verified_prior_policy"] = verified_prior
    effective["_verified_failure_evidence"] = verified_failure
    for key in ("_verified_threshold_policy", "_verified_geometry_plan",
                "_verified_qualification_aperture"):
        effective[key] = prior_policy.get(key)
    return effective


def validate_v12_policy(policy: dict[str, Any], source: Path) -> dict[str, Any]:
    require(set(policy) == {
                "schema", "schema_version", "policy_id", "policy_status",
                "scope", "prior_policy", "failure_evidence",
                "ratification", "limitations"},
            f"{source}: v12 amendment inventory changed")
    require(policy.get("policy_id") ==
                "moving-nested-reference-frontier-v12-parent-stencil-limit-amendment" and
            policy.get("policy_status") == "ratified" and
            isinstance(policy.get("scope"), str) and policy["scope"],
            f"{source}: v12 identity is invalid")
    prior_record = object_value(policy.get("prior_policy"),
                                f"{source}: prior_policy")
    require(set(prior_record) ==
                {"path", "sha256", "schema_version", "policy_id"} and
            prior_record.get("schema_version") == 11 and
            prior_record.get("sha256") == V11_POLICY_SHA256 and
            prior_record.get("policy_id") ==
                "moving-nested-reference-frontier-v11-ordinary-trusted-improvement-amendment",
            f"{source}: v12 prior-policy binding changed")
    verified_prior = resolve_bound_record(
        prior_record, source, f"{source}: prior_policy")
    prior_policy = validate_policy(
        load_json(Path(verified_prior["path"])),
        Path(verified_prior["path"]))

    failure_record = object_value(policy.get("failure_evidence"),
                                  f"{source}: failure_evidence")
    require(set(failure_record) ==
                {"path", "sha256", "schema_version", "reason"} and
            failure_record.get("schema_version") == 1 and
            failure_record.get("sha256") ==
                PARENT_STENCIL_LIMIT_FAILURE_SHA256 and
            failure_record.get("reason") ==
                "parent_stencil_shift_bound_unrepresented",
            f"{source}: v12 failure-evidence binding changed")
    verified_failure = resolve_bound_record(
        failure_record, source, f"{source}: failure_evidence")
    failure = load_json(Path(verified_failure["path"]))
    require(failure.get("schema") ==
                "nimbus.qualification.moving_nested_reference_frontier.movement_failure" and
            failure.get("schema_version") == 1 and
            failure.get("status") == "invalid" and
            failure.get("qualification_disposition") == "incomplete" and
            failure.get("qualification_eligible") is False and
            failure.get("reason") ==
                "parent_stencil_shift_bound_unrepresented" and
            failure.get("comparison_performed") is False and
            failure.get("candidate_metric_values_inspected") is False,
            f"{source}: bound parent-stencil-limit failure identity changed")

    ratification = object_value(policy.get("ratification"),
                                f"{source}: ratification")
    require(set(ratification) == {
                "decision", "review_date", "reviewer_role",
                "failed_v11_parent_stencil_limit_observed",
                "candidate_metric_values_inspected",
                "minimum_input_series_schema_version",
                "parent_stencil_limit_amendment"} and
            ratification.get("decision") ==
                "post_failure_parent_stencil_limit_amendment_of_v11_and_blinded_v2_thresholds" and
            ratification.get(
                "failed_v11_parent_stencil_limit_observed") is True and
            ratification.get("candidate_metric_values_inspected") is False and
            ratification.get("minimum_input_series_schema_version") == 2 and
            isinstance(ratification.get("review_date"), str) and
            isinstance(ratification.get("reviewer_role"), str),
            f"{source}: v12 amendment provenance is incomplete")
    amendment = object_value(
        ratification.get("parent_stencil_limit_amendment"),
        f"{source}: parent_stencil_limit_amendment")
    require(amendment == {
                "reason":
                    "fail_closed_live_parent_stencil_shift_bound_unrepresented",
                "algorithm": "qualified_envelope_hysteresis_v14",
                "legal_bounds_source": "live_one_way_nest_relationship",
                "legal_bounds_units":
                    "signed_parent_cells_relative_to_current_child_origin",
                "candidate_constraint":
                    "intersection_of_policy_cap_and_exact_legal_placement_interval",
                "candidate_paths_constrained": [
                    "ordinary", "projected_relaxation_risk",
                    "orthogonal_release_correction", "trusted_containment",
                    "orthogonal_containment_reserve"],
                "limit_state_eligibility":
                    "an approaching inner-margin breach has a policy-bounded strictly improving shift but no legal strictly improving shift solely because the child is at the exact parent-stencil placement limit, while every selected face remains outside relaxation",
                "limit_state_disposition":
                    "parent_stencil_limited_contained_without_mutation",
                "all_faces_outside_relaxation_required": True,
                "strict_policy_bounded_improvement_required": True,
                "legal_improvement_available": False,
                "movement_performed": False,
                "current_relaxation_entry_recoverable": False,
                "malformed_live_bounds_fail_closed": True,
                "lower_level_relationship_validation_retained": True,
                "maximum_shift_cap_parent_cells": 14,
                "numerical_thresholds_changed": False,
                "geometry_or_aperture_changed": False,
                "measurement_cadence_changed": False,
                "failed_candidate_disposition": "invalid_not_comparable"},
            f"{source}: parent-stencil-limit amendment changed")
    limitations = policy.get("limitations")
    require(isinstance(limitations, list) and limitations and
            all(isinstance(value, str) and value for value in limitations),
            f"{source}: v12 limitations are incomplete")

    effective = copy.deepcopy(prior_policy)
    for key in tuple(effective):
        if key.startswith("_verified_"):
            del effective[key]
    effective.update({
        "schema_version": 12,
        "policy_id": policy["policy_id"],
        "policy_status": "ratified",
        "scope": policy["scope"],
        "prior_policy": copy.deepcopy(prior_record),
        "failure_evidence": copy.deepcopy(failure_record),
        "ratification": copy.deepcopy(ratification),
        "limitations": copy.deepcopy(limitations),
    })
    effective["expected_contract"] = copy.deepcopy(
        prior_policy["expected_contract"])
    effective["expected_contract"]["moving_movement_algorithm"] = \
        "qualified_envelope_hysteresis_v14"
    effective["_verified_prior_policy"] = verified_prior
    effective["_verified_failure_evidence"] = verified_failure
    for key in ("_verified_threshold_policy", "_verified_geometry_plan",
                "_verified_qualification_aperture"):
        effective[key] = prior_policy.get(key)
    return effective


def validate_v13_policy(policy: dict[str, Any], source: Path) -> dict[str, Any]:
    require(set(policy) == {
                "schema", "schema_version", "policy_id", "policy_status",
                "scope", "prior_policy", "backend_evidence",
                "ratification", "limitations"},
            f"{source}: v13 amendment inventory changed")
    require(policy.get("policy_id") ==
                "moving-nested-reference-frontier-v13-vulkan-backend-amendment" and
            policy.get("policy_status") == "ratified" and
            isinstance(policy.get("scope"), str) and policy["scope"],
            f"{source}: v13 identity is invalid")
    prior_record = object_value(policy.get("prior_policy"),
                                f"{source}: prior_policy")
    require(set(prior_record) ==
                {"path", "sha256", "schema_version", "policy_id"} and
            prior_record.get("schema_version") == 12 and
            prior_record.get("sha256") == V12_POLICY_SHA256 and
            prior_record.get("policy_id") ==
                "moving-nested-reference-frontier-v12-parent-stencil-limit-amendment",
            f"{source}: v13 prior-policy binding changed")
    verified_prior = resolve_bound_record(
        prior_record, source, f"{source}: prior_policy")
    prior_policy = validate_policy(
        load_json(Path(verified_prior["path"])), Path(verified_prior["path"]))

    evidence_record = object_value(policy.get("backend_evidence"),
                                   f"{source}: backend_evidence")
    require(set(evidence_record) ==
                {"path", "sha256", "schema_version", "status"} and
            evidence_record.get("schema_version") == 1 and
            evidence_record.get("status") == "pass" and
            evidence_record.get("sha256") ==
                VULKAN_REFERENCE_RUNTIME_EVIDENCE_SHA256,
            f"{source}: v13 backend-evidence binding changed")
    verified_evidence = resolve_bound_record(
        evidence_record, source, f"{source}: backend_evidence")
    evidence = load_json(Path(verified_evidence["path"]))
    require(evidence.get("schema") ==
                "nimbus.evidence.moving_nested_vulkan_reference_runtime" and
            evidence.get("schema_version") == 1 and
            evidence.get("status") == "pass" and
            evidence.get("qualification_eligible") is False and
            evidence.get("candidate_metric_values_inspected") is False,
            f"{source}: bound Vulkan reference evidence identity changed")
    evidence_claims = object_value(evidence.get("claims"),
                                   f"{source}: backend evidence claims")
    require(evidence_claims.get("same_binary_candidate_and_reference") is True and
            evidence_claims.get("same_vulkan_backend_candidate_and_reference") is True and
            evidence_claims.get("same_comparison_level_equations") is True and
            evidence_claims.get("strict_finite_bounded_reference_outputs") is True and
            evidence_claims.get("formal_comparison_complete") is False,
            f"{source}: Vulkan reference evidence overclaims or is incomplete")
    runtime = object_value(evidence.get("reference_runtime"),
                           f"{source}: backend evidence runtime")
    device = object_value(evidence.get("device_evidence"),
                          f"{source}: backend evidence device")
    require(runtime.get("fresh_configured_start") is True and
            runtime.get("uninterrupted_single_process") is True and
            runtime.get("final_time_s") == 4500 and
            runtime.get("tracker_samples") == 150 and
            runtime.get("diagnostic_samples") == 150 and
            runtime.get("portable_checkpoints") == 10 and
            device.get("fallback_off_physical_vulkan") is True and
            device.get("dispatches") == 42000 and
            device.get("host_authoritative_kessler_state_commit") is True,
            f"{source}: Vulkan reference runtime evidence is incomplete")
    for container_name, record_name in (
        ("inputs", "reference_config"),
        ("inputs", "moving_lifetime_chain"),
        ("reference_runtime", "report"),
    ):
        container = object_value(evidence.get(container_name),
                                 f"{source}: evidence {container_name}")
        record = object_value(container.get(record_name),
                              f"{source}: evidence {record_name}")
        path = repository_evidence_path(record.get("path"))
        require(path.is_file() and sha256(path) == record.get("sha256"),
                f"{source}: evidence-bound artifact changed: {path}")

    ratification = object_value(policy.get("ratification"),
                                f"{source}: ratification")
    require(set(ratification) == {
                "decision", "review_date", "reviewer_role",
                "candidate_metric_values_inspected",
                "minimum_input_series_schema_version",
                "backend_identity_amendment"} and
            ratification.get("decision") ==
                "pre_comparison_vulkan_backend_identity_amendment_of_v12" and
            ratification.get("candidate_metric_values_inspected") is False and
            ratification.get("minimum_input_series_schema_version") == 2 and
            isinstance(ratification.get("review_date"), str) and
            isinstance(ratification.get("reviewer_role"), str),
            f"{source}: v13 amendment provenance is incomplete")
    amendment = object_value(
        ratification.get("backend_identity_amendment"),
        f"{source}: backend_identity_amendment")
    require(amendment == {
                "reason": "same_backend_reference_now_available",
                "candidate_backend": "vulkan",
                "reference_backend": "vulkan",
                "fallback_allowed": False,
                "same_binary_required": True,
                "physics_signature_sha256": VULKAN_PHYSICS_SIGNATURE_SHA256,
                "host_authoritative_kessler_state_commit": True,
                "native_smagorinsky_sgs_dispatch_required": True,
                "native_kessler_process_dispatch_required": True,
                "numerical_thresholds_changed": False,
                "movement_algorithm_changed": False,
                "geometry_or_aperture_changed": False,
                "measurement_cadence_changed": False},
            f"{source}: Vulkan backend identity amendment changed")
    limitations = policy.get("limitations")
    require(isinstance(limitations, list) and limitations and
            all(isinstance(value, str) and value for value in limitations),
            f"{source}: v13 limitations are incomplete")

    effective = copy.deepcopy(prior_policy)
    for key in tuple(effective):
        if key.startswith("_verified_"):
            del effective[key]
    effective.update({
        "schema_version": 13,
        "policy_id": policy["policy_id"],
        "policy_status": "ratified",
        "scope": policy["scope"],
        "prior_policy": copy.deepcopy(prior_record),
        "backend_evidence": copy.deepcopy(evidence_record),
        "ratification": copy.deepcopy(ratification),
        "limitations": copy.deepcopy(limitations),
    })
    effective["expected_contract"] = copy.deepcopy(
        prior_policy["expected_contract"])
    effective["expected_contract"]["physics_signature_sha256"] = \
        VULKAN_PHYSICS_SIGNATURE_SHA256
    effective["_verified_prior_policy"] = verified_prior
    effective["_verified_backend_evidence"] = verified_evidence
    for key in ("_verified_threshold_policy", "_verified_geometry_plan",
                "_verified_qualification_aperture"):
        effective[key] = prior_policy.get(key)
    return effective


def validate_v14_policy(policy: dict[str, Any], source: Path) -> dict[str, Any]:
    require(set(policy) == {
                "schema", "schema_version", "policy_id", "policy_status",
                "scope", "prior_policy", "backend_evidence",
                "ratification", "limitations"},
            f"{source}: v14 amendment inventory changed")
    require(policy.get("policy_id") ==
                "moving-nested-reference-frontier-v14-metal-backend-amendment" and
            policy.get("policy_status") == "ratified" and
            isinstance(policy.get("scope"), str) and policy["scope"],
            f"{source}: v14 identity is invalid")
    prior_record = object_value(policy.get("prior_policy"),
                                f"{source}: prior_policy")
    require(set(prior_record) ==
                {"path", "sha256", "schema_version", "policy_id"} and
            prior_record.get("schema_version") == 13 and
            prior_record.get("sha256") == V13_POLICY_SHA256 and
            prior_record.get("policy_id") ==
                "moving-nested-reference-frontier-v13-vulkan-backend-amendment",
            f"{source}: v14 prior-policy binding changed")
    verified_prior = resolve_bound_record(
        prior_record, source, f"{source}: prior_policy")
    prior_policy = validate_policy(
        load_json(Path(verified_prior["path"])), Path(verified_prior["path"]))

    evidence_record = object_value(policy.get("backend_evidence"),
                                   f"{source}: backend_evidence")
    require(set(evidence_record) ==
                {"path", "sha256", "schema_version", "status"} and
            evidence_record.get("schema_version") == 1 and
            evidence_record.get("status") == "pass" and
            valid_sha256(evidence_record.get("sha256")),
            f"{source}: v14 backend-evidence binding changed")
    verified_evidence = resolve_bound_record(
        evidence_record, source, f"{source}: backend_evidence")
    evidence = load_json(Path(verified_evidence["path"]))
    require(evidence.get("schema") ==
                "nimbus.evidence.moving_nested_metal_reference_runtime" and
            evidence.get("schema_version") == 1 and
            evidence.get("status") == "pass" and
            evidence.get("qualification_eligible") is False and
            evidence.get("candidate_metric_values_inspected") is False,
            f"{source}: bound Metal reference evidence identity changed")
    evidence_claims = object_value(evidence.get("claims"),
                                   f"{source}: backend evidence claims")
    require(evidence_claims.get("same_binary_candidate_and_reference") is True and
            evidence_claims.get("same_metal_backend_candidate_and_reference") is True and
            evidence_claims.get("same_comparison_level_equations") is True and
            evidence_claims.get("strict_finite_bounded_reference_outputs") is True and
            evidence_claims.get("formal_comparison_complete") is False and
            evidence_claims.get("public_moving_nest_admitted") is False and
            evidence_claims.get("throughput_promotion") is False,
            f"{source}: Metal reference evidence overclaims or is incomplete")
    runtime = object_value(evidence.get("reference_runtime"),
                           f"{source}: backend evidence runtime")
    device = object_value(evidence.get("device_evidence"),
                          f"{source}: backend evidence device")
    require(runtime.get("fresh_configured_start") is True and
            runtime.get("uninterrupted_single_process") is True and
            runtime.get("final_time_s") == 4500 and
            runtime.get("tracker_samples") == 150 and
            runtime.get("diagnostic_samples") == 150 and
            runtime.get("portable_checkpoints") == 10 and
            device.get("fallback_off_physical_metal") is True and
            type(device.get("dispatches")) is int and
            device["dispatches"] > 0 and
            device.get("host_authoritative_kessler_state_commit") is True,
            f"{source}: Metal reference runtime evidence is incomplete")
    stages = object_value(device.get("stage_dispatches"),
                          f"{source}: Metal stage dispatches")
    require(any(count > 0 for name, count in stages.items()
                if "smagorinsky" in name or "sgs" in name) and
            any(count > 0 for name, count in stages.items()
                if "kessler" in name),
            f"{source}: Metal native stage evidence is incomplete")
    for container_name, record_name in (
        ("inputs", "reference_config"),
        ("inputs", "moving_lifetime_chain"),
        ("reference_runtime", "report"),
    ):
        container = object_value(evidence.get(container_name),
                                 f"{source}: evidence {container_name}")
        record = object_value(container.get(record_name),
                              f"{source}: evidence {record_name}")
        path = repository_evidence_path(record.get("path"))
        require(path.is_file() and sha256(path) == record.get("sha256"),
                f"{source}: evidence-bound artifact changed: {path}")

    ratification = object_value(policy.get("ratification"),
                                f"{source}: ratification")
    require(set(ratification) == {
                "decision", "review_date", "reviewer_role",
                "candidate_metric_values_inspected",
                "minimum_input_series_schema_version",
                "backend_identity_amendment"} and
            ratification.get("decision") ==
                "pre_comparison_metal_backend_identity_amendment_of_v13" and
            ratification.get("candidate_metric_values_inspected") is False and
            ratification.get("minimum_input_series_schema_version") == 2 and
            isinstance(ratification.get("review_date"), str) and
            isinstance(ratification.get("reviewer_role"), str),
            f"{source}: v14 amendment provenance is incomplete")
    amendment = object_value(
        ratification.get("backend_identity_amendment"),
        f"{source}: backend_identity_amendment")
    require(amendment == {
                "reason": "same_backend_reference_now_available",
                "candidate_backend": "metal",
                "reference_backend": "metal",
                "fallback_allowed": False,
                "same_binary_required": True,
                "physics_signature_sha256": METAL_PHYSICS_SIGNATURE_SHA256,
                "host_authoritative_kessler_state_commit": True,
                "native_smagorinsky_sgs_dispatch_required": True,
                "native_kessler_process_dispatch_required": True,
                "numerical_thresholds_changed": False,
                "movement_algorithm_changed": False,
                "geometry_or_aperture_changed": False,
                "measurement_cadence_changed": False},
            f"{source}: Metal backend identity amendment changed")
    limitations = policy.get("limitations")
    require(isinstance(limitations, list) and limitations and
            all(isinstance(value, str) and value for value in limitations),
            f"{source}: v14 limitations are incomplete")

    effective = copy.deepcopy(prior_policy)
    for key in tuple(effective):
        if key.startswith("_verified_"):
            del effective[key]
    effective.update({
        "schema_version": 14,
        "policy_id": policy["policy_id"],
        "policy_status": "ratified",
        "scope": policy["scope"],
        "prior_policy": copy.deepcopy(prior_record),
        "backend_evidence": copy.deepcopy(evidence_record),
        "ratification": copy.deepcopy(ratification),
        "limitations": copy.deepcopy(limitations),
    })
    effective["expected_contract"] = copy.deepcopy(
        prior_policy["expected_contract"])
    effective["expected_contract"]["physics_signature_sha256"] = \
        METAL_PHYSICS_SIGNATURE_SHA256
    effective["_verified_prior_policy"] = verified_prior
    effective["_verified_backend_evidence"] = verified_evidence
    for key in ("_verified_threshold_policy", "_verified_geometry_plan",
                "_verified_qualification_aperture"):
        effective[key] = prior_policy.get(key)
    return effective


def validate_policy(policy: dict[str, Any], source: Path) -> dict[str, Any]:
    require(policy.get("schema") == POLICY_SCHEMA, f"{source}: wrong policy schema")
    schema_version = policy.get("schema_version")
    require(schema_version in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
            f"{source}: wrong policy schema version")
    if schema_version == 14:
        return validate_v14_policy(policy, source)
    if schema_version == 13:
        return validate_v13_policy(policy, source)
    if schema_version == 12:
        return validate_v12_policy(policy, source)
    if schema_version == 11:
        return validate_v11_policy(policy, source)
    if schema_version == 10:
        return validate_v10_policy(policy, source)
    if schema_version == 9:
        return validate_v9_policy(policy, source)
    if schema_version == 8:
        return validate_v8_policy(policy, source)
    if schema_version == 7:
        return validate_v7_policy(policy, source)
    if schema_version == 6:
        return validate_v6_policy(policy, source)
    if schema_version == 1:
        require(policy.get("policy_status") == "measurement_only",
                f"{source}: v1 policy is permanently measurement-only")
    else:
        require(policy.get("policy_status") == "ratified",
                f"{source}: ratified policy must be explicitly ratified")
    required = policy.get("required_metric_families")
    require(required == ["path", "intensity", "cold_pool", "precipitation"],
            f"{source}: v1 requires all four metric families in canonical order")
    object_value(policy.get("expected_contract"), f"{source}: expected_contract")
    object_value(policy.get("coverage_thresholds"), f"{source}: coverage_thresholds")
    thresholds = object_value(policy.get("metric_thresholds"),
                              f"{source}: metric_thresholds")
    require(set(thresholds) == set(required),
            f"{source}: metric thresholds do not cover every required family")
    definitions = object_value(policy.get("derived_definitions"),
                               f"{source}: derived_definitions")
    require(set(definitions) == {"cold_pool", "precipitation"},
            f"{source}: derived definitions are incomplete")
    for family in ("cold_pool", "precipitation"):
        definition = object_value(definitions[family],
                                  f"{source}: derived_definitions.{family}")
        require(isinstance(definition.get("id"), str) and definition["id"],
                f"{source}: derived_definitions.{family}.id must be nonempty")
    if schema_version in {2, 3, 4, 5}:
        expected = object_value(policy["expected_contract"],
                                f"{source}: expected_contract")
        expected_inventory = {
                    "coordinate_frame", "detector_algorithm",
                    "continuity_algorithm", "moving_tracking_source",
                    "reference_tracking_source", "moving_movement_algorithm",
                    "comparison_region", "relaxation_exclusion_cells",
                    "trusted_region_cell_count", "trusted_region_shape_cells",
                    "horizontal_grid_spacing_m",
                    "grid_phase_origin_m", "physics_signature_sha256",
                    "sample_start_s", "sample_interval_s", "duration_s"}
        if schema_version in {3, 4, 5}:
            expected_inventory |= {"geometry_plan", "qualification_aperture"}
        require(set(expected) == expected_inventory,
                f"{source}: v{schema_version} expected contract inventory changed")
        require(valid_sha256(expected.get("physics_signature_sha256")),
                f"{source}: expected physics signature is invalid")
        trusted_count = expected.get("trusted_region_cell_count")
        require(type(trusted_count) is int and trusted_count > 0,
                f"{source}: trusted-region cell count must be positive")
        trusted_shape = object_value(expected.get("trusted_region_shape_cells"),
                                     f"{source}: trusted-region shape")
        require(set(trusted_shape) == {"x", "y"} and
                all(type(value) is int and value > 0
                    for value in trusted_shape.values()) and
                trusted_shape["x"] * trusted_shape["y"] == trusted_count,
                f"{source}: trusted-region shape is invalid")
        phase = object_value(expected.get("grid_phase_origin_m"),
                             f"{source}: expected grid phase")
        require(set(phase) == {"x", "y"},
                f"{source}: expected grid phase inventory changed")
        finite_number(phase.get("x"), f"{source}: expected phase x")
        finite_number(phase.get("y"), f"{source}: expected phase y")
        if schema_version in {3, 4, 5}:
            geometry_record = object_value(
                expected.get("geometry_plan"), f"{source}: geometry_plan")
            aperture_record = object_value(
                expected.get("qualification_aperture"),
                f"{source}: qualification_aperture")
            require(set(geometry_record) ==
                    {"path", "sha256", "schema_version", "plan_id"} and
                geometry_record.get("schema_version") == 1 and
                geometry_record.get("sha256") == GEOMETRY_PLAN_SHA256 and
                geometry_record.get("plan_id") ==
                    "moving_nested_reference_geometry_v1",
                f"{source}: geometry-plan binding changed")
            require(set(aperture_record) ==
                    {"path", "sha256", "schema_version", "aperture_id"} and
                aperture_record.get("schema_version") == 1 and
                aperture_record.get("sha256") ==
                    QUALIFICATION_APERTURE_SHA256 and
                aperture_record.get("aperture_id") ==
                    "moving_nested_fixed_100x96_v1",
                f"{source}: qualification-aperture binding changed")
            verified_geometry = resolve_bound_record(
                geometry_record, source, f"{source}: geometry_plan")
            verified_aperture = resolve_bound_record(
                aperture_record, source, f"{source}: qualification_aperture")
            geometry_document = load_json(Path(verified_geometry["path"]))
            aperture_document = load_json(Path(verified_aperture["path"]))
            require(geometry_document.get("schema") == GEOMETRY_PLAN_SCHEMA and
                geometry_document.get("schema_version") == 1 and
                geometry_document.get("plan_id") == geometry_record["plan_id"],
                f"{source}: geometry-plan artifact identity changed")
            require(aperture_document.get("schema") == APERTURE_SCHEMA and
                aperture_document.get("schema_version") == 1 and
                aperture_document.get("aperture_id") ==
                    aperture_record["aperture_id"] and
                aperture_document.get("shape_cells") == trusted_shape and
                aperture_document.get("cell_count") == trusted_count and
                aperture_document.get("horizontal_grid_spacing_m") ==
                    expected.get("horizontal_grid_spacing_m") and
                aperture_document.get("grid_phase_origin_m") == phase,
                f"{source}: qualification aperture disagrees with policy geometry")
            policy["_verified_geometry_plan"] = verified_geometry
            policy["_verified_qualification_aperture"] = verified_aperture

        cold_definition = definitions["cold_pool"]
        require(cold_definition == {
                    "id": COLD_POOL_ALGORITHM,
                    "level": "first_physical_scalar_center",
                    "required_fields":
                        ["theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"],
                    "theta_rho_formula":
                        "theta * (1 + qv / epsilon) / (1 + qv + qc + qr + qi + qs + qg + qh), epsilon=Rd/Rv",
                    "environment_reference":
                        "horizontally_homogeneous_initial_theta_rho_at_same_level",
                    "cold_pool_threshold_k": -2.0,
                    "integrated_deficit_definition":
                        "sum(max(0,-theta_rho_perturbation_k) * coincident_cell_area_m2)",
                    "centroid_weight": "max(0,-theta_rho_perturbation_k)"},
                f"{source}: v2 cold-pool definition changed")
        precipitation_definition = definitions["precipitation"]
        require(precipitation_definition == {
                    "id": PRECIPITATION_ALGORITHM,
                    "required_fields": ["accumulated_rainfall"],
                    "effective_depth_definition":
                        "published accumulated_rainfall_mm, including the authoritative accumulation residual",
                    "accumulated_mass_definition":
                        "sum(effective_depth_mm * coincident_cell_area_m2)",
                    "footprint_threshold_mm": 1.0},
                f"{source}: v2 precipitation definition changed")

        coverage = object_value(policy["coverage_thresholds"],
                                f"{source}: coverage_thresholds")
        require(set(coverage) == {
                    "minimum_paired_selected_fraction",
                    "maximum_consecutive_unpaired_samples",
                    "require_final_paired_sample"},
                f"{source}: v2 coverage threshold inventory changed")
        paired_fraction = finite_number(
            coverage.get("minimum_paired_selected_fraction"),
            f"{source}: minimum paired fraction")
        require(0.0 < paired_fraction <= 1.0,
                f"{source}: minimum paired fraction must be in (0,1]")
        maximum_gap = coverage.get("maximum_consecutive_unpaired_samples")
        require(type(maximum_gap) is int and maximum_gap >= 0,
                f"{source}: maximum unpaired samples must be a nonnegative integer")
        require(coverage.get("require_final_paired_sample") is True,
                f"{source}: v2 requires a paired final sample")

        expected_threshold_keys = {
            "path": {"rmse_m", "p95_m", "maximum_m", "final_m"},
            "intensity": {
                "maximum_updraft_normalized_rmse",
                "maximum_updraft_peak_relative_error",
                "maximum_updraft_peak_time_error_s",
                "maximum_absolute_uh_normalized_rmse",
                "maximum_absolute_uh_peak_relative_error",
                "maximum_absolute_uh_peak_time_error_s",
                "updraft_scale_floor_ms", "uh_scale_floor_m2s2"},
            "cold_pool": {
                "area_normalized_rmse", "integrated_deficit_normalized_rmse",
                "minimum_theta_rho_perturbation_rmse_k", "centroid_rmse_m",
                "area_scale_floor_m2", "integrated_deficit_scale_floor_k_m2",
                "minimum_active_samples"},
            "precipitation": {
                "final_accumulated_mass_relative_error",
                "footprint_area_normalized_rmse",
                "p95_depth_normalized_rmse", "maximum_depth_normalized_rmse",
                "mass_scale_floor_kg", "area_scale_floor_m2",
                "depth_scale_floor_mm"},
        }
        for family, keys in expected_threshold_keys.items():
            values = object_value(thresholds[family],
                                  f"{source}: metric_thresholds.{family}")
            require(set(values) == keys,
                    f"{source}: v2 {family} threshold inventory changed")
            for key, value in values.items():
                if key == "minimum_active_samples":
                    require(type(value) is int and value > 0,
                            f"{source}: minimum active samples must be positive")
                else:
                    positive_number(value, f"{source}: {family}.{key}")
        path = thresholds["path"]
        require(path["rmse_m"] <= path["p95_m"] <= path["maximum_m"] and
                path["final_m"] <= path["maximum_m"],
                f"{source}: path thresholds are internally inconsistent")

        ratification = object_value(policy.get("ratification"),
                                    f"{source}: ratification")
        if schema_version == 2:
            require(ratification.get("decision") ==
                        "ratified_before_candidate_observation" and
                    ratification.get("candidate_4500_outputs_observed") is False and
                    ratification.get("minimum_input_series_schema_version") == 2 and
                    isinstance(ratification.get("review_date"), str) and
                    isinstance(ratification.get("reviewer_role"), str),
                    f"{source}: v2 ratification provenance is incomplete")
        elif schema_version == 3:
            require(ratification.get("decision") ==
                        "post_failure_geometry_amendment_of_blinded_v2_thresholds" and
                    ratification.get("candidate_4500_geometry_observed") is True and
                    ratification.get("candidate_metric_values_inspected") is False and
                    ratification.get("minimum_input_series_schema_version") == 2 and
                    isinstance(ratification.get("review_date"), str) and
                    isinstance(ratification.get("reviewer_role"), str),
                    f"{source}: v3 amendment provenance is incomplete")
        elif schema_version == 4:
            require(ratification.get("decision") ==
                        "post_failure_projected_containment_amendment_of_v3_geometry_and_blinded_v2_thresholds" and
                    ratification.get("failed_v3_movement_sequence_observed") is True and
                    ratification.get("candidate_metric_values_inspected") is False and
                    ratification.get("minimum_input_series_schema_version") == 2 and
                    isinstance(ratification.get("review_date"), str) and
                    isinstance(ratification.get("reviewer_role"), str),
                    f"{source}: v4 amendment provenance is incomplete")
        else:
            require(ratification.get("decision") ==
                        "post_failure_maximum_safe_projected_shift_amendment_of_v4_and_blinded_v2_thresholds" and
                    ratification.get("failed_v4_post_commit_sequence_observed") is True and
                    ratification.get("candidate_metric_values_inspected") is False and
                    ratification.get("minimum_input_series_schema_version") == 2 and
                    isinstance(ratification.get("review_date"), str) and
                    isinstance(ratification.get("reviewer_role"), str),
                    f"{source}: v5 amendment provenance is incomplete")
        reviewed = ratification.get("reviewed_sources")
        require(isinstance(reviewed, list) and reviewed,
                f"{source}: ratification reviewed-source list is empty")
        for index, record in enumerate(reviewed):
            item = object_value(record, f"{source}: reviewed_sources[{index}]")
            require(isinstance(item.get("purpose"), str) and item["purpose"],
                    f"{source}: reviewed source purpose is empty")
            resolve_bound_record(item, source,
                                 f"{source}: reviewed_sources[{index}]")
        rationale = object_value(ratification.get("threshold_rationale"),
                                 f"{source}: threshold_rationale")
        require(set(rationale) == {
                    "coverage", "path", "intensity", "cold_pool",
                    "precipitation", "truncation"} and
                all(isinstance(value, str) and value for value in rationale.values()),
                f"{source}: threshold rationale is incomplete")
        if schema_version in {3, 4, 5}:
            threshold_record = object_value(
                policy.get("threshold_policy"), f"{source}: threshold_policy")
            require(set(threshold_record) ==
                        {"path", "sha256", "schema_version", "policy_id"} and
                    threshold_record.get("schema_version") == 2 and
                    threshold_record.get("sha256") ==
                        BLINDED_V2_POLICY_SHA256 and
                    threshold_record.get("policy_id") ==
                        "moving-nested-reference-frontier-v2-ratified",
                    f"{source}: v{schema_version} threshold-policy binding changed")
            verified_threshold_policy = resolve_bound_record(
                threshold_record, source, f"{source}: threshold_policy")
            threshold_policy = load_json(Path(verified_threshold_policy["path"]))
            validate_policy(threshold_policy,
                            Path(verified_threshold_policy["path"]))
            for key in ("required_metric_families", "coverage_thresholds",
                        "metric_thresholds", "derived_definitions"):
                require(policy.get(key) == threshold_policy.get(key),
                        f"{source}: v{schema_version} {key} differs from bound blinded v2 policy")
            for key in ("coordinate_frame", "detector_algorithm",
                        "continuity_algorithm", "moving_tracking_source",
                        "reference_tracking_source",
                        "relaxation_exclusion_cells",
                        "trusted_region_cell_count", "trusted_region_shape_cells",
                        "horizontal_grid_spacing_m", "grid_phase_origin_m",
                        "physics_signature_sha256", "sample_start_s",
                        "sample_interval_s", "duration_s"):
                require(expected.get(key) ==
                            threshold_policy["expected_contract"].get(key),
                        f"{source}: v{schema_version} expected_contract.{key} differs from bound v2 policy")
            expected_movement = (
                threshold_policy["expected_contract"].get(
                    "moving_movement_algorithm")
                if schema_version == 3 else
                "qualified_envelope_hysteresis_v6"
                if schema_version == 4 else
                "qualified_envelope_hysteresis_v7")
            require(expected.get("moving_movement_algorithm") == expected_movement,
                    f"{source}: v{schema_version} movement algorithm is invalid")
        if schema_version == 3:
            failure_record = object_value(
                policy.get("failure_evidence"), f"{source}: failure_evidence")
            require(set(failure_record) ==
                        {"path", "sha256", "schema_version", "reason"} and
                    failure_record.get("schema_version") == 1 and
                    failure_record.get("sha256") == GEOMETRY_FAILURE_SHA256 and
                    failure_record.get("reason") == "unresolved_breached_face",
                    f"{source}: v3 failure-evidence binding changed")
            verified_failure = resolve_bound_record(
                failure_record, source, f"{source}: failure_evidence")
            failure = load_json(Path(verified_failure["path"]))
            require(failure.get("schema") ==
                        "nimbus.evidence.moving_nested_reference_geometry_failure" and
                    failure.get("schema_version") == 1 and
                    failure.get("status") == "invalid_fail_closed" and
                    failure.get("reason") == "unresolved_breached_face",
                    f"{source}: bound geometry failure identity changed")
            amendment = object_value(ratification.get("geometry_amendment"),
                                     f"{source}: geometry_amendment")
            require(amendment == {
                        "reason":
                            "fail_closed_operational_child_geometry_repair_after_unresolved_breached_face",
                        "numerical_thresholds_changed": False,
                        "measurement_shape_changed": False,
                        "measurement_spacing_or_phase_changed": False,
                        "operational_child_is_measurement_aperture": False,
                        "failed_candidate_disposition": "invalid_not_comparable"},
                    f"{source}: fail-closed geometry amendment is incomplete")
            policy["_verified_threshold_policy"] = verified_threshold_policy
            policy["_verified_failure_evidence"] = verified_failure
        elif schema_version == 4:
            prior_record = object_value(
                policy.get("prior_policy"), f"{source}: prior_policy")
            require(set(prior_record) ==
                        {"path", "sha256", "schema_version", "policy_id"} and
                    prior_record.get("schema_version") == 3 and
                    prior_record.get("sha256") == V3_POLICY_SHA256 and
                    prior_record.get("policy_id") ==
                        "moving-nested-reference-frontier-v3-geometry-amendment",
                    f"{source}: v4 prior-policy binding changed")
            verified_prior = resolve_bound_record(
                prior_record, source, f"{source}: prior_policy")
            prior_policy = load_json(Path(verified_prior["path"]))
            validate_policy(prior_policy, Path(verified_prior["path"]))
            for key in ("required_metric_families", "coverage_thresholds",
                        "metric_thresholds", "derived_definitions"):
                require(policy.get(key) == prior_policy.get(key),
                        f"{source}: v4 {key} differs from immutable v3")
            for key in ("coordinate_frame", "detector_algorithm",
                        "continuity_algorithm", "moving_tracking_source",
                        "reference_tracking_source", "comparison_region",
                        "relaxation_exclusion_cells", "trusted_region_cell_count",
                        "trusted_region_shape_cells", "horizontal_grid_spacing_m",
                        "grid_phase_origin_m", "physics_signature_sha256",
                        "sample_start_s", "sample_interval_s", "duration_s"):
                require(expected.get(key) ==
                            prior_policy["expected_contract"].get(key),
                        f"{source}: v4 expected_contract.{key} differs from immutable v3")
            for key, identity_key in (("geometry_plan", "plan_id"),
                                      ("qualification_aperture", "aperture_id")):
                current_record = expected[key]
                prior_expected = prior_policy["expected_contract"][key]
                for record_key in ("sha256", "schema_version", identity_key):
                    require(current_record.get(record_key) ==
                                prior_expected.get(record_key),
                            f"{source}: v4 expected_contract.{key} differs from immutable v3")
            failure_record = object_value(
                policy.get("failure_evidence"), f"{source}: failure_evidence")
            require(set(failure_record) ==
                        {"path", "sha256", "schema_version", "reason"} and
                    failure_record.get("schema_version") == 1 and
                    failure_record.get("sha256") ==
                        PROJECTED_CONTAINMENT_FAILURE_SHA256 and
                    failure_record.get("reason") ==
                        "relaxation_zone_entry_after_confirmation_pending",
                    f"{source}: v4 failure-evidence binding changed")
            verified_failure = resolve_bound_record(
                failure_record, source, f"{source}: failure_evidence")
            failure = load_json(Path(verified_failure["path"]))
            require(failure.get("schema") ==
                        "nimbus.qualification.moving_nested_reference_frontier.movement_failure" and
                    failure.get("schema_version") == 1 and
                    failure.get("status") == "invalid" and
                    failure.get("qualification_disposition") == "incomplete" and
                    failure.get("qualification_eligible") is False and
                    failure.get("reason") ==
                        "relaxation_zone_entry_after_confirmation_pending" and
                    failure.get("comparison_performed") is False and
                    failure.get("candidate_metric_values_inspected") is False,
                    f"{source}: bound projected-containment failure identity changed")
            amendment = object_value(
                ratification.get("projected_containment_amendment"),
                f"{source}: projected_containment_amendment")
            require(amendment == {
                        "reason":
                            "fail_closed_relaxation_zone_entry_after_confirmation_pending",
                        "algorithm": "qualified_envelope_hysteresis_v6",
                        "ordinary_confirmation_samples": 2,
                        "eligibility":
                            "authoritative selected track with finite signed per-face motion and a current inner-margin breach",
                        "per_face_projection":
                            "projected_clearance_m = current_clearance_m - max(0,outward_signed_face_velocity_ms) * face_motion_interval_s",
                        "immediate_commit_predicate":
                            "any projected_clearance_m < exact_axis_relaxation_width_m",
                        "commit_preconditions":
                            "complete two-dimensional release box and parent containment are feasible before mutation",
                        "exact_relaxation_boundary_is_trusted": True,
                        "centroid_motion_substitution_allowed": False,
                        "ambiguity_topology_or_missing_face_motion_bypass_allowed": False,
                        "current_relaxation_entry_recoverable": False,
                        "numerical_thresholds_changed": False,
                        "geometry_or_aperture_changed": False,
                        "failed_candidate_disposition": "invalid_not_comparable"},
                    f"{source}: projected-containment amendment is incomplete")
            policy["_verified_threshold_policy"] = verified_threshold_policy
            policy["_verified_prior_policy"] = verified_prior
            policy["_verified_failure_evidence"] = verified_failure
        elif schema_version == 5:
            prior_record = object_value(
                policy.get("prior_policy"), f"{source}: prior_policy")
            require(set(prior_record) ==
                        {"path", "sha256", "schema_version", "policy_id"} and
                    prior_record.get("schema_version") == 4 and
                    prior_record.get("sha256") == V4_POLICY_SHA256 and
                    prior_record.get("policy_id") ==
                        "moving-nested-reference-frontier-v4-projected-containment-amendment",
                    f"{source}: v5 prior-policy binding changed")
            verified_prior = resolve_bound_record(
                prior_record, source, f"{source}: prior_policy")
            prior_policy = load_json(Path(verified_prior["path"]))
            validate_policy(prior_policy, Path(verified_prior["path"]))
            for key in ("required_metric_families", "coverage_thresholds",
                        "metric_thresholds", "derived_definitions"):
                require(policy.get(key) == prior_policy.get(key),
                        f"{source}: v5 {key} differs from immutable v4")
            for key in ("coordinate_frame", "detector_algorithm",
                        "continuity_algorithm", "moving_tracking_source",
                        "reference_tracking_source", "comparison_region",
                        "relaxation_exclusion_cells", "trusted_region_cell_count",
                        "trusted_region_shape_cells", "horizontal_grid_spacing_m",
                        "grid_phase_origin_m", "physics_signature_sha256",
                        "sample_start_s", "sample_interval_s", "duration_s"):
                require(expected.get(key) ==
                            prior_policy["expected_contract"].get(key),
                        f"{source}: v5 expected_contract.{key} differs from immutable v4")
            for key, identity_key in (("geometry_plan", "plan_id"),
                                      ("qualification_aperture", "aperture_id")):
                current_record = expected[key]
                prior_expected = prior_policy["expected_contract"][key]
                for record_key in ("sha256", "schema_version", identity_key):
                    require(current_record.get(record_key) ==
                                prior_expected.get(record_key),
                            f"{source}: v5 expected_contract.{key} differs from immutable v4")
            failure_record = object_value(
                policy.get("failure_evidence"), f"{source}: failure_evidence")
            require(set(failure_record) ==
                        {"path", "sha256", "schema_version", "reason"} and
                    failure_record.get("schema_version") == 1 and
                    failure_record.get("sha256") ==
                        POST_COMMIT_CONTAINMENT_FAILURE_SHA256 and
                    failure_record.get("reason") ==
                        "relaxation_zone_entry_after_minimum_projected_risk_shift",
                    f"{source}: v5 failure-evidence binding changed")
            verified_failure = resolve_bound_record(
                failure_record, source, f"{source}: failure_evidence")
            failure = load_json(Path(verified_failure["path"]))
            require(failure.get("schema") ==
                        "nimbus.qualification.moving_nested_reference_frontier.movement_failure" and
                    failure.get("schema_version") == 1 and
                    failure.get("status") == "invalid" and
                    failure.get("qualification_disposition") == "incomplete" and
                    failure.get("qualification_eligible") is False and
                    failure.get("reason") ==
                        "relaxation_zone_entry_after_minimum_projected_risk_shift" and
                    failure.get("comparison_performed") is False and
                    failure.get("candidate_metric_values_inspected") is False,
                    f"{source}: bound post-commit containment failure identity changed")
            amendment = object_value(
                ratification.get("maximum_safe_projected_shift_amendment"),
                f"{source}: maximum_safe_projected_shift_amendment")
            require(amendment == {
                        "reason":
                            "fail_closed_relaxation_entry_after_minimum_current_release_shift",
                        "algorithm": "qualified_envelope_hysteresis_v7",
                        "projected_risk_axis_shift":
                            "maximum configured whole-parent-cell shift that preserves the opposite release face",
                        "ordinary_confirmed_axis_shift":
                            "minimum whole-parent-cell shift that restores the release box",
                        "orthogonal_nonprojected_axis_shift":
                            "minimum whole-parent-cell shift that restores the release box",
                        "maximum_shift_cap_parent_cells": 14,
                        "complete_release_box_required": True,
                        "parent_containment_required": True,
                        "ambiguity_topology_or_missing_face_motion_bypass_allowed": False,
                        "current_relaxation_entry_recoverable": False,
                        "numerical_thresholds_changed": False,
                        "geometry_or_aperture_changed": False,
                        "failed_candidate_disposition": "invalid_not_comparable"},
                    f"{source}: maximum-safe projected-shift amendment is incomplete")
            policy["_verified_threshold_policy"] = verified_threshold_policy
            policy["_verified_prior_policy"] = verified_prior
            policy["_verified_failure_evidence"] = verified_failure
    return policy


def expected_times(contract: dict[str, Any]) -> list[float]:
    start = contract["sample_start_s"]
    interval = contract["sample_interval_s"]
    duration = contract["duration_s"]
    span = (duration - start) / interval
    count = int(round(span)) + 1
    require(count > 0 and math.isclose(start + (count - 1) * interval, duration,
                                        rel_tol=0.0, abs_tol=1.0e-9),
            "duration is not exactly reachable from sample start and interval")
    return [start + index * interval for index in range(count)]


def percentile(values: list[float], quantile: float) -> float:
    require(values, "cannot calculate a percentile of an empty series")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def rmse(values: list[float]) -> float:
    require(values, "cannot calculate RMSE of an empty series")
    return math.sqrt(sum(value * value for value in values) / len(values))


def normalized_rmse(candidate: list[float], reference: list[float], floor: float) -> float:
    require(len(candidate) == len(reference) and candidate,
            "normalized RMSE requires paired nonempty series")
    normalized = [
        (left - right) / max(abs(right), floor)
        for left, right in zip(candidate, reference)
    ]
    return rmse(normalized)


def maximum_consecutive_false(values: list[bool]) -> int:
    maximum = 0
    current = 0
    for value in values:
        current = 0 if value else current + 1
        maximum = max(maximum, current)
    return maximum


def add_check(checks: list[dict[str, Any]], name: str, value: float,
              threshold: float, comparator: str = "<=") -> None:
    if comparator == "<=":
        passed = value <= threshold
    elif comparator == ">=":
        passed = value >= threshold
    else:
        raise AssertionError(f"unsupported comparator {comparator}")
    checks.append({
        "name": name,
        "value": value,
        "threshold": threshold,
        "comparator": comparator,
        "passed": passed,
    })


def paired_series(samples: list[tuple[dict[str, Any], dict[str, Any]]], key: str) -> tuple[list[float], list[float]]:
    return (
        [finite_number(moving[key], f"moving {key}") for moving, _ in samples],
        [finite_number(reference[key], f"reference {key}") for _, reference in samples],
    )


def paired_nonnegative_series(
    samples: list[tuple[dict[str, Any], dict[str, Any]]], key: str
) -> tuple[list[float], list[float]]:
    return (
        [nonnegative_number(moving.get(key), f"moving {key}") for moving, _ in samples],
        [nonnegative_number(reference.get(key), f"reference {key}") for _, reference in samples],
    )


def derived_record(sample: dict[str, Any], family: str, definition: str,
                   label: str) -> dict[str, Any] | None:
    value = sample.get(family)
    if value is None:
        return None
    record = object_value(value, f"{label}.{family}")
    require(record.get("definition") == definition,
            f"{label}.{family} has the wrong definition")
    return record


def compare_documents(moving_document: dict[str, Any], moving_path: Path,
                      reference_document: dict[str, Any], reference_path: Path,
                      policy_document: dict[str, Any], policy_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "status": "incompatible",
        "policy": {"path": str(policy_path.resolve()), "sha256": sha256(policy_path)},
        "inputs": {},
        "contract_errors": [],
        "incomplete_reasons": [],
        "checks": [],
        "metrics": {},
    }
    try:
        policy = validate_policy(policy_document, policy_path)
        moving = validate_series(moving_document, moving_path, "moving_child")
        reference = validate_series(reference_document, reference_path, "oversized_reference")
        report["inputs"] = {
            "moving": {key: moving[key] for key in ("input_path", "input_sha256", "artifacts")},
            "reference": {key: reference[key] for key in ("input_path", "input_sha256", "artifacts")},
        }

        moving_contract = moving["contract"]
        reference_contract = reference["contract"]
        expected = policy["expected_contract"]
        if policy["schema_version"] in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            minimum_series_version = policy["ratification"][
                "minimum_input_series_schema_version"]
            require(moving["schema_version"] >= minimum_series_version and
                    reference["schema_version"] >= minimum_series_version,
                    "ratified policy requires hash-bound v2 input series")
            for name, series in (("moving", moving), ("reference", reference)):
                source_manifest = series["source_manifest"]
                require(source_manifest.get("schema_version") == 2,
                        f"ratified policy requires a v2 {name} source manifest")
                bound_policy = source_manifest["_verified_policy"]
                policy_record = source_manifest["policy"]
                require(bound_policy["path"] == str(policy_path.resolve()) and
                        bound_policy["sha256"] == report["policy"]["sha256"] and
                        policy_record.get("policy_id") == policy.get("policy_id") and
                        policy_record.get("policy_status") ==
                            policy.get("policy_status"),
                        f"{name} source is not bound to the active ratified policy")
                require(policy_record.get("schema_version") ==
                            policy["schema_version"],
                        f"{name} source policy schema differs from the active policy")
        if policy["schema_version"] in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            for name, series in (("moving", moving), ("reference", reference)):
                source_manifest = series["source_manifest"]
                for key, verified_key in (
                    ("geometry_plan", "_verified_geometry_plan"),
                    ("qualification_aperture",
                     "_verified_qualification_aperture"),
                ):
                    expected_record = expected[key]
                    verified_record = source_manifest[verified_key]
                    require(verified_record["sha256"] ==
                                expected_record["sha256"] and
                            verified_record["path"] ==
                                policy["_verified_" + key]["path"],
                            f"{name} source is not bound to the active {key}")
        require(moving["schema_version"] == reference["schema_version"],
                "moving/reference series schema versions differ")
        if moving["schema_version"] == 2:
            require(moving["artifacts"]["binary"]["sha256"] ==
                    reference["artifacts"]["binary"]["sha256"],
                    "moving/reference binaries differ")
            reference_region_source = resolve_bound_record(
                reference["source_manifest"].get("region_series"),
                Path(reference["artifacts"]["source_manifest"]["path"]),
                "reference region series")
            require(reference_region_source["sha256"] == moving["input_sha256"] and
                    reference_region_source["path"] == moving["input_path"],
                    "reference trusted regions are not bound to the moving input series")
        common_keys = [
            "coordinate_frame", "detector_algorithm", "continuity_algorithm",
            "horizontal_grid_spacing_m", "comparison_region",
            "relaxation_exclusion_cells", "duration_s", "sample_start_s",
            "sample_interval_s",
        ]
        if policy["schema_version"] in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            common_keys += ["geometry_plan", "qualification_aperture"]
        for key in common_keys:
            require(moving_contract.get(key) == reference_contract.get(key),
                    f"moving/reference contract mismatch for {key}")
            require(moving_contract.get(key) == expected.get(key),
                    f"input contract does not match policy for {key}")
        require(moving_contract["physics_signature_sha256"] ==
                reference_contract["physics_signature_sha256"],
                "moving/reference physics signatures differ")
        require(moving_contract["grid_phase_origin_m"] ==
                reference_contract["grid_phase_origin_m"],
                "moving/reference grids are not Earth-phase aligned")
        require(moving_contract["tracking_source"] == expected.get("moving_tracking_source"),
                "moving tracking source does not match policy")
        require(reference_contract["tracking_source"] == expected.get("reference_tracking_source"),
                "reference tracking source does not match policy")
        require(moving_contract.get("movement_algorithm") ==
                expected.get("moving_movement_algorithm"),
                "moving movement algorithm does not match policy")
        if policy["schema_version"] in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            require(moving_contract["physics_signature_sha256"] ==
                    expected["physics_signature_sha256"],
                    "input physics signature does not match ratified policy")
            require(moving_contract["grid_phase_origin_m"] ==
                    expected["grid_phase_origin_m"],
                    "input Earth-grid phase does not match ratified policy")

        times = expected_times(moving_contract)
        moving_by_time = {sample["simulation_time_s"]: sample for sample in moving["samples"]}
        reference_by_time = {sample["simulation_time_s"]: sample for sample in reference["samples"]}
        extra_moving = sorted(set(moving_by_time) - set(times))
        extra_reference = sorted(set(reference_by_time) - set(times))
        require(not extra_moving and not extra_reference,
                "input contains samples outside the declared cadence")
        missing_moving = [time for time in times if time not in moving_by_time]
        missing_reference = [time for time in times if time not in reference_by_time]
        if missing_moving:
            report["incomplete_reasons"].append(
                f"moving series is missing {len(missing_moving)} declared times")
        if missing_reference:
            report["incomplete_reasons"].append(
                f"reference series is missing {len(missing_reference)} declared times")
        common_times = [time for time in times
                        if time in moving_by_time and time in reference_by_time]
        require(common_times, "moving/reference series have no common declared times")
        if moving["schema_version"] == 2:
            for time in common_times:
                moving_region = moving_by_time[time]["region"]
                reference_region = reference_by_time[time]["region"]
                region_keys = [
                    "id", "earth_bounds_m", "horizontal_grid_spacing_m",
                    "grid_phase_origin_m", "cell_count"]
                if policy["schema_version"] == 2:
                    region_keys.append("aperture_id")
                for key in region_keys:
                    require(moving_region.get(key) == reference_region.get(key),
                            f"moving/reference trusted regions differ at {time}s for {key}")
                require(moving_region["horizontal_grid_spacing_m"] ==
                        moving_contract["horizontal_grid_spacing_m"] and
                        moving_region["grid_phase_origin_m"] ==
                        moving_contract["grid_phase_origin_m"],
                        f"sample trusted-region grid contract differs at {time}s")
                if policy["schema_version"] == 2:
                    source_range = moving_region["source_index_range"]
                    trusted_shape = {
                        "x": source_range["upper"]["i"] -
                             source_range["lower"]["i"] + 1,
                        "y": source_range["upper"]["j"] -
                             source_range["lower"]["j"] + 1,
                    }
                    require(moving_region["cell_count"] ==
                            expected["trusted_region_cell_count"] and
                            trusted_shape == expected["trusted_region_shape_cells"],
                            f"trusted-region cell count differs from ratified policy at {time}s")

        reference_selected_times = [
            time for time in common_times if reference_by_time[time]["selected"]
        ]
        require(reference_selected_times,
                "reference never supplies a selected storm")
        mature_start = reference_selected_times[0]
        mature_times = [time for time in common_times if time >= mature_start]
        paired_flags = [
            moving_by_time[time]["selected"] and reference_by_time[time]["selected"]
            for time in mature_times
        ]
        paired_times = [time for time, paired in zip(mature_times, paired_flags) if paired]
        paired_fraction = len(paired_times) / len(mature_times)
        longest_gap = maximum_consecutive_false(paired_flags)
        coverage = {
            "declared_sample_count": len(times),
            "common_sample_count": len(common_times),
            "mature_start_s": mature_start,
            "mature_sample_count": len(mature_times),
            "paired_selected_samples": len(paired_times),
            "paired_selected_fraction": paired_fraction,
            "maximum_consecutive_unpaired_samples": longest_gap,
            "final_sample_paired": bool(paired_flags and paired_flags[-1]),
            "missing_moving_times_s": missing_moving,
            "missing_reference_times_s": missing_reference,
        }
        report["coverage"] = coverage
        coverage_policy = policy["coverage_thresholds"]
        add_check(report["checks"], "coverage.paired_selected_fraction",
                  paired_fraction,
                  finite_number(coverage_policy.get("minimum_paired_selected_fraction"),
                                "minimum paired fraction"), ">=")
        add_check(report["checks"], "coverage.maximum_consecutive_unpaired_samples",
                  float(longest_gap),
                  finite_number(coverage_policy.get("maximum_consecutive_unpaired_samples"),
                                "maximum unpaired samples"))
        require(isinstance(coverage_policy.get("require_final_paired_sample"), bool),
                "coverage require_final_paired_sample must be boolean")
        if coverage_policy["require_final_paired_sample"]:
            add_check(report["checks"], "coverage.final_sample_paired",
                      1.0 if coverage["final_sample_paired"] else 0.0, 1.0, ">=")

        if not paired_times:
            report["incomplete_reasons"].append("path and intensity have no paired selected samples")
        else:
            pairs = [(moving_by_time[time], reference_by_time[time]) for time in paired_times]
            path_complete = all(
                moving_sample.get("core_centroid_m") is not None and
                reference_sample.get("core_centroid_m") is not None
                for moving_sample, reference_sample in pairs
            )
            if not path_complete:
                report["incomplete_reasons"].append(
                    "path inputs are incomplete at paired selected times")
            else:
                path_errors = [
                    math.hypot(
                        moving_sample["core_centroid_m"][0] - reference_sample["core_centroid_m"][0],
                        moving_sample["core_centroid_m"][1] - reference_sample["core_centroid_m"][1],
                    )
                    for moving_sample, reference_sample in pairs
                ]
                path_metrics = {
                    "sample_count": len(path_errors),
                    "rmse_m": rmse(path_errors),
                    "p95_m": percentile(path_errors, 0.95),
                    "maximum_m": max(path_errors),
                    "final_m": path_errors[-1],
                }
                report["metrics"]["path"] = path_metrics
                for key, value in path_metrics.items():
                    if key != "sample_count":
                        add_check(report["checks"], f"path.{key}", value,
                                  finite_number(policy["metric_thresholds"]["path"].get(key),
                                                f"path threshold {key}"))

            moving_intensity_samples = [
                (time, moving_by_time[time]) for time in mature_times
                if moving_by_time[time]["selected"]]
            reference_intensity_samples = [
                (time, reference_by_time[time]) for time in mature_times
                if reference_by_time[time]["selected"]]
            intensity_complete = all(
                sample.get("maximum_updraft_ms") is not None and
                sample.get("maximum_absolute_uh_m2s2") is not None
                for _, sample in moving_intensity_samples +
                                 reference_intensity_samples
            )
            if not intensity_complete:
                report["incomplete_reasons"].append(
                    "intensity inputs are incomplete at paired selected times")
            else:
                moving_w, reference_w = paired_series(pairs, "maximum_updraft_ms")
                moving_uh, reference_uh = paired_series(pairs, "maximum_absolute_uh_m2s2")
                intensity_policy = policy["metric_thresholds"]["intensity"]
                w_floor = nonnegative_number(intensity_policy.get("updraft_scale_floor_ms"),
                                             "updraft scale floor")
                uh_floor = nonnegative_number(intensity_policy.get("uh_scale_floor_m2s2"),
                                              "UH scale floor")

                def intensity_metrics(
                    paired_candidate: list[float],
                    paired_reference: list[float],
                    candidate_samples: list[tuple[float, dict[str, Any]]],
                    reference_samples: list[tuple[float, dict[str, Any]]],
                    key: str,
                    floor: float,
                ) -> dict[str, float]:
                    candidate_values = [
                        finite_number(sample[key], f"moving {key}")
                        for _, sample in candidate_samples]
                    reference_values = [
                        finite_number(sample[key], f"reference {key}")
                        for _, sample in reference_samples]
                    candidate_peak = max(candidate_values)
                    reference_peak = max(reference_values)
                    candidate_time = candidate_samples[
                        candidate_values.index(candidate_peak)][0]
                    reference_time = reference_samples[
                        reference_values.index(reference_peak)][0]
                    return {
                        "normalized_rmse": normalized_rmse(
                            paired_candidate, paired_reference, floor),
                        "peak_relative_error": abs(candidate_peak - reference_peak) /
                                               max(abs(reference_peak), floor),
                        "peak_time_error_s": abs(candidate_time - reference_time),
                    }

                w_metrics = intensity_metrics(
                    moving_w, reference_w, moving_intensity_samples,
                    reference_intensity_samples, "maximum_updraft_ms", w_floor)
                uh_metrics = intensity_metrics(
                    moving_uh, reference_uh, moving_intensity_samples,
                    reference_intensity_samples,
                    "maximum_absolute_uh_m2s2", uh_floor)
                report["metrics"]["intensity"] = {
                    "sample_count": len(pairs),
                    "maximum_updraft": w_metrics,
                    "maximum_absolute_uh": uh_metrics,
                }
                for key, value in w_metrics.items():
                    policy_key = f"maximum_updraft_{key}"
                    add_check(report["checks"], f"intensity.{policy_key}", value,
                              finite_number(intensity_policy.get(policy_key),
                                            f"intensity threshold {policy_key}"))
                for key, value in uh_metrics.items():
                    policy_key = f"maximum_absolute_uh_{key}"
                    add_check(report["checks"], f"intensity.{policy_key}", value,
                              finite_number(intensity_policy.get(policy_key),
                                            f"intensity threshold {policy_key}"))

        definitions = policy["derived_definitions"]
        complete_derived: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for family in ("cold_pool", "precipitation"):
            family_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            missing: list[float] = []
            for time in common_times:
                moving_record = derived_record(
                    moving_by_time[time], family, definitions[family]["id"], f"moving sample {time}")
                reference_record = derived_record(
                    reference_by_time[time], family, definitions[family]["id"], f"reference sample {time}")
                if moving_record is None or reference_record is None:
                    missing.append(time)
                else:
                    family_pairs.append((moving_record, reference_record))
            if missing or len(common_times) != len(times):
                report["incomplete_reasons"].append(
                    f"{family} inputs are incomplete at {len(missing) + len(times) - len(common_times)} times")
            else:
                complete_derived[family] = family_pairs

        if "cold_pool" in complete_derived:
            pairs = complete_derived["cold_pool"]
            policy_values = policy["metric_thresholds"]["cold_pool"]
            moving_area, reference_area = paired_nonnegative_series(pairs, "area_m2")
            moving_deficit, reference_deficit = paired_nonnegative_series(
                pairs, "integrated_deficit_k_m2")
            moving_minimum, reference_minimum = paired_series(
                pairs, "minimum_theta_rho_perturbation_k")
            active_centroid_errors: list[float] = []
            for index, (moving_record, reference_record) in enumerate(pairs):
                moving_area_value = nonnegative_number(moving_record.get("area_m2"), "moving cold-pool area")
                reference_area_value = nonnegative_number(reference_record.get("area_m2"), "reference cold-pool area")
                if moving_area_value > 0.0 and reference_area_value > 0.0:
                    mx, my = coordinate(moving_record.get("centroid_m"), "moving cold-pool centroid")
                    rx, ry = coordinate(reference_record.get("centroid_m"), "reference cold-pool centroid")
                    active_centroid_errors.append(math.hypot(mx - rx, my - ry))
            minimum_active_value = policy_values.get("minimum_active_samples")
            require(isinstance(minimum_active_value, int) and
                    not isinstance(minimum_active_value, bool) and
                    minimum_active_value >= 0,
                    "minimum active cold-pool samples must be a nonnegative integer")
            minimum_active = minimum_active_value
            if len(active_centroid_errors) < minimum_active:
                report["incomplete_reasons"].append(
                    "cold_pool lacks the required paired active centroid samples")
            else:
                cold_metrics = {
                    "sample_count": len(pairs),
                    "active_centroid_samples": len(active_centroid_errors),
                    "area_normalized_rmse": normalized_rmse(
                        moving_area, reference_area,
                        nonnegative_number(policy_values.get("area_scale_floor_m2"), "cold-pool area floor")),
                    "integrated_deficit_normalized_rmse": normalized_rmse(
                        moving_deficit, reference_deficit,
                        nonnegative_number(policy_values.get("integrated_deficit_scale_floor_k_m2"),
                                           "cold-pool deficit floor")),
                    "minimum_theta_rho_perturbation_rmse_k": rmse([
                        left - right for left, right in zip(moving_minimum, reference_minimum)
                    ]),
                    "centroid_rmse_m": rmse(active_centroid_errors),
                }
                report["metrics"]["cold_pool"] = cold_metrics
                for key, value in cold_metrics.items():
                    if key not in {"sample_count", "active_centroid_samples"}:
                        add_check(report["checks"], f"cold_pool.{key}", value,
                                  finite_number(policy_values.get(key), f"cold-pool threshold {key}"))

        if "precipitation" in complete_derived:
            pairs = complete_derived["precipitation"]
            policy_values = policy["metric_thresholds"]["precipitation"]
            moving_mass, reference_mass = paired_nonnegative_series(
                pairs, "accumulated_mass_kg")
            moving_area, reference_area = paired_nonnegative_series(
                pairs, "footprint_area_m2")
            moving_p95, reference_p95 = paired_nonnegative_series(
                pairs, "p95_depth_mm")
            moving_maximum, reference_maximum = paired_nonnegative_series(
                pairs, "maximum_depth_mm")
            mass_floor = nonnegative_number(policy_values.get("mass_scale_floor_kg"),
                                            "precipitation mass floor")
            precip_metrics = {
                "sample_count": len(pairs),
                "final_accumulated_mass_relative_error":
                    abs(moving_mass[-1] - reference_mass[-1]) /
                    max(abs(reference_mass[-1]), mass_floor),
                "footprint_area_normalized_rmse": normalized_rmse(
                    moving_area, reference_area,
                    nonnegative_number(policy_values.get("area_scale_floor_m2"),
                                       "precipitation area floor")),
                "p95_depth_normalized_rmse": normalized_rmse(
                    moving_p95, reference_p95,
                    nonnegative_number(policy_values.get("depth_scale_floor_mm"),
                                       "precipitation depth floor")),
                "maximum_depth_normalized_rmse": normalized_rmse(
                    moving_maximum, reference_maximum,
                    nonnegative_number(policy_values.get("depth_scale_floor_mm"),
                                       "precipitation depth floor")),
            }
            report["metrics"]["precipitation"] = precip_metrics
            for key, value in precip_metrics.items():
                if key != "sample_count":
                    add_check(report["checks"], f"precipitation.{key}", value,
                              finite_number(policy_values.get(key),
                                            f"precipitation threshold {key}"))

        failed = [check for check in report["checks"] if not check["passed"]]
        if report["incomplete_reasons"]:
            report["status"] = "incomplete"
        elif failed:
            report["status"] = "fail"
        elif policy["policy_status"] != "ratified":
            report["status"] = "measurement_only"
        else:
            report["status"] = "pass"
        report["policy"].update({
            "policy_id": policy.get("policy_id"),
            "policy_status": policy["policy_status"],
            "schema_version": policy["schema_version"],
        })
        report["qualification_eligible"] = report["status"] == "pass"
        return report
    except ContractError as exc:
        report["contract_errors"].append(str(exc))
        report["qualification_eligible"] = False
        return report


def compare_paths(moving_path: Path, reference_path: Path, policy_path: Path) -> dict[str, Any]:
    return compare_documents(
        load_json(moving_path), moving_path,
        load_json(reference_path), reference_path,
        load_json(policy_path), policy_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", type=Path, required=True,
                        help="prepared moving-child reference-series JSON")
    parser.add_argument("--reference", type=Path, required=True,
                        help="prepared oversized-reference series JSON")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare_paths(
            args.moving.resolve(), args.reference.resolve(), args.policy.resolve())
    except ContractError as exc:
        report = {
            "schema": REPORT_SCHEMA,
            "schema_version": 1,
            "status": "incompatible",
            "qualification_eligible": False,
            "contract_errors": [str(exc)],
        }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return {"pass": 0, "measurement_only": 3, "incomplete": 3,
            "fail": 1, "incompatible": 2}.get(report["status"], 2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"[moving-nested-reference-frontier] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)

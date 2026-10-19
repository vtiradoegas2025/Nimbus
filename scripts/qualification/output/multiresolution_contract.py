"""Shared durable field-hierarchy validation for NPY, ZFP, and netCDF history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class MultiresolutionContractError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise MultiresolutionContractError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        _fail(f"{label}: expected {expected!r}, got {actual!r}")


def inspect_durable_hierarchies(
    run_dir: Path,
    *,
    expected_format: str,
    require_lod: bool,
) -> dict[str, Any]:
    validated_lods = 0
    hierarchy_fields = 0
    topology: list[tuple[str, tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]]] = []
    steps = sorted(path for path in run_dir.glob("step_*") if path.is_dir())
    if not steps:
        _fail(f"{run_dir} has no published steps")
    for step in steps:
        manifest_path = step / "manifest.json"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        if "nimbus-pending-field-brick-sha256:" in manifest_text:
            _fail(f"{step.name} completion manifest retains a pending hierarchy token")
        manifest = json.loads(manifest_text)
        _require_equal(
            manifest["data_product"]["storage_format"],
            expected_format,
            f"{step.name} storage format",
        )
        if any(".lod" in field for field in manifest["published_fields"]):
            _fail(f"{step.name} advertises LOD sidecars as public fields")
        catalog = {entry["field_id"]: entry for entry in manifest["fields"]}
        current_fields = [
            field for field in manifest["published_fields"]
            if "brick_hierarchy" in catalog[field]
        ]
        if not current_fields:
            _fail(f"{step.name} advertises no durable 3-D field hierarchy")
        for field in current_fields:
            hierarchy_fields += 1
            descriptor = catalog[field]["brick_hierarchy"]
            hierarchy_path = step / descriptor["path"]
            hierarchy_bytes = hierarchy_path.read_bytes()
            _require_equal(
                hashlib.sha256(hierarchy_bytes).hexdigest(),
                descriptor["content_sha256"],
                f"{step.name}/{field} hierarchy checksum",
            )
            hierarchy = json.loads(hierarchy_bytes)
            _require_equal(hierarchy["field"], field, f"{step.name}/{field} identity")
            _require_equal(
                hierarchy["publicationGeneration"],
                manifest["publication_generation"],
                f"{step.name}/{field} generation",
            )
            _require_equal(
                hierarchy["schemaVersion"],
                descriptor["schema_version"],
                f"{step.name}/{field} schema version",
            )
            if hierarchy["schemaVersion"] not in (2, 3):
                _fail(f"{step.name}/{field} hierarchy is not durable v2/v3")
            if hierarchy["sourceDtype"] != "float32" \
                    or hierarchy["sampleLocation"] != "cell_center":
                _fail(f"{step.name}/{field} hierarchy changed its source contract")
            levels = hierarchy["levels"]
            _require_equal(levels[0]["source"]["kind"], "primary_field",
                           f"{step.name}/{field} level-zero source")
            if expected_format == "npy_3d":
                primary = step / catalog[field]["file_pattern"]
            elif expected_format == "zfp":
                primary = step / catalog[field]["file_pattern"]
            elif expected_format == "netcdf":
                primary = step / "history.nc"
                _require_equal(catalog[field]["variable_name"], field,
                               f"{step.name}/{field} netCDF variable")
            else:
                _fail(f"unsupported maintained format {expected_format}")
            if not primary.is_file() or primary.stat().st_size == 0:
                _fail(f"{step.name}/{field} primary payload is absent")

            level_signature: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
            for index, level in enumerate(levels):
                _require_equal(level["level"], index, f"{step.name}/{field} LOD order")
                level_signature.append((
                    index,
                    tuple(level["shape"]),
                    tuple(level["stride"]),
                ))
                if not level["bricks"]:
                    _fail(f"{step.name}/{field} LOD {index} has no brick directory")
                for brick in level["bricks"]:
                    content = brick.get("contentSha256", "")
                    if len(content) != 64 or any(character not in "0123456789abcdef" for character in content):
                        _fail(f"{step.name}/{field} LOD {index} has invalid brick identity")
                if index == 0:
                    continue
                source = level["source"]
                _require_equal(source["kind"], "npy_lod",
                               f"{step.name}/{field} LOD {index} source")
                _require_equal(source["aggregation"], "recursive_cell_mean_2x2x2",
                               f"{step.name}/{field} LOD {index} aggregation")
                payload = step / source["path"]
                _require_equal(_sha256(payload), source["contentSha256"],
                               f"{step.name}/{field} LOD {index} checksum")
                if hierarchy["schemaVersion"] == 3:
                    geometry = source.get("geometry", {})
                    _require_equal(geometry.get("kind"), "physical_volume_centroids",
                                   f"{step.name}/{field} terrain LOD geometry")
                    for component in ("xM", "yM", "zM", "volumeM3"):
                        descriptor = geometry.get(component, {})
                        geometry_path = step / descriptor.get("path", "")
                        _require_equal(_sha256(geometry_path), descriptor.get("contentSha256"),
                                       f"{step.name}/{field} LOD {index} {component} checksum")
                validated_lods += 1
            topology.append((field, tuple(level_signature)))
    if require_lod and validated_lods == 0:
        _fail(f"{expected_format} hierarchy published no reduced LOD payload")
    return {
        "steps": len(steps),
        "hierarchyFields": hierarchy_fields,
        "validatedLods": validated_lods,
        "topology": tuple(topology),
    }

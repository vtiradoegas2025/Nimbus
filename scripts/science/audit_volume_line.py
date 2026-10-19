#!/usr/bin/env python3
"""Extract bounded, unmodified NPY sample lines for renderer/source comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def line_metrics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not 3 <= values.size <= 4096:
        raise ValueError("A line must contain 3–4096 scalar samples")
    finite = np.isfinite(values)
    result = {
        "samples": int(values.size),
        "nan_count": int(np.isnan(values).sum()),
        "positive_infinity_count": int(np.isposinf(values).sum()),
        "negative_infinity_count": int(np.isneginf(values).sum()),
        "minimum": float(values[finite].min()) if finite.any() else None,
        "maximum": float(values[finite].max()) if finite.any() else None,
        "range": None,
        "standard_deviation": None,
        "neighbor_difference_rms": None,
        "second_difference_rms": None,
    }
    if finite.all():
        result.update({
            "range": float(np.ptp(values)),
            "standard_deviation": float(np.std(values)),
            "neighbor_difference_rms": float(np.sqrt(np.mean(np.diff(values) ** 2))),
            "second_difference_rms": float(np.sqrt(np.mean(np.diff(values, n=2) ** 2))),
        })
    return result


def audit_line(path: Path, axis_order: str, start: tuple[int, int, int], axis: str, length: int) -> dict:
    if sorted(axis_order) != ["x", "y", "z"] or axis not in "xyz" or len(axis) != 1:
        raise ValueError("Declare a permutation of xyz and one line axis")
    if len(start) != 3 or any(type(v) is not int or v < 0 for v in start):
        raise ValueError("Start must contain three nonnegative integer xyz indices")
    if type(length) is not int or not 3 <= length <= 4096:
        raise ValueError("Line length must be an integer from 3 to 4096")
    path = path.resolve(strict=True)
    before = path.stat()
    manifest_path = path.parent / "manifest.json"
    manifest_bytes = manifest_path.read_bytes() if manifest_path.exists() else None
    manifest = json.loads(manifest_bytes) if manifest_bytes else {}
    grid = manifest.get("grid", {})
    if "axis_order" in grid and grid["axis_order"] != list(axis_order):
        raise ValueError("Declared array order differs from the saved manifest")
    if manifest.get("publication_state", "complete") != "complete":
        raise ValueError("Cannot audit an incomplete publication")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8):
        raise ValueError("Expected a three-dimensional float32 or float64 NPY field")
    shape_xyz = tuple(int(array.shape[axis_order.index(name)]) for name in "xyz")
    if "dimensions" in grid and tuple(grid["dimensions"].get(name) for name in "xyz") != shape_xyz:
        raise ValueError("Saved grid dimensions differ from the NPY field")
    end = list(start)
    end["xyz".index(axis)] += length - 1
    if any(e >= n for e, n in zip(end, shape_xyz)):
        raise ValueError("Requested line exceeds the saved field")
    valid = grid.get("valid_index_range")
    if valid is not None:
        lower, upper = valid.get("lower", []), valid.get("upper", [])
        if len(lower) != 3 or len(upper) != 3 or any(
            type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi < size
            for lo, hi, size in zip(lower, upper, shape_xyz)
        ):
            raise ValueError("Invalid published index bounds")
        if any(s < lo or e > hi for s, e, lo, hi in zip(start, end, lower, upper)):
            raise ValueError("Requested line includes samples outside the published valid domain")
    selection = [slice(start["xyz".index(name)], end["xyz".index(name)] + 1)
                 if name == axis else start["xyz".index(name)] for name in axis_order]
    values = np.array(array[tuple(selection)], dtype=np.float64, copy=True)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during the audit")
    if manifest_bytes is not None and manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Publication manifest changed during the audit")
    field = next((f for f in manifest.get("fields", []) if f.get("field_id") == path.stem), {})
    return {
        "source": str(path), "source_bytes": before.st_size, "source_mtime_ns": before.st_mtime_ns,
        "dtype": array.dtype.str, "array_axis_order": axis_order, "shape_xyz": shape_xyz,
        "selection": {"start_xyz": start, "end_xyz": end, "axis": axis, "length": length},
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes else None,
        "simulation_time_s": manifest.get("simulation_time_s"),
        "publication_generation": manifest.get("publication_generation"),
        "vertical_geometry": grid.get("vertical_geometry", {}).get("kind", "unspecified"),
        "units": field.get("units"),
        "line_float64_le_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
        "values": [float(v) if np.isfinite(v) else None for v in values],
        "metrics": line_metrics(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fields", nargs="+", type=Path)
    parser.add_argument("--axis-order", required=True, help="NPY array dimensions, e.g. xyz or zyx")
    parser.add_argument("--start", type=int, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--axis", choices=list("xyz"), default="x")
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [audit_line(p, args.axis_order, tuple(args.start), args.axis, args.length) for p in args.fields]
    text = json.dumps({
        "schema": "nimbus.volume-source-lines.v1",
        "interpretation": "Raw index-space samples. Differences are in field units, not physical derivatives. No physical/numerical classification or pass threshold is inferred. Invalid samples remain gaps; whole-line difference metrics are omitted when gaps exist.",
        "lines": rows,
    }, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()

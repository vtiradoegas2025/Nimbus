#!/usr/bin/env python3
"""Inspect bounded, unmodified NPY planes for source-space short-scale structure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _spectrum(values: np.ndarray, taper: bool) -> dict:
    centered = values - np.mean(values)
    if taper:
        centered = centered * np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    power = np.abs(np.fft.fft2(centered)) ** 2
    power[0, 0] = 0
    total = float(np.sum(power))
    if total == 0:
        return {"energy": 0, "peak_frequency_cycles_per_cell": None,
                "high_axis_frequency_energy_fraction": 0, "near_nyquist_energy_fraction": 0}
    fx, fy = np.meshgrid(np.fft.fftfreq(values.shape[0]), np.fft.fftfreq(values.shape[1]), indexing="ij")
    index = np.unravel_index(np.argmax(power), power.shape)
    axis_frequency = np.maximum(np.abs(fx), np.abs(fy))
    return {"energy": total, "peak_frequency_cycles_per_cell": [float(fx[index]), float(fy[index])],
            "high_axis_frequency_energy_fraction": float(np.sum(power[axis_frequency >= 0.25]) / total),
            "near_nyquist_energy_fraction": float(np.sum(power[axis_frequency >= 0.45]) / total)}


def plane_metrics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not all(8 <= count <= 256 for count in values.shape):
        raise ValueError("A plane must have 8–256 samples on each axis")
    finite = np.isfinite(values)
    result = {"samples": int(values.size), "finite_samples": int(np.count_nonzero(finite)),
              "minimum": float(np.min(values[finite])) if finite.any() else None,
              "maximum": float(np.max(values[finite])) if finite.any() else None,
              "standard_deviation": None, "neighbor_difference_rms": None,
              "second_difference_rms": None, "raw_spectrum": None,
              "detrended_hann_spectrum": None, "checkerboard_projection_rms": None}
    if not finite.all():
        return result
    result["standard_deviation"] = float(np.std(values))
    result["neighbor_difference_rms"] = [float(np.sqrt(np.mean(np.diff(values, axis=axis) ** 2))) for axis in (0, 1)]
    result["second_difference_rms"] = [float(np.sqrt(np.mean(np.diff(values, n=2, axis=axis) ** 2))) for axis in (0, 1)]
    x, y = np.indices(values.shape, dtype=np.float64)
    design = np.column_stack([np.ones(values.size), x.ravel(), y.ravel()])
    coefficients, *_ = np.linalg.lstsq(design, values.ravel(), rcond=None)
    departures = values - (design @ coefficients).reshape(values.shape)
    if np.max(np.abs(departures)) <= np.finfo(np.float64).eps * max(1, float(np.max(np.abs(values)))) * 32:
        departures = np.zeros_like(departures)
    checkerboard = np.where((x.astype(int) + y.astype(int)) % 2, -1, 1)
    result["checkerboard_projection_rms"] = float(abs(np.mean(departures * checkerboard)))
    result["raw_spectrum"] = _spectrum(values, False)
    result["detrended_hann_spectrum"] = _spectrum(departures, True)
    return result


def audit_plane(path: Path, axis_order: str, start: tuple[int, int, int], axes: str, shape: tuple[int, int]) -> dict:
    if sorted(axis_order) != ["x", "y", "z"] or len(axes) != 2 or len(set(axes)) != 2 or any(axis not in "xyz" for axis in axes):
        raise ValueError("Declare one xyz array permutation and two distinct plane axes")
    if len(start) != 3 or any(type(value) is not int or value < 0 for value in start):
        raise ValueError("Start must be three nonnegative integer xyz indices")
    if len(shape) != 2 or any(type(value) is not int or not 8 <= value <= 256 for value in shape):
        raise ValueError("Plane extent must be two integers from 8 to 256")
    path = path.resolve(strict=True)
    before = path.stat()
    manifest_path = path.parent / "manifest.json"
    manifest_bytes = manifest_path.read_bytes() if manifest_path.exists() else None
    manifest = json.loads(manifest_bytes) if manifest_bytes else {}
    if manifest.get("publication_state", "complete") != "complete":
        raise ValueError("Cannot inspect an incomplete publication")
    grid = manifest.get("grid", {})
    if "axis_order" in grid and grid["axis_order"] != list(axis_order):
        raise ValueError("Array axis declaration disagrees with the published layout")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8):
        raise ValueError("Expected a rank-3 float32 or float64 NPY field")
    shape_xyz = tuple(int(array.shape[axis_order.index(axis)]) for axis in "xyz")
    dimensions = grid.get("dimensions")
    if dimensions and tuple(dimensions.get(axis) for axis in "xyz") != shape_xyz:
        raise ValueError("Source dimensions disagree with the manifest")
    end = list(start)
    for axis, size in zip(axes, shape):
        end["xyz".index(axis)] += size - 1
    if any(value >= size for value, size in zip(end, shape_xyz)):
        raise ValueError("Plane exceeds the saved source dimensions")
    valid = grid.get("valid_index_range")
    if valid:
        lower, upper = valid.get("lower", []), valid.get("upper", [])
        if len(lower) != 3 or len(upper) != 3 or any(type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi < size
                                                   for lo, hi, size in zip(lower, upper, shape_xyz)):
            raise ValueError("Invalid published physical index bounds")
        if any(first < lo or last > hi for first, last, lo, hi in zip(start, end, lower, upper)):
            raise ValueError("Plane includes samples outside the published physical domain")
    selections = [slice(start["xyz".index(axis)], end["xyz".index(axis)] + 1) if axis in axes
                  else start["xyz".index(axis)] for axis in axis_order]
    values = np.array(array[tuple(selections)], dtype=np.float64, copy=True)
    remaining_axes = [axis for axis in axis_order if axis in axes]
    values = values.transpose(tuple(remaining_axes.index(axis) for axis in axes))
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while reading the bounded plane")
    if manifest_bytes is not None and manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Publication manifest changed during inspection")
    field = next((field for field in manifest.get("fields", []) if field.get("field_id") == path.stem), {})
    return {"source": str(path), "source_bytes": before.st_size, "source_mtime_ns": before.st_mtime_ns,
            "dtype": array.dtype.str, "array_axis_order": axis_order, "shape_xyz": shape_xyz,
            "selection": {"start_xyz": start, "end_xyz": end, "axes": axes, "shape": shape},
            "simulation_time_s": manifest.get("simulation_time_s"), "units": field.get("units"),
            "published_valid_index_range": valid,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes else None,
            "plane_float64_le_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
            "values": [[float(value) if np.isfinite(value) else None for value in row] for row in values],
            "metrics": plane_metrics(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fields", nargs="+", type=Path)
    parser.add_argument("--axis-order", required=True)
    parser.add_argument("--start", nargs=3, type=int, required=True)
    parser.add_argument("--axes", choices=["xy", "yx", "xz", "zx", "yz", "zy"], default="xy")
    parser.add_argument("--shape", nargs=2, type=int, default=(64, 64))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "nimbus.volume-source-planes.v1",
              "interpretation": "Unmodified index-space values. Fourier frequencies are cycles per saved cell; >=0.25 means at most four cells on an axis, >=0.45 is near its two-cell sampling limit. The diagnostic Hann spectrum removes an affine plane and tapers crop boundaries; source values remain unchanged. These measures do not classify physical versus numerical structure or define a scientific pass threshold. Spectra are omitted when any sample is invalid.",
              "planes": [audit_plane(path, args.axis_order, tuple(args.start), args.axes, tuple(args.shape)) for path in args.fields]}
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()

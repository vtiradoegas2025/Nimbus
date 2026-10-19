#!/usr/bin/env python3
"""Validate a Cartesian C-grid or internal A-grid calm-column hold.

The checker intentionally uses only Python's standard library.  It validates
the atomic publication contract and strict-validation summary, reads the
initial/final float32 NPY fields, and evaluates balance, EOS, and physical-cell
inventory metrics. Here ``rho`` is the prognostic carrier inventory,
interpreted on the strongly evidenced dry-carrier basis while SCI-024 remains
open, and ``rho*qv`` is the vapor inventory for this microphysics-off fixture.
Manifest-declared scalar ghosts are excluded from every physical metric.
Headless NPY output has already interpolated C-grid face velocities to scalar
centers, so the checker consumes ``u``, ``v``, and ``w`` directly. Wind bounds
use physical cells and subtract the declared frame velocity; closure planes
may legitimately contain Earth-fixed wall values.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import struct
from typing import Iterator, Mapping, Sequence


GRAVITY_MS2 = 9.81
GAS_CONSTANT_DRY_AIR_JKGK = 287.0
SPECIFIC_HEAT_CP_JKGK = 1004.0
REFERENCE_PRESSURE_PA = 100000.0
LATENT_HEAT_VAPORIZATION_JKG = 2.5e6
OUTPUT_MANIFEST_VERSION = 7
REQUIRED_FIELDS = ("u", "v", "w", "rho", "p", "theta", "qv")
STEP_DIR_PATTERN = re.compile(r"step_[0-9]{6}$")


class QualificationError(ValueError):
    """A malformed artifact or failed scientific threshold."""


@dataclass(frozen=True)
class Float32Array:
    shape: tuple[int, int, int]
    values: tuple[float, ...]

    def at(self, i: int, j: int, k: int) -> float:
        _, ny, nz = self.shape
        return self.values[(i * ny + j) * nz + k]


@dataclass(frozen=True)
class Thresholds:
    max_wind_ms: float = 1.0e-3
    max_pressure_drift_pa: float = 2.0e-1
    max_density_drift_kgm3: float = 2.0e-5
    max_theta_drift_k: float = 2.0e-3
    max_hydrostatic_relative_residual: float = 1.0e-3
    max_eos_relative_residual: float = 5.0e-6
    max_inventory_relative_drift: float = 2.0e-6
    max_total_energy_relative_drift: float = 2.0e-6


@dataclass(frozen=True)
class Frame:
    path: Path
    time_s: float
    shape: tuple[int, int, int]
    dx_m: float
    dy_m: float
    dz_m: float
    published_fields: frozenset[str]
    valid_lower: tuple[int, int, int]
    valid_upper: tuple[int, int, int]
    frame_velocity_ms: tuple[float, float] = (0.0, 0.0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def read_npy_float32(path: Path) -> Float32Array:
    try:
        with path.open("rb") as handle:
            require(handle.read(6) == b"\x93NUMPY", f"{path}: not an NPY file")
            version = handle.read(2)
            require(len(version) == 2, f"{path}: truncated NPY version")
            major, minor = version
            if (major, minor) == (1, 0):
                raw_length = handle.read(2)
                require(len(raw_length) == 2, f"{path}: truncated NPY header length")
                header_length = struct.unpack("<H", raw_length)[0]
            elif major in (2, 3):
                raw_length = handle.read(4)
                require(len(raw_length) == 4, f"{path}: truncated NPY header length")
                header_length = struct.unpack("<I", raw_length)[0]
            else:
                raise QualificationError(
                    f"{path}: unsupported NPY version {major}.{minor}"
                )
            raw_header = handle.read(header_length)
            require(
                len(raw_header) == header_length,
                f"{path}: truncated NPY header",
            )
            encoding = "utf-8" if major == 3 else "latin1"
            try:
                header = ast.literal_eval(raw_header.decode(encoding))
            except (UnicodeDecodeError, SyntaxError, ValueError) as error:
                raise QualificationError(f"{path}: invalid NPY header: {error}") from error
            require(isinstance(header, dict), f"{path}: invalid NPY header object")
            require(
                header.get("descr") in ("<f4", "|f4"),
                f"{path}: expected little-endian float32, got {header.get('descr')!r}",
            )
            require(
                header.get("fortran_order") is False,
                f"{path}: Fortran-order arrays are unsupported",
            )
            raw_shape = header.get("shape")
            require(
                isinstance(raw_shape, tuple)
                and len(raw_shape) == 3
                and all(isinstance(n, int) and not isinstance(n, bool) and n > 0
                        for n in raw_shape),
                f"{path}: expected a positive rank-3 shape, got {raw_shape!r}",
            )
            shape = tuple(raw_shape)
            count = math.prod(shape)
            payload = handle.read()
    except OSError as error:
        raise QualificationError(f"cannot read NPY {path}: {error}") from error

    require(
        len(payload) == count * 4,
        f"{path}: payload size mismatch (expected {count * 4}, got {len(payload)})",
    )
    values = tuple(value for (value,) in struct.iter_unpack("<f", payload))
    require(
        all(math.isfinite(value) for value in values),
        f"{path}: contains NaN or infinity",
    )
    return Float32Array(shape=shape, values=values)


def valid_generation(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def parse_frame(path: Path, expected_generation: str) -> Frame:
    manifest_path = path / "manifest.json"
    manifest = read_json_object(manifest_path)
    require(manifest.get("schema") == "nimbus.step", f"{manifest_path}: wrong schema")
    require(
        manifest.get("manifest_version") == OUTPUT_MANIFEST_VERSION,
        f"{manifest_path}: unsupported manifest version",
    )
    require(
        manifest.get("publication_state") == "complete",
        f"{manifest_path}: frame is not complete",
    )
    require(
        manifest.get("publication_generation") == expected_generation,
        f"{manifest_path}: publication generation mismatch",
    )
    require(
        manifest.get("step_dir") == path.name,
        f"{manifest_path}: step_dir does not match its directory",
    )
    expected_step_index = int(path.name.removeprefix("step_"))
    step_index = manifest.get("step_index")
    require(
        isinstance(step_index, int)
        and not isinstance(step_index, bool)
        and step_index == expected_step_index,
        f"{manifest_path}: step_index does not match its directory",
    )
    time_s = manifest.get("simulation_time_s")
    require(
        isinstance(time_s, (int, float))
        and not isinstance(time_s, bool)
        and math.isfinite(float(time_s)),
        f"{manifest_path}: invalid simulation_time_s",
    )

    grid = manifest.get("grid")
    require(isinstance(grid, dict), f"{manifest_path}: missing grid object")
    require(
        grid.get("coordinate_system") == "cartesian",
        f"{manifest_path}: expected Cartesian grid",
    )
    require(
        grid.get("axis_order") == ["x", "y", "z"],
        f"{manifest_path}: invalid grid.axis_order",
    )
    dimensions_object = grid.get("dimensions")
    require(isinstance(dimensions_object, dict), f"{manifest_path}: missing grid.dimensions")
    dimensions: list[int] = []
    for key in ("x", "y", "z"):
        value = dimensions_object.get(key)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"{manifest_path}: invalid grid.dimensions.{key}",
        )
        dimensions.append(value)
    spacing_object = grid.get("spacing_m")
    require(isinstance(spacing_object, dict), f"{manifest_path}: missing grid.spacing_m")
    spacings: list[float] = []
    for key in ("x", "y", "z"):
        value = spacing_object.get(key)
        require(finite_positive(value), f"{manifest_path}: invalid grid.spacing_m.{key}")
        spacings.append(float(value))

    native_staggering = grid.get("native_model_staggering")
    require(
        native_staggering in {"collocated", "c_grid"},
        f"{manifest_path}: invalid grid.native_model_staggering",
    )
    require(
        grid.get("publication_staggering") == "cell_centered",
        f"{manifest_path}: qualification fields must be cell-centered",
    )
    scalar_origin = grid.get("scalar_origin_m")
    require(
        isinstance(scalar_origin, list)
        and len(scalar_origin) == 3
        and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in scalar_origin),
        f"{manifest_path}: invalid grid.scalar_origin_m",
    )
    physical_bounds = grid.get("physical_bounds_m")
    require(isinstance(physical_bounds, dict), f"{manifest_path}: missing physical bounds")
    lower = physical_bounds.get("lower")
    upper = physical_bounds.get("upper")
    require(
        isinstance(lower, list) and isinstance(upper, list)
        and len(lower) == 3 and len(upper) == 3,
        f"{manifest_path}: invalid physical bounds",
    )
    require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in [*lower, *upper]
        ),
        f"{manifest_path}: non-finite physical bounds",
    )
    valid_range = grid.get("valid_index_range")
    halo_width = grid.get("halo_width")
    require(isinstance(valid_range, dict), f"{manifest_path}: missing valid index range")
    require(isinstance(halo_width, dict), f"{manifest_path}: missing halo width")
    expected_lower = [1 if dimensions[0] >= 3 else 0,
                      1 if dimensions[1] >= 3 else 0,
                      1 if native_staggering == "collocated" and dimensions[2] >= 3 else 0]
    expected_upper = [dimensions[0] - 1 - expected_lower[0],
                      dimensions[1] - 1 - expected_lower[1],
                      dimensions[2] - 1 - expected_lower[2]]
    require(
        valid_range.get("lower") == expected_lower
        and valid_range.get("upper") == expected_upper,
        f"{manifest_path}: inconsistent valid index range",
    )
    require(
        halo_width.get("lower") == expected_lower
        and halo_width.get("upper") == expected_lower,
        f"{manifest_path}: inconsistent halo width",
    )
    horizontal_frame = grid.get("horizontal_frame", {})
    require(
        isinstance(horizontal_frame, dict),
        f"{manifest_path}: invalid horizontal frame",
    )
    live_origin = horizontal_frame.get("grid_origin_earth_m", [0.0, 0.0])
    require(
        isinstance(live_origin, list) and len(live_origin) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in live_origin
        ),
        f"{manifest_path}: invalid live Earth origin",
    )
    frame_velocity = horizontal_frame.get(
        "frame_velocity_earth_ms", [0.0, 0.0]
    )
    require(
        isinstance(frame_velocity, list) and len(frame_velocity) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in frame_velocity
        ),
        f"{manifest_path}: invalid frame velocity",
    )
    for axis in range(3):
        axis_origin = float(live_origin[axis]) if axis < 2 else 0.0
        expected_bound_lower = (
            axis_origin + expected_lower[axis] * spacings[axis]
        )
        expected_bound_upper = (
            axis_origin
            + (dimensions[axis] - expected_lower[axis]) * spacings[axis]
        )
        require(
            math.isclose(float(lower[axis]), expected_bound_lower, abs_tol=1.0e-9)
            and math.isclose(
                float(scalar_origin[axis]),
                axis_origin + 0.5 * spacings[axis],
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
            and math.isclose(
                float(upper[axis]), expected_bound_upper, rel_tol=1.0e-9
            ),
            f"{manifest_path}: inconsistent coordinate metadata on axis {axis}",
        )

    published = manifest.get("published_fields")
    require(
        isinstance(published, list)
        and all(isinstance(field, str) and field for field in published),
        f"{manifest_path}: invalid published_fields",
    )
    published_set = frozenset(published)
    require(
        all(field in published_set for field in REQUIRED_FIELDS),
        f"{manifest_path}: required qualification fields are not published",
    )
    return Frame(
        path=path,
        time_s=float(time_s),
        shape=tuple(dimensions),
        dx_m=spacings[0],
        dy_m=spacings[1],
        dz_m=spacings[2],
        published_fields=published_set,
        valid_lower=tuple(expected_lower),
        valid_upper=tuple(expected_upper),
        frame_velocity_ms=(
            float(frame_velocity[0]),
            float(frame_velocity[1]),
        ),
    )


def physical_indices(frame: Frame) -> Iterator[tuple[int, int, int]]:
    for i in range(frame.valid_lower[0], frame.valid_upper[0] + 1):
        for j in range(frame.valid_lower[1], frame.valid_upper[1] + 1):
            for k in range(frame.valid_lower[2], frame.valid_upper[2] + 1):
                yield i, j, k


def relative_drift(initial: float, final: float) -> float:
    return abs(final - initial) / max(1.0, abs(initial))


def max_physical_difference(
    initial: Float32Array,
    final: Float32Array,
    frame: Frame,
) -> float:
    return max(
        abs(final.at(i, j, k) - initial.at(i, j, k))
        for i, j, k in physical_indices(frame)
    )


def hydrostatic_relative_residual(
    rho: Float32Array, pressure: Float32Array, frame: Frame
) -> float:
    residual = 0.0
    for i in range(frame.valid_lower[0], frame.valid_upper[0] + 1):
        for j in range(frame.valid_lower[1], frame.valid_upper[1] + 1):
            for k in range(frame.valid_lower[2], frame.valid_upper[2]):
                rho_face = 0.5 * (rho.at(i, j, k) + rho.at(i, j, k + 1))
                weight = rho_face * GRAVITY_MS2
                require(weight > 0.0, "hydrostatic residual encountered non-positive density")
                dp_dz = (pressure.at(i, j, k + 1) - pressure.at(i, j, k)) / frame.dz_m
                residual = max(residual, abs(dp_dz + weight) / weight)
    return residual


def eos_relative_residual(
    rho: Float32Array,
    pressure: Float32Array,
    theta: Float32Array,
    frame: Frame,
) -> float:
    kappa = GAS_CONSTANT_DRY_AIR_JKGK / SPECIFIC_HEAT_CP_JKGK
    residual = 0.0
    for i, j, k in physical_indices(frame):
        p_value = pressure.at(i, j, k)
        theta_value = theta.at(i, j, k)
        require(p_value > 0.0 and theta_value > 0.0, "EOS residual encountered invalid p/theta")
        temperature = theta_value * (p_value / REFERENCE_PRESSURE_PA) ** kappa
        diagnosed_rho = p_value / (GAS_CONSTANT_DRY_AIR_JKGK * temperature)
        residual = max(
            residual,
            abs(rho.at(i, j, k) - diagnosed_rho) / diagnosed_rho,
        )
    return residual


def inventories(
    fields: Mapping[str, Float32Array], frame: Frame | float
) -> dict[str, float]:
    """Return carrier inventories over the manifest physical domain.

    The headless writer publishes cell-centered velocities even when the
    prognostic grid is C-staggered. Do not center these arrays a second time.
    A numeric second argument retains the original C-grid helper API used by
    focused unit tests; production qualification always supplies a Frame.
    """
    cv = SPECIFIC_HEAT_CP_JKGK - GAS_CONSTANT_DRY_AIR_JKGK
    kappa = GAS_CONSTANT_DRY_AIR_JKGK / SPECIFIC_HEAT_CP_JKGK
    mass_terms: list[float] = []
    rho_theta_terms: list[float] = []
    water_terms: list[float] = []
    energy_terms: list[float] = []
    if isinstance(frame, Frame):
        indices = physical_indices(frame)
        dz_m = frame.dz_m
    else:
        nx, ny, nz = fields["rho"].shape
        indices = (
            (i, j, k)
            for i in range(1, nx - 1)
            for j in range(1, ny - 1)
            for k in range(nz)
        )
        dz_m = float(frame)
    for i, j, k in indices:
        rho_value = fields["rho"].at(i, j, k)
        theta_value = fields["theta"].at(i, j, k)
        pressure_value = fields["p"].at(i, j, k)
        qv_value = fields["qv"].at(i, j, k)
        u_value = fields["u"].at(i, j, k)
        v_value = fields["v"].at(i, j, k)
        w_value = fields["w"].at(i, j, k)
        require(rho_value > 0.0, "inventory encountered non-positive density")
        require(theta_value > 0.0 and pressure_value > 0.0, "inventory encountered invalid thermodynamics")
        require(qv_value >= 0.0, "inventory encountered negative water vapor")

        temperature = theta_value * (pressure_value / REFERENCE_PRESSURE_PA) ** kappa
        kinetic = 0.5 * rho_value * (
            u_value * u_value + v_value * v_value + w_value * w_value
        )
        internal = rho_value * cv * temperature
        # The existing energy diagnostic defines scalar-center height this way.
        # The broader scalar-origin contract remains tracked separately as SCI-019.
        z_center_m = (float(k) + 0.5) * dz_m
        potential = rho_value * GRAVITY_MS2 * z_center_m
        latent = rho_value * LATENT_HEAT_VAPORIZATION_JKG * qv_value

        mass_terms.append(rho_value)
        rho_theta_terms.append(rho_value * theta_value)
        water_terms.append(rho_value * qv_value)
        energy_terms.append(kinetic + internal + potential + latent)
    return {
        "mass": math.fsum(mass_terms),
        "rho_theta": math.fsum(rho_theta_terms),
        "water": math.fsum(water_terms),
        "total_energy": math.fsum(energy_terms),
    }


def validate_summary(run_dir: Path) -> None:
    path = run_dir / "validation_summary.json"
    summary = read_json_object(path)
    require(summary.get("context") == "run_summary", f"{path}: wrong context")
    require(summary.get("guard_mode") == "strict", f"{path}: strict guard was not active")
    require(summary.get("failed") is False, f"{path}: validation summary failed")
    for key in (
        "failed_report_count",
        "total_violations",
        "total_nonfinite",
        "total_bounds",
        "total_sanitized_nonfinite",
        "total_sanitized_bounds",
    ):
        require(summary.get(key) == 0, f"{path}: {key} must be zero")
    report_count = summary.get("report_count")
    require(
        isinstance(report_count, int) and not isinstance(report_count, bool) and report_count >= 2,
        f"{path}: expected initial/final validation reports",
    )


def load_fields(frame: Frame) -> dict[str, Float32Array]:
    fields: dict[str, Float32Array] = {}
    for field in REQUIRED_FIELDS:
        array = read_npy_float32(frame.path / f"{field}.npy")
        require(
            array.shape == frame.shape,
            f"{frame.path}/{field}.npy: shape {array.shape} != manifest {frame.shape}",
        )
        fields[field] = array
    return fields


def enforce_upper_bound(name: str, value: float, limit: float) -> None:
    require(math.isfinite(value), f"{name}: non-finite metric")
    require(value <= limit, f"{name}={value:.9g} exceeds {limit:.9g}")


def qualify_run(
    run_dir: Path,
    thresholds: Thresholds = Thresholds(),
    minimum_duration_s: float = 3600.0,
) -> dict[str, object]:
    marker_path = run_dir / "nimbus_output.json"
    marker = read_json_object(marker_path)
    require(marker.get("schema") == "nimbus.output", f"{marker_path}: wrong schema")
    require(
        marker.get("manifest_version") == OUTPUT_MANIFEST_VERSION,
        f"{marker_path}: unsupported manifest version",
    )
    require(marker.get("publication_state") == "ready", f"{marker_path}: run is not ready")
    require(
        marker.get("frame_completion") == "atomic_step_manifest",
        f"{marker_path}: wrong frame-completion contract",
    )
    generation = marker.get("publication_generation")
    require(valid_generation(generation), f"{marker_path}: invalid publication generation")
    assert isinstance(generation, str)

    step_paths = sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and STEP_DIR_PATTERN.fullmatch(path.name)
    )
    require(len(step_paths) == 2, f"{run_dir}: expected exactly 2 published frames, found {len(step_paths)}")
    frames = sorted(
        (parse_frame(path, generation) for path in step_paths),
        key=lambda frame: frame.time_s,
    )
    initial_frame, final_frame = frames
    require(abs(initial_frame.time_s) <= 1.0e-9, "initial frame is not at t=0")
    require(
        final_frame.time_s + 1.0e-9 >= minimum_duration_s,
        f"final frame time {final_frame.time_s:g}s is shorter than {minimum_duration_s:g}s",
    )
    require(initial_frame.shape == final_frame.shape, "initial/final shapes differ")
    require(
        (initial_frame.dx_m, initial_frame.dy_m, initial_frame.dz_m)
        == (final_frame.dx_m, final_frame.dy_m, final_frame.dz_m),
        "initial/final grid spacings differ",
    )
    validate_summary(run_dir)

    initial = load_fields(initial_frame)
    final = load_fields(final_frame)
    for field in REQUIRED_FIELDS:
        require(initial[field].shape == final[field].shape, f"{field}: initial/final shapes differ")

    max_wind = max(
        math.sqrt(
            (final["u"].at(i, j, k) - final_frame.frame_velocity_ms[0]) ** 2
            + (final["v"].at(i, j, k) - final_frame.frame_velocity_ms[1]) ** 2
            + final["w"].at(i, j, k) ** 2
        )
        for i, j, k in physical_indices(final_frame)
    )
    max_pressure_drift = max_physical_difference(initial["p"], final["p"], initial_frame)
    max_density_drift = max_physical_difference(initial["rho"], final["rho"], initial_frame)
    max_theta_drift = max_physical_difference(initial["theta"], final["theta"], initial_frame)
    max_hydrostatic = max(
        hydrostatic_relative_residual(initial["rho"], initial["p"], initial_frame),
        hydrostatic_relative_residual(final["rho"], final["p"], final_frame),
    )
    max_eos = max(
        eos_relative_residual(initial["rho"], initial["p"], initial["theta"], initial_frame),
        eos_relative_residual(final["rho"], final["p"], final["theta"], final_frame),
    )

    initial_inventories = inventories(initial, initial_frame)
    final_inventories = inventories(final, final_frame)
    inventory_drifts = {
        name: relative_drift(initial_inventories[name], final_inventories[name])
        for name in ("mass", "rho_theta", "water")
    }
    energy_drift = relative_drift(
        initial_inventories["total_energy"],
        final_inventories["total_energy"],
    )

    enforce_upper_bound("max_wind_ms", max_wind, thresholds.max_wind_ms)
    enforce_upper_bound(
        "max_pressure_drift_pa",
        max_pressure_drift,
        thresholds.max_pressure_drift_pa,
    )
    enforce_upper_bound(
        "max_density_drift_kgm3",
        max_density_drift,
        thresholds.max_density_drift_kgm3,
    )
    enforce_upper_bound("max_theta_drift_k", max_theta_drift, thresholds.max_theta_drift_k)
    enforce_upper_bound(
        "max_hydrostatic_relative_residual",
        max_hydrostatic,
        thresholds.max_hydrostatic_relative_residual,
    )
    enforce_upper_bound(
        "max_eos_relative_residual",
        max_eos,
        thresholds.max_eos_relative_residual,
    )
    for name, value in inventory_drifts.items():
        enforce_upper_bound(
            f"{name}_relative_drift",
            value,
            thresholds.max_inventory_relative_drift,
        )
    enforce_upper_bound(
        "total_energy_relative_drift",
        energy_drift,
        thresholds.max_total_energy_relative_drift,
    )

    physical_cell_count = math.prod(
        upper - lower + 1
        for lower, upper in zip(initial_frame.valid_lower, initial_frame.valid_upper)
    )
    return {
        "status": "PASS",
        "duration_s": final_frame.time_s - initial_frame.time_s,
        "grid_shape": list(initial_frame.shape),
        "physical_cell_count": physical_cell_count,
        "inventories": {
            "initial": initial_inventories,
            "final": final_inventories,
        },
        "metrics": {
            "max_wind_ms": max_wind,
            "max_pressure_drift_pa": max_pressure_drift,
            "max_density_drift_kgm3": max_density_drift,
            "max_theta_drift_k": max_theta_drift,
            "max_hydrostatic_relative_residual": max_hydrostatic,
            "max_eos_relative_residual": max_eos,
            "mass_relative_drift": inventory_drifts["mass"],
            "rho_theta_relative_drift": inventory_drifts["rho_theta"],
            "water_relative_drift": inventory_drifts["water"],
            "total_energy_relative_drift": energy_drift,
        },
        "thresholds": asdict(thresholds),
    }


def non_negative_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--minimum-duration-s", type=non_negative_finite, default=3600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = qualify_run(args.run_dir, minimum_duration_s=args.minimum_duration_s)
    except QualificationError as error:
        parser.exit(1, f"[hydrostatic-qualification] FAIL: {error}\n")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

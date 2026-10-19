#!/usr/bin/env python3
"""Qualify coupled terrain-following C-grid science on the production CPU path."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
import sys
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import Sequence


COMMON_TOOLS = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

from completed_step_energy_qualification import (
    completed_step_energy_identities,
)
from check_hydrostatic_hold import (
    QualificationError,
    read_json_object,
    read_npy_float32,
    require,
    valid_generation,
)


CORE_FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
COORDINATE_FIELDS = ("physical_z_center_m", "physical_z_interface_m")
LEDGER_FILES = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
DT_S = 0.2
RADIATION_CADENCE_S = 5
TRANSPORT_SCALARS = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh")


@dataclass(frozen=True)
class Thresholds:
    maximum_radiation_step_relative_residual: float = 1.0e-10
    maximum_radiation_cumulative_relative_residual: float = 1.0e-12
    maximum_shortwave_power_relative_residual: float = 1.0e-10
    maximum_momentum_relative_drift: float = 1.0e-3
    maximum_moisture_relative_drift: float = 1.0e-4
    maximum_energy_relative_drift: float = 1.0e-4
    maximum_source_heat_relative_residual: float = 1.0e-6
    maximum_source_water_relative_residual: float = 1.0e-6
    maximum_dry_mass_identity_relative_residual: float = 2.0e-15
    maximum_cumulative_dry_mass_identity_relative_residual: float = 2.0e-14
    maximum_dynamics_dry_mass_closure_fraction: float = 1.0e-6
    maximum_acoustic_energy_identity_relative_residual: float = 2.0e-15
    minimum_terrain_relief_m: float = 10.0
    minimum_sgs_dissipation_j: float = 1.0
    minimum_tke_energy_response_j: float = 1.0
    minimum_diffusion_dissipation_j: float = 1.0e6
    minimum_diffusion_scalar_energy_response_j: float = 1.0e6
    minimum_pbl_heat_response_j: float = 1.0
    minimum_pbl_vapor_response_kg: float = 1.0e-6
    minimum_pbl_momentum_work_j: float = 1.0
    maximum_diffusion_storage_rounding_fraction: float = 2.0e-3
    minimum_radiation_energy_j: float = 1.0
    minimum_acoustic_boundary_work_j: float = 1.0
    minimum_acoustic_radiative_work_j: float = 1.0


@dataclass(frozen=True)
class TerrainFrame:
    path: Path
    time_s: float
    shape: tuple[int, int, int]
    spacing_m: tuple[float, float, float]
    valid_lower: tuple[int, int, int]
    valid_upper: tuple[int, int, int]
    center_path: Path
    interface_path: Path


@dataclass(frozen=True)
class Float64Array:
    shape: tuple[int, int, int]
    values: tuple[float, ...]

    def at(self, i: int, j: int, k: int) -> float:
        _, ny, nz = self.shape
        return self.values[(i * ny + j) * nz + k]


def read_npy_float64(path: Path) -> Float64Array:
    try:
        with path.open("rb") as handle:
            require(handle.read(6) == b"\x93NUMPY", f"{path}: not NPY")
            version = handle.read(2)
            require(len(version) == 2, f"{path}: truncated NPY version")
            major, minor = version
            if (major, minor) == (1, 0):
                raw_length = handle.read(2)
                require(len(raw_length) == 2,
                        f"{path}: truncated NPY header length")
                header_length = struct.unpack("<H", raw_length)[0]
            elif major in (2, 3):
                raw_length = handle.read(4)
                require(len(raw_length) == 4,
                        f"{path}: truncated NPY header length")
                header_length = struct.unpack("<I", raw_length)[0]
            else:
                raise QualificationError(
                    f"{path}: unsupported NPY version {major}.{minor}")
            raw_header = handle.read(header_length)
            require(len(raw_header) == header_length,
                    f"{path}: truncated NPY header")
            encoding = "utf-8" if major == 3 else "latin1"
            header = ast.literal_eval(raw_header.decode(encoding))
            require(isinstance(header, dict), f"{path}: invalid NPY header")
            require(header.get("descr") in ("<f8", "|f8"),
                    f"{path}: expected float64")
            require(header.get("fortran_order") is False,
                    f"{path}: column-major NPY is unsupported")
            raw_shape = header.get("shape")
            require(isinstance(raw_shape, tuple) and len(raw_shape) == 3
                    and all(isinstance(n, int) and n > 0 for n in raw_shape),
                    f"{path}: expected a positive rank-3 shape")
            shape = tuple(raw_shape)
            payload = handle.read()
    except (OSError, SyntaxError, ValueError) as error:
        raise QualificationError(f"cannot read NPY {path}: {error}") from error
    require(len(payload) == math.prod(shape) * 8,
            f"{path}: payload size mismatch")
    values = tuple(value for (value,) in struct.iter_unpack("<d", payload))
    require(all(math.isfinite(value) for value in values),
            f"{path}: contains non-finite coordinates")
    return Float64Array(shape=shape, values=values)


def strict_summary(run_dir: Path) -> None:
    path = run_dir / "validation_summary.json"
    summary = read_json_object(path)
    require(summary.get("context") == "run_summary",
            f"{path}: wrong context")
    require(summary.get("guard_mode") == "strict",
            f"{path}: strict validation was not active")
    require(summary.get("failed") is False, f"{path}: validation failed")
    for key in (
        "failed_report_count", "total_violations", "total_nonfinite",
        "total_bounds", "total_sanitized_nonfinite",
        "total_sanitized_bounds",
    ):
        require(summary.get(key) == 0, f"{path}: {key} must be zero")


def parse_terrain_frame(path: Path, generation: str) -> TerrainFrame:
    manifest_path = path / "manifest.json"
    manifest = read_json_object(manifest_path)
    require(manifest.get("schema") == "nimbus.step"
            and manifest.get("manifest_version") == 7
            and manifest.get("publication_state") == "complete",
            f"{manifest_path}: invalid publication contract")
    require(manifest.get("publication_generation") == generation,
            f"{manifest_path}: generation mismatch")
    time_s = manifest.get("simulation_time_s")
    require(isinstance(time_s, (int, float)) and math.isfinite(float(time_s)),
            f"{manifest_path}: invalid time")
    grid = manifest.get("grid")
    require(isinstance(grid, dict), f"{manifest_path}: missing grid")
    require(grid.get("coordinate_system") == "cartesian"
            and grid.get("native_model_staggering") == "c_grid"
            and grid.get("publication_staggering") == "cell_centered",
            f"{manifest_path}: wrong Cartesian C-grid identity")
    dimensions = grid.get("dimensions")
    spacing = grid.get("spacing_m")
    valid = grid.get("valid_index_range")
    require(isinstance(dimensions, dict) and isinstance(spacing, dict)
            and isinstance(valid, dict), f"{manifest_path}: incomplete grid")
    shape = tuple(int(dimensions[key]) for key in ("x", "y", "z"))
    spacing_m = tuple(float(spacing[key]) for key in ("x", "y", "z"))
    lower = tuple(int(value) for value in valid.get("lower", []))
    upper = tuple(int(value) for value in valid.get("upper", []))
    require(shape == (16, 16, 20) and lower == (1, 1, 0)
            and upper == (14, 14, 19),
            f"{manifest_path}: fixture shape/domain changed")
    vertical = grid.get("vertical_geometry")
    require(isinstance(vertical, dict)
            and vertical.get("kind") == "terrain_following"
            and vertical.get("center_coordinate") == "physical_z_center_m"
            and vertical.get("interface_coordinate") == "physical_z_interface_m"
            and vertical.get("dtype") == "float64"
            and vertical.get("layout") == "x_y_z"
            and vertical.get("storage") == "versioned_npy_geometry"
            and vertical.get("contract") == "nimbus.terrain-geometry.v1"
            and vertical.get("publication_generation") == generation,
            f"{manifest_path}: terrain vertical-coordinate contract changed")
    center_relative = vertical.get("center_path")
    interface_relative = vertical.get("interface_path")
    require(isinstance(center_relative, str)
            and isinstance(interface_relative, str),
            f"{manifest_path}: terrain geometry paths are malformed")
    center_path = (path / center_relative).resolve()
    interface_path = (path / interface_relative).resolve()
    require(center_path.is_file() and interface_path.is_file(),
            f"{manifest_path}: terrain geometry artifact is missing")
    published = manifest.get("published_fields")
    require(isinstance(published, list)
            and all(field in published for field in CORE_FIELDS),
            f"{manifest_path}: missing a core field")
    return TerrainFrame(
        path, float(time_s), shape, spacing_m, lower, upper,
        center_path, interface_path,
    )


def terrain_field_path(frame: TerrainFrame, field: str) -> Path:
    if field == "physical_z_center_m":
        return frame.center_path
    if field == "physical_z_interface_m":
        return frame.interface_path
    return frame.path / f"{field}.npy"


def frames(run_dir: Path) -> list[TerrainFrame]:
    marker = read_json_object(run_dir / "nimbus_output.json")
    generation = marker.get("publication_generation")
    require(marker.get("schema") == "nimbus.output"
            and marker.get("publication_state") == "ready"
            and valid_generation(generation),
            f"{run_dir}: invalid output marker")
    assert isinstance(generation, str)
    result = sorted(
        (parse_terrain_frame(path, generation)
         for path in run_dir.glob("step_[0-9][0-9][0-9][0-9][0-9][0-9]")
         if path.is_dir()),
        key=lambda frame: frame.time_s,
    )
    require(result, f"{run_dir}: no frames")
    strict_summary(run_dir)
    return result


def run(binary: Path, config: Path, outdir: Path, duration_s: int,
        checkpoint_s: int, threads: int, resume: Path | None = None) -> None:
    command = [
        str(binary), "--headless", f"--config={config}",
        f"--duration={duration_s}", f"--write-every={duration_s}",
        f"--outdir={outdir}", f"--checkpoint-every={checkpoint_s}",
        f"--threads={threads}", "--compute-backend=cpu",
        "--guard-mode=strict", "--log-profile=quiet",
    ]
    if resume is not None:
        command.append(f"--resume={resume}")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    subprocess.run(command, check=True, env=environment)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        result = list(csv.DictReader(handle))
    require(result, f"{path}: empty ledger")
    for row in result:
        for key, value in row.items():
            if key == "stage":
                continue
            require(value is not None and math.isfinite(float(value)),
                    f"{path}: non-finite {key}")
    return result


def checkpoint_sidecars(run_dir: Path) -> list[dict[str, object]]:
    result = [
        read_json_object(path)
        for path in sorted((run_dir / "checkpoints").glob("*.nimbus.json"))
    ]
    require(result and all(item.get("boundary") == "completed_timestep"
                           for item in result),
            f"{run_dir}: invalid checkpoint set")
    return result


def checkpoint_path(run_dir: Path, sidecar: dict[str, object]) -> Path:
    filename = sidecar.get("checkpoint_file")
    require(isinstance(filename, str) and filename,
            f"{run_dir}: malformed checkpoint sidecar")
    path = run_dir / "checkpoints" / filename
    require(path.is_file(), f"{run_dir}: missing {filename}")
    return path


def require_final_identity(reference_dir: Path, candidate_dir: Path,
                           duration_s: int, label: str) -> None:
    reference = frames(reference_dir)[-1]
    candidate = frames(candidate_dir)[-1]
    require(math.isclose(reference.time_s, duration_s, abs_tol=1.0e-8)
            and math.isclose(candidate.time_s, duration_s, abs_tol=1.0e-8),
            f"{label}: wrong final time")
    for field in (*CORE_FIELDS, *COORDINATE_FIELDS):
        require(terrain_field_path(reference, field).read_bytes()
                == terrain_field_path(candidate, field).read_bytes(),
                f"{label}: final {field} differs")


def require_ledger_identity(reference_dir: Path, candidate_dir: Path,
                            label: str) -> None:
    for filename in LEDGER_FILES:
        require((reference_dir / filename).read_bytes()
                == (candidate_dir / filename).read_bytes(),
                f"{label}: {filename} differs")


def require_restart_suffix(continuous_dir: Path, resumed_dir: Path,
                           label: str) -> None:
    for filename in LEDGER_FILES:
        continuous = (continuous_dir / filename).read_text(
            encoding="utf-8").splitlines()
        resumed = (resumed_dir / filename).read_text(
            encoding="utf-8").splitlines()
        require(continuous and resumed and continuous[0] == resumed[0]
                and len(resumed) > 1,
                f"{label}: malformed {filename}")
        require(continuous[-(len(resumed) - 1):] == resumed[1:],
                f"{label}: {filename} is not an exact restart suffix")


def relative_identity(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))


def relative_balance(lhs: float, *rhs_terms: float) -> float:
    rhs = sum(rhs_terms)
    return abs(lhs - rhs) / max(
        1.0, abs(lhs), *(abs(term) for term in rhs_terms))


def microphysics_moist_energy_identities(
        sources: list[dict[str, str]]) -> dict[str, float]:
    maximum = 0.0
    for row in sources:
        require(
            int(row["microphysics_moist_energy_report_schema_version"]) == 2
            and int(row["microphysics_moist_energy_basis_version"]) == 2
            and int(row["microphysics_moist_energy_complete"]) == 1,
            "mapped microphysics moist-energy currency is partial or changed")
        committed = float(
            row["microphysics_moist_committed_energy_change_j"])
        reservoir_terms = [float(row[key]) for key in (
            "microphysics_moist_dry_internal_change_j",
            "microphysics_moist_water_sensible_change_j",
            "microphysics_moist_phase_reference_change_j",
            "microphysics_moist_carrier_kinetic_change_j",
            "microphysics_moist_gravitational_change_j",
            "microphysics_moist_subgrid_tke_change_j",
        )]
        reservoirs = sum(reservoir_terms)
        closure = float(
            row["microphysics_moist_energy_closure_residual_j"])
        maximum = max(maximum,
            abs(committed - (reservoirs + closure))
            / max(1.0, abs(committed), abs(closure),
                  *(abs(term) for term in reservoir_terms)))
    require(maximum <= 1.0e-14,
            "mapped microphysics moist-energy identity does not close")
    return {
        "maximum_microphysics_moist_energy_identity_relative_residual":
            maximum,
    }


def sgs_moist_energy_identities(
        sources: list[dict[str, str]]) -> dict[str, float]:
    maximum = 0.0
    cumulative_maximum = 0.0
    for row in sources:
        require(
            int(row["sgs_moist_energy_report_schema_version"]) == 1
            and int(row["sgs_moist_energy_basis_version"]) == 2
            and int(row["sgs_moist_energy_step_complete"]) == 1
            and int(row["sgs_moist_energy_cumulative_complete"]) == 1,
            "mapped SGS moist-energy currency is partial or changed")
        step_reservoirs = float(row["sgs_committed_momentum_work_j"]) + sum(
            float(row[key]) for key in (
                "sgs_moist_energy_step_dry_internal_change_j",
                "sgs_moist_energy_step_water_sensible_change_j",
                "sgs_moist_energy_step_phase_reference_change_j",
                "sgs_moist_energy_step_carrier_kinetic_change_j",
                "sgs_moist_energy_step_gravitational_change_j",
                "sgs_moist_energy_step_subgrid_tke_change_j",
            ))
        maximum = max(maximum, relative_identity(
            float(row["sgs_moist_energy_step_committed_energy_change_j"]),
            step_reservoirs
            + float(row["sgs_moist_energy_step_closure_residual_j"])))
        cumulative_reservoirs = float(
            row["sgs_cumulative_committed_momentum_work_j"]) + sum(
                float(row[key]) for key in (
                    "sgs_moist_energy_cumulative_dry_internal_change_j",
                    "sgs_moist_energy_cumulative_water_sensible_change_j",
                    "sgs_moist_energy_cumulative_phase_reference_change_j",
                    "sgs_moist_energy_cumulative_carrier_kinetic_change_j",
                    "sgs_moist_energy_cumulative_gravitational_change_j",
                    "sgs_moist_energy_cumulative_subgrid_tke_change_j",
                ))
        cumulative_maximum = max(cumulative_maximum, relative_identity(
            float(
                row["sgs_moist_energy_cumulative_committed_energy_change_j"]),
            cumulative_reservoirs
            + float(
                row["sgs_moist_energy_cumulative_closure_residual_j"])))
    require(max(maximum, cumulative_maximum) <= 1.0e-14,
            "mapped SGS moist-energy reservoir identity does not close")
    return {
        "maximum_sgs_moist_energy_identity_relative_residual": maximum,
        "maximum_cumulative_sgs_moist_energy_identity_relative_residual":
            cumulative_maximum,
    }


def diffusion_energy_identities(
        sources: list[dict[str, str]], expected_steps: int,
        mode: str, thresholds: Thresholds) -> dict[str, float]:
    require(mode in ("inactive", "scalar", "scalar_momentum"),
            "mapped diffusion qualification mode is invalid")
    require(all(
        int(row["diffusion_moist_energy_report_schema_version"]) == 1
        and int(row["diffusion_moist_energy_basis_version"]) == 2
        for row in sources),
        "mapped diffusion energy currency version changed")
    if mode == "inactive":
        require(all(
            int(row["diffusion_momentum_work_step_count"]) == 0
            and int(row["diffusion_moist_energy_step_count"]) == 0
            and int(row["diffusion_moist_energy_step_complete"]) == 0
            and int(row["diffusion_moist_energy_cumulative_complete"]) == 0
            for row in sources),
            "inactive mapped diffusion authored energy work")
        return {
            "maximum_diffusion_work_identity_relative_residual": 0.0,
            "maximum_cumulative_diffusion_work_identity_relative_residual":
                0.0,
            "maximum_diffusion_moist_energy_identity_relative_residual": 0.0,
            "maximum_cumulative_diffusion_moist_energy_identity_relative_residual":
                0.0,
            "cumulative_diffusion_requested_momentum_work_j": 0.0,
            "cumulative_diffusion_committed_momentum_work_j": 0.0,
            "cumulative_diffusion_committed_energy_change_j": 0.0,
            "cumulative_diffusion_storage_rounding_fraction": 0.0,
        }

    expected = list(range(1, expected_steps + 1))
    momentum_counts = [
        int(row["diffusion_momentum_work_step_count"])
        for row in sources
    ]
    require(momentum_counts == (
        expected if mode == "scalar_momentum" else [0] * expected_steps)
        and [
        int(row["diffusion_moist_energy_step_count"])
        for row in sources
    ] == expected,
        "mapped diffusion work sequence changed")
    require(all(
        int(row["diffusion_moist_energy_step_complete"]) == 1
        and int(row["diffusion_moist_energy_cumulative_complete"]) == 1
        for row in sources),
        "mapped diffusion moist-energy currency is partial")

    work_maxima = {"step": 0.0, "cumulative": 0.0}
    moist_maxima = {"step": 0.0, "cumulative": 0.0}
    for row in sources:
        for cadence in ("step", "cumulative"):
            work_prefix = "diffusion_" if cadence == "step" else (
                "diffusion_cumulative_")
            committed_work = float(
                row[work_prefix + "committed_momentum_work_j"])
            work_maxima[cadence] = max(
                work_maxima[cadence],
                relative_balance(
                    committed_work,
                    float(row[work_prefix + "requested_momentum_work_j"]),
                    float(row[
                        work_prefix
                        + "finite_step_momentum_correction_j"]),
                    float(row[
                        work_prefix + "momentum_storage_rounding_j"])))
            moist_prefix = f"diffusion_moist_energy_{cadence}_"
            moist_terms = [
                committed_work,
                *(float(row[moist_prefix + suffix]) for suffix in (
                    "dry_internal_change_j",
                    "water_sensible_change_j",
                    "phase_reference_change_j",
                    "carrier_kinetic_change_j",
                    "gravitational_change_j",
                    "subgrid_tke_change_j",
                    "closure_residual_j",
                )),
            ]
            moist_maxima[cadence] = max(
                moist_maxima[cadence],
                relative_balance(
                    float(row[
                        moist_prefix + "committed_energy_change_j"]),
                    *moist_terms))
    require(max(*work_maxima.values(), *moist_maxima.values()) <= 1.0e-12,
            "mapped diffusion energy identities do not close")

    final = sources[-1]
    requested = float(
        final["diffusion_cumulative_requested_momentum_work_j"])
    committed = float(
        final["diffusion_cumulative_committed_momentum_work_j"])
    storage = float(
        final["diffusion_cumulative_momentum_storage_rounding_j"])
    cumulative_energy = float(final[
        "diffusion_moist_energy_cumulative_committed_energy_change_j"])
    storage_fraction = abs(storage) / max(1.0, abs(requested))
    if mode == "scalar_momentum":
        require(requested <= -thresholds.minimum_diffusion_dissipation_j
                and committed <= -thresholds.minimum_diffusion_dissipation_j,
                "mapped active diffusion did not exercise resolved dissipation")
        require(storage_fraction
                <= thresholds.maximum_diffusion_storage_rounding_fraction,
                "mapped diffusion FP32 work residual exceeded its bound")
    else:
        require(requested == 0.0 and committed == 0.0 and storage == 0.0,
                "mapped scalar diffusion authored momentum work")
        require(abs(cumulative_energy)
                >= thresholds.minimum_diffusion_scalar_energy_response_j,
                "mapped scalar diffusion did not exercise energy coupling")
    return {
        "maximum_diffusion_work_identity_relative_residual":
            work_maxima["step"],
        "maximum_cumulative_diffusion_work_identity_relative_residual":
            work_maxima["cumulative"],
        "maximum_diffusion_moist_energy_identity_relative_residual":
            moist_maxima["step"],
        "maximum_cumulative_diffusion_moist_energy_identity_relative_residual":
            moist_maxima["cumulative"],
        "cumulative_diffusion_requested_momentum_work_j": requested,
        "cumulative_diffusion_committed_momentum_work_j": committed,
        "cumulative_diffusion_committed_energy_change_j": cumulative_energy,
        "cumulative_diffusion_storage_rounding_fraction": storage_fraction,
    }


def pbl_work_identities(
        sources: list[dict[str, str]], expected_steps: int,
        active: bool, thresholds: Thresholds) -> dict[str, float]:
    if not active:
        require(all(
            int(row["pbl_momentum_work_step_count"]) == 0
            and float(row["pbl_heat_j"]) == 0.0
            and float(row["pbl_vapor_tendency_kg"]) == 0.0
            and float(row["pbl_cloud_liquid_tendency_kg"]) == 0.0
            and float(row["pbl_cloud_ice_tendency_kg"]) == 0.0
            for row in sources),
            "inactive mapped PBL authored source work")
        return {
            "maximum_pbl_momentum_work_identity_relative_residual": 0.0,
            "maximum_cumulative_pbl_momentum_work_identity_relative_residual":
                0.0,
            "maximum_absolute_pbl_heat_j": 0.0,
            "maximum_absolute_pbl_vapor_tendency_kg": 0.0,
            "cumulative_pbl_committed_momentum_work_j": 0.0,
        }

    require([
        int(row["pbl_momentum_work_step_count"]) for row in sources
    ] == list(range(1, expected_steps + 1)),
        "mapped PBL momentum-work sequence changed")
    step_maximum = 0.0
    cumulative_maximum = 0.0
    for row in sources:
        step_maximum = max(step_maximum, relative_balance(
            float(row["pbl_committed_momentum_work_j"]),
            float(row["pbl_requested_momentum_work_j"]),
            float(row["pbl_finite_step_momentum_correction_j"]),
            float(row["pbl_momentum_storage_rounding_j"])))
        cumulative_maximum = max(cumulative_maximum, relative_balance(
            float(row["pbl_cumulative_committed_momentum_work_j"]),
            float(row["pbl_cumulative_requested_momentum_work_j"]),
            float(row[
                "pbl_cumulative_finite_step_momentum_correction_j"]),
            float(row["pbl_cumulative_momentum_storage_rounding_j"])))
    require(max(step_maximum, cumulative_maximum) <= 1.0e-12,
            "mapped PBL momentum-work identities do not close")
    maximum_heat = max(abs(float(row["pbl_heat_j"])) for row in sources)
    maximum_vapor = max(abs(float(
        row["pbl_vapor_tendency_kg"])) for row in sources)
    cumulative_work = float(
        sources[-1]["pbl_cumulative_committed_momentum_work_j"])
    require(maximum_heat >= thresholds.minimum_pbl_heat_response_j
            and maximum_vapor >= thresholds.minimum_pbl_vapor_response_kg
            and abs(cumulative_work)
                >= thresholds.minimum_pbl_momentum_work_j,
            "mapped PBL source coupling was not exercised")
    return {
        "maximum_pbl_momentum_work_identity_relative_residual":
            step_maximum,
        "maximum_cumulative_pbl_momentum_work_identity_relative_residual":
            cumulative_maximum,
        "maximum_absolute_pbl_heat_j": maximum_heat,
        "maximum_absolute_pbl_vapor_tendency_kg": maximum_vapor,
        "cumulative_pbl_committed_momentum_work_j": cumulative_work,
    }


def eos_pressure_energy_identities(
        sources: list[dict[str, str]],
        expected_steps: int) -> dict[str, float]:
    require([int(row["eos_pressure_energy_step_count"]) for row in sources]
            == list(range(1, expected_steps + 1)),
            "mapped EOS pressure-energy report sequence changed")
    maximum = 0.0
    cumulative_maximum = 0.0
    for row in sources:
        require(
            int(row["eos_pressure_energy_report_schema_version"]) == 1
            and int(row["eos_pressure_energy_basis_version"]) == 2
            and int(row["eos_pressure_energy_step_complete"]) == 1
            and int(row["eos_pressure_energy_cumulative_complete"]) == 1,
            "mapped EOS pressure-energy currency is partial or changed")
        for cadence in ("step", "cumulative"):
            prefix = f"eos_pressure_energy_{cadence}_"
            reservoirs = sum(float(row[prefix + suffix]) for suffix in (
                "dry_internal_change_j",
                "water_sensible_change_j",
                "phase_reference_change_j",
                "carrier_kinetic_change_j",
                "gravitational_change_j",
                "subgrid_tke_change_j",
            ))
            residual = relative_identity(
                float(row[prefix + "committed_energy_change_j"]),
                reservoirs + float(row[prefix + "closure_residual_j"]))
            if cadence == "step":
                maximum = max(maximum, residual)
            else:
                cumulative_maximum = max(cumulative_maximum, residual)
    require(max(maximum, cumulative_maximum) <= 1.0e-14,
            "mapped EOS pressure-energy reservoir identity does not close")
    return {
        "maximum_eos_pressure_energy_identity_relative_residual": maximum,
        "maximum_cumulative_eos_pressure_energy_identity_relative_residual":
            cumulative_maximum,
    }


def sponge_energy_identities(
        sources: list[dict[str, str]],
        expected_steps: int) -> dict[str, float]:
    require([int(row["sponge_energy_step_count"]) for row in sources]
            == list(range(1, expected_steps + 1)),
            "mapped sponge-energy report sequence changed")
    energy_maxima = {"step": 0.0, "cumulative": 0.0}
    momentum_maxima = {"step": 0.0, "cumulative": 0.0}
    for row in sources:
        require(
            int(row["sponge_energy_report_schema_version"]) == 1
            and int(row["sponge_energy_basis_version"]) == 2
            and int(row["sponge_energy_step_complete"]) == 1
            and int(row["sponge_energy_cumulative_complete"]) == 1,
            "mapped sponge-energy currency is partial or changed")
        for cadence in ("step", "cumulative"):
            for pass_name in ("upper", "w_only", "lateral"):
                prefix = f"sponge_energy_{cadence}_{pass_name}_"
                require(int(row[prefix + "complete"]) == 1,
                        f"mapped {pass_name} sponge report is incomplete")
                carrier = float(row[prefix + "carrier_kinetic_change_j"])
                requested = float(
                    row[prefix + "momentum_requested_linear_work_j"])
                finite = float(
                    row[prefix + "momentum_finite_step_correction_j"])
                storage = float(
                    row[prefix + "momentum_storage_rounding_j"])
                momentum_maxima[cadence] = max(
                    momentum_maxima[cadence],
                    abs(carrier - (requested + finite + storage))
                    / max(1.0, abs(carrier), abs(requested),
                          abs(finite), abs(storage)))
                reservoir_terms = [float(row[prefix + suffix])
                                   for suffix in (
                                       "dry_internal_change_j",
                                       "water_sensible_change_j",
                                       "phase_reference_change_j",
                                       "carrier_kinetic_change_j",
                                       "gravitational_change_j",
                                       "subgrid_tke_change_j")]
                reservoirs = sum(reservoir_terms)
                closure = float(row[prefix + "closure_residual_j"])
                committed = float(
                    row[prefix + "committed_energy_change_j"])
                energy_maxima[cadence] = max(
                    energy_maxima[cadence],
                    abs(committed - (reservoirs + closure))
                    / max(1.0, abs(committed), abs(closure),
                          *(abs(term) for term in reservoir_terms)))
    require(max(*energy_maxima.values(), *momentum_maxima.values())
            <= 1.0e-14,
            "mapped sponge operator-time energy identity does not close")
    return {
        "maximum_sponge_energy_identity_relative_residual":
            energy_maxima["step"],
        "maximum_cumulative_sponge_energy_identity_relative_residual":
            energy_maxima["cumulative"],
        "maximum_sponge_momentum_identity_relative_residual":
            momentum_maxima["step"],
        "maximum_cumulative_sponge_momentum_identity_relative_residual":
            momentum_maxima["cumulative"],
    }


def final_boundary_energy_identities(
        sources: list[dict[str, str]],
        expected_steps: int) -> dict[str, float]:
    require([int(row["final_boundary_energy_step_count"]) for row in sources]
            == list(range(1, expected_steps + 1)),
            "mapped final-boundary energy report sequence changed")
    maxima = {"step": 0.0, "cumulative": 0.0}
    for row in sources:
        require(
            int(row["final_boundary_energy_report_schema_version"]) == 1
            and int(row["final_boundary_energy_basis_version"]) == 2,
            "mapped final-boundary energy currency changed")
        for cadence in ("step", "cumulative"):
            prefix = f"final_boundary_energy_{cadence}_"
            require(
                int(row[prefix + "applied"]) == 1
                and int(row[prefix + "complete"]) == 1,
                "mapped final-boundary energy report is incomplete")
            carrier = float(row[prefix + "carrier_kinetic_change_j"])
            constraint = float(
                row[prefix + "constraint_committed_work_j"])
            reservoirs = sum(float(row[prefix + suffix])
                             for suffix in (
                                 "dry_internal_change_j",
                                 "water_sensible_change_j",
                                 "phase_reference_change_j",
                                 "carrier_kinetic_change_j",
                                 "gravitational_change_j",
                                 "subgrid_tke_change_j",
                                 "closure_residual_j"))
            committed = float(row[prefix + "committed_energy_change_j"])
            maxima[cadence] = max(
                maxima[cadence],
                relative_identity(constraint, carrier),
                relative_identity(committed, reservoirs))
    require(max(maxima.values()) <= 1.0e-14,
            "mapped final-boundary energy identity does not close")
    return {
        "maximum_final_boundary_energy_identity_relative_residual":
            maxima["step"],
        "maximum_cumulative_final_boundary_energy_identity_relative_residual":
            maxima["cumulative"],
    }


def scalar_energy_identities(
        sources: list[dict[str, str]],
        expected_steps: int) -> dict[str, float]:
    require([
        int(row["advection_thermodynamic_moist_energy_report_step_count"])
        for row in sources
    ] == list(range(1, expected_steps + 1)),
        "mapped scalar moist-energy report sequence changed")
    maxima = [0.0, 0.0, 0.0, 0.0]
    for row in sources:
        require(
            int(row["advection_scalar_energy_report_schema_version"]) == 1
            and int(row["advection_scalar_energy_basis_version"]) == 2
            and int(row[
                "advection_thermodynamic_moist_energy_reports_complete"]) == 1,
            "mapped scalar moist-energy currency is partial or changed")
        for scalar in TRANSPORT_SCALARS:
            for cadence in ("step", "cumulative"):
                prefix = f"advection_{scalar}_energy_{cadence}_"
                committed = float(row[prefix + "committed_state_change_j"])
                reservoir_terms = [float(row[prefix + key]) for key in (
                    "dry_internal_change_j",
                    "water_sensible_change_j",
                    "phase_reference_change_j",
                    "carrier_kinetic_change_j",
                    "gravitational_change_j",
                    "subgrid_tke_change_j",
                )]
                reservoirs = sum(reservoir_terms)
                limited = float(
                    row[prefix + "limited_face_state_change_j"])
                outward = float(row[prefix + "outward_boundary_j"])
                conversion = float(row[prefix + "interior_conversion_j"])
                requested = float(row[prefix + "requested_state_change_j"])
                remap = float(row[prefix + "density_history_remap_j"])
                storage = float(row[prefix + "storage_rounding_j"])
                closure = float(row[prefix + "closure_residual_j"])
                identities = (
                    abs(committed - reservoirs)
                    / max(1.0, abs(committed),
                          *(abs(term) for term in reservoir_terms)),
                    relative_balance(limited, -outward, conversion),
                    relative_identity(requested, remap + limited),
                    relative_identity(
                        committed, requested + storage + closure),
                )
                maxima = [
                    max(current, candidate)
                    for current, candidate in zip(maxima, identities)
                ]
    # Mapped boundary export and interior conversion can strongly cancel, so
    # the face balance uses the authored term scale rather than their small net.
    require(max(maxima) <= 1.0e-14,
            "mapped scalar moist-energy discrete identities do not close: "
            f"reservoir={maxima[0]:.17g}, face={maxima[1]:.17g}, "
            f"requested={maxima[2]:.17g}, commit={maxima[3]:.17g}")
    return {
        "maximum_scalar_energy_reservoir_identity_relative_residual":
            maxima[0],
        "maximum_scalar_energy_face_identity_relative_residual": maxima[1],
        "maximum_scalar_energy_requested_identity_relative_residual":
            maxima[2],
        "maximum_scalar_energy_commit_identity_relative_residual": maxima[3],
    }


def slow_momentum_energy_identities(
        sources: list[dict[str, str]],
        expected_steps: int) -> dict[str, float]:
    require([
        int(row["slow_momentum_energy_step_count"])
        for row in sources
    ] == list(range(1, expected_steps + 1)),
        "mapped slow momentum energy report sequence changed")
    maximum = 0.0
    pressure_maximum = 0.0
    for row in sources:
        require(
            int(row["slow_momentum_report_schema_version"]) == 2
            and int(row["slow_momentum_energy_basis_version"]) == 2,
            "mapped slow momentum energy currency version changed")
        for cadence in ("step", "cumulative"):
            prefix = f"slow_momentum_energy_{cadence}_"
            require(int(row[prefix + "complete"]) == 1,
                    "mapped slow momentum energy report is partial")
            advection_finite = float(
                row[prefix + "advection_finite_step_correction_j"])
            nonadvection_finite = float(
                row[prefix + "nonadvection_finite_step_correction_j"])
            require(advection_finite >= 0.0 and nonadvection_finite >= 0.0,
                    "mapped slow momentum finite-step correction is invalid")
            committed = float(
                row[prefix + "committed_kinetic_change_j"])
            attributed_terms = [float(row[prefix + key]) for key in (
                "advection_requested_linear_work_j",
                "nonadvection_requested_linear_work_j",
                "advection_finite_step_correction_j",
                "nonadvection_finite_step_correction_j",
                "coupled_finite_step_cross_work_j",
                "storage_rounding_j",
                "closure_residual_j",
            )]
            maximum = max(
                maximum, relative_balance(committed, *attributed_terms))
            pressure_committed = float(
                row[prefix + "committed_pressure_energy_change_j"])
            pressure_reservoirs = sum(float(row[prefix + key]) for key in (
                "pressure_dry_internal_change_j",
                "pressure_water_sensible_change_j",
            ))
            pressure_maximum = max(
                pressure_maximum,
                relative_identity(pressure_committed, pressure_reservoirs))
    require(maximum <= 1.0e-14,
            "mapped slow momentum kinetic-work identity does not close")
    require(pressure_maximum <= 1.0e-14,
            "mapped slow pressure reservoir identity does not close")
    return {
        "maximum_slow_momentum_energy_identity_relative_residual": maximum,
        "maximum_slow_pressure_energy_identity_relative_residual":
            pressure_maximum,
    }

def kinetic_transport_identities(
        sources: list[dict[str, str]]) -> dict[str, float]:
    continuity_maximum = 0.0
    combined_maximum = 0.0
    copied_momentum_maximum = 0.0
    for row in sources:
        for cadence in ("step", "cumulative"):
            prefix = ("acoustic_v1_" if cadence == "step"
                      else "acoustic_v1_cumulative_")
            require(int(row[prefix + "kinetic_transport_complete"]) == 1,
                    "mapped kinetic transport report is partial")
            committed = float(row[prefix + "carrier_kinetic_change_j"])
            requested = float(
                row[prefix + "continuity_carrier_requested_change_j"])
            outward = float(
                row[prefix
                    + "continuity_carrier_outward_boundary_transport_j"])
            continuity_interior = float(
                row[prefix + "continuity_carrier_interior_conversion_j"])
            storage = float(
                row[prefix + "continuity_carrier_storage_rounding_j"])
            momentum = float(
                row[prefix
                    + "slow_momentum_advection_requested_linear_work_j"])
            combined_interior = float(
                row[prefix + "kinetic_transport_interior_conversion_j"])
            closure = float(
                row[prefix + "kinetic_transport_closure_residual_j"])
            slow_prefix = f"slow_momentum_energy_{cadence}_"
            continuity_maximum = max(
                continuity_maximum,
                abs(committed - requested - storage)
                    / max(1.0, abs(committed), abs(requested), abs(storage)),
                abs(requested + outward - continuity_interior)
                    / max(1.0, abs(requested), abs(outward),
                          abs(continuity_interior)))
            copied_momentum_maximum = max(
                copied_momentum_maximum,
                relative_identity(
                    momentum,
                    float(row[
                        slow_prefix
                        + "advection_requested_linear_work_j"])))
            combined_maximum = max(
                combined_maximum,
                abs(requested + momentum + outward
                    - combined_interior - closure)
                    / max(1.0, abs(requested), abs(momentum), abs(outward),
                          abs(combined_interior), abs(closure)))
    require(max(
        continuity_maximum, combined_maximum,
        copied_momentum_maximum) <= 1.0e-14,
        "mapped kinetic transport discrete identities do not close")
    return {
        "maximum_continuity_carrier_identity_relative_residual":
            continuity_maximum,
        "maximum_kinetic_transport_identity_relative_residual":
            combined_maximum,
        "maximum_kinetic_transport_momentum_copy_relative_residual":
            copied_momentum_maximum,
    }


def analyze_geometry(run_dir: Path,
                     thresholds: Thresholds) -> dict[str, float]:
    run_frames = frames(run_dir)
    initial = run_frames[0]
    centers = read_npy_float64(initial.center_path)
    interfaces = read_npy_float64(initial.interface_path)
    require(centers.shape == initial.shape
            and interfaces.shape == (
                initial.shape[0], initial.shape[1], initial.shape[2] + 1),
            "terrain coordinate sidecar shape mismatch")
    ground = [
        interfaces.at(i, j, 0)
        for i in range(initial.valid_lower[0], initial.valid_upper[0] + 1)
        for j in range(initial.valid_lower[1], initial.valid_upper[1] + 1)
    ]
    relief = max(ground) - min(ground)
    require(relief >= thresholds.minimum_terrain_relief_m,
            "qualification terrain is effectively flat")
    tops = []
    minimum_thickness = math.inf
    for i in range(initial.valid_lower[0], initial.valid_upper[0] + 1):
        for j in range(initial.valid_lower[1], initial.valid_upper[1] + 1):
            tops.append(interfaces.at(i, j, initial.shape[2]))
            for k in range(initial.shape[2]):
                lower = interfaces.at(i, j, k)
                upper = interfaces.at(i, j, k + 1)
                center = centers.at(i, j, k)
                require(lower < center < upper,
                        "terrain center is outside its physical layer")
                minimum_thickness = min(minimum_thickness, upper - lower)
    top_spread = max(tops) - min(tops)
    require(top_spread <= 1.0e-9,
            "terrain qualification requires a flat physical model top")
    final = run_frames[-1]
    for field in COORDINATE_FIELDS:
        require(terrain_field_path(initial, field).read_bytes()
                == terrain_field_path(final, field).read_bytes(),
                f"{field}: static terrain geometry changed during the run")
    return {
        "terrain_relief_m": relief,
        "minimum_physical_layer_thickness_m": minimum_thickness,
        "physical_model_top_spread_m": top_spread,
    }


def analyze_physics(run_dir: Path, duration_s: int,
                    thresholds: Thresholds,
                    diffusion_mode: str,
                    shortwave_active: bool,
                    pbl_active: bool,
                    tke_active: bool) -> dict[str, float]:
    expected_steps = int(round(duration_s / DT_S))
    radiation = rows(run_dir / "radiation_budget.csv")
    require(len(radiation) == expected_steps
            and [int(row["step"]) for row in radiation]
                == list(range(1, expected_steps + 1)),
            "radiation ledger step sequence changed")
    samples: list[float] = []
    for row in radiation:
        sample = float(row["last_radiation_sample_s"])
        if not samples or sample != samples[-1]:
            samples.append(sample)
    expected_samples = [
        float(value) for value in range(
            0, duration_s, RADIATION_CADENCE_S)
    ]
    require(len(samples) == len(expected_samples)
            and all(math.isclose(actual, expected, abs_tol=1.0e-8)
                    for actual, expected in zip(samples, expected_samples)),
            f"terrain radiation cadence drifted: {samples}")
    maximum_rad_step = max(
        abs(float(row["relative_booking_residual"])) for row in radiation)
    final_rad = radiation[-1]
    cumulative_rad = (
        abs(float(final_rad["cumulative_booking_residual_j"]))
        / max(1.0, abs(float(final_rad["cumulative_expected_energy_j"])))
    )
    total_radiation_energy = abs(
        float(final_rad["cumulative_expected_energy_j"]))
    require(maximum_rad_step
            <= thresholds.maximum_radiation_step_relative_residual
            and cumulative_rad
            <= thresholds.maximum_radiation_cumulative_relative_residual
            and total_radiation_energy >= thresholds.minimum_radiation_energy_j,
            "terrain radiation energy ledger failed")
    shortwave_relative_residual = max(abs(float(
        row["shortwave_relative_power_residual"])) for row in radiation)
    maximum_shortwave_incoming = max(float(
        row["shortwave_incoming_power_w"]) for row in radiation)
    maximum_shortwave_atmospheric = max(float(
        row["shortwave_atmospheric_absorbed_power_w"])
        for row in radiation)
    maximum_shortwave_surface = max(float(
        row["shortwave_surface_absorbed_power_w"]) for row in radiation)
    require(all(
        int(row["shortwave_reflection_converged"]) == 1
        for row in radiation),
        "terrain shortwave reflection did not converge")
    require(shortwave_relative_residual
            <= thresholds.maximum_shortwave_power_relative_residual,
            "terrain shortwave power ledger failed")
    if shortwave_active:
        require(min(
            maximum_shortwave_incoming,
            maximum_shortwave_atmospheric,
            maximum_shortwave_surface) > 1.0,
            "terrain shortwave coupling was not exercised")
    else:
        require(max(
            maximum_shortwave_incoming,
            maximum_shortwave_atmospheric,
            maximum_shortwave_surface) == 0.0,
            "inactive terrain shortwave authored power")

    inventory_specs = (
        ("momentum_budget.csv",
         ("rel_momentum_x", "rel_momentum_y", "rel_momentum_z",
          "rel_angular_momentum_z"),
         thresholds.maximum_momentum_relative_drift),
        ("moisture_budget.csv", ("rel_total_water",),
         thresholds.maximum_moisture_relative_drift),
        ("energy_budget.csv", ("rel_total",),
         thresholds.maximum_energy_relative_drift),
    )
    metrics: dict[str, float] = {}
    for filename, columns, threshold in inventory_specs:
        ledger = rows(run_dir / filename)
        maximum = max(abs(float(row[column]))
                      for row in ledger for column in columns)
        require(all(int(row["warn"]) == 0 for row in ledger),
                f"{filename}: emitted a warning")
        require(maximum <= threshold,
                f"{filename}: drift exceeded the qualification bound")
        metrics[f"maximum_{filename.removesuffix('_budget.csv')}"
                "_relative_drift"] = maximum

    sources = rows(run_dir / "thermodynamic_source_budget.csv")
    stages = [
        row for row in rows(run_dir / "budget.csv")
        if row["stage"] == "microphysics"
    ]
    require(len(sources) == expected_steps and len(stages) == expected_steps,
            "coupled source ledger is incomplete")
    microphysics_species_columns = tuple(
        f"microphysics_{species}_requested_kg"
        for species in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"))
    maximum_microphysics_heat_response = max(
        abs(float(row["microphysics_heat_j"])) for row in sources)
    maximum_microphysics_species_response = max(
        sum(abs(float(row[column]))
            for column in microphysics_species_columns)
        for row in sources)
    maximum_microphysics_energy_response = max(abs(float(
        row["microphysics_moist_committed_energy_change_j"]))
        for row in sources)
    require(all(
        int(row["microphysics_conservation_active"]) == 1
        for row in sources),
        "mapped microphysics conservation report was not active")
    scalar_energy_metrics = scalar_energy_identities(
        sources, expected_steps)
    microphysics_moist_energy_metrics = (
        microphysics_moist_energy_identities(sources))
    sgs_moist_energy_metrics = sgs_moist_energy_identities(sources)
    maximum_tke_energy_response = max(
        abs(float(row["sgs_moist_energy_step_subgrid_tke_change_j"]))
        for row in sources)
    if tke_active:
        require(maximum_tke_energy_response
                >= thresholds.minimum_tke_energy_response_j,
                "mapped prognostic TKE did not produce a reservoir response")
    diffusion_energy_metrics = diffusion_energy_identities(
        sources, expected_steps, diffusion_mode, thresholds)
    pbl_work_metrics = pbl_work_identities(
        sources, expected_steps, pbl_active, thresholds)
    eos_pressure_energy_metrics = eos_pressure_energy_identities(
        sources, expected_steps)
    sponge_energy_metrics = sponge_energy_identities(
        sources, expected_steps)
    final_boundary_energy_metrics = final_boundary_energy_identities(
        sources, expected_steps)
    completed_step_energy_metrics = completed_step_energy_identities(
        sources, rows(run_dir / "energy_budget.csv"),
        list(range(1, expected_steps + 1)), require)
    slow_momentum_energy_metrics = slow_momentum_energy_identities(
        sources, expected_steps)
    kinetic_transport_metrics = kinetic_transport_identities(sources)
    require([int(row["sgs_momentum_work_step_count"]) for row in sources]
            == list(range(1, expected_steps + 1)),
            "mapped SGS work sequence changed")
    requested_sgs = [
        float(row["sgs_requested_momentum_work_j"]) for row in sources
    ]
    committed_sgs = [
        float(row["sgs_committed_momentum_work_j"]) for row in sources
    ]
    sgs_open_counts = [
        int(row["sgs_open_boundary_stability_step_count"])
        for row in sources
    ]
    require(sgs_open_counts[-1] > 0
            and all(current >= previous
                    and current - previous <= 1
                    for previous, current in zip(
                        [0, *sgs_open_counts[:-1]], sgs_open_counts)),
        "mapped SGS open-boundary provenance sequence changed")
    irreversible_sgs = [
        float(row["sgs_irreversible_dissipation_j"]) for row in sources
    ]
    maximum_sgs_stress_residual = max(
        abs(float(row["sgs_stress_closure_residual_j"])) /
        max(1.0,
            abs(float(row["sgs_open_boundary_work_j"])),
            abs(float(row["sgs_irreversible_dissipation_j"])))
        for row in sources)
    require(min(irreversible_sgs) >= 0.0
            and maximum_sgs_stress_residual <= 1.0e-12,
            "mapped SGS open-boundary dissipation provenance failed")
    cumulative_sgs = float(
        sources[-1]["sgs_cumulative_committed_momentum_work_j"])
    cumulative_irreversible_sgs = float(
        sources[-1]["sgs_cumulative_irreversible_dissipation_j"])
    require(cumulative_sgs <= -thresholds.minimum_sgs_dissipation_j
            and cumulative_irreversible_sgs
                >= thresholds.minimum_sgs_dissipation_j,
            "mapped Smagorinsky did not exercise dissipative coupling")

    heat_residual = 0.0
    water_residual = 0.0
    dry_mass_identity = 0.0
    dry_mass_cumulative_identity = 0.0
    acoustic_inventory_identity = 0.0
    acoustic_pressure_identity = 0.0
    acoustic_radiative_identity = 0.0
    acoustic_boundary_identity = 0.0
    maximum_acoustic_boundary_work = 0.0
    maximum_acoustic_radiative_work = 0.0
    require([int(row["acoustic_v1_step_count"]) for row in sources]
            == list(range(1, expected_steps + 1)),
            "mapped acoustic-energy report sequence changed")
    for stage, source in zip(stages, sources):
        require(int(source["acoustic_report_schema_version"]) == 1
                and int(source["acoustic_energy_basis_version"]) == 2,
                "mapped acoustic-energy currency version changed")
        require(int(source["acoustic_v1_substep_count"]) == 6
                and int(source["acoustic_v1_complete"]) == 1
                and int(source["acoustic_v1_boundary_application_count"]) == 7
                and int(source["acoustic_v1_boundary_complete"]) == 1,
                "mapped acoustic-energy report is partial")
        heat_scale = max(
            1.0, abs(float(stage["d_thermal_energy"])),
            abs(float(source["state_heat_change_j"])),
            *(abs(float(source[key])) for key in (
                "microphysics_heat_j", "radiation_heat_j", "pbl_heat_j",
                "theta_repair_j", "theta_storage_rounding_j",
                "theta_storage_residual_j")),
        )
        heat_residual = max(
            heat_residual,
            abs(float(stage["d_thermal_energy"])
                - float(source["state_heat_change_j"])) / heat_scale,
        )
        water_residual = max(
            water_residual,
            relative_identity(
                float(stage["d_total_water"]),
                float(source["committed_water_change_kg"])),
        )
        dry_mass_identity = max(
            dry_mass_identity,
            relative_balance(
                float(source["dynamics_committed_dry_mass_change_kg"]),
                float(source["dynamics_flux_implied_dry_mass_change_kg"]),
                float(source["dynamics_dry_mass_closure_residual_kg"])),
        )
        dry_mass_cumulative_identity = max(
            dry_mass_cumulative_identity,
            relative_balance(
                float(source[
                    "dynamics_cumulative_committed_dry_mass_change_kg"]),
                float(source[
                    "dynamics_cumulative_flux_implied_dry_mass_change_kg"]),
                float(source[
                    "dynamics_cumulative_dry_mass_closure_residual_kg"])),
        )
        acoustic_reservoir_change = sum(float(source[key]) for key in (
            "acoustic_v1_dry_internal_change_j",
            "acoustic_v1_water_sensible_change_j",
            "acoustic_v1_phase_reference_change_j",
            "acoustic_v1_carrier_kinetic_change_j",
            "acoustic_v1_gravitational_change_j",
            "acoustic_v1_subgrid_tke_change_j",
        ))
        acoustic_inventory = float(
            source["acoustic_v1_committed_inventory_change_j"])
        acoustic_pressure = float(
            source["acoustic_v1_pressure_committed_work_j"])
        acoustic_boundary = float(
            source["acoustic_v1_boundary_committed_work_j"])
        acoustic_inventory_sum = (
            acoustic_reservoir_change
            + acoustic_pressure + acoustic_boundary)
        # This identity can contain severe physical cancellation. Scale the
        # roundoff by its largest contributing reservoir, not by the small
        # cancelled net, or a sound ledger is falsely rejected.
        acoustic_inventory_scale = max(
            1.0, abs(acoustic_inventory),
            abs(acoustic_reservoir_change),
            abs(acoustic_pressure), abs(acoustic_boundary))
        acoustic_inventory_identity = max(
            acoustic_inventory_identity,
            abs(acoustic_inventory - acoustic_inventory_sum)
            / acoustic_inventory_scale,
        )
        acoustic_pressure_identity = max(
            acoustic_pressure_identity,
            relative_identity(
                float(source[
                    "acoustic_v1_pressure_committed_work_j"]),
                float(source[
                    "acoustic_v1_pressure_requested_linear_work_j"])
                + float(source[
                    "acoustic_v1_pressure_finite_step_correction_j"])
                + float(source[
                    "acoustic_v1_pressure_storage_rounding_j"])),
        )
        acoustic_radiative_identity = max(
            acoustic_radiative_identity,
            relative_identity(
                float(source[
                    "acoustic_v1_boundary_radiative_committed_work_j"]),
                float(source[
                    "acoustic_v1_boundary_radiative_requested_linear_work_j"])
                + float(source[
                    "acoustic_v1_boundary_radiative_finite_step_correction_j"])
                + float(source[
                    "acoustic_v1_boundary_radiative_storage_rounding_j"])),
        )
        acoustic_boundary_identity = max(
            acoustic_boundary_identity,
            relative_identity(
                float(source["acoustic_v1_boundary_committed_work_j"]),
                float(source[
                    "acoustic_v1_boundary_radiative_committed_work_j"])
                + float(source[
                    "acoustic_v1_boundary_constraint_history_committed_work_j"])),
        )
        maximum_acoustic_boundary_work = max(
            maximum_acoustic_boundary_work,
            abs(float(source["acoustic_v1_boundary_committed_work_j"])),
        )
        maximum_acoustic_radiative_work = max(
            maximum_acoustic_radiative_work,
            abs(float(source[
                "acoustic_v1_boundary_radiative_committed_work_j"])),
        )
    require(heat_residual <= thresholds.maximum_source_heat_relative_residual,
            "terrain-coupled heat attribution did not close")
    require(water_residual
            <= thresholds.maximum_source_water_relative_residual,
            "terrain-coupled water attribution did not close")
    require(dry_mass_identity
            <= thresholds.maximum_dry_mass_identity_relative_residual
            and dry_mass_cumulative_identity
            <= thresholds.
                maximum_cumulative_dry_mass_identity_relative_residual,
            "terrain dynamics dry-mass attribution identity failed")
    acoustic_identity = max(
        acoustic_inventory_identity,
        acoustic_pressure_identity,
        acoustic_radiative_identity,
        acoustic_boundary_identity,
    )
    require(acoustic_identity
            <= thresholds.maximum_acoustic_energy_identity_relative_residual,
            "mapped acoustic-energy attribution identity failed")
    require(maximum_acoustic_boundary_work
            >= thresholds.minimum_acoustic_boundary_work_j
            and maximum_acoustic_radiative_work
            >= thresholds.minimum_acoustic_radiative_work_j,
            "mapped acoustic boundary/radiation work was not exercised")

    first_frame = frames(run_dir)[0]
    rho_initial = read_npy_float32(first_frame.path / "rho.npy")
    z_faces = read_npy_float64(first_frame.interface_path)
    initial_dry_mass = 0.0
    for i in range(first_frame.valid_lower[0],
                   first_frame.valid_upper[0] + 1):
        for j in range(first_frame.valid_lower[1],
                       first_frame.valid_upper[1] + 1):
            for k in range(first_frame.valid_lower[2],
                           first_frame.valid_upper[2] + 1):
                volume = (first_frame.spacing_m[0]
                          * first_frame.spacing_m[1]
                          * (z_faces.at(i, j, k + 1)
                             - z_faces.at(i, j, k)))
                initial_dry_mass += rho_initial.at(i, j, k) * volume
    dry_mass_closure_fraction = (
        abs(float(sources[-1][
            "dynamics_cumulative_dry_mass_closure_residual_kg"]))
        / initial_dry_mass
    )
    require(dry_mass_closure_fraction
            <= thresholds.maximum_dynamics_dry_mass_closure_fraction,
            "terrain dynamics cumulative dry-mass storage closure failed")

    return {
        **scalar_energy_metrics,
        **microphysics_moist_energy_metrics,
        **sgs_moist_energy_metrics,
        **diffusion_energy_metrics,
        **pbl_work_metrics,
        **eos_pressure_energy_metrics,
        **sponge_energy_metrics,
        **final_boundary_energy_metrics,
        **completed_step_energy_metrics,
        **slow_momentum_energy_metrics,
        **kinetic_transport_metrics,
        **metrics,
        "maximum_radiation_step_relative_residual": maximum_rad_step,
        "maximum_shortwave_power_relative_residual":
            shortwave_relative_residual,
        "maximum_shortwave_incoming_power_w": maximum_shortwave_incoming,
        "maximum_shortwave_atmospheric_absorbed_power_w":
            maximum_shortwave_atmospheric,
        "maximum_shortwave_surface_absorbed_power_w":
            maximum_shortwave_surface,
        "radiation_cumulative_relative_residual": cumulative_rad,
        "cumulative_radiation_energy_j": total_radiation_energy,
        "maximum_source_heat_relative_residual": heat_residual,
        "maximum_source_water_relative_residual": water_residual,
        "maximum_dynamics_dry_mass_identity_residual": dry_mass_identity,
        "maximum_dynamics_cumulative_dry_mass_identity_residual":
            dry_mass_cumulative_identity,
        "dynamics_cumulative_dry_mass_closure_fraction":
            dry_mass_closure_fraction,
        "maximum_acoustic_inventory_identity_relative_residual":
            acoustic_inventory_identity,
        "maximum_acoustic_pressure_identity_relative_residual":
            acoustic_pressure_identity,
        "maximum_acoustic_radiative_identity_relative_residual":
            acoustic_radiative_identity,
        "maximum_acoustic_boundary_identity_relative_residual":
            acoustic_boundary_identity,
        "maximum_acoustic_boundary_work_j":
            maximum_acoustic_boundary_work,
        "maximum_acoustic_radiative_work_j":
            maximum_acoustic_radiative_work,
        "cumulative_sgs_committed_momentum_work_j": cumulative_sgs,
        "cumulative_sgs_irreversible_dissipation_j":
            cumulative_irreversible_sgs,
        "maximum_tke_energy_response_j": maximum_tke_energy_response,
        "maximum_microphysics_heat_response_j":
            maximum_microphysics_heat_response,
        "maximum_microphysics_species_response_kg":
            maximum_microphysics_species_response,
        "maximum_microphysics_energy_response_j":
            maximum_microphysics_energy_response,
        "maximum_sgs_stress_closure_relative_residual":
            maximum_sgs_stress_residual,
        "maximum_sgs_requested_momentum_work_j": max(requested_sgs),
        "maximum_sgs_committed_momentum_work_j": max(committed_sgs),
    }


def qualify_case(binary: Path, config: Path, case_root: Path,
                 duration_s: int, parallel_threads: int,
                 thresholds: Thresholds,
                 diffusion_mode: str = "inactive",
                 shortwave_active: bool = False,
                 pbl_active: bool = False,
                 tke_active: bool = False,
                 microphysics_control: Path | None = None,
                 chaos_control: Path | None = None,
                 diffusion_control: Path | None = None) -> dict[str, object]:
    checkpoint_s = duration_s // 2
    serial = case_root / "serial_continuous"
    serial_resumed = case_root / "serial_resumed"
    parallel = case_root / "parallel_continuous"
    parallel_resumed = case_root / "parallel_resumed"

    run(binary, config, serial, duration_s, checkpoint_s, 1)
    serial_sidecars = checkpoint_sidecars(serial)
    require(len(serial_sidecars) == 2
            and all(math.isclose(
                float(sidecar["simulation_time_s"]), expected,
                abs_tol=1.0e-8)
                for sidecar, expected in zip(
                    serial_sidecars, (checkpoint_s, duration_s))),
            "serial run did not publish midpoint/final checkpoints")
    branch = checkpoint_path(serial, serial_sidecars[0])
    run(binary, config, serial_resumed, duration_s, checkpoint_s, 1, branch)
    run(binary, config, parallel, duration_s, checkpoint_s, parallel_threads)
    run(binary, config, parallel_resumed, duration_s, checkpoint_s,
        parallel_threads, branch)

    comparisons = {
        "same_thread_restart": serial_resumed,
        "one_vs_n_thread_continuous": parallel,
        "one_to_n_thread_restart": parallel_resumed,
    }
    for label, candidate in comparisons.items():
        require_final_identity(serial, candidate, duration_s, label)
    require_ledger_identity(serial, parallel, "one-vs-N continuous")
    require_ledger_identity(
        serial_resumed, parallel_resumed, "one-vs-N resumed")
    require_restart_suffix(serial, serial_resumed, "same-thread restart")
    require_restart_suffix(parallel, parallel_resumed,
                           "one-to-N-thread restart")

    final_checkpoint = checkpoint_path(
        serial, serial_sidecars[-1]).read_bytes()
    for label, candidate in comparisons.items():
        candidate_checkpoint = checkpoint_path(
            candidate, checkpoint_sidecars(candidate)[-1]).read_bytes()
        require(final_checkpoint == candidate_checkpoint,
                f"{label}: final checkpoint differs")

    def compare_disabled_control(
            control_config: Path | None, owner: str) -> dict[str, object]:
        if control_config is None:
            return {}
        control = case_root / f"serial_{owner}_none_control"
        run(binary, control_config, control, duration_s, checkpoint_s, 1)
        reference_frame = frames(serial)[-1]
        control_frame = frames(control)[-1]
        distinct_fields = [
            field for field in CORE_FIELDS
            if terrain_field_path(reference_frame, field).read_bytes()
            != terrain_field_path(control_frame, field).read_bytes()
        ]
        require(distinct_fields,
                f"mapped {owner} matched its disabled public-state control")
        control_checkpoint = checkpoint_path(
            control, checkpoint_sidecars(control)[-1]).read_bytes()
        require(final_checkpoint != control_checkpoint,
                f"mapped {owner} matched its disabled checkpoint control")
        return {
            "disabled_control_distinct_fields": distinct_fields,
            "disabled_control_checkpoint_distinct": True,
        }

    microphysics_control_metrics = compare_disabled_control(
        microphysics_control, "microphysics")
    chaos_control_metrics = compare_disabled_control(
        chaos_control, "chaos")
    diffusion_control_metrics = compare_disabled_control(
        diffusion_control, "diffusion")

    return {
        "geometry_metrics": analyze_geometry(serial, thresholds),
        "physics_metrics": analyze_physics(
            serial, duration_s, thresholds, diffusion_mode,
            shortwave_active, pbl_active, tke_active),
        "microphysics_control_metrics": microphysics_control_metrics,
        "chaos_control_metrics": chaos_control_metrics,
        "diffusion_control_metrics": diffusion_control_metrics,
    }


def qualify(work_root: Path, duration_s: int, parallel_threads: int,
            thresholds: Thresholds = Thresholds()) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[4]
    binary = repo / "bin/nimbus_sim"
    bell_config = repo / (
        "tests/fixtures/configs/qualification/cgrid_terrain_smagorinsky_coupled.yaml")
    require(binary.is_file(), f"missing production simulator: {binary}")
    require(bell_config.is_file(),
            f"missing qualification config: {bell_config}")

    bell_text = bell_config.read_text(encoding="utf-8")
    bell_block = """terrain:
  scheme: bell
  coordinate: smoothed
  compute_derivatives: true
  compute_metrics: true
  smoothing_decay: 0.1
  bell:
    h0: 300
    a: 2000
    axisymmetric: true
"""
    schar_block = """terrain:
  scheme: schar
  coordinate: smoothed
  compute_derivatives: true
  compute_metrics: true
  smoothing_decay: 0.1
  schar:
    H: 300
    a: 5000
    ell: 4000
"""
    require(bell_text.count(bell_block) == 1,
            "bell qualification terrain block changed")
    schar_config = work_root / "cgrid_terrain_schar_coupled.yaml"
    schar_config.write_text(
        bell_text.replace(bell_block, schar_block)
        .replace("active bell terrain", "active Schär terrain"),
        encoding="utf-8",
    )
    active_scalar = (
        bell_text.replace("  K_h: 0\n", "  K_h: 80\n", 1)
        .replace("  K_v: 0\n", "  K_v: 160\n", 1)
    )
    active_diffusion = active_scalar.replace(
        "  apply_to: scalars\n", "  apply_to: scalars_and_momentum\n", 1)
    bell_scalar_config = work_root / "cgrid_terrain_bell_active_scalar.yaml"
    bell_scalar_config.write_text(active_scalar, encoding="utf-8")
    schar_scalar_config = work_root / "cgrid_terrain_schar_active_scalar.yaml"
    schar_scalar_config.write_text(
        active_scalar.replace(bell_block, schar_block)
        .replace("active bell terrain", "active Schär terrain"),
        encoding="utf-8",
    )
    bell_active_config = work_root / "cgrid_terrain_bell_active_diffusion.yaml"
    bell_active_config.write_text(active_diffusion, encoding="utf-8")
    schar_active_config = work_root / "cgrid_terrain_schar_active_diffusion.yaml"
    schar_active_config.write_text(
        active_diffusion.replace(bell_block, schar_block)
        .replace("active bell terrain", "active Schär terrain"),
        encoding="utf-8",
    )
    dynamics_block = """dynamics:
  scheme: klemp_wilhelmson_split_cartesian_cgrid
  axisymmetric: false
"""
    require(active_diffusion.count(dynamics_block) == 1,
            "terrain qualification dynamics block changed")
    weno_shortwave = (
        active_diffusion.replace(
            dynamics_block,
            dynamics_block + "\nadvection:\n  scheme: weno5\n", 1)
        .replace("  do_sw: false\n", "  do_sw: true\n", 1)
        .replace(
            "  local_solar_time_at_start_h: 0\n",
            "  local_solar_time_at_start_h: 12\n", 1)
    )
    tvd_shortwave = weno_shortwave.replace(
        "advection:\n  scheme: weno5\n",
        "advection:\n  scheme: tvd\n", 1)
    bell_tvd_shortwave_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_active_diffusion.yaml")
    bell_tvd_shortwave_config.write_text(
        tvd_shortwave, encoding="utf-8")
    schar_weno_shortwave_config = work_root / (
        "cgrid_terrain_schar_weno_shortwave_active_diffusion.yaml")
    schar_weno_shortwave_config.write_text(
        weno_shortwave.replace(bell_block, schar_block)
        .replace("active bell terrain", "active Schär terrain"),
        encoding="utf-8",
    )
    surface_none_block = """surface:
  scheme: none
  apply_fluxes: false
"""
    surface_slab_block = """surface:
  scheme: slab
  apply_fluxes: true
"""
    pbl_none_block = """boundary_layer:
  scheme: none
"""
    pbl_slab_block = """boundary_layer:
  scheme: slab
"""
    require(tvd_shortwave.count(surface_none_block) == 1
            and tvd_shortwave.count(pbl_none_block) == 1,
            "terrain qualification surface/PBL blocks changed")
    bell_pbl = (
        tvd_shortwave.replace(
            surface_none_block, surface_slab_block, 1)
        .replace(pbl_none_block, pbl_slab_block, 1)
        .replace("  mode: 3d\n", "  mode: 2d\n", 1)
        .replace("  tsk_k: 300\n", "  tsk_k: 302\n", 1)
    )
    bell_pbl_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_pbl_active_diffusion.yaml")
    bell_pbl_config.write_text(bell_pbl, encoding="utf-8")
    translating_motion = """domain_motion:
  mode: constant_velocity
  frame_velocity:
    u_ms: 5
    v_ms: -2

"""
    schar_motion = (
        weno_shortwave
        .replace(dynamics_block, translating_motion + dynamics_block, 1)
        .replace(bell_block, schar_block)
        .replace("active bell terrain", "active Schär terrain")
    )
    schar_motion_config = work_root / (
        "cgrid_terrain_schar_weno_shortwave_motion_active_diffusion.yaml")
    schar_motion_config.write_text(schar_motion, encoding="utf-8")
    schar_pbl = (
        weno_shortwave.replace(
            surface_none_block, surface_slab_block, 1)
        .replace(pbl_none_block, pbl_slab_block, 1)
        .replace("  mode: 3d\n", "  mode: 2d\n", 1)
        .replace("  tsk_k: 300\n", "  tsk_k: 302\n", 1)
        .replace(dynamics_block, translating_motion + dynamics_block, 1)
        .replace(bell_block, schar_block)
        .replace("active bell terrain", "active Schär terrain")
    )
    schar_pbl_config = work_root / (
        "cgrid_terrain_schar_weno_shortwave_pbl_active_diffusion.yaml")
    schar_pbl_config.write_text(schar_pbl, encoding="utf-8")
    pbl_ysu_block = """boundary_layer:
  scheme: ysu
"""
    pbl_mynn_block = """boundary_layer:
  scheme: mynn
"""
    bell_ysu = bell_pbl.replace(
        pbl_slab_block, pbl_ysu_block, 1)
    bell_ysu_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_ysu_active_diffusion.yaml")
    bell_ysu_config.write_text(bell_ysu, encoding="utf-8")
    schar_mynn = schar_pbl.replace(
        pbl_slab_block, pbl_mynn_block, 1)
    schar_mynn_config = work_root / (
        "cgrid_terrain_schar_weno_shortwave_mynn_active_diffusion.yaml")
    schar_mynn_config.write_text(schar_mynn, encoding="utf-8")
    bell_ysu_rrtmg = (
        bell_ysu.replace("  scheme: simple_grey\n", "  scheme: rrtmg\n", 1)
        .replace("  do_sw: true\n", "  do_sw: false\n", 1)
        .replace(
            "  clear_sky_only: true\n",
            "  clear_sky_only: false\n  rrtmg_ozone_vmr: 0.000004\n",
            1)
    )
    bell_ysu_rrtmg_config = work_root / (
        "cgrid_terrain_bell_tvd_ysu_rrtmg_lw_active_diffusion.yaml")
    bell_ysu_rrtmg_config.write_text(
        bell_ysu_rrtmg, encoding="utf-8")
    schar_mynn_rrtmg = (
        schar_mynn.replace("  scheme: simple_grey\n", "  scheme: rrtmg\n", 1)
        .replace("  do_sw: true\n", "  do_sw: false\n", 1)
        .replace(
            "  clear_sky_only: true\n",
            "  clear_sky_only: false\n  rrtmg_ozone_vmr: 0.000004\n",
            1)
    )
    schar_mynn_rrtmg_config = work_root / (
        "cgrid_terrain_schar_weno_mynn_rrtmg_lw_active_diffusion.yaml")
    schar_mynn_rrtmg_config.write_text(
        schar_mynn_rrtmg, encoding="utf-8")
    bell_tke = (
        tvd_shortwave.replace(
            "  scheme: smagorinsky\n", "  scheme: tke\n", 1)
        .replace(
            "  mode: 3d\n",
            "  mode: 3d\n  stability_correction: tke_based\n", 1)
        .replace("  Pr_t: 0.7\n", "  Pr_t: 0.3333333333333333\n", 1)
        .replace("  Sc_t: 0.7\n", "  Sc_t: 0.3333333333333333\n", 1)
    )
    bell_tke_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_tke_active_diffusion.yaml")
    bell_tke_config.write_text(bell_tke, encoding="utf-8")
    schar_tke_rrtmg = (
        schar_motion.replace(
            "  scheme: smagorinsky\n", "  scheme: tke\n", 1)
        .replace(
            "  mode: 3d\n",
            "  mode: 3d\n  stability_correction: tke_based\n", 1)
        .replace("  Pr_t: 0.7\n", "  Pr_t: 0.3333333333333333\n", 1)
        .replace("  Sc_t: 0.7\n", "  Sc_t: 0.3333333333333333\n", 1)
        .replace("  scheme: simple_grey\n", "  scheme: rrtmg\n", 1)
        .replace("  do_sw: true\n", "  do_sw: false\n", 1)
        .replace(
            "  clear_sky_only: true\n",
            "  clear_sky_only: false\n  rrtmg_ozone_vmr: 0.000004\n",
            1)
    )
    schar_tke_rrtmg_config = work_root / (
        "cgrid_terrain_schar_weno_tke_rrtmg_lw_active_diffusion.yaml")
    schar_tke_rrtmg_config.write_text(
        schar_tke_rrtmg, encoding="utf-8")
    bell_ysu_tke = (
        bell_ysu.replace(
            "  scheme: smagorinsky\n", "  scheme: tke\n", 1)
        .replace(
            "  mode: 2d\n",
            "  mode: 2d\n  stability_correction: tke_based\n", 1)
        .replace("  Pr_t: 0.7\n", "  Pr_t: 0.3333333333333333\n", 1)
        .replace("  Sc_t: 0.7\n", "  Sc_t: 0.3333333333333333\n", 1)
    )
    bell_ysu_tke_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_ysu_tke_active_diffusion.yaml")
    bell_ysu_tke_config.write_text(bell_ysu_tke, encoding="utf-8")
    schar_mynn_tke_rrtmg = (
        schar_mynn_rrtmg.replace(
            "  scheme: smagorinsky\n", "  scheme: tke\n", 1)
        .replace(
            "  mode: 2d\n",
            "  mode: 2d\n  stability_correction: tke_based\n", 1)
        .replace("  Pr_t: 0.7\n", "  Pr_t: 0.3333333333333333\n", 1)
        .replace("  Sc_t: 0.7\n", "  Sc_t: 0.3333333333333333\n", 1)
        .replace(
            "boundary_layer:\n  scheme: mynn\n",
            "boundary_layer:\n  scheme: mynn\n  pbl_top_method: tke\n",
            1)
    )
    schar_mynn_tke_rrtmg_config = work_root / (
        "cgrid_terrain_schar_weno_mynn_tke_rrtmg_lw_active_diffusion.yaml")
    schar_mynn_tke_rrtmg_config.write_text(
        schar_mynn_tke_rrtmg, encoding="utf-8")
    schar_none_rrtmg = schar_tke_rrtmg.replace(
        "microphysics:\n  scheme: kessler\n",
        "microphysics:\n  scheme: none\n", 1)
    schar_none_rrtmg_config = work_root / (
        "cgrid_terrain_schar_weno_tke_none_rrtmg_lw_active_diffusion.yaml")
    schar_none_rrtmg_config.write_text(
        schar_none_rrtmg, encoding="utf-8")
    def with_cloud_seed(config_text: str) -> str:
        dtheta = "      dtheta_k: 1\n"
        require(config_text.count(dtheta) == 1,
                "terrain qualification bubble block changed")
        return config_text.replace(
            dtheta,
            dtheta + "      cloud_water_kgkg: 0.002\n", 1)

    bell_lin_rrtmg = with_cloud_seed(bell_ysu_rrtmg).replace(
        "microphysics:\n  scheme: kessler\n",
        "microphysics:\n  scheme: lin\n", 1)
    bell_lin_rrtmg_config = work_root / (
        "cgrid_terrain_bell_tvd_ysu_lin_rrtmg_lw_active_diffusion.yaml")
    bell_lin_rrtmg_config.write_text(bell_lin_rrtmg, encoding="utf-8")
    bell_lin_none_config = work_root / (
        "cgrid_terrain_bell_tvd_ysu_none_rrtmg_lw_active_diffusion.yaml")
    bell_lin_none_config.write_text(
        bell_lin_rrtmg.replace("  scheme: lin\n", "  scheme: none\n", 1),
        encoding="utf-8")

    schar_thompson_rrtmg = with_cloud_seed(
        schar_mynn_tke_rrtmg).replace(
            "microphysics:\n  scheme: kessler\n",
            "microphysics:\n  scheme: thompson\n", 1)
    schar_thompson_rrtmg_config = work_root / (
        "cgrid_terrain_schar_weno_mynn_tke_thompson_rrtmg_lw_"
        "active_diffusion.yaml")
    schar_thompson_rrtmg_config.write_text(
        schar_thompson_rrtmg, encoding="utf-8")
    schar_thompson_none_config = work_root / (
        "cgrid_terrain_schar_weno_mynn_tke_none_rrtmg_lw_"
        "active_diffusion.yaml")
    schar_thompson_none_config.write_text(
        schar_thompson_rrtmg.replace(
            "  scheme: thompson\n", "  scheme: none\n", 1),
        encoding="utf-8")

    bell_milbrandt = with_cloud_seed(bell_ysu_tke).replace(
        "microphysics:\n  scheme: kessler\n",
        "microphysics:\n  scheme: milbrandt\n", 1)
    bell_milbrandt_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_ysu_tke_milbrandt_"
        "active_diffusion.yaml")
    bell_milbrandt_config.write_text(bell_milbrandt, encoding="utf-8")
    bell_milbrandt_none_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_ysu_tke_none_"
        "active_diffusion.yaml")
    bell_milbrandt_none_config.write_text(
        bell_milbrandt.replace(
            "  scheme: milbrandt\n", "  scheme: none\n", 1),
        encoding="utf-8")
    bell_milbrandt_cloudy_simple_grey = bell_milbrandt.replace(
        "  clear_sky_only: true\n",
        "  clear_sky_only: false\n"
        "  simple_grey_optical_model: absorbing_water_cloud_path_v1\n",
        1)
    bell_milbrandt_cloudy_simple_grey_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_ysu_tke_milbrandt_"
        "cloudy_simple_grey_active_diffusion.yaml")
    bell_milbrandt_cloudy_simple_grey_config.write_text(
        bell_milbrandt_cloudy_simple_grey, encoding="utf-8")
    bell_milbrandt_cloudy_none_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_ysu_tke_none_"
        "cloudy_simple_grey_active_diffusion.yaml")
    bell_milbrandt_cloudy_none_config.write_text(
        bell_milbrandt_cloudy_simple_grey.replace(
            "  scheme: milbrandt\n", "  scheme: none\n", 1),
        encoding="utf-8")
    def with_chaos(config_text: str, scheme: str) -> str:
        extensions = "extensions:\n  execution:\n"
        require(config_text.count(extensions) == 1,
                "terrain qualification extensions block changed")
        return config_text.replace(
            extensions,
            "extensions:\n"
            "  chaos:\n"
            f"    scheme: {scheme}\n"
            "    seed: 324\n"
            "    member_id: 2\n"
            "  execution:\n",
            1)

    bell_initial_conditions_chaos = with_chaos(
        bell_tke, "initial_conditions")
    bell_initial_conditions_chaos_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_tke_initial_conditions_chaos_"
        "active_diffusion.yaml")
    bell_initial_conditions_chaos_config.write_text(
        bell_initial_conditions_chaos, encoding="utf-8")
    bell_boundary_layer_chaos = with_chaos(
        bell_ysu_tke, "boundary_layer")
    bell_boundary_layer_chaos_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_ysu_tke_boundary_layer_chaos_"
        "active_diffusion.yaml")
    bell_boundary_layer_chaos_config.write_text(
        bell_boundary_layer_chaos, encoding="utf-8")
    schar_full_stochastic = with_chaos(
        schar_thompson_rrtmg, "full_stochastic")
    schar_full_stochastic_config = work_root / (
        "cgrid_terrain_schar_weno_mynn_tke_thompson_rrtmg_lw_"
        "full_stochastic_active_diffusion.yaml")
    schar_full_stochastic_config.write_text(
        schar_full_stochastic, encoding="utf-8")
    schar_full_stochastic_shortwave = (
        schar_full_stochastic
        .replace("  do_sw: false\n", "  do_sw: true\n", 1)
        .replace(
            "  rrtmg_ozone_vmr: 0.000004\n",
            "  rrtmg_ozone_vmr: 0.000004\n"
            "  terrain_sw_cosine_nodes: 1\n"
            "  terrain_sw_azimuth_nodes: 4\n"
            "  terrain_sw_maximum_bounces: 256\n"
            "  terrain_sw_relative_residual_tolerance: 1.0e-8\n"
            "  terrain_sw_absolute_residual_tolerance_w: 1.0e-6\n",
            1)
    )
    schar_full_stochastic_shortwave_config = work_root / (
        "cgrid_terrain_schar_weno_mynn_tke_thompson_rrtmg_sw_"
        "full_stochastic_active_diffusion.yaml")
    schar_full_stochastic_shortwave_config.write_text(
        schar_full_stochastic_shortwave, encoding="utf-8")
    def as_implicit_scalar(config_text: str) -> str:
        result = config_text
        result = result.replace(
            "diffusion:\n  scheme: explicit\n",
            "diffusion:\n  scheme: implicit\n", 1)
        result = result.replace("  K_h: 80\n", "  K_h: 8000\n", 1)
        result = result.replace("  K_v: 160\n", "  K_v: 16000\n", 1)
        require(result != config_text and
                result.count("diffusion:\n  scheme: implicit\n") == 1,
                "terrain qualification implicit diffusion transform failed")
        return result

    def explicit_vertical_control(config_text: str) -> str:
        return config_text.replace(
            "diffusion:\n  scheme: implicit\n",
            "diffusion:\n  scheme: explicit\n", 1)

    bell_cloudy_full_stochastic = with_chaos(
        bell_milbrandt_cloudy_simple_grey, "full_stochastic")
    bell_implicit = as_implicit_scalar(bell_cloudy_full_stochastic)
    bell_implicit_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_ysu_tke_milbrandt_cloudy_"
        "full_stochastic_implicit_diffusion.yaml")
    bell_implicit_config.write_text(bell_implicit, encoding="utf-8")
    bell_implicit_control_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_ysu_tke_milbrandt_cloudy_"
        "full_stochastic_explicit_vertical_diffusion.yaml")
    bell_implicit_control_config.write_text(
        explicit_vertical_control(bell_implicit), encoding="utf-8")
    bell_implicit_chaos_control_config = work_root / (
        "cgrid_terrain_bell_tvd_shortwave_ysu_tke_milbrandt_cloudy_"
        "none_chaos_implicit_diffusion.yaml")
    bell_implicit_chaos_control_config.write_text(
        bell_implicit.replace(
            "    scheme: full_stochastic\n",
            "    scheme: none\n", 1),
        encoding="utf-8")

    schar_implicit = as_implicit_scalar(schar_full_stochastic)
    schar_implicit_config = work_root / (
        "cgrid_terrain_schar_weno_mynn_tke_thompson_rrtmg_lw_"
        "full_stochastic_implicit_diffusion.yaml")
    schar_implicit_config.write_text(schar_implicit, encoding="utf-8")
    schar_implicit_control_config = work_root / (
        "cgrid_terrain_schar_weno_mynn_tke_thompson_rrtmg_lw_"
        "full_stochastic_explicit_vertical_diffusion.yaml")
    schar_implicit_control_config.write_text(
        explicit_vertical_control(schar_implicit), encoding="utf-8")
    schar_implicit_chaos_control_config = work_root / (
        "cgrid_terrain_schar_weno_mynn_tke_thompson_rrtmg_lw_"
        "none_chaos_implicit_diffusion.yaml")
    schar_implicit_chaos_control_config.write_text(
        schar_implicit.replace(
            "    scheme: full_stochastic\n",
            "    scheme: none\n", 1),
        encoding="utf-8")
    schar_implicit_microphysics_control_config = work_root / (
        "cgrid_terrain_schar_weno_mynn_tke_none_rrtmg_lw_"
        "full_stochastic_implicit_diffusion.yaml")
    schar_implicit_microphysics_control_config.write_text(
        schar_implicit.replace(
            "  scheme: thompson\n", "  scheme: none\n", 1),
        encoding="utf-8")

    cases = {
        "bell": qualify_case(
            binary, bell_config, work_root / "bell",
            duration_s, parallel_threads, thresholds),
        "schar": qualify_case(
            binary, schar_config, work_root / "schar",
            duration_s, parallel_threads, thresholds),
        "bell_active_scalar": qualify_case(
            binary, bell_scalar_config, work_root / "bell_active_scalar",
            duration_s, parallel_threads, thresholds, "scalar"),
        "schar_active_scalar": qualify_case(
            binary, schar_scalar_config, work_root / "schar_active_scalar",
            duration_s, parallel_threads, thresholds, "scalar"),
        "bell_active_diffusion": qualify_case(
            binary, bell_active_config, work_root / "bell_active_diffusion",
            duration_s, parallel_threads, thresholds, "scalar_momentum"),
        "schar_active_diffusion": qualify_case(
            binary, schar_active_config, work_root / "schar_active_diffusion",
            duration_s, parallel_threads, thresholds, "scalar_momentum"),
        "bell_tvd_shortwave_active_diffusion": qualify_case(
            binary, bell_tvd_shortwave_config,
            work_root / "bell_tvd_shortwave_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True),
        "schar_weno_shortwave_active_diffusion": qualify_case(
            binary, schar_weno_shortwave_config,
            work_root / "schar_weno_shortwave_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True),
        "schar_weno_shortwave_motion_active_diffusion": qualify_case(
            binary, schar_motion_config,
            work_root / "schar_weno_shortwave_motion_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True),
        "bell_tvd_shortwave_pbl_active_diffusion": qualify_case(
            binary, bell_pbl_config,
            work_root / "bell_tvd_shortwave_pbl_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, True),
        "schar_weno_shortwave_pbl_active_diffusion": qualify_case(
            binary, schar_pbl_config,
            work_root / "schar_weno_shortwave_pbl_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, True),
        "bell_tvd_shortwave_ysu_active_diffusion": qualify_case(
            binary, bell_ysu_config,
            work_root / "bell_tvd_shortwave_ysu_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, True),
        "schar_weno_shortwave_mynn_active_diffusion": qualify_case(
            binary, schar_mynn_config,
            work_root / "schar_weno_shortwave_mynn_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, True),
        "bell_tvd_ysu_rrtmg_lw_active_diffusion": qualify_case(
            binary, bell_ysu_rrtmg_config,
            work_root / "bell_tvd_ysu_rrtmg_lw_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", False, True),
        "schar_weno_mynn_rrtmg_lw_active_diffusion": qualify_case(
            binary, schar_mynn_rrtmg_config,
            work_root / "schar_weno_mynn_rrtmg_lw_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", False, True),
        "bell_tvd_shortwave_tke_active_diffusion": qualify_case(
            binary, bell_tke_config,
            work_root / "bell_tvd_shortwave_tke_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, False, True),
        "schar_weno_tke_rrtmg_lw_active_diffusion": qualify_case(
            binary, schar_tke_rrtmg_config,
            work_root / "schar_weno_tke_rrtmg_lw_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", False, False, True),
        "bell_tvd_shortwave_ysu_tke_active_diffusion": qualify_case(
            binary, bell_ysu_tke_config,
            work_root / "bell_tvd_shortwave_ysu_tke_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, True, True),
        "schar_weno_mynn_tke_rrtmg_lw_active_diffusion": qualify_case(
            binary, schar_mynn_tke_rrtmg_config,
            work_root / "schar_weno_mynn_tke_rrtmg_lw_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", False, True, True),
        "schar_weno_tke_none_rrtmg_lw_active_diffusion": qualify_case(
            binary, schar_none_rrtmg_config,
            work_root / "schar_weno_tke_none_rrtmg_lw_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", False, False, True),
        "bell_tvd_ysu_lin_rrtmg_lw_active_diffusion": qualify_case(
            binary, bell_lin_rrtmg_config,
            work_root / "bell_tvd_ysu_lin_rrtmg_lw_active_diffusion",
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", False, True,
            microphysics_control=bell_lin_none_config),
        "schar_weno_mynn_tke_thompson_rrtmg_lw_active_diffusion":
            qualify_case(
                binary, schar_thompson_rrtmg_config,
                work_root / (
                    "schar_weno_mynn_tke_thompson_rrtmg_lw_"
                    "active_diffusion"),
                duration_s, parallel_threads, thresholds,
                "scalar_momentum", False, True, True,
                microphysics_control=schar_thompson_none_config),
        "bell_tvd_shortwave_ysu_tke_milbrandt_active_diffusion":
            qualify_case(
                binary, bell_milbrandt_config,
                work_root / (
                    "bell_tvd_shortwave_ysu_tke_milbrandt_"
                    "active_diffusion"),
                duration_s, parallel_threads, thresholds,
                "scalar_momentum", True, True, True,
                microphysics_control=bell_milbrandt_none_config),
        "bell_tvd_shortwave_ysu_tke_milbrandt_cloudy_simple_grey_"
        "active_diffusion": qualify_case(
            binary, bell_milbrandt_cloudy_simple_grey_config,
            work_root / (
                "bell_tvd_shortwave_ysu_tke_milbrandt_"
                "cloudy_simple_grey_active_diffusion"),
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, True, True,
            microphysics_control=bell_milbrandt_cloudy_none_config),
        "bell_tvd_shortwave_tke_initial_conditions_chaos_"
        "active_diffusion": qualify_case(
            binary, bell_initial_conditions_chaos_config,
            work_root / (
                "bell_tvd_shortwave_tke_initial_conditions_chaos_"
                "active_diffusion"),
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, False, True,
            chaos_control=bell_tke_config),
        "bell_tvd_shortwave_ysu_tke_boundary_layer_chaos_"
        "active_diffusion": qualify_case(
            binary, bell_boundary_layer_chaos_config,
            work_root / (
                "bell_tvd_shortwave_ysu_tke_boundary_layer_chaos_"
                "active_diffusion"),
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, True, True,
            chaos_control=bell_ysu_tke_config),
        "schar_weno_mynn_tke_thompson_rrtmg_lw_full_stochastic_"
        "active_diffusion": qualify_case(
            binary, schar_full_stochastic_config,
            work_root / (
                "schar_weno_mynn_tke_thompson_rrtmg_lw_"
                "full_stochastic_active_diffusion"),
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", False, True, True,
            microphysics_control=schar_thompson_none_config,
            chaos_control=schar_thompson_rrtmg_config),
        "schar_weno_mynn_tke_thompson_rrtmg_sw_full_stochastic_"
        "active_diffusion": qualify_case(
            binary, schar_full_stochastic_shortwave_config,
            work_root / (
                "schar_weno_mynn_tke_thompson_rrtmg_sw_"
                "full_stochastic_active_diffusion"),
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, True, True),
        "bell_tvd_shortwave_ysu_tke_milbrandt_cloudy_full_stochastic_"
        "implicit_diffusion": qualify_case(
            binary, bell_implicit_config,
            work_root / (
                "bell_tvd_shortwave_ysu_tke_milbrandt_cloudy_"
                "full_stochastic_implicit_diffusion"),
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", True, True, True,
            chaos_control=bell_implicit_chaos_control_config,
            diffusion_control=bell_implicit_control_config),
        "schar_weno_mynn_tke_thompson_rrtmg_lw_full_stochastic_"
        "implicit_diffusion": qualify_case(
            binary, schar_implicit_config,
            work_root / (
                "schar_weno_mynn_tke_thompson_rrtmg_lw_"
                "full_stochastic_implicit_diffusion"),
            duration_s, parallel_threads, thresholds,
            "scalar_momentum", False, True, True,
            microphysics_control=schar_implicit_microphysics_control_config,
            chaos_control=schar_implicit_chaos_control_config,
            diffusion_control=schar_implicit_control_config),
    }
    comparisons = (
        "same_thread_restart",
        "one_vs_n_thread_continuous",
        "one_to_n_thread_restart",
    )
    clear_frame = frames(
        work_root /
        "bell_tvd_shortwave_ysu_tke_milbrandt_active_diffusion" /
        "serial_continuous")[-1]
    cloudy_frame = frames(
        work_root /
        "bell_tvd_shortwave_ysu_tke_milbrandt_"
        "cloudy_simple_grey_active_diffusion" /
        "serial_continuous")[-1]
    cloudy_optics_distinct_fields = [
        field for field in CORE_FIELDS
        if terrain_field_path(clear_frame, field).read_bytes()
        != terrain_field_path(cloudy_frame, field).read_bytes()
    ]
    require(cloudy_optics_distinct_fields,
            "mapped cloudy simple-grey matched its clear-sky control")
    return {
        "status": "PASS",
        "scope":
            "production_cpu_bell_and_schar_terrain_following_cartesian_"
            "cgrid_all_microphysics_smagorinsky_tke_"
            "simple_grey_lw_sw_rrtmg_lw_sw_"
            "auto_tvd_weno_"
            "constant_explicit_diffusion_optional_slab_ysu_mynn_pbl_"
            "horizontal_smagorinsky_or_tke_"
            "explicit_or_implicit_scalar_momentum_all_chaos_constant_motion",
        "duration_s_per_terrain": duration_s,
        "continuous_steps_per_terrain": int(round(duration_s / DT_S)),
        "restart_s": duration_s // 2,
        "serial_threads": 1,
        "parallel_threads": parallel_threads,
        "bitwise_final_fields": [*CORE_FIELDS, *COORDINATE_FIELDS],
        "bitwise_final_checkpoint": True,
        "bitwise_thread_ledgers": list(LEDGER_FILES),
        "exact_restart_ledger_suffixes": list(LEDGER_FILES),
        "comparisons_per_terrain": list(comparisons),
        "cloudy_simple_grey_control_distinct_fields":
            cloudy_optics_distinct_fields,
        "terrain_cases": cases,
        "thresholds": asdict(thresholds),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--parallel-threads", type=int, default=4)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.duration < 20 or args.duration % 10 != 0:
        parser.error("duration must be at least 20 s and a multiple of 10 s")
    if args.parallel_threads < 2 or args.parallel_threads > 256:
        parser.error("parallel-threads must be between 2 and 256")

    created_temp = args.work_dir is None
    work_root = args.work_dir or Path(tempfile.mkdtemp(
        prefix="nimbus-cgrid-terrain-"))
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        result = qualify(work_root, args.duration, args.parallel_threads)
    except (QualificationError, OSError,
            subprocess.CalledProcessError) as error:
        if created_temp and os.environ.get("NIMBUS_QUALIFICATION_KEEP") != "1":
            shutil.rmtree(work_root, ignore_errors=True)
        parser.exit(1, f"[cgrid-terrain] FAIL: {error}\n")

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if created_temp and os.environ.get("NIMBUS_QUALIFICATION_KEEP") == "1":
        print(f"[cgrid-terrain] retained artifacts: {work_root}")
    elif created_temp:
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

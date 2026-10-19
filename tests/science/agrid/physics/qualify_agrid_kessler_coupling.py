#!/usr/bin/env python3
"""Qualify sustained moist Kessler/radiation coupling on the CPU A-grid."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
import sys
from pathlib import Path
import shutil
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
    STEP_DIR_PATTERN,
    load_fields,
    parse_frame,
    physical_indices,
    read_json_object,
    read_npy_float32,
    require,
    valid_generation,
)


@dataclass(frozen=True)
class Thresholds:
    maximum_radiation_step_relative_residual: float = 1.0e-10
    maximum_radiation_cumulative_relative_residual: float = 1.0e-12
    maximum_momentum_relative_drift: float = 1.0e-3
    maximum_moisture_relative_drift: float = 1.0e-6
    maximum_energy_relative_drift: float = 1.0e-5
    maximum_source_water_residual_kg: float = 2.0e-6
    maximum_species_water_residual_kg: float = 1.0e-6
    maximum_microphysics_conservation_relative_residual: float = 1.0e-10
    maximum_cumulative_dynamics_dry_mass_closure_fraction: float = 3.0e-6
    maximum_acoustic_energy_identity_relative_residual: float = 2.0e-12
    minimum_rain_mixing_ratio_kgkg: float = 1.0e-8
    minimum_microphysics_heat_j: float = 1.0
    minimum_qv_response_kgkg: float = 1.0e-7
    minimum_surface_precipitation_kg: float = 1.0e-6
    minimum_acoustic_constraint_work_j: float = 1.0


CORE_FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
LEDGER_FILES = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
WATER_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
TRANSPORT_SCALARS = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh")
DT_S = 0.2


def validate_strict_summary(run_dir: Path) -> None:
    path = run_dir / "validation_summary.json"
    summary = read_json_object(path)
    require(summary.get("context") == "run_summary",
            f"{path}: wrong context")
    require(summary.get("guard_mode") == "strict",
            f"{path}: strict guard was not active")
    require(summary.get("failed") is False,
            f"{path}: validation summary failed")
    for key in (
        "failed_report_count", "total_violations", "total_nonfinite",
        "total_bounds", "total_sanitized_nonfinite",
        "total_sanitized_bounds",
    ):
        require(summary.get(key) == 0, f"{path}: {key} must be zero")


def run(binary: Path, config: Path, outdir: Path, duration_s: int,
        checkpoint_s: int, threads: int,
        resume: Path | None = None) -> None:
    command = [
        str(binary), "--headless", f"--config={config}",
        f"--duration={duration_s}", f"--write-every={duration_s}",
        f"--outdir={outdir}", f"--checkpoint-every={checkpoint_s}",
        "--compute-backend=cpu", "--guard-mode=strict",
        "--log-profile=quiet",
    ]
    if resume is not None:
        command.append(f"--resume={resume}")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    subprocess.run(command, check=True, env=environment)


def frames(run_dir: Path):
    marker = read_json_object(run_dir / "nimbus_output.json")
    require(marker.get("schema") == "nimbus.output",
            f"{run_dir}: wrong output schema")
    require(marker.get("publication_state") == "ready",
            f"{run_dir}: output is not ready")
    generation = marker.get("publication_generation")
    require(valid_generation(generation), f"{run_dir}: invalid generation")
    assert isinstance(generation, str)
    result = sorted(
        (parse_frame(path, generation) for path in run_dir.iterdir()
         if path.is_dir() and STEP_DIR_PATTERN.fullmatch(path.name)),
        key=lambda frame: frame.time_s,
    )
    validate_strict_summary(run_dir)
    return result


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
    paths = sorted((run_dir / "checkpoints").glob("*.nimbus.json"))
    result = [read_json_object(path) for path in paths]
    require(result, f"{run_dir}: no checkpoints")
    require(all(item.get("boundary") == "completed_timestep" for item in result),
            f"{run_dir}: checkpoint outside completed timestep")
    return result


def checkpoint_path(run_dir: Path, sidecar: dict[str, object]) -> Path:
    filename = sidecar.get("checkpoint_file")
    require(isinstance(filename, str) and filename,
            f"{run_dir}: malformed checkpoint sidecar")
    result = run_dir / "checkpoints" / filename
    require(result.is_file(), f"{run_dir}: missing checkpoint {filename}")
    return result


def require_final_identity(reference_dir: Path, candidate_dir: Path,
                           duration_s: int, label: str) -> None:
    reference = frames(reference_dir)
    candidate = frames(candidate_dir)
    require(reference and candidate, f"{label}: missing output frames")
    require(math.isclose(reference[-1].time_s, duration_s, abs_tol=1.0e-8)
            and math.isclose(candidate[-1].time_s, duration_s, abs_tol=1.0e-8),
            f"{label}: wrong final output time")
    for field in CORE_FIELDS:
        require((reference[-1].path / f"{field}.npy").read_bytes()
                == (candidate[-1].path / f"{field}.npy").read_bytes(),
                f"{label}: final {field} bytes differ")


def require_ledger_identity(reference_dir: Path, candidate_dir: Path,
                            label: str) -> None:
    for filename in LEDGER_FILES:
        require((reference_dir / filename).read_bytes()
                == (candidate_dir / filename).read_bytes(),
                f"{label}: {filename} bytes differ")


def require_restart_ledger_suffix(continuous_dir: Path, resumed_dir: Path,
                                  label: str) -> None:
    for filename in LEDGER_FILES:
        continuous = (continuous_dir / filename).read_text(
            encoding="utf-8").splitlines()
        resumed = (resumed_dir / filename).read_text(
            encoding="utf-8").splitlines()
        require(continuous and resumed and continuous[0] == resumed[0],
                f"{label}: {filename} header changed")
        require(len(resumed) > 1,
                f"{label}: {filename} has no resumed diagnostic rows")
        require(continuous[-(len(resumed) - 1):] == resumed[1:],
                f"{label}: {filename} is not an exact restart suffix")


def relative_identity(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))


def microphysics_moist_energy_identities(
        sources: list[dict[str, str]]) -> dict[str, float]:
    maximum = 0.0
    for row in sources:
        require(
            int(row["microphysics_moist_energy_report_schema_version"]) == 2
            and int(row["microphysics_moist_energy_basis_version"]) == 2
            and int(row["microphysics_moist_energy_complete"]) == 1,
            "microphysics moist-energy currency is partial or changed")
        committed = float(
            row["microphysics_moist_committed_energy_change_j"])
        reservoirs = sum(float(row[key]) for key in (
            "microphysics_moist_dry_internal_change_j",
            "microphysics_moist_water_sensible_change_j",
            "microphysics_moist_phase_reference_change_j",
            "microphysics_moist_carrier_kinetic_change_j",
            "microphysics_moist_gravitational_change_j",
            "microphysics_moist_subgrid_tke_change_j",
        ))
        closure = float(
            row["microphysics_moist_energy_closure_residual_j"])
        maximum = max(
            maximum,
            relative_identity(committed, reservoirs + closure))
    require(maximum <= 1.0e-14,
            "microphysics moist-energy reservoir identity does not close")
    return {
        "maximum_microphysics_moist_energy_identity_relative_residual":
            maximum,
    }


def sgs_moist_energy_identities(
        sources: list[dict[str, str]]) -> dict[str, float]:
    if not sources or max(
            int(row["sgs_momentum_work_step_count"]) for row in sources) == 0:
        return {"maximum_sgs_moist_energy_identity_relative_residual": 0.0}
    maximum = 0.0
    cumulative_maximum = 0.0
    for row in sources:
        require(
            int(row["sgs_moist_energy_report_schema_version"]) == 1
            and int(row["sgs_moist_energy_basis_version"]) == 2
            and int(row["sgs_moist_energy_step_complete"]) == 1
            and int(row["sgs_moist_energy_cumulative_complete"]) == 1,
            "SGS moist-energy currency is partial or changed")
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
            "SGS moist-energy reservoir identity does not close")
    return {
        "maximum_sgs_moist_energy_identity_relative_residual": maximum,
        "maximum_cumulative_sgs_moist_energy_identity_relative_residual":
            cumulative_maximum,
    }


def eos_pressure_energy_identities(
        sources: list[dict[str, str]],
        expected_steps: int) -> dict[str, float]:
    require([int(row["eos_pressure_energy_step_count"]) for row in sources]
            == list(range(1, expected_steps + 1)),
            "EOS pressure-energy report sequence changed")
    maximum = 0.0
    cumulative_maximum = 0.0
    for row in sources:
        require(
            int(row["eos_pressure_energy_report_schema_version"]) == 1
            and int(row["eos_pressure_energy_basis_version"]) == 2
            and int(row["eos_pressure_energy_step_complete"]) == 1
            and int(row["eos_pressure_energy_cumulative_complete"]) == 1,
            "EOS pressure-energy currency is partial or changed")
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
            "EOS pressure-energy reservoir identity does not close")
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
            "sponge-energy report sequence changed")
    energy_maxima = {"step": 0.0, "cumulative": 0.0}
    momentum_maxima = {"step": 0.0, "cumulative": 0.0}
    for row in sources:
        require(
            int(row["sponge_energy_report_schema_version"]) == 1
            and int(row["sponge_energy_basis_version"]) == 2
            and int(row["sponge_energy_step_complete"]) == 1
            and int(row["sponge_energy_cumulative_complete"]) == 1,
            "sponge-energy currency is partial or changed")
        for cadence in ("step", "cumulative"):
            for pass_name in ("upper", "w_only", "lateral"):
                prefix = f"sponge_energy_{cadence}_{pass_name}_"
                require(int(row[prefix + "complete"]) == 1,
                        f"{pass_name} sponge report is incomplete")
                carrier = float(row[prefix + "carrier_kinetic_change_j"])
                requested = float(
                    row[prefix + "momentum_requested_linear_work_j"])
                finite = float(
                    row[prefix + "momentum_finite_step_correction_j"])
                storage = float(
                    row[prefix + "momentum_storage_rounding_j"])
                momentum_maxima[cadence] = max(
                    momentum_maxima[cadence],
                    relative_identity(
                        carrier, requested + finite + storage))
                reservoirs = sum(float(row[prefix + suffix])
                                 for suffix in (
                                     "dry_internal_change_j",
                                     "water_sensible_change_j",
                                     "phase_reference_change_j",
                                     "carrier_kinetic_change_j",
                                     "gravitational_change_j",
                                     "subgrid_tke_change_j"))
                energy_maxima[cadence] = max(
                    energy_maxima[cadence],
                    relative_identity(
                        float(row[
                            prefix + "committed_energy_change_j"]),
                        reservoirs
                        + float(row[prefix + "closure_residual_j"])))
    require(max(*energy_maxima.values(), *momentum_maxima.values())
            <= 1.0e-14,
            "sponge operator-time energy identity does not close")
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
            "final-boundary energy report sequence changed")
    maxima = {"step": 0.0, "cumulative": 0.0}
    for row in sources:
        require(
            int(row["final_boundary_energy_report_schema_version"]) == 1
            and int(row["final_boundary_energy_basis_version"]) == 2,
            "final-boundary energy currency changed")
        for cadence in ("step", "cumulative"):
            prefix = f"final_boundary_energy_{cadence}_"
            require(
                int(row[prefix + "applied"]) == 1
                and int(row[prefix + "complete"]) == 1,
                "final-boundary energy report is incomplete")
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
            "final-boundary operator-time energy identity does not close")
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
        "scalar moist-energy report sequence changed")
    maximum_reservoir = 0.0
    maximum_face = 0.0
    maximum_requested = 0.0
    maximum_commit = 0.0
    for row in sources:
        require(
            int(row["advection_scalar_energy_report_schema_version"]) == 1
            and int(row["advection_scalar_energy_basis_version"]) == 2
            and int(row[
                "advection_thermodynamic_moist_energy_reports_complete"]) == 1,
            "scalar moist-energy currency is partial or changed")
        for scalar in TRANSPORT_SCALARS:
            for cadence in ("step", "cumulative"):
                prefix = f"advection_{scalar}_energy_{cadence}_"
                committed = float(row[prefix + "committed_state_change_j"])
                reservoirs = sum(float(row[prefix + key]) for key in (
                    "dry_internal_change_j",
                    "water_sensible_change_j",
                    "phase_reference_change_j",
                    "carrier_kinetic_change_j",
                    "gravitational_change_j",
                    "subgrid_tke_change_j",
                ))
                limited = float(
                    row[prefix + "limited_face_state_change_j"])
                outward = float(row[prefix + "outward_boundary_j"])
                conversion = float(row[prefix + "interior_conversion_j"])
                requested = float(row[prefix + "requested_state_change_j"])
                remap = float(row[prefix + "density_history_remap_j"])
                storage = float(row[prefix + "storage_rounding_j"])
                closure = float(row[prefix + "closure_residual_j"])
                maximum_reservoir = max(
                    maximum_reservoir,
                    relative_identity(committed, reservoirs))
                maximum_face = max(
                    maximum_face,
                    relative_identity(limited, -outward + conversion))
                maximum_requested = max(
                    maximum_requested,
                    relative_identity(requested, remap + limited))
                maximum_commit = max(
                    maximum_commit,
                    relative_identity(
                        committed, requested + storage + closure))
    require(max(
        maximum_reservoir, maximum_face,
        maximum_requested, maximum_commit) <= 1.0e-14,
        "scalar moist-energy discrete identities do not close")
    return {
        "maximum_scalar_energy_reservoir_identity_relative_residual":
            maximum_reservoir,
        "maximum_scalar_energy_face_identity_relative_residual":
            maximum_face,
        "maximum_scalar_energy_requested_identity_relative_residual":
            maximum_requested,
        "maximum_scalar_energy_commit_identity_relative_residual":
            maximum_commit,
    }


def slow_momentum_energy_identities(
        sources: list[dict[str, str]],
        expected_steps: int) -> dict[str, float]:
    require([
        int(row["slow_momentum_energy_step_count"])
        for row in sources
    ] == list(range(1, expected_steps + 1)),
        "slow momentum energy report sequence changed")
    maximum = 0.0
    pressure_maximum = 0.0
    for row in sources:
        require(
            int(row["slow_momentum_report_schema_version"]) == 2
            and int(row["slow_momentum_energy_basis_version"]) == 2,
            "slow momentum energy currency version changed")
        for cadence in ("step", "cumulative"):
            prefix = f"slow_momentum_energy_{cadence}_"
            require(int(row[prefix + "complete"]) == 1,
                    "slow momentum energy report is partial")
            advection_finite = float(
                row[prefix + "advection_finite_step_correction_j"])
            nonadvection_finite = float(
                row[prefix + "nonadvection_finite_step_correction_j"])
            require(advection_finite >= 0.0 and nonadvection_finite >= 0.0,
                    "slow momentum finite-step correction is invalid")
            committed = float(
                row[prefix + "committed_kinetic_change_j"])
            attributed = sum(float(row[prefix + key]) for key in (
                "advection_requested_linear_work_j",
                "nonadvection_requested_linear_work_j",
                "advection_finite_step_correction_j",
                "nonadvection_finite_step_correction_j",
                "coupled_finite_step_cross_work_j",
                "storage_rounding_j",
                "closure_residual_j",
            ))
            maximum = max(
                maximum, relative_identity(committed, attributed))
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
            "slow momentum discrete kinetic-work identity does not close")
    require(pressure_maximum <= 1.0e-14,
            "slow pressure reservoir identity does not close")
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
                    "kinetic transport report is partial")
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
        "kinetic transport discrete identities do not close")
    return {
        "maximum_continuity_carrier_identity_relative_residual":
            continuity_maximum,
        "maximum_kinetic_transport_identity_relative_residual":
            combined_maximum,
        "maximum_kinetic_transport_momentum_copy_relative_residual":
            copied_momentum_maximum,
    }


def analyze_physics(run_dir: Path, duration_s: int,
                    thresholds: Thresholds) -> dict[str, float]:
    run_frames = frames(run_dir)
    require([frame.time_s for frame in run_frames] == [0.0, float(duration_s)],
            f"{run_dir}: expected initial and final frames")
    initial = load_fields(run_frames[0])
    final = load_fields(run_frames[-1])
    final["qr"] = read_npy_float32(run_frames[-1].path / "qr.npy")
    indices = list(physical_indices(run_frames[-1]))
    require(indices, f"{run_dir}: empty physical domain")

    maximum_rain = max(final["qr"].at(*index) for index in indices)
    qv_response = max(abs(final["qv"].at(*index)
                          - initial["qv"].at(*index)) for index in indices)
    require(maximum_rain >= thresholds.minimum_rain_mixing_ratio_kgkg,
            "Kessler produced no resolved rainwater")
    require(qv_response >= thresholds.minimum_qv_response_kgkg,
            "Kessler/radiation coupling produced no resolved vapor response")

    radiation = rows(run_dir / "radiation_budget.csv")
    expected_steps = int(round(duration_s / DT_S))
    require([int(row["step"]) for row in radiation]
            == list(range(1, expected_steps + 1)),
            "radiation step sequence is discontinuous")
    samples: list[float] = []
    for row in radiation:
        sample = float(row["last_radiation_sample_s"])
        if not samples or sample != samples[-1]:
            samples.append(sample)
    expected_samples = [float(value) for value in range(0, duration_s, 10)]
    require(len(samples) == len(expected_samples) and all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-8)
        for actual, expected in zip(samples, expected_samples)),
        f"radiation cadence drifted: {samples}")
    maximum_rad_step = max(float(row["relative_booking_residual"])
                           for row in radiation)
    last_radiation = radiation[-1]
    cumulative_rad = abs(float(last_radiation[
        "cumulative_booking_residual_j"])) / max(
            1.0, abs(float(last_radiation["cumulative_expected_energy_j"])))
    require(maximum_rad_step
            <= thresholds.maximum_radiation_step_relative_residual,
            "radiation step power ledger failed")
    require(cumulative_rad
            <= thresholds.maximum_radiation_cumulative_relative_residual,
            "radiation cumulative power ledger failed")

    momentum = rows(run_dir / "momentum_budget.csv")
    moisture = rows(run_dir / "moisture_budget.csv")
    energy = rows(run_dir / "energy_budget.csv")
    maximum_momentum = max(
        abs(float(row[name])) for row in momentum
        for name in ("rel_momentum_x", "rel_momentum_y", "rel_momentum_z",
                     "rel_angular_momentum_z"))
    maximum_moisture = max(abs(float(row["rel_total_water"]))
                           for row in moisture)
    maximum_energy = max(abs(float(row["rel_total"])) for row in energy)
    require(all(int(row["warn"]) == 0
                for row in momentum + moisture + energy),
            "coupled inventory ledger emitted a warning")
    require(maximum_momentum <= thresholds.maximum_momentum_relative_drift,
            "coupled momentum drift exceeded its bound")
    require(maximum_moisture <= thresholds.maximum_moisture_relative_drift,
            "coupled water drift exceeded its bound")
    require(maximum_energy <= thresholds.maximum_energy_relative_drift,
            "coupled energy drift exceeded its bound")

    sources = rows(run_dir / "thermodynamic_source_budget.csv")
    stages = rows(run_dir / "budget.csv")
    micro_stages = [row for row in stages if row["stage"] == "microphysics"]
    require(len(sources) == expected_steps and len(micro_stages) == expected_steps,
            "thermodynamic source ledger is incomplete")
    scalar_energy_metrics = scalar_energy_identities(
        sources, expected_steps)
    microphysics_moist_energy_metrics = (
        microphysics_moist_energy_identities(sources))
    sgs_moist_energy_metrics = sgs_moist_energy_identities(sources)
    eos_pressure_energy_metrics = eos_pressure_energy_identities(
        sources, expected_steps)
    sponge_energy_metrics = sponge_energy_identities(
        sources, expected_steps)
    final_boundary_energy_metrics = final_boundary_energy_identities(
        sources, expected_steps)
    completed_step_energy_metrics = completed_step_energy_identities(
        sources, energy, list(range(1, expected_steps + 1)), require)
    slow_momentum_energy_metrics = slow_momentum_energy_identities(
        sources, expected_steps)
    kinetic_transport_metrics = kinetic_transport_identities(sources)
    require(all(int(row["microphysics_conservation_active"]) == 1
                for row in sources),
            "microphysics conservation attribution was not active")
    require([int(row["dynamics_boundary_mass_step_count"]) for row in sources]
            == list(range(1, expected_steps + 1)),
            "dynamics boundary-mass step sequence is discontinuous")

    maximum_dynamics_identity = max(
        relative_identity(
            float(row["dynamics_committed_dry_mass_change_kg"]),
            float(row["dynamics_flux_implied_dry_mass_change_kg"])
            + float(row["dynamics_dry_mass_closure_residual_kg"]))
        for row in sources)
    maximum_dynamics_cumulative_identity = max(
        relative_identity(
            float(row["dynamics_cumulative_committed_dry_mass_change_kg"]),
            float(row[
                "dynamics_cumulative_flux_implied_dry_mass_change_kg"])
            + float(row[
                "dynamics_cumulative_dry_mass_closure_residual_kg"]))
        for row in sources)
    maximum_closed_boundary_export = max(
        abs(float(row["dynamics_outward_boundary_dry_mass_kg"]))
        for row in sources)
    initial_dry_mass = (
        sum(float(initial["rho"].at(*index)) for index in indices)
        * run_frames[0].dx_m * run_frames[0].dy_m * run_frames[0].dz_m)
    dynamics_closure_fraction = max(
        abs(float(row[
            "dynamics_cumulative_dry_mass_closure_residual_kg"]))
        / initial_dry_mass for row in sources)
    require(maximum_dynamics_identity <= 1.0e-15,
            "per-step dynamics dry-mass attribution does not close")
    require(maximum_dynamics_cumulative_identity <= 1.0e-15,
            "cumulative dynamics dry-mass attribution does not close")
    require(maximum_closed_boundary_export == 0.0,
            "periodic/rigid dynamics authored a boundary dry-mass flux")
    require(dynamics_closure_fraction <= thresholds.
            maximum_cumulative_dynamics_dry_mass_closure_fraction,
            "cumulative dynamics dry-mass storage closure exceeded its bound")

    acoustic_inventory_identity = 0.0
    acoustic_pressure_identity = 0.0
    acoustic_radiative_identity = 0.0
    acoustic_boundary_identity = 0.0
    maximum_acoustic_constraint_work = 0.0
    require([int(row["acoustic_v1_step_count"]) for row in sources]
            == list(range(1, expected_steps + 1)),
            "A-grid acoustic-energy report sequence changed")
    for source in sources:
        require(int(source["acoustic_report_schema_version"]) == 1
                and int(source["acoustic_energy_basis_version"]) == 2,
                "A-grid acoustic-energy currency version changed")
        require(int(source["acoustic_v1_substep_count"]) == 6
                and int(source["acoustic_v1_complete"]) == 1
                and int(source["acoustic_v1_boundary_application_count"]) == 7
                and int(source["acoustic_v1_boundary_complete"]) == 1,
                "A-grid acoustic-energy report is partial")

        reservoir_change = sum(float(source[key]) for key in (
            "acoustic_v1_dry_internal_change_j",
            "acoustic_v1_water_sensible_change_j",
            "acoustic_v1_phase_reference_change_j",
            "acoustic_v1_carrier_kinetic_change_j",
            "acoustic_v1_gravitational_change_j",
            "acoustic_v1_subgrid_tke_change_j",
        ))
        inventory = float(
            source["acoustic_v1_committed_inventory_change_j"])
        pressure = float(
            source["acoustic_v1_pressure_committed_work_j"])
        boundary = float(
            source["acoustic_v1_boundary_committed_work_j"])
        inventory_scale = max(
            1.0, abs(inventory), abs(reservoir_change),
            abs(pressure), abs(boundary))
        acoustic_inventory_identity = max(
            acoustic_inventory_identity,
            abs(inventory - reservoir_change - pressure - boundary)
            / inventory_scale,
        )
        acoustic_pressure_identity = max(
            acoustic_pressure_identity,
            relative_identity(
                pressure,
                float(source[
                    "acoustic_v1_pressure_requested_linear_work_j"])
                + float(source[
                    "acoustic_v1_pressure_finite_step_correction_j"])
                + float(source[
                    "acoustic_v1_pressure_storage_rounding_j"])),
        )
        radiative = float(
            source["acoustic_v1_boundary_radiative_committed_work_j"])
        acoustic_radiative_identity = max(
            acoustic_radiative_identity,
            relative_identity(
                radiative,
                float(source[
                    "acoustic_v1_boundary_radiative_requested_linear_work_j"])
                + float(source[
                    "acoustic_v1_boundary_radiative_finite_step_correction_j"])
                + float(source[
                    "acoustic_v1_boundary_radiative_storage_rounding_j"])),
        )
        constraint = float(source[
            "acoustic_v1_boundary_constraint_history_committed_work_j"])
        acoustic_boundary_identity = max(
            acoustic_boundary_identity,
            relative_identity(boundary, radiative + constraint),
        )
        require(radiative == 0.0,
                "collocated halo closure fabricated radiative work")
        maximum_acoustic_constraint_work = max(
            maximum_acoustic_constraint_work, abs(constraint))

    acoustic_identity = max(
        acoustic_inventory_identity,
        acoustic_pressure_identity,
        acoustic_radiative_identity,
        acoustic_boundary_identity,
    )
    require(acoustic_identity
            <= thresholds.maximum_acoustic_energy_identity_relative_residual,
            "A-grid acoustic-energy attribution identity failed")
    cumulative_constraint = float(sources[-1][
        "acoustic_v1_cumulative_boundary_constraint_history_committed_work_j"])
    require(maximum_acoustic_constraint_work
            >= thresholds.minimum_acoustic_constraint_work_j
            and cumulative_constraint < 0.0,
            "A-grid null-mode projection work was not exercised")

    maximum_source_water = max(
        abs(float(stage["d_total_water"])
            - float(source["committed_water_change_kg"]))
        for stage, source in zip(micro_stages, sources))
    maximum_species_water = max(
        abs(float(stage[f"d_{species}_mass"])
            - float(source[f"{species}_committed_change_kg"]))
        for stage, source in zip(micro_stages, sources)
        for species in WATER_SPECIES)
    maximum_requested_sum = max(
        relative_identity(
            sum(float(source[f"microphysics_{species}_requested_kg"])
                for species in WATER_SPECIES),
            float(source["microphysics_water_tendency_kg"]))
        for source in sources)
    maximum_committed_sum = max(
        relative_identity(
            sum(float(source[f"{species}_committed_change_kg"])
                for species in WATER_SPECIES),
            float(source["committed_water_change_kg"]))
        for source in sources)
    maximum_micro_conservation = max(
        abs(float(source["microphysics_conservation_residual_kg"]))
        / max(1.0, abs(float(source["microphysics_baseline_water_kg"])))
        for source in sources)
    maximum_microphysics_heat = max(
        abs(float(source["microphysics_heat_j"])) for source in sources)
    cumulative_precipitation = sum(
        float(source["surface_precipitation_kg"]) for source in sources)

    require(maximum_source_water <= thresholds.maximum_source_water_residual_kg,
            "committed total water does not close against its stage ledger")
    require(maximum_species_water
            <= thresholds.maximum_species_water_residual_kg,
            "a committed water species does not close against its stage ledger")
    require(maximum_requested_sum <= 1.0e-10,
            "requested per-species water does not reproduce its total")
    require(maximum_committed_sum <= 1.0e-10,
            "committed per-species water does not reproduce its total")
    require(maximum_micro_conservation <= thresholds.
            maximum_microphysics_conservation_relative_residual,
            "Kessler process water does not close with precipitation")
    require(maximum_microphysics_heat >= thresholds.minimum_microphysics_heat_j,
            "Kessler produced no latent-heating signal")
    require(cumulative_precipitation
            >= thresholds.minimum_surface_precipitation_kg,
            "Kessler did not exercise physical-surface precipitation fallout")

    return {
        **scalar_energy_metrics,
        **microphysics_moist_energy_metrics,
        **sgs_moist_energy_metrics,
        **eos_pressure_energy_metrics,
        **sponge_energy_metrics,
        **final_boundary_energy_metrics,
        **completed_step_energy_metrics,
        **slow_momentum_energy_metrics,
        **kinetic_transport_metrics,
        "maximum_rain_mixing_ratio_kgkg": maximum_rain,
        "maximum_qv_response_kgkg": qv_response,
        "maximum_microphysics_heat_j": maximum_microphysics_heat,
        "cumulative_surface_precipitation_kg": cumulative_precipitation,
        "maximum_radiation_step_relative_residual": maximum_rad_step,
        "radiation_cumulative_relative_residual": cumulative_rad,
        "maximum_momentum_relative_drift": maximum_momentum,
        "maximum_moisture_relative_drift": maximum_moisture,
        "maximum_energy_relative_drift": maximum_energy,
        "maximum_source_water_residual_kg": maximum_source_water,
        "maximum_species_water_residual_kg": maximum_species_water,
        "maximum_species_requested_sum_relative_residual":
            maximum_requested_sum,
        "maximum_species_committed_sum_relative_residual":
            maximum_committed_sum,
        "maximum_microphysics_conservation_relative_residual":
            maximum_micro_conservation,
        "maximum_dynamics_dry_mass_identity_residual":
            maximum_dynamics_identity,
        "maximum_dynamics_cumulative_dry_mass_identity_residual":
            maximum_dynamics_cumulative_identity,
        "maximum_closed_boundary_dry_mass_export_kg":
            maximum_closed_boundary_export,
        "maximum_cumulative_dynamics_dry_mass_closure_fraction":
            dynamics_closure_fraction,
        "maximum_acoustic_energy_identity_relative_residual":
            acoustic_identity,
        "maximum_acoustic_inventory_identity_relative_residual":
            acoustic_inventory_identity,
        "maximum_acoustic_pressure_identity_relative_residual":
            acoustic_pressure_identity,
        "maximum_acoustic_radiative_identity_relative_residual":
            acoustic_radiative_identity,
        "maximum_acoustic_boundary_identity_relative_residual":
            acoustic_boundary_identity,
        "maximum_acoustic_constraint_work_j":
            maximum_acoustic_constraint_work,
        "cumulative_acoustic_constraint_work_j":
            cumulative_constraint,
    }


def qualify(work_root: Path, duration_s: int, parallel_threads: int,
            thresholds: Thresholds = Thresholds(),
            binary_override: Path | None = None) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[4]
    binary = (binary_override or repo / "bin/nimbus_sim").resolve()
    config = repo / "tests/fixtures/configs/qualification/agrid_kessler_radiation_coupled.yaml"
    require(binary.is_file(), f"missing simulator: {binary}")
    require(config.is_file(), f"missing config: {config}")

    checkpoint_s = duration_s // 2
    serial = work_root / "serial_continuous"
    serial_resumed = work_root / "serial_resumed"
    parallel = work_root / "parallel_continuous"
    parallel_resumed = work_root / "parallel_resumed"
    run(binary, config, serial, duration_s, checkpoint_s, 1)
    serial_sidecars = checkpoint_sidecars(serial)
    checkpoint_times = [
        float(item["simulation_time_s"]) for item in serial_sidecars
    ]
    require(len(checkpoint_times) == 2 and all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-8)
        for actual, expected in zip(
            checkpoint_times, (checkpoint_s, duration_s))),
            f"{serial}: expected midpoint and final checkpoints")
    branch_checkpoint = checkpoint_path(serial, serial_sidecars[0])
    run(binary, config, serial_resumed, duration_s, checkpoint_s, 1,
        branch_checkpoint)
    run(binary, config, parallel, duration_s, checkpoint_s, parallel_threads)
    run(binary, config, parallel_resumed, duration_s, checkpoint_s,
        parallel_threads, branch_checkpoint)

    comparisons = {
        "same_thread_restart": serial_resumed,
        "one_vs_n_thread_continuous": parallel,
        "one_to_n_thread_restart": parallel_resumed,
    }
    for label, candidate in comparisons.items():
        require_final_identity(serial, candidate, duration_s, label)
    require_ledger_identity(serial, parallel, "one-vs-N-thread continuous")
    require_ledger_identity(serial_resumed, parallel_resumed,
                            "one-vs-N-thread resumed")
    require_restart_ledger_suffix(serial, serial_resumed,
                                  "same-thread restart")
    require_restart_ledger_suffix(parallel, parallel_resumed,
                                  "one-to-N-thread restart")

    final_serial_checkpoint = checkpoint_path(
        serial, checkpoint_sidecars(serial)[-1]).read_bytes()
    for label, candidate in comparisons.items():
        require(final_serial_checkpoint == checkpoint_path(
            candidate, checkpoint_sidecars(candidate)[-1]).read_bytes(),
            f"{label}: final checkpoint bytes differ")

    metrics = analyze_physics(serial, duration_s, thresholds)
    return {
        "status": "PASS",
        "scope": "cpu_flat_cartesian_agrid_kessler_simple_grey",
        "duration_s": duration_s,
        "continuous_steps": int(round(duration_s / DT_S)),
        "restart_s": checkpoint_s,
        "serial_threads": 1,
        "parallel_threads": parallel_threads,
        "bitwise_final_fields": list(CORE_FIELDS),
        "bitwise_final_checkpoint": True,
        "bitwise_thread_ledgers": list(LEDGER_FILES),
        "exact_restart_ledger_suffixes": list(LEDGER_FILES),
        "comparisons": list(comparisons),
        "metrics": metrics,
        "thresholds": asdict(thresholds),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--parallel-threads", type=int, default=4)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.duration < 20 or args.duration % 20 != 0:
        parser.error("duration must be a positive multiple of 20 s")
    if args.parallel_threads < 2 or args.parallel_threads > 256:
        parser.error("parallel-threads must be between 2 and 256")

    created_temp = args.work_dir is None
    work_root = args.work_dir or Path(tempfile.mkdtemp(
        prefix="nimbus-agrid-kessler-"))
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        result = qualify(
            work_root, args.duration, args.parallel_threads,
            binary_override=args.binary)
    except (QualificationError, OSError,
            subprocess.CalledProcessError) as error:
        if created_temp and os.environ.get("NIMBUS_QUALIFICATION_KEEP") != "1":
            shutil.rmtree(work_root, ignore_errors=True)
        parser.exit(1, f"[agrid-kessler] FAIL: {error}\n")

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if created_temp and os.environ.get("NIMBUS_QUALIFICATION_KEEP") == "1":
        print(f"[agrid-kessler] retained artifacts: {work_root}")
    elif created_temp:
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

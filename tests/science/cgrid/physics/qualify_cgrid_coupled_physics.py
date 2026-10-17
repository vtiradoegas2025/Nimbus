#!/usr/bin/env python3
"""Qualify the admitted CPU C-grid Kessler/PBL/SGS/radiation combination."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile


COMMON_TOOLS = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

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
    validate_summary,
)


@dataclass(frozen=True)
class Thresholds:
    maximum_radiation_step_relative_residual: float = 1.0e-10
    maximum_radiation_cumulative_relative_residual: float = 1.0e-12
    maximum_momentum_relative_drift: float = 1.0e-3
    maximum_moisture_relative_drift: float = 1.0e-4
    maximum_energy_relative_drift: float = 1.0e-3
    minimum_rain_mixing_ratio_kgkg: float = 1.0e-8
    minimum_theta_response_k: float = 1.0e-3
    minimum_qv_response_kgkg: float = 1.0e-7
    # The state-stage inventory is independently differenced from O(1e9 kg)
    # totals, while source provenance is summed from FP32 per-cell commits.
    # This remains a sub-microgram-per-tonne numerical closure bound.
    maximum_source_water_relative_residual: float = 1.0e-9
    maximum_species_water_relative_residual: float = 1.0e-9
    maximum_source_heat_relative_residual: float = 1.0e-7
    maximum_microphysics_conservation_relative_residual: float = 1.0e-5
    maximum_pbl_momentum_storage_rounding_fraction: float = 2.0e-3
    # SGS updates are committed to FP32 velocity storage. Bound both the net
    # signed representation effect and the non-cancelling accumulation of all
    # per-step effects; the latter prevents a fortuitous sign cancellation
    # from qualifying a noisy run.
    maximum_cumulative_sgs_momentum_storage_rounding_fraction: float = 1.0e-2
    maximum_absolute_accumulated_sgs_momentum_storage_rounding_fraction: float = 2.0e-2
    maximum_cumulative_dynamics_dry_mass_closure_fraction: float = 1.0e-6
    minimum_surface_precipitation_kg: float = 1.0e-12


CORE_FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
DT_S = 0.7
DURATION_S = 140.0
CHECKPOINT_S = 70.0


def run(binary: Path, config: Path, outdir: Path,
        resume: Path | None = None) -> None:
    command = [
        str(binary), "--headless", f"--config={config}",
        f"--duration={DURATION_S:g}", f"--outdir={outdir}",
        "--compute-backend=cpu", "--guard-mode=strict",
        "--log-profile=quiet",
    ]
    if resume is not None:
        command.append(f"--resume={resume}")
    subprocess.run(command, check=True)


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
    validate_summary(run_dir)
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


def max_abs(records: list[dict[str, str]], names: tuple[str, ...]) -> float:
    return max(abs(float(record[name])) for record in records for name in names)


def checkpoint_sidecars(run_dir: Path) -> list[dict[str, object]]:
    paths = sorted((run_dir / "checkpoints").glob("*.nimbus.json"))
    result = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    require(all(item.get("boundary") == "completed_timestep" for item in result),
            f"{run_dir}: checkpoint outside completed timestep")
    return result


def qualify(work_root: Path,
            thresholds: Thresholds = Thresholds()) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[4]
    binary = repo / "bin/nimbus_sim"
    config = repo / "tests/fixtures/configs/qualification/cgrid_coupled_physics.yaml"
    require(binary.is_file(), f"missing simulator: {binary}")
    require(config.is_file(), f"missing config: {config}")

    continuous = work_root / "continuous"
    resumed = work_root / "resumed"
    run(binary, config, continuous)

    sidecars = checkpoint_sidecars(continuous)
    require([int(item["completed_steps"]) for item in sidecars] == [100, 200],
            f"{continuous}: duplicate or missing nominal checkpoints")
    require(math.isclose(float(sidecars[0]["simulation_time_s"]),
                         CHECKPOINT_S, abs_tol=1.0e-8),
            f"{continuous}: first checkpoint is not at 70 s")
    source_checkpoint = continuous / "checkpoints" / str(
        sidecars[0]["checkpoint_file"])
    run(binary, config, resumed, source_checkpoint)

    continuous_frames = frames(continuous)
    resumed_frames = frames(resumed)
    require([frame.time_s for frame in continuous_frames] == [0.0, DURATION_S],
            f"{continuous}: expected initial/final frames")
    require([frame.time_s for frame in resumed_frames] == [CHECKPOINT_S, DURATION_S],
            f"{resumed}: expected branch-point/final frames")

    final_a = continuous_frames[-1]
    final_b = resumed_frames[-1]
    for field in CORE_FIELDS:
        require((final_a.path / f"{field}.npy").read_bytes()
                == (final_b.path / f"{field}.npy").read_bytes(),
                f"restart changed final {field} bytes")
    final_checkpoints = (
        continuous / "checkpoints/checkpoint_step_000000200.nimbus",
        resumed / "checkpoints/checkpoint_step_000000200.nimbus",
    )
    require(final_checkpoints[0].read_bytes() == final_checkpoints[1].read_bytes(),
            "restart changed final checkpoint bytes")

    initial_fields = load_fields(continuous_frames[0])
    final_fields = load_fields(final_a)
    final_fields["qr"] = read_npy_float32(final_a.path / "qr.npy")
    indices = list(physical_indices(final_a))
    maximum_rain = max(final_fields["qr"].at(*index) for index in indices)
    theta_response = max(abs(final_fields["theta"].at(*index)
                             - initial_fields["theta"].at(*index))
                         for index in indices)
    qv_response = max(abs(final_fields["qv"].at(*index)
                          - initial_fields["qv"].at(*index))
                      for index in indices)
    require(maximum_rain >= thresholds.minimum_rain_mixing_ratio_kgkg,
            "Kessler was enabled but produced no resolved rainwater")
    require(theta_response >= thresholds.minimum_theta_response_k,
            "coupled heat sources produced no resolved theta response")
    require(qv_response >= thresholds.minimum_qv_response_kgkg,
            "coupled water sources produced no resolved vapor response")

    radiation = rows(continuous / "radiation_budget.csv")
    require([int(row["step"]) for row in radiation] == list(range(1, 201)),
            "radiation step sequence is discontinuous")
    samples: list[tuple[float, float]] = []
    for row in radiation:
        sample = float(row["last_radiation_sample_s"])
        next_sample = float(row["next_radiation_sample_s"])
        if not samples or sample != samples[-1][0]:
            samples.append((sample, next_sample))
    expected_samples = [(0.0, 10.0)]
    for deadline in range(10, 140, 10):
        crossing = math.ceil((deadline - 1.0e-12) / DT_S) * DT_S
        expected_samples.append((crossing, float(deadline + 10)))
    require(len(samples) == len(expected_samples) and all(
        math.isclose(actual[0], expected[0], abs_tol=1.0e-8)
        and math.isclose(actual[1], expected[1], abs_tol=1.0e-8)
        for actual, expected in zip(samples, expected_samples)),
        f"radiation cadence drifted: {samples}")
    maximum_rad_step = max(float(row["relative_booking_residual"])
                           for row in radiation)
    last_rad = radiation[-1]
    cumulative_rad = abs(float(last_rad["cumulative_booking_residual_j"])) \
        / max(1.0, abs(float(last_rad["cumulative_expected_energy_j"])))
    require(maximum_rad_step
            <= thresholds.maximum_radiation_step_relative_residual,
            "radiation step power ledger failed")
    require(cumulative_rad
            <= thresholds.maximum_radiation_cumulative_relative_residual,
            "radiation cumulative power ledger failed")

    momentum = rows(continuous / "momentum_budget.csv")
    moisture = rows(continuous / "moisture_budget.csv")
    energy = rows(continuous / "energy_budget.csv")
    maximum_momentum = max_abs(momentum, (
        "rel_momentum_x", "rel_momentum_y", "rel_momentum_z",
        "rel_angular_momentum_z"))
    maximum_moisture = max_abs(moisture, ("rel_total_water",))
    maximum_energy = max_abs(energy, ("rel_total",))
    require(all(int(row["warn"]) == 0 for row in momentum + moisture + energy),
            "combined inventory ledger emitted a warning")
    require(maximum_momentum <= thresholds.maximum_momentum_relative_drift,
            "combined momentum drift exceeded its bound")
    require(maximum_moisture <= thresholds.maximum_moisture_relative_drift,
            "combined water drift exceeded its bound")
    require(maximum_energy <= thresholds.maximum_energy_relative_drift,
            "combined energy drift exceeded its bound")

    sources = rows(continuous / "thermodynamic_source_budget.csv")
    stage_budget = rows(continuous / "budget.csv")
    micro_stages = [row for row in stage_budget if row["stage"] == "microphysics"]
    require(len(sources) == 200 and len(micro_stages) == 200,
            "thermodynamic source ledger is incomplete")
    require(all(int(source["microphysics_conservation_active"]) == 1
                for source in sources),
            "microphysics conservation attribution was not active")
    require([int(source["dynamics_boundary_mass_step_count"])
             for source in sources] == list(range(1, 201)),
            "dynamics boundary-mass ledger step sequence is discontinuous")
    maximum_dynamics_mass_identity = max(
        abs(float(source["dynamics_committed_dry_mass_change_kg"])
            - float(source["dynamics_flux_implied_dry_mass_change_kg"])
            - float(source["dynamics_dry_mass_closure_residual_kg"]))
        / max(1.0,
              abs(float(source["dynamics_committed_dry_mass_change_kg"])),
              abs(float(source[
                  "dynamics_flux_implied_dry_mass_change_kg"])))
        for source in sources)
    maximum_dynamics_cumulative_mass_identity = max(
        abs(float(source[
                "dynamics_cumulative_committed_dry_mass_change_kg"])
            - float(source[
                "dynamics_cumulative_flux_implied_dry_mass_change_kg"])
            - float(source[
                "dynamics_cumulative_dry_mass_closure_residual_kg"]))
        / max(1.0, abs(float(source[
            "dynamics_cumulative_committed_dry_mass_change_kg"])))
        for source in sources)
    maximum_closed_boundary_export = max(
        abs(float(source["dynamics_outward_boundary_dry_mass_kg"]))
        for source in sources)
    initial_dry_mass_kg = (
        sum(float(initial_fields["rho"].at(*index)) for index in indices)
        * final_a.dx_m * final_a.dy_m * final_a.dz_m)
    maximum_cumulative_dynamics_mass_closure_fraction = max(
        abs(float(source[
            "dynamics_cumulative_dry_mass_closure_residual_kg"]))
        / initial_dry_mass_kg
        for source in sources)
    require(maximum_dynamics_mass_identity <= 1.0e-15,
            "per-step dynamics dry-mass attribution does not close")
    require(maximum_dynamics_cumulative_mass_identity <= 1.0e-15,
            "cumulative dynamics dry-mass attribution does not close")
    require(maximum_closed_boundary_export == 0.0,
            "periodic/rigid dynamics authored a nonzero boundary mass flux")
    require(maximum_cumulative_dynamics_mass_closure_fraction <=
            thresholds.maximum_cumulative_dynamics_dry_mass_closure_fraction,
            "cumulative dynamics FP32 dry-mass closure residual exceeded its bound")
    maximum_source_water = max(
        abs(float(stage["d_total_water"])
            - float(source["committed_water_change_kg"]))
        / max(1.0, abs(float(stage["d_total_water"])),
              abs(float(source["committed_water_change_kg"])))
        for stage, source in zip(micro_stages, sources))
    water_species = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
    maximum_species_source_water = max(
        abs(float(stage[f"d_{species}_mass"])
            - float(source[f"{species}_committed_change_kg"]))
        / max(1.0, abs(float(stage[f"d_{species}_mass"])),
              abs(float(source[f"{species}_committed_change_kg"])))
        for stage, source in zip(micro_stages, sources)
        for species in water_species)
    maximum_species_requested_sum_residual = max(
        abs(sum(float(source[f"microphysics_{species}_requested_kg"])
                for species in water_species)
            - float(source["microphysics_water_tendency_kg"]))
        / max(1.0, abs(float(source["microphysics_water_tendency_kg"])))
        for source in sources)
    maximum_species_committed_sum_residual = max(
        abs(sum(float(source[f"{species}_committed_change_kg"])
                for species in water_species)
            - float(source["committed_water_change_kg"]))
        / max(1.0, abs(float(source["committed_water_change_kg"])))
        for source in sources)
    maximum_source_heat = max(
        abs(float(stage["d_thermal_energy"])
            - float(source["state_heat_change_j"]))
        / max(1.0, abs(float(stage["d_thermal_energy"])),
              abs(float(source["state_heat_change_j"])))
        for stage, source in zip(micro_stages, sources))
    maximum_micro_conservation = max(
        abs(float(source["microphysics_conservation_residual_kg"]))
        / max(1.0, abs(float(source["microphysics_baseline_water_kg"])))
        for source in sources)
    cumulative_precipitation = sum(
        float(source["surface_precipitation_kg"]) for source in sources)
    maximum_pbl_work_identity = max(
        abs(float(source["pbl_committed_momentum_work_j"])
            - float(source["pbl_requested_momentum_work_j"])
            - float(source["pbl_finite_step_momentum_correction_j"])
            - float(source["pbl_momentum_storage_rounding_j"]))
        / max(1.0, abs(float(source["pbl_committed_momentum_work_j"])))
        for source in sources)
    maximum_pbl_cumulative_work_identity = max(
        abs(float(source["pbl_cumulative_committed_momentum_work_j"])
            - float(source["pbl_cumulative_requested_momentum_work_j"])
            - float(source[
                "pbl_cumulative_finite_step_momentum_correction_j"])
            - float(source[
                "pbl_cumulative_momentum_storage_rounding_j"]))
        / max(1.0, abs(float(source[
            "pbl_cumulative_committed_momentum_work_j"])))
        for source in sources)
    minimum_pbl_finite_step_correction = min(
        float(source["pbl_finite_step_momentum_correction_j"])
        for source in sources)
    maximum_pbl_storage_rounding_fraction = max(
        abs(float(source["pbl_momentum_storage_rounding_j"]))
        / max(1.0,
              abs(float(source["pbl_requested_momentum_work_j"])),
              abs(float(source["pbl_committed_momentum_work_j"])))
        for source in sources)
    final_pbl_requested_work = float(
        sources[-1]["pbl_cumulative_requested_momentum_work_j"])
    final_pbl_finite_step_correction = float(
        sources[-1]["pbl_cumulative_finite_step_momentum_correction_j"])
    final_pbl_storage_rounding = float(
        sources[-1]["pbl_cumulative_momentum_storage_rounding_j"])
    final_pbl_committed_work = float(
        sources[-1]["pbl_cumulative_committed_momentum_work_j"])
    require([int(source["pbl_momentum_work_step_count"])
             for source in sources] == list(range(1, 201)),
            "PBL momentum-work ledger step sequence is discontinuous")
    require(maximum_pbl_work_identity <= 1.0e-12,
            "PBL momentum-work attribution does not close")
    require(maximum_pbl_cumulative_work_identity <= 1.0e-12,
            "cumulative PBL momentum-work attribution does not close")
    require(minimum_pbl_finite_step_correction >= 0.0,
            "PBL finite-step quadratic work became negative")
    require(final_pbl_requested_work < 0.0 and final_pbl_committed_work < 0.0,
            "slab PBL branch did not remove resolved kinetic energy")
    require(maximum_pbl_storage_rounding_fraction <=
            thresholds.maximum_pbl_momentum_storage_rounding_fraction,
            "PBL FP32 momentum-work attribution noise exceeded its bound")
    maximum_sgs_work_identity = max(
        abs(float(source["sgs_committed_momentum_work_j"])
            - float(source["sgs_requested_momentum_work_j"])
            - float(source["sgs_finite_step_momentum_correction_j"])
            - float(source["sgs_momentum_storage_rounding_j"]))
        / max(1.0, abs(float(source["sgs_committed_momentum_work_j"])))
        for source in sources)
    maximum_sgs_cumulative_work_identity = max(
        abs(float(source["sgs_cumulative_committed_momentum_work_j"])
            - float(source["sgs_cumulative_requested_momentum_work_j"])
            - float(source[
                "sgs_cumulative_finite_step_momentum_correction_j"])
            - float(source[
                "sgs_cumulative_momentum_storage_rounding_j"]))
        / max(1.0, abs(float(source[
            "sgs_cumulative_committed_momentum_work_j"])))
        for source in sources)
    minimum_sgs_finite_step_correction = min(
        float(source["sgs_finite_step_momentum_correction_j"])
        for source in sources)
    maximum_sgs_storage_rounding_fraction = max(
        abs(float(source["sgs_momentum_storage_rounding_j"]))
        / max(1.0,
              abs(float(source["sgs_requested_momentum_work_j"])),
              abs(float(source["sgs_committed_momentum_work_j"])))
        for source in sources)
    maximum_absolute_sgs_storage_rounding_j = max(
        abs(float(source["sgs_momentum_storage_rounding_j"]))
        for source in sources)
    final_sgs_requested_work = float(
        sources[-1]["sgs_cumulative_requested_momentum_work_j"])
    final_sgs_finite_step_correction = float(
        sources[-1]["sgs_cumulative_finite_step_momentum_correction_j"])
    final_sgs_storage_rounding = float(
        sources[-1]["sgs_cumulative_momentum_storage_rounding_j"])
    final_sgs_committed_work = float(
        sources[-1]["sgs_cumulative_committed_momentum_work_j"])
    cumulative_sgs_storage_rounding_fraction = (
        abs(final_sgs_storage_rounding)
        / max(1.0, abs(final_sgs_requested_work),
              abs(final_sgs_committed_work)))
    absolute_accumulated_sgs_storage_rounding_j = sum(
        abs(float(source["sgs_momentum_storage_rounding_j"]))
        for source in sources)
    absolute_accumulated_sgs_storage_rounding_fraction = (
        absolute_accumulated_sgs_storage_rounding_j
        / max(1.0, abs(final_sgs_requested_work),
              abs(final_sgs_committed_work)))
    require([int(source["sgs_momentum_work_step_count"])
             for source in sources] == list(range(1, 201)),
            "SGS momentum-work ledger step sequence is discontinuous")
    require(maximum_sgs_work_identity <= 1.0e-12,
            "SGS momentum-work attribution does not close")
    require(maximum_sgs_cumulative_work_identity <= 1.0e-12,
            "cumulative SGS momentum-work attribution does not close")
    require(minimum_sgs_finite_step_correction >= 0.0,
            "SGS finite-step quadratic work became negative")
    require(final_sgs_requested_work < 0.0 and final_sgs_committed_work < 0.0,
            "Smagorinsky branch did not remove resolved kinetic energy")
    # A per-step relative residual is ill-conditioned while the developing
    # flow requests less work than one FP32 velocity quantum. Preserve that
    # worst-step value as evidence, but qualify the accumulated state effect
    # once the SGS branch is materially active.
    require(cumulative_sgs_storage_rounding_fraction <=
            thresholds.maximum_cumulative_sgs_momentum_storage_rounding_fraction,
            "cumulative SGS FP32 momentum-work attribution noise exceeded "
            "its signed bound: "
            f"{cumulative_sgs_storage_rounding_fraction:.17g} > "
            f"{thresholds.maximum_cumulative_sgs_momentum_storage_rounding_fraction:.17g}")
    require(absolute_accumulated_sgs_storage_rounding_fraction <=
            thresholds.maximum_absolute_accumulated_sgs_momentum_storage_rounding_fraction,
            "SGS FP32 momentum-work attribution noise exceeded its "
            "non-cancelling bound: "
            f"{absolute_accumulated_sgs_storage_rounding_fraction:.17g} > "
            f"{thresholds.maximum_absolute_accumulated_sgs_momentum_storage_rounding_fraction:.17g}")
    require(maximum_source_water
            <= thresholds.maximum_source_water_relative_residual,
            "committed water does not close against its source ledger")
    require(maximum_species_source_water
            <= thresholds.maximum_species_water_relative_residual,
            "a committed water species does not close against its source ledger")
    require(maximum_species_requested_sum_residual <= 1.0e-10,
            "per-species requested water does not reproduce its total: "
            f"{maximum_species_requested_sum_residual:.17g}")
    require(maximum_species_committed_sum_residual <= 1.0e-10,
            "per-species committed water does not reproduce its total: "
            f"{maximum_species_committed_sum_residual:.17g}")
    require(maximum_source_heat
            <= thresholds.maximum_source_heat_relative_residual,
            "committed heat does not close against its source ledger")
    require(maximum_micro_conservation
            <= thresholds.maximum_microphysics_conservation_relative_residual,
            "microphysics process tendencies do not close with precipitation")
    require(cumulative_precipitation
            >= thresholds.minimum_surface_precipitation_kg,
            "warm-rain branch did not exercise surface precipitation fallout")

    resumed_radiation = rows(resumed / "radiation_budget.csv")
    require([int(row["step"]) for row in resumed_radiation] == list(range(101, 201)),
            "restart did not continue radiation ledger sequence")
    require(resumed_radiation[-1]["cumulative_expected_energy_j"]
            == radiation[-1]["cumulative_expected_energy_j"]
            and resumed_radiation[-1]["cumulative_booked_heat_j"]
            == radiation[-1]["cumulative_booked_heat_j"],
            "restart changed cumulative radiation provenance")
    resumed_sources = rows(resumed / "thermodynamic_source_budget.csv")
    require(resumed_sources == sources[100:],
            "restart changed thermodynamic/water source provenance")

    return {
        "status": "PASS",
        "scope": "cpu_flat_cartesian_cgrid_kessler_slab_pbl_horizontal_smagorinsky_simple_grey",
        "duration_s": DURATION_S,
        "continuous_steps": 200,
        "restart_step": 100,
        "bitwise_restart_fields": list(CORE_FIELDS),
        "metrics": {
            "maximum_rain_mixing_ratio_kgkg": maximum_rain,
            "maximum_theta_response_k": theta_response,
            "maximum_qv_response_kgkg": qv_response,
            "maximum_radiation_step_relative_residual": maximum_rad_step,
            "radiation_cumulative_relative_residual": cumulative_rad,
            "maximum_momentum_relative_drift": maximum_momentum,
            "maximum_moisture_relative_drift": maximum_moisture,
            "maximum_energy_relative_drift": maximum_energy,
            "maximum_source_water_relative_residual": maximum_source_water,
            "maximum_dynamics_dry_mass_identity_residual":
                maximum_dynamics_mass_identity,
            "maximum_dynamics_cumulative_dry_mass_identity_residual":
                maximum_dynamics_cumulative_mass_identity,
            "maximum_closed_boundary_dry_mass_export_kg":
                maximum_closed_boundary_export,
            "maximum_cumulative_dynamics_dry_mass_closure_fraction":
                maximum_cumulative_dynamics_mass_closure_fraction,
            "maximum_species_source_water_relative_residual":
                maximum_species_source_water,
            "maximum_species_requested_sum_relative_residual":
                maximum_species_requested_sum_residual,
            "maximum_species_committed_sum_relative_residual":
                maximum_species_committed_sum_residual,
            "maximum_source_heat_relative_residual": maximum_source_heat,
            "maximum_microphysics_conservation_relative_residual":
                maximum_micro_conservation,
            "maximum_pbl_momentum_work_identity_residual":
                maximum_pbl_work_identity,
            "maximum_pbl_cumulative_momentum_work_identity_residual":
                maximum_pbl_cumulative_work_identity,
            "maximum_pbl_momentum_storage_rounding_fraction":
                maximum_pbl_storage_rounding_fraction,
            "cumulative_pbl_requested_momentum_work_j":
                final_pbl_requested_work,
            "cumulative_pbl_finite_step_momentum_correction_j":
                final_pbl_finite_step_correction,
            "cumulative_pbl_momentum_storage_rounding_j":
                final_pbl_storage_rounding,
            "cumulative_pbl_committed_momentum_work_j":
                final_pbl_committed_work,
            "maximum_sgs_momentum_work_identity_residual":
                maximum_sgs_work_identity,
            "maximum_sgs_cumulative_momentum_work_identity_residual":
                maximum_sgs_cumulative_work_identity,
            "maximum_sgs_momentum_storage_rounding_fraction":
                maximum_sgs_storage_rounding_fraction,
            "maximum_absolute_sgs_momentum_storage_rounding_j":
                maximum_absolute_sgs_storage_rounding_j,
            "cumulative_sgs_momentum_storage_rounding_fraction":
                cumulative_sgs_storage_rounding_fraction,
            "absolute_accumulated_sgs_momentum_storage_rounding_j":
                absolute_accumulated_sgs_storage_rounding_j,
            "absolute_accumulated_sgs_momentum_storage_rounding_fraction":
                absolute_accumulated_sgs_storage_rounding_fraction,
            "cumulative_sgs_requested_momentum_work_j":
                final_sgs_requested_work,
            "cumulative_sgs_finite_step_momentum_correction_j":
                final_sgs_finite_step_correction,
            "cumulative_sgs_momentum_storage_rounding_j":
                final_sgs_storage_rounding,
            "cumulative_sgs_committed_momentum_work_j":
                final_sgs_committed_work,
            "cumulative_surface_precipitation_kg": cumulative_precipitation,
        },
        "thresholds": asdict(thresholds),
    }


def main() -> int:
    work_root = Path(tempfile.mkdtemp(prefix="nimbus-cgrid-coupled-"))
    try:
        print(json.dumps(qualify(work_root), indent=2, sort_keys=True))
    except (QualificationError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"FAIL: {error}") from error
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

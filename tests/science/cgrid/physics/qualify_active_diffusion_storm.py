#!/usr/bin/env python3
"""Qualify a long coupled C-grid storm with active Kh/Kv diffusion."""

from __future__ import annotations

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
    require,
    valid_generation,
    validate_summary,
)


@dataclass(frozen=True)
class Thresholds:
    maximum_work_identity_residual: float = 1.0e-12
    maximum_cumulative_storage_rounding_fraction: float = 2.0e-3
    maximum_step_energy_relative_change: float = 1.0e-3
    maximum_diffusion_dry_mass_relative_change: float = 1.0e-12
    maximum_diffusion_water_relative_change: float = 2.0e-7
    minimum_cumulative_dissipation_j: float = 1.0e6
    minimum_updraft_ms: float = 1.0
    minimum_theta_response_k: float = 0.1
    minimum_qv_response_kgkg: float = 1.0e-5


CORE_FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
DT_S = 0.7
DURATION_S = 700.0
CHECKPOINT_S = 350.0
TOTAL_STEPS = 1000
CHECKPOINT_STEP = 500
SOURCE_LEDGER_COLUMNS = 695


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


def rows(path: Path,
         expected_columns: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        require(fieldnames is not None, f"{path}: missing CSV header")
        require(len(fieldnames) == len(set(fieldnames)),
                f"{path}: duplicate CSV columns")
        if expected_columns is not None:
            require(len(fieldnames) == expected_columns,
                    f"{path}: expected {expected_columns} columns, "
                    f"found {len(fieldnames)}")
        result = list(reader)
    require(result, f"{path}: empty ledger")
    for row in result:
        require(None not in row, f"{path}: data row exceeds CSV header")
        for key, value in row.items():
            if key == "stage":
                continue
            require(value is not None and math.isfinite(float(value)),
                    f"{path}: non-finite {key}")
    return result


def checkpoint_sidecars(run_dir: Path) -> list[dict[str, object]]:
    paths = sorted((run_dir / "checkpoints").glob("*.nimbus.json"))
    result = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    require(all(item.get("boundary") == "completed_timestep"
                for item in result),
            f"{run_dir}: checkpoint outside completed timestep")
    return result


def relative_identity(committed: float, requested: float,
                      correction: float, rounding: float) -> float:
    return abs(committed - requested - correction - rounding) / max(
        1.0, abs(committed), abs(requested), abs(correction), abs(rounding))


def sponge_energy_identities(
        sources: list[dict[str, str]],
        expected_counts: list[int]) -> dict[str, float]:
    require([int(row["sponge_energy_step_count"]) for row in sources]
            == expected_counts,
            "sponge-energy step sequence is discontinuous")
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
                        carrier, requested, finite, storage))
                reservoirs = sum(float(row[prefix + suffix])
                                 for suffix in (
                                     "dry_internal_change_j",
                                     "water_sensible_change_j",
                                     "phase_reference_change_j",
                                     "carrier_kinetic_change_j",
                                     "gravitational_change_j",
                                     "subgrid_tke_change_j",
                                     "closure_residual_j"))
                committed = float(
                    row[prefix + "committed_energy_change_j"])
                energy_maxima[cadence] = max(
                    energy_maxima[cadence],
                    abs(committed - reservoirs)
                    / max(1.0, abs(committed), abs(reservoirs)))
    require(max(*energy_maxima.values(), *momentum_maxima.values())
            <= 1.0e-14,
            "sponge operator-time energy identity did not close")
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
        expected_counts: list[int]) -> dict[str, float]:
    require([int(row["final_boundary_energy_step_count"]) for row in sources]
            == expected_counts,
            "final-boundary energy step sequence is discontinuous")
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
                abs(constraint - carrier)
                / max(1.0, abs(constraint), abs(carrier)),
                abs(committed - reservoirs)
                / max(1.0, abs(committed), abs(reservoirs)))
    require(max(maxima.values()) <= 1.0e-14,
            "final-boundary operator-time energy identity did not close")
    return {
        "maximum_final_boundary_energy_identity_relative_residual":
            maxima["step"],
        "maximum_cumulative_final_boundary_energy_identity_relative_residual":
            maxima["cumulative"],
    }


def qualify(work_root: Path,
            thresholds: Thresholds = Thresholds()) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[4]
    binary = repo / "bin/nimbus_sim"
    config = repo / "tests/fixtures/configs/qualification/cgrid_active_diffusion_storm.yaml"
    require(binary.is_file(), f"missing simulator: {binary}")
    require(config.is_file(), f"missing config: {config}")

    continuous = work_root / "continuous"
    resumed = work_root / "resumed"
    run(binary, config, continuous)
    sidecars = checkpoint_sidecars(continuous)
    require([int(item["completed_steps"]) for item in sidecars]
            == [CHECKPOINT_STEP, TOTAL_STEPS],
            "active-diffusion checkpoints do not match steps 500/1000")
    require(math.isclose(float(sidecars[0]["simulation_time_s"]),
                         CHECKPOINT_S, abs_tol=1.0e-8),
            "active-diffusion branch checkpoint is not at 350 s")
    source_checkpoint = continuous / "checkpoints" / str(
        sidecars[0]["checkpoint_file"])
    run(binary, config, resumed, source_checkpoint)

    continuous_frames = frames(continuous)
    resumed_frames = frames(resumed)
    require([frame.time_s for frame in continuous_frames]
            == [0.0, DURATION_S],
            "continuous active-diffusion run lacks initial/final frames")
    require([frame.time_s for frame in resumed_frames]
            == [CHECKPOINT_S, DURATION_S],
            "resumed active-diffusion run lacks branch/final frames")
    final_a = continuous_frames[-1]
    final_b = resumed_frames[-1]
    for field in CORE_FIELDS:
        require((final_a.path / f"{field}.npy").read_bytes()
                == (final_b.path / f"{field}.npy").read_bytes(),
                f"restart changed final {field} bytes")
    final_checkpoints = (
        continuous / "checkpoints/checkpoint_step_000001000.nimbus",
        resumed / "checkpoints/checkpoint_step_000001000.nimbus",
    )
    require(final_checkpoints[0].read_bytes() ==
            final_checkpoints[1].read_bytes(),
            "restart changed final checkpoint bytes")

    sources = rows(
        continuous / "thermodynamic_source_budget.csv",
        SOURCE_LEDGER_COLUMNS)
    require(len(sources) == TOTAL_STEPS,
            "active-diffusion source ledger is incomplete")
    expected_counts = list(range(1, TOTAL_STEPS + 1))
    require([int(row["diffusion_momentum_work_step_count"])
             for row in sources] == expected_counts,
            "diffusion work step sequence is discontinuous")
    require([int(row["diffusion_moist_energy_step_count"])
             for row in sources] == expected_counts,
            "diffusion moist-energy step sequence is discontinuous")
    require(all(
        int(row["diffusion_moist_energy_report_schema_version"]) == 1
        and int(row["diffusion_moist_energy_basis_version"]) == 2
        and int(row["diffusion_moist_energy_step_complete"]) == 1
        and int(row["diffusion_moist_energy_cumulative_complete"]) == 1
        for row in sources),
        "diffusion moist-energy currency is partial or changed")
    identity_residual = max(relative_identity(
        float(row["diffusion_committed_momentum_work_j"]),
        float(row["diffusion_requested_momentum_work_j"]),
        float(row["diffusion_finite_step_momentum_correction_j"]),
        float(row["diffusion_momentum_storage_rounding_j"]))
        for row in sources)
    cumulative_identity_residual = max(relative_identity(
        float(row["diffusion_cumulative_committed_momentum_work_j"]),
        float(row["diffusion_cumulative_requested_momentum_work_j"]),
        float(row[
            "diffusion_cumulative_finite_step_momentum_correction_j"]),
        float(row["diffusion_cumulative_momentum_storage_rounding_j"]))
        for row in sources)
    require(identity_residual <= thresholds.maximum_work_identity_residual,
            "per-step diffusion momentum-work identity did not close")
    require(cumulative_identity_residual
            <= thresholds.maximum_work_identity_residual,
            "cumulative diffusion momentum-work identity did not close")
    moist_energy_identity = max(
        abs(
            float(row[
                "diffusion_moist_energy_step_committed_energy_change_j"])
            - float(row["diffusion_committed_momentum_work_j"])
            - sum(float(row[key]) for key in (
                "diffusion_moist_energy_step_dry_internal_change_j",
                "diffusion_moist_energy_step_water_sensible_change_j",
                "diffusion_moist_energy_step_phase_reference_change_j",
                "diffusion_moist_energy_step_carrier_kinetic_change_j",
                "diffusion_moist_energy_step_gravitational_change_j",
                "diffusion_moist_energy_step_subgrid_tke_change_j",
                "diffusion_moist_energy_step_closure_residual_j",
            )))
        / max(
            1.0,
            abs(float(row[
                "diffusion_moist_energy_step_committed_energy_change_j"])))
        for row in sources)
    cumulative_moist_energy_identity = max(
        abs(
            float(row[
                "diffusion_moist_energy_cumulative_committed_energy_change_j"])
            - float(row["diffusion_cumulative_committed_momentum_work_j"])
            - sum(float(row[key]) for key in (
                "diffusion_moist_energy_cumulative_dry_internal_change_j",
                "diffusion_moist_energy_cumulative_water_sensible_change_j",
                "diffusion_moist_energy_cumulative_phase_reference_change_j",
                "diffusion_moist_energy_cumulative_carrier_kinetic_change_j",
                "diffusion_moist_energy_cumulative_gravitational_change_j",
                "diffusion_moist_energy_cumulative_subgrid_tke_change_j",
                "diffusion_moist_energy_cumulative_closure_residual_j",
            )))
        / max(
            1.0,
            abs(float(row[
                "diffusion_moist_energy_cumulative_committed_energy_change_j"])))
        for row in sources)
    require(max(moist_energy_identity, cumulative_moist_energy_identity)
            <= thresholds.maximum_work_identity_residual,
            "diffusion moist-energy identity did not close")
    require([int(row["eos_pressure_energy_step_count"])
             for row in sources] == expected_counts,
            "EOS pressure-energy step sequence is discontinuous")
    require(all(
        int(row["eos_pressure_energy_report_schema_version"]) == 1
        and int(row["eos_pressure_energy_basis_version"]) == 2
        and int(row["eos_pressure_energy_step_complete"]) == 1
        and int(row["eos_pressure_energy_cumulative_complete"]) == 1
        for row in sources),
        "EOS pressure-energy currency is partial or changed")
    eos_identity = max(
        abs(
            float(row["eos_pressure_energy_step_committed_energy_change_j"])
            - sum(float(row[key]) for key in (
                "eos_pressure_energy_step_dry_internal_change_j",
                "eos_pressure_energy_step_water_sensible_change_j",
                "eos_pressure_energy_step_phase_reference_change_j",
                "eos_pressure_energy_step_carrier_kinetic_change_j",
                "eos_pressure_energy_step_gravitational_change_j",
                "eos_pressure_energy_step_subgrid_tke_change_j",
                "eos_pressure_energy_step_closure_residual_j",
            )))
        / max(
            1.0,
            abs(float(row[
                "eos_pressure_energy_step_committed_energy_change_j"])))
        for row in sources)
    cumulative_eos_identity = max(
        abs(
            float(row[
                "eos_pressure_energy_cumulative_committed_energy_change_j"])
            - sum(float(row[key]) for key in (
                "eos_pressure_energy_cumulative_dry_internal_change_j",
                "eos_pressure_energy_cumulative_water_sensible_change_j",
                "eos_pressure_energy_cumulative_phase_reference_change_j",
                "eos_pressure_energy_cumulative_carrier_kinetic_change_j",
                "eos_pressure_energy_cumulative_gravitational_change_j",
                "eos_pressure_energy_cumulative_subgrid_tke_change_j",
                "eos_pressure_energy_cumulative_closure_residual_j",
            )))
        / max(
            1.0,
            abs(float(row[
                "eos_pressure_energy_cumulative_committed_energy_change_j"])))
        for row in sources)
    require(max(eos_identity, cumulative_eos_identity)
            <= thresholds.maximum_work_identity_residual,
            "EOS pressure-energy identity did not close")
    sponge_metrics = sponge_energy_identities(
        sources, expected_counts)
    final_boundary_metrics = final_boundary_energy_identities(
        sources, expected_counts)
    final_source = sources[-1]
    requested_work = float(
        final_source["diffusion_cumulative_requested_momentum_work_j"])
    committed_work = float(
        final_source["diffusion_cumulative_committed_momentum_work_j"])
    finite_step = float(final_source[
        "diffusion_cumulative_finite_step_momentum_correction_j"])
    storage_rounding = float(final_source[
        "diffusion_cumulative_momentum_storage_rounding_j"])
    require(requested_work < -thresholds.minimum_cumulative_dissipation_j,
            "active momentum diffusion did not request resolved dissipation")
    require(committed_work < -thresholds.minimum_cumulative_dissipation_j,
            "active momentum diffusion did not commit resolved dissipation")
    require(finite_step >= 0.0,
            "diffusion finite-step energy correction became negative")
    storage_fraction = abs(storage_rounding) / max(1.0, abs(requested_work))
    require(storage_fraction <=
            thresholds.maximum_cumulative_storage_rounding_fraction,
            "diffusion FP32 work residual exceeded its bound")

    resumed_sources = rows(
        resumed / "thermodynamic_source_budget.csv",
        SOURCE_LEDGER_COLUMNS)
    require(resumed_sources == sources[CHECKPOINT_STEP:],
            "restart changed diffusion/source ledger continuation")

    budgets = rows(continuous / "budget.csv")
    diffusion_rows = [row for row in budgets if row["stage"] == "diffusion"]
    require(len(diffusion_rows) == TOTAL_STEPS,
            "stage budget lacks one diffusion row per step")
    require(all(int(row["warn_physical"]) == 0 for row in diffusion_rows),
            "diffusion stage emitted a physical-budget warning")
    max_mass = max(abs(float(row["rel_dry_mass"]))
                   for row in diffusion_rows)
    max_water = max(abs(float(row["rel_total_water"]))
                    for row in diffusion_rows)
    require(max_mass <=
            thresholds.maximum_diffusion_dry_mass_relative_change,
            "diffusion changed dry mass")
    require(max_water <= thresholds.maximum_diffusion_water_relative_change,
            "diffusion water-inventory rounding exceeded its bound")

    energy = rows(continuous / "energy_budget.csv")
    completed_step_metrics = completed_step_energy_identities(
        sources, energy, expected_counts, require)
    max_step_energy = max(abs(float(row["rel_total"])) for row in energy)
    require(max_step_energy <=
            thresholds.maximum_step_energy_relative_change,
            "coupled active-diffusion step energy change exceeded its bound")

    initial_fields = load_fields(continuous_frames[0])
    final_fields = load_fields(final_a)
    indices = list(physical_indices(final_a))
    maximum_updraft = max(abs(final_fields["w"].at(*index))
                          for index in indices)
    theta_response = max(abs(final_fields["theta"].at(*index)
                             - initial_fields["theta"].at(*index))
                         for index in indices)
    qv_response = max(abs(final_fields["qv"].at(*index)
                          - initial_fields["qv"].at(*index))
                      for index in indices)
    require(maximum_updraft >= thresholds.minimum_updraft_ms,
            "active-diffusion storm produced no resolved vertical motion")
    require(theta_response >= thresholds.minimum_theta_response_k,
            "active-diffusion storm produced no thermal response")
    require(qv_response >= thresholds.minimum_qv_response_kgkg,
            "active-diffusion storm produced no moisture response")

    return {
        "status": "PASS",
        "scope": "cpu_flat_cartesian_cgrid_active_explicit_Kh40_Kv80_coupled_storm",
        "duration_s": DURATION_S,
        "continuous_steps": TOTAL_STEPS,
        "restart_step": CHECKPOINT_STEP,
        "bitwise_restart_fields": list(CORE_FIELDS),
        "metrics": {
            "maximum_diffusion_work_identity_residual": identity_residual,
            "maximum_diffusion_cumulative_work_identity_residual":
                cumulative_identity_residual,
            "maximum_diffusion_moist_energy_identity_residual":
                moist_energy_identity,
            "maximum_diffusion_cumulative_moist_energy_identity_residual":
                cumulative_moist_energy_identity,
            "maximum_eos_pressure_energy_identity_residual": eos_identity,
            "maximum_eos_pressure_cumulative_energy_identity_residual":
                cumulative_eos_identity,
            **sponge_metrics,
            **final_boundary_metrics,
            **completed_step_metrics,
            "cumulative_diffusion_requested_momentum_work_j": requested_work,
            "cumulative_diffusion_finite_step_correction_j": finite_step,
            "cumulative_diffusion_storage_rounding_j": storage_rounding,
            "cumulative_diffusion_committed_momentum_work_j": committed_work,
            "cumulative_diffusion_storage_rounding_fraction": storage_fraction,
            "maximum_diffusion_dry_mass_relative_change": max_mass,
            "maximum_diffusion_water_relative_change": max_water,
            "maximum_step_energy_relative_change": max_step_energy,
            "maximum_updraft_ms": maximum_updraft,
            "maximum_theta_response_k": theta_response,
            "maximum_qv_response_kgkg": qv_response,
        },
        "thresholds": asdict(thresholds),
    }


def main() -> int:
    work_root = Path(tempfile.mkdtemp(prefix="nimbus-active-diffusion-"))
    try:
        print(json.dumps(qualify(work_root), indent=2, sort_keys=True))
    except (QualificationError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"FAIL: {error}") from error
    finally:
        if os.environ.get("NIMBUS_QUALIFICATION_KEEP") == "1":
            print(f"[active-diffusion] kept artifacts: {work_root}")
        else:
            shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

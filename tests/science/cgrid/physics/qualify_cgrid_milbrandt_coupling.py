#!/usr/bin/env python3
"""Qualify sustained storm Milbrandt/radiation coupling on the CPU C-grid."""

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

from check_hydrostatic_hold import (
    QualificationError,
    STEP_DIR_PATTERN,
    load_fields,
    parse_frame,
    physical_indices,
    read_json_object,
    require,
    valid_generation,
)


@dataclass(frozen=True)
class Thresholds:
    maximum_radiation_step_relative_residual: float = 1.0e-10
    maximum_radiation_cumulative_relative_residual: float = 1.0e-12
    maximum_momentum_relative_drift: float = 1.0e-3
    maximum_moisture_relative_drift: float = 1.0e-4
    maximum_energy_relative_drift: float = 1.0e-4
    maximum_source_water_relative_residual: float = 2.0e-6
    maximum_species_water_relative_residual: float = 1.0e-6
    maximum_source_heat_relative_residual: float = 1.0e-6
    maximum_microphysics_conservation_relative_residual: float = 1.0e-6
    maximum_cumulative_dynamics_dry_mass_closure_fraction: float = 1.0e-6
    minimum_species_requested_mass_kg: float = 1.0
    minimum_microphysics_heat_j: float = 1.0
    minimum_surface_precipitation_kg: float = 1.0


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
FROZEN_SPECIES = ("qi", "qs", "qg", "qh")
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
        checkpoint_s: int, threads: int, *, mixed_phase: bool,
        resume: Path | None = None) -> None:
    command = [
        str(binary), "--headless", f"--config={config}",
        f"--duration={duration_s}", f"--write-every={duration_s}",
        f"--outdir={outdir}", f"--checkpoint-every={checkpoint_s}",
        "--compute-backend=cpu", "--guard-mode=strict",
        "--log-profile=quiet",
    ]
    if mixed_phase:
        command.append("--qualification-milbrandt-mixed-phase")
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
    require(all(item.get("boundary") == "completed_timestep"
                for item in result),
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
            and math.isclose(candidate[-1].time_s, duration_s,
                             abs_tol=1.0e-8),
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


def analyze_physics(run_dir: Path, duration_s: int,
                    thresholds: Thresholds) -> dict[str, object]:
    run_frames = frames(run_dir)
    require([frame.time_s for frame in run_frames]
            == [0.0, float(duration_s)],
            f"{run_dir}: expected initial and final frames")
    initial = load_fields(run_frames[0])
    indices = list(physical_indices(run_frames[0]))
    require(indices, f"{run_dir}: empty physical domain")
    expected_steps = int(round(duration_s / DT_S))

    radiation = rows(run_dir / "radiation_budget.csv")
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
            1.0, abs(float(last_radiation[
                "cumulative_expected_energy_j"])))
    require(maximum_rad_step
            <= thresholds.maximum_radiation_step_relative_residual,
            "radiation step power ledger failed")
    require(cumulative_rad
            <= thresholds.maximum_radiation_cumulative_relative_residual,
            "radiation cumulative power ledger failed")

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
    inventory_metrics: dict[str, float] = {}
    for filename, columns, threshold in inventory_specs:
        ledger = rows(run_dir / filename)
        maximum = max(abs(float(row[column])) for row in ledger
                      for column in columns)
        require(all(int(row["warn"]) == 0 for row in ledger),
                f"{filename}: inventory ledger emitted a warning")
        require(maximum <= threshold,
                f"{filename}: inventory drift exceeded its bound")
        inventory_metrics[f"maximum_{filename.removesuffix('_budget.csv')}"
                          "_relative_drift"] = maximum

    sources = rows(run_dir / "thermodynamic_source_budget.csv")
    stages = rows(run_dir / "budget.csv")
    micro_stages = [row for row in stages if row["stage"] == "microphysics"]
    require(len(sources) == expected_steps
            and len(micro_stages) == expected_steps,
            "thermodynamic source ledger is incomplete")
    require(all(int(row["microphysics_conservation_active"]) == 1
                for row in sources),
            "microphysics conservation attribution was not active")
    require([int(row["dynamics_boundary_mass_step_count"])
             for row in sources] == list(range(1, expected_steps + 1)),
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

    maximum_source_water = max(
        relative_identity(float(stage["d_total_water"]),
                          float(source["committed_water_change_kg"]))
        for stage, source in zip(micro_stages, sources))
    maximum_species_water = max(
        relative_identity(float(stage[f"d_{species}_mass"]),
                          float(source[f"{species}_committed_change_kg"]))
        for stage, source in zip(micro_stages, sources)
        for species in WATER_SPECIES)
    maximum_requested_sum = max(
        abs(sum(float(source[
                    f"microphysics_{species}_requested_kg"])
                for species in WATER_SPECIES)
            - float(source["microphysics_water_tendency_kg"]))
        / max(
            1.0,
            sum(abs(float(source[
                f"microphysics_{species}_requested_kg"]))
                for species in WATER_SPECIES),
            abs(float(source["microphysics_water_tendency_kg"])))
        for source in sources)
    maximum_committed_sum = max(
        abs(sum(float(source[f"{species}_committed_change_kg"])
                for species in WATER_SPECIES)
            - float(source["committed_water_change_kg"]))
        / max(
            1.0,
            sum(abs(float(source[
                f"{species}_committed_change_kg"]))
                for species in WATER_SPECIES),
            abs(float(source["committed_water_change_kg"])))
        for source in sources)
    heat_attribution = [
        (
            abs(float(stage["d_thermal_energy"])
                - float(source["state_heat_change_j"])),
            max(
                1.0,
                abs(float(stage["d_thermal_energy"])),
                abs(float(source["state_heat_change_j"])),
                abs(float(source["microphysics_heat_j"])),
                abs(float(source["radiation_heat_j"])),
                abs(float(source["pbl_heat_j"])),
                abs(float(source["theta_repair_j"])),
                abs(float(source["theta_storage_rounding_j"])),
                abs(float(source["theta_storage_residual_j"])),
            ),
        )
        for stage, source in zip(micro_stages, sources)
    ]
    maximum_source_heat_absolute = max(
        residual for residual, _ in heat_attribution)
    maximum_source_heat = max(
        residual / scale for residual, scale in heat_attribution)
    maximum_micro_conservation = max(
        abs(float(source["microphysics_conservation_residual_kg"]))
        / max(1.0, abs(float(source["microphysics_baseline_water_kg"])))
        for source in sources)
    maximum_microphysics_heat = max(
        abs(float(source["microphysics_heat_j"])) for source in sources)
    cumulative_precipitation = sum(
        float(source["surface_precipitation_kg"]) for source in sources)
    species_activity = {
        species: max(abs(float(source[
            f"microphysics_{species}_requested_kg"])) for source in sources)
        for species in WATER_SPECIES
    }

    require(maximum_source_water
            <= thresholds.maximum_source_water_relative_residual,
            "committed total water does not close against its stage ledger")
    require(maximum_species_water
            <= thresholds.maximum_species_water_relative_residual,
            "a committed water species does not close against its stage ledger")
    require(maximum_requested_sum <= 1.0e-10,
            "requested per-species water does not reproduce its total")
    require(maximum_committed_sum <= 1.0e-10,
            "committed per-species water does not reproduce its total")
    require(maximum_source_heat
            <= thresholds.maximum_source_heat_relative_residual,
            "committed heat does not close against its stage ledger")
    require(maximum_micro_conservation <= thresholds.
            maximum_microphysics_conservation_relative_residual,
            "Milbrandt process water does not close with precipitation")
    require(maximum_microphysics_heat >= thresholds.minimum_microphysics_heat_j,
            "Milbrandt produced no latent-heating signal")
    require(cumulative_precipitation
            >= thresholds.minimum_surface_precipitation_kg,
            "Milbrandt did not exercise physical-surface precipitation")
    for species in FROZEN_SPECIES:
        require(species_activity[species]
                >= thresholds.minimum_species_requested_mass_kg,
                f"Milbrandt did not materially evolve {species}")

    return {
        **inventory_metrics,
        "maximum_radiation_step_relative_residual": maximum_rad_step,
        "radiation_cumulative_relative_residual": cumulative_rad,
        "maximum_source_water_relative_residual": maximum_source_water,
        "maximum_species_water_relative_residual": maximum_species_water,
        "maximum_species_requested_sum_relative_residual":
            maximum_requested_sum,
        "maximum_species_committed_sum_relative_residual":
            maximum_committed_sum,
        "maximum_source_heat_relative_residual": maximum_source_heat,
        "maximum_source_heat_absolute_residual_j":
            maximum_source_heat_absolute,
        "maximum_microphysics_conservation_relative_residual":
            maximum_micro_conservation,
        "maximum_microphysics_heat_j": maximum_microphysics_heat,
        "maximum_requested_species_mass_kg": species_activity,
        "cumulative_surface_precipitation_kg": cumulative_precipitation,
        "maximum_dynamics_dry_mass_identity_residual":
            maximum_dynamics_identity,
        "maximum_dynamics_cumulative_dry_mass_identity_residual":
            maximum_dynamics_cumulative_identity,
        "maximum_closed_boundary_dry_mass_export_kg":
            maximum_closed_boundary_export,
        "maximum_cumulative_dynamics_dry_mass_closure_fraction":
            dynamics_closure_fraction,
    }


def qualify(work_root: Path, duration_s: int, parallel_threads: int,
            thresholds: Thresholds = Thresholds()) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[4]
    binary = repo / "bin/nimbus_sim_internal_qualification"
    ordinary_binary = repo / "bin/nimbus_sim"
    config = repo / (
        "tests/fixtures/configs/qualification/cgrid_milbrandt_radiation_coupled.yaml")
    require(binary.is_file(), f"missing simulator: {binary}")
    require(ordinary_binary.is_file(),
            f"missing ordinary simulator: {ordinary_binary}")
    require(config.is_file(), f"missing config: {config}")

    ordinary = work_root / "ordinary_runtime"
    checkpoint_s = duration_s // 2
    serial = work_root / "serial_continuous"
    serial_resumed = work_root / "serial_resumed"
    parallel = work_root / "parallel_continuous"
    parallel_resumed = work_root / "parallel_resumed"
    run(ordinary_binary, config, ordinary, 2, 2, 1, mixed_phase=False)
    require([frame.time_s for frame in frames(ordinary)] == [0.0, 2.0],
            "ordinary runtime did not publish a strict Milbrandt run")
    run(binary, config, serial, duration_s, checkpoint_s, 1,
        mixed_phase=True)
    serial_sidecars = checkpoint_sidecars(serial)
    require(len(serial_sidecars) == 2 and all(
        math.isclose(float(item["simulation_time_s"]), expected,
                     rel_tol=0.0, abs_tol=1.0e-8)
        for item, expected in zip(
            serial_sidecars, (checkpoint_s, duration_s))),
            f"{serial}: expected midpoint and final checkpoints")
    branch_checkpoint = checkpoint_path(serial, serial_sidecars[0])
    run(binary, config, serial_resumed, duration_s, checkpoint_s, 1,
        mixed_phase=False, resume=branch_checkpoint)
    run(binary, config, parallel, duration_s, checkpoint_s, parallel_threads,
        mixed_phase=True)
    run(binary, config, parallel_resumed, duration_s, checkpoint_s,
        parallel_threads, mixed_phase=False, resume=branch_checkpoint)

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
        "scope":
            "internal_cpu_flat_cartesian_cgrid_milbrandt_storm_simple_grey",
        "duration_s": duration_s,
        "continuous_steps": int(round(duration_s / DT_S)),
        "restart_s": checkpoint_s,
        "serial_threads": 1,
        "parallel_threads": parallel_threads,
        "ordinary_runtime_selectable": True,
        "ordinary_runtime_reachability_duration_s": 2,
        "bitwise_final_fields": list(CORE_FIELDS),
        "bitwise_final_checkpoint_including_private_moments": True,
        "bitwise_thread_ledgers": list(LEDGER_FILES),
        "exact_restart_ledger_suffixes": list(LEDGER_FILES),
        "comparisons": list(comparisons),
        "metrics": metrics,
        "thresholds": asdict(thresholds),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=400)
    parser.add_argument("--parallel-threads", type=int, default=4)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.duration < 20 or args.duration % 20 != 0:
        parser.error("duration must be a positive multiple of 20 s")
    if args.parallel_threads < 2 or args.parallel_threads > 256:
        parser.error("parallel-threads must be between 2 and 256")

    created_temp = args.work_dir is None
    work_root = args.work_dir or Path(tempfile.mkdtemp(
        prefix="nimbus-cgrid-milbrandt-"))
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        result = qualify(work_root, args.duration, args.parallel_threads)
    except (QualificationError, OSError,
            subprocess.CalledProcessError) as error:
        if created_temp and os.environ.get("NIMBUS_QUALIFICATION_KEEP") != "1":
            shutil.rmtree(work_root, ignore_errors=True)
        parser.exit(1, f"[cgrid-milbrandt] FAIL: {error}\n")

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if created_temp and os.environ.get("NIMBUS_QUALIFICATION_KEEP") == "1":
        print(f"[cgrid-milbrandt] retained artifacts: {work_root}")
    elif created_temp:
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

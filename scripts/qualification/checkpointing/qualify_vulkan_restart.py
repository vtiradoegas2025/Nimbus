#!/usr/bin/env python3
"""Fail-closed same-device restart qualification for report-bearing Vulkan."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


CORE_FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
LEDGERS = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
STEP_PATTERN = re.compile(r"step_[0-9]{6}$")
TELEMETRY_PATTERN = re.compile(
    r"\[GPU TELEMETRY\].*model_steps=([0-9]+).*dispatches=([0-9]+)"
)
MIXED_DEVICE_STAGES = (
    "dynamics/cartesian_cgrid_slow_tendencies.comp.spv",
    "acoustic_fused_cartesian_cgrid",
    "transport_batch_cartesian_cgrid",
    "smagorinsky_fused",
    "microphysics/kessler_pointwise.comp.spv",
    "microphysics/kessler_sedimentation.comp.spv",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(binary: Path, config: Path, outdir: Path, duration: int,
        checkpoint_every: int, resume: Path | None = None,
        require_mixed_physics: bool = False) -> str:
    command = [
        str(binary),
        "--headless",
        f"--config={config}",
        "--compute-backend=vulkan",
        "--guard-mode=off",
        f"--duration={duration}",
        f"--write-every={duration}",
        f"--checkpoint-every={checkpoint_every}",
        f"--outdir={outdir}",
        "--no-timing",
    ]
    if resume is not None:
        command.append(f"--resume={resume}")
    environment = os.environ.copy()
    environment["NIMBUS_GPU_KERNEL_TIMING"] = "1"
    environment["NIMBUS_GPU_RESIDENT"] = "1"
    environment["NIMBUS_GPU_LIN_BATCH"] = "1"
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(
        result.returncode == 0,
        f"{outdir.name}: simulator exited {result.returncode}\n"
        f"{result.stdout[-5000:]}",
    )
    require("selected device" in result.stdout,
            f"{outdir.name}: no physical Vulkan device selected")
    require("Falling back to CPU backend" not in result.stdout,
            f"{outdir.name}: Vulkan fell back to CPU")
    require("[CHECKPOINT] committed" in result.stdout,
            f"{outdir.name}: no checkpoint was committed")
    if resume is not None:
        require("[CHECKPOINT] resumed" in result.stdout,
                f"{outdir.name}: checkpoint was not restored")
    telemetry = TELEMETRY_PATTERN.findall(result.stdout)
    require(telemetry, f"{outdir.name}: missing GPU telemetry")
    model_steps, dispatches = (int(value) for value in telemetry[-1])
    require(model_steps > 0 and dispatches > 0,
            f"{outdir.name}: resumed Vulkan work was vacuous")
    require(
        "[GPU STAGE TELEMETRY] acoustic_fused_cartesian_cgrid "
        in result.stdout,
        f"{outdir.name}: fused report-attributed acoustic stage "
        "did not execute",
    )
    if require_mixed_physics:
        for expected in (
            "Microphysics: kessler",
            "Radiation: simple_grey",
            "Boundary Layer: slab",
            "Turbulence: smagorinsky",
            "Mode = horizontal_only",
        ):
            require(
                expected in result.stdout,
                f"{outdir.name}: mixed-physics selection missing: {expected}",
            )
        for stage in MIXED_DEVICE_STAGES:
            require(
                f"[GPU STAGE TELEMETRY] {stage} " in result.stdout,
                f"{outdir.name}: required device stage did not execute: "
                f"{stage}",
            )
    return result.stdout


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing mixed-physics ledger: {path.name}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(rows, f"mixed-physics ledger is empty: {path.name}")
    return rows


def finite_column(rows: list[dict[str, str]], name: str) -> list[float]:
    require(
        name in rows[0],
        f"mixed-physics ledger is missing required column: {name}",
    )
    values = [float(row[name]) for row in rows]
    require(
        all(math.isfinite(value) for value in values),
        f"mixed-physics ledger column is non-finite: {name}",
    )
    return values


def finite_value(row: dict[str, str], name: str, context: str) -> float:
    require(name in row, f"{context}: missing report column: {name}")
    value = float(row[name])
    require(math.isfinite(value), f"{context}: non-finite report value: {name}")
    return value


def require_true(row: dict[str, str], name: str, context: str) -> None:
    require(
        finite_value(row, name, context) == 1.0,
        f"{context}: report is not complete: {name}",
    )


def require_identity(
    actual: float,
    expected: float,
    context: str,
    identity: str,
    max_ulps: int = 64,
    reference_values: tuple[float, ...] = (),
) -> None:
    if actual == expected:
        return
    scale = max(
        abs(actual),
        abs(expected),
        *(abs(value) for value in reference_values),
    )
    tolerance = max_ulps * math.ulp(scale)
    require(
        abs(actual - expected) <= tolerance,
        f"{context}: {identity} identity failed: "
        f"actual={actual:.17g} expected={expected:.17g} "
        f"difference={actual - expected:.17g}",
    )


def ordered_sum(
    row: dict[str, str],
    names: tuple[str, ...] | list[str],
    context: str,
) -> float:
    result = 0.0
    for name in names:
        result += finite_value(row, name, context)
    return result


def require_report_owner_identities(
    run_dir: Path, minimum_time_s: float | None = None
) -> int:
    """Independently recompose every completed-step owner in the CSV."""
    rows = read_csv_rows(run_dir / "thermodynamic_source_budget.csv")
    if minimum_time_s is not None:
        rows = [
            row for row in rows
            if finite_value(row, "time_s", run_dir.name) >= minimum_time_s
        ]
    scalar_names = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh")
    reservoirs = (
        "dry_internal",
        "water_sensible",
        "phase_reference",
        "carrier_kinetic",
        "gravitational",
        "subgrid_tke",
    )
    checks = 0

    for row_index, row in enumerate(rows, start=1):
        context = f"{run_dir.name}: report row {row_index}"

        # Momentum-work owners close requested + finite-step correction +
        # storage rounding to the committed work, including when inactive.
        for owner in ("pbl", "sgs", "diffusion"):
            committed = finite_value(
                row, f"{owner}_committed_momentum_work_j", context)
            requested = finite_value(
                row, f"{owner}_requested_momentum_work_j", context)
            finite_step = finite_value(
                row,
                f"{owner}_finite_step_momentum_correction_j",
                context,
            )
            require_identity(
                finite_value(
                    row, f"{owner}_momentum_storage_rounding_j", context),
                committed - requested - finite_step,
                context,
                f"{owner} momentum storage",
                reference_values=(committed, requested, finite_step),
            )
            checks += 1

        # Thermodynamic sources own six moist-mixture reservoirs.
        require_true(
            row, "microphysics_moist_energy_complete", context)
        micro_prefix = "microphysics_moist_"
        micro_reservoir_sum = ordered_sum(
            row,
            [f"{micro_prefix}{name}_change_j" for name in reservoirs],
            context,
        )
        micro_committed = finite_value(
            row, f"{micro_prefix}committed_energy_change_j", context)
        require_identity(
            micro_committed,
            micro_reservoir_sum,
            context,
            "thermodynamic-source committed energy",
        )
        require_identity(
            finite_value(
                row, f"{micro_prefix}energy_closure_residual_j", context),
            micro_committed - micro_reservoir_sum,
            context,
            "thermodynamic-source closure",
        )
        checks += 2

        # SGS and diffusion additionally own their committed momentum work.
        for owner in ("sgs", "diffusion"):
            prefix = f"{owner}_moist_energy_step_"
            report_values = [
                finite_value(row, f"{prefix}{name}_change_j", context)
                for name in reservoirs
            ]
            committed = finite_value(
                row, f"{prefix}committed_energy_change_j", context)
            closure = finite_value(
                row, f"{prefix}closure_residual_j", context)
            complete = finite_value(
                row, f"{prefix}complete", context)
            if complete == 0.0:
                require(
                    committed == 0.0
                    and closure == 0.0
                    and all(value == 0.0 for value in report_values),
                    f"{context}: inactive {owner} report is not zero",
                )
            else:
                require(
                    complete == 1.0,
                    f"{context}: invalid {owner} completion flag",
                )
                expected = finite_value(
                    row, f"{owner}_committed_momentum_work_j", context)
                for value in report_values:
                    expected += value
                require_identity(
                    committed, expected, context,
                    f"{owner} committed moist energy")
                require_identity(
                    closure, committed - expected, context,
                    f"{owner} moist-energy closure")
            checks += 2

        # EOS projection is a pure six-reservoir owner.
        require_true(
            row, "eos_pressure_energy_step_complete", context)
        eos_prefix = "eos_pressure_energy_step_"
        eos_reservoir_sum = ordered_sum(
            row,
            [f"{eos_prefix}{name}_change_j" for name in reservoirs],
            context,
        )
        eos_committed = finite_value(
            row, f"{eos_prefix}committed_energy_change_j", context)
        require_identity(
            eos_committed, eos_reservoir_sum, context,
            "EOS committed energy")
        require_identity(
            finite_value(
                row, f"{eos_prefix}closure_residual_j", context),
            eos_committed - eos_reservoir_sum,
            context,
            "EOS closure",
        )
        checks += 2

        # Each sponge pass independently owns its momentum decomposition and
        # six-reservoir state change.
        require_true(row, "sponge_energy_step_complete", context)
        for pass_name in ("upper", "w_only", "lateral"):
            prefix = f"sponge_energy_step_{pass_name}_"
            carrier = finite_value(
                row, f"{prefix}carrier_kinetic_change_j", context)
            momentum_requested = finite_value(
                row,
                f"{prefix}momentum_requested_linear_work_j",
                context,
            )
            momentum_finite = finite_value(
                row,
                f"{prefix}momentum_finite_step_correction_j",
                context,
            )
            require_identity(
                finite_value(
                    row,
                    f"{prefix}momentum_storage_rounding_j",
                    context,
                ),
                carrier - momentum_requested - momentum_finite,
                context,
                f"sponge {pass_name} momentum storage",
            )
            reservoir_sum = ordered_sum(
                row,
                [f"{prefix}{name}_change_j" for name in reservoirs],
                context,
            )
            committed = finite_value(
                row, f"{prefix}committed_energy_change_j", context)
            require_identity(
                committed, reservoir_sum, context,
                f"sponge {pass_name} committed energy")
            require_identity(
                finite_value(
                    row, f"{prefix}closure_residual_j", context),
                committed - reservoir_sum,
                context,
                f"sponge {pass_name} closure",
            )
            checks += 3

        # Final boundary closure owns exactly its constrained kinetic change.
        require_true(
            row, "final_boundary_energy_step_complete", context)
        final_prefix = "final_boundary_energy_step_"
        final_carrier = finite_value(
            row, f"{final_prefix}carrier_kinetic_change_j", context)
        require_identity(
            finite_value(
                row,
                f"{final_prefix}constraint_committed_work_j",
                context,
            ),
            final_carrier,
            context,
            "final-boundary constrained work",
        )
        final_reservoir_sum = ordered_sum(
            row,
            [f"{final_prefix}{name}_change_j" for name in reservoirs],
            context,
        )
        final_committed = finite_value(
            row, f"{final_prefix}committed_energy_change_j", context)
        require_identity(
            finite_value(
                row, f"{final_prefix}closure_residual_j", context),
            final_committed - final_reservoir_sum,
            context,
            "final-boundary closure",
        )
        checks += 2

        # The acoustic owner has three independent work decompositions.
        require_true(row, "acoustic_v1_complete", context)
        require_true(row, "acoustic_v1_boundary_complete", context)
        require_true(
            row, "acoustic_v1_kinetic_transport_complete", context)
        pressure_committed = finite_value(
            row, "acoustic_v1_pressure_committed_work_j", context)
        pressure_requested = finite_value(
            row,
            "acoustic_v1_pressure_requested_linear_work_j",
            context,
        )
        pressure_finite = finite_value(
            row,
            "acoustic_v1_pressure_finite_step_correction_j",
            context,
        )
        require_identity(
            finite_value(
                row,
                "acoustic_v1_pressure_storage_rounding_j",
                context,
            ),
            pressure_committed - pressure_requested - pressure_finite,
            context,
            "acoustic pressure storage",
            reference_values=(
                pressure_committed,
                pressure_requested,
                pressure_finite,
            ),
        )
        radiative_committed = finite_value(
            row,
            "acoustic_v1_boundary_radiative_committed_work_j",
            context,
        )
        radiative_requested = finite_value(
            row,
            "acoustic_v1_boundary_radiative_requested_linear_work_j",
            context,
        )
        radiative_finite = finite_value(
            row,
            "acoustic_v1_boundary_radiative_finite_step_correction_j",
            context,
        )
        require_identity(
            finite_value(
                row,
                "acoustic_v1_boundary_radiative_storage_rounding_j",
                context,
            ),
            radiative_committed - radiative_requested - radiative_finite,
            context,
            "acoustic radiative-boundary storage",
            reference_values=(
                radiative_committed,
                radiative_requested,
                radiative_finite,
            ),
        )
        boundary_expected = (
            radiative_committed
            + finite_value(
                row,
                "acoustic_v1_boundary_constraint_history_committed_work_j",
                context,
            )
        )
        require_identity(
            finite_value(
                row, "acoustic_v1_boundary_committed_work_j", context),
            boundary_expected,
            context,
            "acoustic total boundary work",
            max_ulps=512,
            reference_values=(radiative_committed,),
        )
        continuity_requested = finite_value(
            row,
            "acoustic_v1_continuity_carrier_requested_change_j",
            context,
        )
        continuity_outward = finite_value(
            row,
            "acoustic_v1_continuity_carrier_outward_boundary_transport_j",
            context,
        )
        require_identity(
            finite_value(
                row,
                "acoustic_v1_continuity_carrier_interior_conversion_j",
                context,
            ),
            continuity_requested + continuity_outward,
            context,
            "acoustic continuity transport",
        )
        require_identity(
            finite_value(
                row,
                "acoustic_v1_carrier_kinetic_change_j",
                context,
            ),
            continuity_requested
            + finite_value(
                row,
                "acoustic_v1_continuity_carrier_storage_rounding_j",
                context,
            ),
            context,
            "acoustic continuity storage",
        )
        combined_transport = (
            continuity_requested
            + finite_value(
                row,
                "acoustic_v1_slow_momentum_advection_requested_linear_work_j",
                context,
            )
        )
        kinetic_interior = finite_value(
            row,
            "acoustic_v1_kinetic_transport_interior_conversion_j",
            context,
        )
        require_identity(
            kinetic_interior,
            combined_transport + continuity_outward,
            context,
            "coupled kinetic transport",
        )
        require_identity(
            finite_value(
                row,
                "acoustic_v1_kinetic_transport_closure_residual_j",
                context,
            ),
            combined_transport - (-continuity_outward + kinetic_interior),
            context,
            "coupled kinetic-transport closure",
        )
        acoustic_reservoir_sum = ordered_sum(
            row,
            [
                "acoustic_v1_dry_internal_change_j",
                "acoustic_v1_water_sensible_change_j",
                "acoustic_v1_phase_reference_change_j",
                "acoustic_v1_carrier_kinetic_change_j",
                "acoustic_v1_gravitational_change_j",
                "acoustic_v1_subgrid_tke_change_j",
            ],
            context,
        )
        require_identity(
            finite_value(
                row, "acoustic_v1_committed_inventory_change_j", context),
            acoustic_reservoir_sum
            + pressure_committed
            + finite_value(
                row, "acoustic_v1_boundary_committed_work_j", context),
            context,
            "acoustic committed inventory",
        )
        checks += 8

        # Slow momentum separates advective and nonadvective linear work,
        # their quadratic terms, and their finite-step cross term.
        require_true(
            row, "slow_momentum_energy_step_complete", context)
        slow_prefix = "slow_momentum_energy_step_"
        slow_requested = ordered_sum(
            row,
            [
                f"{slow_prefix}advection_requested_linear_work_j",
                f"{slow_prefix}nonadvection_requested_linear_work_j",
                f"{slow_prefix}advection_finite_step_correction_j",
                f"{slow_prefix}nonadvection_finite_step_correction_j",
                f"{slow_prefix}coupled_finite_step_cross_work_j",
            ],
            context,
        )
        slow_committed = finite_value(
            row, f"{slow_prefix}committed_kinetic_change_j", context)
        require_identity(
            finite_value(
                row, f"{slow_prefix}storage_rounding_j", context),
            slow_committed - slow_requested,
            context,
            "slow-momentum storage",
        )
        require_identity(
            finite_value(
                row, f"{slow_prefix}closure_residual_j", context),
            slow_committed
            - (
                slow_requested
                + finite_value(
                    row, f"{slow_prefix}storage_rounding_j", context)
            ),
            context,
            "slow-momentum closure",
            reference_values=(slow_committed, slow_requested,
                              finite_value(row, f"{slow_prefix}storage_rounding_j", context)),
        )
        require_identity(
            finite_value(
                row,
                f"{slow_prefix}committed_pressure_energy_change_j",
                context,
            ),
            finite_value(
                row,
                f"{slow_prefix}pressure_dry_internal_change_j",
                context,
            )
            + finite_value(
                row,
                f"{slow_prefix}pressure_water_sensible_change_j",
                context,
            ),
            context,
            "slow pressure-energy commit",
        )
        checks += 3

        # Boundary scalar and moist-energy reports must each close before
        # their sum can enter the completed-step report.
        require_true(
            row,
            "advection_thermodynamic_boundary_reports_complete",
            context,
        )
        require_true(
            row,
            "advection_thermodynamic_moist_energy_reports_complete",
            context,
        )
        for scalar in scalar_names:
            unit = "kg_k" if scalar == "theta" else "kg"
            boundary_prefix = f"advection_{scalar}_"
            outward = finite_value(
                row,
                f"{boundary_prefix}outward_boundary_{unit}",
                context,
            )
            flux_change = finite_value(
                row,
                f"{boundary_prefix}flux_implied_change_{unit}",
                context,
            )
            committed = finite_value(
                row,
                f"{boundary_prefix}committed_change_{unit}",
                context,
            )
            require_identity(
                flux_change, -outward, context,
                f"{scalar} boundary flux")
            require_identity(
                finite_value(
                    row,
                    f"{boundary_prefix}closure_residual_{unit}",
                    context,
                ),
                committed - flux_change,
                context,
                f"{scalar} boundary closure",
            )

            energy_prefix = f"advection_{scalar}_energy_step_"
            energy_reservoir_sum = ordered_sum(
                row,
                [
                    f"{energy_prefix}{name}_change_j"
                    for name in reservoirs
                ],
                context,
            )
            energy_committed = finite_value(
                row,
                f"{energy_prefix}committed_state_change_j",
                context,
            )
            limited = finite_value(
                row,
                f"{energy_prefix}limited_face_state_change_j",
                context,
            )
            outward_energy = finite_value(
                row,
                f"{energy_prefix}outward_boundary_j",
                context,
            )
            interior_conversion = finite_value(
                row,
                f"{energy_prefix}interior_conversion_j",
                context,
            )
            require_identity(
                limited,
                -outward_energy + interior_conversion,
                context,
                f"{scalar} limited-face energy",
                reference_values=(
                    limited,
                    outward_energy,
                    interior_conversion,
                ),
            )
            require_identity(
                finite_value(
                    row,
                    f"{energy_prefix}requested_state_change_j",
                    context,
                ),
                finite_value(
                    row,
                    f"{energy_prefix}density_history_remap_j",
                    context,
                )
                + limited,
                context,
                f"{scalar} requested scalar energy",
            )
            require_identity(
                energy_committed,
                energy_reservoir_sum,
                context,
                f"{scalar} committed scalar energy",
            )
            require_identity(
                finite_value(
                    row,
                    f"{energy_prefix}closure_residual_j",
                    context,
                ),
                energy_committed
                - (
                    finite_value(
                        row,
                        f"{energy_prefix}requested_state_change_j",
                        context,
                    )
                    + finite_value(
                        row,
                        f"{energy_prefix}storage_rounding_j",
                        context,
                    )
                ),
                context,
                f"{scalar} scalar-energy closure",
            )
            checks += 6

        # Dry carrier mass is the density-side boundary owner.
        dry_flux = finite_value(
            row, "dynamics_flux_implied_dry_mass_change_kg", context)
        require_identity(
            dry_flux,
            -finite_value(
                row, "dynamics_outward_boundary_dry_mass_kg", context),
            context,
            "dry-mass boundary flux",
        )
        require_identity(
            finite_value(
                row, "dynamics_dry_mass_closure_residual_kg", context),
            finite_value(
                row, "dynamics_committed_dry_mass_change_kg", context)
            - dry_flux,
            context,
            "dry-mass boundary closure",
        )
        checks += 2

        # Finally, independently sum the nine owners and six reservoirs.
        require_true(
            row, "completed_step_energy_step_complete", context)
        completed_prefix = "completed_step_energy_step_"
        owner_sum = ordered_sum(
            row,
            [
                f"{completed_prefix}{name}_committed_change_j"
                for name in (
                    "acoustic",
                    "slow_dynamics",
                    "scalar_transport",
                    "thermodynamic_sources",
                    "sgs",
                    "diffusion",
                    "eos_projection",
                    "sponge",
                    "final_boundary",
                    "unsplit_rk",
                    "coordinate_remap",
                )
                if f"{completed_prefix}{name}_committed_change_j" in row
            ],
            context,
        )
        committed_owner_sum = finite_value(
            row, f"{completed_prefix}committed_owner_sum_j", context)
        require_identity(
            committed_owner_sum, owner_sum, context,
            "completed-step owner sum",
        )
        completed_reservoir_sum = ordered_sum(
            row,
            [
                f"{completed_prefix}{name}_change_j"
                for name in reservoirs
            ],
            context,
        )
        require_identity(
            finite_value(
                row, f"{completed_prefix}closure_residual_j", context),
            committed_owner_sum - completed_reservoir_sum,
            context,
            "completed-step reservoir closure",
        )
        checks += 2

    return checks


def require_mixed_physics_ledgers(run_dir: Path) -> None:
    radiation = read_csv_rows(run_dir / "radiation_budget.csv")
    sample_times = finite_column(radiation, "last_radiation_sample_s")
    held_power = finite_column(radiation, "held_atmospheric_power_w")
    relative_residual = finite_column(
        radiation, "relative_booking_residual")
    require(
        max(sample_times) > 0.0,
        f"{run_dir.name}: radiation cadence never refreshed after t=0",
    )
    require(
        any(abs(value) > 0.0 for value in held_power),
        f"{run_dir.name}: radiation produced no atmospheric forcing",
    )
    require(
        max(abs(value) for value in relative_residual) <= 1.0e-12,
        f"{run_dir.name}: radiation held-power booking did not close",
    )

    sources = read_csv_rows(
        run_dir / "thermodynamic_source_budget.csv")
    for column, label in (
        ("microphysics_heat_j", "Kessler heat"),
        ("radiation_heat_j", "radiation heat"),
        ("pbl_heat_j", "PBL heat"),
        ("pbl_requested_momentum_work_j", "PBL momentum work"),
        ("sgs_requested_momentum_work_j", "SGS momentum work"),
    ):
        values = finite_column(sources, column)
        require(
            any(abs(value) > 0.0 for value in values),
            f"{run_dir.name}: {label} remained vacuous",
        )
    require(
        max(finite_column(sources, "pbl_momentum_work_step_count")) > 0.0,
        f"{run_dir.name}: PBL momentum attribution never committed",
    )
    require(
        max(finite_column(sources, "sgs_momentum_work_step_count")) > 0.0,
        f"{run_dir.name}: SGS momentum attribution never committed",
    )


def checkpoint_sidecars(run_dir: Path) -> list[dict[str, object]]:
    sidecars = sorted((run_dir / "checkpoints").glob("*.nimbus.json"))
    require(sidecars, f"{run_dir.name}: no checkpoint sidecars")
    records = [
        json.loads(path.read_text(encoding="utf-8")) for path in sidecars
    ]
    require(
        all(record.get("boundary") == "completed_timestep"
            for record in records),
        f"{run_dir.name}: checkpoint outside a completed timestep",
    )
    return records


def final_frame(run_dir: Path) -> Path:
    frames = sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and STEP_PATTERN.fullmatch(path.name)
    )
    require(frames, f"{run_dir.name}: no field frames")
    return frames[-1]


def require_restart_suffix(continuous: Path, resumed: Path) -> None:
    for filename in LEDGERS:
        continuous_path = continuous / filename
        resumed_path = resumed / filename
        require(continuous_path.is_file(),
                f"continuous: missing {filename}")
        require(resumed_path.is_file(), f"resumed: missing {filename}")
        continuous_lines = continuous_path.read_text(
            encoding="utf-8").splitlines()
        resumed_lines = resumed_path.read_text(
            encoding="utf-8").splitlines()
        require(
            continuous_lines and resumed_lines
            and continuous_lines[0] == resumed_lines[0],
            f"{filename}: restart changed the ledger schema",
        )
        require(
            len(resumed_lines) > 1
            and continuous_lines[-(len(resumed_lines) - 1):]
            == resumed_lines[1:],
            f"{filename}: resumed ledger is not an exact continuous suffix",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("bin/nimbus_sim"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tests/fixtures/configs/qualification/gpu_report_bearing.yaml"),
    )
    parser.add_argument("--duration", type=int, default=12)
    parser.add_argument("--checkpoint", type=int, default=6)
    parser.add_argument(
        "--require-mixed-physics",
        action="store_true",
        help=(
            "require active Kessler, slab PBL, horizontal Smagorinsky, "
            "simple-grey radiation, and their device/ledger evidence"
        ),
    )
    args = parser.parse_args()
    require(args.binary.is_file(), f"missing simulator: {args.binary}")
    require(args.config.is_file(), f"missing config: {args.config}")
    require(0 < args.checkpoint < args.duration,
            "checkpoint must be positive and earlier than duration")

    with tempfile.TemporaryDirectory(
        prefix="nimbus_vulkan_restart_"
    ) as root_text:
        root = Path(root_text)
        continuous = root / "continuous"
        resumed = root / "resumed"
        run(
            args.binary, args.config, continuous,
            args.duration, args.checkpoint,
            require_mixed_physics=args.require_mixed_physics,
        )
        owner_identity_checks = require_report_owner_identities(continuous)
        if args.require_mixed_physics:
            require_mixed_physics_ledgers(continuous)
        sidecars = checkpoint_sidecars(continuous)
        branch_records = [
            record for record in sidecars
            if abs(float(record["simulation_time_s"]) - args.checkpoint)
            <= 1.0e-9
        ]
        require(len(branch_records) == 1,
                "continuous run did not publish exactly one branch checkpoint")
        branch_checkpoint = (
            continuous / "checkpoints"
            / str(branch_records[0]["checkpoint_file"])
        )
        run(
            args.binary, args.config, resumed,
            args.duration, args.checkpoint, branch_checkpoint,
            require_mixed_physics=args.require_mixed_physics,
        )
        owner_identity_checks += require_report_owner_identities(resumed)
        if args.require_mixed_physics:
            require_mixed_physics_ledgers(resumed)

        continuous_final = final_frame(continuous)
        resumed_final = final_frame(resumed)
        for field in CORE_FIELDS:
            require(
                (continuous_final / f"{field}.npy").read_bytes()
                == (resumed_final / f"{field}.npy").read_bytes(),
                f"restart changed final {field} bytes",
            )

        continuous_records = checkpoint_sidecars(continuous)
        resumed_records = checkpoint_sidecars(resumed)
        continuous_end = (
            continuous / "checkpoints"
            / str(continuous_records[-1]["checkpoint_file"])
        )
        resumed_end = (
            resumed / "checkpoints"
            / str(resumed_records[-1]["checkpoint_file"])
        )
        require(
            continuous_end.read_bytes() == resumed_end.read_bytes(),
            "restart changed the final checkpoint bytes",
        )
        require_restart_suffix(continuous, resumed)

    print(
        "[vulkan-restart] RESULT: PASS "
        f"duration={args.duration}s checkpoint={args.checkpoint}s "
        f"fields={len(CORE_FIELDS)} ledgers={len(LEDGERS)} "
        f"owner_identity_checks={owner_identity_checks} "
        f"mixed_physics={'yes' if args.require_mixed_physics else 'no'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"[vulkan-restart] RESULT: FAIL: {error}")
        raise SystemExit(1)

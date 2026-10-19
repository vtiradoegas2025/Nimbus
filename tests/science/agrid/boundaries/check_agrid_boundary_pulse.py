#!/usr/bin/env python3
"""Qualify rigid reflection and zero-gradient transmission of an A-grid pulse."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


COMMON_TOOLS = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

from check_hydrostatic_hold import (
    Float32Array,
    Frame,
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


GAMMA = 1.4


@dataclass(frozen=True)
class Thresholds:
    minimum_initial_rightgoing_fraction: float = 0.98
    minimum_rigid_energy_retention: float = 0.95
    maximum_rigid_energy_retention: float = 1.05
    minimum_rigid_leftgoing_fraction: float = 0.90
    maximum_zero_gradient_energy_retention: float = 0.15
    maximum_zero_gradient_to_rigid_energy: float = 0.20
    maximum_zero_gradient_reflected_initial_energy: float = 0.05
    maximum_zero_gradient_mass_perturbation_retention: float = 0.15
    maximum_rigid_flux_closure_residual: float = 0.02
    maximum_zero_gradient_flux_closure_residual: float = 0.02


def load_run(
    run_dir: Path,
) -> list[tuple[Frame, Mapping[str, Float32Array]]]:
    marker = read_json_object(run_dir / "nimbus_output.json")
    generation = marker.get("publication_generation")
    require(valid_generation(generation), f"{run_dir}: invalid publication generation")
    assert isinstance(generation, str)
    frames = sorted(
        (
            parse_frame(path, generation)
            for path in run_dir.iterdir()
            if path.is_dir() and STEP_DIR_PATTERN.fullmatch(path.name)
        ),
        key=lambda frame: frame.time_s,
    )
    require(len(frames) >= 50, f"{run_dir}: acoustic-flux history is incomplete")
    duration_s = frames[-1].time_s - frames[0].time_s
    require(
        100.0 <= duration_s <= 100.3 + 1.0e-9,
        f"{run_dir}: expected the first full large-step endpoint at or after 100 seconds",
    )
    validate_summary(run_dir)
    return [(frame, load_fields(frame)) for frame in frames]


def acoustic_state_metrics(
    frame: Frame,
    fields: Mapping[str, Float32Array],
    base: Mapping[tuple[int, int], tuple[float, float, float]],
) -> dict[str, float]:
    energy = 0.0
    right_energy = 0.0
    left_energy = 0.0
    mass_perturbation = 0.0
    for i, j, k in physical_indices(frame):
        pressure_base, density_base, sound_speed = base[(j, k)]
        pressure_perturbation = fields["p"].at(i, j, k) - pressure_base
        velocity = fields["u"].at(i, j, k)
        impedance_velocity = density_base * sound_speed * velocity
        denominator = 4.0 * density_base * sound_speed * sound_speed
        right_energy += (pressure_perturbation + impedance_velocity) ** 2 / denominator
        left_energy += (pressure_perturbation - impedance_velocity) ** 2 / denominator
        energy += (
            0.5 * density_base * velocity * velocity
            + 0.5 * pressure_perturbation * pressure_perturbation
              / (density_base * sound_speed * sound_speed)
        )
        mass_perturbation += fields["rho"].at(i, j, k) - density_base
    cell_volume_m3 = frame.dx_m * frame.dy_m * frame.dz_m
    energy *= cell_volume_m3
    right_energy *= cell_volume_m3
    left_energy *= cell_volume_m3
    mass_perturbation *= cell_volume_m3
    require(energy > 0.0, "acoustic pulse has zero energy")
    return {
        "energy": energy,
        "rightgoing_energy": right_energy,
        "leftgoing_energy": left_energy,
        "rightgoing_fraction": right_energy / energy,
        "leftgoing_fraction": left_energy / energy,
        "mass_perturbation": mass_perturbation,
    }


def outward_acoustic_power_w(
    frame: Frame,
    fields: Mapping[str, Float32Array],
    base: Mapping[tuple[int, int], tuple[float, float, float]],
) -> float:
    """Linear acoustic p' u power through the two physical x boundaries."""
    west_i = frame.valid_lower[0] - 1
    east_i = frame.valid_upper[0] + 1
    require(west_i >= 0 and east_i < frame.shape[0],
            "acoustic flux requires one published x ghost on each side")
    power_density_sum = 0.0
    for j in range(frame.valid_lower[1], frame.valid_upper[1] + 1):
        for k in range(frame.valid_lower[2], frame.valid_upper[2] + 1):
            pressure_base, _, _ = base[(j, k)]
            west_pressure = fields["p"].at(west_i, j, k) - pressure_base
            east_pressure = fields["p"].at(east_i, j, k) - pressure_base
            # Outward normals are -x at west and +x at east.
            power_density_sum += (
                east_pressure * fields["u"].at(east_i, j, k)
                - west_pressure * fields["u"].at(west_i, j, k)
            )
    return power_density_sum * frame.dy_m * frame.dz_m


def integrated_outward_acoustic_energy_j(
    history: Sequence[tuple[Frame, Mapping[str, Float32Array]]],
    base: Mapping[tuple[int, int], tuple[float, float, float]],
) -> float:
    powers = [outward_acoustic_power_w(frame, fields, base)
              for frame, fields in history]
    total = 0.0
    for index in range(1, len(history)):
        dt_s = history[index][0].time_s - history[index - 1][0].time_s
        require(dt_s > 0.0, "acoustic-flux frames are not strictly ordered")
        total += 0.5 * dt_s * (powers[index - 1] + powers[index])
    return total


def initial_base(
    frame: Frame,
    fields: Mapping[str, Float32Array],
) -> dict[tuple[int, int], tuple[float, float, float]]:
    base: dict[tuple[int, int], tuple[float, float, float]] = {}
    for j in range(frame.valid_lower[1], frame.valid_upper[1] + 1):
        for k in range(frame.valid_lower[2], frame.valid_upper[2] + 1):
            pressure_base = min(
                fields["p"].at(i, j, k)
                for i in range(frame.valid_lower[0], frame.valid_upper[0] + 1)
            )
            density_base = min(
                fields["rho"].at(i, j, k)
                for i in range(frame.valid_lower[0], frame.valid_upper[0] + 1)
            )
            require(pressure_base > 0.0 and density_base > 0.0, "invalid acoustic base state")
            base[(j, k)] = (
                pressure_base,
                density_base,
                math.sqrt(GAMMA * pressure_base / density_base),
            )
    return base


def max_initial_difference(
    frame: Frame,
    lhs: Mapping[str, Float32Array],
    rhs: Mapping[str, Float32Array],
) -> float:
    return max(
        abs(lhs[field].at(i, j, k) - rhs[field].at(i, j, k))
        for field in ("u", "v", "w", "rho", "p", "theta", "qv")
        for i, j, k in physical_indices(frame)
    )


def qualify(
    rigid_dir: Path,
    zero_gradient_dir: Path,
    thresholds: Thresholds = Thresholds(),
) -> dict[str, object]:
    rigid_history = load_run(rigid_dir)
    zero_history = load_run(zero_gradient_dir)
    rigid_frame0, rigid0 = rigid_history[0]
    rigid_frame1, rigid1 = rigid_history[-1]
    zero_frame0, zero0 = zero_history[0]
    zero_frame1, zero1 = zero_history[-1]
    require(rigid_frame0.shape == zero_frame0.shape, "pulse grid shapes differ")
    initial_difference = max_initial_difference(rigid_frame0, rigid0, zero0)
    require(initial_difference == 0.0, "boundary runs do not share a bitwise-identical initial state")

    base = initial_base(rigid_frame0, rigid0)
    rigid_initial = acoustic_state_metrics(rigid_frame0, rigid0, base)
    zero_initial = acoustic_state_metrics(zero_frame0, zero0, base)
    require(rigid_initial == zero_initial, "derived initial acoustic states differ")
    require(
        rigid_initial["rightgoing_fraction"] >= thresholds.minimum_initial_rightgoing_fraction,
        "seeded pulse is not predominantly right-going",
    )
    rigid_final = acoustic_state_metrics(rigid_frame1, rigid1, base)
    zero_final = acoustic_state_metrics(zero_frame1, zero1, base)

    rigid_retention = rigid_final["energy"] / rigid_initial["energy"]
    zero_retention = zero_final["energy"] / zero_initial["energy"]
    zero_to_rigid = zero_final["energy"] / rigid_final["energy"]
    zero_reflected_initial = zero_final["leftgoing_energy"] / zero_initial["energy"]
    zero_mass_retention = abs(
        zero_final["mass_perturbation"] / zero_initial["mass_perturbation"]
    )
    rigid_outward_energy = integrated_outward_acoustic_energy_j(
        rigid_history, base)
    zero_outward_energy = integrated_outward_acoustic_energy_j(
        zero_history, base)
    rigid_flux_closure = abs(
        rigid_final["energy"] + rigid_outward_energy
        - rigid_initial["energy"]
    ) / rigid_initial["energy"]
    zero_flux_closure = abs(
        zero_final["energy"] + zero_outward_energy
        - zero_initial["energy"]
    ) / zero_initial["energy"]

    require(
        thresholds.minimum_rigid_energy_retention
        <= rigid_retention
        <= thresholds.maximum_rigid_energy_retention,
        f"rigid energy retention {rigid_retention:.9g} is outside bounds",
    )
    require(
        rigid_final["leftgoing_fraction"] >= thresholds.minimum_rigid_leftgoing_fraction,
        f"rigid pulse did not reflect leftward ({rigid_final['leftgoing_fraction']:.9g})",
    )
    require(
        zero_retention <= thresholds.maximum_zero_gradient_energy_retention,
        f"zero-gradient energy retention {zero_retention:.9g} exceeds bound",
    )
    require(
        zero_to_rigid <= thresholds.maximum_zero_gradient_to_rigid_energy,
        f"zero-gradient/rigid residual-energy ratio {zero_to_rigid:.9g} exceeds bound",
    )
    require(
        zero_reflected_initial <= thresholds.maximum_zero_gradient_reflected_initial_energy,
        f"zero-gradient reflected energy {zero_reflected_initial:.9g} exceeds bound",
    )
    require(
        zero_mass_retention <= thresholds.maximum_zero_gradient_mass_perturbation_retention,
        f"zero-gradient perturbation-mass retention {zero_mass_retention:.9g} exceeds bound",
    )
    require(
        rigid_flux_closure <= thresholds.maximum_rigid_flux_closure_residual,
        f"rigid acoustic-flux closure {rigid_flux_closure:.9g} exceeds bound",
    )
    require(
        zero_flux_closure <= thresholds.maximum_zero_gradient_flux_closure_residual,
        f"zero-gradient acoustic-flux closure {zero_flux_closure:.9g} exceeds bound",
    )

    return {
        "status": "PASS",
        "scope": "production_agrid_acoustic_boundary_pulse",
        "initial_state_max_abs_difference": initial_difference,
        "metrics": {
            "initial_rightgoing_fraction": rigid_initial["rightgoing_fraction"],
            "rigid_energy_retention": rigid_retention,
            "rigid_final_leftgoing_fraction": rigid_final["leftgoing_fraction"],
            "zero_gradient_energy_retention": zero_retention,
            "zero_gradient_to_rigid_residual_energy": zero_to_rigid,
            "zero_gradient_reflected_initial_energy": zero_reflected_initial,
            "zero_gradient_mass_perturbation_retention": zero_mass_retention,
            "rigid_integrated_outward_acoustic_energy_j": rigid_outward_energy,
            "rigid_acoustic_flux_closure_residual": rigid_flux_closure,
            "zero_gradient_integrated_outward_acoustic_energy_j":
                zero_outward_energy,
            "zero_gradient_acoustic_flux_closure_residual": zero_flux_closure,
            "initial_acoustic_energy_j": rigid_initial["energy"],
            "history_frame_count": len(rigid_history),
        },
        "thresholds": asdict(thresholds),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rigid_run", type=Path)
    parser.add_argument("zero_gradient_run", type=Path)
    args = parser.parse_args(argv)
    try:
        result = qualify(args.rigid_run, args.zero_gradient_run)
    except QualificationError as error:
        parser.exit(1, f"[agrid-boundary-pulse] FAIL: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

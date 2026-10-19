#!/usr/bin/env python3
"""Qualify the native warm-bubble response of the production A-grid core."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import sys
from pathlib import Path
from typing import Sequence


COMMON_TOOLS = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

from check_hydrostatic_hold import (
    QualificationError,
    STEP_DIR_PATTERN,
    eos_relative_residual,
    inventories,
    load_fields,
    parse_frame,
    physical_indices,
    read_json_object,
    relative_drift,
    require,
    valid_generation,
    validate_summary,
)


GRAVITY_MS2 = 9.81


@dataclass(frozen=True)
class Thresholds:
    minimum_peak_theta_anomaly_k: float = 0.75
    maximum_peak_theta_anomaly_k: float = 2.1
    minimum_thermal_velocity_correlation: float = 0.90
    minimum_buoyancy_impulse_ratio: float = 0.60
    maximum_buoyancy_impulse_ratio: float = 1.20
    minimum_peak_updraft_ms: float = 0.03
    maximum_peak_updraft_ms: float = 0.12
    maximum_downdraft_magnitude_ms: float = 5.0e-3
    maximum_eos_relative_residual: float = 1.0e-6
    maximum_inventory_relative_drift: float = 5.0e-7
    maximum_total_energy_relative_drift: float = 5.0e-7


def correlation(lhs: list[float], rhs: list[float]) -> float:
    require(len(lhs) == len(rhs) and len(lhs) > 1, "correlation inputs are invalid")
    lhs_mean = math.fsum(lhs) / len(lhs)
    rhs_mean = math.fsum(rhs) / len(rhs)
    covariance = math.fsum(
        (left - lhs_mean) * (right - rhs_mean)
        for left, right in zip(lhs, rhs)
    )
    lhs_variance = math.fsum((value - lhs_mean) ** 2 for value in lhs)
    rhs_variance = math.fsum((value - rhs_mean) ** 2 for value in rhs)
    require(lhs_variance > 0.0 and rhs_variance > 0.0, "correlation field has zero variance")
    return covariance / math.sqrt(lhs_variance * rhs_variance)


def qualify_run(
    run_dir: Path,
    thresholds: Thresholds = Thresholds(),
) -> dict[str, object]:
    marker_path = run_dir / "nimbus_output.json"
    marker = read_json_object(marker_path)
    require(marker.get("schema") == "nimbus.output", f"{marker_path}: wrong schema")
    require(marker.get("publication_state") == "ready", f"{marker_path}: run is not ready")
    generation = marker.get("publication_generation")
    require(valid_generation(generation), f"{marker_path}: invalid publication generation")
    assert isinstance(generation, str)

    paths = sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and STEP_DIR_PATTERN.fullmatch(path.name)
    )
    require(len(paths) == 2, f"{run_dir}: expected exactly two frames")
    initial_frame, final_frame = sorted(
        (parse_frame(path, generation) for path in paths),
        key=lambda frame: frame.time_s,
    )
    require(abs(initial_frame.time_s) <= 1.0e-9, "initial frame is not at t=0")
    duration_s = final_frame.time_s - initial_frame.time_s
    require(math.isclose(duration_s, 2.0, abs_tol=1.0e-9), "expected a two-second response")
    require(initial_frame.shape == final_frame.shape, "initial/final shapes differ")
    validate_summary(run_dir)

    initial = load_fields(initial_frame)
    final = load_fields(final_frame)
    indices = list(physical_indices(initial_frame))
    require(indices, "physical domain is empty")

    theta_by_level: dict[int, list[float]] = {}
    for i, j, k in indices:
        theta_by_level.setdefault(k, []).append(initial["theta"].at(i, j, k))
    level_background = {
        level: min(values) for level, values in theta_by_level.items()
    }
    anomalies = [
        initial["theta"].at(i, j, k) - level_background[k]
        for i, j, k in indices
    ]
    final_w = [final["w"].at(i, j, k) for i, j, k in indices]
    peak_anomaly = max(anomalies)
    require(
        thresholds.minimum_peak_theta_anomaly_k
        <= peak_anomaly
        <= thresholds.maximum_peak_theta_anomaly_k,
        f"peak sampled thermal anomaly {peak_anomaly:.9g} K is outside the expected range",
    )
    initial_max_wind = max(
        math.sqrt(
            initial["u"].at(i, j, k) ** 2
            + initial["v"].at(i, j, k) ** 2
            + initial["w"].at(i, j, k) ** 2
        )
        for i, j, k in indices
    )
    require(initial_max_wind <= 1.0e-7, "bubble did not begin from rest")

    thermal_correlation = correlation(anomalies, final_w)
    require(
        thermal_correlation >= thresholds.minimum_thermal_velocity_correlation,
        f"thermal/updraft correlation {thermal_correlation:.9g} is too weak",
    )
    core = [
        ((i, j, k), anomaly)
        for (i, j, k), anomaly in zip(indices, anomalies)
        if anomaly > 0.05
    ]
    require(len(core) >= 8, "warm-bubble core is under-resolved")
    measured_core_impulse = math.fsum(
        final["w"].at(i, j, k) for (i, j, k), _ in core
    ) / len(core)
    expected_core_impulse = math.fsum(
        GRAVITY_MS2 * anomaly / initial["theta"].at(i, j, k) * duration_s
        for (i, j, k), anomaly in core
    ) / len(core)
    require(expected_core_impulse > 0.0, "thermal has no positive buoyancy impulse")
    impulse_ratio = measured_core_impulse / expected_core_impulse
    require(
        thresholds.minimum_buoyancy_impulse_ratio
        <= impulse_ratio
        <= thresholds.maximum_buoyancy_impulse_ratio,
        f"measured/expected buoyancy impulse {impulse_ratio:.9g} is outside bounds",
    )

    peak_updraft = max(final_w)
    minimum_vertical_velocity = min(final_w)
    require(
        thresholds.minimum_peak_updraft_ms
        <= peak_updraft
        <= thresholds.maximum_peak_updraft_ms,
        f"peak updraft {peak_updraft:.9g} m/s is outside bounds",
    )
    require(
        minimum_vertical_velocity >= -thresholds.maximum_downdraft_magnitude_ms,
        f"spurious downdraft {minimum_vertical_velocity:.9g} m/s exceeds bound",
    )

    for field in ("rho", "p", "theta"):
        require(
            min(final[field].at(i, j, k) for i, j, k in indices) > 0.0,
            f"final {field} is non-positive",
        )
    require(
        min(final["qv"].at(i, j, k) for i, j, k in indices) >= 0.0,
        "final qv is negative",
    )
    max_eos = max(
        eos_relative_residual(
            initial["rho"], initial["p"], initial["theta"], initial_frame
        ),
        eos_relative_residual(
            final["rho"], final["p"], final["theta"], final_frame
        ),
    )
    require(
        max_eos <= thresholds.maximum_eos_relative_residual,
        f"EOS residual {max_eos:.9g} exceeds bound",
    )

    initial_inventory = inventories(initial, initial_frame)
    final_inventory = inventories(final, final_frame)
    drifts = {
        name: relative_drift(initial_inventory[name], final_inventory[name])
        for name in ("mass", "rho_theta", "water", "total_energy")
    }
    for name in ("mass", "rho_theta", "water"):
        require(
            drifts[name] <= thresholds.maximum_inventory_relative_drift,
            f"{name} drift {drifts[name]:.9g} exceeds bound",
        )
    require(
        drifts["total_energy"] <= thresholds.maximum_total_energy_relative_drift,
        f"total-energy drift {drifts['total_energy']:.9g} exceeds bound",
    )

    return {
        "status": "PASS",
        "scope": "native_warm_bubble_production_integrator",
        "duration_s": duration_s,
        "physical_cell_count": len(indices),
        "metrics": {
            "peak_initial_theta_anomaly_k": peak_anomaly,
            "thermal_velocity_correlation": thermal_correlation,
            "core_measured_buoyancy_impulse_ms": measured_core_impulse,
            "core_expected_buoyancy_impulse_ms": expected_core_impulse,
            "buoyancy_impulse_ratio": impulse_ratio,
            "peak_updraft_ms": peak_updraft,
            "minimum_vertical_velocity_ms": minimum_vertical_velocity,
            "max_eos_relative_residual": max_eos,
            "inventory_relative_drifts": drifts,
        },
        "thresholds": asdict(thresholds),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = qualify_run(args.run_dir)
    except QualificationError as error:
        parser.exit(1, f"[agrid-warm-bubble] FAIL: {error}\n")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

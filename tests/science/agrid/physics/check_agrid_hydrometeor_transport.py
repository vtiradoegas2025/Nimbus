#!/usr/bin/env python3
"""Qualify nonzero A-grid rain-water transport through the production step."""

from __future__ import annotations

import argparse
import csv
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
    load_fields,
    parse_frame,
    physical_indices,
    read_json_object,
    read_npy_float32,
    require,
    valid_generation,
    validate_summary,
)


def qualify_transport(run_dir: Path) -> dict[str, object]:
    marker_path = run_dir / "nimbus_output.json"
    marker = read_json_object(marker_path)
    generation = marker.get("publication_generation")
    require(valid_generation(generation), f"{marker_path}: invalid publication generation")
    assert isinstance(generation, str)
    paths = sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and STEP_DIR_PATTERN.fullmatch(path.name)
    )
    require(len(paths) == 2, f"{run_dir}: expected exactly two frames")
    frames = sorted(
        (parse_frame(path, generation) for path in paths),
        key=lambda frame: frame.time_s,
    )
    initial_frame, final_frame = frames
    require(abs(initial_frame.time_s) <= 1.0e-9, "initial frame is not t=0")
    require(final_frame.time_s >= 20.0 - 1.0e-9, "transport run is shorter than 20 seconds")
    require(initial_frame.shape == final_frame.shape, "initial/final shape differs")
    validate_summary(run_dir)

    initial = load_fields(initial_frame)
    final = load_fields(final_frame)
    initial_qr = read_npy_float32(initial_frame.path / "qr.npy")
    final_qr = read_npy_float32(final_frame.path / "qr.npy")
    require(initial_qr.shape == initial_frame.shape, "initial qr shape differs from manifest")
    require(final_qr.shape == final_frame.shape, "final qr shape differs from manifest")

    def weighted_metrics(fields, qr, frame):
        mass_terms: list[float] = []
        rain_terms: list[float] = []
        x_terms: list[float] = []
        q_values: list[float] = []
        u_terms: list[float] = []
        for i, j, k in physical_indices(frame):
            rho_value = fields["rho"].at(i, j, k)
            qr_value = qr.at(i, j, k)
            require(rho_value > 0.0, "non-positive dry density")
            require(qr_value >= 0.0, "negative rain-water mixing ratio")
            rain_mass = rho_value * qr_value
            mass_terms.append(rho_value)
            rain_terms.append(rain_mass)
            x_terms.append(rain_mass * (float(i) + 0.5) * frame.dx_m)
            q_values.append(qr_value)
            u_terms.append(fields["u"].at(i, j, k))
        rain_inventory = math.fsum(rain_terms)
        require(rain_inventory > 0.0, "rain-water inventory is zero")
        return {
            "dry_mass": math.fsum(mass_terms),
            "rain_inventory": rain_inventory,
            "rain_centroid_x_m": math.fsum(x_terms) / rain_inventory,
            "peak_qr_kgkg": max(q_values),
            "mean_u_ms": math.fsum(u_terms) / len(u_terms),
        }

    before = weighted_metrics(initial, initial_qr, initial_frame)
    after = weighted_metrics(final, final_qr, final_frame)
    rain_drift = abs(after["rain_inventory"] - before["rain_inventory"]) / before["rain_inventory"]
    dry_mass_drift = abs(after["dry_mass"] - before["dry_mass"]) / before["dry_mass"]
    displacement = after["rain_centroid_x_m"] - before["rain_centroid_x_m"]
    max_qr_change = max(
        abs(final_qr.at(i, j, k) - initial_qr.at(i, j, k))
        for i, j, k in physical_indices(initial_frame)
    )

    try:
        with (run_dir / "budget.csv").open("r", encoding="utf-8", newline="") as handle:
            budget_rows = list(csv.DictReader(handle))
        with (run_dir / "moisture_budget.csv").open("r", encoding="utf-8", newline="") as handle:
            moisture_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise QualificationError(f"cannot read conservation ledgers: {error}") from error
    boundary_rows = [row for row in budget_rows if row.get("stage") == "bc"]
    total_rows = [row for row in budget_rows if row.get("stage") == "total_step"]
    moisture_total_rows = [row for row in moisture_rows if row.get("stage") == "total_step"]
    require(len(boundary_rows) == 100, f"expected 100 boundary ledger rows, found {len(boundary_rows)}")
    require(len(total_rows) == 100, f"expected 100 total-step ledger rows, found {len(total_rows)}")
    require(len(moisture_total_rows) == 100, f"expected 100 moisture ledger rows, found {len(moisture_total_rows)}")
    try:
        max_boundary_dry_mass = max(abs(float(row["d_dry_mass"])) for row in boundary_rows)
        max_boundary_water = max(abs(float(row["d_total_water"])) for row in boundary_rows)
        max_step_dry_mass_relative = max(abs(float(row["rel_dry_mass"])) for row in total_rows)
        max_step_water_relative = max(abs(float(row["rel_total_water"])) for row in total_rows)
        max_moisture_step_relative = max(abs(float(row["rel_total_water"])) for row in moisture_total_rows)
    except (KeyError, TypeError, ValueError) as error:
        raise QualificationError(f"malformed conservation ledger: {error}") from error

    require(abs(before["mean_u_ms"] - 10.0) <= 1.0e-6, "initial mean u is not 10 m/s")
    require(rain_drift <= 2.0e-6, f"rain inventory drift {rain_drift:.9g} exceeds 2e-6")
    require(dry_mass_drift <= 2.0e-6, f"dry-mass drift {dry_mass_drift:.9g} exceeds 2e-6")
    require(100.0 <= displacement <= 300.0, f"rain centroid displacement {displacement:.9g} m is implausible")
    require(max_qr_change >= 1.0e-5, "rain field did not materially move")
    require(max_boundary_dry_mass == 0.0, "periodic/rigid boundary stage changed dry mass")
    require(max_boundary_water == 0.0, "periodic/rigid boundary stage changed water")
    require(max_step_dry_mass_relative <= 2.0e-6, "per-step dry-mass ledger does not close")
    require(max_step_water_relative <= 2.0e-6, "per-step total-water ledger does not close")
    require(max_moisture_step_relative <= 2.0e-6, "moisture ledger does not close")
    require(
        after["peak_qr_kgkg"] <= before["peak_qr_kgkg"] * (1.0 + 1.0e-5),
        "transport created a new rain-water maximum",
    )
    return {
        "status": "PASS",
        "duration_s": final_frame.time_s - initial_frame.time_s,
        "initial": before,
        "final": after,
        "metrics": {
            "dry_mass_relative_drift": dry_mass_drift,
            "rain_inventory_relative_drift": rain_drift,
            "rain_centroid_displacement_x_m": displacement,
            "max_qr_change_kgkg": max_qr_change,
            "max_boundary_stage_dry_mass_change": max_boundary_dry_mass,
            "max_boundary_stage_water_change": max_boundary_water,
            "max_step_dry_mass_relative_change": max_step_dry_mass_relative,
            "max_step_total_water_relative_change": max_step_water_relative,
            "max_moisture_ledger_relative_change": max_moisture_step_relative,
        },
        "thresholds": {
            "max_dry_mass_relative_drift": 2.0e-6,
            "max_rain_inventory_relative_drift": 2.0e-6,
            "max_step_ledger_relative_change": 2.0e-6,
            "boundary_stage_mass_and_water_change": 0.0,
            "rain_centroid_displacement_x_m": [100.0, 300.0],
            "minimum_max_qr_change_kgkg": 1.0e-5,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = qualify_transport(args.run_dir)
    except QualificationError as error:
        parser.exit(1, f"[agrid-hydrometeor-transport] FAIL: {error}\n")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

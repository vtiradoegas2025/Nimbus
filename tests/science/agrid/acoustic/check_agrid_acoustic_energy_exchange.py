#!/usr/bin/env python3
"""Qualify standing-wave compression/kinetic energy exchange on the A-grid."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Sequence
from check_agrid_production_acoustic_wave import GAMMA, QualificationError, frames, require

def components(frame, reference, fields) -> dict[str, float]:
    compressive = kinetic = 0.0
    for j in range(frame.valid_lower[1], frame.valid_upper[1] + 1):
        for k in range(frame.valid_lower[2], frame.valid_upper[2] + 1):
            indices = range(frame.valid_lower[0], frame.valid_upper[0] + 1)
            count = frame.valid_upper[0] - frame.valid_lower[0] + 1
            pressure_base = math.fsum(reference["p"].at(i, j, k) for i in indices) / count
            density_base = math.fsum(reference["rho"].at(i, j, k) for i in range(frame.valid_lower[0], frame.valid_upper[0] + 1)) / count
            sound_speed = math.sqrt(GAMMA * pressure_base / density_base)
            pressure_mean = math.fsum(fields["p"].at(i, j, k) for i in range(frame.valid_lower[0], frame.valid_upper[0] + 1)) / count
            velocity_means = {name: math.fsum(fields[name].at(i, j, k) for i in range(frame.valid_lower[0], frame.valid_upper[0] + 1)) / count for name in ("u", "v", "w")}
            for i in range(frame.valid_lower[0], frame.valid_upper[0] + 1):
                pressure_prime = fields["p"].at(i, j, k) - pressure_mean
                compressive += 0.5 * pressure_prime * pressure_prime / (density_base * sound_speed * sound_speed)
                kinetic += 0.5 * density_base * sum((fields[name].at(i, j, k) - velocity_means[name]) ** 2 for name in ("u", "v", "w"))
    total = compressive + kinetic
    require(total > 0.0, "zero acoustic perturbation energy")
    return {"compressive": compressive, "kinetic": kinetic, "total": total, "kinetic_fraction": kinetic / total}

def qualify(run: Path) -> dict[str, object]:
    initial_frame, initial, final_frame, final = frames(run)
    before = components(initial_frame, initial, initial)
    after = components(final_frame, initial, final)
    drift = abs(after["total"] - before["total"]) / before["total"]
    require(before["kinetic_fraction"] < 1.0e-8, "standing wave did not begin as compression energy")
    require(after["kinetic_fraction"] > 0.98, "quarter-period state did not convert to kinetic energy")
    require(after["compressive"] / before["compressive"] < 0.02, "compression energy did not leave its initial reservoir")
    require(drift < 0.01, "total acoustic perturbation energy does not close")
    return {"status": "PASS", "initial": before, "quarter_period": after, "total_energy_relative_drift": drift}

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = qualify(args.run_dir)
    except QualificationError as error:
        parser.exit(1, f"[agrid-acoustic-energy-exchange] FAIL: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

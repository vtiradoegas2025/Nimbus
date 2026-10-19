#!/usr/bin/env python3
"""Qualify excited-wave A-grid acoustic-substep/CFL refinement."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Sequence
from check_agrid_production_acoustic_wave import QualificationError, metrics, require

def qualify(runs: dict[str, Path]) -> dict[str, object]:
    results = {name: metrics(path) for name, path in runs.items()}
    for name, result in results.items():
        require(result["phase_error_rad"] < 0.03, f"{name}: phase error too large")
        require(0.98 < result["amplitude_ratio"] < 1.03, f"{name}: amplitude ratio out of range")
        require(result["acoustic_energy_relative_drift"] < 0.01, f"{name}: energy drift too large")
        require(result["dry_mass_relative_drift"] < 2.0e-6, f"{name}: mass drift too large")
    errors = [results[name]["phase_error_rad"] for name in ("3", "6", "12")]
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    require(min(ratios) > 1.1, "phase error does not improve with acoustic-substep refinement")
    return {"status": "PASS", "acoustic_substeps": [3, 6, 12], "phase_error_refinement_ratios": ratios, "runs": results}

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_3", type=Path)
    parser.add_argument("run_6", type=Path)
    parser.add_argument("run_12", type=Path)
    args = parser.parse_args(argv)
    try:
        result = qualify({"3": args.run_3, "6": args.run_6, "12": args.run_12})
    except QualificationError as error:
        parser.exit(1, f"[agrid-production-acoustic-cfl] FAIL: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

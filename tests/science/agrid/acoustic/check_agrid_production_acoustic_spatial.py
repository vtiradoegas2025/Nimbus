#!/usr/bin/env python3
"""Qualify production A-grid acoustic-wave spatial refinement."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Sequence
from check_agrid_production_acoustic_wave import QualificationError, metrics, require

def qualify(runs: dict[str, Path]) -> dict[str, object]:
    results = {name: metrics(path) for name, path in runs.items()}
    require([results[n]["physical_nx"] for n in ("coarse", "medium", "fine")] == [16, 32, 64], "unexpected physical resolution sequence")
    for name, result in results.items():
        require(result["phase_error_rad"] < 0.05, f"{name}: phase error too large")
        require(0.98 < result["amplitude_ratio"] < 1.03, f"{name}: amplitude ratio out of range")
        require(result["acoustic_energy_relative_drift"] < 0.01, f"{name}: acoustic energy drift too large")
        require(result["dry_mass_relative_drift"] < 2.0e-6, f"{name}: mass drift too large")
    errors = [results[n]["phase_error_rad"] for n in ("coarse", "medium", "fine")]
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    require(min(ratios) > 3.0, "phase error does not show strong spatial refinement")
    return {"status": "PASS", "physical_nx": [16, 32, 64], "phase_error_refinement_ratios": ratios, "runs": results}

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("medium", type=Path)
    parser.add_argument("fine", type=Path)
    args = parser.parse_args(argv)
    try:
        result = qualify({"coarse": args.coarse, "medium": args.medium, "fine": args.fine})
    except QualificationError as error:
        parser.exit(1, f"[agrid-production-acoustic-spatial] FAIL: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

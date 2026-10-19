#!/usr/bin/env python3
"""Check real held-radiation response and energy booking after the 300-second refresh."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


def identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(row: dict) -> dict | None:
    axes = dict(part.split("=", 1) for part in row["tuple"].split("|"))
    if axes["radiation"] == "none":
        return None
    output = Path(row["configs"]["metal"]["path"]).parent / "metal-1"
    path = output / "radiation_budget.csv"
    with path.open(newline="") as stream:
        values = list(csv.DictReader(stream))
    if len(values) != 3010:
        raise RuntimeError("long radiation event omitted a model step")
    before = next((item for item in reversed(values) if float(item["sim_time_s"]) < 299.9), None)
    after = next((item for item in values if float(item["sim_time_s"]) >= 300.0), None)
    if before is None or after is None or \
            abs(float(after["last_radiation_sample_s"]) - 300.0) > 1e-6:
        raise RuntimeError("long radiation event did not publish a 300-second held source")
    response = abs(float(after["held_atmospheric_power_w"]) -
                   float(before["held_atmospheric_power_w"]))
    booking = max(abs(float(item["relative_booking_residual"])) for item in values)
    shortwave = max(abs(float(item["shortwave_relative_power_residual"])) for item in values)
    if not all(map(math.isfinite, (response, booking, shortwave))) or \
            response <= 1.0 or booking > 1e-9 or shortwave > 1e-9:
        raise RuntimeError("long radiation event has no physical response or violates booked energy closure")
    return {"id": row["id"], "scheme": axes["radiation"], "status": "pass",
            "heldPowerResponseW": response,
            "maximumRelativeBookingResidual": booking,
            "maximumRelativeShortwaveResidual": shortwave,
            "radiationBudget": identity(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if summary.get("status") != "complete" or len(summary.get("results", [])) != 3 or \
            any(row["status"] != "pass" for row in summary["results"]):
        raise RuntimeError("complete qualified long-cadence evidence is required")
    results = [value for row in summary["results"] if (value := check(row)) is not None]
    if {row["scheme"] for row in results} != {"rrtmg", "simple_grey"}:
        raise RuntimeError("both native radiation source families require a literal response")
    report = {"schema": "nimbus.generated-metal-cadence-response.v1", "status": "pass",
              "claimBoundary": "Bounded physical held-source response and radiation energy booking for two "
              "canonical qualified long-window Metal profiles.",
              "summary": identity(args.summary.resolve()), "results": results}
    if args.output.exists():
        parser.error("use a fresh output path")
    write_json_atomic(args.output.resolve(), report)
    print("qualified generated Metal source refresh: RRTMG and simple-grey response passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bound a Bunkers-left lifetime preflight grid-size threshold without stepping."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.enumerate_flat_boundary_family import replace_once
from scripts.qualification.materialize_retained_fixtures import identity
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


BASE = ROOT / "tests/fixtures/configs/test/domain_lifetime_preflight.yaml"
VALUES = tuple(range(160, 201, 2))


def validate_results(rows: list[dict]) -> int:
    if [row.get("ny") for row in rows] != list(VALUES) or \
            any(row.get("status") not in ("pass", "rejected") for row in rows):
        raise RuntimeError("Bunkers-left numeric probe omitted, duplicated or misordered a grid value")
    passes = [row["ny"] for row in rows if row["status"] == "pass"]
    if not passes or rows[0]["status"] != "rejected" or rows[-1]["status"] != "pass":
        raise RuntimeError("Bunkers-left numeric probe lacks both threshold sides")
    first = min(passes)
    if any((row["status"] == "pass") != (row["ny"] >= first) for row in rows):
        raise RuntimeError("Bunkers-left numeric probe is nonmonotone in the reviewed interval")
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    source = replace_once(BASE.read_text(), "    motion_source: bunkers_right\n",
                          "    motion_source: bunkers_left\n")
    inputs = {"binary": identity(binary), "base": identity(BASE),
              "probe": identity(Path(__file__)),
              "plannerSource": identity(ROOT / "src/core/infra/domain_motion.cpp"),
              "loaderSource": identity(ROOT / "src/init/config/v1_loader.cpp")}
    rows = []
    for ny in VALUES:
        config = output / f"bunkers_left_ny_{ny}.yaml"
        config.write_text(replace_once(source, "  ny: 160\n", f"  ny: {ny}\n"))
        result = subprocess.run([str(binary), "--config", str(config), "--validate-only"],
                                cwd=ROOT, text=True, capture_output=True,
                                timeout=30, check=False)
        log = output / f"bunkers_left_ny_{ny}.log"
        log.write_text(result.stdout + result.stderr)
        lines = [line for line in log.read_text().splitlines() if line.startswith("[VALIDATE]")]
        if not lines or "bunkers_left/" not in log.read_text():
            raise RuntimeError("Bunkers-left numeric probe did not reach the intended planner")
        status = "pass" if result.returncode == 0 and \
            any(line.startswith("[VALIDATE] PASS:") for line in lines) else "rejected"
        if status == "rejected" and result.returncode == 0:
            raise RuntimeError("Bunkers-left preflight has an ambiguous successful exit")
        rows.append({"ny": ny, "id": f"planner:bunkers_left:ny={ny}",
                     "config": identity(config), "log": identity(log),
                     "status": status, "returncode": result.returncode,
                     "validationLines": lines})
    first = validate_results(rows)
    if any(identity(ROOT / entry["path"]) != entry for entry in inputs.values()):
        raise RuntimeError("Bunkers-left numeric probe source or executable changed")
    report = {"schema": "nimbus.bunkers-left-lifetime-grid-threshold.v1",
              "status": "bounded_preflight_pass",
              "claimBoundary": "Fixed 200xnyx50 exploratory lifetime-planner preflight across ny=160..200 step2; not a physical storm simulation, general envelope or Ready claim.",
              "inputs": inputs, "caseCount": len(rows), "gridNyValues": VALUES,
              "firstPassingNy": first, "lastRejectedNy": first - 2,
              "cases": rows, "I1Complete": False, "wholeBackendReady": False}
    write_json_atomic(output / "summary.json", report)
    print(json.dumps({"status": report["status"], "cases": len(rows),
                      "lastRejectedNy": first - 2, "firstPassingNy": first}))


if __name__ == "__main__":
    main()

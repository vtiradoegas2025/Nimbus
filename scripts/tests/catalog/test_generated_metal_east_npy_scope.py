#!/usr/bin/env python3
"""Require the one-digest east-NPY public scope and its guarded neighbors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
QUALIFIER = ROOT / "scripts/qualification/qualify_flat_east_npy_public.py"
GENERATOR = ROOT / "scripts/qualification/generate_metal_east_npy_scope.py"


def run(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=180, check=False)
    if completed.returncode != 0:
        raise RuntimeError("east NPY maintained public gate failed:\n" + completed.stdout[-3000:])
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--helper", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    helper = args.helper.resolve(strict=True)
    verified = run([sys.executable, str(GENERATOR), "--helper", str(helper), "--verify"])
    with tempfile.TemporaryDirectory(prefix="nimbus-east-npy-public-") as temporary:
        out = Path(temporary) / "admission"
        output = run([sys.executable, str(QUALIFIER), "--binary", str(binary),
                      "--output", str(out), "--admission-only"])
        report = json.loads((out / "summary.json").read_text())
        admissions = report["admission"]
        negatives = report["cliNegatives"]
        if report["status"] != "bounded_admission_pass" or len(admissions) != 13 or \
                len(negatives) != 7 or not admissions["metal"]["accepted"] or \
                any(admissions[name]["accepted"] for name in (
                    "netcdf", "zfpHigh", "npyAsync", "eastZeroGradient", "eastRigid",
                    "extraWestRadiative", "topRigid", "bottomSlip", "centeredTrigger")) or \
                any(item["accepted"] for item in negatives.values()):
            raise RuntimeError("east NPY public admission/negative accounting changed")
    print(verified.strip())
    print(output.strip())
    print("[generated-metal-east-npy] PASS: one admitted NPY digest, nine guarded profiles, seven CLI negatives")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run existing native policy gates directly without a shared Make relink."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    report = args.report_output.resolve()
    if raw.exists() or report.exists():
        parser.error("raw and report targets must be new")
    simulator = ROOT / "bin/nimbus_sim"
    helper = ROOT / "bin/nimbus_native_admission_batch"
    identities = {
        "simulatorSha256": sha256(simulator),
        "helperSha256": sha256(helper),
        "runnerSha256": sha256(Path(__file__)),
    }
    raw.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    cases = [
        ("east_loader", [str(ROOT / "bin/test_init_config_v1"), "[generated-metal-east-npy]"]),
        ("zfp_catalog", [str(ROOT / "bin/test_core_scheme_catalog"), "[generated-metal-zfp]"]),
        ("east_scope", ["python3", "scripts/tests/catalog/test_generated_metal_east_npy_scope.py",
                        "--binary", str(simulator), "--helper", str(helper)]),
        ("explicit_zfp", ["python3", "scripts/tests/catalog/test_qualified_zfp_explicit_tolerances.py",
                          "--binary", str(simulator)]),
        ("generated_public", ["python3", "scripts/tests/catalog/test_generated_metal_public_scope.py",
                              "--binary", str(simulator)]),
        ("generated_cli", ["python3", "scripts/tests/catalog/test_generated_metal_cli_scope.py",
                           "--binary", str(simulator)]),
    ]
    results = []
    for name, command in cases:
        completed = subprocess.run(command, cwd=ROOT, env=environment,
                                   capture_output=True, text=True,
                                   timeout=180, check=False)
        log = raw / f"{name}.log"
        log.write_text(completed.stdout + "\n" + completed.stderr,
                       encoding="utf-8")
        results.append({"name": name, "command": command,
                        "exitCode": completed.returncode,
                        "logSha256": sha256(log)})
        if completed.returncode != 0:
            break
    if (sha256(simulator) != identities["simulatorSha256"] or
            sha256(helper) != identities["helperSha256"] or
            sha256(Path(__file__)) != identities["runnerSha256"]):
        raise RuntimeError("native binary or runner changed during guarded regression")
    status = "pass" if len(results) == len(cases) and all(
        row["exitCode"] == 0 for row in results) else "fail"
    summary = {
        "schema": "nimbus.policy11-guarded-native-regression.v1",
        "status": status,
        "claimBoundary": "Existing generated-Metal public/CLI/east and explicit-ZFP gates only; radar v4 public admitted:false.",
        **identities,
        "cases": results,
        "rawRoot": str(raw),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"{status}: {len(results)}/{len(cases)} guarded native cases")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Preserve exact current-source SGS component-test outcomes, including red suites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


INPUTS = (
    "bin/nimbus_sim",
    "bin/test_physics_turbulence_sgs",
    "bin/test_physics_sgs_apply_runtime_state",
    "bin/test_physics_turbulence_runtime_state",
    "tests/physics/test_turbulence_sgs.cpp",
    "tests/physics/test_sgs_apply_runtime_state.cpp",
    "tests/physics/test_turbulence_runtime_state.cpp",
    "src/turbulence/base/eddy_viscosity.cpp",
    "src/turbulence/base/collocated_eddy_viscosity.cpp",
    "src/core/orchestration/dynamics/sgs_apply.cpp",
    "src/turbulence/runtime.cpp",
    "include/turbulence/scheme_descriptor.hpp",
    "build/tests/.build-flags",
)
RUNS = (
    ("analytic_full", "bin/test_physics_turbulence_sgs", None),
    ("sgs_apply_full", "bin/test_physics_sgs_apply_runtime_state", None),
    ("runtime_state_full", "bin/test_physics_turbulence_runtime_state", None),
    ("runtime_invalid_candidate", "bin/test_physics_turbulence_runtime_state",
     "turbulence runtime rejects invalid candidate transactionally"),
    ("runtime_refresh_transaction", "bin/test_physics_turbulence_runtime_state",
     "turbulence refresh reuses two transactional field sets"),
    ("runtime_invalid_cadence", "bin/test_physics_turbulence_runtime_state",
     "SGS cadence rejects unrepresentable progress before publication"),
)


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-runtime-pass", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {path: identity(ROOT / path) for path in INPUTS}
    inputs["auditor"] = identity(Path(__file__))
    rows = []
    for name, binary, selected in RUNS:
        command = [str(ROOT / binary)]
        if selected is not None:
            command.append(selected)
        result = subprocess.run(command, cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=45, check=False)
        log = output / f"{name}.log"
        log.write_text(result.stdout)
        rows.append({"name": name, "binary": binary, "filter": selected,
                     "returncode": result.returncode, "log": identity(log)})
    status = {row["name"]: row["returncode"] for row in rows}
    if status != {"analytic_full": 0, "sgs_apply_full": 0,
                  "runtime_state_full": 0 if args.expect_runtime_pass else 1,
                  "runtime_invalid_candidate": 0,
                  "runtime_refresh_transaction": 0, "runtime_invalid_cadence": 0}:
        raise RuntimeError("SGS component-test outcomes changed; review logs before reporting")
    runtime_log = (output / "runtime_state_full.log").read_text()
    if args.expect_runtime_pass:
        if "All tests passed (122 assertions in 13 test cases)" not in runtime_log:
            raise RuntimeError("SGS runtime full-suite pass has an unexpected assertion count")
    elif "gpu_implementation_coverage == tmv::SchemeCoverage::None" not in runtime_log or \
            "4 == 2" not in runtime_log:
        raise RuntimeError("SGS runtime full-suite failure changed cause")
    if any(identity(ROOT / path) != inputs[path] for path in INPUTS) or \
            identity(Path(__file__)) != inputs["auditor"]:
        raise RuntimeError("SGS component source/binary changed during audit")
    report = {"schema": "nimbus.vertical-only-sgs-component-audit.v1",
              "status": "all_component_suites_pass" if args.expect_runtime_pass
                        else "focused_pass_full_runtime_suite_red",
              "claimBoundary": "Current-source analytic SGS, apply and runtime-state component suites pass; no public qualified vertical-only or whole-backend Ready claim."
              if args.expect_runtime_pass else
              "Current-source analytic SGS and apply suites pass; focused runtime transaction tests pass; full runtime suite has one descriptor expectation failure. No blanket component or qualified Ready claim.",
              "inputs": inputs, "runs": rows,
              "fullSuiteFailure": None if args.expect_runtime_pass else {
                  "test": "Turbulence capabilities are complete without concrete built-ins",
                  "observed": "TKE gpu_implementation_coverage Covered (4)",
                  "expectedByTest": "None (2)",
                  "passedCases": 12, "failedCases": 1,
                  "passedAssertions": 121, "failedAssertions": 1},
              "runtimeFullSuite": {"cases": 13, "assertions": 122,
                                   "passed": args.expect_runtime_pass},
              "I1Complete": False, "wholeBackendReady": False}
    write_json_atomic(output / "summary.json", report)
    print(json.dumps({"status": report["status"], "runs": len(rows)}))


if __name__ == "__main__":
    main()

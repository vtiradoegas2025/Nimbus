#!/usr/bin/env python3
"""Bound the one-east-radiative flat C-grid admission and continuation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.enumerate_flat_boundary_family import family_id
from scripts.qualification.qualify_flat_mixed_boundary_metal import (
    exact_ledgers,
    identity,
    require,
    run_case,
)
from scripts.qualification.survey_generated_metal import (
    compare_fields,
    ledger_suffix,
    read_float32_npy,
    replace_once,
)
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


BASE = ROOT / "tests/fixtures/reference/catalog/flat_mixed_boundary_metal_exploratory_v1.yaml"
SELECTORS = {
    "radiative": "radiative",
    "zero_gradient": "zero_gradient",
    "rigid": "rigid",
    "baseline": "rigid_slip",
}
SOURCE_OWNERS = (
    "src/init/config/v1_loader.cpp",
    "src/core/runtime/runtime_config.cpp",
    "src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
    "src/dynamics/native_cgrid_boundary.cpp",
    "src/numerics/advection/advection_cartesian.cpp",
    "src/numerics/advection/advection_batch.cpp",
    "tests/numerics/test_native_scalar_capture.cpp",
    "tests/core/test_simulation_checkpoint.cpp",
)


def config(source: str, east: str, backend: str, mode: str) -> str:
    result = replace_once(source, "    east: rigid_slip\n", f"    east: {east}\n")
    result = replace_once(result, "    backend: metal\n", f"    backend: {backend}\n")
    result = replace_once(result, "    mode: exploratory\n", f"    mode: {mode}\n")
    result = replace_once(result, "    center_x_m: 4500\n", "    center_x_m: 8000\n")
    result = replace_once(result, "    center_y_m: 4500\n", "    center_y_m: 1000\n")
    return result


def admission(binary: Path, path: Path, accepted: bool, marker: str, log: Path) -> dict:
    result = subprocess.run([str(binary), "--config", str(path), "--validate-only"],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=30, check=False)
    log.write_text(result.stdout)
    require((result.returncode == 0) == accepted and marker in result.stdout,
            f"unexpected admission for {path.name}: {result.stdout[-1200:]}")
    return {"accepted": accepted, "config": identity(path), "log": identity(log)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = BASE.read_text()
    inputs = {"binary": identity(binary), "base": identity(BASE), "gate": identity(Path(__file__))}
    inputs.update({path: identity(ROOT / path) for path in SOURCE_OWNERS})
    cases = {}
    admissions = {}
    configs = {}
    for name, east in SELECTORS.items():
        laterals = ("rigid_slip", east, "rigid_slip", "rigid_slip")
        path = out / f"{name}-cpu-qualified.yaml"
        path.write_text(config(source, east, "cpu", "qualified"))
        configs[name] = {"familyId": family_id(laterals, "rigid_noslip", "open"),
                         "selector": east, "cpu": identity(path)}
        admissions[name] = admission(binary, path, True, "[VALIDATE] PASS:",
                                     out / f"{name}-cpu-admission.log")
    target = out / "radiative-cpu-qualified.yaml"
    metal = out / "radiative-metal-qualified-guarded.yaml"
    metal.write_text(config(source, "radiative", "metal", "qualified"))
    admissions["metalQualified"] = admission(binary, metal, False,
        "qualified Metal execution is limited", out / "radiative-metal-admission.log")
    invalid = out / "unpaired-periodic.yaml"
    invalid.write_text(replace_once(target.read_text(), "    west: rigid_slip\n", "    west: periodic\n"))
    admissions["unpairedPeriodic"] = admission(binary, invalid, False,
        "periodic lateral boundaries", out / "unpaired-periodic-admission.log")
    collocated = out / "collocated-radiative.yaml"
    collocated.write_text(replace_once(target.read_text(), "  staggering: c_grid\n", "  staggering: collocated\n"))
    admissions["collocatedRadiative"] = admission(binary, collocated, False,
        "radiative", out / "collocated-radiative-admission.log")
    for name in SELECTORS:
        path = out / f"{name}-cpu-qualified.yaml"
        cases[name] = run_case(binary, path, out / f"{name}-cpu1", "cpu", 1, 1)
    cases["radiativeCpu4"] = run_case(binary, target, out / "radiative-cpu4", "cpu", 4, 1)
    cases["radiativeContinuous"] = run_case(binary, target, out / "radiative-continuous", "cpu", 1, 2, 1)
    checkpoint = out / "radiative-continuous/checkpoints/checkpoint_step_000000010.nimbus"
    require(checkpoint.is_file(), "radiative midpoint checkpoint is missing")
    cases["radiativeResumed"] = run_case(binary, target, out / "radiative-resumed", "cpu", 4, 2, 0,
                                         checkpoint)
    parity = {"cpuThreads": compare_fields(out / "radiative-cpu1", out / "radiative-cpu4", True),
              "cpuRestart": compare_fields(out / "radiative-continuous", out / "radiative-resumed", True)}
    ledgers = {"cpuThreads": exact_ledgers(out / "radiative-cpu1", out / "radiative-cpu4"),
               "cpuRestartSuffix": ledger_suffix(out / "radiative-continuous", out / "radiative-resumed")}
    output_u = {}
    for name in SELECTORS:
        final_u = sorted((out / f"{name}-cpu1").glob("step_*/u.npy"))[-1]
        _, _, output_u[name] = read_float32_npy(final_u)
    contrast = {name: max(abs(a - b) for a, b in zip(output_u["radiative"], output_u[name]))
                for name in ("zero_gradient", "rigid", "baseline")}
    require(contrast["zero_gradient"] > 1e-4 and contrast["rigid"] > .1,
            "near-corner pulse failed to distinguish radiative face from controls")
    require(all(v["passed"] for v in parity["cpuThreads"].values()) and
            all(ledgers["cpuThreads"].values()), "east-radiative CPU thread identity failed")
    continuation = all(v["passed"] for v in parity["cpuRestart"].values()) and \
        all(ledgers["cpuRestartSuffix"].values())
    unchanged = all(identity(Path(item["path"])) == item for item in inputs.values())
    require(unchanged, "gate inputs changed during execution")
    report = {"schema": "nimbus.flat-east-radiative-boundary-gate.v1",
              "status": "cpu_bounded_pass" if continuation else "cpu_restart_red",
              "claimBoundary": "Fixed one-east-radiative CPU admission/step/restart only; no native Metal integrated equivalence or public Metal readiness claim.",
              "inputs": inputs, "configs": configs, "admission": admissions, "cases": cases,
              "checkpoint": identity(checkpoint), "parity": parity, "ledgers": ledgers,
              "uContrastMaxAbs": contrast,
              "cpuRestartExact": continuation, "nativeMetalIntegrated": "not_run",
              "wholeBackendReady": False}
    write_json_atomic(out / "summary.json", report)
    print(json.dumps({"status": report["status"], "familyId": configs["radiative"]["familyId"],
                      "cpuRestartExact": continuation}))


if __name__ == "__main__":
    main()

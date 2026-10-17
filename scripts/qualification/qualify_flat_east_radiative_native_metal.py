#!/usr/bin/env python3
"""Bound the exact guarded east-radiative native Metal full-step route."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.enumerate_flat_boundary_family import family_id
from scripts.qualification.qualify_flat_east_radiative_boundary import (
    BASE,
    config,
)
from scripts.qualification.qualify_flat_mixed_boundary_metal import (
    exact_ledgers,
    identity,
    require,
    run_case,
)
from scripts.qualification.survey_generated_metal import (
    LEDGERS,
    NATIVE_TVD_OPERATION,
    compare_fields,
    ledger_suffix,
    replace_once,
)
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


FAMILY_ID = family_id(("rigid_slip", "radiative", "rigid_slip", "rigid_slip"),
                      "rigid_noslip", "open")
SOURCE_OWNERS = (
    "src/core/runtime/runtime_config.cpp",
    "src/core/runtime/headless_runtime.cpp",
    "include/core/runtime/headless_runtime.hpp",
    "include/core/runtime/flat_native_tvd_step_transaction.hpp",
    "include/core/runtime/atmospheric_level.hpp",
    "include/core/runtime/simulation_state.hpp",
    "src/core/runtime/simulation_state.cpp",
    "include/diagnostics/level_budget_csv.hpp",
    "src/numerics/advection/advection_batch.cpp",
    "src/numerics/advection/advection_cartesian.cpp",
    "src/dynamics/native_cgrid_boundary.cpp",
    "src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
    "metal/src/compute/backend/advection.inc",
    "metal/src/compute/backend/dynamics/acoustic_cgrid_batched.inc",
    "metal/include/compute/metal/shaders/advection/conservative_cgrid_transport_msl.hpp",
    "tests/numerics/test_native_scalar_capture.cpp",
)
MEMORY_CAP = 128 * 1024 * 1024


def admission(binary: Path, config_path: Path, log: Path, accepted: bool, marker: str) -> dict:
    result = subprocess.run([str(binary), "--config", str(config_path), "--validate-only"],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=30, check=False)
    log.write_text(result.stdout)
    require((result.returncode == 0) == accepted and marker in result.stdout,
            f"unexpected {config_path.name} admission: {result.stdout[-1200:]}")
    return {"accepted": accepted, "config": identity(config_path), "log": identity(log)}


def memory_and_kernel_gate(record: dict, native_updates: int) -> dict:
    log = Path(record["log"]["path"]).read_text()
    fields = [int(value) for value in re.findall(r"\[MEMORY PLAN\].*predicted_fields=(\d+)", log)]
    predicted = [int(value) for value in re.findall(r"\[MEMORY PLAN\].*predicted_bytes=(\d+)", log)]
    residency = [int(value) for value in re.findall(r"\[MEMORY PLAN\].*device_residency_bytes=(\d+)", log)]
    budgets = [int(value) for value in re.findall(r"\[MEMORY PLAN\].*device_budget_bytes=(\d+)", log)]
    peaks = [int(value) for value in re.findall(r"peak_rss_bytes=(\d+)", log)]
    require(len(fields) == len(predicted) == len(residency) == len(budgets) == 1 and peaks,
            "native Metal run omitted exact memory-plan or high-water evidence")
    require(fields[0] >= 352 and predicted[0] >= 26913232 and
            predicted[0] <= MEMORY_CAP and max(peaks) <= MEMORY_CAP and
            residency[0] < budgets[0], "native Metal run exceeded bounded memory plan")
    operations = record["operations"]
    require(operations.get(NATIVE_TVD_OPERATION, 0) == native_updates and
            operations.get("acoustic/acoustic_momentum_cartesian_cgrid.comp", 0) >= 60 and
            operations.get("acoustic/acoustic_pressure_cartesian_cgrid.comp", 0) >= 60,
            "native Metal acoustic/TVD dispatch ownership is incomplete")
    return {"predictedFieldEquivalents": fields[0], "predictedBytes": predicted[0],
            "deviceResidencyBytes": residency[0],
            "deviceBudgetBytes": budgets[0], "peakRssBytes": max(peaks),
            "nativeTvdUpdates": operations[NATIVE_TVD_OPERATION]}


def ledger_gate(outdir: Path) -> dict:
    result = {}
    for filename in LEDGERS:
        path = outdir / filename
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        require(rows, f"{outdir.name}/{filename} has no committed ledger rows")
        for row in rows:
            require(all(value is not None and value != "" and
                        (key == "stage" or math.isfinite(float(value)))
                        for key, value in row.items()),
                    f"{outdir.name}/{filename} has a nonfinite or incomplete owner cell")
        result[filename] = {"rows": len(rows), "sha256": identity(path)["sha256"]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {"binary": identity(binary), "base": identity(BASE), "qualifier": identity(Path(__file__))}
    inputs.update({path: identity(ROOT / path) for path in SOURCE_OWNERS})
    report = {"schema": "nimbus.flat-east-radiative-native-metal.v1",
              "status": "running", "fixedFamilyId": FAMILY_ID, "inputs": inputs,
              "claimBoundary": "Exact guarded one-east-radiative native Metal full-step gate only; no public qualified admission or whole-family Ready claim.",
              "wholeBackendReady": False, "I1Complete": False}
    stage = "configuration"
    try:
        source = BASE.read_text()
        configs = {}
        for name, east, backend, mode in (
                ("cpu", "radiative", "cpu", "qualified"),
                ("vulkan", "radiative", "gpu", "qualified"),
                ("metal", "radiative", "metal", "exploratory"),
                ("metalQualified", "radiative", "metal", "qualified"),
                ("eastZeroGradient", "zero_gradient", "metal", "exploratory"),
                ("eastRigid", "rigid", "metal", "exploratory")):
            path = out / f"{name}.yaml"
            path.write_text(replace_once(config(source, east, backend, mode),
                                         "    memory_admission: warn\n", "    memory_admission: reject\n"))
            configs[name] = identity(path)
        west_extra = out / "westExtraRadiative.yaml"
        west_extra.write_text(replace_once((out / "metal.yaml").read_text(),
                                           "    west: rigid_slip\n", "    west: radiative\n"))
        configs["westExtraRadiative"] = identity(west_extra)
        periodic = out / "unpairedPeriodic.yaml"
        periodic.write_text(replace_once((out / "metal.yaml").read_text(),
                                         "    west: rigid_slip\n", "    west: periodic\n"))
        configs["unpairedPeriodic"] = identity(periodic)
        report["configs"] = configs
        stage = "admission"
        admissions = {}
        for name, accepted, marker in (
                ("cpu", True, "[VALIDATE] PASS:"),
                ("vulkan", True, "[VALIDATE] PASS:"),
                ("metal", True, "[VALIDATE] PASS:"),
                ("metalQualified", False, "qualified Metal execution is limited"),
                ("unpairedPeriodic", False, "periodic lateral boundaries")):
            admissions[name] = admission(binary, out / f"{name}.yaml",
                                         out / f"{name}-admission.log", accepted, marker)
        report["admission"] = admissions
        stage = "one_step"
        cases = {}
        for name, backend, threads in (
                ("cpu1", "cpu", 1), ("cpu4", "cpu", 4),
                ("vulkan1", "vulkan", 1),
                ("metal1", "metal", 1), ("metal4", "metal", 4)):
            config_name = name.rstrip("14")
            cases[name] = run_case(binary, out / f"{config_name}.yaml",
                                   out / name, backend, threads, 1)
        stage = "continuation"
        cases["metalContinuous"] = run_case(binary, out / "metal.yaml", out / "metalContinuous",
                                             "metal", 1, 2, 1)
        checkpoint = out / "metalContinuous/checkpoints/checkpoint_step_000000010.nimbus"
        require(checkpoint.is_file(), "native Metal radiative midpoint checkpoint is missing")
        cases["metalResumed"] = run_case(binary, out / "metal.yaml", out / "metalResumed",
                                          "metal", 4, 2, 0, checkpoint, expected_native_updates=30)
        stage = "negative_neighbors"
        for name in ("eastZeroGradient", "eastRigid", "westExtraRadiative"):
            cases[name] = run_case(binary, out / f"{name}.yaml", out / name,
                                   "metal", 1, 1, expected_native_updates=0)
        report["cases"] = cases
        report["checkpoint"] = identity(checkpoint)
        stage = "memory_kernels"
        report["memoryAndKernels"] = {
            name: memory_and_kernel_gate(cases[name], 30 if name != "metalContinuous" else 60)
            for name in ("metal1", "metal4", "metalContinuous")}
        report["memoryAndKernels"]["metalResumed"] = memory_and_kernel_gate(cases["metalResumed"], 30)
        stage = "parity_ledgers"
        parity = {"cpuMetal": compare_fields(out / "cpu1", out / "metal1", False),
                  "cpuVulkan": compare_fields(out / "cpu1", out / "vulkan1", False),
                  "cpuThreads": compare_fields(out / "cpu1", out / "cpu4", True),
                  "metalThreads": compare_fields(out / "metal1", out / "metal4", True),
                  "metalRestart": compare_fields(out / "metalContinuous", out / "metalResumed", True)}
        ledger_identity = {"cpuThreads": exact_ledgers(out / "cpu1", out / "cpu4"),
                           "metalThreads": exact_ledgers(out / "metal1", out / "metal4"),
                           "metalRestartSuffix": ledger_suffix(out / "metalContinuous", out / "metalResumed")}
        report["parity"] = parity
        report["ledgerIdentity"] = ledger_identity
        report["ledgerOwners"] = {name: ledger_gate(out / name)
                                  for name in ("cpu1", "vulkan1", "metal1", "metal4",
                                               "metalContinuous", "metalResumed")}
        require(all(field["passed"] for values in parity.values() for field in values.values()) and
                all(value for values in ledger_identity.values() for value in values.values()),
                "native Metal east-radiative field or ledger parity failed")
        require(all(identity(Path(item["path"])) == item for item in inputs.values()),
                "source or executable changed during integrated gate")
        report["status"] = "bounded_integrated_pass"
        write_json_atomic(out / "summary.json", report)
        print(json.dumps({"status": report["status"], "familyId": FAMILY_ID,
                          "qualifiedMetalAdmission": "guarded", "nativeTvdUpdates": 30}))
    except Exception as exc:
        report["status"] = "red"
        report["failedStage"] = stage
        report["error"] = str(exc)
        write_json_atomic(out / "summary.json", report)
        raise


if __name__ == "__main__":
    main()

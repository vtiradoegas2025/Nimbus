#!/usr/bin/env python3
"""Bound exact public east-radiative NPY admission and native Metal execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.qualify_flat_east_radiative_native_metal import (
    FAMILY_ID,
    ledger_gate,
    memory_and_kernel_gate,
)
from scripts.qualification.qualify_flat_mixed_boundary_metal import (
    exact_ledgers,
    identity,
    require,
    run_case,
)
from scripts.qualification.survey_generated_metal import compare_fields, ledger_suffix, replace_once
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


EAST = ROOT / "tests/fixtures/reference/catalog/flat_east_radiative_metal_npy_candidate_v1.yaml"
ZFP = ROOT / "tests/fixtures/reference/catalog/generated_metal_zfp_candidate_v1.yaml"
CANONICAL = ROOT / "tests/fixtures/reference/catalog/generated_metal_canonical_v1.yaml"
SOURCE_OWNERS = (
    "src/core/runtime/runtime_config.cpp",
    "src/core/runtime/nimbus_sim.cpp",
    "src/core/runtime/scheme_catalog.cpp",
    "include/core/runtime/runtime_config.hpp",
    "include/core/runtime/generated_metal_east_npy_scope_v1.hpp",
    "include/core/runtime/generated_metal_output_scope_v3.hpp",
    "tests/fixtures/reference/catalog/generated_metal_east_npy_scope_v1.json",
    "tests/fixtures/reference/catalog/generated_metal_output_scope_v3.json",
)


def output_block(source: str) -> str:
    begin = source.index("# Output\n")
    end = source.index("# Nimbus extensions", begin)
    return source[begin:end]


def validate(binary: Path, config: Path, log: Path, accepted: bool,
             marker: str, extra: tuple[str, ...] = ()) -> dict:
    result = subprocess.run([str(binary), "--config", str(config),
                             "--validate-only", "--duration=1", *extra],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=30, check=False)
    log.write_text(result.stdout)
    require((result.returncode == 0) == accepted and marker in result.stdout,
            f"unexpected {config.name} admission {extra}: {result.stdout[-1200:]}")
    return {"accepted": accepted, "config": identity(config),
            "log": identity(log), "cli": list(extra)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission-only", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = EAST.read_text()
    inputs = {"binary": identity(binary), "east": identity(EAST),
              "zfpCandidate": identity(ZFP), "canonical": identity(CANONICAL),
              "qualifier": identity(Path(__file__))}
    inputs.update({path: identity(ROOT / path) for path in SOURCE_OWNERS})
    report = {"schema": "nimbus.flat-east-radiative-public-npy.v1",
              "status": "running", "familyId": FAMILY_ID, "inputs": inputs,
              "claimBoundary": "One exact publicly admitted qualified east-radiative minimal NPY Metal profile; other retained boundaries and east codecs remain guarded.",
              "wholeBackendReady": False, "I1Complete": False}
    stage = "configs"
    try:
        cpu = out / "cpu-qualified.yaml"
        vulkan = out / "vulkan-qualified.yaml"
        cpu.write_text(replace_once(source, "    backend: metal\n", "    backend: cpu\n"))
        vulkan.write_text(replace_once(source, "    backend: metal\n", "    backend: gpu\n"))
        mutated = {
            "netcdf": replace_once(source, "  format: npy_3d\n", "  format: netcdf\n"),
            "zfpHigh": replace_once(source, output_block(source), output_block(ZFP.read_text())),
            "npyAsync": replace_once(source, "  async_io: false\n", "  async_io: true\n"),
            "eastZeroGradient": replace_once(source, "    east: radiative\n", "    east: zero_gradient\n"),
            "eastRigid": replace_once(source, "    east: radiative\n", "    east: rigid\n"),
            "extraWestRadiative": replace_once(source, "    west: rigid_slip\n", "    west: radiative\n"),
            "topRigid": replace_once(source, "  top: open\n", "  top: rigid\n"),
            "bottomSlip": replace_once(source, "  bottom: rigid_noslip\n", "  bottom: rigid_slip\n"),
            "centeredTrigger": replace_once(replace_once(source, "    center_x_m: 8000\n",
                                                        "    center_x_m: 4500\n"),
                                           "    center_y_m: 1000\n", "    center_y_m: 4500\n"),
        }
        configs = {"metal": identity(EAST), "cpu": identity(cpu), "vulkan": identity(vulkan)}
        for name, content in mutated.items():
            path = out / f"negative-{name}.yaml"
            path.write_text(content)
            configs[name] = identity(path)
        report["configs"] = configs
        stage = "admission"
        admissions = {"cpu": validate(binary, cpu, out / "cpu-admission.log", True, "[VALIDATE] PASS:"),
                      "vulkan": validate(binary, vulkan, out / "vulkan-admission.log", True, "[VALIDATE] PASS:"),
                      "metal": validate(binary, EAST, out / "metal-admission.log", True, "[VALIDATE] PASS:"),
                      "canonical": validate(binary, CANONICAL, out / "canonical-admission.log", True,
                                             "[VALIDATE] PASS:")}
        for name in mutated:
            admissions[name] = validate(binary, out / f"negative-{name}.yaml",
                                        out / f"negative-{name}-admission.log", False,
                                        "qualified Metal execution is limited")
        cli_negatives = {
            "write301": (("--write-every=301",), "qualified east-radiative Metal NPY scope requires --write-every=1"),
            "dt": (("--dt-large=0.2",), "qualified generated Metal scope requires --dt-large=0.1"),
            "substeps": (("--acoustic-substeps=5",), "qualified generated Metal scope requires --acoustic-substeps=6"),
            "guardOff": (("--guard-mode=off",), "Qualified execution requires --guard-mode=strict"),
            "guardScope": (("--guard-scope=exported",), "qualified generated Metal scope requires --guard-scope=required"),
            "liveShm": (("--live-shm",), "qualified generated Metal scope does not admit live-SHM CLI overrides"),
            "trajectories": (("--trajectories",), "qualified generated Metal scope does not admit trajectory CLI overrides"),
        }
        report["cliNegatives"] = {name: validate(binary, EAST, out / f"cli-{name}.log",
                                                 False, marker, options)
                                  for name, (options, marker) in cli_negatives.items()}
        report["admission"] = admissions
        if args.admission_only:
            require(all(identity(Path(item["path"])) == item for item in inputs.values()),
                    "source or binary changed during qualified admission gate")
            report["status"] = "bounded_admission_pass"
            write_json_atomic(out / "summary.json", report)
            print(json.dumps({"status": report["status"], "familyId": FAMILY_ID,
                              "qualifiedMetalAdmission": "accepted"}))
            return
        stage = "physical"
        cases = {}
        for name, config, backend, threads in (
                ("cpu1", cpu, "cpu", 1), ("cpu4", cpu, "cpu", 4),
                ("vulkan1", vulkan, "vulkan", 1),
                ("metal1", EAST, "metal", 1), ("metal4", EAST, "metal", 4)):
            cases[name] = run_case(binary, config, out / name, backend, threads, 1)
        cases["metalContinuous"] = run_case(binary, EAST, out / "metalContinuous", "metal", 1, 2, 1)
        checkpoint = out / "metalContinuous/checkpoints/checkpoint_step_000000010.nimbus"
        require(checkpoint.is_file(), "qualified Metal midpoint checkpoint is absent")
        cases["metalResumed"] = run_case(binary, EAST, out / "metalResumed", "metal", 4, 2, 0,
                                          checkpoint, expected_native_updates=30)
        report["cases"] = cases
        report["checkpoint"] = identity(checkpoint)
        report["memoryAndKernels"] = {name: memory_and_kernel_gate(cases[name], count)
                                      for name, count in (("metal1", 30), ("metal4", 30),
                                                          ("metalContinuous", 60), ("metalResumed", 30))}
        stage = "parity_ledgers"
        parity = {"cpuMetal": compare_fields(out / "cpu1", out / "metal1", False),
                  "cpuVulkan": compare_fields(out / "cpu1", out / "vulkan1", False),
                  "cpuThreads": compare_fields(out / "cpu1", out / "cpu4", True),
                  "metalThreads": compare_fields(out / "metal1", out / "metal4", True),
                  "metalRestart": compare_fields(out / "metalContinuous", out / "metalResumed", True)}
        ledgers = {"cpuThreads": exact_ledgers(out / "cpu1", out / "cpu4"),
                   "metalThreads": exact_ledgers(out / "metal1", out / "metal4"),
                   "metalRestartSuffix": ledger_suffix(out / "metalContinuous", out / "metalResumed")}
        report["parity"] = parity
        report["ledgerIdentity"] = ledgers
        report["ledgerOwners"] = {name: ledger_gate(out / name)
                                  for name in ("cpu1", "vulkan1", "metal1", "metal4",
                                               "metalContinuous", "metalResumed")}
        require(all(field["passed"] for values in parity.values() for field in values.values()) and
                all(value for values in ledgers.values() for value in values.values()),
                "qualified east NPY field or ledger parity failed")
        require(all(identity(Path(item["path"])) == item for item in inputs.values()),
                "source or binary changed during qualified public gate")
        report["status"] = "bounded_public_pass"
        write_json_atomic(out / "summary.json", report)
        print(json.dumps({"status": report["status"], "familyId": FAMILY_ID,
                          "qualifiedMetalAdmission": "accepted", "nativeTvdUpdates": 30}))
    except Exception as exc:
        report["status"] = "red"
        report["failedStage"] = stage
        report["error"] = str(exc)
        write_json_atomic(out / "summary.json", report)
        raise


if __name__ == "__main__":
    main()

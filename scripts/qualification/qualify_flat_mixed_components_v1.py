#!/usr/bin/env python3
"""Stage source-bound mixed-open component references and real Metal memory policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification import qualify_flat_mixed_metal_rebind_v1 as rebind
from scripts.qualification.qualify_flat_mixed_boundary_metal import exact_ledgers, identity, require, run_case
from scripts.qualification.qualify_flat_east_radiative_native_metal import ledger_gate, memory_and_kernel_gate
from scripts.qualification.checkpointing.qualify_vulkan_restart import require_report_owner_identities
from scripts.qualification.survey_generated_metal import compare_fields

CASES = (
    {"name": "native_boundary_metal", "target": "bin/test_dynamics_native_cgrid_boundary",
     "filter": "[metal][native_cgrid_boundary]", "backend": "metal", "minimumAssertions": 80000,
     "ownerClass": "CPU oracle × native Metal 3750 boundary combinations plus invalid/history/thread witnesses"},
    {"name": "native_boundary_vulkan", "target": "bin/test_dynamics_native_cgrid_boundary",
     "filter": "[vulkan][native_cgrid_boundary]", "backend": "vulkan", "minimumAssertions": 80000,
     "ownerClass": "CPU oracle × native Vulkan 3750 boundary combinations plus invalid/history/thread witnesses"},
    {"name": "open_top_boundary", "target": "bin/test_dynamics_cartesian_lateral_bc_yaml",
     "filter": "C-grid open top closes the normal face with zero gradient", "backend": "cpu",
     "minimumAssertions": 1, "ownerClass": "independent open-lid normal-face zero-gradient reference"},
    {"name": "boundary_kinetic_closure", "target": "bin/test_dynamics_cartesian_lateral_bc_yaml",
     "filter": "SCI-142 C-grid acoustic boundary commit closes native kinetic inventory", "backend": "cpu",
     "minimumAssertions": 1, "ownerClass": "independent constrained boundary kinetic inventory"},
    {"name": "bottom_no_slip", "target": "bin/test_dynamics_cartesian_vertical_bc_yaml",
     "filter": "CGrid bottom=RigidNoSlip pins u/v at k=0", "backend": "cpu",
     "minimumAssertions": 1, "ownerClass": "independent C-grid bottom tangential-velocity pinning"},
    {"name": "open_top_mass_scalar", "target": "bin/test_numerics_advection_cartesian",
     "filter": "[open-top][SCI-042]",
     "backend": "cpu", "minimumAssertions": 1,
     "ownerClass": "independent retained top dry-mass, scalar and carrier-kinetic outflow reference"},
    {"name": "advection_transaction", "target": "bin/test_numerics_advection_cartesian",
     "filter": "Strict native WENO restores fields and ledgers after partial device failure", "backend": "injected",
     "minimumAssertions": 48, "ownerClass": "WENO/native-dry-TVD caller-owned scalar/report rollback"},
)
EXTRA_OWNERS = (
    "scripts/dev/shaders/generate_mapped_boundary_msl.py",
    "scripts/dev/shaders/verify_spirv.py", "scripts/dev/shaders/verify_kernel_manifest.py",
    "include/dynamics/native_cgrid_boundary.hpp", "include/dynamics/final_boundary_energy_report.hpp",
    "include/boundary_conditions/runtime_state.hpp", "src/boundary_conditions/runtime.cpp",
    "src/dynamics/cartesian_cgrid_momentum_flux.cpp", "include/compute/compute_backend.hpp",
    "include/core/infra/hardware_info.hpp", "src/core/infra/hardware_info.cpp",
    "vulkan/src/compute/compute_backend_vulkan.cpp",
    "scripts/tests/catalog/test_flat_mixed_components_v1.py",
)


def inputs() -> dict:
    return rebind.source_inputs() | {name: identity(ROOT / name) for name in EXTRA_OWNERS} | {
        "componentRunner": identity(Path(__file__))}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def result_record(result: subprocess.CompletedProcess[str], case: dict) -> dict:
    content = result.stdout + result.stderr
    match = re.search(r"All tests passed \((\d+) assertions? in (\d+) test cases?\)", content)
    assertions = int(match[1]) if match else 0
    count = int(match[2]) if match else 0
    passed = result.returncode == 0 and count == 1 and assertions >= case["minimumAssertions"] and \
        "[SCIENTIFIC INVALIDITY]" not in content
    return {**case, "exitCode": result.returncode, "assertions": assertions,
            "testCases": count, "passed": passed}


def stage(root: Path) -> None:
    require(not root.exists(), "component stage root must be new")
    before = inputs()
    fixtures = rebind.staged_configs()
    (root / "fixtures").mkdir(parents=True)
    for name in ("metal_exploratory", "metal_exploratory_memory_reject"):
        (root / "fixtures" / (name + ".yaml")).write_text(fixtures[name], encoding="utf-8")
    require(inputs() == before, "source changed while staging components")
    write_json(root / "stage.json", {
        "schema": "nimbus.flat-mixed-component-stage.v1", "status": "staged_not_executed",
        "familyId": rebind.FAMILY_ID, "sourceAtStage": before, "componentCases": list(CASES),
        "buildTargets": sorted({row["target"] for row in CASES}),
        "fixtures": {name: identity(root / "fixtures" / (name + ".yaml"))
                     for name in ("metal_exploratory", "metal_exploratory_memory_reject")},
        "shaderPrerequisites": ["python3 -B scripts/dev/shaders/generate_mapped_boundary_msl.py --check",
                                "python3 -B scripts/dev/shaders/verify_spirv.py",
                                "python3 -B scripts/dev/shaders/verify_kernel_manifest.py"],
        "componentProofBoundary": "selected existing owner references including exact mixed topology inside native matrix; no public admission or transfer to full boundary-family native transport",
        "resourceControlBoundary": "two actual Metal 1s runs differ only in warn/reject memory policy; intended exact native transport and eight-field/six-ledger byte identity must pass",
        "caps": {"componentSeconds": 180, "componentGroupSeconds": 1260,
                 "componentRawBytes": 16 * 1024 * 1024, "controlRssBytes": 128 * 1024 * 1024,
                 "controlRawBytes": 64 * 1024 * 1024},
        "VulkanComparatorResourceRed": "preserved separately; not repaired or bypassed by this component collector",
        "publicMetalAdmission": "guarded", "physicalExecution": "not_run",
        "I1Complete": False, "wholeBackendReady": False})


def bind(root: Path, binary: Path, expected: str) -> None:
    require(not (root / "binding.json").exists(), "use a new component stage for each binary/source binding")
    staged = json.loads((root / "stage.json").read_text())
    for name, record in staged["fixtures"].items():
        require(identity(Path(record["path"])) == record, "component fixture drifted: " + name)
    require(identity(binary)["sha256"] == expected, "coordinated simulator SHA differs")
    before = inputs()
    targets = {target: identity(ROOT / target) for target in staged["buildTargets"]}
    require(inputs() == before, "source changed while binding")
    write_json(root / "binding.json", {
        "schema": "nimbus.flat-mixed-component-binding.v1", "status": "bound_not_executed",
        "stageSha256": identity(root / "stage.json")["sha256"], "simulator": identity(binary),
        "sourceInputs": before, "testBinaries": targets,
        "sourceChangesSinceStage": [name for name in before if before[name] != staged["sourceAtStage"].get(name)],
        "publicMetalAdmission": "guarded", "I1Complete": False, "wholeBackendReady": False})


def binding(root: Path) -> dict:
    bound = json.loads((root / "binding.json").read_text())
    require(identity(root / "stage.json")["sha256"] == bound["stageSha256"], "component stage drifted")
    require(inputs() == bound["sourceInputs"], "enumerated component source drifted")
    require(identity(Path(bound["simulator"]["path"])) == bound["simulator"], "simulator drifted")
    for name, record in bound["testBinaries"].items():
        require(identity(Path(record["path"])) == record, "test binary drifted: " + name)
    staged = json.loads((root / "stage.json").read_text())
    for name, record in staged["fixtures"].items():
        require(identity(Path(record["path"])) == record, "component fixture drifted: " + name)
    return bound


def components(root: Path) -> None:
    bound = binding(root)
    out = root / "components"
    require(not out.exists(), "component raw root already exists")
    out.mkdir()
    started = time.monotonic()
    rows = []
    for case in CASES:
        command = [bound["testBinaries"][case["target"]]["path"], case["filter"]]
        if case["backend"] == "vulkan":
            command.insert(0, str(ROOT / "scripts/run_with_vulkan_env.sh"))
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False,
                                timeout=180, env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE"})
        log = out / (case["name"] + ".log")
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        rows.append({**result_record(result, case), "log": identity(log)})
        raw_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        write_json(out / "summary.json", {
            "schema": "nimbus.flat-mixed-component-result.v1", "status": "running_or_red",
            "binding": identity(root / "binding.json"), "cases": rows, "rawBytes": raw_bytes,
            "wallSeconds": time.monotonic() - started, "publicMetalAdmission": "guarded",
            "wholeBackendReady": False})
        require(rows[-1]["passed"], "component reference failed: " + case["name"])
        require(raw_bytes < 16 * 1024 * 1024 and time.monotonic() - started < 1260, "component resource cap failed")
        binding(root)
    write_json(out / "summary.json", {
        "schema": "nimbus.flat-mixed-component-result.v1", "status": "bounded_component_pass",
        "binding": identity(root / "binding.json"), "cases": rows,
        "assertions": sum(row["assertions"] for row in rows), "testCases": len(rows),
        "wallSeconds": time.monotonic() - started, "publicMetalAdmission": "guarded",
        "VulkanComparatorResourceRed": "unresolved_separate_gate", "I1Complete": False, "wholeBackendReady": False})


def resource_control(root: Path) -> None:
    bound = binding(root)
    staged = json.loads((root / "stage.json").read_text())
    out = root / "resource_control"
    require(not out.exists(), "resource control raw root already exists")
    out.mkdir()
    started = time.monotonic()
    rows = {}
    for name, fixture in (("warn", "metal_exploratory"), ("reject", "metal_exploratory_memory_reject")):
        rows[name] = run_case(Path(bound["simulator"]["path"]), Path(staged["fixtures"][fixture]["path"]),
                              out / name, "metal", 1, 1)
        content = Path(rows[name]["log"]["path"]).read_text()
        require("[COMPUTE] backend=metal" in content and "Execution mode: exploratory" in content,
                "resource control did not execute intended exploratory Metal branch")
        rows[name]["memoryAndKernels"] = memory_and_kernel_gate(rows[name], 30)
        rows[name]["ownerIdentityChecks"] = require_report_owner_identities(out / name)
        require(rows[name]["ownerIdentityChecks"] >= 850, "resource control omitted owner identity evidence")
        rows[name]["ledgerOwners"] = ledger_gate(out / name)
        binding(root)
    fields = compare_fields(out / "warn", out / "reject", True)
    ledgers = exact_ledgers(out / "warn", out / "reject")
    require(all(row["passed"] for row in fields.values()) and all(ledgers.values()), "memory policy changed equations or ledger results")
    resource = rebind.observed_resource_gate(rows)
    write_json(out / "resource_review.json", resource)
    require(resource["status"] == "bounded_resource_pass", "Metal resource control exceeded unchanged 128-MiB RSS cap")
    raw_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    require(raw_bytes < 64 * 1024 * 1024, "Metal resource control raw bytes exceeded cap")
    require(time.monotonic() - started < 360, "Metal resource control exceeded bounded two-run duration")
    binding(root)
    write_json(out / "summary.json", {
        "schema": "nimbus.flat-mixed-real-metal-memory-policy.v1", "status": "bounded_real_device_control_pass",
        "binding": identity(root / "binding.json"), "cases": rows, "fieldIdentity": fields,
        "ledgerIdentity": ledgers, "resourceReview": identity(out / "resource_review.json"),
        "rawBytes": raw_bytes, "publicMetalAdmission": "guarded",
        "claimBoundary": "actual-device fitting warn/reject policy control at unchanged exact equations; injected failures remain separate receipt",
        "I1Complete": False, "wholeBackendReady": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("stage", "bind", "components", "resource-control"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--expected-binary-sha256")
    args = parser.parse_args()
    root = args.report_root.resolve()
    if args.stage:
        stage(root)
    elif args.bind:
        require(bool(args.expected_binary_sha256), "exact coordinated simulator SHA required")
        bind(root, args.binary.resolve(), args.expected_binary_sha256)
    elif args.components:
        components(root)
    else:
        resource_control(root)
    print("[mixed-open-components] action complete; public admission and qualification remain pending")


if __name__ == "__main__":
    main()

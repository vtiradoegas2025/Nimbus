#!/usr/bin/env python3
"""Stage and source-bind exact mixed-open Metal evidence without opening admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.qualify_flat_mixed_boundary_metal import identity, require
from scripts.qualification.survey_generated_metal import replace_once

FIXTURE = ROOT / "tests/fixtures/reference/catalog/flat_mixed_boundary_metal_exploratory_v1.yaml"
QUALIFIED_SHA = "985c698858be9be4be29dc2adf819f31c20788c13a414747c422e21d7ebe51da"
FAMILY_ID = "boundary:ed765facac077ef0b331283d"
OWNERS = (
    "src/init/config/v1_loader.cpp", "src/init/config/config_v1.cpp",
    "include/init/config/config_v1.hpp", "src/init/config/config_v1_serialize.cpp",
    "src/init/config/config_v1_serialization_common.cpp",
    "src/core/runtime/runtime_config.cpp", "src/core/runtime/nimbus_sim.cpp",
    "include/core/runtime/runtime_config.hpp", "include/core/runtime/generated_metal_scope_v1.hpp",
    "include/core/runtime/generated_metal_east_npy_scope_v1.hpp",
    "include/core/runtime/generated_metal_output_scope_v2.hpp",
    "include/core/runtime/generated_metal_output_scope_v3.hpp",
    "src/core/runtime/headless_runtime.cpp", "include/core/runtime/headless_runtime.hpp",
    "include/core/runtime/flat_native_tvd_step_transaction.hpp",
    "include/core/runtime/atmospheric_level.hpp", "include/core/runtime/simulation_state.hpp",
    "src/core/runtime/simulation_state.cpp", "include/diagnostics/level_budget_csv.hpp",
    "src/numerics/advection/advection_batch.cpp", "src/numerics/advection/advection_cartesian.cpp",
    "src/dynamics/native_cgrid_boundary.cpp",
    "src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
    "metal/src/compute/backend/advection.inc",
    "metal/src/compute/backend/dynamics/acoustic_cgrid_batched.inc",
    "metal/include/compute/metal/shaders/advection/conservative_cgrid_transport_msl.hpp",
    "metal/include/compute/metal/shaders/dynamics/acoustic_bc_face_cartesian_cgrid_msl.hpp",
    "tests/core/test_nested_production_step.cpp", "tests/dynamics/test_native_cgrid_boundary.cpp",
    "tests/dynamics/test_cartesian_lateral_bc_yaml.cpp", "tests/dynamics/test_cartesian_vertical_bc_yaml.cpp",
    "tests/numerics/test_advection_cartesian.cpp", "mk/tests.mk",
    "scripts/qualification/qualify_flat_mixed_boundary_metal.py",
    "scripts/qualification/survey_generated_metal.py",
    "scripts/qualification/qualify_flat_east_radiative_native_metal.py",
    "scripts/qualification/checkpointing/qualify_vulkan_restart.py",
    "scripts/qualification/seal_runtime_archive.py", "scripts/run_with_vulkan_env.sh",
    "scripts/tests/catalog/test_flat_mixed_metal_rebind_v1.py",
    "tests/core/test_flat_mixed_preallocation.cpp",
    "src/core/orchestration/dynamics/equations.cpp",
    "src/compute/compute_backend.cpp", "src/compute/backend_runtime.cpp",
    "include/compute/compute_memory_admission.hpp", "include/compute/backend_types.hpp",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def source_inputs() -> dict[str, dict]:
    return {path: identity(ROOT / path) for path in OWNERS} | {
        "runner": identity(Path(__file__)), "fixture": identity(FIXTURE)}


def future_cli_contracts() -> list[dict]:
    return [
        {"name": name, "flags": list(flags), "accepted": accepted,
         "rejectionMarkers": [] if accepted else cli_rejection_markers(name),
         "proofState": "post_admission_not_executed"}
        for name, flags, accepted in (
            ("baseline", (), True), ("duration_one", ("--duration=1",), True),
            ("write_one", ("--write-every=1",), True),
            ("dt_same", ("--dt-large=0.1",), True),
            ("substeps_same", ("--acoustic-substeps=6",), True),
            ("guard_strict", ("--guard-mode=strict",), True),
            ("write_301", ("--write-every=301",), False),
            ("write_two", ("--write-every=2",), False),
            ("write_disabled", ("--write-every=0",), False),
            ("dt_changed", ("--dt-large=0.2",), False),
            ("substeps_changed", ("--acoustic-substeps=5",), False),
            ("guard_off", ("--guard-mode=off",), False),
            ("guard_exported", ("--guard-scope=exported",), False),
            ("guard_nonfinite_only", ("--guard-fail-on=nonfinite",), False),
            ("live_shm", ("--live-shm",), False),
            ("trajectories", ("--trajectories",), False),
        )]


def cli_rejection_markers(name: str) -> list[str]:
    if name.startswith("write_"):
        return ["qualified mixed-open Metal NPY scope requires --write-every=1"]
    return {"dt_changed": ["qualified generated Metal scope requires --dt-large=0.1"],
            "substeps_changed": ["qualified generated Metal scope requires --acoustic-substeps=6"],
            "guard_off": ["Qualified execution requires --guard-mode=strict"],
            "guard_exported": ["qualified generated Metal scope requires --guard-scope=required"],
            "guard_nonfinite_only": ["Qualified execution requires --guard-fail-on=both"],
            "live_shm": ["qualified generated Metal scope does not admit live-SHM CLI overrides"],
            "trajectories": ["qualified generated Metal scope does not admit trajectory CLI overrides"]}[name]


def public_admission_result(result: subprocess.CompletedProcess[str], accepted: bool,
                            markers: list[str]) -> dict:
    content = result.stdout + result.stderr
    actual = result.returncode == 0 and "[VALIDATE] PASS:" in content
    matched = [marker for marker in markers if marker in content]
    passed = actual if accepted else result.returncode > 0 and "[VALIDATE]" in content and bool(markers) and len(matched) == len(markers)
    return {"actualAccepted": actual, "passed": passed,
            "exitCode": result.returncode, "matchedRejectionMarkers": matched}


def observed_resource_gate(cases: dict[str, dict], rss_cap: int = 128 * 1024 * 1024) -> dict:
    result = {}
    for name, record in cases.items():
        path = Path(record["log"]["path"])
        require(identity(path) == record["log"], "core resource log identity drifted: " + name)
        content = path.read_text(encoding="utf-8")
        peaks = [int(value) for value in re.findall(r"peak_rss_bytes=(\d+)", content)]
        require(bool(peaks), "core run omitted peak-RSS evidence: " + name)
        result[name] = {"peakRssBytes": max(peaks), "rssCapBytes": rss_cap,
                        "passed": max(peaks) <= rss_cap, "log": record["log"]}
        require(identity(path) == record["log"], "core resource log changed during review: " + name)
    return {"status": "bounded_resource_pass" if all(row["passed"] for row in result.values()) else "red_rss_cap",
            "cases": result, "thresholdRelaxed": False}


def staged_configs() -> dict[str, str]:
    source = FIXTURE.read_text(encoding="utf-8")
    qualified = replace_once(source, "    mode: exploratory\n", "    mode: qualified\n")
    require(hashlib.sha256(qualified.encode()).hexdigest() == QUALIFIED_SHA,
            "exact retained mixed-open source bytes changed; review a new stage version")
    return {
        "metal_exploratory": source,
        "metal_qualified": qualified,
        "cpu_qualified": replace_once(qualified, "    backend: metal\n", "    backend: cpu\n"),
        "vulkan_qualified": replace_once(qualified, "    backend: metal\n", "    backend: gpu\n"),
        "metal_exploratory_memory_reject": replace_once(source, "    memory_admission: warn\n", "    memory_admission: reject\n"),
        "negative_netcdf": replace_once(qualified, "  format: npy_3d\n", "  format: netcdf\n"),
        "negative_zfp_high": replace_once(qualified, "  format: npy_3d\n", "  format: zfp_high\n"),
        "negative_npy_async": replace_once(qualified, "  async_io: false\n", "  async_io: true\n"),
        "negative_live_shm": replace_once(qualified, "  live_shm: false\n", "  live_shm: true\n"),
        "negative_trajectories": replace_once(qualified, "  trajectories: false\n", "  trajectories: true\n"),
        "negative_east_rigid": replace_once(qualified, "    east: rigid_slip\n", "    east: rigid\n"),
        "negative_bottom_slip": replace_once(qualified, "  bottom: rigid_noslip\n", "  bottom: rigid_slip\n"),
        "negative_top_rigid": replace_once(qualified, "  top: open\n", "  top: rigid\n"),
        "negative_unpaired_periodic": replace_once(qualified, "    west: rigid_slip\n", "    west: periodic\n"),
    }


def stage(root: Path) -> None:
    require(not root.exists(), "stage root must be new")
    inputs = source_inputs()
    configs = staged_configs()
    (root / "fixtures").mkdir(parents=True)
    for name, content in configs.items():
        (root / "fixtures" / (name + ".yaml")).write_text(content, encoding="utf-8")
    require(source_inputs() == inputs, "source changed while staging")
    write_json(root / "stage.json", {
        "schema": "nimbus.flat-mixed-metal-rebind-stage.v3", "status": "staged_not_executed",
        "familyId": FAMILY_ID, "retainedCombinationId": FAMILY_ID + "/metal",
        "claimBoundary": "One retained 18-cubed mixed-open topology; source plan only, no binary admission or physical execution",
        "sourceInputs": inputs,
        "fixtures": {name: identity(root / "fixtures" / (name + ".yaml")) for name in configs},
        "physicsSelectors": {
            "lateral": ["rigid_slip"] * 4, "bottom": "rigid_noslip", "top": "open_zero_gradient_lid",
            "grid": "flat Cartesian C-grid 18x18x18 at 500/500/250m",
            "dynamics": "KW split explicit dt=0.1, six acoustic substeps, adaptive=false",
            "transport": "retained-mass-flux conservative MC/TVD, microphysics=none",
            "initialCondition": "parametric CAPE0, theta300K, passive qv0.004, centered2K bubble and seeded initial-condition chaos",
            "disabledOwners": ["terrain", "damping", "coriolis", "SGS", "PBL", "surface_fluxes", "radiation"]},
        "incidentalOrOperationalHashKnobs": {
            "memoryAdmission": "warn/reject is a resource policy, not a physics-selector expansion; reject variant is a separate diagnostic fixture",
            "executionMode": "exploratory/qualified is admission policy; strict science guard remains error",
            "output": "synchronous minimal NPY, no live SHM or trajectories; bytes/codec/cadence still need output proof",
            "inactiveDefaults": "stored inactive-owner constants can alter the full canonical hash without changing the selected equation; no automatic normalization or exclusion is authorized"},
        "futureCliContracts": future_cli_contracts(),
        "publicScope": "proposed one exact synchronous minimal NPY profile at write cadence1; generic301 cadence must not be inherited without its own gate",
        "requiredGates": {
            "core": "current-source eight-run CPU/Vulkan/Metal parity, native dispatch, strict ledgers and Metal1-to4-worker restart",
            "rollback": "new exact mixed-open direct15-field/private/report/checkpoint rollback and headless six-ledger/no-artifact/terminal-contract failure cases; uncompiled and unexecuted",
            "components": "current-source native CPU/Metal/Vulkan boundary owner matrix, CPU open-top flux and bottom no-slip references, injected advection transaction",
            "resourceAndMissingKernel": "dedicated injected-factory target rejects missing owners and insufficient range/heap before field allocation; fitting control stops at an allocation sentinel; no fake physics execution",
            "public": "ordinary qualified selection and CLI/output/adjacent negatives after one-profile admission decision; current base guard masks these"},
        "resourceCaps": {"physicalRunSeconds": 180, "rollbackCaseSeconds": 240,
                         "coreGateSeconds": 1800, "rawBytes": 128 * 1024 * 1024,
                         "observedRssBytes": 128 * 1024 * 1024},
        "ordinaryAdmission": "not_run", "physicalExecution": "not_run",
        "I1Complete": False, "wholeBackendReady": False})


def bind(root: Path, binary: Path, helper: Path, core_test: Path, expected: str,
         preallocation_test: Path | None = None) -> None:
    require(not (root / "binding.json").exists(), "use a new stage root instead of replacing a binary binding")
    staged = json.loads((root / "stage.json").read_text())
    for name, record in staged["fixtures"].items():
        require(identity(Path(record["path"])) == record, "staged fixture changed: " + name)
    require(identity(binary)["sha256"] == expected, "simulator SHA does not match coordinated freeze")
    inputs = source_inputs() | {"binary": identity(binary), "helper": identity(helper), "coreTestBinary": identity(core_test)}
    if preallocation_test is not None:
        inputs["preallocationTestBinary"] = identity(preallocation_test)
    write_json(root / "binding.json", {"schema": "nimbus.flat-mixed-metal-rebind-binding.v1",
               "status": "bound_not_executed", "stageSha256": identity(root / "stage.json")["sha256"],
               "inputs": inputs, "I1Complete": False, "wholeBackendReady": False})


def binding(root: Path) -> dict:
    document = json.loads((root / "binding.json").read_text())
    require(identity(root / "stage.json")["sha256"] == document["stageSha256"], "stage plan drifted")
    require(source_inputs() == {key: document["inputs"][key] for key in source_inputs()}, "enumerated source drifted")
    for name, record in document["inputs"].items():
        require(identity(Path(record["path"])) == record, "bound source or binary changed: " + name)
    staged = json.loads((root / "stage.json").read_text())
    for name, record in staged["fixtures"].items():
        require(identity(Path(record["path"])) == record, "staged fixture changed: " + name)
    return document


def core(root: Path) -> None:
    sealed = binding(root)
    destination = root / "core"
    require(not destination.exists(), "core evidence root already exists")
    started = time.monotonic()
    binary = Path(sealed["inputs"]["binary"]["path"])
    helper = Path(sealed["inputs"]["helper"]["path"])
    canonical = subprocess.run([str(helper), "--canonical-config", str(root / "fixtures/metal_qualified.yaml")],
                               cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    (root / "canonical.log").write_text(canonical.stdout + canonical.stderr, encoding="utf-8")
    require(canonical.returncode == 0 and hashlib.sha256(canonical.stdout.encode()).hexdigest() == QUALIFIED_SHA,
            "native typed canonical bytes changed; preserve red and review rather than adding an assumed hash")
    command = [sys.executable, "-B", str(ROOT / "scripts/qualification/qualify_flat_mixed_boundary_metal.py"),
               "--binary", str(binary), "--fixture", str(root / "fixtures/metal_exploratory.yaml"), "--output", str(destination)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800, check=False,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    (root / "core_runner.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    require(result.returncode == 0, "mixed-open core runner failed; raw evidence is preserved")
    from scripts.qualification.checkpointing.qualify_vulkan_restart import require_report_owner_identities
    from scripts.qualification.qualify_flat_east_radiative_native_metal import ledger_gate
    summary = json.loads((destination / "summary.json").read_text())
    resources = observed_resource_gate(summary["cases"])
    write_json(root / "resource_review.json", resources)
    checked = {}
    for name, record in summary["cases"].items():
        run_root = Path(record["validation"]["path"]).parent
        count = require_report_owner_identities(run_root)
        duration = 1 if name == "metalResumed" else record["duration"]
        require(count >= 850 * duration, "incomplete independent report-owner checks: " + name)
        checked[name] = {"ownerIdentityChecks": count, "ledgerOwners": ledger_gate(run_root)}
    raw_bytes = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    require(raw_bytes <= 128 * 1024 * 1024 and time.monotonic() - started < 1800, "core evidence exceeded resource bounds")
    require(resources["status"] == "bounded_resource_pass", "core run exceeded its declared 128-MiB RSS cap; physics evidence is preserved separately")
    binding(root)
    write_json(root / "core_rebind.json", {"schema": "nimbus.flat-mixed-metal-core-rebind.v1",
               "status": "bounded_core_pass", "bindingSha256": identity(root / "binding.json")["sha256"],
               "coreSummary": identity(destination / "summary.json"), "ownerChecks": checked,
               "resourceReview": identity(root / "resource_review.json"),
               "rawBytes": raw_bytes, "wallSeconds": time.monotonic() - started,
               "publicMetalAdmission": "guarded", "publicCliOutputProof": "pending_after_admission",
               "I1Complete": False, "wholeBackendReady": False})


def rollback(root: Path) -> None:
    sealed = binding(root)
    destination = root / "rollback"
    require(not destination.exists(), "rollback evidence root already exists")
    destination.mkdir()
    binary = sealed["inputs"]["coreTestBinary"]["path"]
    rows = []
    for name, selected, marker, mixed in (
        ("direct_mixed", "[metal][native_dry_tvd][mixed_open][full_step_rollback]", "[flat-native-late-failure]", True),
        ("headless_mixed", "[metal][native_dry_tvd][mixed_open][headless_step_rollback]", "failed_step_artifacts_unchanged=1", True),
        ("direct_controls", "Flat canonical and east-radiative native steps roll back a late scalar failure", "[flat-native-late-failure]", False),
        ("headless_control", "Headless east-radiative late scalar failure publishes no failed-step artifacts", "failed_step_artifacts_unchanged=1", False),
    ):
        result = subprocess.run([binary, selected], cwd=ROOT, capture_output=True, text=True, timeout=240,
                                check=False, env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE"})
        log = destination / (name + ".log")
        content = result.stdout + result.stderr
        log.write_text(content, encoding="utf-8")
        passed = result.returncode == 0 and "All tests passed" in content and marker in content and ("mixed_open=1" if mixed else "mixed_open=0") in content
        rows.append({"name": name, "filter": selected, "exitCode": result.returncode,
                     "passed": passed, "log": identity(log)})
        write_json(destination / "summary.json", {"status": "bounded_rollback_pass" if all(row["passed"] for row in rows) and len(rows) == 4 else "incomplete_or_red",
                   "bindingSha256": identity(root / "binding.json")["sha256"], "cases": rows,
                   "publicMetalAdmission": "guarded", "wholeBackendReady": False})
        require(passed, "exact mixed-open rollback failed; raw evidence is preserved")
    binding(root)


def public_scope(root: Path) -> None:
    sealed = binding(root)
    destination = root / "public_scope"
    require(not destination.exists(), "public scope evidence root already exists")
    destination.mkdir()
    staged = json.loads((root / "stage.json").read_text())
    binary = sealed["inputs"]["binary"]["path"]
    cases = [(row["name"], "metal_qualified", row["flags"], row["accepted"], row["rejectionMarkers"])
             for row in staged["futureCliContracts"]]
    cases += [(name, name, [], False,
               ["boundary_conditions.lateral.west:", "opposing pairs"] if name == "negative_unpaired_periodic" else
               ["qualified Metal execution is limited"])
              for name in staged["fixtures"] if name.startswith("negative_")]
    rows = []
    for name, fixture, flags, accepted, markers in cases:
        result = subprocess.run([binary, "--validate-only", "--config", staged["fixtures"][fixture]["path"],
                                 "--threads=1", *flags], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
        log = destination / (name + ".log")
        content = result.stdout + result.stderr
        log.write_text(content, encoding="utf-8")
        outcome = public_admission_result(result, accepted, markers)
        rows.append({"name": name, "expectedAccepted": accepted, **outcome,
                     "cli": flags, "fixture": staged["fixtures"][fixture], "log": identity(log)})
        write_json(destination / "summary.json", {"status": "running_or_red", "cases": rows,
                   "claimBoundary": "Post-admission exact cadence1 minimal NPY contract; positives must pass before negatives are meaningful",
                   "wholeBackendReady": False})
        require(outcome["passed"], "post-admission scope failed; current closed base guard must not be counted as negative proof")
    binding(root)
    write_json(destination / "summary.json", {"status": "bounded_public_scope_pass", "cases": rows,
               "bindingSha256": identity(root / "binding.json")["sha256"],
               "physicalExecution": "not_run_by_this_scope_gate", "wholeBackendReady": False, "I1Complete": False})


def preallocation(root: Path) -> None:
    sealed = binding(root)
    require("preallocationTestBinary" in sealed["inputs"], "bind the dedicated admission-only test binary first")
    destination = root / "preallocation"
    require(not destination.exists(), "preallocation evidence root already exists")
    destination.mkdir()
    binary = sealed["inputs"]["preallocationTestBinary"]["path"]
    selected = "[mixed_open][preallocation][injected_factory]"
    result = subprocess.run([binary, selected], cwd=ROOT, capture_output=True, text=True, timeout=90,
                            check=False, env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE"})
    content = result.stdout + result.stderr
    log = destination / "admission_only.log"
    log.write_text(content, encoding="utf-8")
    rows = re.findall(r"\[mixed-open-admission-only\] fault=(\d+) reject_memory=(\d+) allocation_boundary_calls=(\d+) field_elements=(\d+) dispatch_calls=(\d+)", content)
    expected_rows = [(str(fault), str(policy), str(boundary), "0", "0") for fault, policy, boundary in
                     ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (3, 1, 0), (4, 1, 0), (5, 1, 1))]
    passed = result.returncode == 0 and "All tests passed" in content and len(rows) == 7 and \
        sorted(rows) == sorted(expected_rows)
    write_json(destination / "summary.json", {"status": "bounded_injected_admission_pass" if passed else "red",
               "bindingSha256": identity(root / "binding.json")["sha256"], "testBinary": sealed["inputs"]["preallocationTestBinary"],
               "exitCode": result.returncode, "log": identity(log), "faultRows": rows,
               "claimBoundary": "Injected Metal factory capability/memory admission only; fitting control stops at field-allocation boundary; no real GPU or simulation physics",
               "physicalExecution": "not_run", "wholeBackendReady": False})
    require(passed, "dedicated preallocation gate failed; preserve the diagnostic")
    binding(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    for option in ("stage", "bind", "core", "rollback", "public-scope", "preallocation"):
        action.add_argument("--" + option, action="store_true")
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--helper", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    parser.add_argument("--core-test-binary", type=Path, default=ROOT / "bin/test_core_nested_production_step")
    parser.add_argument("--preallocation-test-binary", type=Path)
    parser.add_argument("--expected-binary-sha256")
    args = parser.parse_args()
    root = args.report_root.resolve()
    if args.stage:
        stage(root)
    elif args.bind:
        require(bool(args.expected_binary_sha256), "exact coordinated binary SHA is required")
        bind(root, args.binary.resolve(), args.helper.resolve(), args.core_test_binary.resolve(), args.expected_binary_sha256,
             args.preallocation_test_binary.resolve() if args.preallocation_test_binary else None)
    elif args.core:
        core(root)
    elif args.rollback:
        rollback(root)
    elif args.preallocation:
        preallocation(root)
    else:
        public_scope(root)
    print("[mixed-open-rebind] action complete; Ready and scientific qualification remain unclaimed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bound the exact static vertical-only Smagorinsky CPU candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/tests/science"))
from npy_support import read_float32_npy
from scripts.qualification.checkpointing.qualify_vulkan_restart import (
    require_report_owner_identities, require_restart_suffix,
)


SOURCE = ROOT / "tests/fixtures/configs/qualification/cgrid_dynamic_smagorinsky_ready.yaml"
STATIC_SOURCE = ROOT / "tests/fixtures/configs/vertical_only/static_cpu.yaml"
PBL_NEGATIVE = ROOT / "tests/fixtures/reference/turbulence/vertical_only_active_pbl_invalid.yaml"
FIELDS = ("p", "rho", "theta", "u", "v", "w", "qv", "qr")
LEDGERS = ("budget.csv", "energy_budget.csv", "moisture_budget.csv",
           "momentum_budget.csv", "radiation_budget.csv",
           "thermodynamic_source_budget.csv")
OWNERS = (
    ROOT / "src/init/config/config_v1_serialization_common.cpp",
    ROOT / "src/turbulence/schemes/smagorinsky/dynamic_smagorinsky.cpp",
    ROOT / "src/turbulence/schemes/smagorinsky/smagorinsky.cpp",
    ROOT / "src/turbulence/base/eddy_viscosity.cpp",
    ROOT / "src/core/orchestration/dynamics/sgs_apply.cpp",
    ROOT / "src/core/orchestration/dynamics/dynamics.cpp",
    ROOT / "src/core/runtime/runtime_config.cpp",
    ROOT / "include/core/runtime/runtime_config.hpp",
    ROOT / "include/core/runtime/flat_native_tvd_step_transaction.hpp",
    ROOT / "src/core/runtime/headless_runtime.cpp",
    ROOT / "include/core/runtime/science_profile.hpp",
    ROOT / "scripts/qualification/checkpointing/qualify_vulkan_restart.py",
    ROOT / "scripts/tests/science/npy_support.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise RuntimeError(f"expected one source token: {before!r}")
    return source.replace(before, after, 1)


def manifest_at(root: Path, time_s: float) -> Path:
    for manifest in sorted(root.glob("step_*/manifest.json")):
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if abs(float(document["simulation_time_s"]) - time_s) < 1.0e-9:
            return manifest
    raise RuntimeError(f"missing complete output frame at t={time_s}: {root}")


def checkpoint_at(root: Path, time_s: float) -> Path:
    for checkpoint in sorted((root / "checkpoints").glob("*.nimbus")):
        sidecar = Path(str(checkpoint) + ".json")
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if abs(float(metadata["simulation_time_s"]) - time_s) < 1.0e-9:
                return checkpoint
    raise RuntimeError(f"missing checkpoint at t={time_s}: {root}")


def sgs_owner(root: Path) -> dict[str, object]:
    ledger = root / "thermodynamic_source_budget.csv"
    with ledger.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("active SGS omitted science owner rows")
    keys = ("sgs_momentum_work_step_count", "sgs_committed_momentum_work_j",
            "sgs_moist_energy_step_complete")
    if any(key not in rows[0] for key in keys):
        raise RuntimeError("active SGS omitted a required work/energy owner")
    values = [[float(row[key]) for row in rows] for key in keys]
    if any(not math.isfinite(value) for series in values for value in series) or \
            max(values[0]) <= 0 or any(value != 1.0 for value in values[2]):
        raise RuntimeError("active SGS work/closure is incomplete or nonfinite")
    identities = require_report_owner_identities(root)
    if identities < 85 * len(rows):
        raise RuntimeError("active SGS report-owner identities are incomplete")
    return {"rows": len(rows), "ownerIdentityChecks": identities,
            "maximumMomentumWorkStepCount": max(values[0]),
            "maximumCommittedMomentumWorkJ": max(map(abs, values[1])),
            "ledgerSha256": sha256(ledger)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--guarded-exploratory", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()
    raw = args.raw_root.resolve()
    report = args.report_output.resolve()
    if not binary.is_file() or raw.exists() or report.exists():
        parser.error("binary must exist and outputs must be new")
    identity = {"simulatorSha256": sha256(binary),
                "canonicalFixtureSha256": sha256(SOURCE),
                "staticFixtureSha256": sha256(STATIC_SOURCE),
                "pblNegativeSha256": sha256(PBL_NEGATIVE),
                "runnerSha256": sha256(Path(__file__)),
                "ownerSha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                                for path in OWNERS}}
    source = SOURCE.read_text(encoding="utf-8")
    expected_static_vertical = replace_once(replace_once(
        source, "  mode: 3d\n", "  mode: vertical_only\n"),
        "  dynamic_coefficient: true\n", "  dynamic_coefficient: false\n")
    qualified_static_vertical = STATIC_SOURCE.read_text(encoding="utf-8")
    if qualified_static_vertical != expected_static_vertical:
        raise RuntimeError("static vertical-only fixture differs from its reviewed reference")
    qualified_static_three_d = replace_once(
        source, "  dynamic_coefficient: true\n", "  dynamic_coefficient: false\n")
    static_vertical = replace_once(
        qualified_static_vertical, "    mode: qualified\n", "    mode: exploratory\n") \
        if args.guarded_exploratory else qualified_static_vertical
    static_three_d = replace_once(
        qualified_static_three_d, "    mode: qualified\n", "    mode: exploratory\n") \
        if args.guarded_exploratory else qualified_static_three_d
    dynamic_vertical = replace_once(source, "  mode: 3d\n", "  mode: vertical_only\n")
    exploratory_dynamic_vertical = replace_once(
        dynamic_vertical, "    mode: qualified\n", "    mode: exploratory\n")
    raw.mkdir(parents=True)
    inputs = raw / "inputs"
    runs = raw / "runs"
    logs = raw / "logs"
    inputs.mkdir()
    runs.mkdir()
    logs.mkdir()
    configs = {
        "static_vertical": static_vertical,
        "static_three_d": static_three_d,
        "dynamic_three_d": source,
        "dynamic_vertical_qualified": dynamic_vertical,
        "dynamic_vertical_exploratory": exploratory_dynamic_vertical,
    }
    if args.guarded_exploratory:
        configs["qualified_static_vertical"] = qualified_static_vertical
    paths = {}
    for name, text in configs.items():
        path = inputs / f"{name}.yaml"
        path.write_text(text, encoding="utf-8")
        paths[name] = path
    paths["active_pbl_overlap"] = PBL_NEGATIVE
    rows: list[dict[str, object]] = []
    started = time.monotonic()
    passing_status = "pass_guarded_exploratory_cpu_candidate" if \
        args.guarded_exploratory else "pass_static_cpu_candidate"
    status = passing_status

    for name, config in paths.items():
        command = [str(binary), "--config", str(config), "--validate-only"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                   text=True, timeout=45, check=False)
        log = logs / f"{name}_validate.log"
        log.write_text(completed.stdout + "\n" + completed.stderr,
                       encoding="utf-8")
        expected_positive = name in {"static_vertical", "static_three_d",
                                     "dynamic_three_d"}
        passed = (completed.returncode == 0) == expected_positive
        if name == "qualified_static_vertical":
            passed = passed and "guarded pending whole-step failure atomicity" in \
                log.read_text(encoding="utf-8")
        if name == "dynamic_vertical_qualified":
            diagnostic = log.read_text(encoding="utf-8")
            passed = passed and ("dynamic Smagorinsky is qualified only" in diagnostic or
                                 "requires turbulence.mode=3d" in diagnostic)
        if name == "dynamic_vertical_exploratory":
            passed = passed and "requires turbulence.mode=3d" in \
                log.read_text(encoding="utf-8")
        if name == "active_pbl_overlap":
            passed = passed and "vertical momentum/heat/moisture mixing" in \
                log.read_text(encoding="utf-8")
        rows.append({"name": name, "kind": "validate", "exitCode": completed.returncode,
                     "sourceSha256": sha256(config), "logSha256": sha256(log),
                     "pass": passed})
        if not passed:
            status = "admission_deviation"
            break

    def run_case(name: str, config: Path, threads: int,
                 resume: Path | None = None) -> bool:
        nonlocal status
        if time.monotonic() - started >= 240:
            status = "wall_cap"
            return False
        outdir = runs / name
        command = [str(binary), "--headless", "--config", str(config),
                   "--duration=2", "--write-every=1", "--checkpoint-every=1",
                   f"--threads={threads}", f"--outdir={outdir}",
                   "--log-profile=normal", "--no-timing"]
        if resume is not None:
            command.append(f"--resume={resume}")
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(threads)
        environment["OMP_DYNAMIC"] = "FALSE"
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment,
                                       capture_output=True, text=True,
                                       timeout=120, check=False)
            code = completed.returncode
            message = completed.stdout + "\n" + completed.stderr
        except subprocess.TimeoutExpired as error:
            code = 124
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            message = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
        log = logs / f"{name}.log"
        log.write_text(message, encoding="utf-8")
        execution_mode = "exploratory" if args.guarded_exploratory else "qualified"
        passed = code == 0 and "[SCIENTIFIC INVALIDITY]" not in message and \
            f"Execution mode: {execution_mode}" in message and \
            "[COMPUTE] backend=cpu" in message and "fallback=off" in message
        memory_fields = None
        if name.startswith("vertical_"):
            match = re.search(r"solver_resident=(\d+) fields", message)
            memory_fields = int(match.group(1)) if match else None
            passed = passed and memory_fields is not None and memory_fields >= 292
        if passed and resume is not None:
            passed = "[CHECKPOINT] resumed" in message
        if passed:
            summary = json.loads((outdir / "validation_summary.json").read_text(
                encoding="utf-8"))
            passed = summary.get("failed") is False and \
                summary.get("total_violations") == 0
        if passed:
            for time_s in (2.0,) if resume else (0.0, 1.0, 2.0):
                step = manifest_at(outdir, time_s).parent
                for field in FIELDS:
                    shape, _, values = read_float32_npy(step / f"{field}.npy")
                    if shape != (12, 12, 16) or any(not math.isfinite(value)
                                                    for value in values):
                        passed = False
                        break
                if not passed:
                    break
        owner = sgs_owner(outdir) if passed else None
        rows.append({"name": name, "kind": "physical", "exitCode": code,
                     "threads": threads, "resume": str(resume) if resume else None,
                     "sourceSha256": sha256(config), "logSha256": sha256(log),
                     "solverResidentFieldEquivalents": memory_fields,
                     "sgsOwner": owner, "pass": passed})
        files = [item for item in raw.rglob("*") if item.is_file()]
        if len(files) >= 3000 or sum(item.stat().st_size for item in files) >= 128 * 1024 * 1024:
            status = "resource_cap"
            return False
        if not passed:
            status = "physical_or_science_deviation"
        return passed

    if status == passing_status:
        if run_case("vertical_t1", paths["static_vertical"], 1) and \
                run_case("vertical_t4", paths["static_vertical"], 4) and \
                run_case("static_three_d_t1", paths["static_three_d"], 1):
            checkpoint = checkpoint_at(runs / "vertical_t1", 1.0)
            run_case("vertical_resumed_t4", paths["static_vertical"], 4,
                     checkpoint)

    comparisons = {}
    if status == passing_status:
        one = runs / "vertical_t1"
        four = runs / "vertical_t4"
        resumed = runs / "vertical_resumed_t4"
        for time_s in (0.0, 1.0, 2.0):
            one_step = manifest_at(one, time_s).parent
            four_step = manifest_at(four, time_s).parent
            for field in FIELDS:
                if (one_step / f"{field}.npy").read_bytes() != \
                        (four_step / f"{field}.npy").read_bytes():
                    status = "thread_field_mismatch"
                    break
            if status != passing_status:
                break
        if status == passing_status:
            comparisons["threadLedgers"] = {name: sha256(one / name) == sha256(four / name)
                                            for name in LEDGERS}
            if not all(comparisons["threadLedgers"].values()):
                status = "thread_ledger_mismatch"
        if status == passing_status:
            for field in FIELDS:
                if (manifest_at(one, 2.0).parent / f"{field}.npy").read_bytes() != \
                        (manifest_at(resumed, 2.0).parent / f"{field}.npy").read_bytes():
                    status = "restart_field_mismatch"
                    break
        if status == passing_status:
            require_restart_suffix(one, resumed)
            comparisons["restartLedgerSuffix"] = True
            _, _, vertical_u = read_float32_npy(manifest_at(one, 2.0).parent / "u.npy")
            _, _, control_u = read_float32_npy(
                manifest_at(runs / "static_three_d_t1", 2.0).parent / "u.npy")
            comparisons["modeResponseMaximumU"] = max(
                abs(a - b) for a, b in zip(vertical_u, control_u))
            if comparisons["modeResponseMaximumU"] <= 0:
                status = "inert_mode"

    after = {"simulatorSha256": sha256(binary),
             "canonicalFixtureSha256": sha256(SOURCE),
             "staticFixtureSha256": sha256(STATIC_SOURCE),
             "pblNegativeSha256": sha256(PBL_NEGATIVE),
             "runnerSha256": sha256(Path(__file__)),
             "ownerSha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                             for path in OWNERS}}
    if identity != after:
        status = "source_identity_drift"
    files = [item for item in raw.rglob("*") if item.is_file()]
    report.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": "nimbus.vertical-only-static-cpu-candidate.v1",
              "status": status,
              "claimBoundary": (
                  "Exact static vertical-only flat CPU candidate exercised through guarded-exploratory mode; "
                  "public qualified acceptance is reported separately."
                  if args.guarded_exploratory else
                  "Exact static vertical-only flat CPU candidate exercised through ordinary qualified admission."
              ) + " Dynamic+non3D and PBL overlap reject; broader backend parity and Ready remain separate.",
              "inputs": identity,
              "cases": rows,
              "comparisons": comparisons,
              "wallSeconds": time.monotonic() - started,
              "rawFiles": len(files),
              "rawBytes": sum(item.stat().st_size for item in files),
              "rawRoot": str(raw),
              "guardedExploratory": args.guarded_exploratory,
              "engineeringIncomplete": True}
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"{status}: {len(rows)} bounded cases, {result['rawFiles']} files, "
          f"{result['rawBytes']} bytes")
    return 0 if status == passing_status else 1


if __name__ == "__main__":
    raise SystemExit(main())

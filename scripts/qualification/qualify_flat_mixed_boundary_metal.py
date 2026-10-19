#!/usr/bin/env python3
"""Bound the guarded native Metal mixed-boundary dry C-grid transport branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.survey_generated_metal import (
    FIELDS,
    LEDGERS,
    NATIVE_TVD_OPERATION,
    compare_fields,
    ledger_suffix,
    replace_once,
)
from scripts.tests.catalog.qualify_catalog_bounded_runs import dispatch_counts, write_json_atomic


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def run_case(binary: Path, config: Path, outdir: Path, backend: str, threads: int,
             duration: int, checkpoint_every: int = 0, resume: Path | None = None,
             expected_native_updates: int | None = None) -> dict:
    command = [str(binary), "--headless", "--config", str(config), f"--duration={duration}",
               "--write-every=1", f"--checkpoint-every={checkpoint_every}",
               f"--threads={threads}", f"--outdir={outdir}", "--log-profile=normal", "--no-timing"]
    if resume is not None:
        command.append(f"--resume={resume}")
    if backend == "vulkan":
        command = [str(ROOT / "scripts/run_with_vulkan_env.sh"), *command]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=180, check=False,
                               env={**os.environ, "OMP_NUM_THREADS": str(threads), "OMP_DYNAMIC": "FALSE"})
    log = outdir.with_suffix(".log")
    log.write_text(completed.stdout)
    require(completed.returncode == 0 and "[SCIENTIFIC INVALIDITY]" not in completed.stdout,
            f"{backend}/{threads} mixed-boundary runtime failed: {completed.stdout[-1000:]}")
    validation = outdir / "validation_summary.json"
    document = json.loads(validation.read_text())
    require(not document["failed"] and document["total_violations"] == 0,
            f"{backend}/{threads} strict validation failed")
    final = sorted(outdir.glob("step_*/manifest.json"))[-1]
    manifest = json.loads(final.read_text())
    require(sorted(manifest["published_fields"]) == sorted(FIELDS) and
            manifest["grid"]["dimensions"] == {"x": 18, "y": 18, "z": 18},
            f"{backend}/{threads} published the wrong minimal fields or grid")
    reports, operations = dispatch_counts(completed.stdout, backend)
    record = {"backend": backend, "threads": threads, "duration": duration,
              "config": identity(config), "log": identity(log), "validation": identity(validation),
              "manifest": identity(final), "dispatchReports": reports,
              "operations": operations, "fallbackOff": "fallback=off" in completed.stdout}
    require(record["fallbackOff"], f"{backend}/{threads} used a fallback")
    if backend == "metal":
        expected_updates = duration * 30 if expected_native_updates is None else expected_native_updates
        require(operations.get(NATIVE_TVD_OPERATION, 0) == expected_updates,
                f"Metal/{threads} did not dispatch native conservative TVD transport")
    return record


def exact_ledgers(left: Path, right: Path) -> dict[str, bool]:
    return {name: (left / name).read_bytes() == (right / name).read_bytes() for name in LEDGERS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--fixture", type=Path,
                        default=ROOT / "tests/fixtures/reference/catalog/flat_mixed_boundary_metal_exploratory_v1.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    fixture = args.fixture.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {key: identity(path) for key, path in
              (("binary", binary), ("fixture", fixture), ("qualifier", Path(__file__)),
               ("comparator", ROOT / "scripts/qualification/survey_generated_metal.py"),
               ("vulkanWrapper", ROOT / "scripts/run_with_vulkan_env.sh"))}
    source = fixture.read_text()
    require(source.count("    backend: metal\n") == 1 and
            source.count("    mode: exploratory\n") == 1,
            "mixed-boundary fixture must be exploratory Metal with one exact selector")
    cpu_config = out / "cpu-qualified.yaml"
    vulkan_config = out / "vulkan-qualified.yaml"
    qualified_metal = out / "metal-qualified-rejected.yaml"
    structural_neighbor = out / "metal-structural-neighbor.yaml"
    invalid_periodic = out / "metal-invalid-periodic.yaml"
    cpu_config.write_text(replace_once(replace_once(source, "    backend: metal\n", "    backend: cpu\n"),
                                       "    mode: exploratory\n", "    mode: qualified\n"))
    vulkan_config.write_text(replace_once(replace_once(source, "    backend: metal\n", "    backend: gpu\n"),
                                          "    mode: exploratory\n", "    mode: qualified\n"))
    qualified_metal.write_text(replace_once(source, "    mode: exploratory\n", "    mode: qualified\n"))
    structural_neighbor.write_text(replace_once(source, "    east: rigid_slip\n", "    east: rigid\n"))
    invalid_periodic.write_text(replace_once(source, "    west: rigid_slip\n", "    west: periodic\n"))
    admission = {}
    for backend, path, accepted in (("cpu", cpu_config, True),
                                    ("vulkan", vulkan_config, True),
                                    ("metal", qualified_metal, False)):
        completed = subprocess.run([str(binary), "--config", str(path), "--validate-only"],
                                   cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=30, check=False)
        log = out / f"{backend}-admission.log"
        log.write_text(completed.stdout)
        require((completed.returncode == 0) == accepted and
                ("[VALIDATE] PASS:" in completed.stdout if accepted else
                 "qualified Metal execution is limited" in completed.stdout),
                f"{backend} admission did not preserve the reviewed public boundary")
        admission[backend] = {"accepted": accepted, "config": identity(path), "log": identity(log)}
    invalid = subprocess.run([str(binary), "--config", str(invalid_periodic), "--validate-only"],
                             cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=30, check=False)
    invalid_log = out / "invalid-periodic-admission.log"
    invalid_log.write_text(invalid.stdout)
    require(invalid.returncode != 0 and "periodic lateral boundaries" in invalid.stdout,
            "unpaired periodic lateral face did not fail strict input admission")
    cases = {}
    for name, path, backend, threads in (
            ("cpu1", cpu_config, "cpu", 1), ("cpu4", cpu_config, "cpu", 4),
            ("vulkan1", vulkan_config, "vulkan", 1),
            ("metal1", fixture, "metal", 1), ("metal4", fixture, "metal", 4)):
        cases[name] = run_case(binary, path, out / name, backend, threads, 1)
    continuous = out / "metal-continuous"
    cases["metalContinuous"] = run_case(binary, fixture, continuous, "metal", 1, 2, 1)
    checkpoint = continuous / "checkpoints/checkpoint_step_000000010.nimbus"
    require(checkpoint.is_file(), "native Metal mixed-boundary halfway checkpoint is missing")
    resumed = out / "metal-resumed"
    cases["metalResumed"] = run_case(binary, fixture, resumed, "metal", 4, 2, 0, checkpoint,
                                      expected_native_updates=30)
    cases["metalStructuralNeighbor"] = run_case(
        binary, structural_neighbor, out / "metal-structural-neighbor", "metal", 1, 1,
        expected_native_updates=0)
    parity = {
        "cpuMetal": compare_fields(out / "cpu1", out / "metal1", False),
        "cpuVulkan": compare_fields(out / "cpu1", out / "vulkan1", False),
        "cpuThreads": compare_fields(out / "cpu1", out / "cpu4", True),
        "metalThreads": compare_fields(out / "metal1", out / "metal4", True),
        "metalRestart": compare_fields(continuous, resumed, True),
    }
    ledgers = {"cpuThreads": exact_ledgers(out / "cpu1", out / "cpu4"),
               "metalThreads": exact_ledgers(out / "metal1", out / "metal4"),
               "metalRestartSuffix": ledger_suffix(continuous, resumed)}
    require(all(field["passed"] for result in parity.values() for field in result.values()) and
            all(passed for result in ledgers.values() for passed in result.values()),
            "mixed-boundary field or ledger equivalence failed")
    if any(identity(Path(record["path"])) != record for record in inputs.values()):
        raise RuntimeError("mixed-boundary evidence inputs changed during qualification")
    report = {"schema": "nimbus.flat-mixed-boundary-metal-probe.v1", "status": "bounded_pass",
              "claimBoundary": "Qualified CPU/Vulkan and strict exploratory native Metal mixed-boundary evidence; qualified Metal public admission remains guarded and full boundary matrix is not Ready.",
              "inputs": inputs, "admission": admission, "cases": cases,
              "negativeScope": {"unpairedPeriodic": {"config": identity(invalid_periodic),
                                                      "log": identity(invalid_log)},
                                "adjacentBoundary": identity(structural_neighbor)},
              "checkpoint": identity(checkpoint), "parity": parity, "ledgers": ledgers,
              "wholeBackendReady": False}
    write_json_atomic(out / "summary.json", report)
    print(json.dumps({"status": "bounded_pass", "metalNativeUpdates": 30,
                      "cpuVulkanMetalParity": True, "threadAndRestartExact": True,
                      "qualifiedMetalAdmission": "guarded"}))


if __name__ == "__main__":
    main()

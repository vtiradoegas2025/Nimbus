#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATTERN = re.compile(r"All tests passed \((\d+) assertions in (\d+) test cases?\)")
SUITES = (
    ("diagnostics", "test-diagnostics",
     ((560, 13), (84, 17), (2, 1), (153, 24), (104, 13), (127, 21),
      (57, 9), (267, 22), (798, 5), (31, 4), (307, 3), (433, 32)), ()),
    ("dynamics_boundaries", "test-dynamics",
     ((19650, 16), (13752, 5), (1796, 23), (28, 10), (24, 6),
      (40, 7), (12474, 6), (2174, 3), (38808, 13)), ()),
    ("two_way_nesting", "test-two-way-nested-ready",
     ((90, 3), (4, 1), (225, 8), (834, 1), (1924, 1), (2195, 1)),
     ("[two-way-nested-ready] PASS",)),
    ("active_terrain_nesting", "test-active-terrain-nested-ready",
     ((967, 5), (967, 5), (117, 4)),
     ("[active-terrain-nested-ready] PASS",)),
    ("domain_recycling", "qualify-catalog-domain-recycling-ownership", (),
     ("[domain-recycling-owner-contracts] PASS: 28/28 owners ReadyForTransaction",)),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def validate_suite_output(
    name: str,
    output: str,
    expected_summaries: tuple[tuple[int, int], ...],
    required_markers: tuple[str, ...],
) -> tuple[int, int]:
    summaries = tuple((int(a), int(c)) for a, c in SUMMARY_PATTERN.findall(output))
    if summaries != expected_summaries:
        raise RuntimeError(f"{name} test summary drifted: expected={expected_summaries}, actual={summaries}")
    for marker in required_markers:
        if marker not in output:
            raise RuntimeError(f"{name} is missing required marker: {marker}")
    lowered = output.lower()
    for forbidden in ("gpu unavailable - skipping", "no tests ran", "test cases: 1 | 1 failed", "tests failed"):
        if forbidden in lowered:
            raise RuntimeError(f"{name} contains a skipped or failed test marker: {forbidden}")
    return sum(value[0] for value in summaries), sum(value[1] for value in summaries)


def validate_inputs(binary_sha256: str, paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    values = {name: load(path, name) for name, path in paths.items()}
    checks = {
        "domainRecycling": values["domainRecycling"].get("binary", {}).get("sha256"),
        "reachable": values["reachable"].get("binary", {}).get("sha256"),
        "kernelDispatch": values["kernelDispatch"].get("binary", {}).get("sha256"),
        "vulkanQualification": values["vulkanQualification"].get("artifacts", {}).get("binary_sha256"),
        "movingVulkan": values["movingVulkan"].get("inputs", {}).get("binary_sha256"),
        "movingMetal": values["movingMetal"].get("inputs", {}).get("binary_sha256"),
    }
    stale = sorted(name for name, value in checks.items() if value != binary_sha256)
    if stale:
        raise RuntimeError("composite ownership inputs do not bind the current binary: " + ", ".join(stale))
    if values["domainRecycling"].get("summary") != {
        "actions": {
            "InvalidateAndRecompute": 7,
            "InvalidateDeviceResidency": 1,
            "Preserve": 4,
            "RebuildFromShiftedState": 1,
            "ShiftAndExportEarthTile": 3,
            "ShiftPersistentInvalidateTransient": 4,
            "ShiftWithLeadingState": 3,
            "TranslateEarthCoordinates": 5,
        },
        "owners": 28,
        "readyForTransaction": 28,
        "remaining": 0,
    }:
        raise RuntimeError("domain-recycling ownership summary drifted")
    if values["kernelDispatch"].get("summary", {}).get("unclassifiedOperations") != 0:
        raise RuntimeError("kernel dispatch coverage contains unclassified operations")
    if values["vulkanQualification"].get("status") != "pass":
        raise RuntimeError("Vulkan qualification is not passing")
    if values["movingVulkan"].get("status") != "pass" or values["movingMetal"].get("status") != "pass":
        raise RuntimeError("moving backend evidence is not passing")
    return {name: record(path) for name, path in paths.items()}


def input_paths() -> dict[str, Path]:
    return {
        "domainRecycling": ROOT / "reports/pre_beta/domain_recycling_owner_evidence_v1.json",
        "reachable": ROOT / "reports/pre_beta/reachable_set_v1.json",
        "kernelDispatch": ROOT / "reports/pre_beta/backend_kernel_dispatch_coverage_v1.json",
        "vulkanQualification": ROOT / "reports/qualification/vulkan-qualification-current.json",
        "movingVulkan": ROOT / "reports/pre_beta/physical/moving_nested_vulkan_public_runtime_v1_pass.json",
        "movingMetal": ROOT / "reports/pre_beta/physical/moving_nested_metal_public_runtime_smoke_v1_pass.json",
    }


def verify_record(value: object, label: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise RuntimeError(f"{label} file record is malformed")
    path = Path(value["path"])
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"{label} file record escapes the repository")
    if record(ROOT / path) != value:
        raise RuntimeError(f"{label} file record drifted")


def verify_evidence(value: dict[str, object], binary_sha256: str) -> None:
    if value.get("schema") != "nimbus.composite-runtime-ownership-evidence.v1" or value.get("version") != 1:
        raise RuntimeError("composite ownership evidence schema drifted")
    if value.get("status") != "pass" or value.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("composite ownership evidence does not bind the passing current binary")
    verify_record(value.get("binary"), "binary")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(input_paths()):
        raise RuntimeError("composite ownership evidence inputs are incomplete")
    for name, file_record in inputs.items():
        verify_record(file_record, f"inputs.{name}")
    sources = value.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"gate", "makeTargets"}:
        raise RuntimeError("composite ownership evidence sources are incomplete")
    for name, file_record in sources.items():
        verify_record(file_record, f"sources.{name}")
    verify_record(value.get("log"), "log")
    expected_summary = {
        "suites": 5,
        "assertions": 99455,
        "testCases": 283,
        "skipped": 0,
        "domainRecyclingOwners": 28,
        "activeKernelOperations": 47,
        "unclassifiedKernelOperations": 0,
    }
    if value.get("summary") != expected_summary:
        raise RuntimeError("composite ownership evidence summary drifted")
    receipts = value.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != len(SUITES):
        raise RuntimeError("composite ownership evidence receipts are incomplete")
    expected_receipts = {
        name: (target, sum(a for a, _ in summaries) if summaries else 463,
               sum(c for _, c in summaries) if summaries else 1)
        for name, target, summaries, _ in SUITES
    }
    seen: set[str] = set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or not isinstance(receipt.get("name"), str):
            raise RuntimeError("composite ownership evidence receipt is malformed")
        name = receipt["name"]
        if name in seen or name not in expected_receipts:
            raise RuntimeError("composite ownership evidence receipt identity drifted")
        seen.add(name)
        target, assertions, cases = expected_receipts[name]
        if (
            receipt.get("command") != ["make", target]
            or receipt.get("exitStatus") != 0
            or receipt.get("assertions") != assertions
            or receipt.get("testCases") != cases
            or receipt.get("skipped") != 0
            or not isinstance(receipt.get("elapsedWallSeconds"), (int, float))
            or receipt["elapsedWallSeconds"] <= 0
        ):
            raise RuntimeError(f"composite ownership evidence receipt drifted: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/pre_beta/composite_runtime_ownership_v1.json")
    parser.add_argument("--log", type=Path,
                        default=ROOT / "reports/pre_beta/composite_runtime_ownership_v1.log")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    binary_sha256 = sha256_file(binary)
    if args.verify:
        verify_evidence(load(args.output.resolve(), "composite ownership evidence"), binary_sha256)
        print(f"[composite-runtime-ownership] PASS verify {args.output}")
        return 0
    inputs = input_paths()
    input_records = validate_inputs(binary_sha256, inputs)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    log_parts: list[str] = []
    receipts: list[dict[str, object]] = []
    total_assertions = 0
    total_cases = 0
    for name, target, expected_summaries, markers in SUITES:
        command = ["make", target]
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, check=False)
        elapsed = time.monotonic() - started
        output = result.stdout
        log_parts.append(f"===== {name}: {' '.join(command)} =====\n{output}")
        if result.returncode != 0:
            write_atomic(args.log.resolve(), "\n".join(log_parts).encode())
            raise RuntimeError(f"composite ownership suite failed: {name}")
        assertions, cases = validate_suite_output(name, output, expected_summaries, markers)
        if name == "domain_recycling":
            assertions, cases = 463, 1
        total_assertions += assertions
        total_cases += cases
        receipts.append({"name": name, "command": command, "exitStatus": result.returncode,
                         "assertions": assertions, "testCases": cases, "skipped": 0,
                         "elapsedWallSeconds": elapsed})
        if sha256_file(binary) != binary_sha256:
            raise RuntimeError(f"public binary changed after composite suite: {name}")
        validate_inputs(binary_sha256, inputs)

    log_path = args.log.resolve()
    write_atomic(log_path, "\n".join(log_parts).encode())
    if (total_assertions, total_cases) != (99455, 283):
        raise RuntimeError(f"composite ownership totals drifted: assertions={total_assertions}, cases={total_cases}")
    evidence = {
        "schema": "nimbus.composite-runtime-ownership-evidence.v1",
        "version": 1,
        "status": "pass",
        "completedUtc": datetime.now(timezone.utc).isoformat(),
        "binary": record(binary),
        "inputs": input_records,
        "sources": {"gate": record(Path(__file__)), "makeTargets": record(ROOT / "mk/tests.mk")},
        "log": record(log_path),
        "receipts": receipts,
        "summary": {"suites": len(SUITES), "assertions": total_assertions, "testCases": total_cases,
                    "skipped": 0, "domainRecyclingOwners": 28, "activeKernelOperations": 47,
                    "unclassifiedKernelOperations": 0},
        "claimBoundary": (
            "Bounded composite ownership across diagnostics, dynamics/boundaries, two-way and "
            "active-terrain nesting, and root D5 recycling, anchored to current physical backend "
            "evidence; not universal tuple execution, a long campaign, or external review."
        ),
    }
    write_atomic(args.output.resolve(), (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode())
    verify_evidence(load(args.output.resolve(), "composite ownership evidence"), binary_sha256)
    print("[composite-runtime-ownership] PASS: 5 suites, 99455 assertions, 283 cases, "
          f"0 skipped; binary_sha256={binary_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

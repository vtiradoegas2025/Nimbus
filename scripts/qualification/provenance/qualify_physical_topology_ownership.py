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
TEST_BINARY = "./bin/test_core_nested_production_step"
VULKAN_WRAPPER = "./scripts/run_with_vulkan_env.sh"
SUITES = (
    ("mapped_terrain", ["make", "test-native-mapped-terrain-ready"],
     ((5792, 8), (6637, 3)),
     ("[native-mapped-terrain-ready] PASS backend=vulkan",
      "[native-mapped-terrain-ready] PASS backend=metal")),
    ("constant_velocity_vulkan", ["make", "qualify-constant-velocity-vulkan"], (),
     ("[constant-velocity-vulkan] PASS", "owner_identity_checks=765", "ledgers=6")),
    ("stationary_nested_vulkan", ["make", "qualify-stationary-nested-vulkan-runtime"], (),
     ("[stationary-nested-admission] PASS", "[stationary-nested-vulkan-runtime] PASS")),
    ("stationary_nested_smagorinsky_vulkan",
     ["make", "qualify-stationary-nested-vulkan-smagorinsky-runtime"], (),
     ("[stationary-nested-smagorinsky-admission] PASS", "non-vacuous native Smagorinsky work")),
    ("stationary_nested_kessler_smagorinsky_vulkan",
     ["make", "qualify-stationary-nested-vulkan-kessler-smagorinsky-runtime"], (),
     ("[stationary-nested-kessler-smagorinsky-admission] PASS",
      "conservative active native Kessler work and fallout")),
    ("stationary_nested_radiation_vulkan",
     ["make", "qualify-stationary-nested-radiation-runtime"], (),
     ("[stationary-nested-radiation-admission] PASS", "host-owned longwave activity")),
    ("stationary_one_way_vulkan",
     [VULKAN_WRAPPER, TEST_BINARY, "stationary one-way child executes counted physical Vulkan work"],
     ((45433, 1),), ()),
    ("stationary_active_vulkan",
     [VULKAN_WRAPPER, TEST_BINARY,
      "stationary nested Kessler and Smagorinsky physics qualify physical Vulkan parity"],
     ((29136, 1),), ()),
    ("stationary_wave_vulkan",
     [VULKAN_WRAPPER, TEST_BINARY, "stationary child transmits a physical Vulkan acoustic wave"],
     ((15169, 1),), ()),
    ("relocation_vulkan",
     [VULKAN_WRAPPER, TEST_BINARY, "synchronized child relocation resumes exact physical Vulkan work"],
     ((59035, 1),), ()),
    ("recycling_metal",
     [TEST_BINARY, "constant-velocity recycling commits exact all-owner Metal transaction"],
     ((71, 1),), ()),
    ("relocation_metal",
     [TEST_BINARY, "synchronized child relocation resumes exact physical Metal work"],
     ((59035, 1),), ()),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {"path": resolved.relative_to(ROOT).as_posix(), "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved)}


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


def validate_output(name: str, output: str, summaries: tuple[tuple[int, int], ...],
                    markers: tuple[str, ...]) -> tuple[int, int]:
    actual = tuple((int(a), int(c)) for a, c in SUMMARY_PATTERN.findall(output))
    if actual != summaries:
        raise RuntimeError(f"{name} physical summary drifted: expected={summaries}, actual={actual}")
    for marker in markers:
        if marker not in output:
            raise RuntimeError(f"{name} is missing physical marker: {marker}")
    lowered = output.lower()
    for forbidden in ("gpu unavailable - skipping", "no tests ran", "tests failed", "fallback=on"):
        if forbidden in lowered:
            raise RuntimeError(f"{name} contains a skipped, failed, or fallback marker: {forbidden}")
    return sum(a for a, _ in actual), sum(c for _, c in actual)


def anchor_paths() -> dict[str, Path]:
    return {
        "compositeRuntime": ROOT / "reports/pre_beta/composite_runtime_ownership_v1.json",
        "kernelDispatch": ROOT / "reports/pre_beta/backend_kernel_dispatch_coverage_v1.json",
        "vulkanQualification": ROOT / "reports/qualification/vulkan-qualification-current.json",
        "movingVulkan": ROOT / "reports/pre_beta/physical/moving_nested_vulkan_public_runtime_v1_pass.json",
        "movingMetal": ROOT / "reports/pre_beta/physical/moving_nested_metal_public_runtime_smoke_v1_pass.json",
    }


def validate_anchors(binary_sha256: str) -> dict[str, dict[str, object]]:
    paths = anchor_paths()
    values = {name: load(path, name) for name, path in paths.items()}
    identities = {
        "compositeRuntime": values["compositeRuntime"].get("binary", {}).get("sha256"),
        "kernelDispatch": values["kernelDispatch"].get("binary", {}).get("sha256"),
        "vulkanQualification": values["vulkanQualification"].get("artifacts", {}).get("binary_sha256"),
        "movingVulkan": values["movingVulkan"].get("inputs", {}).get("binary_sha256"),
        "movingMetal": values["movingMetal"].get("inputs", {}).get("binary_sha256"),
    }
    stale = sorted(name for name, identity in identities.items() if identity != binary_sha256)
    if stale:
        raise RuntimeError("physical topology anchors are stale: " + ", ".join(stale))
    if values["compositeRuntime"].get("status") != "pass":
        raise RuntimeError("composite runtime ownership anchor is not passing")
    if values["kernelDispatch"].get("summary", {}).get("unclassifiedOperations") != 0:
        raise RuntimeError("kernel dispatch anchor has unclassified operations")
    if values["vulkanQualification"].get("status") != "pass":
        raise RuntimeError("Vulkan qualification anchor is not passing")
    if values["movingVulkan"].get("status") != "pass" or values["movingMetal"].get("status") != "pass":
        raise RuntimeError("moving topology anchors are not passing")
    return {name: record(path) for name, path in paths.items()}


def verify_record(value: object, label: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise RuntimeError(f"{label} record is malformed")
    path = Path(value["path"])
    if path.is_absolute() or ".." in path.parts or record(ROOT / path) != value:
        raise RuntimeError(f"{label} record drifted")


def expected_receipts() -> dict[str, tuple[list[str], int, int]]:
    return {
        name: (command, sum(a for a, _ in summaries), sum(c for _, c in summaries))
        for name, command, summaries, _ in SUITES
    }


def verify_evidence(value: dict[str, object], binary_sha256: str) -> None:
    if value.get("schema") != "nimbus.physical-topology-ownership-evidence.v1" or value.get("version") != 1:
        raise RuntimeError("physical topology evidence schema drifted")
    if value.get("status") != "pass" or value.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("physical topology evidence does not bind the passing current binary")
    verify_record(value.get("binary"), "binary")
    verify_record(value.get("physicalSnapshot"), "physicalSnapshot")
    if value.get("physicalSnapshot", {}).get("sha256") != binary_sha256:
        raise RuntimeError("physical topology snapshot differs from the public binary")
    anchors = value.get("anchors")
    if not isinstance(anchors, dict) or set(anchors) != set(anchor_paths()):
        raise RuntimeError("physical topology anchors are incomplete")
    for name, file_record in anchors.items():
        verify_record(file_record, f"anchors.{name}")
    sources = value.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"gate", "testsMake", "workflowsMake"}:
        raise RuntimeError("physical topology sources are incomplete")
    for name, file_record in sources.items():
        verify_record(file_record, f"sources.{name}")
    verify_record(value.get("log"), "log")
    if value.get("summary") != {"suites": 12, "assertions": 220308, "testCases": 17, "skipped": 0,
                                "fallbackUsed": False, "physicalVulkan": True, "physicalMetal": True}:
        raise RuntimeError("physical topology summary drifted")
    receipts = value.get("receipts")
    expected = expected_receipts()
    if not isinstance(receipts, list) or len(receipts) != len(expected):
        raise RuntimeError("physical topology receipts are incomplete")
    seen: set[str] = set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or receipt.get("name") not in expected or receipt["name"] in seen:
            raise RuntimeError("physical topology receipt identity drifted")
        name = receipt["name"]
        seen.add(name)
        command, assertions, cases = expected[name]
        if (receipt.get("command") != command or receipt.get("exitStatus") != 0
                or receipt.get("assertions") != assertions or receipt.get("testCases") != cases
                or receipt.get("skipped") != 0 or receipt.get("fallbackUsed") is not False
                or not isinstance(receipt.get("elapsedWallSeconds"), (int, float))
                or receipt["elapsedWallSeconds"] <= 0):
            raise RuntimeError(f"physical topology receipt drifted: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "bin/nimbus_sim_physical_qualification")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/pre_beta/physical_topology_ownership_v1.json")
    parser.add_argument("--log", type=Path,
                        default=ROOT / "reports/pre_beta/physical_topology_ownership_v1.log")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    snapshot = args.snapshot.resolve(strict=True)
    binary_sha256 = sha256_file(binary)
    if args.verify:
        verify_evidence(load(args.output.resolve(), "physical topology evidence"), binary_sha256)
        print(f"[physical-topology-ownership] PASS verify {args.output}")
        return 0
    if sha256_file(snapshot) != binary_sha256:
        raise RuntimeError("physical qualification snapshot differs from the public binary")
    anchors = validate_anchors(binary_sha256)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["NIMBUS_GPU_REQUIRED"] = "1"
    environment["PHYSICAL_QUALIFICATION_BINARY"] = snapshot.relative_to(ROOT).as_posix()
    log_parts: list[str] = []
    receipts: list[dict[str, object]] = []
    total_assertions = 0
    total_cases = 0
    for name, command, summaries, markers in SUITES:
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, check=False)
        elapsed = time.monotonic() - started
        log_parts.append(f"===== {name}: {' '.join(command)} =====\n{result.stdout}")
        if result.returncode != 0:
            write_atomic(args.log.resolve(), "\n".join(log_parts).encode())
            raise RuntimeError(f"physical topology suite failed: {name}")
        assertions, cases = validate_output(name, result.stdout, summaries, markers)
        total_assertions += assertions
        total_cases += cases
        receipts.append({"name": name, "command": command, "exitStatus": 0,
                         "assertions": assertions, "testCases": cases, "skipped": 0,
                         "fallbackUsed": False, "elapsedWallSeconds": elapsed})
        if sha256_file(binary) != binary_sha256 or sha256_file(snapshot) != binary_sha256:
            raise RuntimeError(f"binary continuity failed after physical topology suite: {name}")
        validate_anchors(binary_sha256)
    if (total_assertions, total_cases) != (220308, 17):
        raise RuntimeError(f"physical topology totals drifted: assertions={total_assertions}, cases={total_cases}")
    log_path = args.log.resolve()
    write_atomic(log_path, "\n".join(log_parts).encode())
    evidence = {
        "schema": "nimbus.physical-topology-ownership-evidence.v1", "version": 1, "status": "pass",
        "completedUtc": datetime.now(timezone.utc).isoformat(), "binary": record(binary),
        "physicalSnapshot": record(snapshot), "anchors": anchors,
        "sources": {"gate": record(Path(__file__)), "testsMake": record(ROOT / "mk/tests.mk"),
                    "workflowsMake": record(ROOT / "mk/workflows.mk")},
        "log": record(log_path), "receipts": receipts,
        "summary": {"suites": 12, "assertions": 220308, "testCases": 17, "skipped": 0,
                    "fallbackUsed": False, "physicalVulkan": True, "physicalMetal": True},
        "claimBoundary": (
            "Physical mapped-terrain, constant-motion, stationary nested, and synchronized relocation "
            "ownership on maintained Vulkan/Metal paths; not universal topology execution, a long campaign, "
            "or external review."
        ),
    }
    write_atomic(args.output.resolve(), (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode())
    verify_evidence(load(args.output.resolve(), "physical topology evidence"), binary_sha256)
    print("[physical-topology-ownership] PASS: 12 suites, 220308 assertions, 17 cases, "
          f"0 skipped, fallback prohibited; binary_sha256={binary_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

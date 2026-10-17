#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, label: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument(
        "--kernel-ownership", type=Path,
        default=ROOT / "reports/pre_beta/backend_kernel_ownership_v1.json",
    )
    parser.add_argument(
        "--tuple-observation", type=Path,
        default=ROOT / "reports/pre_beta/backend_tuple_operation_observation_v1.json",
    )
    parser.add_argument(
        "--vulkan-report", type=Path,
        default=ROOT / "reports/qualification/vulkan-qualification-current.json",
    )
    parser.add_argument(
        "--vulkan-log", type=Path,
        default=ROOT / "reports/qualification/vulkan-qualification-current.log",
    )
    parser.add_argument(
        "--metal-report", type=Path,
        default=ROOT / "reports/qualification/metal/bootstrap-current.json",
    )
    parser.add_argument(
        "--metal-log", type=Path,
        default=ROOT / "reports/qualification/metal/bootstrap-current.log",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/pre_beta/backend_kernel_dispatch_coverage_v1.json",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    binary_sha256 = sha256_file(binary)
    ownership_path = args.kernel_ownership.resolve()
    tuple_path = args.tuple_observation.resolve()
    vulkan_report_path = args.vulkan_report.resolve()
    vulkan_log_path = args.vulkan_log.resolve()
    metal_report_path = args.metal_report.resolve()
    metal_log_path = args.metal_log.resolve()
    ownership = load(ownership_path, "kernel ownership evidence")
    tuples = load(tuple_path, "tuple operation observation")
    vulkan = load(vulkan_report_path, "Vulkan qualification report")
    metal = load(metal_report_path, "Metal qualification report")

    if ownership.get("schema") != "nimbus.kernel-ownership-evidence.v1":
        raise RuntimeError("kernel ownership evidence schema drifted")
    if ownership.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("kernel ownership evidence does not bind the current binary")
    catalog = ownership.get("catalog")
    if not isinstance(catalog, dict) or catalog.get("summary") != {
        "operations": 50,
        "active": 47,
        "deferred": 0,
        "orphan": 3,
        "runtimeCovered": 19,
        "vulkanHardwareCovered": 43,
        "metalHardwareCovered": 47,
    }:
        raise RuntimeError("kernel ownership catalog summary drifted")
    kernel_records = catalog.get("records")
    if not isinstance(kernel_records, list):
        raise RuntimeError("kernel ownership records are missing")
    active_records = [
        item
        for item in kernel_records
        if isinstance(item, dict) and item.get("availability") == "active"
    ]
    if len(active_records) != 47:
        raise RuntimeError(f"expected 47 active operation records, got {len(active_records)}")
    active = {
        item.get("operationId"): item
        for item in active_records
        if isinstance(item.get("operationId"), str) and item["operationId"]
    }
    if len(active) != len(active_records):
        raise RuntimeError("active operation identifiers are missing or duplicated")

    if tuples.get("schema") != "nimbus.backend-tuple-operation-observation.v1":
        raise RuntimeError("tuple operation observation schema drifted")
    if tuples.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("tuple operation observation does not bind the current binary")
    tuple_summary = tuples.get("summary")
    if not isinstance(tuple_summary, dict) or {
        key: tuple_summary.get(key)
        for key in ("acceleratorFixtures", "vulkanFixtures", "metalFixtures")
    } != {
        "acceleratorFixtures": 16,
        "vulkanFixtures": 10,
        "metalFixtures": 6,
    }:
        raise RuntimeError("tuple operation observation is incomplete")
    observed_operation_ids = tuple_summary.get("observedOperationIds")
    if not isinstance(observed_operation_ids, list):
        raise RuntimeError("tuple operation identifiers are missing")
    maintained = set(observed_operation_ids)
    if (
        len(observed_operation_ids) != 27
        or len(maintained) != 27
        or not maintained <= set(active)
    ):
        raise RuntimeError("maintained tuple operation set drifted")
    focused = set(active) - maintained
    if len(focused) != 20:
        raise RuntimeError(f"focused kernel partition expected 20 operations, got {len(focused)}")

    if (
        vulkan.get("schema") != "nimbus.vulkan-qualification-evidence.v1"
        or vulkan.get("status") != "pass"
        or vulkan.get("artifacts", {}).get("binary_sha256") != binary_sha256
        or vulkan.get("combination_summary", {}).get("case_count") != 168
        or vulkan.get("combination_summary", {}).get("physical_pass_count") != 90
        or vulkan.get("combination_summary", {}).get("expected_preflight_rejection_count") != 78
    ):
        raise RuntimeError("Vulkan qualification evidence is incomplete or stale")
    if vulkan.get("artifacts", {}).get("qualification_log_sha256") != sha256_file(vulkan_log_path):
        raise RuntimeError("Vulkan qualification log identity drifted")

    metal_manifest = metal.get("manifest")
    metal_manifest_hash = (
        metal_manifest.get("canonical_fnv1a64")
        if isinstance(metal_manifest, dict)
        else None
    )
    if (
        metal.get("schema") != "nimbus.metal-bootstrap-qualification.v1"
        or metal.get("result") != "pass"
        or metal.get("qualification_scope") != "isolated_native_metal_compute"
        or metal.get("qualification_eligible") is not False
        or metal_manifest != {
            "active": 47,
            "canonical_fnv1a64": metal_manifest_hash,
            "deferred": 0,
            "kernels": 50,
            "orphan": 3,
            "ready": 47,
        }
        or not isinstance(metal_manifest_hash, str)
        or re.fullmatch(r"[0-9a-f]{16}", metal_manifest_hash) is None
        or metal.get("tests") != {"assertions": 75655, "test_cases": 43}
        or metal.get("test_output_sha256") != sha256_file(metal_log_path)
    ):
        raise RuntimeError("Metal qualification evidence is incomplete or stale")

    coverage: list[dict[str, object]] = []
    for operation_id, item in sorted(active.items()):
        vulkan_contract = item.get("vulkan")
        metal_contract = item.get("metal")
        qualification = item.get("qualification")
        if not all(isinstance(value, dict) for value in (vulkan_contract, metal_contract, qualification)):
            raise RuntimeError(f"malformed kernel ownership record: {operation_id}")
        vulkan_test = vulkan_contract.get("hardwareTest")
        runtime_sample = qualification.get("runtimeSample")
        if vulkan_test is None and runtime_sample is None:
            raise RuntimeError(f"operation lacks a Vulkan coverage route: {operation_id}")
        if metal_contract.get("hardwareTest") != "tests/metal/test_metal_compute_backend.cpp":
            raise RuntimeError(f"operation lacks the canonical Metal suite: {operation_id}")
        coverage.append({
            "operationId": operation_id,
            "coverageClass": "maintained_tuple" if operation_id in maintained else "focused_backend_suite",
            "vulkanRoute": {
                "hardwareTest": vulkan_test,
                "runtimeSample": runtime_sample,
            },
            "metalRoute": {
                "hardwareTest": metal_contract["hardwareTest"],
                "sourceFnv1a64": metal_contract["sourceFnv1a64"],
            },
        })

    result = {
        "schema": "nimbus.backend-kernel-dispatch-coverage.v1",
        "version": 1,
        "status": "pass",
        "binary": record(binary),
        "inputs": {
            "kernelOwnership": record(ownership_path),
            "tupleObservation": record(tuple_path),
            "vulkanQualification": record(vulkan_report_path),
            "vulkanLog": record(vulkan_log_path),
            "metalQualification": record(metal_report_path),
            "metalLog": record(metal_log_path),
        },
        "summary": {
            "activeOperations": 47,
            "maintainedTupleOperations": 27,
            "focusedBackendSuiteOperations": 20,
            "unclassifiedOperations": 0,
            "vulkanCombinationCases": 168,
            "vulkanPhysicalCases": 90,
            "metalAssertions": 75655,
            "metalTestCases": 43,
        },
        "coverage": coverage,
        "claimBoundary": (
            "Physical maintained-tuple and focused backend-operation coverage; "
            "not universal generated-tuple execution, a long campaign, or external review."
        ),
    }
    write_atomic(args.output.resolve(), result)
    print(
        "[backend-kernel-dispatch-coverage] PASS: 47/47 active operations classified; "
        "27 maintained-tuple, 20 focused-suite, 0 unclassified; "
        f"binary_sha256={binary_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

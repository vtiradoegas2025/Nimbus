#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
DEFAULT_TOOL = ROOT / "bin/nimbus_kernel_ownership"
ORPHAN_OPERATIONS = {
    "diffusion.cartesian",
    "archive.legacy_thompson_lin_pointwise",
    "archive.legacy_thompson_lin_sedimentation",
}
VULKAN_ONLY_OPERATIONS = {"dynamics.acoustic_retained_flux_fp64"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv1a64(path: Path) -> str:
    value = 14695981039346656037
    for byte in path.read_bytes():
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def require_file(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} path is missing")
    path = ROOT / value
    if not path.is_file():
        raise RuntimeError(f"{label} path does not exist: {value}")
    return path


def validate(catalog: dict[str, object]) -> None:
    if catalog.get("schema") != "nimbus.kernel-ownership-catalog.v1" or catalog.get("version") != 1:
        raise RuntimeError("kernel ownership catalog identity drifted")
    records = catalog.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("kernel ownership catalog has no records")
    if any(not isinstance(record, dict) or not isinstance(record.get("vulkan"), dict) for record in records):
        raise RuntimeError("kernel ownership artifact records are malformed")
    artifact_manifest = json.loads((ROOT / "vulkan/shaders/manifest.json").read_text())
    prefix = "vulkan/shaders/compute/"
    expected_artifacts = {entry["artifact"][len(prefix):] for entry in artifact_manifest["shaders"]
                          if entry["artifact"].startswith(prefix)}
    actual_artifacts = {record.get("vulkan", {}).get("artifact") for record in records if isinstance(record, dict)}
    if len(records) != len(expected_artifacts) or actual_artifacts != expected_artifacts:
        raise RuntimeError("kernel ownership records differ from the compute artifact inventory")

    operations: set[str] = set()
    kernels: set[str] = set()
    observed_orphans: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("index") != index:
            raise RuntimeError("kernel ownership indices are not contiguous")
        operation = record.get("operationId")
        kernel = record.get("kernelName")
        if not isinstance(operation, str) or not operation or operation in operations:
            raise RuntimeError(f"duplicate or empty operation identity at {index}")
        if not isinstance(kernel, str) or not kernel or kernel in kernels:
            raise RuntimeError(f"duplicate or empty kernel identity at {index}")
        operations.add(operation)
        kernels.add(kernel)
        owners = record.get("owners")
        vulkan = record.get("vulkan")
        metal = record.get("metal")
        qualification = record.get("qualification")
        if not all(isinstance(value, dict) for value in (owners, vulkan, metal, qualification)):
            raise RuntimeError(f"kernel ownership record {index} is malformed")
        if owners.get("cpu") != "reference" or not isinstance(record.get("cpuReference"), str):
            raise RuntimeError(f"kernel ownership record {index} lacks a CPU reference")

        artifact = require_file(
            f"vulkan/shaders/compute/{vulkan.get('artifact')}",
            f"kernel {operation} Vulkan artifact",
        )
        if vulkan.get("artifactFnv1a64") != fnv1a64(artifact):
            raise RuntimeError(f"kernel {operation} Vulkan artifact hash drifted")

        availability = record.get("availability")
        if availability == "active":
            vulkan_only = operation in VULKAN_ONLY_OPERATIONS
            if owners != {"cpu": "reference", "vulkan": "device", "metal": "unavailable" if vulkan_only else "device"}:
                raise RuntimeError(f"active kernel {operation} has incomplete backend ownership")
            if vulkan_only:
                if any(metal.get(key) is not None for key in ("library", "sourceFnv1a64", "preciseMathRequired", "hardwareTest")):
                    raise RuntimeError(f"Vulkan-only kernel {operation} claims a Metal implementation")
            else:
                if not isinstance(metal.get("library"), str) or not isinstance(metal.get("sourceFnv1a64"), str):
                    raise RuntimeError(f"active kernel {operation} lacks Metal source identity")
                if len(metal["sourceFnv1a64"]) != 16:
                    raise RuntimeError(f"active kernel {operation} has malformed Metal source hash")
                if not isinstance(metal.get("preciseMathRequired"), bool):
                    raise RuntimeError(f"active kernel {operation} omits its Metal compiler math policy")
            if operation in ("advection.conservative_mass_flux_cartesian_cgrid",
                             "advection.conservative_update_cartesian_cgrid",
                             "terrain.mapped_scalar_limiter",
                             "terrain.mapped_scalar_update",
                             "chaos.earth_control_evolution") and not metal["preciseMathRequired"]:
                raise RuntimeError(f"kernel {operation} requires precise Metal math")
            if not isinstance(qualification.get("outputs"), str) or not qualification["outputs"]:
                raise RuntimeError(f"active kernel {operation} lacks checked outputs")
            if qualification.get("requiredHardware") != ("vulkan" if vulkan_only else "vulkan_and_metal"):
                raise RuntimeError(f"active kernel {operation} does not require its implemented physical backends")
            if qualification.get("waiver") is not None:
                raise RuntimeError(f"active kernel {operation} unexpectedly has a waiver")
            runtime_sample = qualification.get("runtimeSample")
            if runtime_sample is not None and (not isinstance(runtime_sample, str) or not runtime_sample):
                raise RuntimeError(f"kernel {operation} has an invalid runtime replay owner")
            vulkan_test = vulkan.get("hardwareTest")
            if runtime_sample is None and vulkan_test is None:
                raise RuntimeError(f"active kernel {operation} lacks Vulkan runtime/hardware coverage")
            if vulkan_test is not None:
                require_file(vulkan_test, f"kernel {operation} Vulkan hardware test")
            if not vulkan_only:
                require_file(metal.get("hardwareTest"), f"kernel {operation} Metal hardware test")
            elif vulkan_test is None:
                raise RuntimeError(f"Vulkan-only kernel {operation} lacks a physical Vulkan fixture")
        elif availability == "orphan":
            observed_orphans.add(operation)
            if owners != {"cpu": "reference", "vulkan": "unavailable", "metal": "unavailable"}:
                raise RuntimeError(f"orphan kernel {operation} has an executable backend owner")
            if metal.get("library") is not None or metal.get("sourceFnv1a64") is not None:
                raise RuntimeError(f"orphan kernel {operation} retains a Metal implementation")
            if metal.get("preciseMathRequired") is not None:
                raise RuntimeError(f"orphan kernel {operation} retains an executable Metal math policy")
            if qualification.get("requiredHardware") != "none" or not isinstance(qualification.get("waiver"), str):
                raise RuntimeError(f"orphan kernel {operation} lacks an explicit waiver")
        else:
            raise RuntimeError(f"unexpected kernel availability {availability!r} for {operation}")
    if observed_orphans != ORPHAN_OPERATIONS:
        raise RuntimeError(f"orphan kernel set drifted: {observed_orphans}")
    if not VULKAN_ONLY_OPERATIONS.issubset(operations):
        raise RuntimeError("Vulkan-only kernel inventory is incomplete")
    summary = {
        "operations": len(records),
        "active": sum(record["availability"] == "active" for record in records),
        "deferred": 0,
        "orphan": len(observed_orphans),
        "runtimeCovered": sum(record["qualification"]["runtimeSample"] is not None for record in records),
        "vulkanHardwareCovered": sum(record["vulkan"]["hardwareTest"] is not None for record in records),
        "metalHardwareCovered": sum(record["metal"]["hardwareTest"] is not None for record in records),
    }
    if catalog.get("summary") != summary:
        raise RuntimeError(f"kernel ownership summary differs from validated records: {catalog.get('summary')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    tool = args.tool.resolve()
    for path, label in ((binary, "simulator"), (tool, "kernel ownership tool")):
        if not path.is_file():
            raise RuntimeError(f"missing {label}: {path}")
    binary_sha256 = sha256_file(binary)
    tool_sha256 = sha256_file(tool)
    result = subprocess.run(
        [str(tool)], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "kernel ownership tool failed:\n" +
            result.stderr.decode("utf-8", errors="replace")
        )
    catalog = json.loads(result.stdout)
    validate(catalog)
    if sha256_file(binary) != binary_sha256 or sha256_file(tool) != tool_sha256:
        raise RuntimeError("binary identity changed while kernel ownership evidence was generated")

    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        evidence = {
            "schema": "nimbus.kernel-ownership-evidence.v1",
            "version": 1,
            "binary": file_record(binary),
            "tool": file_record(tool),
            "catalogSha256": hashlib.sha256(result.stdout).hexdigest(),
            "sources": [
                file_record(ROOT / "src/compute/kernel_manifest.cpp"),
                file_record(ROOT / "tests/compute/support/parity_registry.cpp"),
                file_record(ROOT / "metal/src/compute/internal/metal_source_catalog.hpp"),
            ],
            "catalog": catalog,
            "claimBoundary": (
                "Backend ownership and qualification-contract inventory only; "
                "physical execution must be established by a current hardware gate receipt."
            ),
        }
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)

    print(
        f"[kernel-ownership-catalog] PASS: {catalog['summary']['operations']} operations; "
        f"{catalog['summary']['active']} active with CPU references, "
        "Vulkan artifacts, applicable Metal source identities, and backend qualification anchors; "
        f"{catalog['summary']['orphan']} orphan; binary_sha256={binary_sha256}; tool_sha256={tool_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

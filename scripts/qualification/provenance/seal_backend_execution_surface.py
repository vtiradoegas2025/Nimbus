#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.catalog_inventory import maintained_cases, inventory_counts, restart_class_scripts
QUALIFICATION = ROOT / "tests/fixtures/configs/qualification"
PUBLIC_REPRESENTATIVES = ROOT / "configs/1000meterResolution"
CLASS_SCRIPTS = restart_class_scripts()


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


def verify_record(value: object, label: str) -> Path:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise RuntimeError(f"{label} file record is malformed")
    relative = Path(value["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{label} file record escapes the repository")
    path = ROOT / relative
    if record(path) != value:
        raise RuntimeError(f"{label} file record drifted")
    return path


def load(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def configured_backend(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("backend:"):
            return stripped.split(":", 1)[1].strip().lower()
    return "cpu"


def moving_duration(path: Path) -> int:
    return 30 if "preset: moving_reference_v1" in path.read_text(encoding="utf-8") else 1


def expected_cpu_cases() -> list[dict[str, object]]:
    qualified, representatives = maintained_cases()
    cases = list(enumerate(qualified + representatives))
    return [
        {
            "index": index,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "backend": "cpu",
            "durationSeconds": moving_duration(path),
        }
        for index, path in cases
        if configured_backend(path) == "cpu"
    ]


def validate_bounded(value: dict[str, object], binary_sha256: str) -> None:
    if value.get("schema") != "nimbus.catalog-bounded-partition-evidence.v1" or value.get("version") != 1:
        raise RuntimeError("bounded CPU evidence schema drifted")
    if value.get("status") != "pass" or value.get("partition") != "cpu":
        raise RuntimeError("bounded CPU evidence is not a passing CPU partition")
    if value.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("bounded CPU evidence does not bind the current binary")
    verify_record(value.get("binary"), "bounded.binary")
    verify_record(value.get("runner"), "bounded.runner")
    if value.get("runContract") != {
        "cpuWorkers": 6,
        "threadsPerCase": 1,
        "ordinaryDurationSeconds": 1,
        "movingReferenceDurationSeconds": 30,
        "ordinaryPublicationRequired": True,
    }:
        raise RuntimeError("bounded CPU run contract drifted")
    counts = inventory_counts()
    if value.get("summary") != {**counts, "completed": counts["cpuCases"]}:
        raise RuntimeError("bounded CPU evidence summary drifted")
    if value.get("cases") != expected_cpu_cases():
        raise RuntimeError("bounded CPU case inventory drifted")
    if not isinstance(value.get("elapsedWallSeconds"), (int, float)) or value["elapsedWallSeconds"] <= 0:
        raise RuntimeError("bounded CPU elapsed time is invalid")


def validate_restart(value: dict[str, object], binary_sha256: str) -> None:
    if value.get("schema") != "nimbus.catalog-restart-threading-evidence.v1" or value.get("version") != 1:
        raise RuntimeError("restart/threading evidence schema drifted")
    if value.get("status") != "pass" or value.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("restart/threading evidence does not bind the passing current binary")
    verify_record(value.get("binary"), "restart.binary")
    verify_record(value.get("runner"), "restart.runner")
    log_path = verify_record(value.get("log"), "restart.log")
    if value.get("summary") != {"classes": len(CLASS_SCRIPTS), "passed": len(CLASS_SCRIPTS), "skipped": 0}:
        raise RuntimeError("restart/threading evidence summary drifted")
    classes = value.get("classes")
    if not isinstance(classes, list) or len(classes) != len(CLASS_SCRIPTS):
        raise RuntimeError("restart/threading class receipts are incomplete")
    seen: list[str] = []
    for entry, expected in zip(classes, CLASS_SCRIPTS, strict=True):
        if not isinstance(entry, dict):
            raise RuntimeError("restart/threading class receipt is malformed")
        script = verify_record(entry.get("script"), "restart.class.script")
        seen.append(script.relative_to(ROOT).as_posix())
        if (
            entry.get("exitStatus") != 0
            or entry.get("skipped") != 0
            or not isinstance(entry.get("elapsedWallSeconds"), (int, float))
            or entry["elapsedWallSeconds"] <= 0
        ):
            raise RuntimeError(f"restart/threading class receipt drifted: {expected}")
    if tuple(seen) != CLASS_SCRIPTS:
        raise RuntimeError("restart/threading class order or identity drifted")
    log = log_path.read_text(encoding="utf-8")
    if sum("] PASS" in line for line in log.splitlines()) != len(CLASS_SCRIPTS) or "FAILED" in log:
        raise RuntimeError("restart/threading log markers drifted")
    if not isinstance(value.get("elapsedWallSeconds"), (int, float)) or value["elapsedWallSeconds"] <= 0:
        raise RuntimeError("restart/threading elapsed time is invalid")


def input_paths() -> dict[str, Path]:
    return {
        "boundedCpu": ROOT / "reports/pre_beta/catalog_bounded_cpu_v1.json",
        "restartThreading": ROOT / "reports/pre_beta/catalog_restart_threading_v1.json",
        "acceleratorObservation": ROOT / "reports/pre_beta/backend_tuple_operation_observation_v1.json",
        "kernelDispatch": ROOT / "reports/pre_beta/backend_kernel_dispatch_coverage_v1.json",
        "physicalTopology": ROOT / "reports/pre_beta/physical_topology_ownership_v1.json",
        "backendIo": ROOT / "reports/pre_beta/backend_io_ownership_v1.json",
        "reachable": ROOT / "reports/pre_beta/reachable_set_v1.json",
        "generatedBackendTuples": ROOT / "reports/pre_beta/generated_backend_tuple_evidence_v1.json",
        "nativeAdmission": ROOT / "reports/pre_beta/native_admission_evidence_v1.json",
        "nativeFullPreflight": ROOT / "reports/pre_beta/native_full_preflight_evidence_v1.json",
        "nativeYamlReload": ROOT / "reports/pre_beta/native_yaml_reload_v1.json",
    }


def validate_inputs(binary_sha256: str) -> dict[str, dict[str, object]]:
    paths = input_paths()
    values = {name: load(path, name) for name, path in paths.items()}
    validate_bounded(values["boundedCpu"], binary_sha256)
    validate_restart(values["restartThreading"], binary_sha256)
    for name in ("kernelDispatch", "physicalTopology", "backendIo", "reachable",
                 "generatedBackendTuples", "nativeAdmission", "nativeFullPreflight", "nativeYamlReload"):
        if values[name].get("binary", {}).get("sha256") != binary_sha256:
            raise RuntimeError(f"{name} does not bind the current binary")
    if values["generatedBackendTuples"].get("summary") != {
        "evaluated": 1697760,
        "accepted": 365064,
        "ownershipSignatures": 628,
        "duplicateTuples": 0,
    }:
        raise RuntimeError("generated backend tuple evidence summary drifted")
    if values["nativeAdmission"].get("summary") != {
        "categoricalAccepted": 365064,
        "nativeScienceAdmitted": 195432,
        "nativeScienceRejected": 169632,
        "rejectionReasons": 11,
        "duplicateAdmittedTuples": 0,
    }:
        raise RuntimeError("native admission evidence summary drifted")
    if values["nativeFullPreflight"].get("summary") != {
        "categoricalAccepted": 365064,
        "nativeScienceAdmitted": 195432,
        "nativeFullPreflightAdmitted": 195432,
        "nativeFullPreflightRejected": 0,
        "distinctSoundingMaterializations": 2,
        "duplicateAdmittedTuples": 0,
    }:
        raise RuntimeError("native full-preflight evidence summary drifted")
    if values["nativeYamlReload"].get("summary") != {
        "inputTuples": 195432,
        "ownershipSignatures": 488,
        "axisRequirements": 196,
        "representatives": 561,
        "passed": 561,
        "failed": 0,
        "skipped": 0,
        "representativeSha256": "8fe6475741813c05c8c53cb3e537c620d6465cd16b287031bca126bb79dc3c58",
        "partitions": {
            "c_grid.cpu.false": 83,
            "c_grid.cpu.true": 45,
            "c_grid.gpu.false": 156,
            "c_grid.gpu.true": 46,
            "collocated.cpu.false": 79,
            "collocated.gpu.false": 152,
        },
    }:
        raise RuntimeError("native YAML reload evidence summary drifted")
    observation = values["acceleratorObservation"]
    if observation.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("accelerator observation does not bind the current binary")
    if observation.get("summary", {}).get("acceleratorFixtures") != 16:
        raise RuntimeError("accelerator observation is incomplete")
    kernel_input = values["kernelDispatch"].get("inputs", {}).get("tupleObservation")
    if kernel_input != record(paths["acceleratorObservation"]):
        raise RuntimeError("kernel dispatch seal does not bind the accelerator observation")
    if values["physicalTopology"].get("status") != "pass" or values["backendIo"].get("status") != "pass":
        raise RuntimeError("backend execution surface anchor is not passing")
    return {name: record(path) for name, path in paths.items()}


def write_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_evidence(value: dict[str, object], binary_sha256: str) -> None:
    if value.get("schema") != "nimbus.backend-execution-surface-evidence.v1" or value.get("version") != 1:
        raise RuntimeError("backend execution surface schema drifted")
    if value.get("status") != "pass" or value.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("backend execution surface does not bind the passing current binary")
    verify_record(value.get("binary"), "surface.binary")
    verify_record(value.get("sealer"), "surface.sealer")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(input_paths()):
        raise RuntimeError("backend execution surface inputs are incomplete")
    for name, file_record in inputs.items():
        verify_record(file_record, f"surface.inputs.{name}")
    if value.get("summary") != {
        "catalogCases": inventory_counts()["catalogCases"],
        "cpuCases": inventory_counts()["cpuCases"],
        "acceleratorCases": inventory_counts()["acceleratorCases"],
        "restartThreadingClasses": len(CLASS_SCRIPTS),
        "generatedAcceptedTuples": 365064,
        "ownershipSignatures": 628,
        "nativeScienceAdmittedTuples": 195432,
        "nativeFullPreflightAdmittedTuples": 195432,
        "nativeAdmittedOwnershipSignatures": 488,
        "nativeYamlReloadRepresentatives": 561,
        "skipped": 0,
    }:
        raise RuntimeError("backend execution surface summary drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/pre_beta/backend_execution_surface_v1.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    binary_sha256 = sha256_file(binary)
    if args.verify:
        verify_evidence(load(args.output.resolve(), "backend execution surface"), binary_sha256)
        validate_inputs(binary_sha256)
        print(f"[backend-execution-surface] PASS verify {args.output}")
        return 0
    evidence = {
        "schema": "nimbus.backend-execution-surface-evidence.v1",
        "version": 1,
        "status": "pass",
        "binary": record(binary),
        "sealer": record(Path(__file__)),
        "inputs": validate_inputs(binary_sha256),
        "summary": {
            "catalogCases": inventory_counts()["catalogCases"],
            "cpuCases": inventory_counts()["cpuCases"],
            "acceleratorCases": inventory_counts()["acceleratorCases"],
            "restartThreadingClasses": len(CLASS_SCRIPTS),
            "generatedAcceptedTuples": 365064,
            "ownershipSignatures": 628,
            "nativeScienceAdmittedTuples": 195432,
            "nativeFullPreflightAdmittedTuples": 195432,
            "nativeAdmittedOwnershipSignatures": 488,
            "nativeYamlReloadRepresentatives": 561,
            "skipped": 0,
        },
        "claimBoundary": (
            "Current-binary maintained backend execution, restart/threading, topology, native I/O, and "
            "allocation-free preflight of every native-science-admitted canonical tuple, plus public strict-YAML "
            "reload for every admitted ownership signature and categorical-axis witness; not universal generated-"
            "tuple file loading, allocation/timestep execution, frontend admission parity, a long campaign, or "
            "external review."
        ),
    }
    write_atomic(args.output.resolve(), evidence)
    verify_evidence(load(args.output.resolve(), "backend execution surface"), binary_sha256)
    print(
        f"[backend-execution-surface] PASS: {inventory_counts()['catalogCases']} maintained bounded cases and "
        f"{len(CLASS_SCRIPTS)} restart/threading classes; binary_sha256={binary_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

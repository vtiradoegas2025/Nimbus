#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
from scripts.qualification.catalog_admission import (
    load_baseline, baseline_records, verify_baseline_records, verify_provenance_path,
    pipeline_bytes, native_stream, stream_identity,
)
BASELINE = load_baseline()
EXPECTED_SUMMARY = BASELINE["admission"]["summary"]
EXPECTED_PARTITIONS = BASELINE["admission"]["partitions"]
EXPECTED_INPUT_SHA256 = BASELINE["admission"]["inputTupleSha256"]
EXPECTED_ADMITTED_SHA256 = BASELINE["admission"]["admittedTupleSha256"]
EXPECTED_REJECTIONS_SHA256 = hashlib.sha256(json.dumps(BASELINE["admission"]["rejections"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def verify_record(value: object, label: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise RuntimeError(f"{label} file record is malformed")
    verify_provenance_path(value, label, Path(__file__))
    relative = Path(value["path"])
    if relative.is_absolute() or ".." in relative.parts or record(ROOT / relative) != value:
        raise RuntimeError(f"{label} file record drifted")


def pipeline(generated_tool: Path, native_tool: Path, native_args: tuple[str, ...] = ()) -> bytes:
    return pipeline_bytes(generated_tool, native_tool, native_args)

def rejection_sha256(rejections: object) -> str:
    return hashlib.sha256(
        json.dumps(rejections, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_catalog(value: dict[str, object]) -> None:
    if value.get("schema") != "nimbus.native-admission-batch.v1" or value.get("version") != 1:
        raise RuntimeError("native admission batch schema drifted")
    if value.get("summary") != EXPECTED_SUMMARY:
        raise RuntimeError("native admission summary drifted")
    if value.get("inputTupleSha256") != EXPECTED_INPUT_SHA256:
        raise RuntimeError("native admission input identity drifted")
    if value.get("admittedTupleSha256") != EXPECTED_ADMITTED_SHA256:
        raise RuntimeError("native admitted tuple identity drifted")
    if value.get("partitions") != EXPECTED_PARTITIONS:
        raise RuntimeError("native admission partitions drifted")
    rejections = value.get("rejections")
    if not isinstance(rejections, list) or len(rejections) != EXPECTED_SUMMARY["rejectionReasons"]:
        raise RuntimeError("native admission rejection catalog is incomplete")
    if sum(item.get("count", 0) for item in rejections if isinstance(item, dict)) != EXPECTED_SUMMARY["rejected"]:
        raise RuntimeError("native admission rejection counts drifted")
    if rejection_sha256(rejections) != EXPECTED_REJECTIONS_SHA256:
        raise RuntimeError("native admission rejection identity drifted")


def stream_admitted(generated_tool: Path, native_tool: Path) -> tuple[str, dict[str, int]]:
    with native_stream(generated_tool, native_tool, ("--admitted",)) as stream:
        return stream_identity(stream, EXPECTED_SUMMARY["admitted"],
                               EXPECTED_ADMITTED_SHA256, EXPECTED_PARTITIONS)

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
    if value.get("schema") != "nimbus.native-admission-evidence.v1" or value.get("version") != 1:
        raise RuntimeError("native admission evidence schema drifted")
    if value.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("native admission evidence does not bind the current binary")
    for key in ("binary", "generatedTool", "nativeTool", "generatedSource", "nativeSource",
                "runtimeAdmissionSource", "canonicalBase", "generatedEvidence", "qualifier"):
        verify_record(value.get(key), key)
    from scripts.tests.catalog import test_generated_backend_tuples as upstream_guard
    upstream_path = ROOT / value["generatedEvidence"]["path"]
    upstream_guard.verify_evidence(json.loads(upstream_path.read_text()), binary_sha256)
    verify_baseline_records(value, record)
    validate_catalog(value.get("catalog"))
    if value.get("summary") != {
        "categoricalAccepted": EXPECTED_SUMMARY["input"],
        "nativeScienceAdmitted": BASELINE["admission"]["summary"]["admitted"],
        "nativeScienceRejected": EXPECTED_SUMMARY["rejected"],
        "rejectionReasons": EXPECTED_SUMMARY["rejectionReasons"],
        "duplicateAdmittedTuples": 0,
    }:
        raise RuntimeError("native admission evidence summary drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--generated-tool", type=Path, default=ROOT / "bin/nimbus_generated_backend_tuples")
    parser.add_argument("--native-tool", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    generated_tool = args.generated_tool.resolve(strict=True)
    native_tool = args.native_tool.resolve(strict=True)
    binary_sha256 = sha256_file(binary)
    if args.verify:
        if args.output is None:
            raise RuntimeError("--verify requires --output")
        evidence = json.loads(args.output.resolve().read_text(encoding="utf-8"))
        verify_evidence(evidence, binary_sha256)
        print(f"[native-admission-batch] PASS verify {args.output}")
        return 0
    first = pipeline(generated_tool, native_tool)
    second = pipeline(generated_tool, native_tool)
    if first != second:
        raise RuntimeError("native admission batch output is not deterministic")
    catalog = json.loads(first)
    validate_catalog(catalog)
    admitted_sha256, partitions = stream_admitted(generated_tool, native_tool)
    if admitted_sha256 != catalog["admittedTupleSha256"] or partitions != catalog["partitions"]:
        raise RuntimeError("native admitted tuple stream and compact catalog disagree")
    if sha256_file(binary) != binary_sha256:
        raise RuntimeError("simulator identity changed during qualification")
    if args.output is not None:
        generated_evidence = ROOT / "reports/pre_beta/generated_backend_tuple_evidence_v1.json"
        upstream = json.loads(generated_evidence.read_text(encoding="utf-8"))
        from scripts.tests.catalog import test_generated_backend_tuples as upstream_guard
        upstream_guard.verify_evidence(upstream, binary_sha256)
        if upstream.get("catalog", {}).get("acceptedTupleSha256") != EXPECTED_INPUT_SHA256:
            raise RuntimeError("upstream generated tuple evidence identity drifted")
        evidence = {
            **baseline_records(record),
            "qualifier": record(Path(__file__)),
            "schema": "nimbus.native-admission-evidence.v1",
            "version": 1,
            "binary": record(binary),
            "generatedTool": record(generated_tool),
            "nativeTool": record(native_tool),
            "generatedSource": record(ROOT / "tests/compute/generated_backend_tuple_catalog_main.cpp"),
            "nativeSource": record(ROOT / "tests/compute/native_admission_batch_main.cpp"),
            "runtimeAdmissionSource": record(ROOT / "src/core/runtime/nimbus_sim.cpp"),
            "canonicalBase": record(ROOT / "tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml"),
            "generatedEvidence": record(generated_evidence),
            "toolOutputSha256": hashlib.sha256(first).hexdigest(),
            "catalog": catalog,
            "summary": {
                "categoricalAccepted": EXPECTED_SUMMARY["input"],
                "nativeScienceAdmitted": BASELINE["admission"]["summary"]["admitted"],
                "nativeScienceRejected": EXPECTED_SUMMARY["rejected"],
                "rejectionReasons": EXPECTED_SUMMARY["rejectionReasons"],
                "duplicateAdmittedTuples": 0,
            },
            "claimBoundary": (
                "Canonical ConfigV1 materialization through the exact native science-admission function; "
                "does not include per-tuple YAML loading, sounding construction, lifetime preflight, "
                "field allocation, timestep execution, a long campaign, or external review."
            ),
        }
        write_atomic(args.output.resolve(), evidence)
    print(
        f"[native-admission-batch] PASS: {EXPECTED_SUMMARY['admitted']} admitted / {EXPECTED_SUMMARY['input']} categorical accepted; "
        f"{EXPECTED_SUMMARY['rejected']} rejected across {EXPECTED_SUMMARY['rejectionReasons']} reasons; admitted_sha256={admitted_sha256}; "
        f"binary_sha256={binary_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

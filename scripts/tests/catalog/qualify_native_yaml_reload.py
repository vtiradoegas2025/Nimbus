#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
from scripts.qualification.catalog_admission import (
    load_baseline, baseline_records, verify_baseline_records, verify_provenance_path,
    sounding_records, verify_sounding_records, native_stream, select_witnesses, parse_tuple,
)
BASELINE = load_baseline()
EXPECTED_SELECTION = BASELINE["yamlReload"]["summary"]
EXPECTED_INPUT_TUPLES = EXPECTED_SELECTION["inputTuples"]
EXPECTED_OWNERSHIP_SIGNATURES = EXPECTED_SELECTION["ownershipSignatures"]
EXPECTED_AXIS_REQUIREMENTS = EXPECTED_SELECTION["axisRequirements"]
EXPECTED_REPRESENTATIVES = EXPECTED_SELECTION["representatives"]
EXPECTED_SIGNED_STREAM_SHA256 = BASELINE["yamlReload"]["signedStreamSha256"]
EXPECTED_REPRESENTATIVE_SHA256 = EXPECTED_SELECTION["representativeSha256"]
EXPECTED_PARTITIONS = EXPECTED_SELECTION["partitions"]


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


def verify_record(value: object, label: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise RuntimeError(f"{label} file record is malformed")
    verify_provenance_path(value, label, Path(__file__))
    relative = Path(value["path"])
    if relative.is_absolute() or ".." in relative.parts or record(ROOT / relative) != value:
        raise RuntimeError(f"{label} file record drifted")


def select_representatives(raw):
    selected, summary, digest = select_witnesses(raw)
    if digest != EXPECTED_SIGNED_STREAM_SHA256:
        raise RuntimeError("signed admitted tuple stream identity drifted")
    if summary != EXPECTED_SELECTION:
        raise RuntimeError(f"native YAML representative identity drifted: {summary}")
    return selected, summary


def validate_representative(
    binary: Path,
    native_tool: Path,
    config_path: Path,
    signature: str,
    tuple_text: str,
) -> dict[str, object]:
    yaml_result = subprocess.run(
        [str(native_tool), "--yaml"], cwd=ROOT, input=(tuple_text + "\n").encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if yaml_result.returncode != 0:
        raise RuntimeError(
            f"native YAML serialization failed for {signature}: "
            + yaml_result.stderr.decode(errors="replace")
        )
    yaml_bytes = yaml_result.stdout
    if b"\nschema: nimbus.v1\n" not in yaml_bytes:
        raise RuntimeError(f"serialized representative omitted strict schema: {signature}")
    if b"allow_fallback: false" not in yaml_bytes:
        raise RuntimeError(f"serialized representative permits backend fallback: {signature}")
    config_path.write_bytes(yaml_bytes)
    result = subprocess.run(
        [str(binary), "--config", str(config_path), "--validate-only"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    output = result.stdout
    if result.returncode != 0 or output.count(b"[VALIDATE] PASS:") != 1:
        raise RuntimeError(
            f"public strict YAML reload failed for {signature}:\n" + output.decode(errors="replace")
        )
    if b"[VALIDATE WARNING]" in output:
        raise RuntimeError(
            f"public strict YAML reload warned for {signature}:\n"
            + output.decode(errors="replace")
        )
    return {
        "ownershipSignature": signature,
        "tuple": tuple_text,
        "yamlSha256": hashlib.sha256(yaml_bytes).hexdigest(),
        "validationOutputSha256": hashlib.sha256(output).hexdigest(),
        "status": "pass",
    }


def aggregate_sha256(representatives: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in representatives:
        digest.update(
            (
                f'{entry["ownershipSignature"]}\t{entry["tuple"]}\t{entry["yamlSha256"]}\t'
                f'{entry["validationOutputSha256"]}\n'
            ).encode()
        )
    return digest.hexdigest()


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
    if value.get("schema") != "nimbus.native-yaml-reload-evidence.v1" or value.get("version") != 1:
        raise RuntimeError("native YAML reload evidence schema drifted")
    if value.get("status") != "pass" or value.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("native YAML reload evidence does not bind the passing current binary")
    for key in (
        "binary", "generatedTool", "nativeTool", "generatedSource", "nativeSource",
        "runtimeAdmissionSource", "qualifier", "generatedEvidence", "nativeFullPreflightEvidence",
    ):
        verify_record(value.get(key), key)
    from scripts.tests.catalog import test_native_full_preflight as upstream_guard
    upstream_path = ROOT / value["nativeFullPreflightEvidence"]["path"]
    upstream_guard.verify_evidence(json.loads(upstream_path.read_text()), binary_sha256)
    verify_baseline_records(value, record)
    verify_sounding_records(value, record)
    expected_summary = {
        "inputTuples": EXPECTED_INPUT_TUPLES,
        "ownershipSignatures": EXPECTED_OWNERSHIP_SIGNATURES,
        "axisRequirements": EXPECTED_AXIS_REQUIREMENTS,
        "representatives": EXPECTED_REPRESENTATIVES,
        "passed": EXPECTED_REPRESENTATIVES,
        "failed": 0,
        "skipped": 0,
        "representativeSha256": EXPECTED_REPRESENTATIVE_SHA256,
        "partitions": EXPECTED_PARTITIONS,
    }
    if value.get("summary") != expected_summary:
        raise RuntimeError("native YAML reload evidence summary drifted")
    representatives = value.get("representativeResults")
    if not isinstance(representatives, list) or len(representatives) != EXPECTED_REPRESENTATIVES:
        raise RuntimeError("native YAML reload representative results are incomplete")
    if any(entry.get("status") != "pass" for entry in representatives if isinstance(entry, dict)):
        raise RuntimeError("native YAML reload representative result is not passing")
    signatures = {entry.get("ownershipSignature") for entry in representatives if isinstance(entry, dict)}
    tuples = {entry.get("tuple") for entry in representatives if isinstance(entry, dict)}
    if len(signatures) != EXPECTED_OWNERSHIP_SIGNATURES or len(tuples) != EXPECTED_REPRESENTATIVES:
        raise RuntimeError("native YAML reload representative coverage drifted")
    if value.get("validationAggregateSha256") != aggregate_sha256(representatives):
        raise RuntimeError("native YAML reload validation aggregate drifted")
    if not isinstance(value.get("elapsedWallSeconds"), (int, float)) or value["elapsedWallSeconds"] <= 0:
        raise RuntimeError("native YAML reload elapsed duration is invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--generated-tool", type=Path, default=ROOT / "bin/nimbus_generated_backend_tuples")
    parser.add_argument("--native-tool", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pre_beta/native_yaml_reload_v1.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    generated_tool = args.generated_tool.resolve(strict=True)
    native_tool = args.native_tool.resolve(strict=True)
    output = args.output.resolve()
    binary_sha256 = sha256_file(binary)
    if args.verify:
        verify_evidence(json.loads(output.read_text(encoding="utf-8")), binary_sha256)
        print(f"[native-yaml-reload] PASS verify {output}")
        return 0

    started = time.monotonic()
    with native_stream(generated_tool, native_tool, ("--full-admitted-with-signatures",), signed=True) as stream:
        selected, selection_summary = select_representatives(stream)
    representative_results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="nimbus-native-yaml-reload-") as temporary:
        config_path = Path(temporary) / "representative.yaml"
        for signature, tuple_text in selected:
            representative_results.append(
                validate_representative(binary, native_tool, config_path, signature, tuple_text)
            )
    if sha256_file(binary) != binary_sha256:
        raise RuntimeError("simulator identity changed during YAML validation")
    elapsed = time.monotonic() - started
    generated_evidence = ROOT / "reports/pre_beta/generated_backend_tuple_evidence_v1.json"
    full_preflight_evidence = ROOT / "reports/pre_beta/native_full_preflight_evidence_v1.json"
    from scripts.tests.catalog import test_native_full_preflight as upstream_guard
    upstream_guard.verify_evidence(json.loads(full_preflight_evidence.read_text()), binary_sha256)
    evidence = {
        **baseline_records(record),
        "soundingInputs": sounding_records(record),
        "schema": "nimbus.native-yaml-reload-evidence.v1",
        "version": 1,
        "status": "pass",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "binary": record(binary),
        "generatedTool": record(generated_tool),
        "nativeTool": record(native_tool),
        "generatedSource": record(ROOT / "tests/compute/generated_backend_tuple_catalog_main.cpp"),
        "nativeSource": record(ROOT / "tests/compute/native_admission_batch_main.cpp"),
        "runtimeAdmissionSource": record(ROOT / "src/core/runtime/nimbus_sim.cpp"),
        "qualifier": record(Path(__file__)),
        "generatedEvidence": record(generated_evidence),
        "nativeFullPreflightEvidence": record(full_preflight_evidence),
        "summary": {
            **selection_summary,
            "passed": len(representative_results),
            "failed": 0,
            "skipped": 0,
        },
        "representativeResults": representative_results,
        "validationAggregateSha256": aggregate_sha256(representative_results),
        "elapsedWallSeconds": elapsed,
        "claimBoundary": (
            "The public simulator strictly reloaded and allocation-free validated one serialized YAML document "
            "for every native-admitted backend ownership signature plus witnesses for every admitted categorical "
            "field/value in each backend/maturity partition. This is representative strict-file loading, not "
            "all-tuple YAML loading, 3-D field allocation, timestep execution, a campaign, or external review."
        ),
    }
    write_atomic(output, evidence)
    verify_evidence(json.loads(output.read_text(encoding="utf-8")), binary_sha256)
    print(
        f"[native-yaml-reload] PASS: {len(representative_results)}/{len(representative_results)} "
        f"representatives, {EXPECTED_OWNERSHIP_SIGNATURES} ownership signatures, 0 skipped; "
        f"binary_sha256={binary_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOOL = ROOT / "bin/nimbus_generated_backend_tuples"
import sys
sys.path.insert(0, str(ROOT))
from scripts.qualification.catalog_admission import load_baseline, baseline_records, verify_baseline_records, verify_provenance_path

BASELINE = load_baseline()
EXPECTED_SUMMARY = BASELINE["generated"]["summary"]
EXPECTED_PARTITIONS = BASELINE["generated"]["partitions"]
EXPECTED_TUPLE_SHA256 = BASELINE["generated"]["acceptedTupleSha256"]


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


def run_catalog(tool: Path) -> tuple[dict[str, object], bytes]:
    result = subprocess.run(
        [str(tool)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("generated backend tuple tool failed:\n" + result.stderr.decode(errors="replace"))
    repeated = subprocess.run(
        [str(tool)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if repeated.returncode != 0 or repeated.stdout != result.stdout:
        raise RuntimeError("generated backend tuple catalog is not deterministic")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("generated backend tuple catalog is not a JSON object")
    validate_catalog(value)
    return value, result.stdout


def validate_catalog(value: dict[str, object]) -> None:
    if value.get("schema") != "nimbus.generated-backend-tuples.v1" or value.get("version") != 1:
        raise RuntimeError("generated backend tuple schema drifted")
    if value.get("summary") != EXPECTED_SUMMARY:
        raise RuntimeError("generated backend tuple summary drifted")
    if value.get("partitions") != EXPECTED_PARTITIONS:
        raise RuntimeError("generated backend tuple partitions drifted")
    if value.get("acceptedTupleSha256") != EXPECTED_TUPLE_SHA256:
        raise RuntimeError("generated accepted tuple identity drifted")
    groups = value.get("ownershipGroups")
    if not isinstance(groups, list) or len(groups) != EXPECTED_SUMMARY["ownershipSignatures"]:
        raise RuntimeError("generated ownership groups are incomplete")
    signatures: set[str] = set()
    representatives: set[str] = set()
    count = 0
    for group in groups:
        if not isinstance(group, dict):
            raise RuntimeError("generated ownership group is malformed")
        signature = group.get("signature")
        representative = group.get("representative")
        group_count = group.get("count")
        if (
            not isinstance(signature, str)
            or "unsupported" in signature
            or signature in signatures
            or not isinstance(representative, str)
            or representative in representatives
            or not isinstance(group_count, int)
            or group_count <= 0
        ):
            raise RuntimeError("generated ownership group identity drifted")
        signatures.add(signature)
        representatives.add(representative)
        count += group_count
    if count != EXPECTED_SUMMARY["accepted"]:
        raise RuntimeError("generated ownership group counts do not cover the accepted set")
    group_digest = hashlib.sha256(json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if group_digest != BASELINE["generated"]["ownershipGroupsSha256"]:
        raise RuntimeError("generated ownership group identity drifted")


def stream_tuple_identity(tool: Path) -> tuple[str, dict[str, int]]:
    process = subprocess.Popen(
        [str(tool), "--tuples"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    identities: set[bytes] = set()
    partitions = {name: 0 for name in EXPECTED_PARTITIONS}
    count = 0
    for raw_line in process.stdout:
        if not raw_line.endswith(b"\n") or b"\n" in raw_line[:-1]:
            process.kill()
            raise RuntimeError("generated tuple stream contains a malformed line")
        digest.update(raw_line)
        line = raw_line[:-1]
        identity = hashlib.sha256(line).digest()
        if identity in identities:
            process.kill()
            raise RuntimeError("generated tuple stream contains a duplicate identity")
        identities.add(identity)
        text = line.decode("ascii")
        if text.startswith("grid=collocated|backend=cpu|"):
            partition = "collocated.cpu"
        elif text.startswith("grid=collocated|backend=gpu|"):
            partition = "collocated.gpu"
        elif text.startswith("grid=c_grid|backend=cpu|qualified=false|"):
            partition = "c_grid.cpu.exploratory"
        elif text.startswith("grid=c_grid|backend=cpu|qualified=true|"):
            partition = "c_grid.cpu.qualified"
        elif text.startswith("grid=c_grid|backend=gpu|qualified=false|"):
            partition = "c_grid.gpu.exploratory"
        elif text.startswith("grid=c_grid|backend=gpu|qualified=true|"):
            partition = "c_grid.gpu.qualified"
        else:
            process.kill()
            raise RuntimeError("generated tuple stream has an unknown partition")
        partitions[partition] += 1
        count += 1
    stderr = process.stderr.read().decode(errors="replace") if process.stderr is not None else ""
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"generated tuple stream failed ({returncode}): {stderr}")
    if count != EXPECTED_SUMMARY["accepted"] or partitions != EXPECTED_PARTITIONS:
        raise RuntimeError("generated tuple stream cardinality drifted")
    if digest.hexdigest() != EXPECTED_TUPLE_SHA256:
        raise RuntimeError("generated tuple stream SHA-256 drifted")
    return digest.hexdigest(), partitions


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
    if value.get("schema") != "nimbus.generated-backend-tuple-evidence.v1" or value.get("version") != 1:
        raise RuntimeError("generated backend tuple evidence schema drifted")
    if value.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("generated backend tuple evidence does not bind the current binary")
    for key in ("binary", "tool", "source", "qualifier"):
        verify_record(value.get(key), key)
    verify_baseline_records(value, record)
    validate_catalog(value.get("catalog"))
    if value.get("summary") != {
        "evaluated": EXPECTED_SUMMARY["evaluated"],
        "accepted": EXPECTED_SUMMARY["accepted"],
        "ownershipSignatures": EXPECTED_SUMMARY["ownershipSignatures"],
        "duplicateTuples": 0,
    }:
        raise RuntimeError("generated backend tuple evidence summary drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    tool = args.tool.resolve(strict=True)
    binary_sha256 = sha256_file(binary)
    if args.verify:
        if args.output is None:
            raise RuntimeError("--verify requires --output")
        value = json.loads(args.output.resolve().read_text(encoding="utf-8"))
        verify_evidence(value, binary_sha256)
        print(f"[generated-backend-tuples] PASS verify {args.output}")
        return 0
    catalog, output = run_catalog(tool)
    tuple_sha256, partitions = stream_tuple_identity(tool)
    if tuple_sha256 != catalog["acceptedTupleSha256"] or partitions != catalog["partitions"]:
        raise RuntimeError("generated tuple stream and catalog disagree")
    if sha256_file(binary) != binary_sha256:
        raise RuntimeError("simulator identity changed during qualification")
    if args.output is not None:
        evidence = {
            **baseline_records(record),
            "qualifier": record(Path(__file__)),
            "schema": "nimbus.generated-backend-tuple-evidence.v1",
            "version": 1,
            "binary": record(binary),
            "tool": record(tool),
            "source": record(ROOT / "tests/compute/generated_backend_tuple_catalog_main.cpp"),
            "toolOutputSha256": hashlib.sha256(output).hexdigest(),
            "catalog": catalog,
            "summary": {
                "evaluated": EXPECTED_SUMMARY["evaluated"],
                "accepted": EXPECTED_SUMMARY["accepted"],
                "ownershipSignatures": EXPECTED_SUMMARY["ownershipSignatures"],
                "duplicateTuples": 0,
            },
            "claimBoundary": (
                "Canonical identity and ownership-signature partition for generated accepted backend tuples; "
                "not genuine per-tuple runtime execution, a long campaign, or external review."
            ),
        }
        write_atomic(args.output.resolve(), evidence)
    print(
        f"[generated-backend-tuples] PASS: {EXPECTED_SUMMARY['accepted']} accepted / {EXPECTED_SUMMARY['evaluated']} evaluated; "
        f"{EXPECTED_SUMMARY['ownershipSignatures']} ownership signatures; tuple_sha256={tuple_sha256}; binary_sha256={binary_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

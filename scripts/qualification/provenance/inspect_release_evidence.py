#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.qualification.provenance import release_manifest
from scripts.qualification.provenance.record_gate_result import (
    REQUIRED_GATE_POLICY,
    SCHEMA as GATE_SCHEMA,
)

DEFAULT_BINARY = Path("bin/nimbus_sim")
DEFAULT_GATE_DIRECTORY = Path("reports/pre_beta/gates")
DEFAULT_MANIFESTS = (
    Path("reports/pre_beta/local_release_candidate_v2.json"),
    Path("reports/pre_beta/local_release_candidate_v2_smoke.json"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> tuple[object | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as error:
        return None, str(error)


def inspect_receipt(
    path: Path, target: str, binary_sha256: str, root: Path = ROOT,
) -> dict[str, object]:
    record: dict[str, object] = {"target": target, "path": path.as_posix()}
    if not path.is_file():
        return {**record, "classification": "missing"}
    payload, error = load_json(path)
    if error is not None or not isinstance(payload, dict):
        return {**record, "classification": "invalid", "reason": error or "root is not an object"}
    if payload.get("schema") != GATE_SCHEMA or payload.get("target") != target:
        return {**record, "classification": "invalid", "reason": "schema or target mismatch"}
    receipt_sha256 = payload.get("binarySha256")
    record["binarySha256"] = receipt_sha256
    if receipt_sha256 != binary_sha256:
        return {**record, "classification": "stale"}
    if payload.get("status") != "pass":
        return {**record, "classification": "invalid", "reason": "status is not pass"}
    log = payload.get("log")
    log_sha256 = payload.get("logSha256")
    if not isinstance(log, str) or not isinstance(log_sha256, str):
        return {**record, "classification": "invalid", "reason": "log identity is missing"}
    log_path = root / log
    if not log_path.is_file() or sha256_file(log_path) != log_sha256:
        return {**record, "classification": "invalid", "reason": "log identity does not verify"}
    return {**record, "classification": "current"}


def inspect_manifest(
    path: Path, binary_sha256: str, root: Path = ROOT,
) -> dict[str, object]:
    record: dict[str, object] = {"path": path.as_posix()}
    if not path.is_file():
        return {**record, "classification": "missing"}
    payload, error = load_json(path)
    if error is not None or not isinstance(payload, dict):
        return {**record, "classification": "invalid", "reason": error or "root is not an object"}
    binary = payload.get("binary")
    manifest_sha256 = binary.get("sha256") if isinstance(binary, dict) else None
    record["binarySha256"] = manifest_sha256
    if manifest_sha256 != binary_sha256:
        return {**record, "classification": "stale"}
    try:
        release_manifest.verify_manifest(payload, root)
    except release_manifest.ManifestError as error:
        return {**record, "classification": "invalid", "reason": str(error)}
    return {**record, "classification": "current"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--gate-directory", type=Path, default=DEFAULT_GATE_DIRECTORY)
    parser.add_argument("--manifest", action="append", type=Path)
    parser.add_argument("--require-current", action="store_true")
    args = parser.parse_args()
    binary = (ROOT / args.binary).resolve() if not args.binary.is_absolute() else args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"missing simulator: {binary}")
    binary_sha256 = sha256_file(binary)
    gate_directory = ROOT / args.gate_directory
    receipts = [
        inspect_receipt(
            gate_directory / f"{target}.json", target, binary_sha256, ROOT,
        )
        for target in REQUIRED_GATE_POLICY
    ]
    manifests = [
        inspect_manifest(ROOT / path, binary_sha256, ROOT)
        for path in (args.manifest or DEFAULT_MANIFESTS)
    ]
    report = {
        "schema": "nimbus.release_evidence_inspection",
        "version": 1,
        "binary": {
            "path": (
                binary.relative_to(ROOT).as_posix()
                if binary.is_relative_to(ROOT) else binary.as_posix()
            ),
            "sha256": binary_sha256,
        },
        "receipts": receipts,
        "manifests": manifests,
        "summary": {
            classification: sum(
                item["classification"] == classification
                for item in (*receipts, *manifests)
            )
            for classification in ("current", "stale", "missing", "invalid")
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_current and any(
        item["classification"] != "current" for item in (*receipts, *manifests)
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

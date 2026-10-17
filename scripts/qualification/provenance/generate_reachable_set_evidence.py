#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.tests.catalog.test_catalog_authority import (  # noqa: E402
    canonical_surface,
    fingerprint,
    validate_catalog,
    validated_readiness,
)


SCHEMA = "nimbus.catalog.reachable-set-evidence"
VERSION = 1


class EvidenceError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, root: Path) -> tuple[Path, str]:
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise EvidenceError(f"input is a symbolic link: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise EvidenceError(f"input is not a regular file: {candidate}")
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"input is outside the evidence root: {candidate}") from exc
    return resolved, relative


def file_record(path: Path, root: Path) -> dict[str, object]:
    resolved, relative = checked_file(path, root)
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def module_schemes(reference: dict[str, object]) -> dict[str, list[str]]:
    modules = reference.get("modules")
    if not isinstance(modules, list):
        raise EvidenceError("catalog authority modules are malformed")
    result: dict[str, list[str]] = {}
    for module in modules:
        if not isinstance(module, dict) or not isinstance(module.get("id"), str):
            raise EvidenceError("catalog authority module is malformed")
        schemes = module.get("schemes")
        if not isinstance(schemes, list) or not all(
            isinstance(value, str) for value in schemes
        ):
            raise EvidenceError("catalog authority scheme list is malformed")
        result[module["id"]] = schemes
    return result


def expected_matrices(reference: dict[str, object]) -> list[dict[str, object]]:
    axes = module_schemes(reference)
    collocated_physics = (
        len(axes["microphysics"]) * len(axes["radiation"]) *
        len(axes["boundary_layer"]) * len(axes["turbulence"]) *
        len(axes["chaos"]) * len(axes["terrain"])
    )
    collocated = collocated_physics * len(axes["advection"]) * 2 * 3
    cgrid = collocated_physics * len(axes["diffusion"]) * 2 * 2 * 3 * 4 * 4 * 2
    if collocated != 38880 or cgrid != 1658880:
        raise EvidenceError("generated matrix cardinality drifted")
    return [
        {
            "id": "gpu_apply_controls",
            "evaluated": 2 * 2 * 2 * 5 * 3 * 2 * 2 * 2,
            "scope": "backend/qualification/dynamics/microphysics/turbulence/apply flags",
        },
        {
            "id": "boundary_topology",
            "evaluated": 2 * 6 * 6 * 6 * 6,
            "scope": "staggering and four lateral boundary selections",
        },
        {
            "id": "collocated_physics",
            "evaluated": collocated,
            "accepted": 5640,
            "ownershipBound": 5640,
            "scope": (
                "six descriptor-backed collocated physics modules plus public "
                "advection/turbulence-mode and all ten engine process/state-commit owners across CPU/GPU"
            ),
        },
        {
            "id": "cgrid_engine",
            "evaluated": cgrid,
            "accepted": 359424,
            "ownershipBound": 359424,
            "scope": (
                "descriptor physics plus backend/qualification/mode/surface/"
                "initialization/output with all ten engine process/state-commit owners"
            ),
        },
    ]


def build_evidence(
    *, root: Path, binary: Path, cross_product_binary: Path,
    authority: Path, output: Path,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    binary_path, _ = checked_file(binary, root)
    cross_product_path, _ = checked_file(cross_product_binary, root)
    authority_path, _ = checked_file(authority, root)
    source_path, _ = checked_file(
        root / "tests/core/test_scheme_catalog.cpp", root,
    )

    catalog_result = subprocess.run(
        [str(binary_path), "--scheme-catalog=json"], cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if catalog_result.returncode != 0:
        raise EvidenceError(
            "exact binary did not publish its catalog: " +
            catalog_result.stderr.decode("utf-8", errors="replace")
        )
    catalog = json.loads(catalog_result.stdout)
    reference = json.loads(authority_path.read_text(encoding="utf-8"))
    module_count, scheme_count, snapshot = validate_catalog(catalog)
    actual_authority = {
        "schema": catalog.get("schema"),
        "catalogVersion": catalog.get("catalogVersion"),
        "authorityFingerprintSha256": fingerprint(canonical_surface(catalog)),
        "readinessFingerprintSha256": fingerprint(validated_readiness(catalog)),
        "moduleCount": module_count,
        "schemeCount": scheme_count,
        "modules": snapshot,
    }
    if actual_authority != reference:
        raise EvidenceError("exact-binary catalog differs from catalog authority")

    log_path = output.with_suffix(".cross_product.log")
    command = [str(cross_product_path), "[cross-product]"]
    cross_product = subprocess.run(
        command, cwd=root, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    write_atomic(log_path, cross_product.stdout)
    if cross_product.returncode != 0:
        raise EvidenceError(
            "generated cross-product gate failed; see " + str(log_path)
        )

    matrices = expected_matrices(reference)

    scope = {
        "categoricalExactBinary": True,
        "continuousParametersEnumerated": False,
        "runtimeQualificationUniversal": False,
    }
    total_evaluated = sum(int(matrix["evaluated"]) for matrix in matrices)
    reachable_identity = {
        "catalogAuthorityFingerprintSha256": actual_authority[
            "authorityFingerprintSha256"
        ],
        "scope": scope,
        "matrices": matrices,
        "totalEvaluated": total_evaluated,
    }
    evidence: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "binary": file_record(binary_path, root),
        "catalogAuthority": file_record(authority_path, root),
        "catalog": actual_authority,
        "crossProductBinary": file_record(cross_product_path, root),
        "crossProductSource": file_record(source_path, root),
        "matrices": matrices,
        "totalEvaluated": total_evaluated,
        "receipt": {
            "target": "test-catalog-reachable-matrix/cross-product",
            "command": [
                file_record(cross_product_path, root)["path"],
                "[cross-product]",
            ],
            "exitStatus": cross_product.returncode,
            "skipped": 0,
            "fallbackUsed": False,
            "requiredHardware": False,
            "log": file_record(log_path, root),
        },
    }
    evidence["reachableSetSha256"] = hashlib.sha256(
        canonical_bytes(reachable_identity)
    ).hexdigest()
    evidence["integritySealSha256"] = hashlib.sha256(
        canonical_bytes(evidence)
    ).hexdigest()
    return evidence


def verify_file_record(record: object, root: Path, label: str) -> None:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise EvidenceError(f"{label} file record is malformed")
    path = Path(record["path"])
    if path.is_absolute() or ".." in path.parts:
        raise EvidenceError(f"{label} path escapes the evidence root")
    if file_record(root / path, root) != record:
        raise EvidenceError(f"{label} file record drifted")


def verify_evidence(evidence: object, root: Path) -> None:
    root = root.resolve(strict=True)
    if not isinstance(evidence, dict):
        raise EvidenceError("reachable evidence root is malformed")
    if evidence.get("schema") != SCHEMA or evidence.get("version") != VERSION:
        raise EvidenceError("reachable evidence schema is unsupported")
    integrity_seal = evidence.get("integritySealSha256")
    unsigned = dict(evidence)
    unsigned.pop("integritySealSha256", None)
    if integrity_seal != hashlib.sha256(canonical_bytes(unsigned)).hexdigest():
        raise EvidenceError("reachable evidence seal is invalid")
    for key in (
        "binary", "catalogAuthority", "crossProductBinary",
        "crossProductSource",
    ):
        verify_file_record(evidence.get(key), root, key)
    matrices = evidence.get("matrices")
    if not isinstance(matrices, list) or len(matrices) != 4:
        raise EvidenceError("reachable evidence matrices are incomplete")
    if evidence.get("totalEvaluated") != sum(
        int(matrix.get("evaluated", -1))
        for matrix in matrices if isinstance(matrix, dict)
    ):
        raise EvidenceError("reachable evidence total is inconsistent")
    scope = evidence.get("scope")
    if not isinstance(scope, dict) or scope.get("categoricalExactBinary") is not True:
        raise EvidenceError("reachable evidence scope is invalid")
    catalog = evidence.get("catalog")
    if not isinstance(catalog, dict):
        raise EvidenceError("reachable evidence catalog is invalid")
    if matrices != expected_matrices(catalog):
        raise EvidenceError("reachable evidence matrix contract drifted")
    reachable_identity = {
        "catalogAuthorityFingerprintSha256": catalog.get(
            "authorityFingerprintSha256"
        ),
        "scope": scope,
        "matrices": matrices,
        "totalEvaluated": evidence.get("totalEvaluated"),
    }
    if evidence.get("reachableSetSha256") != hashlib.sha256(
        canonical_bytes(reachable_identity)
    ).hexdigest():
        raise EvidenceError("reachable set fingerprint is invalid")
    receipt = evidence.get("receipt")
    if (
        not isinstance(receipt, dict) or receipt.get("exitStatus") != 0 or
        receipt.get("skipped") != 0 or receipt.get("fallbackUsed") is not False
    ):
        raise EvidenceError("reachable evidence receipt is not a clean pass")
    verify_file_record(receipt.get("log"), root, "receipt.log")


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--binary", type=Path, default=Path("bin/nimbus_sim"))
    parser.add_argument(
        "--cross-product-binary", type=Path,
        default=Path("bin/test_core_scheme_catalog"),
    )
    parser.add_argument(
        "--authority", type=Path,
        default=Path("tests/fixtures/reference/catalog/authority_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify:
            verify_evidence(load(args.output), args.root)
            print(f"[reachable-set-evidence] PASS verify {args.output}")
            return 0
        evidence = build_evidence(
            root=args.root, binary=args.binary,
            cross_product_binary=args.cross_product_binary,
            authority=args.authority, output=args.output,
        )
        write_atomic(
            args.output,
            (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode(),
        )
        verify_evidence(load(args.output), args.root)
        print(
            "[reachable-set-evidence] PASS: "
            f"{evidence['totalEvaluated']} categorical cases, "
            f"sha256={evidence['reachableSetSha256']}"
        )
    except (EvidenceError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

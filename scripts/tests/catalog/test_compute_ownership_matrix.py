#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_owner(record: dict[str, object]) -> dict[str, str]:
    backend = record["backend"]
    microphysics = record["microphysics"]
    turbulence = record["turbulence"]
    apply = record["apply"]
    if not isinstance(apply, dict):
        raise RuntimeError("ownership record apply field is not an object")

    if backend == "cpu":
        return {
            "microphysicsStateCommit": "inactive" if microphysics == "none" else "host",
            "slowStateCommit": "host",
            "sgsStateCommit": "inactive" if turbulence == "none" else "host",
        }

    if microphysics == "none":
        microphysics_owner = "inactive"
    elif not apply["microphysics"]:
        microphysics_owner = "host"
    elif microphysics in ("kessler", "lin", "thompson"):
        microphysics_owner = "device"
    else:
        microphysics_owner = "unsupported"

    if turbulence == "none":
        sgs_owner = "inactive"
    elif not apply["sgs"]:
        sgs_owner = "host"
    elif turbulence in ("smagorinsky", "tke"):
        sgs_owner = "device"
    else:
        sgs_owner = "unsupported"

    return {
        "microphysicsStateCommit": microphysics_owner,
        "slowStateCommit": "device" if apply["slow"] else "host",
        "sgsStateCommit": sgs_owner,
    }


def validate(matrix: dict[str, object]) -> None:
    if matrix.get("schema") != "nimbus.compute-ownership-matrix.v1":
        raise RuntimeError("compute ownership matrix schema drifted")
    if matrix.get("version") != 1 or matrix.get("scope") != "state_commit_controls":
        raise RuntimeError("compute ownership matrix identity drifted")
    records = matrix.get("records")
    if not isinstance(records, list):
        raise RuntimeError("compute ownership matrix records are missing")

    backends = ("cpu", "gpu")
    dynamics = (
        "klemp_wilhelmson_split_cartesian_cgrid",
        "klemp_wilhelmson_split_cartesian_agrid",
    )
    microphysics = ("none", "kessler", "lin", "thompson", "milbrandt")
    turbulence = ("none", "smagorinsky", "tke")
    expected_signatures = set(itertools.product(
        backends, (False, True), dynamics, microphysics, turbulence,
        (False, True), (False, True), (False, True),
    ))
    actual_signatures: set[tuple[object, ...]] = set()
    supported = 0
    unsupported = 0
    for expected_index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("index") != expected_index:
            raise RuntimeError("compute ownership record indices are not contiguous")
        apply = record.get("apply")
        owners = record.get("owners")
        errors = record.get("errors")
        if not isinstance(apply, dict) or not isinstance(owners, dict) or not isinstance(errors, list):
            raise RuntimeError(f"compute ownership record {expected_index} is malformed")
        signature = (
            record.get("backend"), record.get("requireQualified"), record.get("dynamics"),
            record.get("microphysics"), record.get("turbulence"), apply.get("microphysics"),
            apply.get("slow"), apply.get("sgs"),
        )
        if signature in actual_signatures:
            raise RuntimeError(f"duplicate compute ownership signature at record {expected_index}")
        actual_signatures.add(signature)
        expected = expected_owner(record)
        if owners != expected:
            raise RuntimeError(
                f"compute ownership record {expected_index} differs from the independent oracle: "
                f"expected={expected}, actual={owners}"
            )
        valid = "unsupported" not in owners.values()
        if record.get("valid") != valid or (len(errors) == 0) != valid:
            raise RuntimeError(f"compute ownership record {expected_index} has inconsistent validity")
        if valid:
            supported += 1
        else:
            unsupported += 1

    if actual_signatures != expected_signatures:
        missing = expected_signatures - actual_signatures
        extra = actual_signatures - expected_signatures
        raise RuntimeError(f"compute ownership axes drifted: missing={len(missing)}, extra={len(extra)}")
    expected_summary = {
        "evaluated": len(expected_signatures),
        "supported": supported,
        "unsupported": unsupported,
    }
    if matrix.get("summary") != expected_summary or expected_summary != {
        "evaluated": 960, "supported": 912, "unsupported": 48,
    }:
        raise RuntimeError(f"compute ownership summary drifted: {matrix.get('summary')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"missing simulator: {binary}")
    binary_sha256 = sha256_file(binary)
    result = subprocess.run(
        [str(binary), "--compute-ownership-matrix=json"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "compute ownership matrix command failed:\n" +
            result.stderr.decode("utf-8", errors="replace")
        )
    matrix = json.loads(result.stdout)
    validate(matrix)
    if sha256_file(binary) != binary_sha256:
        raise RuntimeError("simulator changed while compute ownership evidence was generated")

    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        evidence = {
            "schema": "nimbus.compute-ownership-evidence.v1",
            "version": 1,
            "binary": {
                "path": str(binary.relative_to(ROOT)),
                "sha256": binary_sha256,
            },
            "matrixSha256": sha256_bytes(result.stdout),
            "matrix": matrix,
        }
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)

    print(
        "[compute-ownership-matrix] PASS: 960 state-commit combinations; "
        f"supported=912; unsupported=48; binary_sha256={binary_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

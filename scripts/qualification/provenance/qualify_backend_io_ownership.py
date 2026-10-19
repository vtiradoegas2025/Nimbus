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
SUITES = (
    ("output_writer", ["./bin/test_core_output_writer"], ((2749, 39),), ()),
    ("netcdf_history", ["./bin/test_core_netcdf_history_writer"], ((101, 6),), ()),
    ("zfp_history", ["./bin/test_core_zfp_roundtrip"], ((47121, 23),), ()),
    ("shm_transport", ["./bin/test_core_shm"], ((181, 15),), ()),
    ("durable_file", ["./bin/test_core_durable_file"], ((30, 12),), ()),
    ("checkpoint", ["./bin/test_core_simulation_checkpoint"], ((1374, 50),), ()),
    ("trajectories", ["./bin/test_diagnostics_trajectories"], ((267, 22),), ()),
    ("live_shm", ["python3", "scripts/qualification/output/qualify_live_shm.py"], (),
     ("[live-shm] PASS", "1/4-thread identity", "no durable frame duplication")),
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
        raise RuntimeError(f"{name} I/O summary drifted: expected={summaries}, actual={actual}")
    for marker in markers:
        if marker not in output:
            raise RuntimeError(f"{name} is missing I/O marker: {marker}")
    lowered = output.lower()
    for forbidden in ("no tests ran", "tests failed", "skipping live shm", "unsupported: skipping"):
        if forbidden in lowered:
            raise RuntimeError(f"{name} contains a skipped or failed marker: {forbidden}")
    return sum(a for a, _ in actual), sum(c for _, c in actual)


def anchor_paths() -> dict[str, Path]:
    return {
        "compositeRuntime": ROOT / "reports/pre_beta/composite_runtime_ownership_v1.json",
        "physicalTopology": ROOT / "reports/pre_beta/physical_topology_ownership_v1.json",
        "reachable": ROOT / "reports/pre_beta/reachable_set_v1.json",
    }


def validate_anchors(binary_sha256: str) -> dict[str, dict[str, object]]:
    paths = anchor_paths()
    values = {name: load(path, name) for name, path in paths.items()}
    stale = sorted(
        name for name, value in values.items()
        if value.get("binary", {}).get("sha256") != binary_sha256
    )
    if stale:
        raise RuntimeError("backend I/O anchors are stale: " + ", ".join(stale))
    if any(value.get("status") != "pass" for value in values.values() if "status" in value):
        raise RuntimeError("backend I/O ownership anchor is not passing")
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
    if value.get("schema") != "nimbus.backend-io-ownership-evidence.v1" or value.get("version") != 1:
        raise RuntimeError("backend I/O evidence schema drifted")
    if value.get("status") != "pass" or value.get("binary", {}).get("sha256") != binary_sha256:
        raise RuntimeError("backend I/O evidence does not bind the passing current binary")
    verify_record(value.get("binary"), "binary")
    anchors = value.get("anchors")
    if not isinstance(anchors, dict) or set(anchors) != set(anchor_paths()):
        raise RuntimeError("backend I/O anchors are incomplete")
    for name, file_record in anchors.items():
        verify_record(file_record, f"anchors.{name}")
    sources = value.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"gate", "testsMake"}:
        raise RuntimeError("backend I/O sources are incomplete")
    for name, file_record in sources.items():
        verify_record(file_record, f"sources.{name}")
    verify_record(value.get("log"), "log")
    if value.get("summary") != {"suites": 8, "assertions": 51823, "testCases": 167,
                                "skipped": 0, "frontendIncluded": False}:
        raise RuntimeError("backend I/O summary drifted")
    receipts = value.get("receipts")
    expected = expected_receipts()
    if not isinstance(receipts, list) or len(receipts) != len(expected):
        raise RuntimeError("backend I/O receipts are incomplete")
    seen: set[str] = set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or receipt.get("name") not in expected or receipt["name"] in seen:
            raise RuntimeError("backend I/O receipt identity drifted")
        name = receipt["name"]
        seen.add(name)
        command, assertions, cases = expected[name]
        if (receipt.get("command") != command or receipt.get("exitStatus") != 0
                or receipt.get("assertions") != assertions or receipt.get("testCases") != cases
                or receipt.get("skipped") != 0
                or not isinstance(receipt.get("elapsedWallSeconds"), (int, float))
                or receipt["elapsedWallSeconds"] <= 0):
            raise RuntimeError(f"backend I/O receipt drifted: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/pre_beta/backend_io_ownership_v1.json")
    parser.add_argument("--log", type=Path,
                        default=ROOT / "reports/pre_beta/backend_io_ownership_v1.log")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    binary_sha256 = sha256_file(binary)
    if args.verify:
        verify_evidence(load(args.output.resolve(), "backend I/O evidence"), binary_sha256)
        print(f"[backend-io-ownership] PASS verify {args.output}")
        return 0
    anchors = validate_anchors(binary_sha256)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
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
            raise RuntimeError(f"backend I/O suite failed: {name}")
        assertions, cases = validate_output(name, result.stdout, summaries, markers)
        total_assertions += assertions
        total_cases += cases
        receipts.append({"name": name, "command": command, "exitStatus": 0,
                         "assertions": assertions, "testCases": cases, "skipped": 0,
                         "elapsedWallSeconds": elapsed})
        if sha256_file(binary) != binary_sha256:
            raise RuntimeError(f"public binary changed after backend I/O suite: {name}")
        validate_anchors(binary_sha256)
    if (total_assertions, total_cases) != (51823, 167):
        raise RuntimeError(f"backend I/O totals drifted: assertions={total_assertions}, cases={total_cases}")
    log_path = args.log.resolve()
    write_atomic(log_path, "\n".join(log_parts).encode())
    evidence = {
        "schema": "nimbus.backend-io-ownership-evidence.v1", "version": 1, "status": "pass",
        "completedUtc": datetime.now(timezone.utc).isoformat(), "binary": record(binary),
        "anchors": anchors,
        "sources": {"gate": record(Path(__file__)), "testsMake": record(ROOT / "mk/tests.mk")},
        "log": record(log_path), "receipts": receipts,
        "summary": {"suites": 8, "assertions": 51823, "testCases": 167,
                    "skipped": 0, "frontendIncluded": False},
        "claimBoundary": (
            "Native backend history, NetCDF, ZFP, SHM, durable-file, checkpoint, output, and trajectory "
            "ownership; excludes Studio/Tauri/TypeScript I/O and is not the full qualify-io release gate."
        ),
    }
    write_atomic(args.output.resolve(), (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode())
    verify_evidence(load(args.output.resolve(), "backend I/O evidence"), binary_sha256)
    print("[backend-io-ownership] PASS: 8 suites, 51823 assertions, 167 cases, "
          f"0 skipped; frontend excluded; binary_sha256={binary_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

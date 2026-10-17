#!/usr/bin/env python3
"""Bound native-Metal rollback regression after sharing the flat-step transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
TEST_SOURCE = ROOT / "tests/core/test_nested_production_step.cpp"
TRANSACTION = ROOT / "include/core/runtime/flat_native_tvd_step_transaction.hpp"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-binary", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.test_binary.resolve()
    simulator = args.simulator.resolve()
    raw = args.raw_root.resolve()
    report = args.report_output.resolve()
    if not binary.is_file() or not simulator.is_file() or raw.exists() or report.exists():
        parser.error("binaries must exist and outputs must be new")
    identity = {"testBinarySha256": sha256(binary),
                "simulatorSha256": sha256(simulator),
                "testSourceSha256": sha256(TEST_SOURCE),
                "transactionSourceSha256": sha256(TRANSACTION),
                "runnerSha256": sha256(Path(__file__))}
    raw.mkdir(parents=True)
    rows = []
    started = time.monotonic()
    status = "pass"
    for name, selected, expected in (
        ("direct_canonical_east", "[metal][native_dry_tvd][full_step_rollback]", 0),
        ("headless_east", "[metal][native_dry_tvd][headless_step_rollback]", 0),
    ):
        try:
            completed = subprocess.run(
                [str(binary), selected], cwd=ROOT, capture_output=True,
                text=True, timeout=120, check=False,
            )
            code = completed.returncode
            message = completed.stdout + "\n" + completed.stderr
        except subprocess.TimeoutExpired as error:
            code = 124
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            message = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
        log = raw / f"{name}.log"
        log.write_text(message, encoding="utf-8")
        passed = code == expected and "All tests passed" in message
        rows.append({"name": name, "filter": selected,
                     "returnCode": code, "logSha256": sha256(log),
                     "pass": passed})
        if not passed:
            status = "rollback_regression_red"
            break
    if identity != {"testBinarySha256": sha256(binary),
                    "simulatorSha256": sha256(simulator),
                    "testSourceSha256": sha256(TEST_SOURCE),
                    "transactionSourceSha256": sha256(TRANSACTION),
                    "runnerSha256": sha256(Path(__file__))}:
        status = "source_identity_drift"
    files = [item for item in raw.rglob("*") if item.is_file()]
    bytes_used = sum(item.stat().st_size for item in files)
    if time.monotonic() - started >= 240 or len(files) >= 100 or \
            bytes_used >= 16 * 1024 * 1024:
        status = "resource_cap"
    report.parent.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "nimbus.static-vertical-shared-transaction-metal-regression.v1",
               "status": status,
               "claimBoundary": "Existing native-Metal canonical/east direct and headless rollback regression after transaction generalization; static CPU public admission and whole-backend Ready remain separate.",
               **identity, "cases": rows,
               "wallSeconds": time.monotonic() - started,
               "rawFiles": len(files), "rawBytes": bytes_used,
               "rawRoot": str(raw)}
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"{status}: {len(rows)}/2 native-Metal rollback cases")
    return 0 if status == "pass" and len(rows) == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())

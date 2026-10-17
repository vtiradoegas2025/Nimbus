#!/usr/bin/env python3
"""Bind warm-stage no-observer byte replay and focused observer tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = "b54d9d68d2bb4ccd5642208482cd23e05b1842bc3c8936a0b1210d06c8898a6d"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-binary", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.test_binary.resolve(strict=True)
    simulator = args.simulator.resolve(strict=True)
    raw = args.raw_root.resolve()
    if raw.exists() or args.report_output.exists():
        parser.error("use new raw/report paths to preserve prior evidence")
    raw.mkdir(parents=True)
    header = ROOT / "src/microphysics/schemes/milbrandt/warm_rain.hpp"
    test_source = ROOT / "tests/physics/test_milbrandt_warm_observer.cpp"
    before = {"header": digest(header), "test": digest(test_source),
              "binary": digest(binary), "simulator": digest(simulator)}
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [str(binary)], cwd=ROOT, capture_output=True,
            text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as error:
        (raw / "focused.log").write_text(str(error), encoding="utf-8")
        raise RuntimeError("warm observer focused gate exceeded 20-second cap") from error
    log = raw / "focused.log"
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    after = {"header": digest(header), "test": digest(test_source),
             "binary": digest(binary), "simulator": digest(simulator)}
    if before != after:
        raise RuntimeError("warm observer source or binary drifted during gate")
    if completed.returncode != 0 or f"[warm-observer-baseline] sha256={EXPECTED}" not in log.read_text():
        raise RuntimeError("warm observer baseline bytes drifted")
    report = {
        "schema": "nimbus.milbrandt-warm-observer.v1",
        "status": "pass",
        "claimBoundary": "No-observer warm-stage rate bytes only unless observer assertions are present in focused log.",
        "expectedRateBytesSha256": EXPECTED,
        "returnCode": completed.returncode,
        "wallSeconds": time.monotonic() - start,
        "inputSha256": before,
        "runnerSha256": digest(Path(__file__)),
        "logSha256": digest(log),
        "rawRoot": str(raw),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "rateBytesSha256": EXPECTED,
                      "simulatorSha256": before["simulator"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Preserve the first event-buffer raw-struct comparison red."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


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
    result = subprocess.run(
        [str(binary)], cwd=ROOT, capture_output=True,
        text=True, timeout=20, check=False)
    log = raw / "focused.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    content = log.read_text(encoding="utf-8")
    if result.returncode != 16 or "94 passed | 16 failed" not in content or \
            "warm-observer-observed] sha256=b54d9d68d2bb4ccd5642208482cd23e05b1842bc3c8936a0b1210d06c8898a6d" not in content:
        raise RuntimeError("expected padded-struct comparison red did not reproduce")
    report = {
        "schema": "nimbus.milbrandt-warm-observer-padding-red.v1",
        "status": "red",
        "returnCode": result.returncode,
        "causeBoundary": "One/four-thread rates match; raw padded event-struct memcmp differs.",
        "testBinarySha256": digest(binary),
        "simulatorSha256": digest(simulator),
        "headerSha256": digest(ROOT / "src/microphysics/schemes/milbrandt/warm_rain.hpp"),
        "testSourceSha256": digest(ROOT / "tests/physics/test_milbrandt_warm_observer.cpp"),
        "runnerSha256": digest(Path(__file__)),
        "logSha256": digest(log),
        "rawRoot": str(raw),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "red", "returnCode": result.returncode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

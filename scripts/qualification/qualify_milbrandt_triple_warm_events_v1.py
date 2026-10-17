#!/usr/bin/env python3
"""Seal the bounded default-off Milbrandt rain Z event component gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/microphysics/schemes/milbrandt/triple_warm_events.hpp",
    "src/microphysics/schemes/milbrandt/triple_warm_events.cpp",
    "src/microphysics/schemes/milbrandt/warm_rain.hpp",
    "tests/physics/test_milbrandt_triple_warm_events.cpp",
    "mk/tests.mk",
)


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
    source_before = {name: digest(ROOT / name) for name in SOURCES}
    test_before = digest(binary)
    simulator_before = digest(simulator)
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [str(binary)], cwd=ROOT, capture_output=True,
            text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as error:
        (raw / "focused.log").write_text(str(error), encoding="utf-8")
        raise RuntimeError("rain event gate exceeded 20-second cap") from error
    log = raw / "focused.log"
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    source_after = {name: digest(ROOT / name) for name in SOURCES}
    if source_before != source_after or test_before != digest(binary) or \
            simulator_before != digest(simulator):
        raise RuntimeError("rain event gate source or binary drifted")
    if completed.returncode != 0 or \
            "All tests passed (128 assertions in 4 test cases)" not in log.read_text():
        raise RuntimeError("rain event focused gate failed; preserve raw log")
    report = {
        "schema": "nimbus.milbrandt-triple-warm-events.v1",
        "status": "pass",
        "claimBoundary": "Pure default-off rain B3/B4 API fed caller-supplied admitted events; not warm_rain_cell output or runtime integration.",
        "assertions": 128,
        "testCases": 4,
        "wallSeconds": time.monotonic() - start,
        "testBinarySha256": test_before,
        "simulatorSha256": simulator_before,
        "runnerSha256": digest(Path(__file__)),
        "sourceSha256": source_before,
        "logSha256": digest(log),
        "rawRoot": str(raw),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "assertions": 128,
                      "simulatorSha256": simulator_before}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

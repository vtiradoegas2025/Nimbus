#!/usr/bin/env python3
"""Record bounded default-off phase-aware rain-Z evaluator/adapter evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/microphysics/schemes/milbrandt/warm_rain.hpp",
    "src/microphysics/schemes/milbrandt/triple_warm_events.hpp",
    "src/microphysics/schemes/milbrandt/triple_warm_events.cpp",
    "src/microphysics/schemes/milbrandt/triple_warm_phase.hpp",
    "src/microphysics/schemes/milbrandt/triple_warm_phase.cpp",
    "src/microphysics/schemes/milbrandt/triple_warm_adapter.hpp",
    "src/microphysics/schemes/milbrandt/triple_warm_adapter.cpp",
    "tests/physics/test_milbrandt_triple_warm_phase_adapter.cpp",
    "mk/tests.mk",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-binary", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--expect", choices=("pass", "red"), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.test_binary.resolve(strict=True)
    simulator = args.simulator.resolve(strict=True)
    raw = args.raw_root.resolve()
    if raw.exists() or args.report_output.exists():
        parser.error("use new raw/report paths to preserve prior evidence")
    raw.mkdir(parents=True)
    before = {name: digest(ROOT / name) for name in SOURCES}
    test_before = digest(binary)
    simulator_before = digest(simulator)
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [str(binary)], cwd=ROOT, capture_output=True,
            text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as error:
        (raw / "focused.log").write_text(str(error), encoding="utf-8")
        raise RuntimeError("phase-aware focused gate exceeded 20-second cap") from error
    log = raw / "focused.log"
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    after = {name: digest(ROOT / name) for name in SOURCES}
    if before != after or test_before != digest(binary) or \
            simulator_before != digest(simulator):
        raise RuntimeError("phase-aware source or binary drifted")
    actual = "pass" if completed.returncode == 0 else "red"
    report = {
        "schema": "nimbus.milbrandt-triple-warm-phase-adapter.v1",
        "status": actual,
        "claimBoundary": "Default-off pure phase-aware B3/B4 plus warm observer adapter; no live MilbrandtScheme/public/native integration.",
        "returnCode": completed.returncode,
        "wallSeconds": time.monotonic() - start,
        "sourceSha256": before,
        "testBinarySha256": test_before,
        "simulatorSha256": simulator_before,
        "runnerSha256": digest(Path(__file__)),
        "logSha256": digest(log),
        "rawRoot": str(raw),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if actual != args.expect:
        raise RuntimeError(f"phase-aware gate was {actual}, expected {args.expect}")
    print(json.dumps({"status": actual, "returnCode": completed.returncode,
                      "simulatorSha256": simulator_before}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

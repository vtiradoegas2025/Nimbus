#!/usr/bin/env python3
"""Record bounded isolated five-species q/N/Z sedimentation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/microphysics/schemes/milbrandt/triple_moment.hpp",
    "src/microphysics/schemes/milbrandt/triple_moment.cpp",
    "src/microphysics/schemes/milbrandt/triple_sedimentation.hpp",
    "src/microphysics/schemes/milbrandt/triple_sedimentation.cpp",
    "src/microphysics/schemes/milbrandt/sedimentation.hpp",
    "tests/physics/test_milbrandt_triple_five_sedimentation.cpp",
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
    test_binary = args.test_binary.resolve(strict=True)
    simulator = args.simulator.resolve(strict=True)
    raw = args.raw_root.resolve()
    if raw.exists() or args.report_output.exists():
        parser.error("use new raw/report paths to preserve prior evidence")
    raw.mkdir(parents=True)
    source_before = {name: digest(ROOT / name) for name in SOURCES}
    binary_before = digest(test_binary)
    simulator_before = digest(simulator)
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [str(test_binary)], cwd=ROOT, capture_output=True,
            text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as error:
        (raw / "focused.log").write_text(str(error), encoding="utf-8")
        raise RuntimeError("five-species test exceeded 20-second cap") from error
    log = raw / "focused.log"
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    source_after = {name: digest(ROOT / name) for name in SOURCES}
    if source_before != source_after or binary_before != digest(test_binary) or \
            simulator_before != digest(simulator):
        raise RuntimeError("five-species source or binary drifted during gate")
    actual = "pass" if completed.returncode == 0 else "red"
    report = {
        "schema": "nimbus.milbrandt-triple-five-sedimentation.v1",
        "status": actual,
        "claimBoundary": "Default-off isolated CPU five-species columns only; no MilbrandtScheme/runtime, transport, checkpoint or native integration.",
        "returnCode": completed.returncode,
        "wallSeconds": time.monotonic() - start,
        "testBinarySha256": binary_before,
        "simulatorSha256": simulator_before,
        "runnerSha256": digest(Path(__file__)),
        "sourceSha256": source_before,
        "logSha256": digest(log),
        "rawRoot": str(raw),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if actual != args.expect:
        raise RuntimeError(f"five-species gate was {actual}, expected {args.expect}")
    print(json.dumps({"status": actual, "returnCode": completed.returncode,
                      "simulatorSha256": simulator_before}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

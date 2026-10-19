#!/usr/bin/env python3
"""Seal post-observer real-device Milbrandt warm-stage regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/microphysics/schemes/milbrandt/warm_rain.hpp",
    "tests/physics/test_native_milbrandt_warm.cpp",
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
    before = {name: digest(ROOT / name) for name in SOURCES}
    test_before = digest(binary)
    simulator_before = digest(simulator)
    start = time.monotonic()
    environment = os.environ.copy()
    environment["NIMBUS_GPU_REQUIRED"] = "1"
    try:
        completed = subprocess.run(
            [str(binary)], cwd=ROOT, capture_output=True, text=True,
            env=environment, timeout=120, check=False)
    except subprocess.TimeoutExpired as error:
        (raw / "native_warm.log").write_text(str(error), encoding="utf-8")
        raise RuntimeError("native warm regression exceeded 120-second cap") from error
    log = raw / "native_warm.log"
    content = completed.stdout + completed.stderr
    log.write_text(content, encoding="utf-8")
    after = {name: digest(ROOT / name) for name in SOURCES}
    if before != after or test_before != digest(binary) or simulator_before != digest(simulator):
        raise RuntimeError("native warm source or binary drifted")
    if completed.returncode != 0 or \
            "All tests passed (17424 assertions in 2 test cases)" not in content or \
            "backend=metal, requested=metal" not in content or \
            "backend=vulkan, requested=vulkan" not in content or \
            content.count("fallback=off, fallback_allowed=off") < 2:
        raise RuntimeError("native warm real-device regression failed; preserve raw log")
    report = {
        "schema": "nimbus.milbrandt-warm-post-observer-native.v1",
        "status": "pass",
        "claimBoundary": "Existing double-moment Metal/Vulkan warm-stage parity after optional CPU observer source edit; triple Z remains unintegrated.",
        "assertions": 17424,
        "testCases": 2,
        "wallSeconds": time.monotonic() - start,
        "testBinarySha256": test_before,
        "simulatorSha256": simulator_before,
        "sourceSha256": before,
        "runnerSha256": digest(Path(__file__)),
        "logSha256": digest(log),
        "rawRoot": str(raw),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "assertions": 17424,
                      "simulatorSha256": simulator_before}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

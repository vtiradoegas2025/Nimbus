#!/usr/bin/env python3
"""Seal the bounded default-off hail q/N/Z sedimentation component gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
INPUTS = (
    "src/microphysics/schemes/milbrandt/triple_moment.hpp",
    "src/microphysics/schemes/milbrandt/triple_moment.cpp",
    "src/microphysics/schemes/milbrandt/triple_sedimentation.hpp",
    "src/microphysics/schemes/milbrandt/triple_sedimentation.cpp",
    "src/microphysics/schemes/milbrandt/sedimentation.hpp",
    "tests/physics/test_milbrandt_triple_sedimentation.cpp",
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
    before = {relative: digest(ROOT / relative) for relative in INPUTS}
    simulator_before = digest(simulator)
    binary_before = digest(binary)
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [str(binary)], cwd=ROOT, capture_output=True, text=True,
            timeout=20, check=False)
    except subprocess.TimeoutExpired as error:
        (raw / "focused.log").write_text(str(error), encoding="utf-8")
        raise RuntimeError("focused hail gate exceeded its 20-second cap") from error
    log = raw / "focused.log"
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    after = {relative: digest(ROOT / relative) for relative in INPUTS}
    if before != after or simulator_before != digest(simulator) or binary_before != digest(binary):
        raise RuntimeError("hail gate source or binary drifted")
    if completed.returncode != 0 or "All tests passed (327 assertions in 4 test cases)" not in log.read_text():
        raise RuntimeError("focused hail gate failed; preserve raw log")
    report = {
        "schema": "nimbus.milbrandt-triple-hail-sedimentation.v1",
        "status": "pass",
        "claimBoundary": "Default-off isolated CPU hail column only; no MilbrandtScheme/runtime, transport, checkpoint or native integration.",
        "assertions": 327,
        "testCases": 4,
        "wallSeconds": time.monotonic() - start,
        "testBinarySha256": binary_before,
        "simulatorSha256": simulator_before,
        "runnerSha256": digest(Path(__file__)),
        "sourceSha256": before,
        "logSha256": digest(log),
        "rawRoot": str(raw),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "assertions": 327,
                      "simulatorSha256": simulator_before}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

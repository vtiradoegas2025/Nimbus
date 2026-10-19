#!/usr/bin/env python3
"""Seal the default-off hail q/N/Z gate and a bounded accepted work-cap edge."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
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


def execute(binary: Path, args: list[str], log: Path, expected: str) -> dict[str, object]:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [str(binary), *args], cwd=ROOT, capture_output=True,
            text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as error:
        log.write_text(str(error), encoding="utf-8")
        raise RuntimeError(f"hail gate {args} exceeded 20-second cap") from error
    content = completed.stdout + completed.stderr
    log.write_text(content, encoding="utf-8")
    if completed.returncode != 0 or expected not in content:
        raise RuntimeError(f"hail gate {args} failed; preserve raw log")
    return {
        "args": args,
        "wallSeconds": time.monotonic() - start,
        "logSha256": digest(log),
        "returnCode": completed.returncode,
    }


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
    edge = execute(binary, ["[cap_edge]"], raw / "cap_edge.log",
                   "All tests passed (21 assertions in 1 test case)")
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    edge["peakRssBytes"] = int(rss if platform.system() == "Darwin" else rss * 1024)
    full = execute(binary, [], raw / "focused.log",
                   "All tests passed (352 assertions in 5 test cases)")
    after = {relative: digest(ROOT / relative) for relative in INPUTS}
    if before != after or simulator_before != digest(simulator) or binary_before != digest(binary):
        raise RuntimeError("hail gate source or binary drifted")
    report = {
        "schema": "nimbus.milbrandt-triple-hail-sedimentation.v2",
        "status": "pass",
        "claimBoundary": "Default-off isolated CPU hail column only; no MilbrandtScheme/runtime, transport, checkpoint or native integration.",
        "assertions": 352,
        "testCases": 5,
        "acceptedEdge": {"levels": 16, "maximumSubsteps": 4096,
                         "maximumCellSubsteps": 65536, "durationSeconds": 50,
                         **edge},
        "fullGate": full,
        "testBinarySha256": binary_before,
        "simulatorSha256": simulator_before,
        "runnerSha256": digest(Path(__file__)),
        "sourceSha256": before,
        "rawRoot": str(raw),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "assertions": 352,
                      "peakRssBytes": edge["peakRssBytes"],
                      "simulatorSha256": simulator_before}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

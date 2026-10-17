#!/usr/bin/env python3
"""Seal focused and full Stage-1 post-primary hook runtime-state gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "include/microphysics/microphysics_base.hpp",
    "src/core/orchestration/physics/microphysics_step.cpp",
    "tests/physics/test_microphysics_runtime_state.cpp",
    "src/microphysics/grid_velocity.cpp",
    "src/terrain/base/conservative_transport.cpp",
    "mk/tests.mk",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(binary: Path, args: list[str], log: Path,
            expected: str) -> dict[str, object]:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [str(binary), *args], cwd=ROOT, capture_output=True,
            text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as error:
        log.write_text(str(error), encoding="utf-8")
        raise RuntimeError("post-primary focused test exceeded 20-second cap") from error
    content = completed.stdout + completed.stderr
    log.write_text(content, encoding="utf-8")
    if completed.returncode != 0 or expected not in content:
        raise RuntimeError(f"post-primary gate {args} failed; preserve raw log")
    return {
        "args": args,
        "returnCode": completed.returncode,
        "wallSeconds": time.monotonic() - start,
        "logSha256": digest(log),
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
    source_before = {name: digest(ROOT / name) for name in SOURCES}
    binary_before = digest(binary)
    simulator_before = digest(simulator)
    focused = execute(
        binary, ["[post_primary]"], raw / "focused.log",
        "All tests passed (33 assertions in 2 test cases)")
    full = execute(
        binary, [], raw / "full.log",
        "All tests passed (262 assertions in 37 test cases)")
    source_after = {name: digest(ROOT / name) for name in SOURCES}
    if source_before != source_after or binary_before != digest(binary) or \
            simulator_before != digest(simulator):
        raise RuntimeError("post-primary source or binary drifted")
    report = {
        "schema": "nimbus.milbrandt-post-primary-stage1.v1",
        "status": "pass",
        "claimBoundary": "Default-no-op base hook and fake-scheme post-primary rollback/commit tests; no triple Z owner or relinked simulator claim.",
        "sourceSha256": source_before,
        "testBinarySha256": binary_before,
        "simulatorSha256": simulator_before,
        "runnerSha256": digest(Path(__file__)),
        "focused": focused,
        "full": full,
        "rawRoot": str(raw),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "focusedAssertions": 33,
                      "fullAssertions": 262,
                      "simulatorSha256": simulator_before}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

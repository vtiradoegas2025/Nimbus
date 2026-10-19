#!/usr/bin/env python3
"""Seal double-moment native cold preservation after optional event tracing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/microphysics/schemes/milbrandt/cold.hpp",
    "src/microphysics/schemes/milbrandt/cold_rain_freezing_events.hpp",
    "src/microphysics/schemes/milbrandt/milbrandt.cpp",
    "include/microphysics/milbrandt_cold_parameters.hpp",
    "tests/physics/test_native_milbrandt_cold.cpp",
    "metal/include/compute/metal/shaders/microphysics/milbrandt_cold_msl.hpp",
    "vulkan/shaders/compute/microphysics/milbrandt_cold.comp",
    "vulkan/shaders/compute/microphysics/milbrandt_cold.comp.spv",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--test-binary", type=Path, required=True)
    parser.add_argument("--expected-simulator-sha256", required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    report = args.report_root.resolve()
    simulator = args.simulator.resolve(strict=True)
    binary = args.test_binary.resolve(strict=True)
    if report.exists() or sha(simulator) != args.expected_simulator_sha256:
        parser.error("report must be new and simulator identity must match")
    paths = {path: ROOT / path for path in SOURCES}
    paths.update(simulator=simulator, testBinary=binary, runner=Path(__file__).resolve())
    before = {key: sha(path) for key, path in paths.items()}
    report.mkdir(parents=True)
    environment = dict(os.environ, NIMBUS_GPU_REQUIRED="1")
    start = time.monotonic()
    run = subprocess.run([str(binary)], cwd=ROOT, env=environment, text=True,
                         capture_output=True, timeout=120, check=False)
    output = run.stdout + run.stderr
    log = report / "native_cold.log"
    log.write_text(output)
    after = {key: sha(path) for key, path in paths.items()}
    footer = re.search(r"All tests passed \((\d+) assertions in 2 test cases\)", output)
    passed = (run.returncode == 0 and before == after and footer is not None and
              "backend=metal, requested=metal" in output and
              "backend=vulkan, requested=vulkan" in output and
              output.count("fallback=off, fallback_allowed=off") >= 2)
    summary = {
        "schema": "nimbus.milbrandt-cold-post-observer-native.v1",
        "status": "pass" if passed else "red", "exitCode": run.returncode,
        "before": before, "after": after, "logSha256": sha(log),
        "assertions": int(footer.group(1)) if footer else None,
        "testCases": 2 if footer else None, "timeoutSeconds": 120,
        "elapsedSeconds": time.monotonic() - start,
        "claimBoundary": "Preserved double-moment CPU/Metal/Vulkan cold rates; full cold Z coupling and independent scientific review remain unfinished.",
    }
    (report / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "assertions", "testCases", "exitCode")}))
    if not passed:
        raise RuntimeError("native cold preservation gate failed; preserve raw evidence")


if __name__ == "__main__":
    main()

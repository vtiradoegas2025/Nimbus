#!/usr/bin/env python3
"""Run the bounded static vertical-only CPU late-step failure probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
TEST_SOURCE = ROOT / "tests/core/test_nested_production_step.cpp"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-binary", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--qualified-public", action="store_true")
    args = parser.parse_args()
    binary = args.test_binary.resolve()
    simulator = args.simulator.resolve()
    config = args.config.resolve()
    raw = args.raw_root.resolve()
    report = args.report_output.resolve()
    if not all(path.is_file() for path in (binary, simulator, config)) or \
            raw.exists() or report.exists():
        parser.error("test binary, simulator and config must exist; outputs must be new")
    report.parent.mkdir(parents=True, exist_ok=True)
    alias = config
    if not args.qualified_public:
        source = config.read_text(encoding="utf-8")
        if source.count("    mode: qualified\n") != 1:
            raise RuntimeError("exact candidate lacks one qualified execution selector")
        alias = report.parent / "exploratory_static_vertical.yaml"
        if alias.exists():
            parser.error("exploratory alias target must be new")
        alias.write_text(source.replace("    mode: qualified\n",
                                        "    mode: exploratory\n", 1),
                         encoding="utf-8")
    identity = {"testBinarySha256": sha256(binary),
                "simulatorSha256": sha256(simulator),
                "candidateConfigSha256": sha256(config),
                "exploratoryAliasSha256": sha256(alias),
                "testSourceSha256": sha256(TEST_SOURCE),
                "runnerSha256": sha256(Path(__file__))}
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["NIMBUS_TEST_VERTICAL_ONLY_STATIC_CONFIG"] = str(alias)
    environment["NIMBUS_TEST_VERTICAL_ONLY_FAILURE_ROOT"] = str(raw)
    started = time.monotonic()
    cases = []
    for name, selected in (
        ("direct", "[turbulence][vertical_only][cpu][direct_late_failure]"),
        ("headless", "[turbulence][vertical_only][cpu][late_failure]"),
    ):
        try:
            completed = subprocess.run(
                [str(binary), selected], cwd=ROOT, env=environment,
                capture_output=True, text=True, timeout=60, check=False,
            )
            code = completed.returncode
            message = completed.stdout + "\n" + completed.stderr
        except subprocess.TimeoutExpired as error:
            code = 124
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            message = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
        log = report.parent / f"{name}.log"
        log.write_text(message, encoding="utf-8")
        post_sgs_fault = (name == "direct" and code == 0 and
                          "All tests passed" in message) or \
            "active diffusion configuration has no initialized scheme" in message
        cases.append({"name": name, "returnCode": code,
                      "logSha256": sha256(log),
                      "postSgsFaultReached": post_sgs_fault})
        if code != 0 or not cases[-1]["postSgsFaultReached"]:
            break
    after = {"testBinarySha256": sha256(binary),
             "simulatorSha256": sha256(simulator),
             "candidateConfigSha256": sha256(config),
             "exploratoryAliasSha256": sha256(alias),
             "testSourceSha256": sha256(TEST_SOURCE),
             "runnerSha256": sha256(Path(__file__))}
    files = [path for path in raw.rglob("*") if path.is_file()] if raw.exists() else []
    bytes_used = sum(path.stat().st_size for path in files)
    status = "pass" if len(cases) == 2 and all(
        row["returnCode"] == 0 and row["postSgsFaultReached"] for row in cases
    ) else "late_step_failure_red"
    if time.monotonic() - started >= 60 or len(files) >= 2000 or \
            bytes_used >= 64 * 1024 * 1024:
        status = "resource_cap"
    if after != identity:
        status = "source_identity_drift"
    summary = {"schema": "nimbus.vertical-only-static-cpu-late-failure.v1",
               "status": status,
               "claimBoundary": ("Ordinary qualified exact static flat-CPU source exercises direct/headless post-SGS whole-step rollback; broader Ready remains separate."
                                 if args.qualified_public else
                                 "One test-only exploratory alias of the exact static flat-CPU source exercises the post-SGS whole-step transaction while qualified public admission stays guarded; Ready remains separate."),
               **identity,
               "qualifiedPublic": args.qualified_public,
               "cases": cases,
               "wallSeconds": time.monotonic() - started,
               "rawFiles": len(files),
               "rawBytes": bytes_used,
               "rawRoot": str(raw)}
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"{status}: {len(cases)}/2 direct/headless, {len(files)} files/{bytes_used} bytes")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

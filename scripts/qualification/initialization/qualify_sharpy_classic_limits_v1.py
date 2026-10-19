#!/usr/bin/env python3
"""Capture focused and ordinary classic CDF shared-limit evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "tests/init/fixtures/validate_only_cm1_sounding.yaml"
KEYS = {
    "fixed_100000": (True, "[VALIDATE] PASS"),
    "fixed_100001": (False, "100,000-level safety limit"),
    "record_33": (True, "[VALIDATE] PASS"),
    "record_100001": (False, "100,000-level safety limit"),
    "optional_100001": (False, "100,000-level safety limit"),
    "station_1023": (True, "[VALIDATE] PASS"),
    "station_1024": (False, "1,023-byte safety limit"),
    "timestamp_1023": (True, "[VALIDATE] PASS"),
    "timestamp_1024": (False, "1,023-byte safety limit"),
    "latitude_scalar": (True, "[VALIDATE] PASS"),
    "latitude_array": (False, "must be scalar numeric"),
}
SOURCES = {
    "reader": ROOT / "src/soundings/schemes/sharpy/sharpy_native_reader.cpp",
    "routing": ROOT / "src/soundings/schemes/sharpy/sharpy_sounding.cpp",
    "interface": ROOT / "src/soundings/schemes/sharpy/sharpy_native_reader.hpp",
    "limits": ROOT / "include/soundings/sounding_input_limits.hpp",
    "tests": ROOT / "tests/data/test_soundings.cpp",
    "fixtureWriter": ROOT / "scripts/qualification/initialization/write_sharpy_classic_limits_fixtures_v1.py",
    "qualifier": Path(__file__).resolve(), "baseConfig": BASE,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("focused", "public"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--expect-red", action="store_true")
    parser.add_argument("--full-soundings", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    report = args.report_root.resolve()
    if report.exists():
        parser.error("preserve existing report; use a new root")
    if sha(binary) != args.expected_binary_sha256:
        parser.error("binary differs from coordinated identity")
    fixtures = {
        f"{key}_cdf{version}": ROOT / f"tests/init/fixtures/sharpy_classic_limits_{key}_cdf{version}.nc"
        for version in (1, 2) for key in KEYS
    }
    tracked = {"binary": binary, **SOURCES, **fixtures}
    before = {key: sha(path) for key, path in tracked.items()}
    report.mkdir(parents=True)
    cases = []

    def run(name: str, command: list[str], accepted: bool, diagnostic: str,
            outdir: Path | None = None) -> None:
        start = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                timeout=30 if args.mode == "focused" else 20, check=False)
        output = result.stdout + result.stderr
        log = report / f"{name}.log"
        log.write_text(output)
        published = [path.relative_to(report).as_posix() for path in outdir.rglob("*")
                     if path.is_file()] if outdir else []
        passed = ((result.returncode == 0) == accepted and diagnostic in output and not published)
        if accepted and args.mode == "public":
            passed = passed and "Loaded sounding via native backend: netcdf_classic" in output
        cases.append({"name": name, "command": command, "exitCode": result.returncode,
                      "expectedAccepted": accepted, "expectedDiagnostic": diagnostic,
                      "passed": passed, "elapsedSeconds": time.monotonic() - start,
                      "logSha256": sha(log), "publishedFiles": published})

    if args.mode == "focused":
        run("focused", [str(binary), "[classic][limits]"], True, "All tests passed")
        if args.full_soundings:
            run("full_soundings", [str(binary)], True, "All tests passed")
    else:
        inputs = report / "inputs"
        inputs.mkdir()
        base = BASE.read_text()
        marker = "file_path: tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE"
        if base.count(marker) != 1:
            raise RuntimeError("base YAML sounding path changed")
        for version in (1, 2):
            for key, (accepted, diagnostic) in KEYS.items():
                name = f"{key}_cdf{version}"
                fixture = fixtures[name]
                config = inputs / f"{name}.yaml"
                config.write_text(base.replace(marker, f"file_path: {json.dumps(str(fixture))}"))
                outdir = report / "runs" / name
                run(name, [str(binary), "--headless", "--validate-only", "--config", str(config),
                           f"--outdir={outdir}"], accepted, diagnostic, outdir)
                cases[-1]["configSha256"] = sha(config)
    after = {key: sha(path) for key, path in tracked.items()}
    if before != after:
        raise RuntimeError("source, binary, fixture or runner changed during gate")
    status = "pass" if all(case["passed"] for case in cases) else "red"
    summary = {"schema": "nimbus.sharpy-classic-shared-limits.v1", "status": status,
               "mode": args.mode, "before": before, "after": after, "cases": cases,
               "claimBoundary": "Classic CDF selected-profile and metadata bounds; broad metadata budgets remain open."}
    (report / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": status, "mode": args.mode, "cases": len(cases),
                      "binarySha256": before["binary"]}, sort_keys=True))
    if (status == "red") != args.expect_red:
        raise RuntimeError("gate result differs from expected outcome; preserve raw evidence")


if __name__ == "__main__":
    main()

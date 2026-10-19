#!/usr/bin/env python3
"""Bound ordinary SHARPY NetCDF4 numeric profile admission and pre-read rejection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
BASE_YAML = ROOT / "tests/init/fixtures/validate_only_cm1_sounding.yaml"
SOURCES = {
    "netcdfReader": ROOT / "src/soundings/schemes/sharpy/sharpy_netcdf_dynamic_reader.cpp",
    "typedInterface": ROOT / "src/soundings/schemes/sharpy/sharpy_native_reader.hpp",
    "routing": ROOT / "src/soundings/schemes/sharpy/sharpy_sounding.cpp",
    "levelLimit": ROOT / "include/soundings/sounding_input_limits.hpp",
    "baseYaml": BASE_YAML,
    "safeNetcdf": ROOT / "tests/init/fixtures/sharpy_netcdf_public_safe.nc",
    "oversizedNetcdf": ROOT / "tests/init/fixtures/sharpy_netcdf_numeric_oversized_100001.nc",
    "qualifier": Path(__file__),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    raw = args.raw_root.resolve()
    report_path = args.report_output.resolve()
    if raw.exists() or report_path.exists():
        parser.error("raw root and report must be new paths")
    binary_sha = sha256(binary)
    if binary_sha != args.expected_binary_sha256:
        parser.error("simulator differs from exact coordinated build")
    source_before = {name: sha256(path) for name, path in SOURCES.items()}
    raw.mkdir(parents=True)
    inputs = raw / "inputs"
    inputs.mkdir()
    yaml = BASE_YAML.read_text(encoding="utf-8")
    marker = "file_path: tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE"
    if yaml.count(marker) != 1:
        raise RuntimeError("reviewed CM1 baseline YAML changed")
    cases = [
        ("safe_33_levels", SOURCES["safeNetcdf"], True, "[VALIDATE] PASS"),
        ("oversized_100001_levels", SOURCES["oversizedNetcdf"], False,
         "NetCDF numeric profile exceeds the 100,000-level safety limit"),
    ]
    results = []
    status = "pass"
    for name, sounding, accepted, expected in cases:
        config = inputs / f"{name}.yaml"
        config.write_text(yaml.replace(marker, f'file_path: "{sounding}"'), encoding="utf-8")
        outdir = raw / "runs" / name
        command = [str(binary), "--headless", "--validate-only", "--config", str(config),
                   f"--outdir={outdir}"]
        try:
            run = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=20, check=False)
            output = run.stdout
            exit_code = run.returncode
        except subprocess.TimeoutExpired as error:
            output = str(error)
            exit_code = -1
        log = raw / f"{name}.log"
        log.write_text(output, encoding="utf-8")
        published = [path.relative_to(raw).as_posix() for path in outdir.rglob("*") if path.is_file()]
        results.append({
            "case": name, "config": config.relative_to(ROOT).as_posix(),
            "configSha256": sha256(config), "soundingSha256": sha256(sounding),
            "command": command, "expectedAccepted": accepted,
            "expectedDiagnostic": expected, "exitCode": exit_code,
            "logSha256": sha256(log), "publishedFiles": published,
        })
        if ((exit_code == 0) != accepted or expected not in output or published or
                sha256(binary) != binary_sha or
                any(sha256(path) != source_before[key] for key, path in SOURCES.items())):
            status = "red"
            break
    report = {
        "schema": "nimbus.sharpy-netcdf-numeric-volume-public-guard.v1",
        "status": status,
        "claimBoundary": "One 33-level ordinary NetCDF4 control and a compact 100001-level dynamic-reader rejection; classic NetCDF and global attributes are separate owners.",
        "binarySha256Before": binary_sha,
        "binarySha256After": sha256(binary),
        "sourceSha256Before": source_before,
        "sourceSha256After": {name: sha256(path) for name, path in SOURCES.items()},
        "cases": results,
        "measuredPeakMemory": None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "pass":
        raise RuntimeError(f"SHARPY NetCDF numeric gate stopped red at {results[-1]['case']}")
    print(json.dumps({"status": status, "binarySha256": binary_sha,
                      "caseCount": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

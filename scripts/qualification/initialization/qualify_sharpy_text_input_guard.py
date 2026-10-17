#!/usr/bin/env python3
"""Bounded ordinary strict-YAML SHARPY/CM1 sounding admission and failure gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
POSITIVE_YAML = ROOT / "tests/init/fixtures/validate_only_file_sounding.yaml"
CM1_YAML = ROOT / "tests/init/fixtures/validate_only_cm1_sounding.yaml"
SHARPY_TEXT = ROOT / "tests/init/fixtures/test_supercell.tmv"
SOURCES = {
    "limits": ROOT / "include/soundings/sounding_input_limits.hpp",
    "sharpyReader": ROOT / "src/soundings/schemes/sharpy/sharpy_profile_parse.cpp",
    "cm1Reader": ROOT / "src/soundings/schemes/cm1/cm1_sounding.cpp",
    "dataTests": ROOT / "tests/data/test_soundings.cpp",
    "initializationTests": ROOT / "tests/init/test_file_sounding.cpp",
    "sharpyFixture": SHARPY_TEXT,
    "sharpyYaml": POSITIVE_YAML,
    "cm1Yaml": CM1_YAML,
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
    if sha256(binary) != args.expected_binary_sha256:
        parser.error("simulator binary differs from the reviewed build")
    source_before = {name: sha256(path) for name, path in SOURCES.items()}
    raw.mkdir(parents=True)
    inputs = raw / "inputs"
    inputs.mkdir()
    base = SHARPY_TEXT.read_bytes()
    yaml = CM1_YAML.read_text(encoding="utf-8")
    marker = "file_path: tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE"
    if base.count(b"levels\t33") != 1 or yaml.count(marker) != 1:
        raise RuntimeError("reviewed SHARPY fixture contract changed")

    positive_sharpy = inputs / "sharpy_positive.yaml"
    positive_sharpy.write_text(
        yaml.replace(marker, "file_path: tests/init/fixtures/test_supercell.tmv"),
        encoding="utf-8")

    mutations = [
        ("levels_over_cap", base.replace(b"levels\t33", b"levels\t100001"),
         "100,000-level safety limit"),
        ("levels_overflow", base.replace(b"levels\t33", b"levels\t184467440737095516160000"),
         "invalid levels header"),
        ("levels_trailing_junk", base.replace(b"levels\t33", b"levels\t33junk"),
         "invalid levels header"),
        ("duplicate_levels", base + b"levels\t33\n", "duplicate levels header"),
        ("overlong_line", base + b"#" + b"x" * 4097 + b"\n", "4096-byte line safety limit"),
        ("truncated_levels", base.replace(b"levels\t33", b"levels\t34"),
         "does not match declared levels"),
    ]
    cases = [
        ("sharpy_positive", positive_sharpy, True, "[VALIDATE] PASS"),
        ("cm1_positive", CM1_YAML, True, "[VALIDATE] PASS"),
    ]
    for name, data, diagnostic in mutations:
        sounding = inputs / f"{name}.tmv"
        sounding.write_bytes(data)
        config = inputs / f"{name}.yaml"
        config.write_text(yaml.replace(marker, f'file_path: "{sounding}"'), encoding="utf-8")
        cases.append((name, config, False, diagnostic))
    oversized = inputs / "oversized_file.tmv"
    oversized.write_bytes(base)
    with oversized.open("r+b") as handle:
        handle.truncate(8 * 1024 * 1024 + 1)
    oversized_yaml = inputs / "oversized_file.yaml"
    oversized_yaml.write_text(yaml.replace(marker, f'file_path: "{oversized}"'), encoding="utf-8")
    cases.append(("oversized_file", oversized_yaml, False, "8 MiB safety limit"))

    results = []
    status = "pass"
    for name, config, accepted, expected in cases:
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
            "configSha256": sha256(config), "command": command,
            "expectedAccepted": accepted, "exitCode": exit_code,
            "expectedDiagnostic": expected, "logSha256": sha256(log),
            "publishedFiles": published,
        })
        if ((exit_code == 0) != accepted or expected not in output or published or
                sha256(binary) != args.expected_binary_sha256 or
                any(sha256(path) != source_before[key] for key, path in SOURCES.items())):
            status = "red"
            break
    report = {
        "schema": "nimbus.sharpy-text-input-guard.v1",
        "status": status,
        "claimBoundary": "One strict SHARPY text fixture, one CM1 control and malformed-input neighbors; not full initialization readiness.",
        "binarySha256Before": args.expected_binary_sha256,
        "binarySha256After": sha256(binary),
        "sourceSha256Before": source_before,
        "sourceSha256After": {name: sha256(path) for name, path in SOURCES.items()},
        "cases": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "pass":
        raise RuntimeError(f"SHARPY strict input gate stopped red at {results[-1]['case']}")
    print(json.dumps({"status": status, "binarySha256": args.expected_binary_sha256,
                      "caseCount": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

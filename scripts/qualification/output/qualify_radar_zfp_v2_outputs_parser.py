#!/usr/bin/env python3
"""Bind Studio's mixed ZFP v2 manifest parser tests to frozen native ownership."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
RUST_MANIFEST = ROOT / "ui/src-tauri/Cargo.toml"
PARSER = ROOT / "ui/src-tauri/src/commands/outputs.rs"
NATIVE_CONTRACT = ROOT / "include/core/output/step_manifest.hpp"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo-target-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    if args.report_output.exists():
        parser.error("use a new report path")
    raw = args.raw_root.resolve()
    raw.mkdir(parents=True, exist_ok=False)
    binary = ROOT / "bin/nimbus_sim"
    source_before = {"simulator": sha256(binary), "parser": sha256(PARSER),
                     "nativeManifestOwner": sha256(NATIVE_CONTRACT)}
    catalog_bytes = subprocess.check_output([str(binary), "--scheme-catalog=json"], cwd=ROOT)
    catalog = json.loads(catalog_bytes)
    readiness = {item["id"]: item for item in catalog["readiness"]["items"]}
    if readiness["output.zfp_bounded_history"]["state"] != "engineering_incomplete":
        raise RuntimeError("public radar ZFP guard was lifted before native repair")
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(args.cargo_target_dir.resolve())
    cargo = ["cargo", "test", "--locked", "--manifest-path", str(RUST_MANIFEST)]
    cases = [
        ("mixed_v2", cargo + ["mixed_zfp_v2", "--lib"], "2 passed; 0 failed"),
        ("outputs", cargo + ["commands::outputs", "--lib"], "64 passed; 0 failed; 5 ignored"),
        ("lib_serial", cargo + ["--lib", "--", "--test-threads=1"],
         "227 passed; 0 failed; 11 ignored"),
    ]
    results = []
    for label, command, expected in cases:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                text=True, capture_output=True, timeout=120)
        log = raw / f"{label}.log"
        log.write_text(result.stdout + "\n" + result.stderr)
        if result.returncode != 0 or expected not in result.stdout:
            raise RuntimeError(f"{label} failed or test count changed: {log}")
        results.append({"case": label, "expectedSummary": expected,
                        "exitCode": result.returncode, "logPath": str(log),
                        "logSha256": sha256(log)})
    if (sha256(binary) != source_before["simulator"] or
            sha256(PARSER) != source_before["parser"] or
            sha256(NATIVE_CONTRACT) != source_before["nativeManifestOwner"]):
        raise RuntimeError("simulator or parser/contract source changed during the gate")
    report = {
        "schema": "nimbus.r4-radar-zfp-studio-mixed-manifest-v2.v1",
        "status": "pass_strict_mixed_v2_parser_only",
        "claimBoundary": "Strict Studio mixed v2 product/field execution provenance and exact v4 radar/v3 non-radar header metadata pass focused, outputs and serial full-lib tests. The native writer and reader, public qualified radar admission, campaigns and scientific review remain separate. A prior parallel full-lib attempt had one isolated netCDF dataset test failure; that test passed alone and the serial full-lib rerun passed.",
        "currentSimulatorSha256": source_before["simulator"],
        "currentCatalogSha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "parserSourceSha256": source_before["parser"],
        "nativeManifestOwnerSha256": source_before["nativeManifestOwner"],
        "qualifierSha256": sha256(Path(__file__)),
        "rawRoot": str(raw),
        "tests": results,
    }
    output = args.report_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(results),
                      "simulatorSha256": source_before["simulator"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

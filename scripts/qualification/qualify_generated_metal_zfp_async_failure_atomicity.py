#!/usr/bin/env python3
"""Qualify scoped async ZFP writer failure against a normal public run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
NPY = ROOT / "tests/fixtures/reference/catalog/generated_metal_canonical_v1.yaml"
LEDGERS = (
    "budget.csv", "energy_budget.csv", "moisture_budget.csv",
    "momentum_budget.csv", "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
RUST_MANIFEST = ROOT / "ui/src-tauri/Cargo.toml"
READER_TEST = "qualified_simulator_zfp_matches_npy_reference"


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def execute(binary: Path, config: Path, output: Path,
            inject: bool, checkpoint_every: int) -> dict:
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE"}
    if inject:
        environment["NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST"] = "1"
    command = [str(binary), "--headless", "--config", str(config),
               "--duration=6", "--write-every=1",
               f"--checkpoint-every={checkpoint_every}",
               f"--outdir={output}", "--threads=1", "--log-profile=normal"]
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=180, check=False)
    elapsed = time.monotonic() - started
    log = output.with_suffix(".log")
    log.write_text(result.stdout)
    return {"returncode": result.returncode, "wallSeconds": elapsed,
            "log": identity(log), "injected": inject,
            "invalidScience": "[SCIENTIFIC INVALIDITY]" in result.stdout,
            "fallbackOff": "fallback=off" in result.stdout}


def ledger_rows(directory: Path) -> dict:
    result = {}
    for name in LEDGERS:
        path = directory / name
        require(path.is_file(), f"missing expected science ledger: {path}")
        rows = path.read_text().splitlines()
        result[name] = {**identity(path), "nonHeaderRows": max(0, len(rows) - 1)}
    return result


def committed_steps(directory: Path) -> dict:
    result = {}
    for step in sorted(directory.glob("step_*")):
        if not step.is_dir():
            continue
        manifest = step / "manifest.json"
        result[step.name] = {
            "manifestCommitted": manifest.is_file(),
            "manifest": identity(manifest) if manifest.is_file() else None,
            "zfpPayloads": sum(path.is_file() for path in step.glob("*.zfp3d")),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rust-target", type=Path,
                        default=Path("/private/tmp/nimbus-catalog-rust-20260927"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-binary-sha", required=True)
    parser.add_argument("--expected-baseline-sha", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    baseline = args.baseline_binary.resolve(strict=True)
    config = args.config.resolve(strict=True)
    inputs = {name: identity(path) for name, path in
              (("binary", binary), ("baselineBinary", baseline),
               ("qualifiedZfpConfig", config), ("npyConfig", NPY),
               ("headlessSource", ROOT / "src/core/runtime/headless_runtime.cpp"),
               ("writerSource", ROOT / "src/core/output/output_writer.cpp"),
               ("indexSource", ROOT / "src/core/output/history_index.cpp"),
               ("readerSource", ROOT / "ui/src-tauri/src/zfp3d.rs"),
               ("checker", Path(__file__)))}
    require(inputs["binary"]["sha256"] == args.expected_binary_sha and
            inputs["baselineBinary"]["sha256"] == args.expected_baseline_sha,
            "async failure gate binary identities changed")

    failed_dir = output / "injected-failure"
    failure = execute(binary, config, failed_dir, True, 1)
    failed_log = Path(failure["log"]["path"]).read_text()
    failed_steps = committed_steps(failed_dir)
    failed_ledgers = ledger_rows(failed_dir)
    failed_contract_path = failed_dir / "nimbus_output.json"
    failed_contract = json.loads(failed_contract_path.read_text())
    first_manifest_path = failed_dir / "step_000000/manifest.json"
    first_manifest = json.loads(first_manifest_path.read_text()) if first_manifest_path.is_file() else None
    pending = failed_dir / "history_index.pending.json"
    head = failed_dir / "history_index.json"
    checkpoints = list((failed_dir / "checkpoints").glob("*")) \
        if (failed_dir / "checkpoints").is_dir() else []
    require(failure["returncode"] != 0 and not failure["invalidScience"] and
            failure["fallbackOff"] and
            "injected post-manifest history-index failure" in failed_log and
            set(failed_steps) == {"step_000000"} and
            failed_steps["step_000000"]["manifestCommitted"] and
            failed_steps["step_000000"]["zfpPayloads"] == 8 and
            pending.is_file() and not head.exists() and not checkpoints and
            all(value["nonHeaderRows"] == 0 for value in failed_ledgers.values()) and
            failed_contract.get("publication_state") == "failed" and
            first_manifest is not None and
            first_manifest.get("publication_generation") ==
                failed_contract.get("publication_generation"),
            "qualified ZFP async failure did not stop science/ledger/checkpoint or fail run metadata")

    positive_dir = output / "qualified-positive"
    positive = execute(binary, config, positive_dir, False, 1)
    baseline_dir = output / "baseline-positive"
    prior = execute(baseline, config, baseline_dir, False, 1)
    npy_dir = output / "qualified-npy"
    npy = execute(binary, NPY, npy_dir, False, 0)
    positive_contract = json.loads((positive_dir / "nimbus_output.json").read_text())
    positive_validation = json.loads((positive_dir / "validation_summary.json").read_text())
    require(all(run["returncode"] == 0 and run["fallbackOff"] and
                not run["invalidScience"] for run in (positive, prior, npy)) and
            positive_contract.get("publication_state") == "ready" and
            positive_validation["failed"] is False and
            positive_validation["total_violations"] == 0 and
            set(committed_steps(positive_dir)) ==
                {"step_000004", "step_000005", "step_000006"},
            "normal qualified ZFP/NPY or old-binary reference failed")
    payloads = sorted(path.relative_to(positive_dir) for path in
                      positive_dir.glob("step_*/*.zfp3d"))
    require(len(payloads) == 24 and all(
        (positive_dir / path).read_bytes() == (baseline_dir / path).read_bytes()
        for path in payloads) and all(
        (positive_dir / name).read_bytes() == (baseline_dir / name).read_bytes()
        for name in LEDGERS),
        "qualified ZFP writer barrier changed completed science or compressed fields")
    environment = {**os.environ, "CARGO_TARGET_DIR": str(args.rust_target.resolve()),
                   "NIMBUS_TEST_ZFP_RUN": str(positive_dir),
                   "NIMBUS_TEST_NPY_RUN": str(npy_dir)}
    reader = subprocess.run(["cargo", "test", "--manifest-path", str(RUST_MANIFEST),
                             READER_TEST, "--lib", "--", "--ignored"],
                            cwd=ROOT, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=180, check=False)
    reader_log = output / "studio-reader.log"
    reader_log.write_text(reader.stdout)
    require(reader.returncode == 0 and
            f"test zfp3d::tests::{READER_TEST} ... ok" in reader.stdout and
            "1 passed; 0 failed" in reader.stdout,
            "qualified ZFP reader failed after the async publication barrier")
    require(all(identity(Path(value["path"])) == value for value in inputs.values()),
            "async ZFP failure gate input changed during execution")
    report = {
        "schema": "nimbus.generated-metal-zfp-qualified-async-failure.v1",
        "status": "pass_bounded_failure_and_positive",
        "readiness": "Engineering incomplete",
        "claimBoundary": "Exact qualified Metal/ZFP async post-manifest failure blocks subsequent science ledger/checkpoint commits and marks same-generation run failed. One normal six-second run preserves baseline compressed science; wall times are bounded diagnostics, not C6 release thresholds.",
        "inputs": inputs,
        "results": [
            {"id": "injectedFailure", "log": failure["log"]},
            {"id": "qualifiedPositive", "log": positive["log"]},
            {"id": "baselinePositive", "log": prior["log"]},
            {"id": "qualifiedNpy", "log": npy["log"]},
            {"id": "studioReader", "log": identity(reader_log)},
        ],
        "injectedFailure": {**failure, "steps": failed_steps,
                            "scienceLedgers": failed_ledgers,
                            "runContract": identity(failed_contract_path),
                            "publicationState": failed_contract["publication_state"],
                            "pendingIndex": identity(pending),
                            "historyHeadPresent": False,
                            "checkpointFiles": 0},
        "qualifiedPositive": {**positive,
                              "runContract": identity(positive_dir / "nimbus_output.json"),
                              "validation": identity(positive_dir / "validation_summary.json")},
        "baselinePositive": prior,
        "qualifiedNpy": npy,
        "compressedPayloadsExactToBaseline": len(payloads),
        "scienceLedgersExactToBaseline": list(LEDGERS),
        "studioReader": {"status": "pass", "log": identity(reader_log)},
        "singleRunWallSeconds": {"oldAsyncOverlap": prior["wallSeconds"],
                                 "qualifiedCommitBarrier": positive["wallSeconds"]},
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "failureLedgerRows": 0,
                      "durableInitialFrame": True,
                      "wallSeconds": report["singleRunWallSeconds"]}), flush=True)


if __name__ == "__main__":
    main()

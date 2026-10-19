#!/usr/bin/env python3
"""Observe exact qualified Metal/ZFP behavior after injected post-manifest index failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LEDGERS = (
    "budget.csv", "energy_budget.csv", "moisture_budget.csv",
    "momentum_budget.csv", "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-binary-sha", required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    config = args.config.resolve(strict=True)
    if identity(binary)["sha256"] != args.expected_binary_sha:
        raise RuntimeError("post-manifest diagnostic binary differs from the frozen simulator")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output / "run"
    command = [str(binary), "--headless", "--config", str(config),
               "--duration=2", "--write-every=1", "--checkpoint-every=0",
               f"--outdir={run}", "--threads=1", "--log-profile=normal"]
    result = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=120, check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
             "NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST": "1"})
    log = output / "runtime.log"
    log.write_text(result.stdout)
    steps = []
    if run.exists():
        for step in sorted(run.glob("step_*")):
            if not step.is_dir():
                continue
            manifest = step / "manifest.json"
            steps.append({"name": step.name, "manifestCommitted": manifest.is_file(),
                          "manifest": identity(manifest) if manifest.is_file() else None,
                          "zfpPayloads": sum(path.is_file() for path in step.glob("*.zfp3d"))})
    ledgers = {}
    for name in LEDGERS:
        path = run / name
        if path.is_file():
            content = path.read_text()
            ledgers[name] = {**identity(path), "lineCount": len(content.splitlines()),
                             "nonHeaderRows": max(0, len(content.splitlines()) - 1)}
    index = run / "history_index.json"
    pending = run / "history_index.pending.json"
    contract = run / "nimbus_output.json"
    report = {
        "schema": "nimbus.generated-metal-zfp-post-manifest-diagnostic.v1",
        "status": "observed_failure" if result.returncode != 0 else "unexpected_success",
        "claimBoundary": "Test-only failure after durable step manifest, before history-index append. A manifest-committed frame is allowed to remain; this diagnostic does not assert zero publication or ledger atomicity.",
        "inputs": {"binary": identity(binary), "config": identity(config),
                   "checker": identity(Path(__file__))},
        "injection": "NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST=1",
        "returncode": result.returncode, "log": identity(log),
        "injectedFailureObserved": "injected post-manifest history-index failure" in result.stdout,
        "fallbackOff": "fallback=off" in result.stdout,
        "steps": steps,
        "committedStepManifests": sum(step["manifestCommitted"] for step in steps),
        "historyHead": identity(index) if index.is_file() else None,
        "pendingIndex": identity(pending) if pending.is_file() else None,
        "outputContract": identity(contract) if contract.is_file() else None,
        "scienceLedgers": ledgers,
        "checkpoints": sum(path.is_file() for path in (run / "checkpoints").glob("*"))
        if (run / "checkpoints").is_dir() else 0,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "observed_failure" or not report["injectedFailureObserved"] or \
            not report["fallbackOff"]:
        raise RuntimeError("post-manifest ZFP diagnostic did not reach the intended native writer failure")
    if identity(binary) != report["inputs"]["binary"] or \
            identity(config) != report["inputs"]["config"]:
        raise RuntimeError("post-manifest ZFP diagnostic input changed during execution")
    print(json.dumps({"status": report["status"],
                      "committedManifests": report["committedStepManifests"],
                      "budgetRows": ledgers.get("budget.csv", {}).get("nonHeaderRows", 0)}), flush=True)


if __name__ == "__main__":
    main()

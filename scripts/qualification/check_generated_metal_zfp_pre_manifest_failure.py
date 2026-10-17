#!/usr/bin/env python3
"""Seal test-only ZFP writer pre-manifest failure and adjacent commit semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "bin/test_core_output_writer"
SIMULATOR = ROOT / "bin/nimbus_sim"
INPUTS = {
    "testBinary": TEST,
    "simulatorBinary": SIMULATOR,
    "testSource": ROOT / "tests/core/test_output_writer.cpp",
    "writerSource": ROOT / "src/core/output/output_writer.cpp",
    "historyIndexSource": ROOT / "src/core/output/history_index.cpp",
    "checker": Path(__file__),
}
CASES = (
    ("pre_manifest", "[pre-manifest]", "9 assertions in 1 test case"),
    ("post_manifest_history_index", "[history_index]", "295 assertions in 5 test cases"),
    ("full_output_writer", None, "2809 assertions in 44 test cases"),
)


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-simulator-sha", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {name: identity(path.resolve(strict=True)) for name, path in INPUTS.items()}
    if inputs["simulatorBinary"]["sha256"] != args.expected_simulator_sha:
        raise RuntimeError("ZFP pre-manifest test is not bound to the frozen simulator")
    results = []
    for name, selection, expected in CASES:
        command = [str(TEST.resolve(strict=True))]
        if selection is not None:
            command.append(selection)
        result = subprocess.run(command, cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=120, check=False)
        log = output / f"{name}.log"
        log.write_text(result.stdout)
        if result.returncode != 0 or expected not in result.stdout or \
                "All tests passed" not in result.stdout:
            raise RuntimeError(f"ZFP writer test {name} failed: {result.stdout[-2500:]}")
        results.append({"id": name, "status": "pass", "filter": selection,
                        "expectedAssertions": expected, "log": identity(log)})
    if any(identity(Path(value["path"])) != value for value in inputs.values()):
        raise RuntimeError("ZFP writer test input changed during verification")
    report = {
        "schema": "nimbus.metal-zfp-pre-manifest-writer-failure.v1",
        "status": "pass_writer_level_only",
        "readiness": "Engineering incomplete",
        "claimBoundary": "Test-only async serial ZFP writer failure before manifest commit. It proves no completed manifest/history head, unchanged preexisting ledger bytes and terminal writer rejection; it does not prove integrated solver rollback or post-manifest zero publication.",
        "inputs": inputs,
        "preManifestAssertions": [
            "writer flush fails terminally after pending-index self-symlink",
            "snapshots_written remains zero",
            "no completed step manifest or history head is published",
            "preexisting budget.csv bytes remain unchanged",
            "the next snapshot submission is rejected",
        ],
        "postManifestBoundary": "A separate existing test intentionally preserves a completed frame when history-index append fails after its durable manifest.",
        "integratedPostManifestDiagnostic": "pending_physical_device",
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "tests": len(results)}), flush=True)


if __name__ == "__main__":
    main()

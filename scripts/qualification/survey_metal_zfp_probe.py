#!/usr/bin/env python3
"""Seal the bounded exploratory Metal/ZFP history and readback probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LEDGERS = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
FIELDS = ("u", "v", "w", "theta", "p", "rho", "qv", "qr")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    early = root / "nimbus-metal-zfp-exploratory-20260927"
    threaded = root / "nimbus-metal-zfp-exploratory-t4-20260927"
    retained = root / "nimbus-metal-zfp-retained-20260927"
    reference = root / "nimbus-metal-zfp-retained-npy-20260927"
    for directory in (early, threaded, retained, reference):
        require(directory.is_dir(), f"missing probe directory: {directory}")
        validation = json.loads((directory / "validation_summary.json").read_text())
        require(not validation["failed"] and validation["total_violations"] == 0,
                f"strict validation failed: {directory}")
    payloads = sorted(path.relative_to(early) for path in early.rglob("*.zfp3d"))
    require(len(payloads) == 40, "expected five histories of eight ZFP fields")
    for path in payloads:
        require(sha(early / path) == sha(threaded / path), f"threaded ZFP payload differs: {path}")
    for ledger in LEDGERS:
        require(sha(early / ledger) == sha(threaded / ledger), f"threaded ledger differs: {ledger}")
    lineage = []
    for step, role, parents in ((4, "keyframe", []), (5, "simple_delta", [4]),
                                (6, "predictive_delta", [5, 4])):
        manifest = json.loads((retained / f"step_{step:06d}" / "manifest.json").read_text())
        observed = {field["field_id"]: field["compression"] for field in manifest["fields"]
                    if field["field_id"] in FIELDS}
        require(set(observed) == set(FIELDS), f"missing compressed field at step {step}")
        for field, compression in observed.items():
            require(compression["frame_role"] == role and
                    compression["parent_step_indices"] == parents,
                    f"incorrect {field} predictive lineage at step {step}")
        lineage.append({"step": step, "role": role, "parents": parents,
                        "manifestSha256": sha(retained / f"step_{step:06d}" / "manifest.json")})
    logs = {
        "fieldReadback": ("retained-readback.log", "qualified_simulator_zfp_matches_npy_reference"),
        "predictiveReadback": ("retained-decode.log", "qualified_retained_predictive_zfp_latest_decodes"),
    }
    readback = {}
    for label, (name, test) in logs.items():
        path = root / name
        content = path.read_text()
        require(f"test zfp3d::tests::{test} ... ok" in content and
                "1 passed; 0 failed" in content, f"Rust readback failed: {name}")
        readback[label] = {"test": test, "sha256": sha(path)}
    require(args.candidate.is_file() and args.binary.is_file(), "binary or candidate missing")
    report = {
        "schema": "nimbus.metal-zfp-exploratory-probe.v1",
        "status": "pass_exploratory_only",
        "claimBoundary": "Valid source/reader probe; qualified public admission, matrix coverage, restart and release evidence remain open.",
        "binarySha256": sha(args.binary),
        "candidateSha256": sha(args.candidate),
        "threadedCompressedPayloadsExact": len(payloads),
        "threadedLedgersExact": list(LEDGERS),
        "retainedLineage": lineage,
        "studioReadback": readback,
        "directories": {directory.name: {"files": sum(path.is_file() for path in directory.rglob("*"))}
                        for directory in (early, threaded, retained, reference)},
    }
    if args.output.exists():
        parser.error("use a new output path to preserve the previous probe")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "payloads": len(payloads),
                      "lineage": [row["role"] for row in lineage]}))


if __name__ == "__main__":
    main()

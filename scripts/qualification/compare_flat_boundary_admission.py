#!/usr/bin/env python3
"""Compare exact fixed-boundary admission records across simulator binaries."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def read(summary_path: Path) -> tuple[dict, dict[str, dict]]:
    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != "nimbus.flat-boundary-admission-merge.v1" or \
            summary.get("status") != "complete" or summary.get("totalCases") != 4608:
        raise RuntimeError("flat boundary comparison needs a complete source admission sweep")
    record_path = Path(summary["records"]["path"])
    record_path = (record_path if record_path.is_absolute() else ROOT / record_path).resolve(strict=True)
    if identity(record_path)["sha256"] != summary["records"]["sha256"]:
        raise RuntimeError("flat boundary admission record archive changed")
    rows = {}
    digest = hashlib.sha256()
    with gzip.open(record_path, "rb") as stream:
        for line in stream:
            digest.update(line)
            record = json.loads(line)
            if record["id"] in rows:
                raise RuntimeError("flat boundary admission record ID is duplicated")
            rows[record["id"]] = record
    if len(rows) != 4608 or digest.hexdigest() != summary["records"]["uncompressedSha256"]:
        raise RuntimeError("flat boundary admission record stream omitted an ID")
    return summary, rows


def compare(old_path: Path, current_path: Path) -> dict:
    old_summary, old = read(old_path)
    current_summary, current = read(current_path)
    if old.keys() != current.keys() or \
            old_summary["inputs"]["binary"]["sha256"] == \
            current_summary["inputs"]["binary"]["sha256"]:
        raise RuntimeError("flat boundary admission comparison has missing IDs or one executable")
    status_changes = []
    reason_changes = []
    source_changes = []
    canonical_changes = []
    output_hash_changes = []
    for identifier in sorted(old):
        before, after = old[identifier], current[identifier]
        if before["index"] != after["index"] or before["familyId"] != after["familyId"] or \
                before["backend"] != after["backend"]:
            raise RuntimeError("flat boundary stable ID changed meaning across binaries")
        if before["ordinaryQualifiedAdmission"] != after["ordinaryQualifiedAdmission"]:
            status_changes.append(identifier)
        if before["reason"] != after["reason"]:
            reason_changes.append(identifier)
        if before["sourceYamlSha256"] != after["sourceYamlSha256"]:
            source_changes.append(identifier)
        if before["canonicalYamlSha256"] != after["canonicalYamlSha256"]:
            canonical_changes.append(identifier)
        if before["validationOutputSha256"] != after["validationOutputSha256"]:
            output_hash_changes.append(identifier)
    counts = Counter((row["backend"], row["ordinaryQualifiedAdmission"])
                     for row in current.values())
    return {"schema": "nimbus.flat-boundary-admission-binary-comparison.v1",
            "status": "complete", "claimBoundary":
            "Exact fixed-family admission/status/config comparison across two binaries; no physical current-binary execution or Ready claim.",
            "inputs": {"oldSummary": identity(old_path), "currentSummary": identity(current_path),
                       "comparator": identity(Path(__file__))},
            "oldBinarySha256": old_summary["inputs"]["binary"]["sha256"],
            "currentBinarySha256": current_summary["inputs"]["binary"]["sha256"],
            "cases": len(current), "statusChanges": status_changes,
            "reasonChanges": reason_changes, "sourceYamlChanges": source_changes,
            "canonicalYamlChanges": canonical_changes,
            "validationOutputHashChangeCount": len(output_hash_changes),
            "currentCounts": [{"backend": backend, "status": status, "count": count}
                              for (backend, status), count in sorted(counts.items())],
            "I1Complete": False, "wholeBackendReady": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new comparison path")
    result = compare(args.old.resolve(strict=True), args.current.resolve(strict=True))
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "cases": result["cases"],
                      "statusChanges": len(result["statusChanges"]),
                      "sourceChanges": len(result["sourceYamlChanges"])}), flush=True)


if __name__ == "__main__":
    main()

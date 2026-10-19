#!/usr/bin/env python3
"""Seal the durable pre-fault members of one bounded headless failure receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document(report: Path) -> dict:
    outdir = report / "raw/failed-two-second"
    rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact is a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(outdir).as_posix()
        if not (relative.startswith("step_") or relative.startswith("checkpoints/") or
                relative == "history_index.json"):
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)})
    if len(rows) != 22 or any(row["path"].startswith("step_000002/") or
                              "checkpoint_step_000000020" in row["path"] for row in rows):
        raise RuntimeError("pre-fault artifact membership changed")
    serialized = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"schema": "nimbus.headless-prior-artifact-index.v1", "status": "verified",
            "memberCount": len(rows), "totalBytes": sum(row["bytes"] for row in rows),
            "memberSetSha256": hashlib.sha256(serialized).hexdigest(), "members": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    report = args.report.resolve(strict=True)
    index = report / "prior_artifact_index.json"
    expected = document(report)
    if args.verify:
        if not index.is_file() or json.loads(index.read_text()) != expected:
            raise RuntimeError("prior artifact index drifted")
    else:
        if index.exists():
            parser.error("index exists; use --verify")
        write_json_atomic(index, expected)
    print(json.dumps({"status": expected["status"], "memberCount": expected["memberCount"],
                      "memberSetSha256": expected["memberSetSha256"]}))


if __name__ == "__main__":
    main()

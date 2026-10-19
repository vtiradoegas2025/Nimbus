#!/usr/bin/env python3
"""Seal or verify every regular member of a bounded backend evidence report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


INDEX = "archive_index.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def members(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"report contains a symlink: {path}")
        if not path.is_file() or path.name == INDEX and path.parent == root:
            continue
        relative = path.relative_to(root).as_posix()
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)})
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise RuntimeError("report member set is empty or contains duplicate identities")
    return rows


def document(root: Path) -> dict:
    rows = members(root)
    serialized = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"schema": "nimbus.backend-report-member-index.v1", "status": "verified",
            "root": str(root), "producerSha256": sha(Path(__file__)),
            "memberCount": len(rows), "totalBytes": sum(row["bytes"] for row in rows),
            "memberSetSha256": hashlib.sha256(serialized).hexdigest(), "members": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.report.resolve(strict=True)
    if not root.is_dir():
        parser.error("report must be a directory")
    index = root / INDEX
    expected = document(root)
    if args.verify:
        if not index.is_file() or json.loads(index.read_text()) != expected:
            raise RuntimeError("report member index drifted")
    else:
        if index.exists():
            parser.error("archive_index.json already exists; use --verify")
        write_json_atomic(index, expected)
    print(json.dumps({"status": "verified", "memberCount": expected["memberCount"],
                      "totalBytes": expected["totalBytes"],
                      "memberSetSha256": expected["memberSetSha256"]}))


if __name__ == "__main__":
    main()

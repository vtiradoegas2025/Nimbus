#!/usr/bin/env python3
"""Verify and seal every member of a bounded runtime evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    archive = args.archive.resolve(strict=True)
    summary = args.summary.resolve(strict=True)
    if args.output.exists():
        parser.error("use a new output path to preserve the prior seal")
    bound = json.loads(summary.read_text())
    expected = {}
    bound_rows = bound.get("results")
    if bound_rows is None:
        bound_rows = [bound] if "runs" in bound else []
    for row in bound_rows:
        records = [row.get(key) for key in
                   ("config", "log", "validation", "checkpoint", "checkpointSidecar")]
        records.extend(row.get("configs", {}).values())
        records.extend(row.get("manifests", {}).values())
        for run in row.get("runs", {}).values():
            records.extend(run.get(key) for key in ("log", "validation"))
        restart = row.get("restart", {})
        records.extend(restart.get(key) for key in ("checkpoint", "checkpointSidecar"))
        records.extend(restart.get("manifests", {}).values())
        for run in (restart.get("continuous"), restart.get("resumed")):
            if run is not None:
                records.extend(run.get(key) for key in ("log", "validation"))
        for record in records:
            if record is None:
                continue
            path = Path(record["path"]).resolve(strict=True)
            if not path.is_relative_to(source_root):
                raise RuntimeError("summary references a file outside the archive source")
            expected[path.relative_to(source_root).as_posix()] = record["sha256"]
    aggregate = hashlib.sha256()
    members = []
    found = set()
    total_bytes = 0
    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != source_root.name:
                raise RuntimeError(f"archive has an unsafe or unexpected member path: {member.name}")
            relative = Path(*name.parts[1:])
            original = source_root / relative
            if not original.is_file() or original.stat().st_size != member.size:
                raise RuntimeError(f"archive member differs from source: {relative}")
            stream = bundle.extractfile(member)
            if stream is None:
                raise RuntimeError(f"archive member cannot be read: {relative}")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            value = digest.hexdigest()
            if value != sha(original) or \
                    (relative.as_posix() in expected and value != expected[relative.as_posix()]):
                raise RuntimeError(f"archive member hash differs: {relative}")
            found.add(relative.as_posix())
            members.append({"path": relative.as_posix(), "bytes": member.size, "sha256": value})
            aggregate.update((relative.as_posix() + "\t" + str(member.size) + "\t" + value + "\n").encode())
            total_bytes += member.size
    if found != {path.relative_to(source_root).as_posix() for path in source_root.rglob("*") if path.is_file()} or \
            not set(expected).issubset(found):
        raise RuntimeError("archive omitted source or summary files")
    report = {"schema": "nimbus.runtime-evidence-archive.v1", "status": "verified",
              "archive": {"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha(archive)},
              "summary": {"path": str(summary), "sha256": sha(summary)},
              "sourceRoot": str(source_root), "memberCount": len(members),
              "uncompressedBytes": total_bytes, "memberManifestSha256": aggregate.hexdigest(),
              "allMemberHashesVerified": True, "members": members}
    write_json_atomic(args.output, report)
    print(json.dumps({"status": report["status"], "members": len(members),
                      "archiveSha256": report["archive"]["sha256"]}), flush=True)


if __name__ == "__main__":
    main()

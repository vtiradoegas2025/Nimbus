#!/usr/bin/env python3
"""Recheck source, run, parity and archive seals for the guarded boundary probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(value: object):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for item in value.values():
            yield from records(item)
    elif isinstance(value, list):
        for item in value:
            yield from records(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.summary.resolve(strict=True)
    seal_path = args.archive_manifest.resolve(strict=True)
    summary = json.loads(summary_path.read_text())
    seal = json.loads(seal_path.read_text())
    if summary.get("schema") != "nimbus.flat-mixed-boundary-metal-probe.v1" or \
            summary.get("status") != "bounded_pass" or summary.get("wholeBackendReady") is not False:
        raise RuntimeError("mixed-boundary report has an invalid status or claim boundary")
    checked = set()
    for record in records(summary):
        path = Path(record["path"]).resolve(strict=True)
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise RuntimeError(f"mixed-boundary evidence source changed: {path}")
        checked.add(path)
    if summary["admission"]["metal"]["accepted"] is not False or \
            summary["admission"]["cpu"]["accepted"] is not True or \
            summary["admission"]["vulkan"]["accepted"] is not True:
        raise RuntimeError("mixed-boundary public admission changed")
    operation = "advection/advect_conservative_update_cartesian_cgrid.comp"
    for name, count in (("metal1", 30), ("metal4", 30), ("metalContinuous", 60),
                        ("metalResumed", 30), ("metalStructuralNeighbor", 0)):
        if summary["cases"][name]["operations"].get(operation, 0) != count:
            raise RuntimeError(f"mixed-boundary native dispatch changed: {name}")
    if any(not field["passed"] for result in summary["parity"].values() for field in result.values()) or \
            any(not passed for result in summary["ledgers"].values() for passed in result.values()):
        raise RuntimeError("mixed-boundary field or ledger evidence failed")
    archive = Path(seal["archive"]["path"]).resolve(strict=True)
    if seal.get("schema") != "nimbus.runtime-evidence-archive.v1" or \
            seal.get("status") != "verified" or not seal.get("allMemberHashesVerified") or \
            seal["summary"]["sha256"] != sha(summary_path) or \
            seal["archive"]["sha256"] != sha(archive) or \
            seal["memberCount"] != len(seal["members"]) or \
            seal["memberCount"] < 250:
        raise RuntimeError("mixed-boundary archive seal changed")
    print(f"[flat-mixed-boundary-evidence] PASS: {len(checked)} hashed report inputs, "
          f"{seal['memberCount']} sealed raw members, guarded Metal admission")


if __name__ == "__main__":
    main()

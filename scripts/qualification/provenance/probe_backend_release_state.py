#!/usr/bin/env python3
"""Read-only R9/C8 probe of release receipts, source identity, and local package state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.provenance.inspect_release_evidence import (
    DEFAULT_GATE_DIRECTORY,
    DEFAULT_MANIFESTS,
    inspect_manifest,
    inspect_receipt,
)
from scripts.qualification.provenance.record_gate_result import (
    REQUIRED_GATE_POLICY,
    write_atomic,
)
from scripts.qualification.provenance import release_manifest
from scripts.qualification.validate_c6_threshold_candidate import validate_policy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_release_source_roots() -> list[Path]:
    makefile = (ROOT / "mk/tests.mk").read_text(encoding="utf-8")
    source = makefile.split("RELEASE_SOURCE_ROOTS :=", 1)[1].split("RELEASE_SOURCE_ARGS", 1)[0]
    roots = [Path(token) for token in source.replace("\\\n", " ").split()]
    if not roots or roots[0] != Path("include") or roots[-1] != Path("ui/src-tauri/tauri.linux.conf.json"):
        raise ValueError("canonical Make release-source root declaration changed")
    return roots


def source_inventory_probe(roots: list[Path]) -> dict:
    try:
        inventory = release_manifest.inventory_record(roots, ROOT)
    except (OSError, release_manifest.ManifestError) as error:
        return {"status": "blocked", "rootCount": len(roots), "reason": str(error)}
    return {
        "status": "enumerated_not_clean_release",
        "rootCount": len(roots),
        "fileCount": inventory["count"],
        "sha256": inventory["sha256"],
    }


def package_probe() -> dict:
    macos = ROOT / "ui/src-tauri/target/release/bundle/macos"
    production_archive = ROOT / "reports/pre_beta/Nimbus-Studio-macos-arm64.tar.zst"
    smoke_path = ROOT / "reports/pre_beta/local_release_candidate_v2_smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else {}
    return {
        "localMacosBundleDirectoryExists": macos.is_dir(),
        "localMacosApps": sorted(path.name for path in macos.glob("*.app")) if macos.is_dir() else [],
        "productionArchivePath": production_archive.relative_to(ROOT).as_posix(),
        "productionArchiveExists": production_archive.is_file(),
        "smokeManifestPath": smoke_path.relative_to(ROOT).as_posix(),
        "smokePackageKind": smoke.get("package", {}).get("kind"),
        "smokePackageProductionEligible": release_manifest.is_production_package(smoke.get("package")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--c6-policy", type=Path, default=ROOT / "tests/fixtures/reference/performance/threshold_policy_v1.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    policy_path = args.c6_policy.resolve(strict=True)
    binary_sha = sha256(binary)
    policy_sha = sha256(policy_path)
    revision_before = release_manifest.git_identity(ROOT)
    gate_directory = ROOT / DEFAULT_GATE_DIRECTORY
    receipts = [
        inspect_receipt(gate_directory / f"{target}.json", target, binary_sha, ROOT)
        for target in REQUIRED_GATE_POLICY
    ]
    manifests = [inspect_manifest(ROOT / path, binary_sha, ROOT) for path in DEFAULT_MANIFESTS]
    source = source_inventory_probe(canonical_release_source_roots())
    package = package_probe()
    policy = validate_policy(json.loads(policy_path.read_text(encoding="utf-8")), ROOT, binary)
    revision_after = release_manifest.git_identity(ROOT)
    if sha256(binary) != binary_sha or sha256(policy_path) != policy_sha:
        raise RuntimeError("release probe binary or threshold policy changed during inspection")
    report = {
        "schema": "nimbus.r9-c8-backend-release-probe.v1",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "binary": {"path": str(binary), "sha256": binary_sha},
        "probeScriptSha256": sha256(Path(__file__)),
        "c6Policy": {"path": str(policy_path), "sha256": policy_sha, "result": policy},
        "receipts": receipts,
        "manifests": manifests,
        "summary": {
            name: sum(row["classification"] == name for row in (*receipts, *manifests))
            for name in ("current", "stale", "missing", "invalid")
        },
        "revision": revision_after,
        "revisionStableDuringProbe": revision_before == revision_after,
        "sourceInventory": source,
        "package": package,
        "releaseReady": False,
    }
    output = args.output.resolve()
    if output.exists():
        parser.error("use a new output path; prior release probes are immutable")
    write_atomic(output, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"binarySha256": binary_sha, "summary": report["summary"],
                      "sourceInventory": source["status"], "releaseReady": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

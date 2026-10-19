#!/usr/bin/env python3
"""Prospective C6 v3 gate with measurement-bound paired time-gain claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.qualification import validate_c6_threshold_candidate_v2 as common
from scripts.qualification.c6_paired_measurements import validate_claim

release_manifest = common.release_manifest
FIELDS, LEDGERS, REQUIRED_WORKLOADS = common.FIELDS, common.LEDGERS, common.REQUIRED_WORKLOADS
REQUIRED_MODES = common.REQUIRED_MODES
HISTORICAL_V1_PATH, HISTORICAL_V1_SHA256 = common.HISTORICAL_V1_PATH, common.HISTORICAL_V1_SHA256
VALIDATED_FILES = common.VALIDATED_FILES
require, digest, valid_sha = common.require, common.digest, common.valid_sha
file_identity, linked_json = common.file_identity, common.linked_json
canonical_source_seal, validate_profile = common.canonical_source_seal, common.validate_profile
C6V3PolicyError = common.C6V2PolicyError
POLICY_SCHEMA = "nimbus.c6-release-threshold-candidate.v3"
RECEIPT_SCHEMA = "nimbus.c6-workload-receipt.v3"
LIMIT_SCHEMA = common.LIMIT_SCHEMA
REVIEW_SCHEMA = "nimbus.c6-threshold-activation-review.v2"


def validate_complete_row(row: dict, policy: dict, root: Path, source_record: dict) -> list[dict]:
    source_sha = source_record["inventory"]["sha256"]
    receipt = linked_json(row.get("receipt"), root, f"{row['id']} receipt")
    limits = linked_json(row.get("limits"), root, f"{row['id']} limits")
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("status") == "complete" and
            receipt.get("workloadId") == row["id"] and
            receipt.get("binarySha256") == policy["releaseBinary"]["sha256"] and
            receipt.get("sourceInventorySha256") == source_sha, f"stale or incomplete workload receipt: {row['id']}")
    require(limits.get("schema") == LIMIT_SCHEMA and limits.get("status") == "reviewed" and
            limits.get("workloadId") == row["id"] and limits.get("receiptSha256") == row["receipt"]["sha256"],
            f"unreviewed or stale workload limits: {row['id']}")
    review = limits.get("review")
    require(isinstance(review, dict) and isinstance(review.get("reviewer"), str) and review["reviewer"].strip() and
            review.get("decision") == "approved" and review["reviewer"] != receipt.get("operator"),
            f"missing independent limit review: {row['id']}")
    review_record = linked_json(review.get("record"), root, f"{row['id']} limit review")
    require(review_record.get("schema") == "nimbus.c6-independent-limit-review.v1" and
            review_record.get("workloadId") == row["id"] and
            review_record.get("reviewer") == review["reviewer"] and
            review_record.get("decision") == "approved" and
            review_record.get("binarySha256") == receipt["binarySha256"] and
            review_record.get("sourceInventorySha256") == source_sha and
            review_record.get("limitProfilesSha256") ==
                hashlib.sha256(release_manifest.canonical_bytes(limits.get("profiles"))).hexdigest(),
            f"limit review does not bind exact limits: {row['id']}")
    require(isinstance(receipt.get("operator"), str) and receipt["operator"].strip(),
            f"missing workload operator: {row['id']}")
    profiles = receipt.get("profiles")
    limit_profiles = limits.get("profiles")
    expected = set(REQUIRED_MODES[row["id"]])
    require(isinstance(profiles, list) and isinstance(limit_profiles, list) and
            len(profiles) == len(limit_profiles) == len(expected) and
            {p.get("mode") for p in profiles if isinstance(p, dict)} == expected and
            {p.get("mode") for p in limit_profiles if isinstance(p, dict)} == expected,
            f"wrong size/backend applicability in {row['id']}")
    by_mode = {p["mode"]: p for p in limit_profiles}
    results = [validate_profile(profile, by_mode[profile["mode"]], row, policy, root, source_record)
               for profile in profiles]
    paired = validate_claim(receipt.get("repairClaim"), profiles, policy, root)
    if paired is not None:
        for result in results:
            result["pairedRepairProof"] = paired["profiles"][result["mode"]]
    return results


def _validate_policy(policy: dict, root: Path, binary: Path, *, source_seal: dict | None = None) -> dict:
    require(policy.get("schema") == POLICY_SCHEMA, "wrong C6 v3 policy schema")
    require(policy.get("status") in {"draft_not_activated", "reviewed_active"}, "invalid C6 v3 status")
    acceptance = policy.get("acceptance", {})
    require(acceptance.get("minimumSteadyStateTrials") == 7 and
            acceptance.get("minimumNamedWallFraction") == .95 and
            acceptance.get("minimumWholeStepGainPercentExclusive") == 2.0 and
            acceptance.get("minimumPairedMadMultipleExclusive") == 3.0 and
            acceptance.get("memoryAlternative") == "at_least_one_live_full_field_or_two_percent_of_admitted_peak" and
            set(acceptance.get("requiredExactFields", [])) == FIELDS and
            set(acceptance.get("requiredExactLedgers", [])) == LEDGERS,
            "C6 acceptance gates changed")
    require(policy.get("historicalV1") == {"path": HISTORICAL_V1_PATH, "sha256": HISTORICAL_V1_SHA256} or
            root != ROOT, "historical v1 lineage changed")
    legacy = linked_json(policy.get("historicalV1"), root, "historical v1 policy")
    require(legacy.get("schema") == "nimbus.c6-release-threshold-candidate.v1" and
            legacy.get("status") == "draft_not_activated" and
            any(x.get("id") == "cpu_milbrandt_cold_gamma_ab" and x.get("disposition") == "time_gain_rejected"
                for x in legacy.get("observations", [])), "historical rejected evidence was lost")
    release = policy.get("releaseBinary")
    require(isinstance(release, dict) and valid_sha(release.get("sha256")) and release.get("path") == "bin/nimbus_sim",
            "missing release binary identity")
    require(digest(binary) == release["sha256"], "stale release binary")
    source = policy.get("canonicalSource")
    require(isinstance(source, dict) and source.get("rootsDeclarationPath") == "mk/tests.mk",
            "missing canonical source-root owner")
    if source.get("rootsDeclarationSha256") is None:
        require(policy["status"] == "draft_not_activated" and source.get("inventorySha256") is None,
                "unbound canonical source-root declaration cannot activate")
    else:
        require(source["rootsDeclarationSha256"] == digest(root / "mk/tests.mk"),
                "stale canonical source-root declaration")
    rows = policy.get("workloadClasses")
    require(isinstance(rows, list) and len(rows) == 10 and
            {row.get("id") for row in rows if isinstance(row, dict)} == REQUIRED_WORKLOADS,
            "release workload classes are missing, duplicated, or unreviewed")
    legacy_rows = {row["id"]: row for row in legacy.get("workloadClasses", [])}
    require(set(legacy_rows) == REQUIRED_WORKLOADS, "historical workload inventory changed")
    completed = [row for row in rows if row.get("receipt") is not None or row.get("limits") is not None]
    if source_seal is None and (completed or policy["status"] == "reviewed_active"):
        source_seal = canonical_source_seal(root)
    if source_seal is not None:
        require(source.get("inventorySha256") == source_seal.get("inventorySha256") and
                source.get("rootCount") == source_seal.get("rootCount") and
                source.get("fileCount") == source_seal.get("fileCount") and
                source.get("rootsDeclarationSha256") == source_seal.get("rootsDeclarationSha256"),
                "stale canonical source inventory")
        recorded_inventory = linked_json(source.get("inventoryArtifact"), root, "canonical source inventory")
        require(recorded_inventory == source_seal.get("record"), "incomplete or stale canonical source inventory artifact")
    missing = []
    results = {}
    for row in rows:
        identifier = row["id"]
        require(row.get("requiredModes") == list(REQUIRED_MODES[identifier]), f"wrong backend applicability: {identifier}")
        require(all(row.get(key) == legacy_rows[identifier].get(key)
                    for key in ("grid", "config", "fullFieldBytes")),
                f"workload size/config differs from retained inventory: {identifier}")
        grid = row.get("grid")
        if identifier == "retained_terrain_postimplementation":
            require(grid is None and row.get("config") is None and row.get("fullFieldBytes") is None,
                    "terrain size/config cannot be invented before retained scope closes")
        else:
            require(isinstance(grid, list) and len(grid) == 3 and all(type(x) is int and x > 0 for x in grid) and
                    row.get("fullFieldBytes") == math.prod(grid) * 4,
                    f"wrong workload size: {identifier}")
        if row.get("receipt") is None or row.get("limits") is None:
            require(row.get("receipt") is None and row.get("limits") is None and
                    isinstance(row.get("missing"), list) and row["missing"],
                    f"placeholder or partial workload evidence: {identifier}")
            missing.append(identifier)
        else:
            require(row.get("missing") == [], f"completed workload still declares missing: {identifier}")
            require(source_seal is not None, "source inventory required for receipts")
            results[identifier] = validate_complete_row(row, policy, root, source_seal["record"])
    ready = not missing and policy["status"] == "reviewed_active"
    if ready:
        review = linked_json(policy.get("activationReview"), root, "activation review")
        unsigned = dict(policy)
        unsigned["activationReview"] = None
        require(review.get("schema") == REVIEW_SCHEMA and review.get("decision") == "approved" and
                isinstance(review.get("reviewer"), str) and review["reviewer"].strip() and
                review.get("candidateUnsignedSha256") == hashlib.sha256(release_manifest.canonical_bytes(unsigned)).hexdigest() and
                review.get("binarySha256") == release["sha256"] and
                review.get("sourceInventorySha256") == source_seal["inventorySha256"],
                "stale or unreviewed activation")
    else:
        require(policy["status"] == "draft_not_activated" and policy.get("activationReview") is None,
                "draft cannot carry activation or active status")
    return {"status": "reviewed_active" if ready else "validated_draft_not_release_ready",
            "releaseReady": ready, "releaseBinarySha256": release["sha256"],
            "canonicalSourceInventorySha256": source_seal["inventorySha256"] if source_seal else None,
            "missingWorkloadClasses": missing, "validatedWorkloads": results}


def validate_policy(policy: dict, root: Path, binary: Path) -> dict:
    files: dict[str, str] = {}
    token = VALIDATED_FILES.set(files)
    try:
        initial_binary = release_manifest.file_record(binary, root)
        rows = policy.get("workloadClasses", [])
        needs_inventory = policy.get("status") == "reviewed_active" or any(
            isinstance(row, dict) and (row.get("receipt") is not None or row.get("limits") is not None) for row in rows)
        initial_source = canonical_source_seal(root) if needs_inventory else None
        result = _validate_policy(policy, root, binary, source_seal=initial_source)
        if initial_source is not None:
            require(canonical_source_seal(root) == initial_source, "canonical source drift during validation")
        require(release_manifest.file_record(binary, root) == initial_binary, "release binary drift during validation")
        for relative, sha256 in files.items():
            file_identity({"path": relative, "sha256": sha256}, root, "final-seal evidence")
        return result
    finally:
        VALIDATED_FILES.reset(token)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--require-release-ready", action="store_true")
    arguments = parser.parse_args()
    try:
        policy = json.loads(arguments.policy.read_text(encoding="utf-8"))
        result = validate_policy(policy, ROOT, arguments.binary)
        require(not arguments.require_release_ready or result["releaseReady"], "C6 release thresholds are not activated")
    except (OSError, KeyError, TypeError, ValueError, release_manifest.ManifestError) as error:
        parser.exit(1, f"C6 v3 threshold candidate rejected: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


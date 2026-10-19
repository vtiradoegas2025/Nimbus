"""Strict structural provenance contracts for maintained admission checks."""

from __future__ import annotations

import hashlib

import json

EVIDENCE = ("science_reference", "conservation", "invalid_input",
            "failure_atomicity", "restart", "threading", "backend_parity",
            "output_membership", "resources")


def compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_id(kind: str, value: object) -> str:
    return kind + ":" + hashlib.sha256(compact(value).encode()).hexdigest()[:24]

def structural_groups(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        selected = row.get("canonicalStructuralId")
        if selected is None:
            continue
        signature = row["structuralSignature"]
        if selected in grouped and grouped[selected]["signature"] != signature:
            raise RuntimeError("conditional structural ID collision or inconsistent owner signature")
        group = grouped.setdefault(selected, {"id": selected, "signature": signature,
                                              "sourceIds": [], "exactConfigIds": set(),
                                              "membershipClasses": set(),
                                              "terrainDeferred": row["terrainDeferred"]})
        group["sourceIds"].append(row["id"])
        group["exactConfigIds"].add(row["exactConfigId"])
        group["membershipClasses"].add(row["fullProductMembership"])
    result = []
    for group in grouped.values():
        result.append({"id": group["id"], "signature": group["signature"],
                       "sourceIds": sorted(group["sourceIds"]),
                       "exactConfigIds": sorted(group["exactConfigIds"]),
                       "sourceIntentCount": len(group["sourceIds"]),
                       "exactConfigCount": len(group["exactConfigIds"]),
                       "fullProductMember": any(value.startswith("retained_")
                                                for value in group["membershipClasses"]),
                       "membershipClasses": sorted(group["membershipClasses"]),
                       "proposedBetaMembership": "undecided_requires_product_scope_join",
                       "terrainDeferred": group["terrainDeferred"],
                       "engineering": "unreconciled", "admission": "not_checked_by_join",
                       "boundedEvidence": "not_bound_by_join", "campaign": "not_established",
                       "externalReview": "not_established",
                       "requiredEvidenceClasses": {gate: "required_unbound" for gate in EVIDENCE},
                       "nextTask": "I1-full-categorical-and-owner-specific-execution"})
    result.sort(key=lambda row: row["id"])
    validate_structural_provenance(result, rows)
    return result

def validate_structural_records(groups: list[dict], sources: list[dict]) -> None:
    ids = [row["id"] for row in groups]
    if len(ids) != len(set(ids)) or not ids:
        raise RuntimeError("duplicate or deleted conditional structural ID")
    mapped = [row for row in sources if row.get("canonicalStructuralId")]
    source_ids = [source_id for group in groups for source_id in group["sourceIds"]]
    if len(source_ids) != len(mapped) or len(set(source_ids)) != len(mapped) or \
            set(source_ids) != {row["id"] for row in mapped}:
        raise RuntimeError("conditional structural group has orphan, duplicated or skipped source")
    sources_by_id = {row["id"]: row for row in mapped}
    for group in groups:
        if stable_id("conditional-combination", group["signature"]) != group["id"]:
            raise RuntimeError("conditional structural stable ID changed")
        for source_id in group["sourceIds"]:
            source = sources_by_id[source_id]
            if source["canonicalStructuralId"] != group["id"] or \
                    source.get("structuralSignature") != group["signature"]:
                raise RuntimeError("conditional source reverse joined to a different structural owner: " + source_id)

def validate_structural_provenance(groups: list[dict], sources: list[dict]) -> None:
    validate_structural_records(groups, sources)
    mapped = {row["id"]: row for row in sources if row.get("canonicalStructuralId")}
    membership_classes = {
        "retained_maintained_execution_fixture",
        "retained_current_public_profile_fixture",
        "provisional_canonical_first_party_pending_review",
        "provisional_legacy_missing_stamp_pending_review",
    }
    for group in groups:
        owner_sources = [mapped[source_id] for source_id in group["sourceIds"]]
        for source in owner_sources:
            if not isinstance(source.get("exactConfigId"), str) or not source["exactConfigId"] or \
                    source.get("fullProductMembership") not in membership_classes or \
                    type(source.get("terrainDeferred")) is not bool:
                raise RuntimeError("conditional structural source provenance is missing or unreviewed: " + source["id"])
        expected_sources = sorted(source["id"] for source in owner_sources)
        expected_configs = sorted({source["exactConfigId"] for source in owner_sources})
        expected_classes = sorted({source["fullProductMembership"] for source in owner_sources})
        if group["sourceIds"] != expected_sources or group.get("exactConfigIds") != expected_configs:
            raise RuntimeError("conditional structural exact-config or source reverse coverage differs: " + group["id"])
        for name, expected in (("sourceIntentCount", len(expected_sources)), ("exactConfigCount", len(expected_configs))):
            if type(group.get(name)) is not int or group[name] != expected:
                raise RuntimeError("conditional structural provenance count differs: " + group["id"])
        expected_retained = any(value.startswith("retained_") for value in expected_classes)
        if group.get("membershipClasses") != expected_classes or \
                type(group.get("fullProductMember")) is not bool or group["fullProductMember"] is not expected_retained:
            raise RuntimeError("conditional structural retained/provisional membership differs: " + group["id"])
        terrain = owner_sources[0]["terrainDeferred"]
        if any(source["terrainDeferred"] is not terrain for source in owner_sources) or \
                type(group.get("terrainDeferred")) is not bool or group["terrainDeferred"] is not terrain:
            raise RuntimeError("conditional structural terrain provenance differs: " + group["id"])

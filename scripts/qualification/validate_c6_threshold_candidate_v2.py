#!/usr/bin/env python3
"""Fail-closed C6 release-threshold policy; the repository candidate is a draft."""

from __future__ import annotations

import argparse
from contextvars import ContextVar
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.qualification.provenance import release_manifest  # noqa: E402
from scripts.qualification import collect_c6_evidence as evidence  # noqa: E402
from scripts.qualification.validate_c6_threshold_candidate import (  # noqa: E402
    FIELDS,
    LEDGERS,
    REQUIRED_WORKLOADS,
    qualifies_memory_gain,
)


POLICY_SCHEMA = "nimbus.c6-release-threshold-candidate.v2"
RECEIPT_SCHEMA = "nimbus.c6-workload-receipt.v2"
LIMIT_SCHEMA = "nimbus.c6-reviewed-workload-limit.v1"
REVIEW_SCHEMA = "nimbus.c6-threshold-activation-review.v1"
HISTORICAL_V1_PATH = "tests/fixtures/reference/performance/threshold_policy_v1.json"
HISTORICAL_V1_SHA256 = "ee0670f93f860fd8db72075ba4893eb55899cc6e95e4764fb9527d98494cda95"
REQUIRED_MODES = {
    "cpu_milbrandt_24x24x20": ("cpu",),
    "cpu_active_32x32x24": ("cpu",),
    "cpu_student_64x64x48": ("cpu",),
    "cpu_workstation_128x128x64": ("cpu",),
    "metal_npy_18x18x18": ("metal",),
    "metal_netcdf_18x18x18": ("metal",),
    "vulkan_direct_12x12x48": ("vulkan_direct",),
    "vulkan_staged_12x12x48": ("vulkan_staged",),
    "zfp_history_64x64x48": ("cpu_writer_studio_reader",),
    "retained_terrain_postimplementation": ("cpu", "metal", "vulkan_staged"),
}
GPU_MODES = {"metal", "vulkan_direct", "vulkan_staged"}
VALIDATED_FILES: ContextVar[dict | None] = ContextVar("c6_validated_files", default=None)


class C6V2PolicyError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C6V2PolicyError(message)


def digest(path: Path) -> str:
    return release_manifest.sha256_file(path)


def valid_sha(value: object) -> bool:
    return release_manifest.valid_sha256(value)


def positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def nonnegative(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def close(left: object, right: float) -> bool:
    return type(left) in (int, float) and math.isfinite(left) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def record_validated_file(relative: str, sha256: str) -> None:
    files = VALIDATED_FILES.get()
    if files is not None:
        require(relative not in files or files[relative] == sha256, f"evidence identity drift during validation: {relative}")
        files[relative] = sha256


def linked_json(link: object, root: Path, label: str) -> dict:
    require(isinstance(link, dict) and set(link) == {"path", "sha256"}, f"invalid {label} link")
    require(valid_sha(link["sha256"]), f"invalid {label} hash")
    try:
        path, relative = release_manifest.checked_relative_file(Path(link["path"]), root)
        require(relative == link["path"], f"noncanonical {label} path")
        require(digest(path) == link["sha256"], f"stale {label} report")
        value = json.loads(path.read_text(encoding="utf-8"))
        require(digest(path) == link["sha256"], f"{label} changed while reading report")
        record_validated_file(relative, link["sha256"])
    except (OSError, ValueError, release_manifest.ManifestError) as error:
        raise C6V2PolicyError(f"invalid {label} report: {error}") from error
    require(isinstance(value, dict), f"invalid {label} root")
    return value


def file_identity(record: object, root: Path, label: str) -> dict:
    require(isinstance(record, dict) and set(record) == {"path", "sha256"}, f"invalid {label} identity")
    require(valid_sha(record["sha256"]), f"invalid {label} hash")
    try:
        path, relative = release_manifest.checked_relative_file(Path(record["path"]), root)
    except (OSError, release_manifest.ManifestError) as error:
        raise C6V2PolicyError(f"missing {label}: {error}") from error
    require(relative == record["path"] and digest(path) == record["sha256"], f"stale {label}")
    record_validated_file(relative, record["sha256"])
    return record


def canonical_source_seal(root: Path) -> dict:
    try:
        return evidence.source_seal(evidence.source_inventory(root))
    except (OSError, ValueError, release_manifest.ManifestError) as error:
        raise C6V2PolicyError(f"canonical release source inventory unavailable: {error}") from error


def hardware_identity(link: object, inline: object, schema: str, root: Path, label: str) -> dict:
    record = linked_json(link, root, label)
    try:
        identity = evidence.validate_identity(record, schema)
    except evidence.C6EvidenceError as error:
        raise C6V2PolicyError(f"invalid {label}: {error}") from error
    require(inline == identity, f"{label} artifact disagrees with profile")
    return record


def validate_trial_measurement(trial: dict, profile: dict, root: Path, label: str,
                               source_record: dict, host_record: dict, device_record: dict | None) -> set[str]:
    expected = {key: trial[key] for key in (
        "binarySha256", "sourceInventorySha256", "configSha256", "hostFingerprintSha256",
        "deviceFingerprintSha256", "mode", "grid",
    )}
    require(trial.get("identityBefore") == trial.get("identityAfter") == expected,
            f"trial before/after identity drift in {label}")
    record = linked_json(trial.get("measurementArtifact"), root, f"{label} measurement")
    payload = dict(trial)
    payload.pop("measurementArtifact", None)
    require(record.get("schema") == evidence.MEASUREMENT_SCHEMA and record.get("trial") == payload,
            f"raw measurement artifact disagrees in {label}")
    snapshots = [record.get("snapshotBefore"), record.get("snapshotAfter")]
    require(all(isinstance(snapshot, dict) and set(snapshot) == {
        "schema", "binary", "config", "source", "host", "device",
    } for snapshot in snapshots), f"missing before/after snapshot in {label}")
    require(snapshots[0] == snapshots[1], f"collection snapshot drift in {label}")
    snapshot = snapshots[0]
    require(snapshot["schema"] == evidence.SNAPSHOT_SCHEMA and snapshot["source"] == source_record and
            snapshot["host"] == host_record and snapshot["device"] == device_record and
            snapshot["binary"] == release_manifest.file_record(root / "bin/nimbus_sim", root) and
            snapshot["config"] == release_manifest.file_record(root / profile["config"]["path"], root),
            f"collection snapshot applicability mismatch in {label}")
    require(evidence.trial_identity(snapshot, profile["mode"], profile["grid"]) == expected,
            f"collection snapshot identity mismatch in {label}")
    raw = trial.get("rawArtifacts")
    require(isinstance(raw, dict) and "process" in raw and all(isinstance(key, str) and key.strip() for key in raw),
            f"missing per-trial raw artifacts in {label}")
    paths = set()
    for name, link in raw.items():
        file_identity(link, root, f"{label} raw {name}")
        require(link["path"] not in paths, f"duplicate raw artifact in {label}")
        paths.add(link["path"])
    process = linked_json(raw["process"], root, f"{label} process")
    require(process.get("schema") == evidence.PROCESS_SCHEMA and process.get("trialIndex") == trial["index"] and
            type(process.get("exitStatus")) is int and process["exitStatus"] == 0 and
            process.get("timedOut") is False and process.get("fallbackUsed") is False and
            process.get("identity") == expected and process.get("measurementWindow") == profile["measurementWindow"] and
            process.get("measurements") == {key: trial.get(key) for key in evidence.MEASURED_FIELDS} and
            process.get("rawLog") == trial.get("rawLog"), f"process measurement artifact disagrees in {label}")
    file_identity(trial.get("rawLog"), root, f"{label} trial log")
    for link in (trial["rawLog"], trial["measurementArtifact"], trial["validationArtifact"]):
        require(link["path"] not in paths, f"reused raw artifact role in {label}")
        paths.add(link["path"])
    return paths


def exact_science(value: object, root: Path, label: str) -> tuple[tuple, set[str]]:
    require(isinstance(value, dict), f"missing exact {label}")
    checkpoint = value.get("checkpoint")
    require(isinstance(checkpoint, dict) and set(checkpoint) == {"path", "sha256", "bytes"} and
            type(checkpoint.get("bytes")) is int and positive(checkpoint["bytes"]),
            f"missing exact {label} checkpoint")
    file_identity({key: checkpoint[key] for key in ("path", "sha256")}, root, f"{label} checkpoint")
    require((root / checkpoint["path"]).stat().st_size == checkpoint["bytes"], f"wrong {label} checkpoint bytes")
    fields = value.get("fields")
    ledgers = value.get("ledgers")
    require(isinstance(fields, dict) and set(fields) == FIELDS,
            f"missing exact {label} eight fields")
    require(isinstance(ledgers, dict) and set(ledgers) == LEDGERS,
            f"missing exact {label} six ledgers")
    paths = {checkpoint["path"]}
    for name, record in [*fields.items(), *ledgers.items()]:
        file_identity(record, root, f"{label} {name}")
        require(record["path"] not in paths, f"duplicate exact science artifact in {label}")
        paths.add(record["path"])
    identity = (checkpoint["sha256"], checkpoint["bytes"],
                tuple(sorted((name, record["sha256"]) for name, record in fields.items())),
                tuple(sorted((name, record["sha256"]) for name, record in ledgers.items())))
    return identity, paths


def validate_profile(profile: dict, limit: dict, row: dict, policy: dict, root: Path, source_record: dict) -> dict:
    source_sha = source_record["inventory"]["sha256"]
    mode = profile.get("mode")
    label = f"{row['id']}/{mode}"
    require(mode in REQUIRED_MODES[row["id"]], f"wrong backend mode: {label}")
    require(profile.get("binarySha256") == policy["releaseBinary"]["sha256"], f"stale binary in {label}")
    require(profile.get("sourceInventorySha256") == source_sha, f"stale source in {label}")
    config = file_identity(profile.get("config"), root, f"{label} config")
    if row.get("config") is not None:
        require(config["path"] == row["config"], f"wrong config in {label}")
    grid = profile.get("grid")
    require(isinstance(grid, list) and len(grid) == 3 and all(type(x) is int and x > 0 for x in grid),
            f"invalid grid in {label}")
    if row.get("grid") is not None:
        require(grid == row["grid"], f"wrong size in {label}")
    require(profile.get("fullFieldBytes") == math.prod(grid) * 4, f"wrong full-field bytes in {label}")
    host = profile.get("host")
    host_record = hardware_identity(profile.get("hostArtifact"), host, evidence.HOST_SCHEMA, root, f"{label} host")
    device = profile.get("device")
    device_record = None
    if mode in GPU_MODES:
        device_record = hardware_identity(profile.get("deviceArtifact"), device, evidence.DEVICE_SCHEMA, root,
                                          f"{label} device")
    else:
        require(device is None and profile.get("deviceArtifact") is None, f"unexpected device in {label}")
    reference_identity, science_paths = exact_science(profile.get("referenceScience"), root,
                                                      f"reference science in {label}")
    window = profile.get("measurementWindow")
    require(isinstance(window, dict) and positive(window.get("warmupSteps")) and
            positive(window.get("measuredSteps")) and positive(window.get("timeoutSeconds")),
            f"missing steady-state window in {label}")
    trials = profile.get("trials")
    require(isinstance(trials, list) and len(trials) >= 7, f"fewer than seven steady trials in {label}")
    require([trial.get("index") for trial in trials if isinstance(trial, dict)] == list(range(1, len(trials) + 1)),
            f"skipped or duplicate trial in {label}")
    times = []
    host_peaks = []
    device_peaks = []
    reader_peaks = []
    coverages = []
    raw_paths = set()
    for trial in trials:
        require(trial.get("status") == "completed" and trial.get("skipped") is False and
                trial.get("fallbackUsed") is False, f"skipped, failed, or fallback trial in {label}")
        require(trial.get("binarySha256") == profile["binarySha256"] and
                trial.get("sourceInventorySha256") == source_sha and
                trial.get("configSha256") == config["sha256"] and
                trial.get("hostFingerprintSha256") == host["fingerprintSha256"] and
                trial.get("deviceFingerprintSha256") == (device["fingerprintSha256"] if device else None) and
                trial.get("mode") == mode and trial.get("grid") == grid,
                f"trial identity drift in {label}")
        require(positive(trial.get("avgStepMs")) and positive(trial.get("stepWallSeconds")), f"invalid timing in {label}")
        coverage = trial.get("namedWallFraction")
        require(type(coverage) in (int, float) and math.isfinite(coverage) and .95 <= coverage <= 1,
                f"less than 95% named wall in {label}")
        require(positive(trial.get("peakHostBytes")), f"missing measured host peak in {label}")
        if mode in GPU_MODES:
            require(positive(trial.get("peakDeviceBytes")), f"missing measured device peak in {label}")
            device_peaks.append(trial["peakDeviceBytes"])
        else:
            require(trial.get("peakDeviceBytes") is None, f"inapplicable device peak in {label}")
        if mode == "cpu_writer_studio_reader":
            require(positive(trial.get("peakReaderHostBytes")) and positive(trial.get("readerBytesDecoded")),
                    f"missing ZFP reader resource receipt in {label}")
            reader_peaks.append(trial["peakReaderHostBytes"])
        else:
            require(trial.get("peakReaderHostBytes") is None, f"inapplicable reader peak in {label}")
        validation = trial.get("validation")
        require(isinstance(validation, dict) and validation.get("violations") == 0 and
                validation.get("exitStatus") == 0 and validation.get("skipped") == 0 and
                validation.get("trialIndex") == trial["index"] and
                validation.get("binarySha256") == profile["binarySha256"] and
                validation.get("sourceInventorySha256") == source_sha and
                validation.get("configSha256") == config["sha256"],
                f"nonzero validation in {label}")
        require(linked_json(trial.get("validationArtifact"), root, f"{label} validation") == validation,
                f"validation artifact disagrees in {label}")
        trial_identity, trial_paths = exact_science(trial.get("science"), root, f"trial science in {label}")
        require(trial_identity == reference_identity, f"exact science mismatch in {label}")
        require(not science_paths.intersection(trial_paths), f"reused science artifact across trials in {label}")
        require(not raw_paths.intersection(trial_paths), f"science artifact reused as earlier raw artifact in {label}")
        science_paths.update(trial_paths)
        trial_raw_paths = validate_trial_measurement(trial, profile, root, label, source_record, host_record, device_record)
        require(not raw_paths.intersection(trial_raw_paths), f"reused raw artifact across trials in {label}")
        require(not science_paths.intersection(trial_raw_paths), f"raw artifact reused as science in {label}")
        raw_paths.update(trial_raw_paths)
        times.append(trial["avgStepMs"])
        host_peaks.append(trial["peakHostBytes"])
        coverages.append(coverage)
    median = statistics.median(times)
    mad = statistics.median(abs(value - median) for value in times)
    cv = statistics.pstdev(times) / statistics.mean(times)
    aggregate = profile.get("aggregate")
    require(isinstance(aggregate, dict) and close(aggregate.get("medianStepMs"), median) and
            close(aggregate.get("madStepMs"), mad) and close(aggregate.get("cv"), cv) and
            aggregate.get("maxPeakHostBytes") == max(host_peaks) and
            aggregate.get("maxPeakDeviceBytes") == (max(device_peaks) if device_peaks else None) and
            aggregate.get("maxPeakReaderHostBytes") == (max(reader_peaks) if reader_peaks else None) and
            close(aggregate.get("minNamedWallFraction"), min(coverages)),
            f"variance or peak aggregate disagrees with trials in {label}")
    require(limit.get("mode") == mode and limit.get("binarySha256") == profile["binarySha256"] and
            limit.get("sourceInventorySha256") == source_sha and limit.get("configSha256") == config["sha256"] and
            limit.get("hostFingerprintSha256") == host["fingerprintSha256"] and
            limit.get("deviceFingerprintSha256") == (device["fingerprintSha256"] if device else None) and
            limit.get("grid") == grid, f"limit applicability mismatch in {label}")
    require(positive(limit.get("maxMedianStepMs")) and positive(limit.get("maxCv")) and
            positive(limit.get("maxPeakHostBytes")), f"unreviewed or placeholder limit in {label}")
    if mode in GPU_MODES:
        require(positive(limit.get("maxPeakDeviceBytes")), f"missing device peak limit in {label}")
    else:
        require(limit.get("maxPeakDeviceBytes") is None, f"inapplicable device peak limit in {label}")
    if mode == "cpu_writer_studio_reader":
        require(positive(limit.get("maxPeakReaderHostBytes")), f"missing reader peak limit in {label}")
    else:
        require(limit.get("maxPeakReaderHostBytes") is None, f"inapplicable reader peak limit in {label}")
    require(median <= limit["maxMedianStepMs"] and cv <= limit["maxCv"] and
            max(host_peaks) <= limit["maxPeakHostBytes"] and
            (not device_peaks or max(device_peaks) <= limit["maxPeakDeviceBytes"]) and
            (not reader_peaks or max(reader_peaks) <= limit["maxPeakReaderHostBytes"]),
            f"reviewed release threshold exceeded in {label}")
    return {"mode": mode, "trials": len(trials), "medianStepMs": median, "cv": cv,
            "minimumNamedWallFraction": min(coverages)}


def validate_repair_claim(claim: object, policy: dict) -> None:
    if claim is None:
        return
    require(isinstance(claim, dict), "invalid repair claim")
    kind = claim.get("kind")
    if kind == "paired_time_gain":
        gains = claim.get("pairedGainPercent")
        require(isinstance(gains, list) and len(gains) >= 7 and all(type(x) in (int, float) and math.isfinite(x) for x in gains),
                "invalid paired gain trials")
        median = statistics.median(gains)
        mad = statistics.median(abs(x - median) for x in gains)
        require(median > policy["acceptance"]["minimumWholeStepGainPercentExclusive"] and
                median > policy["acceptance"]["minimumPairedMadMultipleExclusive"] * mad,
                "paired time gain fails >2% and >3 MAD")
    elif kind == "live_memory_gain":
        require(all(type(claim.get(key)) is int for key in
                    ("fullFieldBytes", "admittedPeakBytes", "liveFullFieldsRemoved", "peakReductionBytes")) and
                type(claim.get("costShiftedOutsideMeasurement")) is bool,
                "invalid memory gain evidence")
        require(qualifies_memory_gain(claim.get("fullFieldBytes"), claim.get("admittedPeakBytes"),
                                      claim.get("liveFullFieldsRemoved"), claim.get("peakReductionBytes"),
                                      claim.get("costShiftedOutsideMeasurement")), "memory gain fails full-field/2% gate")
    else:
        raise C6V2PolicyError("unreviewed repair claim kind")


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
    validate_repair_claim(receipt.get("repairClaim"), policy)
    return results


def _validate_policy(policy: dict, root: Path, binary: Path, *, source_seal: dict | None = None) -> dict:
    require(policy.get("schema") == POLICY_SCHEMA, "wrong C6 v2 policy schema")
    require(policy.get("status") in {"draft_not_activated", "reviewed_active"}, "invalid C6 v2 status")
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
    parser.add_argument("--policy", type=Path, default=ROOT / "tests/fixtures/reference/performance/threshold_policy_v2.json")
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--require-release-ready", action="store_true")
    arguments = parser.parse_args()
    try:
        policy = json.loads(arguments.policy.read_text(encoding="utf-8"))
        result = validate_policy(policy, ROOT, arguments.binary)
        require(not arguments.require_release_ready or result["releaseReady"], "C6 release thresholds are not activated")
    except (OSError, KeyError, TypeError, ValueError, release_manifest.ManifestError) as error:
        parser.exit(1, f"C6 v2 threshold candidate rejected: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

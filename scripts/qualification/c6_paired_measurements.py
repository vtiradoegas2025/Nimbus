"""Versioned, artifact-bound paired time-gain proof; memory proof remains open."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True

from scripts.qualification import collect_c6_evidence as evidence
from scripts.qualification import validate_c6_threshold_candidate_v2 as common


CLAIM_SCHEMA = "nimbus.c6-paired-repair-claim.v3"
IDENTITY_KEYS = {
    "binarySha256", "sourceInventorySha256", "configSha256", "hostFingerprintSha256",
    "deviceFingerprintSha256", "mode", "grid",
}
COMPATIBLE_KEYS = {"configSha256", "hostFingerprintSha256", "deviceFingerprintSha256", "mode", "grid"}


def _measurement(link: object, root: Path, label: str) -> tuple[dict, dict, tuple, set[str]]:
    record = common.linked_json(link, root, label)
    common.require(record.get("schema") == evidence.MEASUREMENT_SCHEMA, f"wrong {label} schema")
    trial = record.get("trial")
    common.require(isinstance(trial, dict) and "measurementArtifact" not in trial, f"invalid {label} trial")
    common.require(type(trial.get("index")) is int and trial["index"] > 0
                   and trial.get("status") == "completed" and trial.get("skipped") is False
                   and trial.get("fallbackUsed") is False, f"incomplete {label} trial")
    expected = {key: trial.get(key) for key in IDENTITY_KEYS}
    common.require(trial.get("identityBefore") == trial.get("identityAfter") == expected,
                   f"before/after identity drift in {label}")
    snapshot = record.get("snapshotBefore")
    common.require(isinstance(snapshot, dict) and snapshot == record.get("snapshotAfter")
                   and set(snapshot) == {"schema", "binary", "config", "source", "host", "device"}
                   and snapshot.get("schema") == evidence.SNAPSHOT_SCHEMA,
                   f"snapshot drift or missing source in {label}")
    for key in ("binary", "config"):
        identity = snapshot[key]
        common.require(isinstance(identity, dict) and set(identity) == {"path", "bytes", "sha256"}
                       and isinstance(identity["path"], str) and identity["path"]
                       and type(identity["bytes"]) is int and identity["bytes"] > 0
                       and common.valid_sha(identity["sha256"]), f"invalid recorded {label} {key}")
    source = snapshot["source"]
    common.require(isinstance(source, dict) and set(source) == {"schema", "rootsDeclaration", "inventory"}
                   and source.get("schema") == evidence.SOURCE_SCHEMA
                   and isinstance(source.get("inventory"), dict), f"invalid recorded {label} source")
    inventory = source["inventory"]
    files = inventory.get("files")
    common.require(set(inventory) == {"roots", "files", "count", "sha256"}
                   and isinstance(files, list) and files and type(inventory.get("count")) is int
                   and inventory["count"] == len(files)
                   and all(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"}
                           and isinstance(item["path"], str) and item["path"]
                           and type(item["bytes"]) is int and item["bytes"] >= 0
                           and common.valid_sha(item["sha256"]) for item in files)
                   and len({item["path"] for item in files}) == len(files)
                   and files == sorted(files, key=lambda item: item["path"])
                   and isinstance(inventory.get("roots"), list) and inventory["roots"]
                   and all(isinstance(item, dict) and set(item) == {"path", "kind"}
                           and isinstance(item["path"], str) and item["path"]
                           and item["kind"] in ("file", "directory") for item in inventory["roots"])
                   and len({item["path"] for item in inventory["roots"]}) == len(inventory["roots"])
                   and source["rootsDeclaration"] in files
                   and inventory["sha256"] == hashlib.sha256(
                       common.release_manifest.canonical_bytes(files)).hexdigest(),
                   f"recorded source inventory seal differs in {label}")
    try:
        evidence.validate_identity(snapshot["host"], evidence.HOST_SCHEMA)
        if trial.get("mode") in common.GPU_MODES:
            evidence.validate_identity(snapshot["device"], evidence.DEVICE_SCHEMA)
        else:
            common.require(snapshot["device"] is None, f"unexpected recorded device in {label}")
    except evidence.C6EvidenceError as error:
        raise common.C6V2PolicyError(f"invalid recorded {label} hardware: {error}") from error
    grid = trial.get("grid")
    common.require(isinstance(grid, list) and len(grid) == 3
                   and all(type(value) is int and value > 0 for value in grid), f"invalid {label} grid")
    common.require(evidence.trial_identity(snapshot, trial.get("mode"), grid) == expected,
                   f"snapshot identity differs in {label}")
    common.require(common.positive(trial.get("avgStepMs")) and common.positive(trial.get("stepWallSeconds")),
                   f"invalid positive finite timing in {label}")
    named = trial.get("namedWallFraction")
    common.require(type(named) in (int, float) and math.isfinite(named) and .95 <= named <= 1,
                   f"less than 95% named wall in {label}")
    validation = common.linked_json(trial.get("validationArtifact"), root, f"{label} validation")
    common.require(validation == trial.get("validation")
                   and type(validation.get("violations")) is int and validation["violations"] == 0
                   and type(validation.get("exitStatus")) is int and validation["exitStatus"] == 0
                   and type(validation.get("skipped")) is int and validation["skipped"] == 0
                   and validation.get("trialIndex") == trial["index"]
                   and all(validation.get(key) == expected[key]
                           for key in ("binarySha256", "sourceInventorySha256", "configSha256")),
                   f"invalid scientific validation in {label}")
    science, paths = common.exact_science(trial.get("science"), root, label)
    raw = trial.get("rawArtifacts")
    common.require(isinstance(raw, dict) and "process" in raw
                   and all(isinstance(key, str) and key.strip() for key in raw), f"missing {label} raw artifacts")
    links = [link, trial["validationArtifact"], trial.get("rawLog"), *raw.values()]
    for item in links:
        common.file_identity(item, root, f"{label} artifact")
        common.require(item["path"] not in paths, f"reused artifact role in {label}")
        paths.add(item["path"])
    process = common.linked_json(raw["process"], root, f"{label} process")
    common.require(process.get("schema") == evidence.PROCESS_SCHEMA
                   and type(process.get("trialIndex")) is int and process["trialIndex"] == trial["index"]
                   and type(process.get("exitStatus")) is int and process["exitStatus"] == 0
                   and process.get("timedOut") is False and process.get("fallbackUsed") is False
                   and process.get("identity") == expected and process.get("rawLog") == trial.get("rawLog")
                   and process.get("measurements") == {key: trial.get(key) for key in evidence.MEASURED_FIELDS},
                   f"raw measurement disagrees in {label}")
    window = process.get("measurementWindow")
    common.require(isinstance(window, dict) and set(window) == {"warmupSteps", "measuredSteps", "timeoutSeconds"}
                   and all(type(window.get(key)) is int and window[key] > 0
                           for key in ("warmupSteps", "measuredSteps"))
                   and common.positive(window.get("timeoutSeconds")), f"invalid steady-state window in {label}")
    return trial, process, science, paths


def _validate_claim(claim: object, profiles: list[dict], policy: dict, root: Path) -> dict | None:
    if claim is None:
        return None
    common.require(isinstance(claim, dict) and claim.get("schema") == CLAIM_SCHEMA, "paired claim requires v3 schema")
    common.require(claim.get("kind") != "live_memory_gain",
                   "v3 live-memory proof is Engineering incomplete; unbound allocation counts are unsupported")
    common.require(set(claim) == {"schema", "kind", "profiles"} and claim["kind"] == "paired_time_gain",
                   "detached gain arrays or unreviewed paired claim fields")
    entries = claim["profiles"]
    common.require(isinstance(entries, list) and len(entries) == len(profiles)
                   and all(isinstance(entry, dict) and set(entry) == {"mode", "pairs"} for entry in entries),
                   "missing or duplicate paired profile")
    by_mode = {profile["mode"]: profile for profile in profiles}
    common.require(len(by_mode) == len(profiles)
                   and len({entry["mode"] for entry in entries}) == len(entries)
                   and {entry["mode"] for entry in entries} == set(by_mode), "paired profile modes differ")
    all_paths = set()
    results = {}
    for entry in entries:
        profile = by_mode[entry["mode"]]
        pairs = entry["pairs"]
        trials = profile["trials"]
        common.require(isinstance(pairs, list) and len(pairs) == len(trials) and len(pairs) >= 7,
                       "missing or duplicated paired trials")
        gains = []
        cohort = None
        reference_science, reference_paths = common.exact_science(profile.get("referenceScience"), root,
                                                                 f"paired {entry['mode']} reference")
        common.require(not all_paths.intersection(reference_paths), "reused profile reference artifact")
        all_paths.update(reference_paths)
        for index, (pair, candidate_trial) in enumerate(zip(pairs, trials), start=1):
            label = f"paired {entry['mode']} trial {index}"
            common.require(isinstance(pair, dict) and set(pair) == {"baseline", "candidate"}, f"invalid {label} links")
            common.require(pair["candidate"] == candidate_trial["measurementArtifact"], f"candidate order or artifact differs in {label}")
            baseline, baseline_process, baseline_science, baseline_paths = _measurement(pair["baseline"], root, f"{label} baseline")
            candidate, candidate_process, candidate_science, candidate_paths = _measurement(pair["candidate"], root, f"{label} candidate")
            expected_candidate = {key: value for key, value in candidate_trial.items() if key != "measurementArtifact"}
            common.require(candidate == expected_candidate, f"detached candidate measurement in {label}")
            expected_identity = {"binarySha256": profile["binarySha256"],
                "sourceInventorySha256": profile["sourceInventorySha256"],
                "configSha256": profile["config"]["sha256"],
                "hostFingerprintSha256": profile["host"]["fingerprintSha256"],
                "deviceFingerprintSha256": profile["device"]["fingerprintSha256"] if profile["device"] else None,
                "mode": profile["mode"], "grid": profile["grid"]}
            common.require(candidate["identityBefore"] == expected_identity, f"candidate profile applicability differs in {label}")
            common.require(baseline["index"] == candidate["index"] == index, f"swapped trial indices in {label}")
            common.require(all(baseline[key] == candidate[key] for key in COMPATIBLE_KEYS)
                           and baseline_process["measurementWindow"] == candidate_process["measurementWindow"] == profile["measurementWindow"],
                           f"incompatible config, mode, grid, hardware or window in {label}")
            common.require(baseline_science == candidate_science == reference_science, f"paired exact science differs in {label}")
            identities = (baseline["identityBefore"], candidate["identityBefore"])
            common.require(cohort is None or cohort == identities, f"source or binary cohort drift in {label}")
            cohort = identities
            common.require(not baseline_paths.intersection(candidate_paths)
                           and not all_paths.intersection(baseline_paths | candidate_paths), f"reused paired artifact in {label}")
            all_paths.update(baseline_paths | candidate_paths)
            gain = 100.0 * (1.0 - candidate["avgStepMs"] / baseline["avgStepMs"])
            common.require(math.isfinite(gain) and gain < 100.0, f"impossible paired time reduction in {label}")
            gains.append(gain)
        median = statistics.median(gains)
        mad = statistics.median(abs(value - median) for value in gains)
        common.require(median > policy["acceptance"]["minimumWholeStepGainPercentExclusive"]
                       and median > policy["acceptance"]["minimumPairedMadMultipleExclusive"] * mad,
                       "computed paired time gain fails >2% and >3 MAD")
        results[entry["mode"]] = {"pairs": len(gains), "pairedGainPercent": gains,
            "medianGainPercent": median, "madGainPercent": mad,
            "baselineIdentity": cohort[0], "candidateIdentity": cohort[1]}
    return {"schema": CLAIM_SCHEMA, "kind": "paired_time_gain", "profiles": results,
            "historicalBaselineSourceNotReboundCurrent": True, "memoryProofComplete": False}


def validate_claim(claim: object, profiles: list[dict], policy: dict, root: Path) -> dict | None:
    existing = common.VALIDATED_FILES.get()
    files = existing if existing is not None else {}
    token = common.VALIDATED_FILES.set(files) if existing is None else None
    try:
        result = _validate_claim(claim, profiles, policy, root)
        for relative, sha256 in files.items():
            common.file_identity({"path": relative, "sha256": sha256}, root, "final paired-proof seal")
        return result
    finally:
        if token is not None:
            common.VALIDATED_FILES.reset(token)

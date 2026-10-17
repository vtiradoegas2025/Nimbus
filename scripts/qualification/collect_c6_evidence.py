#!/usr/bin/env python3
"""Capture and seal C6 evidence identities without choosing performance limits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.provenance import release_manifest  # noqa: E402
from scripts.qualification.provenance.probe_backend_release_state import (  # noqa: E402
    canonical_release_source_roots,
)


SOURCE_SCHEMA = "nimbus.c6-source-inventory.v1"
HOST_SCHEMA = "nimbus.c6-host-identity.v1"
DEVICE_SCHEMA = "nimbus.c6-device-identity.v1"
SNAPSHOT_SCHEMA = "nimbus.c6-collection-snapshot.v1"
MEASUREMENT_SCHEMA = "nimbus.c6-trial-measurement.v1"
PROCESS_SCHEMA = "nimbus.c6-trial-process.v1"
HOST_FIELDS = {"os", "architecture", "cpuModel", "logicalThreads", "ramBytes", "hostname"}
DEVICE_FIELDS = {"name", "driver", "backendVersion"}
MEASURED_FIELDS = {
    "avgStepMs", "stepWallSeconds", "namedWallFraction", "peakHostBytes",
    "peakDeviceBytes", "peakReaderHostBytes", "readerBytesDecoded",
}


class C6EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C6EvidenceError(message)


def fingerprint(identity: dict) -> str:
    return hashlib.sha256(release_manifest.canonical_bytes(identity)).hexdigest()


def identity_record(identity: dict, schema: str) -> dict:
    record = {"schema": schema, "identity": identity, "fingerprintSha256": fingerprint(identity)}
    if schema == DEVICE_SCHEMA:
        record["evidenceKind"] = "supplied_identity_artifact"
    return record


def validate_identity(record: object, schema: str) -> dict:
    keys = {"schema", "identity", "fingerprintSha256"}
    if schema == DEVICE_SCHEMA:
        keys.add("evidenceKind")
    require(isinstance(record, dict) and set(record) == keys
            and record["schema"] == schema, "invalid hardware identity artifact")
    if schema == DEVICE_SCHEMA:
        require(record["evidenceKind"] == "supplied_identity_artifact", "unverified device identity provenance")
    identity = record["identity"]
    fields = HOST_FIELDS if schema == HOST_SCHEMA else DEVICE_FIELDS
    require(isinstance(identity, dict) and set(identity) == fields, "incomplete hardware identity")
    text_fields = fields - {"logicalThreads", "ramBytes"}
    require(all(isinstance(identity[key], str) and identity[key].strip() for key in text_fields),
            "missing hardware identity value")
    if schema == HOST_SCHEMA:
        require(all(type(identity[key]) is int and identity[key] > 0 for key in ("logicalThreads", "ramBytes")),
                "invalid host capacity")
    require(record["fingerprintSha256"] == fingerprint(identity), "hardware fingerprint does not match identity")
    return {**identity, "fingerprintSha256": record["fingerprintSha256"]}


def _sysctl(name: str) -> str:
    try:
        result = subprocess.run(["/usr/sbin/sysctl", "-n", name], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
    except subprocess.CalledProcessError as error:
        raise C6EvidenceError(f"host probe {name} failed: {error.stderr.strip()}") from error
    return result.stdout.strip()


def capture_host() -> dict:
    if sys.platform == "darwin":
        cpu = _sysctl("machdep.cpu.brand_string")
        memory = int(_sysctl("hw.memsize"))
    elif sys.platform.startswith("linux"):
        entries = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        models = [line.split(":", 1)[1].strip() for line in entries
                  if line.split(":", 1)[0].strip() in {"model name", "Hardware"} and ":" in line]
        cpu = models[0] if models else platform.processor()
        memory = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    else:
        raise C6EvidenceError(f"host fingerprint capture unavailable on {sys.platform}")
    record = identity_record({
        "os": platform.platform(), "architecture": platform.machine(), "cpuModel": cpu,
        "logicalThreads": os.cpu_count(), "ramBytes": memory, "hostname": platform.node(),
    }, HOST_SCHEMA)
    validate_identity(record, HOST_SCHEMA)
    return record


def source_inventory(root: Path, *, _roots: list[Path] | None = None) -> dict:
    require(_roots is not None or root == ROOT, "canonical source roots are defined only for this repository")
    roots = canonical_release_source_roots() if _roots is None else _roots
    declaration = release_manifest.file_record(root / "mk/tests.mk", root)
    inventory = release_manifest.inventory_record(roots, root)
    require(declaration == release_manifest.file_record(root / "mk/tests.mk", root),
            "source-root declaration changed during inventory capture")
    return {"schema": SOURCE_SCHEMA, "rootsDeclaration": declaration, "inventory": inventory}


def source_seal(source: dict) -> dict:
    inventory = source["inventory"]
    return {
        "rootsDeclarationPath": source["rootsDeclaration"]["path"],
        "rootsDeclarationSha256": source["rootsDeclaration"]["sha256"],
        "rootCount": len(inventory["roots"]), "fileCount": inventory["count"],
        "inventorySha256": inventory["sha256"], "record": source,
    }


def capture_snapshot(root: Path, binary: Path, config: Path, *, device: dict | None = None,
                     _roots: list[Path] | None = None) -> dict:
    if device is not None:
        validate_identity(device, DEVICE_SCHEMA)
    binary_record = release_manifest.file_record(binary, root)
    config_record = release_manifest.file_record(config, root)
    result = {
        "schema": SNAPSHOT_SCHEMA, "binary": binary_record, "config": config_record,
        "source": source_inventory(root, _roots=_roots), "host": capture_host(), "device": device,
    }
    require(binary_record == release_manifest.file_record(binary, root) and
            config_record == release_manifest.file_record(config, root),
            "binary or config changed during snapshot capture")
    return result


def trial_identity(snapshot: dict, mode: str, grid: list[int]) -> dict:
    return {
        "binarySha256": snapshot["binary"]["sha256"],
        "sourceInventorySha256": snapshot["source"]["inventory"]["sha256"],
        "configSha256": snapshot["config"]["sha256"],
        "hostFingerprintSha256": snapshot["host"]["fingerprintSha256"],
        "deviceFingerprintSha256": snapshot["device"]["fingerprintSha256"] if snapshot["device"] else None,
        "mode": mode, "grid": grid,
    }


def measurement_record(trial: dict, before: dict, after: dict) -> dict:
    require(before.get("schema") == after.get("schema") == SNAPSHOT_SCHEMA, "invalid collection snapshot")
    for key in ("binary", "config", "source", "host", "device"):
        require(before.get(key) == after.get(key), f"{key} drift during trial collection")
    expected = trial_identity(before, trial["mode"], trial["grid"])
    require(all(trial.get(key) == value for key, value in expected.items()), "trial identity differs from snapshot")
    require(all(key not in trial or trial[key] == expected for key in ("identityBefore", "identityAfter")),
            "supplied trial before/after identity differs from snapshot")
    payload = copy.deepcopy(trial)
    payload.pop("measurementArtifact", None)
    payload["identityBefore"] = expected
    payload["identityAfter"] = expected
    return {"schema": MEASUREMENT_SCHEMA, "trial": payload, "snapshotBefore": before, "snapshotAfter": after}


def validate_trial_artifacts(trial: dict, root: Path) -> None:
    def verify(link: dict) -> Path:
        require(isinstance(link, dict) and set(link) in ({"path", "sha256"}, {"path", "sha256", "bytes"}),
                "invalid raw evidence link")
        path, relative = release_manifest.checked_relative_file(Path(link["path"]), root)
        require(relative == link["path"] and release_manifest.sha256_file(path) == link["sha256"],
                f"stale raw evidence: {relative}")
        require("bytes" not in link or path.stat().st_size == link["bytes"], f"raw evidence size drift: {relative}")
        return path

    raw = trial.get("rawArtifacts")
    require(isinstance(raw, dict) and "process" in raw, "missing per-trial raw artifacts")
    science = trial.get("science")
    require(isinstance(science, dict) and isinstance(science.get("fields"), dict) and
            isinstance(science.get("ledgers"), dict), "missing per-trial science links")
    links = [trial.get("rawLog"), trial.get("validationArtifact"), *raw.values(), science.get("checkpoint"),
             *science["fields"].values(), *science["ledgers"].values()]
    paths = set()
    for link in links:
        path = verify(link)
        require(path not in paths, "reused per-trial artifact role")
        paths.add(path)
    validation = json.loads(verify(trial["validationArtifact"]).read_text(encoding="utf-8"))
    require(validation == trial.get("validation"), "raw validation artifact disagrees with trial")
    process = json.loads(verify(raw["process"]).read_text(encoding="utf-8"))
    require(process.get("schema") == PROCESS_SCHEMA and process.get("trialIndex") == trial["index"] and
            type(process.get("exitStatus")) is int and process["exitStatus"] == 0 and
            process.get("timedOut") is False and process.get("fallbackUsed") is False and
            process.get("measurements") == {key: trial.get(key) for key in MEASURED_FIELDS} and
            process.get("rawLog") == trial["rawLog"], "raw process artifact disagrees with trial")
    for link in links:
        verify(link)


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--trial", type=Path)
    arguments = parser.parse_args()
    try:
        require((arguments.before is None) == (arguments.trial is None), "--before and --trial must be supplied together")
        output = arguments.output.resolve()
        require(output.is_relative_to(ROOT / "reports"), "evidence output must be under reports, outside source roots")
        require(not output.exists(), "evidence output exists; choose a new version")
        device = json.loads(arguments.device_artifact.read_text(encoding="utf-8")) if arguments.device_artifact else None
        snapshot = capture_snapshot(ROOT, arguments.binary, arguments.config, device=device)
        value = snapshot
        if arguments.before is not None:
            before = json.loads(arguments.before.read_text(encoding="utf-8"))
            trial = json.loads(arguments.trial.read_text(encoding="utf-8"))
            validate_trial_artifacts(trial, ROOT)
            value = measurement_record(trial, before, snapshot)
        write_new_json(output, value)
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError, release_manifest.ManifestError) as error:
        parser.exit(1, f"C6 evidence capture rejected: {error}\n")
    print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": release_manifest.sha256_file(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

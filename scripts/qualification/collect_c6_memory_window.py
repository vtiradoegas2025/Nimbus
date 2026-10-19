#!/usr/bin/env python3
"""Produce artifact-bound Vulkan allocation-window measurements, not release approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import selectors
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification import collect_c6_evidence as evidence
from scripts.qualification.provenance import release_manifest

SCHEMA = "nimbus.c6-vulkan-memory-window-measurement.v1"
PROCESS_SCHEMA = "nimbus.c6-vulkan-memory-window-process.v1"
EVENT_SCHEMA = "nimbus.c6-vulkan-memory-window-event.v1"
PREFIX = "[C6 MEMORY WINDOW] "
SOURCE = "vulkan/include/compute/vulkan_compute_telemetry.hpp"
METRICS = {
    "staging_buffer_extent_bytes": "bytes", "device_buffer_extent_bytes": "bytes",
    "staging_memory_reserved_bytes": "bytes", "device_memory_reserved_bytes": "bytes",
    "staging_memory_allocation_count": "allocations", "device_memory_allocation_count": "allocations",
}
EVENT_KEYS = {"schema", "runToken", "pid", "owner", "source", "scope", "aggregation",
    "recordCapacity", "firstBoundary", "endBoundaryExclusive", "event", "boundary", "completed",
    "valid", "allocationHookEvents", "metrics"}
ENV_KEYS = ("NIMBUS_C6_MEMORY_RUN_TOKEN", "NIMBUS_C6_MEMORY_FIRST_BOUNDARY",
            "NIMBUS_C6_MEMORY_END_BOUNDARY_EXCLUSIVE")
LIMIT = (1 << 64) - 1


class MemoryEvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise MemoryEvidenceError(message)


def unsigned(value):
    return type(value) is int and 0 <= value <= LIMIT


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs)


def window(first, end):
    require(unsigned(first) and unsigned(end) and 0 < first < end, "invalid boundary window")
    return {"firstBoundary": first, "endBoundaryExclusive": end, "recordCapacity": 2,
            "scope": "backend_residency_boundaries"}


def parse_events(text, token, pid, first, end):
    requested = window(first, end)
    require(release_manifest.valid_sha256(token) and unsigned(pid) and pid > 0, "invalid process binding")
    records = []
    for line in text.splitlines():
        require("[C6 MEMORY WINDOW REJECTED]" not in line, "producer rejected request")
        if not line.startswith(PREFIX):
            continue
        require(len(records) < 2 and len(line) <= 16_384, "memory event capacity exceeded")
        try:
            record = strict_json(line[len(PREFIX):])
        except (ValueError, RecursionError) as error:
            raise MemoryEvidenceError("invalid memory event JSON") from error
        require(isinstance(record, dict) and set(record) == EVENT_KEYS, "invalid memory event fields")
        require(record["schema"] == EVENT_SCHEMA and record["runToken"] == token
                and type(record["pid"]) is int and record["pid"] == pid
                and record["owner"] == "VulkanComputeBackend" and record["source"] == SOURCE
                and record["aggregation"] == "separate_nonadditive_compartments"
                and all(type(record[key]) is type(value) and record[key] == value
                        for key, value in requested.items()), "wrong event source/process/window binding")
        require(record["valid"] is True and unsigned(record["allocationHookEvents"]),
                "invalid allocation accounting")
        expected_event, expected_boundary, completed = ("begin", first, False) if not records else ("end", end, True)
        require(record["event"] == expected_event and type(record["boundary"]) is int
                and record["boundary"] == expected_boundary and record["completed"] is completed,
                "incomplete or unordered memory window")
        require(isinstance(record["metrics"], dict) and set(record["metrics"]) == set(METRICS),
                "missing or additional allocation compartment")
        for name, unit in METRICS.items():
            metric = record["metrics"][name]
            require(isinstance(metric, dict) and set(metric) == {"unit", "begin", "current", "highWater"}
                    and metric["unit"] == unit and all(unsigned(metric[key]) for key in ("begin", "current", "highWater"))
                    and metric["highWater"] >= max(metric["begin"], metric["current"]), "invalid metric units/bounds")
            if not records:
                require(metric["begin"] == metric["current"] == metric["highWater"], "begin metric differs")
            else:
                require(metric["begin"] == records[0]["metrics"][name]["begin"], "metric source drift")
        if not records:
            require(record["allocationHookEvents"] == 0, "nonzero begin hook count")
        records.append(record)
    require(len(records) == 2, "truncated memory window")
    return records


def launch(command, environment, raw_path, timeout_s, max_log_bytes):
    require(type(max_log_bytes) is int and max_log_bytes > 0
            and type(timeout_s) in (int, float) and math.isfinite(timeout_s) and timeout_s > 0,
            "invalid capture resource bounds")
    started = time.monotonic()
    timed_out = truncated = False
    count = 0
    with raw_path.open("xb") as raw:
        process = subprocess.Popen(command, cwd=environment.pop("_C6_WORKDIR"), env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = timeout_s - (time.monotonic() - started)
                    if remaining <= 0:
                        timed_out = True
                        process.kill()
                        break
                    for key, _ in selector.select(min(remaining, 1)):
                        chunk = os.read(key.fileobj.fileno(), 65_536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        available = max_log_bytes - count
                        raw.write(chunk[:available])
                        count += min(len(chunk), available)
                        if len(chunk) > available:
                            truncated = True
                            process.kill()
                            break
                    if truncated:
                        break
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
    return {"pid": process.pid, "exitStatus": process.returncode, "timedOut": timed_out,
            "truncated": truncated, "capturedBytes": count}


def snapshot_members(snapshot):
    source = snapshot["source"]
    inventory = source["inventory"]
    files = inventory["files"]
    require(type(inventory["count"]) is int and inventory["count"] == len(files)
            and len({item["path"] for item in files}) == len(files)
            and files == sorted(files, key=lambda item: item["path"])
            and inventory["sha256"] == evidence.fingerprint(files) and source["rootsDeclaration"] in files,
            "invalid source inventory seal")
    members = {}
    for record in [*files, snapshot["binary"], snapshot["config"]]:
        require(set(record) == {"path", "bytes", "sha256"} and isinstance(record["path"], str)
                and unsigned(record["bytes"]) and release_manifest.valid_sha256(record["sha256"]),
                "invalid archived identity")
        require(record["path"] not in members or members[record["path"]] == record, "conflicting source member")
        members[record["path"]] = record
    require(SOURCE in members, "native measurement producer absent from source snapshot")
    return members


def archive_snapshot(root, snapshot, destination):
    with tarfile.open(destination, "x:gz") as archive:
        for relative, expected in sorted(snapshot_members(snapshot).items()):
            path, checked = release_manifest.checked_relative_file(Path(relative), root)
            require(checked == relative and release_manifest.file_record(path, root) == expected, "source drift before archive")
            archive.add(path, arcname=relative, recursive=False)


def verify_link(root, link):
    require(isinstance(link, dict) and set(link) == {"path", "bytes", "sha256"}, "invalid artifact link")
    path, relative = release_manifest.checked_relative_file(Path(link["path"]), root)
    require(relative == link["path"] and release_manifest.file_record(path, root) == link, "stale artifact")
    return path


def read_measurement(root, link):
    measurement = strict_json(verify_link(root, link).read_text())
    require(set(measurement) == {"schema", "identity", "window", "runToken", "snapshotBefore", "snapshotAfter",
            "sourceArchive", "process", "rawLog", "metrics", "releaseAcceptance", "scienceValidation"}
            and measurement["schema"] == SCHEMA and measurement["releaseAcceptance"] is False
            and measurement["scienceValidation"] == "not_performed", "invalid measurement scope")
    before = measurement["snapshotBefore"]
    require(before == measurement["snapshotAfter"] and before["schema"] == evidence.SNAPSHOT_SCHEMA,
            "source/process identity drift")
    evidence.validate_identity(before["host"], evidence.HOST_SCHEMA)
    evidence.validate_identity(before["device"], evidence.DEVICE_SCHEMA)
    identity = measurement["identity"]
    require(isinstance(identity, dict) and isinstance(identity.get("grid"), list) and len(identity["grid"]) == 3
            and all(type(value) is int and value > 0 for value in identity["grid"])
            and identity == evidence.trial_identity(before, "vulkan", identity["grid"]), "invalid measurement identity")
    members = snapshot_members(before)
    with tarfile.open(verify_link(root, measurement["sourceArchive"]), "r:gz") as archive:
        rows = archive.getmembers()
        require(len(rows) == len(members) and {row.name for row in rows} == set(members), "archive membership differs")
        for row in rows:
            expected = members[row.name]
            require(row.isfile() and row.size == expected["bytes"], "archive type/size differs")
            digest = hashlib.sha256()
            with archive.extractfile(row) as stream:
                while chunk := stream.read(65_536):
                    digest.update(chunk)
            require(digest.hexdigest() == expected["sha256"], "archived source differs")
    process = strict_json(verify_link(root, measurement["process"]).read_text())
    require(process["schema"] == PROCESS_SCHEMA and process["identity"] == identity
            and process["window"] == measurement["window"] and process["runToken"] == measurement["runToken"]
            and process["rawLog"] == measurement["rawLog"] and type(process["exitStatus"]) is int
            and process["exitStatus"] == 0 and process["timedOut"] is False and process["truncated"] is False,
            "invalid process evidence")
    require(process["environment"] == dict(zip(ENV_KEYS, (measurement["runToken"],
            str(measurement["window"]["firstBoundary"]), str(measurement["window"]["endBoundaryExclusive"]))))
            and process["command"][0] == before["binary"]["path"], "detached process request")
    require(isinstance(process.get("workingDirectory"), str) and Path(process["workingDirectory"]).is_absolute(),
            "missing captured process working directory")
    producer_root = Path(process["workingDirectory"])
    require(process["command"][1:5] == ["--headless", "--config", str(producer_root / before["config"]["path"]),
            "--compute-backend=vulkan"] and all(isinstance(value, str) for value in process["command"])
            and not any(value.split("=", 1)[0] in {"--config", "--compute-backend"}
                        for value in process["command"][5:]), "detached config/backend selection")
    raw = verify_link(root, measurement["rawLog"])
    require(type(process["capturedBytes"]) is int and process["capturedBytes"] == raw.stat().st_size, "raw byte count differs")
    requested = measurement["window"]
    require(requested == window(requested["firstBoundary"], requested["endBoundaryExclusive"]), "invalid window request")
    events = parse_events(raw.read_text(), measurement["runToken"], process["pid"],
                          requested["firstBoundary"], requested["endBoundaryExclusive"])
    require(measurement["metrics"] == events[1]["metrics"], "detached metric summary")
    require(len({link["path"], measurement["sourceArchive"]["path"], measurement["rawLog"]["path"],
                 measurement["process"]["path"]}) == 4, "duplicate evidence role")
    return measurement


def collect(root, binary, config, device, grid, first, end, arguments, output,
            timeout_s=60, max_log_bytes=16_777_216, *, _roots=None):
    requested = window(first, end)
    require(not output.exists(), "output exists; select a new measurement version")
    require(isinstance(grid, list) and len(grid) == 3
            and all(type(value) is int and value > 0 for value in grid), "invalid grid identity")
    require(isinstance(arguments, list) and arguments[:4] == ["--headless", "--config", str(config.resolve()),
            "--compute-backend=vulkan"] and all(isinstance(value, str) for value in arguments)
            and not any(value.split("=", 1)[0] in {"--config", "--compute-backend"} for value in arguments[4:]),
            "invalid config/backend command binding")
    before = evidence.capture_snapshot(root, binary, config, device=device, _roots=_roots)
    require(before["device"] is not None, "Vulkan measurement requires explicit device identity")
    output.mkdir(parents=True)
    archive = output / "source_and_binary.tar.gz"
    archive_snapshot(root, before, archive)
    token = secrets.token_hex(32)
    values = dict(zip(ENV_KEYS, (token, str(first), str(end))))
    environment = {**os.environ, **values, "_C6_WORKDIR": str(root)}
    command = [str(binary.resolve()), *arguments]
    raw = output / "process.log"
    status = launch(command, environment, raw, timeout_s, max_log_bytes)
    after = evidence.capture_snapshot(root, binary, config, device=device, _roots=_roots)
    identity = evidence.trial_identity(before, "vulkan", grid)
    process = {"schema": PROCESS_SCHEMA, **status, "identity": identity, "window": requested,
        "runToken": token, "command": [before["binary"]["path"], *arguments], "environment": values,
        "workingDirectory": str(root),
        "rawLog": release_manifest.file_record(raw, root)}
    process_path = output / "process.json"
    evidence.write_new_json(process_path, process)
    require(before == after, "identity drift during process")
    require(status["exitStatus"] == 0 and not status["timedOut"] and not status["truncated"], "incomplete process")
    events = parse_events(raw.read_text(), token, status["pid"], first, end)
    measurement = {"schema": SCHEMA, "identity": identity, "window": requested, "runToken": token,
        "snapshotBefore": before, "snapshotAfter": after, "sourceArchive": release_manifest.file_record(archive, root),
        "process": release_manifest.file_record(process_path, root), "rawLog": process["rawLog"],
        "metrics": events[1]["metrics"], "releaseAcceptance": False, "scienceValidation": "not_performed"}
    destination = output / "measurement.json"
    evidence.write_new_json(destination, measurement)
    link = release_manifest.file_record(destination, root)
    read_measurement(root, link)
    return link


def compatible_pair(root, baseline_link, candidate_link):
    require(baseline_link != candidate_link, "reused paired measurement")
    baseline = read_measurement(root, baseline_link)
    candidate = read_measurement(root, candidate_link)
    baseline_process = strict_json(verify_link(root, baseline["process"]).read_text())
    candidate_process = strict_json(verify_link(root, candidate["process"]).read_text())
    require(baseline["runToken"] != candidate["runToken"]
            and baseline["window"] == candidate["window"]
            and baseline_process["command"][1:] == candidate_process["command"][1:]
            and all(baseline["identity"][key] == candidate["identity"][key]
            for key in ("configSha256", "hostFingerprintSha256", "deviceFingerprintSha256", "mode", "grid")),
            "incompatible paired measurement")
    return {"compatibleAllocationWindows": True, "releaseAcceptance": False, "scienceValidation": "not_performed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device-artifact", type=Path, required=True)
    parser.add_argument("--grid", type=int, nargs=3, required=True)
    parser.add_argument("--first-boundary", type=int, required=True)
    parser.add_argument("--end-boundary-exclusive", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-log-bytes", type=int, default=16_777_216)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runtime_arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    try:
        output = options.output.resolve()
        require(output.is_relative_to(ROOT / "reports"), "output must be under reports")
        arguments = options.runtime_arguments
        if arguments[:1] == ["--"]:
            arguments = arguments[1:]
        command = ["--headless", "--config", str(options.config.resolve()), "--compute-backend=vulkan", *arguments]
        device = json.loads(options.device_artifact.read_text())
        link = collect(ROOT, options.binary.resolve(), options.config.resolve(), device, options.grid,
                       options.first_boundary, options.end_boundary_exclusive, command, output,
                       options.timeout, options.max_log_bytes)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, release_manifest.ManifestError) as error:
        parser.exit(1, f"memory capture rejected: {error}\n")
    print(json.dumps(link, sort_keys=True))


if __name__ == "__main__":
    main()

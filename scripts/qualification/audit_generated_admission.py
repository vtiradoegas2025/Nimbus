#!/usr/bin/env python3
"""Classify the generated tuple inventory without replacing qualification pins."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
STATUSES = {"science_rejected", "preflight_rejected", "preflight_admitted"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(generated: Path, native: Path, binary: Path, output: Path) -> dict:
    paths = [generated, native, binary, Path(__file__).resolve(),
             ROOT / "tests/compute/generated_backend_tuple_catalog_main.cpp",
             ROOT / "tests/compute/native_admission_batch_main.cpp",
             ROOT / "tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml",
             ROOT / "tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt",
             ROOT / "tests/fixtures/configs/qualification/soundings/catalog_cm1_profile.txt"]
    identities = {str(path): sha(path) for path in paths}
    first = subprocess.run([str(generated)], cwd=ROOT, capture_output=True, check=True).stdout
    catalog = json.loads(first)
    expected_count = catalog["summary"]["accepted"]
    expected_digest = catalog["acceptedTupleSha256"]
    if (catalog.get("schema") != "nimbus.generated-backend-tuples.v1"
            or not isinstance(expected_count, int) or expected_count <= 0
            or sum(catalog["partitions"].values()) != expected_count):
        raise RuntimeError("generated catalog has inconsistent accounting")
    output.mkdir(parents=True, exist_ok=False)
    (output / "catalog.json").write_bytes(first)
    temporary = output / "classifications.jsonl.gz.partial"
    processes = []
    counts = Counter({status: 0 for status in sorted(STATUSES)})
    partitions = defaultdict(Counter)
    input_partitions = Counter()
    reasons = {}
    axes = defaultdict(lambda: defaultdict(Counter))
    tuples = set()
    configs = set()
    digest = hashlib.sha256()
    try:
        with (output / "producer.stderr").open("wb") as producer_error, \
                (output / "consumer.stderr").open("wb") as consumer_error:
            producer = subprocess.Popen([str(generated), "--tuples"], cwd=ROOT,
                                        stdout=subprocess.PIPE, stderr=producer_error)
            processes.append(producer)
            consumer = subprocess.Popen(
                [str(native), "--classify", "--expected-count", str(expected_count),
                 "--expected-sha256", expected_digest], cwd=ROOT,
                stdin=producer.stdout, stdout=subprocess.PIPE, stderr=consumer_error)
            processes.append(consumer)
            assert producer.stdout is not None and consumer.stdout is not None
            producer.stdout.close()
            with gzip.open(temporary, "wb", compresslevel=4) as evidence:
                for raw in consumer.stdout:
                    row = json.loads(raw)
                    framed = (row["tuple"] + "\n").encode("ascii")
                    identity = hashlib.sha256(framed).digest()
                    if identity in tuples:
                        raise RuntimeError("duplicate canonical tuple in admission classification")
                    tuples.add(identity)
                    digest.update(framed)
                    status = row["admissionStatus"]
                    if status not in STATUSES or bool(row["reason"]) == (status == "preflight_admitted"):
                        raise RuntimeError("inconsistent admission classification")
                    values = dict(field.split("=", 1) for field in row["tuple"].split("|"))
                    expected_partition = (f"{values['grid']}.{values['backend']}."
                                          + ("qualified" if values["qualified"] == "true" else "exploratory"))
                    if row["partition"] != expected_partition:
                        raise RuntimeError("classification partition differs from the tuple")
                    partition = row["partition"]
                    input_partition = (partition.removesuffix(".exploratory")
                                       if partition.startswith("collocated.") else partition)
                    input_partitions[input_partition] += 1
                    counts[status] += 1
                    partitions[partition][status] += 1
                    for axis, value in values.items():
                        axes[axis][value][status] += 1
                    if row["reason"]:
                        key = (status, row["reason"])
                        item = reasons.setdefault(key, {"stage": status, "reason": row["reason"],
                                                        "count": 0, "representative": row["tuple"]})
                        item["count"] += 1
                    if row["configSha256"]:
                        configs.add(row["configSha256"])
                    if status == "preflight_admitted" and len(row["configSha256"]) != 64:
                        raise RuntimeError("admitted tuple omitted its serialized configuration identity")
                    if values.get("sounding") == "cm1" and not row["soundingPath"].endswith("catalog_cm1_profile.txt"):
                        raise RuntimeError("CM1 tuple selected a different sounding fixture")
                    evidence.write(raw)
                    if len(tuples) % 10000 == 0:
                        print(f"classified {len(tuples)}/{expected_count}: {dict(counts)}", flush=True)
            consumer.stdout.close()
            consumer_status = consumer.wait()
            producer_status = producer.wait()
        if producer_status or consumer_status:
            raise RuntimeError(f"inventory pipeline failed: producer={producer_status}, "
                               f"consumer={consumer_status}; see stderr artifacts in {output}")
        if (len(tuples) != expected_count or digest.hexdigest() != expected_digest
                or dict(input_partitions) != catalog["partitions"]):
            raise RuntimeError("classified stream does not match the complete generated inventory")
        if identities != {str(path): sha(path) for path in paths}:
            raise RuntimeError("audit inputs changed during classification")
        classifications = output / "classifications.jsonl.gz"
        temporary.replace(classifications)
        summary = {
            "schema": "nimbus.generated-admission-audit.v1",
            "status": "inventory_classified",
            "readinessStatus": "Engineering incomplete",
            "inputTuples": len(tuples), "inputTupleSha256": digest.hexdigest(),
            "counts": dict(counts), "partitions": dict(partitions), "axes": dict(axes),
            "distinctSerializedConfigs": len(configs),
            "rejections": sorted(reasons.values(), key=lambda row: (-row["count"], row["reason"])),
            "inputs": identities,
            "classifications": {"path": classifications.name, "sha256": sha(classifications)},
            "scope": {"domainMotion": "stationary", "nesting": False,
                      "nativeCommitFlags": False, "deviceAxis": "generic gpu",
                      "numericParameters": "canonical materialization, not all supported parameter profiles"},
            "claimBoundary": "Exhaustive admission/preflight classification of this generated universe; "
                             "not timestep execution, complete intended-combination coverage, readiness, "
                             "a long campaign, or external review. Historical qualification pins are unchanged.",
        }
        pending = output / "summary.json.partial"
        pending.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        pending.replace(output / "summary.json")
        return summary
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-tool", type=Path, default=ROOT / "bin/nimbus_generated_backend_tuples")
    parser.add_argument("--native-tool", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.generated_tool.resolve(), args.native_tool.resolve(),
                   args.binary.resolve(), args.output.resolve())
    print(f"generated admission audit complete: {result['inputTuples']} tuples; {result['counts']}")


if __name__ == "__main__":
    main()

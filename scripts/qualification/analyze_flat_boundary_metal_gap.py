#!/usr/bin/env python3
"""Classify guarded fixed-boundary Metal IDs against exact native/public owners."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


OWNERS = (
    "src/core/runtime/runtime_config.cpp",
    "include/core/runtime/runtime_config.hpp",
    "src/core/runtime/nimbus_sim.cpp",
    "src/numerics/advection/advection_batch.cpp",
    "src/dynamics/native_cgrid_boundary.cpp",
    "src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
    "metal/src/compute/backend/advection.inc",
    "metal/src/compute/backend/dynamics/acoustic_cgrid_batched.inc",
    "metal/include/compute/metal/shaders/advection/conservative_cgrid_transport_msl.hpp",
    "metal/include/compute/metal/shaders/dynamics/acoustic_bc_face_cartesian_cgrid_msl.hpp",
)


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def category(selectors: dict) -> str:
    lateral = selectors["lateral"]
    values = tuple(lateral[face] for face in ("west", "east", "south", "north"))
    if values == ("rigid",) * 4 and selectors["bottom"] == "rigid_slip" and \
            selectors["top"] == "rigid":
        return "canonical_native_and_public"
    if values == ("rigid_slip",) * 4 and selectors["bottom"] == "rigid_noslip" and \
            selectors["top"] == "open":
        return "structural_native_but_public_guarded"
    return "not_structural_native_explicit_host_transport"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combination-map", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    source_path = args.combination_map.resolve(strict=True)
    source = json.loads(source_path.read_text())
    if source.get("schema") != "nimbus.fixed-boundary-combination-evidence-map.v1" or \
            source.get("combinationCount") != 4608 or \
            source.get("currentBinarySha256") != identity(binary)["sha256"] or \
            source.get("I1Complete") is not False:
        raise RuntimeError("Metal gap analysis requires the exact current combination map")
    records_path = Path(source["records"]["path"]).resolve(strict=True)
    if identity(records_path)["sha256"] != source["records"]["sha256"]:
        raise RuntimeError("fixed boundary combination records changed")
    inputs = {"binary": identity(binary), "combinationMap": identity(source_path),
              "combinationRecords": identity(records_path), "analyzer": identity(Path(__file__))}
    inputs.update({path: identity(ROOT / path) for path in OWNERS})
    reviewed = []
    stream_digest = hashlib.sha256()
    with gzip.open(records_path, "rb") as stream:
        for line in stream:
            stream_digest.update(line)
            row = json.loads(line)
            if row["backend"] != "metal":
                continue
            group = category(row["selectors"])
            expected = "admitted" if group == "canonical_native_and_public" else "rejected"
            if row["publicAdmission"]["status"] != expected or \
                    row["status"]["engineering"] != "engineering_incomplete":
                raise RuntimeError("Metal native/public owner classification disagreed with actual admission")
            reviewed.append({"id": row["id"], "familyId": row["familyId"],
                             "ownerClass": group,
                             "publicAdmission": row["publicAdmission"],
                             "sourceYamlSha256": row["sourceYamlSha256"],
                             "canonicalYamlSha256": row["canonicalYamlSha256"],
                             "physicalBoundaryClosureId": row["physicalBoundaryClosureId"],
                             "requiredEvidenceClasses": row["requiredEvidenceClasses"],
                             "nextTask": "R4-current-binary-physical-regression"
                             if group == "canonical_native_and_public" else
                             "R4-promote-proven-structural-mixed-profile"
                             if group == "structural_native_but_public_guarded" else
                             "R4-expand-native-TVD-structural-ownership"})
    if stream_digest.hexdigest() != source["records"]["uncompressedSha256"] or \
            len(reviewed) != 1536 or len({row["id"] for row in reviewed}) != 1536:
        raise RuntimeError("Metal gap analysis omitted a fixed-boundary selector")
    counts = Counter(row["ownerClass"] for row in reviewed)
    if counts != {"canonical_native_and_public": 1,
                  "structural_native_but_public_guarded": 1,
                  "not_structural_native_explicit_host_transport": 1534}:
        raise RuntimeError("Metal structural/public ownership partition changed")
    class_counts = Counter(name for row in reviewed for name in row["requiredEvidenceClasses"])
    records = output / "metal_gap_records.jsonl.gz"
    uncompressed = hashlib.sha256()
    with records.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        for row in reviewed:
            line = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            stream.write(line)
            uncompressed.update(line)
    if any(identity(ROOT / path) != inputs[path] for path in OWNERS) or \
            identity(binary) != inputs["binary"] or \
            identity(source_path) != inputs["combinationMap"]:
        raise RuntimeError("Metal gap source or binary changed during analysis")
    report = {"schema": "nimbus.flat-boundary-metal-gap-decision.v1",
              "status": "source_reviewed_open",
              "claimBoundary": "Exact fixed dry Metal boundary owner/admission partition and repair plan; source interpretation, not all-profile native execution, Ready or scientific qualification.",
              "inputs": inputs, "metalProfiles": len(reviewed),
              "guardedProfiles": 1535,
              "ownerClassCounts": dict(sorted(counts.items())),
              "requiredEvidenceClassCounts": dict(sorted(class_counts.items())),
              "records": {**identity(records),
                          "uncompressedSha256": uncompressed.hexdigest()},
              "decision": {
                  "first": "Keep the 1,535 qualified Metal profiles guarded. The existing native TVD structural predicate admits only the canonical closed and all-rigid-slip/no-slip/open topologies; the public generated-Metal hash scope includes only the canonical one.",
                  "second": "For the one structural-native guarded mixed-open profile, reuse the existing strict exploratory native Metal parity/thread/restart and closure witnesses only after rebinding exact current source, then add an exact versioned public scope entry with negative neighbors and CLI/output gates.",
                  "third": "For the other 1,534 profiles, widen the separately reviewed native TVD structural boundary predicate only after identical-input acoustic/boundary/transport references, radiative-history/zero-gradient/open-top flux attribution, mixed-corner conservation, late-failure rollback, memory/kernel preflight and CPU/Vulkan/Metal parity are proven by owner classes. The Metal acoustic boundary ABI already accepts enum values 0..4; that is not by itself complete numerical ownership.",
                  "fourth": "After structural proof, generate exact public hashes for the fixed family, retain invalid/alias/adjacent negative scopes, run complete ordinary qualified admission and one-second physical Metal coverage, plus representative thread/restart/cadence and resource gates. Do not generalize from this fixed 18-cubed NPY family to numeric/output/motion variants."},
              "I1Complete": False, "wholeBackendReady": False}
    write_json_atomic(output / "summary.json", report)
    print(json.dumps({"status": report["status"], "profiles": len(reviewed),
                      "ownerClasses": report["ownerClassCounts"]}), flush=True)


if __name__ == "__main__":
    main()

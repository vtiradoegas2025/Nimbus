#!/usr/bin/env python3
"""Verify complete, disjoint maintained-config shards and merge canonical identities."""

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
from scripts.qualification.catalog_inventory import maintained_cases
from scripts.qualification.materialize_retained_fixtures import identity, sha
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


def validate_shard_headers(documents: list[dict]) -> dict:
    if not documents:
        raise RuntimeError("no maintained-configuration shard summaries supplied")
    first = documents[0]
    count = first.get("shardCount")
    if not isinstance(count, int) or count < 1 or len(documents) != count:
        raise RuntimeError("maintained-configuration merge has a skipped or extra shard")
    if not isinstance(first.get("sourceCount"), int) or first["sourceCount"] < 1:
        raise RuntimeError("maintained-configuration source count is invalid")
    if {document.get("shardIndex") for document in documents} != set(range(count)):
        raise RuntimeError("maintained-configuration shard IDs are missing or duplicated")
    if any(document.get("schema") != "nimbus.retained-fixture-combination-shard.v1" or
           document.get("status") != "complete" or document.get("shardCount") != count or
           document.get("sourceCount") != first.get("sourceCount") or
           document.get("inputs") != first.get("inputs") or
           document.get("catalogSha256") != first.get("catalogSha256") for document in documents):
        raise RuntimeError("maintained-configuration shard schema or source identities differ")
    return first


def validate_shards(summaries: list[Path], binary: Path, native: Path) -> tuple[list[dict], dict]:
    documents = [json.loads(path.read_text()) for path in summaries]
    first = validate_shard_headers(documents)
    count = first["shardCount"]
    if identity(binary) != first["inputs"]["binary"] or identity(native) != first["inputs"]["native"]:
        raise RuntimeError("maintained-configuration shards bind a stale binary or native materializer")
    for source in first["inputs"].values():
        if identity(ROOT / source["path"]) != source:
            raise RuntimeError(f"maintained-configuration input changed: {source['path']}")
    qualified, public = maintained_cases()
    expected = sorted([(path, "qualified_fixture") for path in qualified] +
                      [(path, "public_preset") for path in public],
                      key=lambda pair: pair[0].relative_to(ROOT).as_posix())
    if len(expected) != first["sourceCount"]:
        raise RuntimeError("maintained execution manifest cardinality changed")
    rows = {}
    for summary_path, document in zip(summaries, documents):
        index = document["shardIndex"]
        named_path = Path(document["records"]["path"])
        path = (named_path if named_path.is_absolute() else summary_path.parent / named_path).resolve(strict=True)
        if sha(path) != document["records"]["sha256"] or path.stat().st_size != document["records"]["bytes"]:
            raise RuntimeError(f"maintained-configuration shard {index} record file changed")
        record_hash = hashlib.sha256()
        seen = []
        with gzip.open(path, "rb") as stream:
            for line in stream:
                record_hash.update(line)
                row = json.loads(line)
                source_index = row["sourceIndex"]
                if source_index in rows or source_index % count != index or \
                        not 0 <= source_index < len(expected):
                    raise RuntimeError("maintained-configuration record is duplicated or in the wrong shard")
                source_path, kind = expected[source_index]
                if row["schema"] != "nimbus.retained-fixture-combination.v1" or \
                        row["id"] != "fixture:" + source_path.relative_to(ROOT).as_posix() or \
                        row["sourceKind"] != kind or row["source"] != identity(source_path) or \
                        row["effectiveConfigId"] != "config:" + row["canonicalYamlSha256"]:
                    raise RuntimeError("maintained-configuration record does not reverse-map to its source")
                rows[source_index] = row
                seen.append(source_index)
        if seen != document["selectedSourceIndices"] or len(seen) != document["selectedCount"] or \
                record_hash.hexdigest() != document["records"]["uncompressedSha256"]:
            raise RuntimeError("maintained-configuration shard record order or digest changed")
    if set(rows) != set(range(len(expected))):
        raise RuntimeError("maintained-configuration merge omitted a retained source")
    return [rows[index] for index in range(len(expected))], first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-summary", type=Path, action="append", required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--native", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [path.resolve(strict=True) for path in args.shard_summary]
    binary = args.binary.resolve(strict=True)
    native = args.native.resolve(strict=True)
    rows, first = validate_shards(summaries, binary, native)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = output / "records.jsonl.gz"
    record_hash = hashlib.sha256()
    with records.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        for row in rows:
            line = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            stream.write(line)
            record_hash.update(line)
    families = Counter(row["familyId"] for row in rows)
    backends = Counter(row["physicalBackend"] for row in rows)
    backend_aliases = Counter(row["backendAlias"] for row in rows if row["backendAlias"] is not None)
    unresolved = Counter(key for row in rows for key in row["conditionalOrUnresolvedSelectors"])
    inactive = Counter(reason for row in rows for reason in row["inactiveConditionalValues"].values())
    ids = Counter(row["effectiveConfigId"] for row in rows)
    conditional = Counter(selection["source"] for row in rows
                          for selection in row["catalogSelections"].values()
                          if selection["source"] == "conditional_auto_by_grid_staggering")
    product_sets = [row["source"]["path"] for row in rows
                    if row["catalogSelections"].get("radar.products", {}).get("productIds")]
    products = Counter(product for row in rows
                       for product in row["catalogSelections"].get("radar.products", {}).get("productIds", []))
    report = {
        "schema": "nimbus.retained-fixture-combination-merge.v1",
        "status": "complete",
        "claimBoundary": "Complete exact-loader identities for maintained YAMLs only; unenumerated retained combinations and all Ready gates remain open.",
        "binary": identity(binary),
        "nativeMaterializer": identity(native),
        "inputs": first["inputs"],
        "mergeTool": identity(Path(__file__)),
        "catalogSha256": first["catalogSha256"],
        "shardCount": len(summaries),
        "shards": [{"summary": str(path), "sha256": sha(path)} for path in sorted(summaries)],
        "sourceCount": len(rows),
        "uniqueEffectiveConfigs": len(ids),
        "duplicateEffectiveConfigGroups": sum(value > 1 for value in ids.values()),
        "familyCounts": dict(sorted(families.items())),
        "physicalBackendCounts": dict(sorted(backends.items())),
        "backendAliasCounts": dict(sorted(backend_aliases.items())),
        "conditionalOrUnresolvedSelectorCounts": dict(sorted(unresolved.items())),
        "conditionalResolutionCounts": dict(sorted(conditional.items())),
        "activeProductSetPaths": product_sets,
        "selectedRadarProductCounts": dict(sorted(products.items())),
        "productSetCombinationExpansionPending": True,
        "inactiveConditionalReasonCounts": dict(sorted(inactive.items())),
        "records": {"path": records.name, "bytes": records.stat().st_size,
                    "sha256": sha(records), "uncompressedSha256": record_hash.hexdigest()},
        "scope": {"I1Complete": False, "wholeBackendReady": False,
                  "campaignValidated": False, "externallyReviewed": False},
    }
    write_json_atomic(output / "summary.json", report)
    print(json.dumps({"status": "complete", "sources": len(rows),
                      "uniqueEffectiveConfigs": len(ids), "unresolvedSelectors": dict(unresolved)}))


if __name__ == "__main__":
    main()

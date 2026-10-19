#!/usr/bin/env python3
"""Reverse-audit all first-party YAML sources beyond the maintained execution set."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.backend_completion_inventory import explicit_yaml_scalars_text
from scripts.qualification.materialize_retained_fixtures import identity
from scripts.qualification.catalog_inventory import maintained_cases
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


def source_paths() -> list[Path]:
    paths = sorted({*ROOT.joinpath("configs").rglob("*.yaml"),
                    *ROOT.joinpath("tests/fixtures").rglob("*.yaml")},
                   key=lambda path: path.relative_to(ROOT).as_posix())
    if not paths or any(not path.is_file() or not path.is_relative_to(ROOT) for path in paths):
        raise RuntimeError("first-party YAML source discovery failed")
    return paths


def classify(path: Path, native: Path, axes: dict[str, dict]) -> dict:
    source = path.read_text()
    explicit = explicit_yaml_scalars_text(source, str(path))
    result = subprocess.run([str(native), "--canonical-config", str(path)], cwd=ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=30, check=False,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    row = {"id": "source:" + path.relative_to(ROOT).as_posix(),
           "source": identity(path),
           "explicitAxisValues": {key: explicit[key] for key in axes if key in explicit}}
    if result.returncode != 0:
        row.update({"strictLoader": "rejected", "exitCode": result.returncode,
                    "stderrSha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
                    "rejectionExcerpt": result.stderr[:500],
                    "retainedDisposition": "needs_source_intent_review",
                    "evidenceClass": "strict_loader_rejection_only"})
        return row
    if not result.stdout.endswith("\n") or "\nschema: nimbus.v1\n" not in result.stdout:
        raise RuntimeError(f"strict-loader serialization is malformed for {path}")
    effective = explicit_yaml_scalars_text(result.stdout, str(path) + ":canonical")
    unexpected = {key: value for key in axes if (value := effective.get(key)) is not None
                  and value not in axes[key]["allowed"]}
    if unexpected:
        raise RuntimeError(f"strict-loader emitted unexpected categorical values: {path}: {unexpected}")
    row.update({"strictLoader": "canonicalized",
                "effectiveConfigId": "config:" + hashlib.sha256(result.stdout.encode()).hexdigest(),
                "effectiveAxisValues": {key: effective[key] for key in axes if key in effective},
                "canonicalYamlBytes": len(result.stdout.encode()),
                "retainedDisposition": "needs_source_intent_review",
                "evidenceClass": "strict_loader_canonicalization_only"})
    return row


def summarize(rows: list[dict], maintained_ids: set[str], axes: dict[str, dict]) -> dict:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) & maintained_ids:
        raise RuntimeError("unmaintained YAML survey duplicated or absorbed a maintained source")
    effective = {key: Counter() for key in axes}
    rejected_explicit = {key: Counter() for key in axes}
    statuses = Counter()
    for row in rows:
        statuses[row["strictLoader"]] += 1
        if row["strictLoader"] == "canonicalized":
            for key, value in row["effectiveAxisValues"].items():
                effective[key][value] += 1
        else:
            for key, value in row["explicitAxisValues"].items():
                rejected_explicit[key][value] += 1
    return {"strictLoaderCounts": dict(sorted(statuses.items())),
            "axisRows": [
                {"id": "axis:" + key,
                 "canonicalizedAdditionalValueCounts": dict(sorted(effective[key].items())),
                 "rejectedSourceExplicitValueCounts": dict(sorted(rejected_explicit[key].items())),
                 "stillUnrepresentedAmongAllCanonicalizedSources": sorted(
                     set(axes[key]["allowed"]) - set(effective[key]) -
                     set(axes[key]["effectiveCounts"])),
                 "newlyRepresentedNonAliasChoices": sorted(
                     set(effective[key]) - set(axes[key]["effectiveCounts"]) -
                     set(axes[key]["reviewedAliasDispositions"])),
                 "remainingNonAliasChoices": sorted(
                     set(axes[key]["allowed"]) - set(effective[key]) -
                     set(axes[key]["effectiveCounts"]) -
                     set(axes[key]["reviewedAliasDispositions"]))}
                for key in sorted(axes)]}


def validated_maintained_sources(summary_path: Path, summary: dict) -> dict[str, str]:
    records_path = (summary_path.parent / summary["records"]["path"]).resolve(strict=True)
    if not records_path.is_relative_to(summary_path.parent) or \
            identity(records_path)["sha256"] != summary["records"]["sha256"]:
        raise RuntimeError("maintained canonical source records changed")
    ids = {}
    digest = hashlib.sha256()
    with gzip.open(records_path, "rb") as stream:
        for line in stream:
            digest.update(line)
            record = json.loads(line)
            path = ROOT / record["source"]["path"]
            identifier = "source:" + record["source"]["path"]
            if identifier in ids or identity(path) != record["source"]:
                raise RuntimeError("maintained canonical source is duplicated or stale")
            ids[identifier] = record["effectiveConfigId"]
    if len(ids) != summary["sourceCount"] or \
            digest.hexdigest() != summary["records"]["uncompressedSha256"]:
        raise RuntimeError("maintained canonical source records are incomplete")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--axis-audit", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--native", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or args.output.exists():
        parser.error("worker count must be 1–16 and output must be a new path")
    binary = args.binary.resolve(strict=True)
    native = args.native.resolve(strict=True)
    summary_path = args.canonical_summary.resolve(strict=True)
    axis_path = args.axis_audit.resolve(strict=True)
    summary = json.loads(summary_path.read_text())
    audit = json.loads(axis_path.read_text())
    if summary.get("schema") != "nimbus.retained-fixture-combination-merge.v1" or \
            summary.get("sourceCount") != 755 or summary.get("status") != "complete" or \
            summary.get("binary") != identity(binary) or \
            summary.get("nativeMaterializer") != identity(native):
        raise RuntimeError("first-party source survey requires current maintained canonical identities")
    if audit.get("schema") != "nimbus.retained-loader-categorical-axis-audit.v1" or \
            audit.get("axisCount") != 44 or \
            audit.get("canonicalSummarySha256") != identity(summary_path)["sha256"]:
        raise RuntimeError("first-party source survey requires the 44-axis current audit")
    axes = {row["path"]: row for row in audit["rows"]}
    if len(axes) != 44:
        raise RuntimeError("first-party source survey has duplicate categorical axes")
    paths = source_paths()
    qualified, public = maintained_cases()
    maintained_ids = validated_maintained_sources(summary_path, summary)
    manifest_ids = {"source:" + path.relative_to(ROOT).as_posix()
                    for path in (*qualified, *public)}
    discovered = {"source:" + path.relative_to(ROOT).as_posix() for path in paths}
    if not maintained_ids.keys() <= discovered or maintained_ids.keys() != manifest_ids:
        raise RuntimeError("first-party source discovery omitted maintained YAMLs")
    pending = [path for path in paths
               if "source:" + path.relative_to(ROOT).as_posix() not in maintained_ids]
    inputs = {"binary": identity(binary), "native": identity(native),
              "maintainedSummary": identity(summary_path), "axisAudit": identity(axis_path),
              "surveyTool": identity(Path(__file__))}
    source_identities = [identity(path) for path in paths]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda path: classify(path, native, axes), pending))
    if [identity(path) for path in paths] != source_identities or \
            identity(binary) != inputs["binary"] or identity(native) != inputs["native"] or \
            identity(summary_path) != inputs["maintainedSummary"] or \
            identity(axis_path) != inputs["axisAudit"] or \
            identity(Path(__file__)) != inputs["surveyTool"]:
        raise RuntimeError("first-party YAML source or survey input changed during classification")
    result = summarize(rows, set(maintained_ids), axes)
    canonical_ids = {row["effectiveConfigId"] for row in rows
                     if row["strictLoader"] == "canonicalized"}
    maintained_config_ids = set(maintained_ids.values())
    report = {"schema": "nimbus.first-party-config-source-axis-survey.v1", "status": "complete",
              "claimBoundary": "All configs/ and tests/fixtures/ YAML source files are classified by strict-loader canonicalization; rejected inputs may be negative tests, and canonicalized extras are candidates, not automatically retained or Ready combinations.",
              "inputs": inputs, "discoveredSources": len(paths), "maintainedSources": len(maintained_ids),
              "additionalSources": len(rows), "sourceIdentitiesSha256": hashlib.sha256(
                  json.dumps(source_identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "additionalUniqueEffectiveConfigs": len(canonical_ids),
              "effectiveConfigsNewToMaintainedSet": len(canonical_ids - maintained_config_ids),
              "effectiveConfigsAlreadyInMaintainedSet": len(canonical_ids & maintained_config_ids),
              **result, "rows": rows, "I1Complete": False}
    write_json_atomic(args.output, report)
    print(json.dumps({"status": "complete", "discovered": len(paths),
                      "additional": len(rows), **result["strictLoaderCounts"]}))


if __name__ == "__main__":
    main()

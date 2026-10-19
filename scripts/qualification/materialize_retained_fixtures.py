#!/usr/bin/env python3
"""Bind maintained YAMLs to strict-loader effective configurations in deterministic shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.backend_completion_inventory import explicit_yaml_scalars_text
from scripts.qualification.catalog_inventory import INVENTORY, maintained_cases
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


NUMERIC = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
PRODUCT_SET = re.compile(r"\[(?:[a-z0-9_]+(?:,\s*[a-z0-9_]+)*)?\]\Z")
GATES = ("science_reference", "conservation", "invalid_input", "failure_atomicity",
         "restart", "threading", "backend_parity", "output", "resources")
BRANCH_KEYS = (
    "grid.coordinate_system", "grid.staggering", "extensions.execution.mode",
    "extensions.compute.backend", "extensions.compute.allow_fallback",
    "domain_motion.mode", "domain_motion.nesting.enabled",
    "domain_motion.nesting.feedback", "domain_motion.tracking.enabled",
    "terrain.scheme", "advection.scheme", "time.integrator",
    "diffusion.scheme", "microphysics.scheme", "radiation.scheme",
    "boundary_layer.scheme", "turbulence.scheme", "extensions.chaos.scheme",
    "surface.scheme", "output.format", "output.preset",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha(path)}


def ordered_product_ids(expression: str | None) -> list[str]:
    if expression is None or PRODUCT_SET.fullmatch(expression) is None:
        raise RuntimeError("strict-loader product set is missing or malformed")
    products = [part.strip() for part in expression[1:-1].split(",") if part.strip()]
    if len(products) != len(set(products)):
        raise RuntimeError("strict-loader product set contains a duplicate product")
    return products


def scalar_identity(catalog: dict, raw: dict[str, str], effective: dict[str, str]) -> tuple[dict, list[str]]:
    selections = {}
    unresolved = []
    for module in catalog["modules"]:
        path = module["selectionPath"]
        policy = module["studioSelection"]
        selected = effective.get(path)
        original = raw.get(path)
        if policy["mode"] == "fixed_v1":
            selections[path] = {"canonicalId": policy["fixedSchemeId"], "source": "fixed_catalog_policy"}
            continue
        if policy["mode"] == "product_set_v1":
            products = ordered_product_ids(selected)
            owners = []
            for product in products:
                matches = [scheme["id"] for scheme in module["schemes"]
                           if product == scheme["id"] or product in scheme["aliases"]]
                if len(matches) != 1:
                    raise RuntimeError(f"product {product!r} has no unique catalog descriptor owner")
                owners.append(matches[0])
            selections[path] = {"effectiveExpression": selected, "explicitExpression": original,
                                "productIds": products, "descriptorOwnerIds": owners,
                                "source": "ordered_product_set_native_field_ids",
                                "owner": "src/init/config/v1_loader.cpp"}
            continue
        if selected is None:
            selections[path] = {"effectiveExpression": None, "explicitExpression": original,
                                "source": "conditional_or_unresolved"}
            unresolved.append(path)
            continue
        normalized = selected.strip("\"'")
        if path == "advection.scheme" and normalized == "auto":
            staggering = effective.get("grid.staggering")
            if staggering in {"c_grid", "collocated"}:
                selections[path] = {"canonicalId": "weno5" if staggering == "c_grid" else "tvd",
                                    "effectiveExpression": selected, "explicitExpression": original,
                                    "source": "conditional_auto_by_grid_staggering",
                                    "owner": "src/init/config/config_v1_physics.cpp"}
                continue
        matches = [scheme for scheme in module["schemes"]
                   if normalized == scheme["id"] or normalized in scheme["aliases"]]
        if len(matches) != 1:
            selections[path] = {"effectiveExpression": selected, "explicitExpression": original,
                                "source": "runtime_resolution_or_unmapped_alias_required"}
            unresolved.append(path)
            continue
        canonical = matches[0]["id"]
        selections[path] = {"canonicalId": canonical, "effectiveExpression": selected,
                            "explicitExpression": original,
                            "source": "explicit" if original is not None else "loader_default",
                            "alias": normalized if normalized != canonical else None}
    return selections, unresolved


def conditional_activity(effective: dict[str, str]) -> dict[str, str]:
    inactive = {}
    rules = (
        ("domain_motion.lifetime_planning.",
         effective.get("domain_motion.lifetime_planning.enabled") != "true",
         "lifetime planning is disabled"),
        ("domain_motion.nesting.", effective.get("domain_motion.nesting.enabled") != "true",
         "nesting is disabled"),
        ("domain_motion.tracking.", effective.get("domain_motion.tracking.enabled") != "true",
         "tracking is disabled"),
        ("terrain.", effective.get("terrain.scheme") == "none", "terrain scheme is none"),
        ("radiation.rrtmg_", effective.get("radiation.scheme") != "rrtmg", "RRTMG is inactive"),
        ("microphysics.milbrandt.", effective.get("microphysics.scheme") != "milbrandt",
         "Milbrandt is inactive"),
        ("output.zfp_", effective.get("output.format") not in
         {"zfp", "zfp_low", "zfp_medium", "zfp_high"}, "ZFP output is inactive"),
        ("output.netcdf_", effective.get("output.format") != "netcdf", "netCDF output is inactive"),
    )
    activating = {"domain_motion.lifetime_planning.enabled", "domain_motion.nesting.enabled",
                  "domain_motion.tracking.enabled", "terrain.scheme"}
    for key in effective:
        if key in activating:
            continue
        for prefix, disabled, reason in rules:
            if disabled and key.startswith(prefix):
                inactive[key] = reason
                break
    return inactive


def physical_backend(value: str | None) -> str:
    if value == "cpu":
        return "cpu"
    if value in {"gpu", "vulkan"}:
        return "vulkan"
    if value == "metal":
        return "metal"
    raise RuntimeError(f"strict-loader configuration has an unknown compute backend: {value!r}")


def materialize(entry: tuple[int, Path, str], native: Path, catalog: dict) -> dict:
    index, path, kind = entry
    completed = subprocess.run([str(native), "--canonical-config", str(path)], cwd=ROOT,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=30, check=False,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if completed.returncode != 0:
        raise RuntimeError(f"strict-loader canonicalization failed for {path}: {completed.stderr[-1000:]}")
    canonical = completed.stdout
    if not canonical.endswith("\n") or "\nschema: nimbus.v1\n" not in canonical:
        raise RuntimeError(f"strict-loader serialization is malformed for {path}")
    source = explicit_yaml_scalars_text(path.read_text(), str(path))
    effective = explicit_yaml_scalars_text(canonical, str(path) + ":canonical")
    selectors, unresolved = scalar_identity(catalog, source, effective)
    normalized_sha = hashlib.sha256(canonical.encode()).hexdigest()
    branches = {key: effective.get(key) for key in BRANCH_KEYS}
    numeric = {key: value for key, value in effective.items() if NUMERIC.fullmatch(value)}
    inactive = conditional_activity(effective)
    backend = physical_backend(effective.get("extensions.compute.backend"))
    nested = "nested_" + effective.get("domain_motion.nesting.feedback", "unknown") \
        if effective.get("domain_motion.nesting.enabled") == "true" else "not_nested"
    family = ".".join((effective.get("grid.staggering", "unknown"),
                       backend,
                       effective.get("terrain.scheme", "unknown"),
                       effective.get("domain_motion.mode", "unknown"),
                       nested,
                       effective.get("output.format", "unknown")))
    return {
        "schema": "nimbus.retained-fixture-combination.v1",
        "sourceIndex": index,
        "id": "fixture:" + path.relative_to(ROOT).as_posix(),
        "effectiveConfigId": "config:" + normalized_sha,
        "familyId": family,
        "sourceKind": kind,
        "source": identity(path),
        "canonicalYamlSha256": normalized_sha,
        "canonicalYamlBytes": len(canonical.encode()),
        "explicitScalars": source,
        "effectiveScalars": effective,
        "branchValues": branches,
        "physicalBackend": backend,
        "backendAlias": effective.get("extensions.compute.backend")
                        if effective.get("extensions.compute.backend") != backend else None,
        "numericValues": numeric,
        "activeNumericValues": {key: value for key, value in numeric.items() if key not in inactive},
        "inactiveConditionalValues": inactive,
        "catalogSelections": selectors,
        "conditionalOrUnresolvedSelectors": unresolved,
        "evidenceRequirements": {gate: "unreconciled_for_this_exact_combination" for gate in GATES},
        "status": {"engineering": "unreconciled", "admission": "not_checked_by_materializer",
                   "boundedEvidence": "unreconciled", "campaign": "not_established",
                   "externalReview": "not_established"},
    }


def run(args: argparse.Namespace) -> None:
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count or \
            not 1 <= args.workers <= 16:
        raise RuntimeError("shard index/count or worker count is invalid")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    native = args.native.resolve(strict=True)
    inputs = {key: identity(path) for key, path in
              (("binary", binary), ("native", native), ("maintainedManifest", INVENTORY),
               ("materializer", Path(__file__)),
               ("declarationParser", ROOT / "scripts/qualification/backend_completion_inventory.py"),
               ("maintainedInventoryTool", ROOT / "scripts/qualification/catalog_inventory.py"),
               ("atomicWriter", ROOT / "scripts/tests/catalog/qualify_catalog_bounded_runs.py"),
               ("autoAdvectionResolver", ROOT / "src/init/config/config_v1_physics.cpp"),
               ("backendResolver", ROOT / "src/init/config/config_v1_output.cpp"),
               ("loaderSource", ROOT / "src/init/config/v1_loader.cpp"),
               ("serializerSource", ROOT / "src/init/config/config_v1_serialize.cpp"))}
    catalog = json.loads(subprocess.check_output([str(binary), "--scheme-catalog=json"], cwd=ROOT))
    catalog_sha = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    qualified, public = maintained_cases()
    entries = sorted([(path, "qualified_fixture") for path in qualified] +
                     [(path, "public_preset") for path in public],
                     key=lambda pair: pair[0].relative_to(ROOT).as_posix())
    selected = [(index, path, kind) for index, (path, kind) in enumerate(entries)
                if index % args.shard_count == args.shard_index]
    digest = hashlib.sha256()
    records = output / "records.jsonl.gz"
    with ThreadPoolExecutor(max_workers=args.workers) as pool, records.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for number, record in enumerate(pool.map(lambda item: materialize(item, native, catalog), selected), 1):
                line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                payload = line.encode()
                compressed.write(payload)
                digest.update(payload)
                if number % 100 == 0:
                    print(f"materialized shard {args.shard_index}: {number}/{len(selected)}", flush=True)
    for key, original in inputs.items():
        if identity(ROOT / original["path"]) != original:
            raise RuntimeError(f"{key} changed while materializing maintained configurations")
    report = {
        "schema": "nimbus.retained-fixture-combination-shard.v1",
        "status": "complete",
        "claimBoundary": "Exact normalized maintained YAML identities, not the full retained product cross-product, bounded execution, Ready, or qualification.",
        "inputs": inputs,
        "catalogSha256": catalog_sha,
        "shardIndex": args.shard_index,
        "shardCount": args.shard_count,
        "sourceCount": len(entries),
        "selectedCount": len(selected),
        "selectedSourceIndices": [index for index, _, _ in selected],
        "records": {"path": records.name, "bytes": records.stat().st_size,
                    "sha256": sha(records), "uncompressedSha256": digest.hexdigest()},
    }
    write_json_atomic(output / "summary.json", report)
    print(json.dumps({"status": "complete", "shardIndex": args.shard_index,
                      "selectedCount": len(selected)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--native", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())

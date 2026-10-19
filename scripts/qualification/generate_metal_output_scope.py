#!/usr/bin/env python3
"""Generate the versioned exact NPY/netCDF extension of the Metal scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/fixtures/reference/catalog/generated_metal_scope_v1.json"
MANIFEST = ROOT / "tests/fixtures/reference/catalog/generated_metal_output_scope_v1.json"
HEADER = ROOT / "include/core/runtime/generated_metal_output_scope_v1.hpp"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_from_base() -> dict:
    base = json.loads(BASE.read_text())
    if base.get("schema") != "nimbus.generated-metal-scope.v1" or \
            base.get("tupleCount") != 2016 or base.get("distinctConfigCount") != 1512:
        raise RuntimeError("base generated Metal scope identity drifted")
    return {"schema": "nimbus.generated-metal-output-scope.v1", "version": 1,
            "claimBoundary": "Exact canonical dry flat Metal NPY/netCDF output variants only; "
            "other retained output, numeric, boundary, motion, nesting and terrain combinations stay open.",
            "baseManifestSha256": sha(BASE), "sourceTuples": 2016,
            "outputFormats": ["npy_3d", "netcdf"], "totalTuples": 4032,
            "distinctConfigCount": 3024,
            "netcdfDefaults": {"deflateLevel": 2, "shuffle": True, "targetChunkKib": 1024},
            "normalization": "replace netcdf with npy_3d after validating netcdf-only defaults, then compare canonical YAML SHA-256 with the base scope"}


def header_for(manifest: dict) -> bytes:
    if manifest.get("schema") != "nimbus.generated-metal-output-scope.v1" or \
            manifest.get("version") != 1 or manifest.get("sourceTuples") != 2016 or \
            manifest.get("outputFormats") != ["npy_3d", "netcdf"] or \
            manifest.get("totalTuples") != 4032 or manifest.get("distinctConfigCount") != 3024 or \
            manifest.get("netcdfDefaults") != {"deflateLevel": 2, "shuffle": True, "targetChunkKib": 1024}:
        raise RuntimeError("generated Metal output scope manifest is malformed")
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return (
        "#pragma once\n\n#include <string_view>\n\n"
        "namespace tmv::runtime\n{\n"
        f'inline constexpr std::string_view kGeneratedMetalOutputScopeManifestSha256 = "{digest}";\n'
        "inline constexpr int kGeneratedMetalOutputScopeTupleCount = 4032;\n"
        "inline constexpr int kGeneratedMetalOutputScopeDistinctConfigCount = 3024;\n"
        "} // namespace tmv::runtime\n"
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(MANIFEST.read_text())
        expected = (json.dumps(manifest, indent=2) + "\n").encode()
        if MANIFEST.read_bytes() != expected or HEADER.read_bytes() != header_for(manifest):
            raise RuntimeError("generated Metal output scope artifacts drifted")
        if BASE.is_file() and manifest_from_base() != manifest:
            raise RuntimeError("generated Metal output scope differs from its base manifest")
    else:
        if MANIFEST.exists() or HEADER.exists():
            raise RuntimeError("refusing to overwrite generated Metal output scope artifacts")
        manifest = manifest_from_base()
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        HEADER.write_bytes(header_for(manifest))
    print("generated Metal output scope: 4032 tuples, 3024 exact configurations")


if __name__ == "__main__":
    main()

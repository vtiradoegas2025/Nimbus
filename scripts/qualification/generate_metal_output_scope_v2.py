#!/usr/bin/env python3
"""Emit or verify the combined exact Metal NPY/netCDF/ZFP output scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/fixtures/reference/catalog/generated_metal_scope_v1.json"
NPY_NETCDF = ROOT / "tests/fixtures/reference/catalog/generated_metal_output_scope_v1.json"
ZFP_TYPED = ROOT / "tests/fixtures/reference/catalog/generated_metal_zfp_typed_output_scope_v1.json"
MANIFEST = ROOT / "tests/fixtures/reference/catalog/generated_metal_output_scope_v2.json"
HEADER = ROOT / "include/core/runtime/generated_metal_output_scope_v2.hpp"
CADENCES = {"npy_3d": [1, 301], "netcdf": [1, 301], "zfp_high": [1]}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_from_sources() -> dict:
    base = json.loads(BASE.read_text())
    npy_netcdf = json.loads(NPY_NETCDF.read_text())
    zfp = json.loads(ZFP_TYPED.read_text())
    if base.get("schema") != "nimbus.generated-metal-scope.v1" or \
            base.get("tupleCount") != 2016 or base.get("distinctConfigCount") != 1512 or \
            npy_netcdf.get("schema") != "nimbus.generated-metal-output-scope.v1" or \
            npy_netcdf.get("baseManifestSha256") != sha(BASE) or \
            npy_netcdf.get("totalTuples") != 4032 or \
            npy_netcdf.get("distinctConfigCount") != 3024 or \
            npy_netcdf.get("outputFormats") != ["npy_3d", "netcdf"] or \
            zfp.get("schema") != "nimbus.generated-metal-zfp-typed-output-scope.v1" or \
            zfp.get("baseScopeManifestSha256") != sha(BASE) or \
            zfp.get("typedSerializedOutputBlockBytes") != 964:
        raise RuntimeError("combined Metal output-scope source identity drifted")
    return {
        "schema": "nimbus.generated-metal-output-scope.v2",
        "version": 2,
        "claimBoundary": "Exact dry flat Metal NPY/netCDF scopes and typed minimal ZFP candidate only; other retained output, numeric, boundary, motion, nesting and terrain combinations remain open.",
        "baseManifestSha256": sha(BASE),
        "npyNetcdfManifestSha256": sha(NPY_NETCDF),
        "zfpTypedOutputManifestSha256": sha(ZFP_TYPED),
        "sourceTuples": 2016,
        "outputFormats": ["npy_3d", "netcdf", "zfp_high"],
        "totalTuples": 6048,
        "distinctConfigCount": 4536,
        "writeCadenceSecondsByFormat": CADENCES,
        "normalization": "NPY/netCDF preserve the v1 output manifest; exact typed ZFP output block normalizes to canonical minimal NPY before the base 1512-hash membership check.",
    }


def header_for(value: dict) -> str:
    if value.get("schema") != "nimbus.generated-metal-output-scope.v2" or \
            value.get("version") != 2 or value.get("sourceTuples") != 2016 or \
            value.get("outputFormats") != ["npy_3d", "netcdf", "zfp_high"] or \
            value.get("totalTuples") != 6048 or value.get("distinctConfigCount") != 4536 or \
            value.get("writeCadenceSecondsByFormat") != CADENCES:
        raise RuntimeError("combined Metal output-scope manifest is malformed")
    encoded = (json.dumps(value, indent=2) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return (
        "#pragma once\n\n#include <algorithm>\n#include <array>\n#include <string_view>\n\n"
        "namespace tmv::runtime\n{\n"
        f'inline constexpr std::string_view kGeneratedMetalOutputScopeV2ManifestSha256 = "{digest}";\n'
        "inline constexpr int kGeneratedMetalOutputScopeV2TupleCount = 6048;\n"
        "inline constexpr int kGeneratedMetalOutputScopeV2DistinctConfigCount = 4536;\n"
        "inline constexpr std::array<int, 2> kGeneratedMetalNpyWriteCadenceSeconds{1, 301};\n"
        "inline constexpr std::array<int, 2> kGeneratedMetalNetcdfWriteCadenceSeconds{1, 301};\n"
        "inline constexpr std::array<int, 1> kGeneratedMetalZfpHighWriteCadenceSeconds{1};\n\n"
        "inline bool admits_generated_metal_output_write_cadence(std::string_view format, int seconds) noexcept\n"
        "{\n"
        "    if (format == \"npy_3d\")\n"
        "    {\n"
        "        return std::binary_search(kGeneratedMetalNpyWriteCadenceSeconds.begin(),\n"
        "                                  kGeneratedMetalNpyWriteCadenceSeconds.end(), seconds);\n"
        "    }\n"
        "    if (format == \"netcdf\")\n"
        "    {\n"
        "        return std::binary_search(kGeneratedMetalNetcdfWriteCadenceSeconds.begin(),\n"
        "                                  kGeneratedMetalNetcdfWriteCadenceSeconds.end(), seconds);\n"
        "    }\n"
        "    if (format == \"zfp_high\")\n"
        "    {\n"
        "        return std::binary_search(kGeneratedMetalZfpHighWriteCadenceSeconds.begin(),\n"
        "                                  kGeneratedMetalZfpHighWriteCadenceSeconds.end(), seconds);\n"
        "    }\n"
        "    return false;\n"
        "}\n"
        "} // namespace tmv::runtime\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--emit-header", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    value = manifest_from_sources()
    encoded = json.dumps(value, indent=2) + "\n"
    if args.verify:
        if MANIFEST.read_text() != encoded or HEADER.read_text() != header_for(value):
            raise RuntimeError("versioned combined Metal output-scope artifacts drifted")
        print("generated Metal output scope v2: verified 6048 tuples, 4536 configurations")
    elif args.emit_header:
        print(header_for(value), end="")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

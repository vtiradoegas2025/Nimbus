#!/usr/bin/env python3
"""Emit or verify the one-digest east-NPY extension and combined scope v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/reference/catalog/flat_east_radiative_metal_npy_candidate_v1.yaml"
HELPER = ROOT / "bin/nimbus_native_admission_batch"
PRIOR = ROOT / "tests/fixtures/reference/catalog/generated_metal_output_scope_v2.json"
NO_ZFP_PRIOR = ROOT / "tests/fixtures/reference/catalog/generated_metal_output_scope_v1.json"
EXTENSION = ROOT / "tests/fixtures/reference/catalog/generated_metal_east_npy_scope_v1.json"
EXTENSION_HEADER = ROOT / "include/core/runtime/generated_metal_east_npy_scope_v1.hpp"
COMBINED = ROOT / "tests/fixtures/reference/catalog/generated_metal_output_scope_v3.json"
COMBINED_HEADER = ROOT / "include/core/runtime/generated_metal_output_scope_v3.hpp"
NO_ZFP_COMBINED = ROOT / "tests/fixtures/reference/catalog/generated_metal_output_scope_nozfp_v2.json"
NO_ZFP_HEADER = ROOT / "include/core/runtime/generated_metal_output_scope_nozfp_v2.hpp"
FAMILY_ID = "boundary:2b5bfbe6dfa466c6baf991e3"
DIGEST = "a244b58833f7e3cd7060d941e8453d4e5face3dff9e18e87ebf61dd581a24d55"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: dict) -> str:
    return json.dumps(value, indent=2) + "\n"


def strict_fixture(helper: Path) -> bytes:
    raw = FIXTURE.read_bytes()
    source = raw.decode()
    for anchor in ("    east: radiative\n", "  format: npy_3d\n",
                   "  preset: minimal\n", "  async_io: false\n",
                   "    backend: metal\n", "    mode: qualified\n"):
        if source.count(anchor) != 1:
            raise RuntimeError(f"east NPY fixture anchor drifted: {anchor!r}")
    completed = subprocess.run([str(helper), "--canonical-config", str(FIXTURE)],
                               cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=30, check=False)
    if completed.returncode != 0 or completed.stdout != raw or sha(raw) != DIGEST:
        raise RuntimeError("east NPY source is not the exact strict-loader serialization")
    return raw


def extension(helper: Path) -> dict:
    raw = strict_fixture(helper)
    return {
        "schema": "nimbus.generated-metal-east-npy-scope.v1",
        "version": 1,
        "claimBoundary": "One exact qualified dry flat east-radiative C-grid Metal/TVD minimal NPY configuration only; no netCDF/ZFP normalization or other boundary profile.",
        "profileCount": 1,
        "familyId": FAMILY_ID,
        "fixtureSha256": sha(raw),
        "strictCanonicalYamlSha256": DIGEST,
        "strictCanonicalYamlBytes": len(raw),
        "outputFormats": ["npy_3d"],
        "preset": "minimal",
        "asyncIo": False,
        "writeCadenceSeconds": [1],
        "scalarTransportOwner": "native_conservative_mc",
        "backend": "metal",
        "executionMode": "qualified",
        "normalizedCodecsAdmitted": [],
    }


def combined(value: dict) -> dict:
    prior = json.loads(PRIOR.read_text())
    if prior.get("schema") != "nimbus.generated-metal-output-scope.v2" or \
            prior.get("totalTuples") != 6048 or prior.get("distinctConfigCount") != 4536 or \
            prior.get("outputFormats") != ["npy_3d", "netcdf", "zfp_high"] or \
            prior.get("writeCadenceSecondsByFormat") != {
                "npy_3d": [1, 301], "netcdf": [1, 301], "zfp_high": [1]}:
        raise RuntimeError("prior generated-Metal output scope drifted")
    return {
        "schema": "nimbus.generated-metal-output-scope.v3",
        "version": 3,
        "claimBoundary": "Existing canonical NPY/netCDF/ZFP scope plus exactly one separately admitted east-radiative minimal NPY configuration; other retained boundaries and codecs remain guarded.",
        "priorOutputScopeV2ManifestSha256": sha(PRIOR.read_bytes()),
        "baseManifestSha256": prior["baseManifestSha256"],
        "npyNetcdfManifestSha256": prior["npyNetcdfManifestSha256"],
        "zfpTypedOutputManifestSha256": prior["zfpTypedOutputManifestSha256"],
        "sourceTuples": prior["sourceTuples"],
        "outputFormats": prior["outputFormats"],
        "totalTuples": 6049,
        "distinctConfigCount": 4537,
        "writeCadenceSecondsByFormat": prior["writeCadenceSecondsByFormat"],
        "additionalBoundaryNpyScope": {
            "version": 1,
            "profileCount": 1,
            "familyId": FAMILY_ID,
            "strictCanonicalYamlSha256": DIGEST,
            "outputFormats": ["npy_3d"],
            "writeCadenceSeconds": [1],
            "manifestSha256": sha(encoded(value).encode()),
        },
        "normalization": "Existing v2 canonical NPY/netCDF/ZFP normalization is unchanged; the additional east-radiative NPY digest is checked before normalization and never admits netCDF or ZFP.",
    }


def combined_no_zfp(value: dict) -> dict:
    prior = json.loads(NO_ZFP_PRIOR.read_text())
    if prior.get("schema") != "nimbus.generated-metal-output-scope.v1" or \
            prior.get("totalTuples") != 4032 or prior.get("distinctConfigCount") != 3024 or \
            prior.get("outputFormats") != ["npy_3d", "netcdf"]:
        raise RuntimeError("prior non-ZFP generated-Metal output scope drifted")
    return {
        "schema": "nimbus.generated-metal-output-scope-nozfp.v2",
        "version": 2,
        "claimBoundary": "Existing canonical NPY/netCDF scope plus exactly one separately admitted east-radiative minimal NPY configuration; this binary has no ZFP codec.",
        "priorOutputScopeV1ManifestSha256": sha(NO_ZFP_PRIOR.read_bytes()),
        "baseManifestSha256": prior["baseManifestSha256"],
        "sourceTuples": prior["sourceTuples"],
        "outputFormats": prior["outputFormats"],
        "totalTuples": 4033,
        "distinctConfigCount": 3025,
        "writeCadenceSecondsByFormat": {"npy_3d": [1, 301], "netcdf": [1, 301]},
        "additionalBoundaryNpyScope": {
            "version": 1,
            "profileCount": 1,
            "familyId": FAMILY_ID,
            "strictCanonicalYamlSha256": DIGEST,
            "outputFormats": ["npy_3d"],
            "writeCadenceSeconds": [1],
            "manifestSha256": sha(encoded(value).encode()),
        },
        "normalization": "Existing v1 canonical NPY/netCDF normalization is unchanged; east-radiative NPY is a raw-format digest with no codec normalization.",
    }


def extension_header(value: dict) -> str:
    return (
        "#pragma once\n\n#include <string_view>\n\n"
        "namespace tmv::runtime\n{\n"
        f'inline constexpr std::string_view kGeneratedMetalEastNpyScopeManifestSha256 = "{sha(encoded(value).encode())}";\n'
        f'inline constexpr std::string_view kGeneratedMetalEastNpyCanonicalYamlSha256 = "{DIGEST}";\n'
        f'inline constexpr std::string_view kGeneratedMetalEastNpyFamilyId = "{FAMILY_ID}";\n'
        "} // namespace tmv::runtime\n"
    )


def combined_header(value: dict) -> str:
    return (
        "#pragma once\n\n#include <string_view>\n\n"
        "namespace tmv::runtime\n{\n"
        f'inline constexpr std::string_view kGeneratedMetalOutputScopeV3ManifestSha256 = "{sha(encoded(value).encode())}";\n'
        "inline constexpr int kGeneratedMetalOutputScopeV3TupleCount = 6049;\n"
        "inline constexpr int kGeneratedMetalOutputScopeV3DistinctConfigCount = 4537;\n"
        "inline constexpr int kGeneratedMetalEastNpyWriteCadenceSeconds = 1;\n"
        "} // namespace tmv::runtime\n"
    )


def no_zfp_header(value: dict) -> str:
    return (
        "#pragma once\n\n#include <string_view>\n\n"
        "namespace tmv::runtime\n{\n"
        f'inline constexpr std::string_view kGeneratedMetalOutputScopeNoZfpV2ManifestSha256 = "{sha(encoded(value).encode())}";\n'
        "inline constexpr int kGeneratedMetalOutputScopeNoZfpV2TupleCount = 4033;\n"
        "inline constexpr int kGeneratedMetalOutputScopeNoZfpV2DistinctConfigCount = 3025;\n"
        "} // namespace tmv::runtime\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, default=HELPER)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("emit-extension", "emit-extension-header", "emit-combined",
                 "emit-combined-header", "emit-nozfp", "emit-nozfp-header", "verify"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    east = extension(args.helper.resolve(strict=True))
    scope = combined(east)
    no_zfp_scope = combined_no_zfp(east)
    if args.verify:
        if EXTENSION.read_text() != encoded(east) or \
                EXTENSION_HEADER.read_text() != extension_header(east) or \
                COMBINED.read_text() != encoded(scope) or \
                COMBINED_HEADER.read_text() != combined_header(scope) or \
                NO_ZFP_COMBINED.read_text() != encoded(no_zfp_scope) or \
                NO_ZFP_HEADER.read_text() != no_zfp_header(no_zfp_scope):
            raise RuntimeError("versioned east NPY scope artifacts drifted")
        print("generated Metal east NPY scope: verified 6049 tuples, 4537 configurations")
    elif args.emit_extension:
        print(encoded(east), end="")
    elif args.emit_extension_header:
        print(extension_header(east), end="")
    elif args.emit_combined:
        print(encoded(scope), end="")
    elif args.emit_nozfp:
        print(encoded(no_zfp_scope), end="")
    elif args.emit_nozfp_header:
        print(no_zfp_header(no_zfp_scope), end="")
    else:
        print(combined_header(scope), end="")


if __name__ == "__main__":
    main()

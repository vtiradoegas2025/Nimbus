#!/usr/bin/env python3
"""Emit or verify the strict-loader-serialized Metal/ZFP output scope seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TEST_BINARY = ROOT / "bin/test_init_config_v1"
TEST_SOURCE = ROOT / "tests/init/config/test_v1_loader.cpp"
SERIALIZER = ROOT / "src/init/config/config_v1_serialize.cpp"
BASE_SCOPE = ROOT / "tests/fixtures/reference/catalog/generated_metal_scope_v1.json"
CANDIDATE_SCOPE = ROOT / "tests/fixtures/reference/catalog/generated_metal_zfp_candidate_scope_v1.json"
CANDIDATE = ROOT / "tests/fixtures/reference/catalog/generated_metal_zfp_candidate_v1.yaml"
MANIFEST = ROOT / "tests/fixtures/reference/catalog/generated_metal_zfp_typed_output_scope_v1.json"
HEADER = ROOT / "include/core/runtime/generated_metal_zfp_typed_scope_v1.hpp"
FILTER = "[generated-metal-zfp-typed-scope]"
TOKEN = re.compile(r"^NIMBUS_TYPED_ZFP_OUTPUT_HEX ([0-9a-f]+)$", re.MULTILINE)

TYPED_KEYS = (
    "preset", "format", "async_io", "live_shm", "live_shm_fields",
    "live_shm_interval_s", "live_shm_full_rate", "trajectories",
    "trajectory_seed_time_s", "trajectory_count", "trajectory_sample_interval_s",
    "trajectory_max_samples_per_parcel", "zfp_tolerance", "zfp_mode",
    "zfp_precision_bits", "zfp_rate_bps", "zfp_keyframe_interval",
    "zfp_sparse_threshold", "zfp_float16_prefilter", "zfp_predictive_delta",
    "zfp_field_execution", "zfp_field_workers", "zfp_field_scratch_mib",
    "zfp_per_field_tolerances", "zfp_tight_tolerance",
    "zfp_moderate_tolerance", "zfp_loose_tolerance", "zfp_field_tolerance",
    "keep_last_frames", "maximum_retained_history_files", "minimum_free_space_mib",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_block() -> bytes:
    source = CANDIDATE.read_text()
    if source.count("output:\n") != 1 or source.count("\n# Nimbus extensions") != 1:
        raise RuntimeError("candidate has no unique raw output block")
    return source[source.index("output:\n"):source.index("\n# Nimbus extensions")].encode()


def typed_block(binary: Path) -> bytes:
    result = subprocess.run([str(binary), FILTER], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=60, check=False,
                            env={**os.environ, "NIMBUS_EMIT_TYPED_ZFP_SCOPE": "1"})
    matches = TOKEN.findall(result.stdout)
    if result.returncode != 0 or len(matches) != 1 or \
            "All tests passed" not in result.stdout:
        raise RuntimeError("strict-loader ZFP typed-scope test failed or emitted no unique block:\n" +
                           result.stdout[-2500:])
    block = bytes.fromhex(matches[0])
    keys = re.findall(r"^  ([a-z0-9_]+):", block.decode(), re.MULTILINE)
    if keys != list(TYPED_KEYS):
        raise RuntimeError(f"typed ZFP output key sequence drifted: {keys}")
    if set(re.findall(r"^    ([a-z_]+):", block.decode(), re.MULTILINE)) != \
            {"p", "qr", "qv", "rho", "theta", "u", "v", "w"}:
        raise RuntimeError("typed ZFP minimal field tolerance inventory drifted")
    return block


def manifest(binary: Path) -> dict:
    candidate = json.loads(CANDIDATE_SCOPE.read_text())
    if candidate.get("schema") != "nimbus.generated-metal-zfp-candidate-scope.v1" or \
            candidate.get("sourceTupleCount") != 2016 or \
            candidate.get("baseScopeManifestSha256") != sha(BASE_SCOPE):
        raise RuntimeError("ZFP candidate/base scope identity drifted")
    raw = raw_block()
    typed = typed_block(binary)
    if hashlib.sha256(raw).hexdigest() != candidate["zfpOutputBlockSha256"] or \
            raw == typed:
        raise RuntimeError("raw/typed ZFP output scope relation drifted")
    return {
        "schema": "nimbus.generated-metal-zfp-typed-output-scope.v1",
        "version": 1,
        "claimBoundary": "Strict ConfigV1-serialized ZFP output block for the guarded minimal dry Metal candidate; CLI overrides and qualified public admission are separate gates.",
        "baseScopeManifestSha256": sha(BASE_SCOPE),
        "candidateScopeManifestSha256": sha(CANDIDATE_SCOPE),
        "candidateFixtureSha256": sha(CANDIDATE),
        "serializerSourceSha256": sha(SERIALIZER),
        "loaderTestSourceSha256": sha(TEST_SOURCE),
        "rawCandidateOutputBlockSha256": hashlib.sha256(raw).hexdigest(),
        "typedSerializedOutputBlockSha256": hashlib.sha256(typed).hexdigest(),
        "typedSerializedOutputBlockBytes": len(typed),
        "typedSerializedTopLevelKeys": list(TYPED_KEYS),
        "typedFieldToleranceIds": ["p", "qr", "qv", "rho", "theta", "u", "v", "w"],
        "inactiveInZfpFormat": [
            "netcdf_deflate_level", "netcdf_shuffle", "netcdf_target_chunk_kib",
            "qualification_diagnostics_interval_s when qualification_diagnostics_preset=none",
            "write_cadence_medium_s and write_cadence_slow_s when tiered_write_cadence=false",
        ],
        "normalization": "Strict loader/serializer round-trip, then replace the entire OutputSection with the canonical NPY OutputSection and exploratory mode with qualified; exact serialized base profile identity is required.",
    }


def header_for(value: dict) -> str:
    encoded = (json.dumps(value, indent=2) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return (
        "#pragma once\n\n#include <string_view>\n\n"
        "namespace tmv::runtime\n{\n"
        f'inline constexpr std::string_view kGeneratedMetalZfpTypedScopeManifestSha256 = "{digest}";\n'
        f'inline constexpr std::string_view kGeneratedMetalZfpTypedOutputBlockSha256 = "{value["typedSerializedOutputBlockSha256"]}";\n'
        "} // namespace tmv::runtime\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-binary", type=Path, default=TEST_BINARY)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--emit-header", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    value = manifest(args.test_binary.resolve(strict=True))
    encoded = json.dumps(value, indent=2) + "\n"
    if args.verify:
        if MANIFEST.read_text() != encoded:
            raise RuntimeError("versioned typed ZFP output scope manifest drifted")
        if HEADER.read_text() != header_for(value):
            raise RuntimeError("versioned typed ZFP output scope header drifted")
        print("generated Metal ZFP typed output scope: verified", value["typedSerializedOutputBlockSha256"])
    elif args.emit_header:
        print(header_for(value), end="")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

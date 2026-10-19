#!/usr/bin/env python3
"""Generate the exact reviewed canonical Metal admission scope."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/fixtures/reference/catalog/generated_metal_scope_v1.json"
HEADER = ROOT / "include/core/runtime/generated_metal_scope_v1.hpp"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_from_projection(projection_path: Path) -> dict:
    projection = json.loads(projection_path.read_text())
    records = Path(projection["records"]["path"])
    if projection["status"] != "complete" or projection["projected"] != 2016 or \
            sha(records) != projection["records"]["sha256"]:
        raise RuntimeError("the completed generated Metal projection is required")
    with gzip.open(records, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 2016 or len({row["id"] for row in rows}) != 2016:
        raise RuntimeError("generated Metal scope tuples are missing or duplicated")
    hashes = sorted({row["metalYamlSha256"] for row in rows})
    if len(hashes) != 1512 or any(len(value) != 64 for value in hashes):
        raise RuntimeError("generated Metal canonical configuration identity drifted")
    return {"schema": "nimbus.generated-metal-scope.v1", "version": 1,
            "claimBoundary": "Exact canonical generated Metal configurations only. Other retained numeric, "
            "motion, nesting, boundary, output and terrain combinations remain tracked.",
            "projectionSha256": sha(projection_path), "projectionRecordsSha256": sha(records),
            "sourceBinarySha256": projection["inputs"]["binary"]["sha256"],
            "tupleCount": len(rows), "distinctConfigCount": len(hashes),
            "canonicalYamlSha256": hashes}


def header_from_manifest(manifest: dict) -> bytes:
    hashes = manifest.get("canonicalYamlSha256")
    if manifest.get("schema") != "nimbus.generated-metal-scope.v1" or manifest.get("version") != 1 or \
            manifest.get("tupleCount") != 2016 or manifest.get("distinctConfigCount") != 1512 or \
            not isinstance(hashes, list) or len(hashes) != 1512 or \
            any(not isinstance(value, str) or len(value) != 64 for value in hashes) or \
            hashes != sorted(set(hashes)):
        raise RuntimeError("versioned canonical Metal scope manifest is malformed")
    entries = ",\n".join(f'    "{value}"' for value in hashes)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    header = (
        "#pragma once\n\n"
        "#include <array>\n#include <string_view>\n\n"
        "namespace tmv::runtime\n{\n"
        f'inline constexpr std::string_view kGeneratedMetalScopeManifestSha256 = "{manifest_sha256}";\n'
        f"inline constexpr std::array<std::string_view, {len(hashes)}> kGeneratedMetalCanonicalYamlSha256{{{{\n"
        f"{entries}\n"
        "}};\n} // namespace tmv::runtime\n"
    )
    return header.encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--projection", type=Path,
                        help="explicit external projection summary for generation or evidence comparison")
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(MANIFEST.read_text())
        manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
        if MANIFEST.read_bytes() != manifest_bytes or HEADER.read_bytes() != header_from_manifest(manifest):
            raise RuntimeError("generated Metal scope header or manifest drifted")
        if args.projection is not None and manifest_from_projection(args.projection) != manifest:
            raise RuntimeError("generated Metal scope differs from its retained projection")
    else:
        if args.projection is None:
            parser.error("generation requires --projection; verification uses the maintained manifest")
        manifest = manifest_from_projection(args.projection)
        if MANIFEST.exists() or HEADER.exists():
            raise RuntimeError("refusing to overwrite an existing Metal scope artifact")
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        HEADER.write_bytes(header_from_manifest(manifest))
    print(f"generated Metal scope: 2016 tuples, 1512 distinct exact configurations")


if __name__ == "__main__":
    main()

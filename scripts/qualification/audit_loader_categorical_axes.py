#!/usr/bin/env python3
"""Audit literal strict-loader categorical domains against maintained effective YAMLs."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def call_arguments(source: str, opening: int) -> list[str]:
    depth = 0
    quote = False
    escape = False
    begin = opening + 1
    args = []
    for index in range(opening, len(source)):
        char = source[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                quote = False
            continue
        if char == '"':
            quote = True
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                args.append(source[begin:index].strip())
                return args
        elif char == ',' and depth == 1:
            args.append(source[begin:index].strip())
            begin = index + 1
    raise RuntimeError("strict-loader categorical call has unmatched parentheses")


def literal_axes(source: str, minimum_axes: int = 30, include_special: bool = True) -> dict[str, dict]:
    variables = dict(re.findall(r'const\s+char\s*\*\s*(\w+)\s*=\s*"([^"\n]+)"', source))
    axes = {}
    for match in re.finditer(r"\btake_enum\s*\(", source):
        args = call_arguments(source, source.index("(", match.start()))
        if len(args) < 4 or re.fullmatch(r'"[A-Za-z_][A-Za-z_0-9.]*"', args[0]) is None:
            continue
        key = args[0][1:-1]
        expression = re.sub(r"//[^\n]*", "", args[3]).strip()
        allowed = variables.get(expression)
        if allowed is None:
            literals = re.findall(r'"([^"\n]*)"', expression)
            if not literals:
                raise RuntimeError(f"strict-loader categorical domain is not literal: {key}")
            allowed = "".join(literals)
        values = [re.sub(r"\s*\([^)]*\)$", "", token.strip())
                  for token in allowed.split(",")]
        if not values or any(not re.fullmatch(r"[a-z0-9][a-z0-9_]*", value) for value in values) or \
                len(values) != len(set(values)):
            raise RuntimeError(f"strict-loader categorical domain is malformed: {key}")
        descriptor = {"path": key, "allowed": values, "allowedExpression": allowed,
                      "parser": args[2].strip()}
        if key in axes and axes[key] != descriptor:
            raise RuntimeError(f"strict-loader categorical key has conflicting domains: {key}")
        axes[key] = descriptor
    special = {
        "take_coordinate_system": ("grid.coordinate_system", "cartesian", "parse_coordinate_system"),
        "take_stagger_type": ("grid.staggering", "collocated, c_grid", "parse_stagger_type"),
    }
    if include_special:
        for call, (key, expected, parser) in special.items():
            pattern = rf'\b{call}\s*\(\s*"{re.escape(key)}"\s*,'
            if len(re.findall(pattern, source)) != 1 or expected not in source:
                raise RuntimeError(f"strict-loader special categorical domain changed: {key}")
            axes[key] = {"path": key, "allowed": expected.split(", "),
                         "allowedExpression": expected, "parser": parser}
    if len(axes) < minimum_axes:
        raise RuntimeError("strict-loader categorical parser missed declared axes")
    return dict(sorted(axes.items()))


def alias_disposition(path: str, value: str) -> dict | None:
    if path.startswith("boundary_conditions.lateral.") and value == "open":
        return {"kind": "conditional_legacy_alias",
                "targetByStaggering": {"c_grid": "radiative", "collocated": "zero_gradient"},
                "owner": "src/init/config/v1_loader.cpp"}
    if path == "extensions.compute.backend" and value in {"vulkan", "vk", "mtl"}:
        return {"kind": "parser_alias", "serializedTarget": "metal" if value == "mtl" else "gpu",
                "physicalBackend": "metal" if value == "mtl" else "vulkan",
                "owner": "src/init/config/config_v1_output.cpp"}
    if path == "grid.coordinate_system" and value == "cart":
        return {"kind": "parser_alias", "serializedTarget": "cartesian",
                "owner": "src/core/infra/coordinate_system.cpp"}
    if path == "grid.staggering" and value in {"a_grid", "cgrid"}:
        return {"kind": "parser_alias",
                "serializedTarget": "collocated" if value == "a_grid" else "c_grid",
                "owner": "src/core/infra/coordinate_system.cpp"}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.canonical_summary.resolve(strict=True)
    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != "nimbus.retained-fixture-combination-merge.v1" or \
            summary.get("status") != "complete" or summary.get("sourceCount") != 755:
        raise RuntimeError("categorical axis audit requires the complete maintained canonical merge")
    records = (summary_path.parent / summary["records"]["path"]).resolve(strict=True)
    if sha(records) != summary["records"]["sha256"]:
        raise RuntimeError("maintained canonical records changed")
    loader = ROOT / "src/init/config/v1_loader.cpp"
    axes = literal_axes(loader.read_text())
    effective = {key: Counter() for key in axes}
    explicit = {key: Counter() for key in axes}
    digest = hashlib.sha256()
    count = 0
    with gzip.open(records, "rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            count += 1
            for key in axes:
                if key in row["effectiveScalars"]:
                    effective[key][row["effectiveScalars"][key]] += 1
                if key in row["explicitScalars"]:
                    explicit[key][row["explicitScalars"][key]] += 1
    if count != 755 or digest.hexdigest() != summary["records"]["uncompressedSha256"]:
        raise RuntimeError("maintained canonical axis audit omitted or changed records")
    rows = []
    for key, axis in axes.items():
        observed = set(effective[key])
        unexpected = observed - set(axis["allowed"])
        if unexpected:
            raise RuntimeError(f"strict-loader effective value is outside advertised domain: {key} {unexpected}")
        unseen = sorted(set(axis["allowed"]) - observed)
        aliases = {value: disposition for value in unseen
                   if (disposition := alias_disposition(key, value)) is not None}
        rows.append({"id": "axis:" + key, "path": key,
                     "loaderOwner": "src/init/config/v1_loader.cpp",
                     "parser": axis["parser"], "allowed": axis["allowed"],
                     "effectiveCounts": dict(sorted(effective[key].items())),
                     "explicitCounts": dict(sorted(explicit[key].items())),
                     "notInMaintainedEffectiveSet": unseen,
                     "reviewedAliasDispositions": aliases,
                     "unrepresentedEffectiveChoices": [value for value in unseen if value not in aliases],
                     "nextTask": "I1-conditional-axis-and-other-fixture-reconciliation"})
    report = {"schema": "nimbus.retained-loader-categorical-axis-audit.v1",
              "status": "complete",
              "claimBoundary": "Advertised literal strict-loader enum values versus maintained effective YAMLs; other parsers, numeric envelopes, conditional interactions and complete product combinations remain open.",
              "loaderSha256": sha(loader),
              "backendAliasParserSha256": sha(ROOT / "src/init/config/config_v1_output.cpp"),
              "coordinateParserSha256": sha(ROOT / "src/core/infra/coordinate_system.cpp"),
              "axisToolSha256": sha(Path(__file__)),
              "canonicalSummarySha256": sha(summary_path),
              "canonicalRecordSha256": summary["records"]["uncompressedSha256"],
              "axisCount": len(rows),
              "missingEffectiveAxisCount": sum(bool(row["notInMaintainedEffectiveSet"]) for row in rows),
              "unrepresentedEffectiveAxisCount": sum(bool(row["unrepresentedEffectiveChoices"]) for row in rows),
              "axisRowsSha256": hashlib.sha256(json.dumps(rows, sort_keys=True,
                                                       separators=(",", ":")).encode()).hexdigest(),
              "rows": rows,
              "I1Complete": False}
    if args.output.exists():
        parser.error("use a new output path to preserve the previous axis audit")
    write_json_atomic(args.output, report)
    print(json.dumps({"status": "complete", "axes": len(rows),
                      "axesWithMissingEffectiveValues": report["missingEffectiveAxisCount"]}))


if __name__ == "__main__":
    main()

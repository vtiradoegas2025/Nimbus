#!/usr/bin/env python3
"""Seal a completed same-backend moving/reference frontier comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_BACKENDS = ("vulkan", "metal")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def artifact(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "sha256": sha256(path)}


def resolve_record(record: object, label: str) -> Path:
    require(isinstance(record, dict), f"missing artifact record: {label}")
    path = (ROOT / str(record.get("path"))).resolve()
    require(path.is_file() and sha256(path) == record.get("sha256"),
            f"campaign artifact binding changed: {label}")
    return path


def build_evidence(compute_backend: str,
                   campaign_report_path: Path) -> dict[str, Any]:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    campaign_report_path = campaign_report_path.resolve()
    campaign = load_json(campaign_report_path)
    claims = campaign.get("claims", {})
    checks = campaign.get("checks", [])
    require(
        campaign.get("schema") ==
            f"nimbus.qualification.moving_nested_{compute_backend}_reference_frontier" and
        campaign.get("schema_version") == 1 and
        campaign.get("status") == "pass" and
        campaign.get("qualification_eligible") is True and
        isinstance(checks, list) and len(checks) == 21 and
        all(isinstance(item, dict) and item.get("passed") is True
            for item in checks),
        "campaign is not a complete 21-check qualification pass",
    )
    require(
        claims.get("same_binary_candidate_and_reference") is True and
        claims.get(
            f"same_{compute_backend}_backend_candidate_and_reference") is True and
        claims.get(
            f"fallback_off_physical_{compute_backend}_both_branches") is True and
        claims.get("formal_reference_comparison_complete") is True and
        claims.get("public_moving_nest_admitted") is False and
        claims.get("device_resident_relocation") is False and
        claims.get("throughput_promotion") is False,
        "campaign claims are incomplete or overbroad",
    )
    inputs = campaign.get("inputs", {})
    bound_inputs = {
        name: artifact(resolve_record(inputs.get(name), f"inputs.{name}"))
        for name in (
            "policy", "moving_lifetime_chain", "moving_runtime_report",
            "reference_runtime_report")
    }
    comparison = campaign.get("comparison_artifacts", {})
    bound_comparison = {
        name: artifact(resolve_record(record, f"comparison_artifacts.{name}"))
        for name, record in comparison.items()
    }
    require(set(bound_comparison) == {
                "moving_series", "moving_source_manifest", "reference_series",
                "reference_source_manifest", "comparison_report"},
            "comparison artifact inventory changed")
    return {
        "schema":
            f"nimbus.evidence.moving_nested_{compute_backend}_reference_frontier",
        "schema_version": 1,
        "status": "pass",
        "qualification_eligible": True,
        "scope": (
            f"same_binary_same_backend_physical_{compute_backend}_moving_child_"
            "against_oversized_stationary_reference"),
        "campaign_report": artifact(campaign_report_path),
        "inputs": {
            "binary_sha256": inputs.get("binary_sha256"),
            **bound_inputs,
        },
        "comparison_artifacts": bound_comparison,
        "coverage": campaign.get("coverage"),
        "metrics": campaign.get("metrics"),
        "checks": checks,
        "runtime_summary": campaign.get("runtime_summary"),
        "claims": claims,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compute-backend", choices=SUPPORTED_BACKENDS, default="vulkan")
    parser.add_argument("--campaign-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default_campaign = ROOT / "tests/artifacts" / (
        "moving_nested_vulkan_reference_frontier_v13_20260819"
        if args.compute_backend == "vulkan" else
        "moving_nested_metal_reference_frontier_v14") / "campaign_report.json"
    campaign = (args.campaign_report or default_campaign).resolve()
    output = (args.output or ROOT / "reports/qualification" / (
        f"moving_nested_{args.compute_backend}_reference_frontier_v1_pass.json"
    )).resolve()
    evidence = build_evidence(args.compute_backend, campaign)
    write_json(output, evidence)
    print(json.dumps({
        "status": evidence["status"],
        "qualification_eligible": evidence["qualification_eligible"],
        "output": relative(output),
        "sha256": sha256(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[moving-nested-accelerator-frontier-evidence] FAIL: {exc}",
              file=sys.stderr)
        raise SystemExit(1)

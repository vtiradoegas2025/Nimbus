#!/usr/bin/env python3
"""Create the pre-comparison Metal backend amendment after evidence is sealed."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_moving_nested_supercell_reference as comparator


ROOT = Path(__file__).resolve().parents[3]
PRIOR_POLICY = (
    ROOT / "tests/qualification/fixtures/"
    "moving_nested_supercell_reference_frontier_v13.json"
)


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


def relative_to_output(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.resolve().parent)


def build_policy(evidence_path: Path, output_path: Path,
                 review_date: str) -> dict[str, Any]:
    evidence_path = evidence_path.resolve()
    output_path = output_path.resolve()
    evidence = load_json(evidence_path)
    claims = evidence.get("claims", {})
    require(evidence.get("schema") ==
                "nimbus.evidence.moving_nested_metal_reference_runtime" and
            evidence.get("schema_version") == 1 and
            evidence.get("status") == "pass" and
            evidence.get("qualification_eligible") is False and
            evidence.get("candidate_metric_values_inspected") is False and
            claims.get("same_binary_candidate_and_reference") is True and
            claims.get("same_metal_backend_candidate_and_reference") is True and
            claims.get("formal_comparison_complete") is False and
            claims.get("public_moving_nest_admitted") is False,
            "Metal reference evidence is not eligible for pre-comparison ratification")
    require(len(review_date) == 10 and review_date[4] == "-" and
            review_date[7] == "-",
            "review date must use YYYY-MM-DD")
    return {
        "schema": "nimbus.qualification.moving_nested_reference_frontier.policy",
        "schema_version": 14,
        "policy_id":
            "moving-nested-reference-frontier-v14-metal-backend-amendment",
        "policy_status": "ratified",
        "scope": (
            "same-equation 600 m moving-child versus oversized stationary "
            "reference on one physical Metal backend with all v13 thresholds, "
            "geometry, aperture, cadence, and movement semantics inherited "
            "unchanged"),
        "prior_policy": {
            "path": relative_to_output(PRIOR_POLICY, output_path),
            "sha256": comparator.V13_POLICY_SHA256,
            "schema_version": 13,
            "policy_id":
                "moving-nested-reference-frontier-v13-vulkan-backend-amendment",
        },
        "backend_evidence": {
            "path": relative_to_output(evidence_path, output_path),
            "sha256": sha256(evidence_path),
            "schema_version": 1,
            "status": "pass",
        },
        "ratification": {
            "decision":
                "pre_comparison_metal_backend_identity_amendment_of_v13",
            "review_date": review_date,
            "reviewer_role": "independent_science_policy_audit",
            "candidate_metric_values_inspected": False,
            "minimum_input_series_schema_version": 2,
            "backend_identity_amendment": {
                "reason": "same_backend_reference_now_available",
                "candidate_backend": "metal",
                "reference_backend": "metal",
                "fallback_allowed": False,
                "same_binary_required": True,
                "physics_signature_sha256":
                    comparator.METAL_PHYSICS_SIGNATURE_SHA256,
                "host_authoritative_kessler_state_commit": True,
                "native_smagorinsky_sgs_dispatch_required": True,
                "native_kessler_process_dispatch_required": True,
                "numerical_thresholds_changed": False,
                "movement_algorithm_changed": False,
                "geometry_or_aperture_changed": False,
                "measurement_cadence_changed": False,
            },
        },
        "limitations": [
            "All 21 numerical thresholds, metric formulas, geometry, aperture, measurement cadence, and movement rules are inherited unchanged from the immutable v13 policy chain.",
            "The Metal backend identity and comparison-level numerical signature are ratified before candidate-versus-reference metric values are inspected.",
            "Both branches must bind one binary and prove fallback-off physical Metal execution.",
            "Kessler process kernels and Smagorinsky/SGS state application are native Metal while the authoritative Kessler state commit remains on the host.",
            "A pass remains bounded to this exact flat Cartesian C-grid, one-way, Kessler plus 3-D Smagorinsky supercell tuple and does not establish device-resident relocation, terrain, broader physics combinations, or throughput promotion.",
        ],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-date", default=date.today().isoformat())
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    output = args.output.resolve()
    policy = build_policy(evidence, output, args.review_date)
    write_json(output, policy)
    comparator.validate_policy(load_json(output), output)
    print(json.dumps({
        "status": "ratified",
        "candidate_metric_values_inspected": False,
        "output": str(output),
        "sha256": sha256(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError,
            comparator.ContractError) as exc:
        print(f"[moving-nested-metal-policy] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)

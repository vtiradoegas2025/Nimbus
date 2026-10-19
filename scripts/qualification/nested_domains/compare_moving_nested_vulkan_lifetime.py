#!/usr/bin/env python3
"""Compare the complete segmented fresh-start Vulkan lifetime to the frozen CPU reference.

The ratified 21 v12 formulas and thresholds are reused unchanged.  This remains
diagnostic because the candidate is segmented by exact checkpoint/restart and
the frozen reference was produced by a different backend identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_moving_nested_vulkan_continuation as continuation
import compare_moving_nested_supercell_reference as comparator
import extract_moving_nested_reference_series as extractor
import inspect_moving_nested_vulkan_lifetime_chain as chain


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = (
    ROOT / "tests" / "artifacts" /
    "moving_nested_reference_frontier_v12_20260818" / "comparison"
)
DEFAULT_REFERENCE = REFERENCE_ROOT / "oversized_reference.json"
DEFAULT_BASELINE_REPORT = REFERENCE_ROOT / "comparison_report.json"
DEFAULT_POLICY = (
    ROOT / "tests" / "qualification" / "fixtures" /
    "moving_nested_supercell_reference_frontier_v12.json"
)
DEFAULT_CHAIN_REPORT = chain.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    ROOT / "tests" / "artifacts" /
    "moving_nested_vulkan_lifetime_v2_20260819_reference_diagnostic.json"
)
SCHEMA = "nimbus.diagnostic.moving_nested_vulkan_lifetime_reference"


def artifact(path: Path) -> dict[str, str]:
    return {"path": chain.relative(path), "sha256": continuation.sha256(path)}


def build_report(reference_path: Path, baseline_path: Path, policy_path: Path,
                 chain_report_path: Path) -> dict[str, Any]:
    reference = continuation.load_json(reference_path)
    baseline = continuation.load_json(baseline_path)
    lifetime = continuation.load_json(chain_report_path)
    continuation.require(
        reference.get("schema_version") == 2 and
        reference.get("role") == "oversized_reference",
        "reference is not the frozen v2 oversized series")
    continuation.require(
        baseline.get("status") == "pass" and
        baseline.get("qualification_eligible") is True and
        len(baseline.get("checks", [])) == 21 and
        all(check.get("passed") is True for check in baseline["checks"]),
        "frozen baseline comparison is not a 21/21 qualification pass")
    continuation.require(
        lifetime.get("schema") == chain.SCHEMA and
        lifetime.get("status") == "complete" and
        lifetime.get("qualification_eligible") is False and
        lifetime.get("progress", {}).get("completed_through_s") == 4500.0 and
        lifetime.get("progress", {}).get("tracker_samples") == 150 and
        lifetime.get("progress", {}).get("checkpoint_count") == 10 and
        lifetime.get("claims", {}).get("single_binary_config_identity") is True and
        lifetime.get("claims", {}).get("uninterrupted_single_process") is True and
        lifetime.get("claims", {}).get("checkpoint_lineage_exact") is True,
        "Vulkan lifetime chain is not the complete validated 4500-second campaign")

    try:
        policy = comparator.validate_policy(
            continuation.load_json(policy_path), policy_path)
    except comparator.ContractError as exc:
        raise continuation.DiagnosticError(f"invalid comparison policy: {exc}") from exc
    times = comparator.expected_times(policy["expected_contract"])
    continuation.require(times[0] == 30.0 and times[-1] == 4500.0 and len(times) == 150,
                         "policy is not the frozen 30-4500 second cadence")
    aperture_record = policy.get("_verified_qualification_aperture")
    continuation.require(isinstance(aperture_record, dict),
                         "policy lacks a verified aperture")
    aperture = continuation.load_json(Path(aperture_record["path"]))
    extractor.validate_aperture(aperture)

    candidate_samples: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    relocations: list[dict[str, Any]] = []
    segment_inputs: list[dict[str, Any]] = []
    consumed_times: list[float] = []
    for segment_record in lifetime.get("segments", []):
        start_s = continuation.number(segment_record.get("start_s"), "segment start")
        end_s = continuation.number(segment_record.get("end_s"), "segment end")
        segment_path = (ROOT / str(segment_record.get("directory"))).resolve()
        report_path = segment_path / "moving_nested_vulkan_runtime_report.json"
        continuation.require(
            continuation.sha256(report_path) ==
            segment_record.get("report", {}).get("sha256"),
            f"chain-bound segment report changed: {segment_path.name}")
        report = continuation.load_json(report_path)
        continuation.require(
            report.get("status") == "pass" and
            report.get("claims", {}).get("fallback_off_physical_vulkan") is True and
            report.get("inputs", {}).get("binary_sha256") ==
            lifetime.get("inputs", {}).get("binary_sha256") and
            report.get("inputs", {}).get("config_sha256") ==
            lifetime.get("inputs", {}).get("config_sha256"),
            f"segment runtime identity changed: {segment_path.name}")
        segment_times = [time_s for time_s in times if start_s < time_s <= end_s]
        continuation.require(segment_times and segment_times[0] == start_s + 30.0 and
                             segment_times[-1] == end_s,
                             f"segment does not align to policy cadence: {segment_path.name}")
        samples, provenance = continuation.extract_vulkan_suffix(
            segment_path, segment_times, policy["expected_contract"], aperture)
        candidate_samples.extend(samples)
        inventories.extend(provenance["artifacts"])
        relocations.extend(provenance["relocations"])
        consumed_times.extend(segment_times)
        segment_inputs.append({
            "start_s": start_s,
            "end_s": end_s,
            "directory": chain.relative(segment_path),
            "runtime_report": artifact(report_path),
        })
    continuation.require(consumed_times == times,
                         "segmented Vulkan diagnostics do not exactly cover policy cadence")

    result = continuation.metric_result(candidate_samples, reference.get("samples", []), policy)
    failed = [check["name"] for check in result["checks"] if not check["passed"]]
    status = "diagnostic_pass" if not failed else "diagnostic_fail"
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": status,
        "qualification_eligible": False,
        "scope": "uninterrupted_fresh_start_physical_vulkan_lifetime_against_frozen_cpu_reference",
        "claims": {
            "formal_qualification_comparison": False,
            "single_candidate_binary_config_identity": True,
            "uninterrupted_candidate_process": True,
            "exact_checkpoint_lineage": True,
            "fresh_configured_start": True,
            "physical_vulkan_fallback_off_full_lifetime": True,
            "ratified_v12_formulas_and_thresholds_reused_unchanged": True,
            "bounded_science_lifetime": not failed,
        },
        "limitations": [
            "The frozen oversized reference uses a different CPU backend identity.",
            "The diagnostic therefore cannot replace the formal same-binary, single-backend comparator.",
            "Microphysics state commits remain host-authoritative in this Vulkan profile.",
            "Moving-nest relocation remains host-owned rather than device-resident.",
        ],
        "lifetime": {
            "start_s": 0.0,
            "final_time_s": times[-1],
            "sample_count": len(candidate_samples),
            "segment_count": len(segment_inputs),
            "checkpoint_count": lifetime["progress"]["checkpoint_count"],
            "relocation_count": len(relocations),
            "relocations": relocations,
        },
        "inputs": {
            "oversized_reference_series": artifact(reference_path),
            "baseline_qualification_report": artifact(baseline_path),
            "policy": artifact(policy_path),
            "lifetime_chain_report": artifact(chain_report_path),
            "binary_sha256": lifetime["inputs"]["binary_sha256"],
            "config_sha256": lifetime["inputs"]["config_sha256"],
            "segments": segment_inputs,
        },
        "threshold_result": {
            "check_count": len(result["checks"]),
            "passed_check_count": result["passed_check_count"],
            "failed_check_count": result["failed_check_count"],
            "failed_checks": failed,
        },
        "coverage": result["coverage"],
        "metrics": result["metrics"],
        "checks": result["checks"],
        "vulkan_artifacts": inventories,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--chain-report", type=Path, default=DEFAULT_CHAIN_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(
            args.reference.resolve(), args.baseline_report.resolve(),
            args.policy.resolve(), args.chain_report.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    except (continuation.DiagnosticError, extractor.ExtractionError) as exc:
        print(f"moving nested Vulkan lifetime diagnostic failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"],
        "qualification_eligible": report["qualification_eligible"],
        **report["threshold_result"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "diagnostic_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure a CPU-prefix/Vulkan-suffix moving storm against the frozen reference.

This is intentionally a diagnostic, not the formal qualification comparator.
The candidate has heterogeneous binary/backend provenance, so a successful
threshold comparison remains ``qualification_eligible: false``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_moving_nested_supercell_reference as comparator
import extract_moving_nested_reference_series as extractor


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = (
    ROOT / "tests" / "artifacts" /
    "moving_nested_reference_frontier_v12_20260818" / "comparison"
)
DEFAULT_MOVING = REFERENCE_ROOT / "moving_child.json"
DEFAULT_REFERENCE = REFERENCE_ROOT / "oversized_reference.json"
DEFAULT_BASELINE_REPORT = REFERENCE_ROOT / "comparison_report.json"
DEFAULT_POLICY = (
    ROOT / "tests" / "qualification" / "fixtures" /
    "moving_nested_supercell_reference_frontier_v12.json"
)
DEFAULT_VULKAN_RUN = (
    ROOT / "tests" / "artifacts" /
    "moving_nested_vulkan_live_storm_v1_20260819"
)
DEFAULT_OUTPUT = DEFAULT_VULKAN_RUN / "hybrid_reference_diagnostic.json"
SCHEMA = "nimbus.diagnostic.moving_nested_vulkan_reference_continuation"
MOVEMENT_ALGORITHM = "qualified_envelope_hysteresis_v14"


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def number(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def nonnegative(value: object, label: str) -> float:
    result = number(value, label)
    require(result >= 0.0, f"{label} must be nonnegative")
    return result


def coordinate(value: object, label: str) -> tuple[float, float]:
    require(isinstance(value, dict), f"{label} must be an object")
    return number(value.get("x"), f"{label}.x"), number(value.get("y"), f"{label}.y")


def add_check(checks: list[dict[str, Any]], name: str, value: float,
              threshold: float, comparison: str = "<=") -> None:
    passed = value <= threshold if comparison == "<=" else value >= threshold
    checks.append({
        "name": name,
        "value": value,
        "threshold": threshold,
        "comparator": comparison,
        "passed": passed,
    })


def metric_result(candidate_samples: list[dict[str, Any]],
                  reference_samples: list[dict[str, Any]],
                  policy: dict[str, Any]) -> dict[str, Any]:
    """Apply the formal comparator's 21 formulas without its provenance claim."""
    times = comparator.expected_times(policy["expected_contract"])
    candidate = {number(item.get("simulation_time_s"), "candidate time"): item
                 for item in candidate_samples}
    reference = {number(item.get("simulation_time_s"), "reference time"): item
                 for item in reference_samples}
    require(set(candidate) == set(times), "candidate does not exactly cover policy cadence")
    require(set(reference) == set(times), "reference does not exactly cover policy cadence")

    definitions = policy["derived_definitions"]
    for label, series in (("candidate", candidate), ("reference", reference)):
        for time_s, sample in series.items():
            require(type(sample.get("selected")) is bool,
                    f"{label} selection is invalid at {time_s}s")
            for family in ("cold_pool", "precipitation"):
                record = sample.get(family)
                require(isinstance(record, dict) and
                        record.get("definition") == definitions[family]["id"],
                        f"{label} {family} definition changed at {time_s}s")

    reference_selected = [time_s for time_s in times if reference[time_s]["selected"]]
    require(reference_selected, "reference never selects a storm")
    mature_start = reference_selected[0]
    mature_times = [time_s for time_s in times if time_s >= mature_start]
    paired_flags = [candidate[time_s]["selected"] and reference[time_s]["selected"]
                    for time_s in mature_times]
    paired_times = [time_s for time_s, paired in zip(mature_times, paired_flags) if paired]
    require(paired_times, "candidate and reference have no paired selected samples")

    paired_fraction = len(paired_times) / len(mature_times)
    longest_gap = comparator.maximum_consecutive_false(paired_flags)
    coverage = {
        "declared_sample_count": len(times),
        "common_sample_count": len(times),
        "mature_start_s": mature_start,
        "mature_sample_count": len(mature_times),
        "paired_selected_samples": len(paired_times),
        "paired_selected_fraction": paired_fraction,
        "maximum_consecutive_unpaired_samples": longest_gap,
        "final_sample_paired": paired_flags[-1],
        "missing_candidate_times_s": [],
        "missing_reference_times_s": [],
    }
    checks: list[dict[str, Any]] = []
    coverage_policy = policy["coverage_thresholds"]
    add_check(checks, "coverage.paired_selected_fraction", paired_fraction,
              number(coverage_policy["minimum_paired_selected_fraction"],
                     "minimum paired fraction"), ">=")
    add_check(checks, "coverage.maximum_consecutive_unpaired_samples", float(longest_gap),
              number(coverage_policy["maximum_consecutive_unpaired_samples"],
                     "maximum unpaired samples"))
    if coverage_policy["require_final_paired_sample"]:
        add_check(checks, "coverage.final_sample_paired",
                  1.0 if paired_flags[-1] else 0.0, 1.0, ">=")

    pairs = [(candidate[time_s], reference[time_s]) for time_s in paired_times]
    path_errors = []
    for left, right in pairs:
        lx, ly = coordinate(left.get("core_centroid_m"), "candidate core centroid")
        rx, ry = coordinate(right.get("core_centroid_m"), "reference core centroid")
        path_errors.append(math.hypot(lx - rx, ly - ry))
    path_metrics = {
        "sample_count": len(path_errors),
        "rmse_m": comparator.rmse(path_errors),
        "p95_m": comparator.percentile(path_errors, 0.95),
        "maximum_m": max(path_errors),
        "final_m": path_errors[-1],
    }
    for key, value in path_metrics.items():
        if key != "sample_count":
            add_check(checks, f"path.{key}", value,
                      number(policy["metric_thresholds"]["path"][key], f"path {key}"))

    intensity_policy = policy["metric_thresholds"]["intensity"]
    selected_candidate = [(time_s, candidate[time_s]) for time_s in mature_times
                          if candidate[time_s]["selected"]]
    selected_reference = [(time_s, reference[time_s]) for time_s in mature_times
                          if reference[time_s]["selected"]]

    def intensity(key: str, floor: float) -> dict[str, float]:
        paired_left = [nonnegative(left.get(key), f"candidate {key}") for left, _ in pairs]
        paired_right = [nonnegative(right.get(key), f"reference {key}") for _, right in pairs]
        left_values = [nonnegative(sample.get(key), f"candidate {key}")
                       for _, sample in selected_candidate]
        right_values = [nonnegative(sample.get(key), f"reference {key}")
                        for _, sample in selected_reference]
        left_peak = max(left_values)
        right_peak = max(right_values)
        left_time = selected_candidate[left_values.index(left_peak)][0]
        right_time = selected_reference[right_values.index(right_peak)][0]
        return {
            "normalized_rmse": comparator.normalized_rmse(
                paired_left, paired_right, floor),
            "peak_relative_error": abs(left_peak - right_peak) /
                                   max(abs(right_peak), floor),
            "peak_time_error_s": abs(left_time - right_time),
        }

    updraft = intensity("maximum_updraft_ms",
                        nonnegative(intensity_policy["updraft_scale_floor_ms"],
                                    "updraft scale floor"))
    uh = intensity("maximum_absolute_uh_m2s2",
                   nonnegative(intensity_policy["uh_scale_floor_m2s2"], "UH scale floor"))
    intensity_metrics = {
        "sample_count": len(pairs),
        "maximum_updraft": updraft,
        "maximum_absolute_uh": uh,
    }
    for prefix, values in (("maximum_updraft", updraft), ("maximum_absolute_uh", uh)):
        for key, value in values.items():
            policy_key = f"{prefix}_{key}"
            add_check(checks, f"intensity.{policy_key}", value,
                      number(intensity_policy[policy_key], f"intensity {policy_key}"))

    cold_pairs = [(candidate[time_s]["cold_pool"], reference[time_s]["cold_pool"])
                  for time_s in times]
    cold_policy = policy["metric_thresholds"]["cold_pool"]
    cold_area = ([nonnegative(left.get("area_m2"), "candidate cold-pool area")
                  for left, _ in cold_pairs],
                 [nonnegative(right.get("area_m2"), "reference cold-pool area")
                  for _, right in cold_pairs])
    cold_deficit = ([nonnegative(left.get("integrated_deficit_k_m2"), "candidate deficit")
                     for left, _ in cold_pairs],
                    [nonnegative(right.get("integrated_deficit_k_m2"), "reference deficit")
                     for _, right in cold_pairs])
    cold_minimum = ([number(left.get("minimum_theta_rho_perturbation_k"), "candidate minimum")
                     for left, _ in cold_pairs],
                    [number(right.get("minimum_theta_rho_perturbation_k"), "reference minimum")
                     for _, right in cold_pairs])
    centroid_errors = []
    for left, right in cold_pairs:
        if nonnegative(left.get("area_m2"), "candidate cold-pool area") > 0.0 and \
                nonnegative(right.get("area_m2"), "reference cold-pool area") > 0.0:
            lx, ly = coordinate(left.get("centroid_m"), "candidate cold-pool centroid")
            rx, ry = coordinate(right.get("centroid_m"), "reference cold-pool centroid")
            centroid_errors.append(math.hypot(lx - rx, ly - ry))
    require(len(centroid_errors) >= cold_policy["minimum_active_samples"],
            "insufficient active cold-pool centroid samples")
    cold_metrics = {
        "sample_count": len(cold_pairs),
        "active_centroid_samples": len(centroid_errors),
        "area_normalized_rmse": comparator.normalized_rmse(
            *cold_area, nonnegative(cold_policy["area_scale_floor_m2"], "cold area floor")),
        "integrated_deficit_normalized_rmse": comparator.normalized_rmse(
            *cold_deficit, nonnegative(cold_policy["integrated_deficit_scale_floor_k_m2"],
                                       "cold deficit floor")),
        "minimum_theta_rho_perturbation_rmse_k": comparator.rmse(
            [left - right for left, right in zip(*cold_minimum)]),
        "centroid_rmse_m": comparator.rmse(centroid_errors),
    }
    for key, value in cold_metrics.items():
        if key not in {"sample_count", "active_centroid_samples"}:
            add_check(checks, f"cold_pool.{key}", value,
                      number(cold_policy[key], f"cold pool {key}"))

    precip_pairs = [(candidate[time_s]["precipitation"],
                     reference[time_s]["precipitation"]) for time_s in times]

    def precip_series(key: str) -> tuple[list[float], list[float]]:
        return ([nonnegative(left.get(key), f"candidate precipitation {key}")
                 for left, _ in precip_pairs],
                [nonnegative(right.get(key), f"reference precipitation {key}")
                 for _, right in precip_pairs])

    precip_policy = policy["metric_thresholds"]["precipitation"]
    mass = precip_series("accumulated_mass_kg")
    area = precip_series("footprint_area_m2")
    p95 = precip_series("p95_depth_mm")
    maximum = precip_series("maximum_depth_mm")
    precip_metrics = {
        "sample_count": len(precip_pairs),
        "final_accumulated_mass_relative_error": abs(mass[0][-1] - mass[1][-1]) /
            max(abs(mass[1][-1]), nonnegative(precip_policy["mass_scale_floor_kg"],
                                              "precipitation mass floor")),
        "footprint_area_normalized_rmse": comparator.normalized_rmse(
            *area, nonnegative(precip_policy["area_scale_floor_m2"], "precipitation area floor")),
        "p95_depth_normalized_rmse": comparator.normalized_rmse(
            *p95, nonnegative(precip_policy["depth_scale_floor_mm"], "precipitation depth floor")),
        "maximum_depth_normalized_rmse": comparator.normalized_rmse(
            *maximum, nonnegative(precip_policy["depth_scale_floor_mm"],
                                  "precipitation depth floor")),
    }
    for key, value in precip_metrics.items():
        if key != "sample_count":
            add_check(checks, f"precipitation.{key}", value,
                      number(precip_policy[key], f"precipitation {key}"))

    require(len(checks) == 21, f"expected 21 checks, produced {len(checks)}")
    return {
        "coverage": coverage,
        "metrics": {
            "path": path_metrics,
            "intensity": intensity_metrics,
            "cold_pool": cold_metrics,
            "precipitation": precip_metrics,
        },
        "checks": checks,
        "passed_check_count": sum(check["passed"] for check in checks),
        "failed_check_count": sum(not check["passed"] for check in checks),
    }


def extract_vulkan_suffix(run: Path, suffix_times: list[float],
                          expected: dict[str, Any], aperture: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    history = run / "nests" / "child_0000" / "qualification_diagnostics"
    steps = extractor.step_publications(history, "nested_child_qualification_diagnostics")
    trackers = extractor.tracker_publications(run)
    require(set(steps) == set(suffix_times), "Vulkan diagnostic stream does not exactly cover suffix")
    require(set(trackers) == set(suffix_times), "Vulkan tracker stream does not exactly cover suffix")
    exclusion = expected["relaxation_exclusion_cells"]
    samples: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    relocations: list[dict[str, Any]] = []
    for time_s in suffix_times:
        step_path, manifest = steps[time_s]
        tracker_path, tracker = trackers[time_s]
        require(tracker.get("tracking_source") == "synchronized_nested_child" and
                tracker.get("detector_coordinate_frame") == "earth_absolute",
                f"Vulkan tracker frame/source changed at {time_s}s")
        authority = tracker.get("movement_authority")
        require(isinstance(authority, dict) and
                authority.get("algorithm") == MOVEMENT_ALGORITHM,
                f"Vulkan movement authority changed at {time_s}s")
        bounds = authority.get("legal_parent_shift_bounds_cells")
        require(isinstance(bounds, dict) and bounds.get("available") is True,
                f"Vulkan tracker lacks live placement bounds at {time_s}s")
        validity = tracker.get("moving_nest_scientific_validity")
        require(isinstance(validity, dict) and validity.get("status") != "invalid",
                f"Vulkan moving nest is invalid at {time_s}s")
        region = extractor.moving_region(manifest, exclusion, aperture)
        cold_pool, precipitation, _ = extractor.diagnostic_records(step_path, manifest, region)
        selected = tracker.get("selected_feature_available")
        require(type(selected) is bool, f"Vulkan selection is invalid at {time_s}s")
        sample: dict[str, Any] = {
            "simulation_time_s": time_s,
            "selected": selected,
            "region": region,
            "core_centroid_m": None,
            "maximum_updraft_ms": None,
            "maximum_absolute_uh_m2s2": None,
            "cold_pool": cold_pool,
            "precipitation": precipitation,
        }
        if selected:
            centroid = tracker.get("core_centroid_m")
            rotation = tracker.get("rotation")
            require(isinstance(centroid, dict) and isinstance(rotation, dict),
                    f"selected Vulkan tracker lacks storm data at {time_s}s")
            sample["core_centroid_m"] = {
                "x": number(centroid.get("x"), "Vulkan centroid x"),
                "y": number(centroid.get("y"), "Vulkan centroid y"),
            }
            sample["maximum_updraft_ms"] = number(
                tracker.get("maximum_updraft_ms"), "Vulkan maximum updraft")
            sample["maximum_absolute_uh_m2s2"] = abs(number(
                rotation.get("maximum_absolute_updraft_helicity_m2s2"), "Vulkan maximum UH"))
        samples.append(sample)
        inventory.append({
            "simulation_time_s": time_s,
            "step_manifest": artifact(step_path / "manifest.json"),
            "cold_pool_boundary": artifact(step_path / "cold_pool_boundary.npy"),
            "accumulated_rainfall": artifact(step_path / "accumulated_rainfall.npy"),
            "storm_record": artifact(tracker_path),
        })
        if authority.get("commit_performed") is True:
            relocations.append({
                "simulation_time_s": time_s,
                "shift_cells": copy.deepcopy(authority.get("committed_shift_cells")),
                "cumulative_shift_cells": copy.deepcopy(
                    authority.get("cumulative_committed_shift_cells")),
                "reason": authority.get("reason"),
            })
    return samples, {"artifacts": inventory, "relocations": relocations}


def build_report(moving_path: Path, reference_path: Path, baseline_path: Path,
                 policy_path: Path, run: Path, split_time_s: float) -> dict[str, Any]:
    moving = load_json(moving_path)
    reference = load_json(reference_path)
    baseline = load_json(baseline_path)
    require(moving.get("schema_version") == 2 and moving.get("role") == "moving_child",
            "CPU prefix is not the frozen v2 moving-child series")
    require(reference.get("schema_version") == 2 and
            reference.get("role") == "oversized_reference",
            "reference is not the frozen v2 oversized series")
    require(baseline.get("status") == "pass" and
            baseline.get("qualification_eligible") is True and
            len(baseline.get("checks", [])) == 21 and
            all(check.get("passed") is True for check in baseline["checks"]),
            "frozen baseline comparison is not a 21/21 qualification pass")

    raw_policy = load_json(policy_path)
    try:
        policy = comparator.validate_policy(raw_policy, policy_path)
    except comparator.ContractError as exc:
        raise DiagnosticError(f"invalid comparison policy: {exc}") from exc
    times = comparator.expected_times(policy["expected_contract"])
    require(split_time_s in times and split_time_s < times[-1], "split time is outside policy cadence")
    prefix_times = [time_s for time_s in times if time_s <= split_time_s]
    suffix_times = [time_s for time_s in times if time_s > split_time_s]

    moving_by_time = {number(sample.get("simulation_time_s"), "CPU moving time"): sample
                      for sample in moving.get("samples", [])}
    require(set(moving_by_time) == set(times), "frozen CPU moving series cadence changed")

    aperture_record = policy.get("_verified_qualification_aperture")
    require(isinstance(aperture_record, dict), "policy lacks a verified aperture")
    aperture = load_json(Path(aperture_record["path"]))
    extractor.validate_aperture(aperture)

    runtime_report_path = run / "moving_nested_vulkan_runtime_report.json"
    runtime_report = load_json(runtime_report_path)
    require(runtime_report.get("status") == "pass" and
            runtime_report.get("qualification_eligible") is False and
            runtime_report.get("claims", {}).get("fallback_off_physical_vulkan") is True and
            number(runtime_report.get("inputs", {}).get("source_checkpoint_time_s"),
                   "runtime checkpoint time") == split_time_s,
            "Vulkan runtime report is not the expected diagnostic pass")
    binary_path = ROOT / runtime_report["inputs"]["binary"]
    config_path = ROOT / runtime_report["inputs"]["config"]
    require(sha256(binary_path) == runtime_report["inputs"]["binary_sha256"],
            "Vulkan runtime binary hash changed")
    require(sha256(config_path) == runtime_report["inputs"]["config_sha256"],
            "Vulkan runtime config hash changed")

    suffix, suffix_provenance = extract_vulkan_suffix(
        run, suffix_times, policy["expected_contract"], aperture)
    hybrid = [copy.deepcopy(moving_by_time[time_s]) for time_s in prefix_times] + suffix
    result = metric_result(hybrid, reference.get("samples", []), policy)
    failed = [check["name"] for check in result["checks"] if not check["passed"]]
    status = "diagnostic_pass" if not failed else "diagnostic_fail"
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": status,
        "qualification_eligible": False,
        "scope": "heterogeneous_cpu_prefix_physical_vulkan_suffix_against_frozen_reference",
        "claims": {
            "formal_qualification_comparison": False,
            "heterogeneous_binary_backend_provenance": True,
            "ratified_v12_formulas_and_thresholds_reused_unchanged": True,
            "physical_vulkan_fallback_off_suffix": True,
            "bounded_science_continuation": not failed,
        },
        "limitations": [
            "The CPU prefix and Vulkan suffix do not share one binary/backend identity.",
            "The diagnostic cannot replace the formal same-binary, single-backend qualification comparator.",
            "Only the 4050-4500 second suffix was executed on Vulkan.",
            "Microphysics state commits remain host-authoritative in this Vulkan profile.",
        ],
        "split": {
            "cpu_prefix_inclusive_through_s": split_time_s,
            "vulkan_suffix_start_s": suffix_times[0],
            "final_time_s": suffix_times[-1],
            "cpu_sample_count": len(prefix_times),
            "vulkan_sample_count": len(suffix_times),
            "total_sample_count": len(hybrid),
        },
        "inputs": {
            "cpu_moving_series": artifact(moving_path),
            "oversized_reference_series": artifact(reference_path),
            "baseline_qualification_report": artifact(baseline_path),
            "policy": artifact(policy_path),
            "vulkan_runtime_report": artifact(runtime_report_path),
            "vulkan_binary": artifact(binary_path),
            "vulkan_config": artifact(config_path),
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
        "vulkan_suffix": suffix_provenance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving", type=Path, default=DEFAULT_MOVING)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--vulkan-run", type=Path, default=DEFAULT_VULKAN_RUN)
    parser.add_argument("--split-time-s", type=float, default=4050.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(
            args.moving.resolve(), args.reference.resolve(),
            args.baseline_report.resolve(), args.policy.resolve(),
            args.vulkan_run.resolve(), args.split_time_s)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    except (DiagnosticError, extractor.ExtractionError) as exc:
        print(f"moving nested Vulkan continuation diagnostic failed: {exc}", file=sys.stderr)
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

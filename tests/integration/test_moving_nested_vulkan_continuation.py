#!/usr/bin/env python3
"""Regression tests for the heterogeneous moving-nest continuation metric lane."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "qualification" / "nested_domains"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_moving_nested_supercell_reference as comparator
import compare_moving_nested_vulkan_continuation as diagnostic


HAS_FROZEN_ARTIFACTS = all(path.is_file() for path in (
    diagnostic.DEFAULT_MOVING,
    diagnostic.DEFAULT_REFERENCE,
    diagnostic.DEFAULT_BASELINE_REPORT,
))


def synthetic_case() -> tuple[dict, list[dict], list[dict]]:
    policy = {
        "expected_contract": {
            "sample_start_s": 30.0,
            "sample_interval_s": 30.0,
            "duration_s": 60.0,
        },
        "coverage_thresholds": {
            "minimum_paired_selected_fraction": 1.0,
            "maximum_consecutive_unpaired_samples": 0,
            "require_final_paired_sample": True,
        },
        "derived_definitions": {
            "cold_pool": {"id": "cold"},
            "precipitation": {"id": "rain"},
        },
        "metric_thresholds": {
            "path": {"rmse_m": 1.0, "p95_m": 1.0, "maximum_m": 1.0, "final_m": 1.0},
            "intensity": {
                "updraft_scale_floor_ms": 1.0,
                "uh_scale_floor_m2s2": 1.0,
                "maximum_updraft_normalized_rmse": 0.01,
                "maximum_updraft_peak_relative_error": 0.01,
                "maximum_updraft_peak_time_error_s": 1.0,
                "maximum_absolute_uh_normalized_rmse": 0.01,
                "maximum_absolute_uh_peak_relative_error": 0.01,
                "maximum_absolute_uh_peak_time_error_s": 1.0,
            },
            "cold_pool": {
                "area_scale_floor_m2": 1.0,
                "integrated_deficit_scale_floor_k_m2": 1.0,
                "minimum_active_samples": 1,
                "area_normalized_rmse": 0.01,
                "integrated_deficit_normalized_rmse": 0.01,
                "minimum_theta_rho_perturbation_rmse_k": 0.01,
                "centroid_rmse_m": 1.0,
            },
            "precipitation": {
                "mass_scale_floor_kg": 1.0,
                "area_scale_floor_m2": 1.0,
                "depth_scale_floor_mm": 1.0,
                "final_accumulated_mass_relative_error": 0.01,
                "footprint_area_normalized_rmse": 0.01,
                "p95_depth_normalized_rmse": 0.01,
                "maximum_depth_normalized_rmse": 0.01,
            },
        },
    }
    samples = []
    for time_s in (30.0, 60.0):
        samples.append({
            "simulation_time_s": time_s,
            "selected": True,
            "core_centroid_m": {"x": time_s, "y": 2.0 * time_s},
            "maximum_updraft_ms": 20.0 + time_s,
            "maximum_absolute_uh_m2s2": 100.0 + time_s,
            "cold_pool": {
                "definition": "cold", "area_m2": 10.0 + time_s,
                "integrated_deficit_k_m2": 50.0 + time_s,
                "minimum_theta_rho_perturbation_k": -3.0,
                "centroid_m": {"x": time_s, "y": time_s},
            },
            "precipitation": {
                "definition": "rain", "accumulated_mass_kg": 100.0 + time_s,
                "footprint_area_m2": 20.0 + time_s, "p95_depth_mm": 2.0,
                "maximum_depth_mm": 4.0,
            },
        })
    return policy, copy.deepcopy(samples), copy.deepcopy(samples)


class MovingNestedVulkanContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not HAS_FROZEN_ARTIFACTS:
            return
        cls.moving = diagnostic.load_json(diagnostic.DEFAULT_MOVING)
        cls.reference = diagnostic.load_json(diagnostic.DEFAULT_REFERENCE)
        cls.baseline = diagnostic.load_json(diagnostic.DEFAULT_BASELINE_REPORT)
        cls.policy = comparator.validate_policy(
            diagnostic.load_json(diagnostic.DEFAULT_POLICY), diagnostic.DEFAULT_POLICY)

    @unittest.skipUnless(HAS_FROZEN_ARTIFACTS, "frozen physical campaign artifacts are absent")
    def test_metric_implementation_exactly_reproduces_frozen_21_checks(self) -> None:
        result = diagnostic.metric_result(
            self.moving["samples"], self.reference["samples"], self.policy)
        self.assertEqual(result["passed_check_count"], 21)
        self.assertEqual(result["failed_check_count"], 0)
        self.assertEqual(result["checks"], self.baseline["checks"])
        for key in (
            "declared_sample_count", "common_sample_count", "mature_start_s",
            "mature_sample_count", "paired_selected_samples",
            "paired_selected_fraction", "maximum_consecutive_unpaired_samples",
            "final_sample_paired",
        ):
            self.assertEqual(result["coverage"][key], self.baseline["coverage"][key])

    def test_large_suffix_path_drift_fails_closed(self) -> None:
        policy, candidate, reference = synthetic_case()
        final = candidate[-1]
        final["core_centroid_m"]["x"] += 100000.0
        result = diagnostic.metric_result(candidate, reference, policy)
        failed = {check["name"] for check in result["checks"] if not check["passed"]}
        self.assertIn("path.maximum_m", failed)
        self.assertIn("path.final_m", failed)
        self.assertGreater(result["failed_check_count"], 0)

    def test_identical_synthetic_series_passes_all_metric_families(self) -> None:
        policy, candidate, reference = synthetic_case()
        result = diagnostic.metric_result(candidate, reference, policy)
        self.assertEqual(result["passed_check_count"], 21)
        self.assertEqual(result["failed_check_count"], 0)

    def test_report_contract_cannot_claim_formal_qualification(self) -> None:
        source = (ROOT / "scripts" / "qualification" / "nested_domains" / "compare_moving_nested_vulkan_continuation.py").read_text(
            encoding="utf-8")
        self.assertIn('"qualification_eligible": False', source)
        self.assertIn('"formal_qualification_comparison": False', source)

    def test_lifetime_report_contract_cannot_claim_formal_qualification(self) -> None:
        source = (ROOT / "scripts" / "qualification" / "nested_domains" / "compare_moving_nested_vulkan_lifetime.py").read_text(
            encoding="utf-8")
        self.assertIn('"qualification_eligible": False', source)
        self.assertIn('"formal_qualification_comparison": False', source)
        self.assertIn('"single_candidate_binary_config_identity": True', source)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Regression tests for the versioned CPU benchmark policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / "scripts" / "benchmarks" / "cpu" / "check_cpu_benchmark.py"
SPEC = importlib.util.spec_from_file_location("check_cpu_benchmark", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def report(step_ms: list[float], rss: list[float] | None = None) -> dict:
    rss = rss or [1000.0] * len(step_ms)
    return {
        "schema": checker.REPORT_SCHEMA,
        "host": {"system": "Darwin", "machine": "arm64", "logical_cpu_count": 16},
        "build": {"profile": "release", "compiler": {"version": "clang test"}},
        "run": {
            "config_sha256": "fixture",
            "trials": len(step_ms),
            "threads": 16,
            "environment": {"OMP_NUM_THREADS": "16"},
            "metric_scope": "model_step",
            "measured_steps": 100,
            "warmup_steps": 20,
        },
        "trials": [
            {
                "avg_step_ms": value,
                "sim_s_per_wall_s": 1000.0 / value,
                "peak_rss_bytes": rss_value,
            }
            for value, rss_value in zip(step_ms, rss)
        ],
    }


def policy() -> dict:
    return {
        "schema": checker.POLICY_SCHEMA,
        "policy_id": "test-v1",
        "minimum_trials": 7,
        "minimum_measured_steps": 100,
        "identity_fields": [
            "host.system", "host.machine", "host.logical_cpu_count",
            "build.profile", "build.compiler.version", "run.config_sha256",
            "run.threads", "run.environment", "run.metric_scope",
            "run.warmup_steps",
        ],
        "metrics": {
            "avg_step_ms": {
                "aggregate": "median", "direction": "lower_is_better",
                "max_relative_regression": 0.05, "baseline_mad_multiplier": 3.0,
            },
            "peak_rss_bytes": {
                "aggregate": "max", "direction": "lower_is_better",
                "max_relative_regression": 0.05, "absolute_allowance": 64.0,
            },
        },
    }


class CpuBenchmarkPolicyTests(unittest.TestCase):
    def test_matching_measurement_passes(self) -> None:
        baseline = report([98, 99, 100, 100, 101, 101, 102])
        current = report([99, 100, 101, 101, 102, 102, 103])
        result = checker.compare_reports(baseline, current, policy())
        self.assertEqual(result["status"], "pass")

    def test_material_slowdown_fails(self) -> None:
        baseline = report([98, 99, 100, 100, 101, 101, 102])
        current = report([118, 119, 120, 120, 121, 121, 122])
        result = checker.compare_reports(baseline, current, policy())
        self.assertEqual(result["status"], "regression")
        self.assertFalse(result["comparisons"][0]["passed"])

    def test_identity_mismatch_is_incompatible(self) -> None:
        baseline = report([100] * 7)
        current = report([100] * 7)
        current["run"]["threads"] = 8
        result = checker.compare_reports(baseline, current, policy())
        self.assertEqual(result["status"], "incompatible")
        self.assertEqual(result["identity_mismatches"][0]["field"], "run.threads")

    def test_short_report_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer than 7 trials"):
            checker.compare_reports(report([100] * 6), report([100] * 7), policy())


if __name__ == "__main__":
    unittest.main()

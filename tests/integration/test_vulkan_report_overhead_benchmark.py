#!/usr/bin/env python3
"""Regression tests for the physical Vulkan report-overhead benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "scripts" / "benchmarks" / "gpu" / "benchmark_vulkan_report_overhead.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_vulkan_report_overhead", BENCHMARK_PATH
)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)
sys.modules["benchmark_vulkan_report_overhead"] = benchmark

SCALAR_BENCHMARK_PATH = (
    ROOT / "scripts" / "benchmarks" / "gpu" / "benchmark_vulkan_scalar_attribution.py"
)
SCALAR_SPEC = importlib.util.spec_from_file_location(
    "benchmark_vulkan_scalar_attribution", SCALAR_BENCHMARK_PATH
)
assert SCALAR_SPEC is not None and SCALAR_SPEC.loader is not None
scalar_benchmark = importlib.util.module_from_spec(SCALAR_SPEC)
SCALAR_SPEC.loader.exec_module(scalar_benchmark)


class VulkanReportOverheadBenchmarkTests(unittest.TestCase):
    def test_config_pair_differs_only_by_report_cadence(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="nimbus_vulkan_report_config_"
        ) as temp:
            root = Path(temp)
            free = root / "free.yaml"
            report = root / "report.yaml"
            free.write_text(
                "schema: nimbus.v1\n"
                "diagnostics:\n"
                "  scientific_every_steps: 0 # disabled\n",
                encoding="utf-8",
            )
            report.write_text(
                "schema: nimbus.v1\n"
                "diagnostics:\n"
                "  scientific_every_steps: 1 # enabled\n",
                encoding="utf-8",
            )
            benchmark.require_config_pair_contract(free, report)
            report.write_text(
                report.read_text(encoding="utf-8") + "physics: changed\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                benchmark.require_config_pair_contract(free, report)

    def test_stage_telemetry_parser_and_global_recomposition(self) -> None:
        payload = (
            "[GPU STAGE TELEMETRY] stage_a "
            "command_buffers=2 dispatches=4 host_upload_bytes=10 "
            "host_download_bytes=11 device_upload_bytes=12 "
            "device_download_bytes=13 resident_hits=14 resident_misses=15 "
            "resident_upload_bytes_saved=16 "
            "writeonly_output_upload_bytes_saved=17 "
            "descriptor_cache_hits=18 descriptor_cache_misses=19\n"
            "[GPU STAGE TELEMETRY] stage_b "
            "command_buffers=3 dispatches=5 host_upload_bytes=20 "
            "host_download_bytes=21 device_upload_bytes=22 "
            "device_download_bytes=23 resident_hits=24 resident_misses=25 "
            "resident_upload_bytes_saved=26 "
            "writeonly_output_upload_bytes_saved=27 "
            "descriptor_cache_hits=28 descriptor_cache_misses=29\n"
        )
        stages = benchmark.parse_stage_telemetry(payload, "fixture")
        global_values = {
            key: sum(stage[key] for stage in stages.values())
            for key in benchmark.STAGE_TELEMETRY_KEYS
        }
        self.assertEqual(
            benchmark.reconcile_stage_telemetry(
                stages, global_values, "fixture"),
            {key: 0 for key in benchmark.STAGE_TELEMETRY_KEYS},
        )
        global_values["host_download_bytes"] += 4
        residual = benchmark.reconcile_stage_telemetry(
            stages, global_values, "fixture")
        self.assertEqual(residual["host_download_bytes"], 4)
        global_values["dispatches"] += 1
        with self.assertRaises(RuntimeError):
            benchmark.reconcile_stage_telemetry(
                stages, global_values, "fixture")

    def test_summary_separates_times_fractions_and_stage_counters(self) -> None:
        def trial(report_bearing: bool) -> dict[str, object]:
            command_buffers = 8 if report_bearing else 4
            dispatches = 16 if report_bearing else 60
            stage = "acoustic_fused" if report_bearing else "acoustic_batched"
            telemetry = {
                key: 0 for key in benchmark.TELEMETRY_KEYS
            }
            telemetry.update({
                "model_steps": 4,
                "command_buffers": command_buffers,
                "dispatches": dispatches,
            })
            stage_values = {
                key: 0 for key in benchmark.STAGE_TELEMETRY_KEYS
            }
            stage_values.update({
                "command_buffers": command_buffers,
                "dispatches": dispatches,
            })
            return {
                "measured_steps": 4,
                "avg_step_ms": 2.0 if report_bearing else 1.0,
                "process_wall_s": 1.0,
                "peak_rss_kib": 100,
                "telemetry": telemetry,
                "stage_telemetry": {stage: stage_values},
                "dynamics_perf_s": {
                    "dynamics_total_s":
                        0.008 if report_bearing else 0.004,
                    "dynamics_profiled_fraction":
                        0.9 if report_bearing else 0.8,
                },
            }

        summary = benchmark.summarize(
            [trial(False) for _ in range(3)],
            [trial(True) for _ in range(3)],
        )
        self.assertEqual(summary["measured_steps_per_trial"], 4)
        self.assertEqual(
            summary["step_time_statistics"]["paired_delta_ms"]["median"],
            1.0,
        )
        self.assertEqual(
            summary["step_time_statistics"]["paired_delta_ms"][
                "three_robust_sigma"
            ],
            0.0,
        )
        total = summary["dynamics_perf"]["dynamics_total_s"]
        self.assertEqual(total["unit"], "seconds_per_run")
        self.assertEqual(total["delta_ms_per_step"], 1.0)
        fraction = summary["dynamics_perf"]["dynamics_profiled_fraction"]
        self.assertEqual(fraction["unit"], "fraction")
        self.assertNotIn("delta_ms_per_step", fraction)
        batched = summary["stage_telemetry"]["acoustic_batched"]
        self.assertEqual(
            batched["dispatches"]["report_free_median_per_step"], 15.0
        )
        fused = summary["stage_telemetry"]["acoustic_fused"]
        self.assertEqual(
            fused["command_buffers"][
                "report_bearing_median_per_step"
            ],
            2.0,
        )


class VulkanScalarAttributionBenchmarkTests(unittest.TestCase):
    def test_speedup_gate_requires_effect_and_noise_clearance(self) -> None:
        accepted = scalar_benchmark.evaluate_speedup_gate(
            [9.0, 9.0, 9.0],
            [10.0, 10.0, 10.0],
            2.0,
        )
        self.assertTrue(accepted["passes_speedup_and_noise"])
        self.assertEqual(accepted["median_improvement_ms"], 1.0)
        self.assertEqual(accepted["minimum_improvement_ms"], 0.2)

        below_effect = scalar_benchmark.evaluate_speedup_gate(
            [9.9, 9.9, 9.9],
            [10.0, 10.0, 10.0],
            2.0,
        )
        self.assertFalse(below_effect["passes_speedup_and_noise"])

        below_noise = scalar_benchmark.evaluate_speedup_gate(
            [9.9, 9.0, 8.1],
            [10.0, 10.0, 10.0],
            2.0,
        )
        self.assertFalse(below_noise["passes_speedup_and_noise"])
        self.assertGreater(
            below_noise["three_robust_sigma_ms"],
            below_noise["median_improvement_ms"],
        )

    def test_speedup_gate_rejects_invalid_pairs(self) -> None:
        with self.assertRaises(RuntimeError):
            scalar_benchmark.evaluate_speedup_gate(
                [9.0, 9.0, 9.0],
                [10.0, 10.0],
                2.0,
            )
        with self.assertRaises(RuntimeError):
            scalar_benchmark.evaluate_speedup_gate(
                [9.0, float("nan"), 9.0],
                [10.0, 10.0, 10.0],
                2.0,
            )

    def test_attribution_overhead_gate_requires_control_normalized_win(
        self,
    ) -> None:
        accepted = scalar_benchmark.evaluate_attribution_overhead_gate(
            [4.0, 4.0, 4.0],
            [9.0, 9.0, 9.0],
            [4.0, 4.0, 4.0],
            [10.0, 10.0, 10.0],
            2.0,
        )
        self.assertTrue(accepted["passes_speedup_and_noise"])
        self.assertEqual(accepted["median_improvement_ms"], 1.0)
        self.assertEqual(accepted["minimum_improvement_ms"], 0.12)

        below_effect = scalar_benchmark.evaluate_attribution_overhead_gate(
            [4.0, 4.0, 4.0],
            [9.9, 9.9, 9.9],
            [4.0, 4.0, 4.0],
            [10.0, 10.0, 10.0],
            2.0,
        )
        self.assertFalse(below_effect["passes_speedup_and_noise"])

        below_noise = scalar_benchmark.evaluate_attribution_overhead_gate(
            [4.0, 4.0, 4.0],
            [9.9, 9.0, 8.1],
            [4.0, 4.0, 4.0],
            [10.0, 10.0, 10.0],
            2.0,
        )
        self.assertFalse(below_noise["passes_speedup_and_noise"])
        self.assertGreater(
            below_noise["three_robust_sigma_ms"],
            below_noise["median_improvement_ms"],
        )

    def test_attribution_overhead_gate_rejects_invalid_controls(self) -> None:
        with self.assertRaises(RuntimeError):
            scalar_benchmark.evaluate_attribution_overhead_gate(
                [4.0, 4.0],
                [9.0, 9.0, 9.0],
                [4.0, 4.0, 4.0],
                [10.0, 10.0, 10.0],
                2.0,
            )
        with self.assertRaises(RuntimeError):
            scalar_benchmark.evaluate_attribution_overhead_gate(
                [10.0, 10.0, 10.0],
                [11.0, 11.0, 11.0],
                [10.0, 10.0, 10.0],
                [9.0, 9.0, 9.0],
                2.0,
            )

    def test_gate_policy_matches_experiment_scope(self) -> None:
        passing = {"passes_speedup_and_noise": True}
        rejected = {"passes_speedup_and_noise": False}
        self.assertTrue(
            scalar_benchmark.passes_gate_policy(
                "whole-step", passing, rejected
            )
        )
        self.assertFalse(
            scalar_benchmark.passes_gate_policy(
                "report-attribution", passing, rejected
            )
        )
        self.assertTrue(
            scalar_benchmark.passes_gate_policy(
                "report-attribution", passing, passing
            )
        )
        with self.assertRaises(RuntimeError):
            scalar_benchmark.passes_gate_policy(
                "invalid", passing, passing
            )


if __name__ == "__main__":
    unittest.main()

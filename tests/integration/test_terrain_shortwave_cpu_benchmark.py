#!/usr/bin/env python3
"""Regression tests for the terrain-shortwave CPU benchmark harness."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PATH = ROOT / "scripts" / "benchmarks" / "cpu" / "benchmark_active_physics_allocations.py"
ACTIVE_SPEC = importlib.util.spec_from_file_location(
    "benchmark_active_physics_allocations", ACTIVE_PATH
)
assert ACTIVE_SPEC is not None and ACTIVE_SPEC.loader is not None
active_benchmark = importlib.util.module_from_spec(ACTIVE_SPEC)
ACTIVE_SPEC.loader.exec_module(active_benchmark)
sys.modules["benchmark_active_physics_allocations"] = active_benchmark

BENCHMARK_PATH = ROOT / "scripts" / "benchmarks" / "cpu" / "benchmark_terrain_shortwave_cpu.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_terrain_shortwave_cpu", BENCHMARK_PATH
)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)
sys.modules["benchmark_terrain_shortwave_cpu"] = benchmark

SCALING_PATH = (
    ROOT / "scripts" / "benchmarks" / "cpu" / "benchmark_terrain_shortwave_cpu_scaling.py"
)
SCALING_SPEC = importlib.util.spec_from_file_location(
    "benchmark_terrain_shortwave_cpu_scaling", SCALING_PATH
)
assert SCALING_SPEC is not None and SCALING_SPEC.loader is not None
scaling = importlib.util.module_from_spec(SCALING_SPEC)
SCALING_SPEC.loader.exec_module(scaling)


class TerrainShortwaveCpuBenchmarkTests(unittest.TestCase):
    def test_thread_counts_accept_ordered_unique_positive_values(self) -> None:
        self.assertEqual(benchmark.thread_counts("1,4,16"), (1, 4, 16))

    def test_thread_counts_reject_duplicates(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            benchmark.thread_counts("4,4")

    def test_thread_counts_reject_nonpositive_values(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            benchmark.thread_counts("1,0")

    def test_metric_reports_median_and_extent(self) -> None:
        self.assertEqual(
            benchmark.metric([3.0, 1.0, 2.0]),
            {"median": 2.0, "min": 1.0, "max": 3.0},
        )

    def test_parser_defaults_cover_one_and_four_workers(self) -> None:
        args = benchmark.parser().parse_args(["--binary", "nimbus"])
        self.assertEqual(args.thread_counts, (1, 4))
        self.assertEqual(args.duration, 10)
        self.assertEqual(args.trials, 5)

    def test_scaling_config_changes_only_grid_shape(self) -> None:
        base = "grid:\n  nx: 16\n  ny: 16\ntrigger:\n  nx: 99\n"
        self.assertEqual(
            scaling.scaled_config_text(base, 64),
            "grid:\n  nx: 64\n  ny: 64\ntrigger:\n  nx: 99\n",
        )

    def test_scaling_sizes_reject_tiny_domains(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            scaling.horizontal_sizes("4,16")

    def test_scaling_defaults_use_seven_paired_trials(self) -> None:
        args = scaling.parser().parse_args(["--binary", "nimbus"])
        self.assertEqual(args.trials, 7)
        self.assertEqual(args.thread_counts, (1, 4))
        self.assertEqual(args.horizontal_sizes, (16, 32, 64))

    def test_paired_scaling_accepts_material_low_noise_gain(self) -> None:
        evidence = scaling.paired_scaling_evidence(
            [100.0, 101.0, 99.0, 100.5, 99.5],
            [70.0, 71.0, 69.0, 70.5, 69.5],
        )
        self.assertEqual(evidence["classification"], "accelerates")
        self.assertEqual(evidence["improvement_ms"]["median"], 30.0)

    def test_paired_scaling_rejects_noisy_small_gain(self) -> None:
        evidence = scaling.paired_scaling_evidence(
            [100.0, 105.0, 95.0, 104.0, 96.0],
            [99.0, 103.0, 94.0, 106.0, 94.0],
        )
        self.assertEqual(evidence["classification"], "unresolved")

    def test_paired_scaling_detects_material_regression(self) -> None:
        evidence = scaling.paired_scaling_evidence(
            [100.0, 101.0, 99.0, 100.5, 99.5],
            [130.0, 131.0, 129.0, 130.5, 129.5],
        )
        self.assertEqual(evidence["classification"], "regresses")


if __name__ == "__main__":
    unittest.main()

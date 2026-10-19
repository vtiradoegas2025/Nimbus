#!/usr/bin/env python3
"""Regression tests for the exact-endpoint CPU active-physics benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "scripts" / "benchmarks" / "cpu" / "benchmark_active_physics_allocations.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_active_physics_allocations", BENCHMARK_PATH
)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)
sys.modules["benchmark_active_physics_allocations"] = benchmark


class ActivePhysicsBenchmarkTests(unittest.TestCase):
    def dynamics_line(self) -> str:
        metrics = {
            "dynamics_total_s": 10.0,
            "dynamics_core_s": 1.0,
            "dynamics_advection_s": 1.0,
            "dynamics_microphysics_transport_s": 0.1,
            "dynamics_microphysics_s": 1.0,
            # SCI-244 may extend this named family without breaking the
            # benchmark timing ABI.
            "dynamics_microphysics_input_validation_s": 0.2,
            "dynamics_turbulence_s": 2.0,
            "dynamics_turbulence_refresh_s": 0.8,
            "dynamics_turbulence_chaos_s": 0.1,
            "dynamics_turbulence_stability_s": 0.2,
            "dynamics_turbulence_apply_s": 0.8,
            "dynamics_turbulence_commit_s": 0.1,
            "dynamics_diffusion_s": 1.0,
            "dynamics_eos_s": 1.0,
            "dynamics_sponge_s": 1.0,
            "dynamics_final_boundary_s": 1.0,
            "dynamics_profiled_s": 9.0,
            "dynamics_residual_s": 1.0,
            "dynamics_profiled_fraction": 0.9,
            "dynamics_step_fraction": 0.8,
        }
        return "[DYNAMICS PERF] " + ", ".join(
            f"{key}={value}" for key, value in metrics.items()
        )

    def test_parser_retains_new_named_subphases(self) -> None:
        parsed = benchmark.parse_dynamics_perf(self.dynamics_line())
        self.assertEqual(parsed["microphysics_input_validation"], 0.2)
        self.assertEqual(parsed["turbulence"], 2.0)
        self.assertEqual(parsed["profiled_fraction"], 0.9)

    def test_parser_rejects_missing_required_owner(self) -> None:
        line = self.dynamics_line().replace(
            ", dynamics_final_boundary_s=1.0", ""
        )
        with self.assertRaisesRegex(ValueError, "missing dynamics stage"):
            benchmark.parse_dynamics_perf(line)

    def test_parser_rejects_nonfinite_extension(self) -> None:
        line = self.dynamics_line().replace(
            "dynamics_microphysics_input_validation_s=0.2",
            "dynamics_microphysics_input_validation_s=nan",
        )
        with self.assertRaisesRegex(ValueError, "invalid dynamics stage"):
            benchmark.parse_dynamics_perf(line)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Regression tests for the controlled CPU benchmark harness."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts" / "benchmarks" / "cpu" / "run_cpu_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_cpu_benchmark", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class CpuBenchmarkRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="nimbus benchmark native ")
        cls.addClassCleanup(temporary.cleanup)
        cls.fake_binary = Path(temporary.name) / (
            "fake_benchmark.exe" if sys.platform == "win32" else "fake_benchmark"
        )
        configured = os.environ.get("CXX")
        if configured:
            compiler = (
                [configured] if Path(configured).is_file() else shlex.split(configured)
            )
        else:
            executable = next(
                (path for name in ("c++", "clang++", "g++")
                 if (path := shutil.which(name))),
                None,
            )
            if executable is None:
                raise RuntimeError(
                    "CPU runner integration tests require a C++17 compiler; set CXX"
                )
            compiler = [executable]
        completed = subprocess.run(
            [
                *compiler, "-std=c++17",
                str(Path(__file__).with_name("cpu_benchmark_fixture.cpp")),
                "-o", str(cls.fake_binary),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Could not compile native CPU runner fixture:\n{completed.stdout}{completed.stderr}"
            )

    def test_nearest_rank_p95(self) -> None:
        self.assertEqual(runner.nearest_rank_p95([1.0, 2.0, 3.0]), 3.0)
        self.assertEqual(runner.nearest_rank_p95(list(range(1, 101))), 95)

    def test_summary_identity_validation(self) -> None:
        valid = {
            "measured_steps": 4,
            "step_wall_s": 2.0,
            "avg_step_ms": 500.0,
            "sim_s_per_wall_s": 3.0,
            "measured_simulated_s": 6.0,
        }
        self.assertIsNone(runner.validate_perf_summary(valid))

        bad_step = dict(valid, avg_step_ms=400.0)
        self.assertIn("avg_step_ms", runner.validate_perf_summary(bad_step) or "")

        bad_rate = dict(valid, sim_s_per_wall_s=4.0)
        self.assertIn("sim_s_per_wall_s", runner.validate_perf_summary(bad_rate) or "")

        nonfinite = dict(valid, step_wall_s=math.inf)
        self.assertIn("non-finite", runner.validate_perf_summary(nonfinite) or "")

        nonpositive = dict(
            valid, measured_simulated_s=0.0, sim_s_per_wall_s=0.0
        )
        self.assertIn("positive", runner.validate_perf_summary(nonpositive) or "")

    def test_tuning_environment_is_explicit_and_scoped(self) -> None:
        self.assertEqual(
            runner.tuning_environment("OMP_PROC_BIND=close"),
            ("OMP_PROC_BIND", "close"),
        )
        with self.assertRaises(Exception):
            runner.tuning_environment("NIMBUS_DEBUG_EXPORTS=1")
        with self.assertRaises(Exception):
            runner.tuning_environment("OMP_NUM_THREADS=64")

    def test_peak_rss_platform_units(self) -> None:
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(runner.peak_rss_bytes(4096), 4096)
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(runner.peak_rss_bytes(4096), 4096 * 1024)
        with mock.patch.object(sys, "platform", "win32"):
            self.assertEqual(runner.peak_rss_bytes(4096), 4096)

    def test_worker_uses_portable_child_runner(self) -> None:
        request = {
            "command": ["nimbus.exe"],
            "environment": {"OMP_NUM_THREADS": "2"},
            "clear_environment_prefixes": runner.CLEARED_ENVIRONMENT_PREFIXES,
            "timeout_s": 5,
        }
        summary = (
            "[PERF SUMMARY] measured_steps=4, warmup_steps=2, scope=test, "
            "step_wall_s=2, avg_step_ms=500, sim_s_per_wall_s=3, "
            "measured_simulated_s=6"
        )
        stdout = io.StringIO()
        with mock.patch.object(
            runner.sys, "stdin", io.StringIO(json.dumps(request))
        ), mock.patch.object(runner.sys, "stdout", stdout), mock.patch.object(
            runner, "run_child", return_value=(0, False, 2.5, 123456, summary, "")
        ) as run_child:
            self.assertEqual(runner.run_worker(), 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["peak_rss_bytes"], 123456)
        self.assertEqual(payload["summary"]["scope"], "test")
        run_child.assert_called_once()

    def test_windows_timeout_terminates_complete_process_tree(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        process.pid = 42
        process.wait.return_value = 0
        with mock.patch.object(runner.sys, "platform", "win32"), mock.patch.object(
            runner.subprocess, "run"
        ) as taskkill:
            runner.terminate_process_tree(process)
        taskkill.assert_called_once()
        self.assertEqual(taskkill.call_args.args[0], ["taskkill", "/PID", "42", "/T", "/F"])

    def test_memory_milestone_parser_preserves_phase_order(self) -> None:
        parsed = runner.parse_memory_milestones(
            "[MEMORY MILESTONE] phase=startup_ready, peak_rss_bytes=123\n"
            "[MEMORY MILESTONE] phase=first_dynamics, peak_rss_bytes=456\n",
            "",
        )
        self.assertEqual(
            parsed,
            [
                {"phase": "startup_ready", "peak_rss_bytes": 123},
                {"phase": "first_dynamics", "peak_rss_bytes": 456},
            ],
        )

    def test_memory_plan_parser_and_calibration(self) -> None:
        plan = runner.parse_memory_plan(
            "[MEMORY PLAN] cells=100, field_bytes=400, "
            "predicted_fields=75, fixed_overhead_bytes=2000, "
            "predicted_bytes=32000\n",
            "",
        )
        self.assertEqual(
            plan,
            {
                "cells": 100,
                "field_bytes": 400,
                "predicted_fields": 75,
                "fixed_overhead_bytes": 2000,
                "predicted_bytes": 32000,
            },
        )
        calibration = runner.derive_memory_calibration(
            plan,
            28000,
            [{"phase": "first_dynamics", "peak_rss_bytes": 24000}],
        )
        assert calibration is not None
        self.assertEqual(calibration["observed_peak_field_equivalents"], 70.0)
        self.assertEqual(calibration["plan_margin_bytes"], 4000)
        self.assertTrue(calibration["within_plan"])
        self.assertEqual(calibration["phases"][0]["field_equivalents"], 60.0)

    def test_main_clears_inherited_environment_and_records_explicit_tuning(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_benchmark_test_") as temp:
            temp_path = Path(temp)
            config = temp_path / "fixture.yaml"
            config.write_text("schema: fake\n", encoding="utf-8")
            environment = os.environ.copy()
            environment["NIMBUS_DEBUG_EXPORTS"] = "1"
            environment["OMP_THREAD_LIMIT"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--binary", str(self.fake_binary),
                    "--config", str(config),
                    "--trials", "1",
                    "--threads", "3",
                    "--env", "OMP_PROC_BIND=close",
                    "--extra-arg=--fixture=environment",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(
                report["run"]["environment"],
                {"OMP_NUM_THREADS": "3", "OMP_PROC_BIND": "close"},
            )
            self.assertEqual(report["run"]["metric_scope"], "test_scope")
            self.assertEqual(report["run"]["measured_simulated_s"], 6.0)
            self.assertEqual(
                report["trials"][0]["memory_milestones"],
                [{"phase": "startup_ready", "peak_rss_bytes": 123456}],
            )
            self.assertEqual(
                report["run"]["memory_plan"]["predicted_fields"], 40000
            )
            self.assertTrue(
                report["trials"][0]["memory_calibration"]["within_plan"]
            )
            self.assertGreater(report["trials"][0]["peak_rss_bytes"], 0)

    def test_main_rejects_heterogeneous_trials(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_benchmark_test_") as temp:
            temp_path = Path(temp)
            counter = temp_path / "counter"
            config = temp_path / "fixture.yaml"
            config.write_text("schema: fake\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--binary", str(self.fake_binary),
                    "--config", str(config),
                    "--trials", "2",
                    "--threads", "2",
                    "--extra-arg=--fixture=heterogeneous",
                    f"--extra-arg=--counter={counter}",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("refusing to aggregate heterogeneous trials", completed.stderr)
            self.assertEqual(counter.read_text(), "2")


if __name__ == "__main__":
    unittest.main()

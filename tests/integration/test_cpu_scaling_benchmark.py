#!/usr/bin/env python3
"""Regression tests for the CPU thread-frontier benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCALING_PATH = ROOT / "scripts" / "benchmarks" / "cpu" / "benchmark_cpu_scaling.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_cpu_scaling", SCALING_PATH
)
assert SPEC is not None and SPEC.loader is not None
scaling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scaling)


class CpuScalingBenchmarkTests(unittest.TestCase):
    def test_thread_parser_rejects_duplicates_and_nonpositive_values(self) -> None:
        self.assertEqual(scaling.thread_counts("4,1,2"), [1, 2, 4])
        with self.assertRaises(Exception):
            scaling.thread_counts("1,2,2")
        with self.assertRaises(Exception):
            scaling.thread_counts("0,1")

    def test_default_frontier_includes_heterogeneous_performance_cores(self) -> None:
        completed = subprocess.CompletedProcess(
            ["sysctl"], returncode=0, stdout="12\n", stderr=""
        )
        with (
            mock.patch.object(scaling.os, "cpu_count", return_value=16),
            mock.patch.object(scaling.platform, "system", return_value="Darwin"),
            mock.patch.object(scaling.subprocess, "run", return_value=completed),
        ):
            self.assertEqual(
                scaling.default_thread_counts(),
                [1, 2, 4, 8, 12, 16],
            )

    def test_pareto_frontier_rejects_time_and_memory_dominance(self) -> None:
        points = [
            {
                "threads": 1,
                "avg_step_ms": {"median": 10.0},
                "peak_rss_bytes": {"max": 100},
            },
            {
                "threads": 2,
                "avg_step_ms": {"median": 6.0},
                "peak_rss_bytes": {"max": 110},
            },
            {
                "threads": 4,
                "avg_step_ms": {"median": 7.0},
                "peak_rss_bytes": {"max": 120},
            },
        ]
        self.assertEqual(scaling.pareto_frontier(points), [1, 2])

    def test_end_to_end_frontier_uses_identical_workload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_scaling_test_") as temp:
            root = Path(temp)
            fake = root / "fake_cpu.py"
            config = root / "fixture.yaml"
            output = root / "scaling.json"
            config.write_text("schema: fake\n", encoding="utf-8")
            fake.write_text(
                "#!/usr/bin/env python3\n"
                + textwrap.dedent(
                    """
                    import os
                    from pathlib import Path
                    import sys

                    threads = int(os.environ["OMP_NUM_THREADS"])
                    for argument in sys.argv[1:]:
                        if argument.startswith("--checkpoint-dir="):
                            checkpoint_dir = Path(argument.split("=", 1)[1])
                            checkpoint_dir.mkdir(parents=True, exist_ok=True)
                            (checkpoint_dir / "checkpoint.nimbus").write_bytes(
                                b"thread-independent-state"
                            )
                    wall = 4.0 / threads
                    print(
                        "[PERF SUMMARY] measured_steps=4, warmup_steps=2, "
                        f"scope=test_scope, step_wall_s={wall}, "
                        f"avg_step_ms={1000.0 / threads}, "
                        f"sim_s_per_wall_s={threads}, "
                        "measured_simulated_s=4"
                    )
                    """
                ),
                encoding="utf-8",
            )
            fake.chmod(0o755)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCALING_PATH),
                    "--binary",
                    str(fake),
                    "--config",
                    str(config),
                    "--trials",
                    "2",
                    "--thread-counts",
                    "1,2,4",
                    "--endpoint-duration",
                    "4",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], scaling.REPORT_SCHEMA)
            self.assertEqual(report["frontier"]["recommended_threads"], 4)
            self.assertTrue(
                report["endpoint_equivalence"]["exact_across_thread_counts"]
            )
            self.assertEqual(
                len(report["endpoint_equivalence"]["results"]),
                3,
            )
            self.assertEqual(
                [point["speedup_over_one_thread"] for point in report["points"]],
                [1.0, 2.0, 4.0],
            )
            self.assertEqual(
                len({
                    source["run"]["config_sha256"]
                    for source in report["source_reports"]
                }),
                1,
            )

    def test_endpoint_mismatch_writes_evidence_and_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_scaling_mismatch_") as temp:
            root = Path(temp)
            fake = root / "fake_cpu.py"
            config = root / "fixture.yaml"
            output = root / "scaling.json"
            config.write_text("schema: fake\n", encoding="utf-8")
            fake.write_text(
                "#!/usr/bin/env python3\n"
                + textwrap.dedent(
                    """
                    import os
                    from pathlib import Path
                    import sys

                    threads = int(os.environ["OMP_NUM_THREADS"])
                    for argument in sys.argv[1:]:
                        if argument.startswith("--checkpoint-dir="):
                            checkpoint_dir = Path(argument.split("=", 1)[1])
                            checkpoint_dir.mkdir(parents=True, exist_ok=True)
                            (checkpoint_dir / "checkpoint.nimbus").write_text(
                                f"threads={threads}", encoding="utf-8"
                            )
                    print(
                        "[PERF SUMMARY] measured_steps=1, warmup_steps=0, "
                        "scope=test_scope, step_wall_s=1, avg_step_ms=1000, "
                        "sim_s_per_wall_s=1, measured_simulated_s=1"
                    )
                    """
                ),
                encoding="utf-8",
            )
            fake.chmod(0o755)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCALING_PATH),
                    "--binary",
                    str(fake),
                    "--config",
                    str(config),
                    "--trials",
                    "1",
                    "--thread-counts",
                    "1,2",
                    "--endpoint-duration",
                    "1",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(
                report["endpoint_equivalence"]["exact_across_thread_counts"]
            )
            self.assertIn("checkpoints differ", completed.stderr)


if __name__ == "__main__":
    unittest.main()

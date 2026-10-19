#!/usr/bin/env python3
"""Regression tests for the nested relocation benchmark harness."""

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
RUNNER_PATH = ROOT / "scripts" / "benchmarks" / "domain" / "benchmark_nested_relocation.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_nested_relocation", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


RECORD = (
    "[NESTED RELOCATION BENCHMARK] profile=flat_active, child_nx=16, "
    "child_ny=20, nz=4, "
    "warmup_moves=2, measured_moves=4, total_ms=8, avg_ms=2, "
    "heap_allocations=0, heap_requested_bytes=0, "
    "physics_field_allocations=0, physics_field_allocated_bytes=0, "
    "rollback_journal_bytes=4096, full_primary_owner_values=20000"
)
COMBINED_RECORD = RECORD.replace(
    "profile=flat_active", "profile=flat_active_milbrandt")


class NestedRelocationBenchmarkTests(unittest.TestCase):
    def test_parser_and_identity_validation(self) -> None:
        measurement = runner.parse_benchmark_output(RECORD)
        assert measurement is not None
        expected = {
            "profile": "flat_active",
            "child_nx": 16,
            "child_ny": 20,
            "nz": 4,
            "warmup_moves": 2,
            "measured_moves": 4,
        }
        self.assertIsNone(runner.validate_measurement(measurement, expected))
        self.assertEqual(measurement["rollback_journal_bytes"], 4096)

        inconsistent = dict(measurement, avg_ms=3.0)
        self.assertIn(
            "inconsistent",
            runner.validate_measurement(inconsistent, expected) or "",
        )
        allocated = dict(measurement, heap_allocations=1)
        self.assertIn(
            "allocated", runner.validate_measurement(allocated, expected) or ""
        )

    def test_peak_rss_platform_units(self) -> None:
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(runner.peak_rss_bytes(1234), 1234)
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(runner.peak_rss_bytes(1234), 1234 * 1024)

    def test_runner_emits_machine_readable_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_nested_benchmark_") as temp:
            temp_path = Path(temp)
            fake = temp_path / "fake_nested_benchmark.py"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                + textwrap.dedent(
                    f"""
                    import os

                    assert os.environ["OMP_NUM_THREADS"] == "3"
                    assert os.environ["NIMBUS_NESTED_BENCH_CHILD_NX"] == "16"
                    assert os.environ["NIMBUS_NESTED_BENCH_CHILD_NY"] == "20"
                    assert os.environ["NIMBUS_NESTED_BENCH_PROFILE"] == "flat_active_milbrandt"
                    print({COMBINED_RECORD!r})
                    """
                ),
                encoding="utf-8",
            )
            fake.chmod(0o755)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--binary", str(fake),
                    "--child-nx", "16",
                    "--child-ny", "20",
                    "--nz", "4",
                    "--warmup-moves", "2",
                    "--measured-moves", "4",
                    "--trials", "2",
                    "--threads", "3",
                    "--profile", "flat_active_milbrandt",
                    "--compiler", "fixture-cxx",
                    "--cxxflags", "-O3 -DNDEBUG",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(
                report["schema"], "nimbus.nested-relocation-benchmark.v1"
            )
            self.assertEqual(report["run"]["trials"], 2)
            self.assertEqual(report["artifact"]["compiler"], "fixture-cxx")
            self.assertEqual(report["aggregate"]["avg_ms"]["median"], 2.0)
            self.assertGreater(report["trials"][0]["peak_rss_bytes"], 0)


if __name__ == "__main__":
    unittest.main()

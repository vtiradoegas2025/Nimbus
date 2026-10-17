#!/usr/bin/env python3
"""Exercise ZFP benchmark report binding with synthetic execution artifacts."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/benchmarks/benchmark_zfp_field_parallel.py"
SPEC = importlib.util.spec_from_file_location("benchmark_zfp_field_parallel", RUNNER)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ZfpFieldParallelBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="nimbus_zfp_report_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "bin/nimbus_sim"
        self.binary.parent.mkdir()
        self.binary.write_bytes(b"#!/bin/sh\nexit 97\n")
        self.binary.chmod(0o755)
        self.fixture = self.root / runner.FIXTURE.relative_to(ROOT)
        self.fixture.parent.mkdir(parents=True)
        self.fixture.write_bytes(runner.FIXTURE.read_bytes())
        for relative in (
            "src/core/output/output_writer.cpp",
            "include/core/output/output_writer.hpp",
            "src/core/output/output_config.cpp",
            "include/core/output/output_config.hpp",
            "src/core/runtime/headless_runtime.cpp",
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"synthetic source identity: {relative}\n", encoding="utf-8")
        self.output = self.root / "report.json"
        self.parallel_mismatch = False
        self.serial_mismatch = False
        self.identity_mismatch: str | None = None
        self.calls: list[str] = []

    def synthetic_case(self, config: Path, output: Path) -> tuple[float, str]:
        text = config.read_text(encoding="utf-8")
        parallel = "  zfp_field_execution: field_parallel" in text
        policy = "parallel" if parallel else "serial"
        self.calls.append(policy)
        self.assertIn("  nx: 128", text)
        self.assertIn("  ny: 128", text)
        self.assertIn("  nz: 32", text)
        self.assertIn("  ztop: 16000", text)
        if parallel:
            self.assertIn("  zfp_field_workers: 4", text)
        else:
            self.assertIn("  zfp_field_execution: serial", text)
        for field in ("theta", "qv"):
            path = output / "step_000001" / f"{field}.zfp3d"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"synthetic opaque artifact: {field}\n".encode()
            if ((parallel and self.parallel_mismatch)
                    or (self.serial_mismatch and self.calls.count("serial") > 1)):
                payload += b"changed\n"
            path.write_bytes(payload)
        cpu = "synthetic CPU"
        codec = "synthetic codec version"
        if parallel and self.identity_mismatch == "cpu":
            cpu += " changed"
        if parallel and self.identity_mismatch == "codec":
            codec += " changed"
        log = (f"[HARDWARE] CPU: {cpu}\n"
               f"ZFP codec provenance: version={codec}, libzfp_execution_policy=serial\n")
        return (1.0 if parallel else 2.0), log

    def synthetic_git(self, command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        if command == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")
        if command == ["git", "status", "--porcelain=v1"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"benchmark contract test must not execute a process: {command}")

    def generate(self) -> dict:
        with (mock.patch.object(runner, "ROOT", self.root),
              mock.patch.object(runner, "BINARY", self.binary),
              mock.patch.object(runner, "FIXTURE", self.fixture),
              mock.patch.object(runner, "run_case", side_effect=self.synthetic_case),
              mock.patch.object(runner.subprocess, "run", side_effect=self.synthetic_git),
              mock.patch.object(runner.platform, "platform", return_value="synthetic platform"),
              mock.patch.object(sys, "argv", [str(RUNNER), "--repeats", "2",
                                               "--output", str(self.output)]),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(runner.main(), 0)
        return json.loads(self.output.read_text(encoding="utf-8"))

    def test_report_binds_execution_and_artifacts(self) -> None:
        evidence = self.generate()
        self.assertEqual(self.calls, ["serial", "parallel", "serial", "parallel"])
        self.assertEqual(evidence["schema"], "nimbus.benchmark.zfp-field-parallel")
        self.assertEqual(evidence["schema_version"], 1)
        for key in (
            "generated_at_utc", "binary_sha256", "fixture_sha256",
            "source_fingerprint_sha256", "worktree_status_sha256",
            "platform", "cpu", "libzfp_version", "claim_boundary",
        ):
            self.assertTrue(evidence.get(key), key)
        for key in ("binary_sha256", "fixture_sha256", "source_fingerprint_sha256"):
            self.assertRegex(evidence[key], r"^[0-9a-f]{64}$")
        self.assertEqual(evidence["binary_sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())
        self.assertEqual(evidence["fixture_sha256"], hashlib.sha256(self.fixture.read_bytes()).hexdigest())
        self.assertEqual(evidence["cpu"], "synthetic CPU")
        self.assertEqual(evidence["libzfp_version"], "synthetic codec version")
        self.assertEqual(evidence["serial_execution_policy"], "serial")
        self.assertEqual(evidence["parallel_execution_policy"], "field_parallel")
        self.assertEqual(evidence["serial_workers"], 1)
        self.assertGreater(evidence["parallel_workers"], 1)
        self.assertGreater(evidence["parallel_scratch_budget_mib"], 0)
        self.assertTrue(evidence["artifact_bytes_exact"])
        digests = evidence["artifact_sha256"]
        self.assertEqual(len(digests), evidence["artifact_count"])
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in digests.values()))
        self.assertEqual(digests, {
            f"step_000001/{field}.zfp3d": hashlib.sha256(
                f"synthetic opaque artifact: {field}\n".encode()).hexdigest()
            for field in ("theta", "qv")
        })
        self.assertIn("no universal throughput claim", evidence["claim_boundary"])
        self.assertIn("no change to the default serial qualified policy", evidence["claim_boundary"])

    def test_parallel_artifact_mismatch_prevents_report(self) -> None:
        self.parallel_mismatch = True
        with self.assertRaisesRegex(RuntimeError, "parallel artifacts differ"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_serial_repeat_mismatch_prevents_report(self) -> None:
        self.serial_mismatch = True
        with self.assertRaisesRegex(RuntimeError, "serial repeat is not deterministic"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_runtime_identity_mismatch_prevents_report(self) -> None:
        for identity, message in (("cpu", "different CPUs"),
                                  ("codec", "different libzfp versions")):
            with self.subTest(identity=identity):
                self.identity_mismatch = identity
                with self.assertRaisesRegex(RuntimeError, message):
                    self.generate()
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

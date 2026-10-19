#!/usr/bin/env python3
"""Regression tests for the predictive-ZFP working-set benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmarks" / "storage" / "benchmark_zfp_history_working_set.py"
SPEC = importlib.util.spec_from_file_location("zfp_history_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def write_history(root: Path, generation: str) -> None:
    (root / "nimbus_output.json").write_text(
        json.dumps({
            "schema": "nimbus.output",
            "publication_generation": generation,
            "publication_state": "ready",
        }),
        encoding="utf-8",
    )
    for step, flags in enumerate((0, 1, 3)):
        step_dir = root / f"step_{step:06d}"
        step_dir.mkdir()
        (step_dir / "manifest.json").write_text(
            json.dumps({
                "schema": "nimbus.step",
                "publication_generation": generation,
                "step_index": step,
            }),
            encoding="utf-8",
        )
        payload = bytearray(40)
        payload[:4] = b"ZFP3"
        payload[36] = flags
        payload[39] = step
        (step_dir / "theta.zfp3d").write_bytes(payload)


class ZfpHistoryWorkingSetBenchmarkTests(unittest.TestCase):
    def test_digest_excludes_only_run_unique_generation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_zfp_history_test_") as temp:
            root = Path(temp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            write_history(first, "0" * 32)
            write_history(second, "f" * 32)

            self.assertEqual(
                benchmark.history_digest(first),
                benchmark.history_digest(second),
            )

    def test_digest_detects_encoded_payload_change(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_zfp_history_test_") as temp:
            root = Path(temp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            write_history(first, "0" * 32)
            write_history(second, "0" * 32)
            payload = bytearray((second / "step_000002" / "theta.zfp3d").read_bytes())
            payload[39] ^= 1
            (second / "step_000002" / "theta.zfp3d").write_bytes(payload)

            first_digest = benchmark.history_digest(first)
            second_digest = benchmark.history_digest(second)
            self.assertNotEqual(
                first_digest["zfp_payload_sha256"],
                second_digest["zfp_payload_sha256"],
            )

    def test_digest_requires_predictive_discriminator(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_zfp_history_test_") as temp:
            root = Path(temp)
            write_history(root, "0" * 32)
            payload_path = root / "step_000002" / "theta.zfp3d"
            payload = bytearray(payload_path.read_bytes())
            payload[36] = 1
            payload_path.write_bytes(payload)
            with self.assertRaisesRegex(RuntimeError, "non-discriminating"):
                benchmark.history_digest(root)

    def test_retain_keyframe_fixture_selects_real_keyframe_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_zfp_fixture_test_") as temp:
            root = Path(temp)
            history = root / "history"
            history.mkdir()
            write_history(history, "0" * 32)
            destination = root / "retained" / "writer-keyframe.zfp3d"

            digest = benchmark.retain_keyframe_fixture(history, destination)

            expected = (history / "step_000000" / "theta.zfp3d").read_bytes()
            self.assertEqual(destination.read_bytes(), expected)
            self.assertEqual(digest, benchmark.hashlib.sha256(expected).hexdigest())

    def test_robust_stats_reports_median_and_mad(self) -> None:
        self.assertEqual(
            benchmark.robust_stats([1.0, 2.0, 100.0]),
            {
                "samples": [1.0, 2.0, 100.0],
                "median": 2.0,
                "median_absolute_deviation": 1.0,
                "three_mad": 3.0,
            },
        )

    def test_status_never_turns_memory_acceptance_into_throughput(self) -> None:
        self.assertEqual(
            benchmark.classify_status(False, 1_048_576, True),
            ("accepted_memory_only", True),
        )
        self.assertEqual(
            benchmark.classify_status(False, 1_048_576, False),
            ("rejected", False),
        )
        self.assertEqual(
            benchmark.classify_status(True, 0, False),
            ("accepted", True),
        )

    def test_lifecycle_only_acceptance_requires_explicit_gate(self) -> None:
        self.assertEqual(
            benchmark.classify_status(False, 0, False, True, True),
            ("accepted_lifecycle_only", True),
        )
        self.assertEqual(
            benchmark.classify_status(False, 0, False, True, False),
            ("rejected", False),
        )

    def test_workspace_and_phase_summaries_are_strictly_parsed(self) -> None:
        workspace = benchmark.WORKSPACE_SUMMARY.search(
            "[OUTPUT] ZFP codec workspace: initializations=1, "
            "bitstream_bindings=1, bitstream_capacity_bytes=542744"
        )
        self.assertIsNotNone(workspace)
        self.assertEqual(workspace.group("bindings"), "1")

        assembly = benchmark.ASSEMBLY_SUMMARY.search(
            "[OUTPUT] Snapshot assembly: frames=9, total_s=0.121, "
            "max_s=0.019, max_bytes=12582912"
        )
        self.assertIsNotNone(assembly)
        self.assertEqual(assembly.group("frames"), "9")
        self.assertEqual(assembly.group("bytes"), "12582912")

        phases = benchmark.PHASE_SUMMARY.search(
            "[OUTPUT] ZFP instrumented phases: control_s=7.4462e-05, "
            "codec_s=0.101231, container_io_s=0.0139268"
        )
        self.assertIsNotNone(phases)
        self.assertEqual(phases.group("codec"), "0.101231")
        provenance = benchmark.CODEC_PROVENANCE.search(
            "[OUTPUT] ZFP codec provenance: "
            "version=zfp version 1.0.1 (December 15, 2023), "
            "execution_policy=serial"
        )
        self.assertIsNotNone(provenance)
        self.assertEqual(
            provenance.group("version"),
            "zfp version 1.0.1 (December 15, 2023)",
        )

    def test_workspace_lifecycle_gate_requires_exact_work_and_phases(self) -> None:
        history = {
            "zfp_frame_types": {
                "keyframe": 2,
                "simple": 2,
                "predictive": 6,
            }
        }
        phases = {
            "control_s": 0.001,
            "codec_s": 0.1,
            "container_io_s": 0.01,
        }
        pair = {
            "optimized": {
                "history": history,
                "zfp_codec_workspace": {
                    "initializations": 1,
                    "bitstream_bindings": 1,
                    "bitstream_capacity_bytes": 1024,
                },
                "zfp_instrumented_phases": phases,
                "zfp_codec_provenance": {
                    "version": "1.0.1",
                    "execution_policy": "serial",
                },
            },
            "comparator": {
                "history": history,
                "zfp_codec_workspace": {
                    "initializations": 10,
                    "bitstream_bindings": 20,
                    "bitstream_capacity_bytes": 1024,
                },
                "zfp_instrumented_phases": phases,
                "zfp_codec_provenance": {
                    "version": "1.0.1",
                    "execution_policy": "serial",
                },
            },
        }
        self.assertTrue(benchmark.workspace_lifecycle_pair_pass(pair))
        pair["optimized"]["zfp_instrumented_phases"] = None
        self.assertFalse(benchmark.workspace_lifecycle_pair_pass(pair))


if __name__ == "__main__":
    unittest.main()

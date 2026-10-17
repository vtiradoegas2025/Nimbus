#!/usr/bin/env python3
"""Focused contract tests for the paired reference-frontier runner."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/qualification/nested_domains/run_moving_nested_reference_frontier.py"
SPEC = importlib.util.spec_from_file_location(
    "run_moving_nested_reference_frontier", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def checkpoint(root: Path, step: int, time_s: float,
               run_id: str = "root", parent: dict | None = None) -> Path:
    path = root / "checkpoints" / f"checkpoint_step_{step:09d}.nimbus"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"checkpoint:{step}".encode("ascii"))
    sidecar = {
        "schema": "nimbus.checkpoint.lifecycle",
        "schema_version": 1,
        "checkpoint_file": path.name,
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_fnv1a64": "0123456789abcdef",
        "completed_steps": step,
        "simulation_time_s": time_s,
        "run_id": run_id,
        "parent_run_id": "" if parent is None else parent["run_id"],
        "parent_checkpoint": "" if parent is None else parent["checkpoint"]["path"],
        "boundary": "completed_timestep",
        "portable_across_compute_memory_topology": True,
    }
    write_json(Path(str(path) + ".json"), sidecar)
    return path


class MovingNestedReferenceCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="nimbus_moving_reference_campaign_")
        self.root = Path(self.temporary.name)
        self.binary = self.root / "identity_only_executable"
        self.binary.write_text("#!/bin/sh\nexit 97\n", encoding="ascii")
        self.binary.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_committed_configs_freeze_sci325_and_one_common_signature(self) -> None:
        with mock.patch.object(runner.subprocess, "run",
                               side_effect=AssertionError("parser test must not execute a process")):
            inputs = runner.validate_inputs(
                self.binary, runner.MOVING_CONFIG, runner.REFERENCE_CONFIG,
                runner.POLICY, validate_binary=False)
        self.assertEqual(inputs["policy_status"], "ratified")
        self.assertEqual(inputs["policy"]["path"], str(runner.POLICY.resolve()))
        self.assertEqual(
            json.loads(runner.POLICY.read_text(encoding="utf-8"))["schema_version"],
            12)
        self.assertEqual(inputs["threshold_policy"]["sha256"],
                         runner.comparator.BLINDED_V2_POLICY_SHA256)
        self.assertEqual(inputs["prior_policy"]["sha256"],
                         runner.comparator.V11_POLICY_SHA256)
        self.assertEqual(inputs["failure_evidence"]["sha256"],
                         runner.comparator.PARENT_STENCIL_LIMIT_FAILURE_SHA256)
        self.assertEqual(inputs["duration_s"], 4500.0)
        self.assertEqual(inputs["sample_count"], 150)
        signature = inputs["numerical_signature"]
        self.assertEqual(
            signature["moving_derivation"]["effective_dt_large_s"], 0.75)
        self.assertEqual(
            signature["reference_derivation"]["effective_dt_large_s"], 0.75)
        self.assertEqual(signature["values"]["comparison_level.effective_dx_m"], 600.0)

    def test_moving_tuple_drift_is_rejected(self) -> None:
        candidate = self.root / "moving.yaml"
        text = runner.MOVING_CONFIG.read_text(encoding="utf-8")
        candidate.write_text(
            text.replace("    inner_margin_m: 15600\n",
                         "    inner_margin_m: 15000\n"), encoding="utf-8")
        with self.assertRaisesRegex(runner.CampaignError, "drifted from the SCI-325"):
            runner.exact_moving_tuple(candidate, runner.SCI325_TUPLE)

    def test_sealed_report_detects_mutation(self) -> None:
        report = runner.seal_report({
            "schema": runner.REPORT_SCHEMA,
            "status": "measurement_only",
            "qualification_eligible": False,
        })
        self.assertTrue(runner.verify_sealed_report(report))
        report["status"] = "pass"
        self.assertFalse(runner.verify_sealed_report(report))

    def test_checkpoint_cadence_and_hash_bindings_are_validated(self) -> None:
        checkpoint(self.root, 1500, 2250.0)
        checkpoint(self.root, 3000, 4500.0)
        records = runner.validate_checkpoints(self.root, 4500.0, 2250)
        self.assertEqual(
            [record["simulation_time_s"] for record in records],
            [2250.0, 4500.0])
        self.assertTrue(all(len(record["checkpoint"]["sha256"]) == 64
                            for record in records))
        Path(str(self.root / "checkpoints/checkpoint_step_000003000.nimbus") +
             ".json").unlink()
        with self.assertRaises(runner.CampaignError):
            runner.validate_checkpoints(self.root, 4500.0, 2250)

    def test_resumed_checkpoint_lineage_is_fail_closed(self) -> None:
        prefix = self.root / "prefix"
        split_path = checkpoint(prefix, 3000, 2250.0, run_id="continuous")
        split = runner.checkpoint_lifecycle(split_path)
        suffix = self.root / "suffix"
        checkpoint(suffix, 6000, 4500.0, run_id="resumed", parent=split)
        records = runner.validate_checkpoints(suffix, 4500.0, 2250, split)
        self.assertEqual(len(records), 1)
        sidecar = Path(str(records[0]["checkpoint"]["path"]) + ".json")
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        document["parent_run_id"] = "wrong"
        write_json(sidecar, document)
        with self.assertRaisesRegex(runner.CampaignError, "lineage"):
            runner.validate_checkpoints(suffix, 4500.0, 2250, split)

    def test_composition_uses_prefix_through_split_and_suffix_after(self) -> None:
        prefix = self.root / "prefix"
        suffix = self.root / "suffix"
        destination = self.root / "evidence"
        for run, label in ((prefix, "prefix"), (suffix, "suffix")):
            run.mkdir(parents=True)
            for name in ("nimbus_output.json", "qualification_input.yaml",
                         "qualification_binary.sha256", "qualification_run.log"):
                (run / name).write_text(label + "\n", encoding="utf-8")
            for time_s in (30.0, 60.0, 90.0):
                step = run / "qualification_diagnostics" / f"step_{int(time_s):09d}"
                write_json(step / "manifest.json", {
                    "simulation_time_s": time_s, "owner": label})
                tracker = run / "storm_track" / f"track_step_{int(time_s):09d}.json"
                write_json(tracker, {"simulation_time_s": time_s, "owner": label})
        runner.compose_resumed_run(
            prefix, suffix, destination, "oversized_reference", 60.0)
        owners = []
        for path in sorted((destination / "qualification_diagnostics").glob("step_*")):
            owners.append(json.loads((path / "manifest.json").read_text())["owner"])
        self.assertEqual(owners, ["prefix", "prefix", "suffix"])
        self.assertEqual(
            (destination / "qualification_input.yaml").read_text(), "suffix\n")

    def test_validate_only_does_not_start_a_simulation_or_claim_result(self) -> None:
        fake = self.root / "fake_nimbus"
        fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        os.chmod(fake, 0o755)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = runner.main(["--validate-only", "--binary", str(fake)])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "validated")
        self.assertEqual(report["policy_status"], "ratified")
        self.assertFalse(report["qualification_eligible"])
        self.assertTrue(runner.verify_sealed_report(report))

    def test_output_cadence_resolves_compact_v12_policy(self) -> None:
        run = self.root / "reference"
        run.mkdir()
        write_json(run / "nimbus_output.json", {
            "schema": "nimbus.output",
            "manifest_version": 6,
            "publication_state": "ready",
            "data_products": ["qualification_diagnostics"],
        })
        shutil.copyfile(runner.REFERENCE_CONFIG, run / "qualification_input.yaml")
        (run / "qualification_binary.sha256").write_text(
            runner.sha256(self.binary) + "\n", encoding="ascii")
        (run / "qualification_run.log").write_text("complete\n", encoding="utf-8")

        times = [30.0, 60.0]

        def expected_times(policy: dict) -> tuple[list[float], dict]:
            self.assertEqual(policy["schema_version"], 12)
            self.assertIn("expected_contract", policy)
            return times, policy["expected_contract"]

        with (mock.patch.object(runner.extractor, "expected_times",
                                side_effect=expected_times),
              mock.patch.object(runner.extractor, "step_publications",
                                return_value={time_s: {} for time_s in times}),
              mock.patch.object(runner.extractor, "tracker_publications",
                                return_value={time_s: {} for time_s in times}),
              mock.patch.object(runner.subprocess, "run",
                                side_effect=AssertionError("parser test must not execute a process"))):
            result = runner.validate_output_cadence(
                run, "oversized_reference", runner.REFERENCE_CONFIG,
                self.binary, runner.POLICY)

        self.assertEqual(result["diagnostic_samples"], 2)
        self.assertEqual(result["tracker_samples"], 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.qualification import qualify_flat_mixed_metal_rebind_v1 as gate


class MixedMetalRebindStageTests(unittest.TestCase):
    def test_exact_retained_bytes_and_resource_policy_are_separate(self) -> None:
        configs = gate.staged_configs()
        self.assertEqual(hashlib.sha256(configs["metal_qualified"].encode()).hexdigest(), gate.QUALIFIED_SHA)
        source = configs["metal_exploratory"]
        self.assertEqual(configs["metal_exploratory_memory_reject"].replace(
            "    memory_admission: reject\n", "    memory_admission: warn\n"), source)
        self.assertIn("  bottom: rigid_noslip\n  top: open\n", source)
        self.assertEqual(source.count("    east: rigid_slip\n"), 1)
        self.assertIn("    native_generation_required: false\n", source)

    def test_proposed_public_cadence_does_not_inherit_301(self) -> None:
        contracts = {row["name"]: row for row in gate.future_cli_contracts()}
        self.assertEqual(len(contracts), 16)
        self.assertTrue(contracts["baseline"]["accepted"])
        self.assertTrue(contracts["write_one"]["accepted"])
        self.assertFalse(contracts["write_301"]["accepted"])
        self.assertFalse(contracts["write_two"]["accepted"])
        self.assertTrue(all(row["proofState"] == "post_admission_not_executed" for row in contracts.values()))

    def test_stage_is_nonexecuting_and_preserves_existing_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate.subprocess, "run") as run:
            root = Path(temporary) / "stage"
            gate.stage(root)
            run.assert_not_called()
            document = json.loads((root / "stage.json").read_text())
            self.assertEqual(document["physicalExecution"], "not_run")
            self.assertFalse(document["wholeBackendReady"])
            self.assertEqual(len(document["fixtures"]), 14)
            with self.assertRaisesRegex(RuntimeError, "must be new"):
                gate.stage(root)

    def test_changed_fixtures_and_binary_refuse_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "stage"
            gate.stage(root)
            fake = Path(temporary) / "binary"
            fake.write_text("fake binary")
            with self.assertRaisesRegex(RuntimeError, "coordinated freeze"):
                gate.bind(root, fake, fake, fake, "0" * 64)
            target = root / "fixtures/metal_qualified.yaml"
            target.write_text(target.read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "fixture changed"):
                gate.bind(root, fake, fake, fake, gate.identity(fake)["sha256"])

    def test_base_guard_cannot_make_post_admission_negatives_green(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "stage"
            gate.stage(root)
            sealed = {"inputs": {"binary": {"path": "never executed"}}}
            rejected = subprocess.CompletedProcess([], 2, "[VALIDATE] qualified Metal execution is limited", "")
            with mock.patch.object(gate, "binding", return_value=sealed), \
                    mock.patch.object(gate.subprocess, "run", return_value=rejected) as run:
                with self.assertRaisesRegex(RuntimeError, "closed base guard"):
                    gate.public_scope(root)
                self.assertEqual(run.call_count, 1)
            document = json.loads((root / "public_scope/summary.json").read_text())
            self.assertEqual(document["status"], "running_or_red")
            self.assertEqual(document["cases"][0]["name"], "baseline")
            self.assertFalse(document["cases"][0]["passed"])

    def test_crash_zero_exit_and_wrong_diagnostic_do_not_prove_rejection(self) -> None:
        marker = "qualified mixed-open Metal NPY scope requires --write-every=1"
        for code, content in ((-11, marker), (0, marker), (2, "[VALIDATE] unrelated error"), (2, marker)):
            with self.subTest(code=code, content=content):
                outcome = gate.public_admission_result(subprocess.CompletedProcess([], code, content, ""), False, [marker])
                self.assertFalse(outcome["passed"])
        outcome = gate.public_admission_result(subprocess.CompletedProcess([], 2, "[VALIDATE] " + marker, ""), False, [marker])
        self.assertTrue(outcome["passed"])

    def test_declared_rss_cap_rejects_preserved_vulkan_peak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "run.log"
            log.write_text("peak_rss_bytes=204685312\n")
            cases = {"vulkan1": {"log": gate.identity(log)}}
            result = gate.observed_resource_gate(cases)
            self.assertEqual(result["status"], "red_rss_cap")
            self.assertFalse(result["thresholdRelaxed"])
            log.write_text("peak_rss_bytes=45842432\n")
            with self.assertRaisesRegex(RuntimeError, "log identity drifted"):
                gate.observed_resource_gate(cases)
            cases = {"metal1": {"log": gate.identity(log)}}
            self.assertEqual(gate.observed_resource_gate(cases)["status"], "bounded_resource_pass")
            log.write_text("missing peak\n")
            cases = {"metal1": {"log": gate.identity(log)}}
            with self.assertRaisesRegex(RuntimeError, "omitted peak-RSS"):
                gate.observed_resource_gate(cases)


if __name__ == "__main__":
    unittest.main()

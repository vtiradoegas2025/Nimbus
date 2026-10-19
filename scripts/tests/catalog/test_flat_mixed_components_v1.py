from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.qualification import qualify_flat_mixed_components_v1 as gate


class MixedComponentCollectorTests(unittest.TestCase):
    def test_seven_owner_cases_and_four_targets_are_separate_from_public_admission(self) -> None:
        self.assertEqual(len(gate.CASES), 7)
        self.assertEqual(len({row["target"] for row in gate.CASES}), 4)
        self.assertEqual({row["backend"] for row in gate.CASES}, {"metal", "vulkan", "cpu", "injected"})
        self.assertTrue(all("," not in row["filter"] for row in gate.CASES))
        self.assertEqual(next(row for row in gate.CASES if row["name"] == "open_top_mass_scalar")["filter"],
                         "[open-top][SCI-042]")

    def test_skipped_missing_and_crashed_native_matrix_cannot_pass(self) -> None:
        case = gate.CASES[0]
        self.assertTrue(gate.result_record(subprocess.CompletedProcess([], 0,
            "All tests passed (86963 assertions in 1 test case)", ""), case)["passed"])
        for code, text in ((0, "No tests ran"), (-11, "All tests passed (86963 assertions in 1 test case)"),
                           (0, "All tests passed (10 assertions in 1 test case)"),
                           (0, "All tests passed (86963 assertions in 2 test cases)")):
            with self.subTest(code=code, text=text):
                self.assertFalse(gate.result_record(subprocess.CompletedProcess([], code, text, ""), case)["passed"])

    def test_stage_runs_no_binary_and_preserves_roots_and_equations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate.subprocess, "run") as run:
            root = Path(temporary) / "stage"
            gate.stage(root)
            run.assert_not_called()
            document = json.loads((root / "stage.json").read_text())
            self.assertEqual(document["status"], "staged_not_executed")
            self.assertEqual(document["publicMetalAdmission"], "guarded")
            self.assertFalse(document["wholeBackendReady"])
            warn = (root / "fixtures/metal_exploratory.yaml").read_text()
            reject = (root / "fixtures/metal_exploratory_memory_reject.yaml").read_text()
            self.assertEqual(reject.replace("    memory_admission: reject\n", "    memory_admission: warn\n"), warn)
            with self.assertRaisesRegex(RuntimeError, "must be new"):
                gate.stage(root)

    def test_fixture_or_source_binary_drift_refuses_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "stage"
            gate.stage(root)
            binary = Path(temporary) / "simulator"
            binary.write_text("fake simulator not executed")
            with self.assertRaisesRegex(RuntimeError, "SHA differs"):
                gate.bind(root, binary, "0" * 64)
            fixture = root / "fixtures/metal_exploratory.yaml"
            fixture.write_text(fixture.read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "fixture drifted"):
                gate.bind(root, binary, gate.identity(binary)["sha256"])


if __name__ == "__main__":
    unittest.main()

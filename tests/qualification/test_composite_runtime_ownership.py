from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualification/provenance/qualify_composite_runtime_ownership.py"
SPEC = importlib.util.spec_from_file_location("composite_ownership", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPOSITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPOSITE)


class CompositeRuntimeOwnershipTests(unittest.TestCase):
    def test_exact_suite_summary_is_accepted(self) -> None:
        output = (
            "All tests passed (3 assertions in 1 test case)\n"
            "All tests passed (7 assertions in 2 test cases)\n"
            "[required] PASS\n"
        )
        self.assertEqual(
            COMPOSITE.validate_suite_output(
                "synthetic", output, ((3, 1), (7, 2)), ("[required] PASS",)
            ),
            (10, 3),
        )

    def test_summary_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "summary drifted"):
            COMPOSITE.validate_suite_output(
                "synthetic", "All tests passed (4 assertions in 1 test case)\n", ((3, 1),), ()
            )

    def test_missing_marker_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing required marker"):
            COMPOSITE.validate_suite_output("synthetic", "", (), ("PASS",))

    def test_gpu_skip_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "skipped or failed"):
            COMPOSITE.validate_suite_output("synthetic", "GPU unavailable - skipping\n", (), ())

    def test_current_evidence_inputs_bind_the_binary(self) -> None:
        binary = ROOT / "bin/nimbus_sim"
        records = COMPOSITE.validate_inputs(COMPOSITE.sha256_file(binary), COMPOSITE.input_paths())
        self.assertEqual(set(records), set(COMPOSITE.input_paths()))

    def test_current_composite_evidence_verifies(self) -> None:
        binary = ROOT / "bin/nimbus_sim"
        evidence = json.loads(
            (ROOT / "reports/pre_beta/composite_runtime_ownership_v1.json").read_text()
        )
        COMPOSITE.verify_evidence(evidence, COMPOSITE.sha256_file(binary))

    def test_mutated_receipt_fails_closed(self) -> None:
        binary = ROOT / "bin/nimbus_sim"
        evidence = json.loads(
            (ROOT / "reports/pre_beta/composite_runtime_ownership_v1.json").read_text()
        )
        evidence["receipts"][0]["skipped"] = 1
        with self.assertRaisesRegex(RuntimeError, "receipt drifted"):
            COMPOSITE.verify_evidence(evidence, COMPOSITE.sha256_file(binary))


if __name__ == "__main__":
    unittest.main()

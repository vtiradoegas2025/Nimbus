from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualification/provenance/qualify_backend_io_ownership.py"
SPEC = importlib.util.spec_from_file_location("backend_io", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BACKEND_IO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND_IO)


class BackendIoOwnershipTests(unittest.TestCase):
    def test_exact_output_contract(self) -> None:
        output = "All tests passed (7 assertions in 2 test cases)\n[io] PASS\n"
        self.assertEqual(BACKEND_IO.validate_output("x", output, ((7, 2),), ("[io] PASS",)), (7, 2))

    def test_skip_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "skipped or failed"):
            BACKEND_IO.validate_output("x", "skipping live SHM", (), ())

    def test_current_anchors_bind_the_binary(self) -> None:
        binary = ROOT / "bin/nimbus_sim"
        records = BACKEND_IO.validate_anchors(BACKEND_IO.sha256_file(binary))
        self.assertEqual(set(records), set(BACKEND_IO.anchor_paths()))

    def test_current_evidence_verifies(self) -> None:
        binary = ROOT / "bin/nimbus_sim"
        evidence = json.loads((ROOT / "reports/pre_beta/backend_io_ownership_v1.json").read_text())
        BACKEND_IO.verify_evidence(evidence, BACKEND_IO.sha256_file(binary))

    def test_mutated_receipt_fails_closed(self) -> None:
        binary = ROOT / "bin/nimbus_sim"
        evidence = json.loads((ROOT / "reports/pre_beta/backend_io_ownership_v1.json").read_text())
        evidence["receipts"][0]["skipped"] = 1
        with self.assertRaisesRegex(RuntimeError, "receipt drifted"):
            BACKEND_IO.verify_evidence(evidence, BACKEND_IO.sha256_file(binary))


if __name__ == "__main__":
    unittest.main()

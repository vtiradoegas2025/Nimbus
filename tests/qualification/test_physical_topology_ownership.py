from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualification/provenance/qualify_physical_topology_ownership.py"
SPEC = importlib.util.spec_from_file_location("physical_topology", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TOPOLOGY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOPOLOGY)


class PhysicalTopologyOwnershipTests(unittest.TestCase):
    def test_exact_output_contract(self) -> None:
        output = "All tests passed (7 assertions in 2 test cases)\n[physical] PASS\n"
        self.assertEqual(TOPOLOGY.validate_output("x", output, ((7, 2),), ("[physical] PASS",)), (7, 2))

    def test_skip_and_fallback_fail_closed(self) -> None:
        for output in ("GPU unavailable - skipping", "fallback=on"):
            with self.subTest(output=output):
                with self.assertRaisesRegex(RuntimeError, "skipped, failed, or fallback"):
                    TOPOLOGY.validate_output("x", output, (), ())

    def test_current_anchors_bind_the_binary(self) -> None:
        binary = ROOT / "bin/nimbus_sim"
        records = TOPOLOGY.validate_anchors(TOPOLOGY.sha256_file(binary))
        self.assertEqual(set(records), set(TOPOLOGY.anchor_paths()))

    def test_current_evidence_verifies(self) -> None:
        binary = ROOT / "bin/nimbus_sim"
        evidence = json.loads(
            (ROOT / "reports/pre_beta/physical_topology_ownership_v1.json").read_text()
        )
        TOPOLOGY.verify_evidence(evidence, TOPOLOGY.sha256_file(binary))

    def test_mutated_receipt_fails_closed(self) -> None:
        binary = ROOT / "bin/nimbus_sim"
        evidence = json.loads(
            (ROOT / "reports/pre_beta/physical_topology_ownership_v1.json").read_text()
        )
        evidence["receipts"][0]["fallbackUsed"] = True
        with self.assertRaisesRegex(RuntimeError, "receipt drifted"):
            TOPOLOGY.verify_evidence(evidence, TOPOLOGY.sha256_file(binary))


if __name__ == "__main__":
    unittest.main()

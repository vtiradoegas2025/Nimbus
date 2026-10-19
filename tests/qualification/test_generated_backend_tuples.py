from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/tests/catalog/test_generated_backend_tuples.py"
SPEC = importlib.util.spec_from_file_location("generated_backend_tuples", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TUPLES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TUPLES)


class GeneratedBackendTupleTests(unittest.TestCase):
    def test_current_catalog_contract(self) -> None:
        evidence = json.loads(
            (ROOT / "reports/pre_beta/generated_backend_tuple_evidence_v1.json").read_text()
        )
        TUPLES.validate_catalog(evidence["catalog"])

    def test_current_evidence_verifies(self) -> None:
        evidence = json.loads(
            (ROOT / "reports/pre_beta/generated_backend_tuple_evidence_v1.json").read_text()
        )
        TUPLES.verify_evidence(evidence, TUPLES.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_partition_mutation_fails_closed(self) -> None:
        evidence = json.loads(
            (ROOT / "reports/pre_beta/generated_backend_tuple_evidence_v1.json").read_text()
        )
        evidence["catalog"]["partitions"]["collocated.cpu"] -= 1
        with self.assertRaisesRegex(RuntimeError, "partitions drifted"):
            TUPLES.verify_evidence(evidence, TUPLES.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_tuple_hash_mutation_fails_closed(self) -> None:
        evidence = json.loads(
            (ROOT / "reports/pre_beta/generated_backend_tuple_evidence_v1.json").read_text()
        )
        evidence["catalog"]["acceptedTupleSha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "identity drifted"):
            TUPLES.verify_evidence(evidence, TUPLES.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_unsupported_owner_fails_closed(self) -> None:
        evidence = json.loads(
            (ROOT / "reports/pre_beta/generated_backend_tuple_evidence_v1.json").read_text()
        )
        evidence["catalog"]["ownershipGroups"][0]["signature"] += "|x=unsupported"
        with self.assertRaisesRegex(RuntimeError, "group identity drifted"):
            TUPLES.verify_evidence(evidence, TUPLES.sha256_file(ROOT / "bin/nimbus_sim"))


if __name__ == "__main__":
    unittest.main()

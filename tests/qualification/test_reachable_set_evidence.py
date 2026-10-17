from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "scripts/qualification/provenance/"
    "generate_reachable_set_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("reachable_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REACHABLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REACHABLE)


class ReachableSetEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="nimbus-reachable-evidence-", dir=ROOT,
        )
        cls.output = Path(cls.temporary.name) / "reachable.json"
        cls.evidence = REACHABLE.build_evidence(
            root=ROOT,
            binary=ROOT / "bin/nimbus_sim",
            cross_product_binary=ROOT / "bin/test_core_scheme_catalog",
            authority=(
                ROOT /
                "tests/fixtures/reference/catalog/authority_v1.json"
            ),
            output=cls.output,
        )
        REACHABLE.write_atomic(
            cls.output,
            (json.dumps(cls.evidence, indent=2, sort_keys=True) + "\n").encode(),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_generated_evidence_verifies_and_is_explicitly_bounded(self) -> None:
        REACHABLE.verify_evidence(self.evidence, ROOT)
        authority = json.loads((ROOT / "tests/fixtures/reference/catalog/authority_v1.json").read_text())
        self.assertEqual(
            self.evidence["catalog"]["readinessFingerprintSha256"],
            authority["readinessFingerprintSha256"],
        )
        self.assertEqual(self.evidence["totalEvaluated"], 1701312)
        collocated = next(
            matrix for matrix in self.evidence["matrices"]
            if matrix["id"] == "collocated_physics"
        )
        self.assertEqual(collocated["evaluated"], 38880)
        self.assertEqual(collocated["accepted"], 5640)
        self.assertEqual(collocated["ownershipBound"], collocated["accepted"])
        cgrid = next(
            matrix for matrix in self.evidence["matrices"]
            if matrix["id"] == "cgrid_engine"
        )
        self.assertEqual(cgrid["accepted"], 359424)
        self.assertEqual(cgrid["ownershipBound"], cgrid["accepted"])
        self.assertFalse(
            self.evidence["scope"]["continuousParametersEnumerated"]
        )
        self.assertFalse(
            self.evidence["scope"]["runtimeQualificationUniversal"]
        )

    def test_matrix_or_scope_mutation_breaks_seal(self) -> None:
        mutated = json.loads(json.dumps(self.evidence))
        mutated["matrices"][0]["evaluated"] += 1
        with self.assertRaisesRegex(REACHABLE.EvidenceError, "seal"):
            REACHABLE.verify_evidence(mutated, ROOT)

    def test_resealed_ownership_count_mutation_fails_closed(self) -> None:
        mutated = json.loads(json.dumps(self.evidence))
        mutated["matrices"][2]["ownershipBound"] -= 1
        mutated.pop("integritySealSha256")
        mutated["integritySealSha256"] = REACHABLE.hashlib.sha256(
            REACHABLE.canonical_bytes(mutated)
        ).hexdigest()
        with self.assertRaisesRegex(REACHABLE.EvidenceError, "matrix contract"):
            REACHABLE.verify_evidence(mutated, ROOT)

    def test_resealed_skip_or_fallback_fails_closed(self) -> None:
        for key, value in (("skipped", 1), ("fallbackUsed", True)):
            mutated = json.loads(json.dumps(self.evidence))
            mutated["receipt"][key] = value
            mutated.pop("integritySealSha256")
            mutated["integritySealSha256"] = REACHABLE.hashlib.sha256(
                REACHABLE.canonical_bytes(mutated)
            ).hexdigest()
            with self.assertRaisesRegex(REACHABLE.EvidenceError, "clean pass"):
                REACHABLE.verify_evidence(mutated, ROOT)

    def test_resealed_path_escape_fails_closed(self) -> None:
        mutated = json.loads(json.dumps(self.evidence))
        mutated["crossProductSource"]["path"] = "../test_scheme_catalog.cpp"
        mutated.pop("integritySealSha256")
        mutated["integritySealSha256"] = REACHABLE.hashlib.sha256(
            REACHABLE.canonical_bytes(mutated)
        ).hexdigest()
        with self.assertRaisesRegex(REACHABLE.EvidenceError, "escapes"):
            REACHABLE.verify_evidence(mutated, ROOT)


if __name__ == "__main__":
    unittest.main()

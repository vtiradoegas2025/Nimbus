"""Prospective v3 integration; all workload/limit artifacts are synthetic."""

import copy
import hashlib
import unittest
from unittest import mock

from scripts.qualification import validate_c6_threshold_candidate_v2 as v2
from scripts.qualification import validate_c6_threshold_candidate_v3 as v3
from scripts.tests.performance.test_c6_threshold_candidate_v2 import write_json
if __package__:
    from .test_c6_paired_measurements import PairedFixture
else:
    from test_c6_paired_measurements import PairedFixture


class C6V3ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.fixture = PairedFixture()
        self.legacy = self.fixture.legacy
        self.policy = self.legacy.policy
        self.policy["schema"] = v3.POLICY_SCHEMA
        for index in range(len(self.policy["workloadClasses"])):
            receipt = self.legacy.read_receipt(index)
            receipt["schema"] = v3.RECEIPT_SCHEMA
            self.legacy.reseal_receipt(receipt, index)
        self.activate()

    def tearDown(self):
        self.fixture.close()

    def activate(self):
        unsigned = copy.deepcopy(self.policy)
        unsigned["activationReview"] = None
        review = {"schema": v3.REVIEW_SCHEMA, "decision": "approved", "reviewer": "synthetic-reviewer",
            "candidateUnsignedSha256": hashlib.sha256(v3.release_manifest.canonical_bytes(unsigned)).hexdigest(),
            "binarySha256": self.policy["releaseBinary"]["sha256"], "sourceInventorySha256": self.legacy.source_sha}
        self.policy["activationReview"] = write_json(self.legacy.root, "activation-v3.json", review)

    def validate(self):
        return v3._validate_policy(self.policy, self.legacy.root, self.legacy.binary, source_seal=self.legacy.seal)

    def test_v3_activates_only_complete_synthetic_workloads_and_limits(self):
        result = self.validate()
        self.assertTrue(result["releaseReady"])
        self.assertEqual(len(result["validatedWorkloads"]), 10)

    def test_v3_complete_row_binds_and_returns_computed_paired_proof(self):
        receipt = self.legacy.read_receipt()
        receipt["repairClaim"] = self.fixture.claim
        self.legacy.reseal_receipt(receipt)
        self.activate()
        with mock.patch.object(v3, "canonical_source_seal", return_value=self.legacy.seal):
            result = v3.validate_policy(self.policy, self.legacy.root, self.legacy.binary)
        proof = result["validatedWorkloads"]["cpu_milbrandt_24x24x20"][0]["pairedRepairProof"]
        self.assertEqual(proof["medianGainPercent"], 50)
        self.assertEqual(proof["pairs"], 7)

    def test_v2_receipts_and_detached_gain_arrays_cannot_migrate_implicitly(self):
        receipt = self.legacy.read_receipt()
        receipt["schema"] = v2.RECEIPT_SCHEMA
        self.legacy.reseal_receipt(receipt)
        self.activate()
        with self.assertRaisesRegex(v3.C6V3PolicyError, "workload receipt"):
            self.validate()
        receipt["schema"] = v3.RECEIPT_SCHEMA
        receipt["repairClaim"] = {"kind": "paired_time_gain", "pairedGainPercent": [100.0] * 7}
        self.legacy.reseal_receipt(receipt)
        self.activate()
        with self.assertRaisesRegex(v3.C6V3PolicyError, "v3 schema"):
            self.validate()

    def test_unreviewed_v3_activation_fails_and_v2_contract_is_unchanged(self):
        self.policy["activationReview"]["sha256"] = "e" * 64
        with self.assertRaisesRegex(v3.C6V3PolicyError, "stale"):
            self.validate()
        legacy = PairedFixture()
        try:
            self.assertTrue(legacy.legacy.validate()["releaseReady"])
            v2.validate_repair_claim({"kind": "paired_time_gain", "pairedGainPercent": [100.0] * 7}, legacy.policy)
        finally:
            legacy.close()


if __name__ == "__main__":
    unittest.main()

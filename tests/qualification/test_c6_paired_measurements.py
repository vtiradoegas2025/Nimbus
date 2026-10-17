"""Synthetic artifact mutations for C6's prospective paired-proof contract."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

from scripts.qualification import c6_paired_measurements as paired
from scripts.qualification import collect_c6_evidence as evidence
from scripts.qualification import validate_c6_threshold_candidate_v2 as common
from scripts.tests.performance import test_c6_threshold_candidate_v2 as legacy_tests

write_json = legacy_tests.write_json


class PairedFixture:
    def __init__(self):
        self.legacy = legacy_tests.C6V2ThresholdTests("test_actual_repository_policy_is_draft")
        self.legacy.setUp()
        self.root = self.legacy.root
        self.policy = self.legacy.policy
        self.receipt = self.legacy.read_receipt()
        self.profile = self.receipt["profiles"][0]
        self.claim = {"schema": paired.CLAIM_SCHEMA, "kind": "paired_time_gain",
            "profiles": [{"mode": self.profile["mode"], "pairs": []}]}
        for index, trial in enumerate(self.profile["trials"], start=1):
            record = json.loads((self.root / trial["measurementArtifact"]["path"]).read_text())
            baseline = record["trial"]
            baseline["avgStepMs"] *= 2
            baseline["stepWallSeconds"] *= 2
            baseline["science"] = self.legacy.science(f"baseline/trial{index}/science")
            baseline["rawLog"] = self.bytes(f"baseline/trial{index}/raw.log", b"synthetic baseline log")
            baseline["rawArtifacts"] = {}
            for snapshot in (record["snapshotBefore"], record["snapshotAfter"]):
                snapshot["binary"]["sha256"] = "b" * 64
                snapshot["binary"]["bytes"] = 42
            baseline["binarySha256"] = "b" * 64
            link = self.write_measurement(record, f"baseline/trial{index}/measurement.json")
            self.claim["profiles"][0]["pairs"].append({"baseline": link,
                "candidate": copy.deepcopy(trial["measurementArtifact"])})

    def bytes(self, relative, payload):
        from scripts.tests.performance.test_c6_threshold_candidate_v2 import write_bytes
        return write_bytes(self.root, relative, payload)

    def write_measurement(self, record, relative):
        trial = record["trial"]
        identity = evidence.trial_identity(record["snapshotBefore"], trial["mode"], trial["grid"])
        trial["identityBefore"] = copy.deepcopy(identity)
        trial["identityAfter"] = copy.deepcopy(identity)
        for key, value in identity.items():
            trial[key] = copy.deepcopy(value)
        prefix = str(Path(relative).parent)
        trial["validation"] = {"violations": 0, "exitStatus": 0, "skipped": 0,
            "trialIndex": trial["index"], **{key: trial[key] for key in (
                "binarySha256", "sourceInventorySha256", "configSha256")}}
        trial["validationArtifact"] = write_json(self.root, prefix + "/validation.json", trial["validation"])
        process = {"schema": evidence.PROCESS_SCHEMA, "trialIndex": trial["index"],
            "exitStatus": 0, "timedOut": False, "fallbackUsed": False, "identity": identity,
            "measurementWindow": copy.deepcopy(self.profile["measurementWindow"]),
            "measurements": {key: trial.get(key) for key in evidence.MEASURED_FIELDS},
            "rawLog": trial["rawLog"]}
        trial["rawArtifacts"]["process"] = write_json(self.root, prefix + "/process.json", process)
        return write_json(self.root, relative, record)

    def rewrite_baseline(self, index, mutate):
        pair = self.claim["profiles"][0]["pairs"][index]
        record = json.loads((self.root / pair["baseline"]["path"]).read_text())
        mutate(record)
        pair["baseline"] = self.write_measurement(record, pair["baseline"]["path"])

    def validate(self):
        return paired.validate_claim(self.claim, [self.profile], self.policy, self.root)

    def close(self):
        self.legacy.tearDown()


class C6PairedMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = PairedFixture()

    def tearDown(self):
        self.fixture.close()

    def test_gain_is_computed_from_paired_artifacts_and_historical_binary_is_not_rebound(self):
        result = self.fixture.validate()
        proof = result["profiles"]["cpu"]
        self.assertEqual(proof["pairs"], 7)
        self.assertEqual(proof["pairedGainPercent"], [50.0] * 7)
        self.assertEqual(proof["medianGainPercent"], 50)
        self.assertEqual(proof["baselineIdentity"]["binarySha256"], "b" * 64)
        self.assertTrue(result["historicalBaselineSourceNotReboundCurrent"])
        self.assertFalse(result["memoryProofComplete"])

    def test_detached_percent_arrays_and_unmeasured_memory_counts_are_rejected(self):
        self.fixture.claim["pairedGainPercent"] = [100.0] * 7
        with self.assertRaisesRegex(common.C6V2PolicyError, "detached"):
            self.fixture.validate()
        self.fixture.claim = {"schema": paired.CLAIM_SCHEMA, "kind": "live_memory_gain",
            "fullFieldBytes": 4096, "peakReductionBytes": 4096, "liveFullFieldsRemoved": 1}
        with self.assertRaisesRegex(common.C6V2PolicyError, "Engineering incomplete"):
            self.fixture.validate()

    def test_missing_duplicate_and_swapped_pairing_rejects(self):
        original = copy.deepcopy(self.fixture.claim)
        for mutation in ("missing", "baseline_duplicate", "candidate_duplicate", "swap"):
            self.fixture.claim = copy.deepcopy(original)
            pairs = self.fixture.claim["profiles"][0]["pairs"]
            if mutation == "missing":
                pairs.pop()
            elif mutation == "swap":
                pairs[0], pairs[1] = pairs[1], pairs[0]
            else:
                role = "baseline" if mutation == "baseline_duplicate" else "candidate"
                pairs[1][role] = copy.deepcopy(pairs[0][role])
            with self.subTest(mutation=mutation), self.assertRaises(common.C6V2PolicyError):
                self.fixture.validate()

    def test_positive_finite_measurements_required_even_when_artifacts_are_resealed(self):
        for value in (0, -1, True, float("nan"), float("inf")):
            self.fixture.rewrite_baseline(0, lambda record: record["trial"].update(avgStepMs=value))
            with self.subTest(value=value), self.assertRaisesRegex(common.C6V2PolicyError, "timing"):
                self.fixture.validate()

    def test_config_hardware_and_grid_cannot_be_swapped_between_pairs(self):
        for mutation in ("config", "hardware", "grid"):
            fresh = PairedFixture()
            try:
                def mutate(record):
                    if mutation == "config":
                        for snapshot in (record["snapshotBefore"], record["snapshotAfter"]):
                            snapshot["config"]["sha256"] = "c" * 64
                    elif mutation == "hardware":
                        for snapshot in (record["snapshotBefore"], record["snapshotAfter"]):
                            identity = copy.deepcopy(snapshot["host"]["identity"])
                            identity["hostname"] = "other-host"
                            snapshot["host"] = evidence.identity_record(identity, evidence.HOST_SCHEMA)
                    else:
                        record["trial"]["grid"] = [1, 2, 3]
                fresh.rewrite_baseline(0, mutate)
                with self.subTest(mutation=mutation), self.assertRaisesRegex(common.C6V2PolicyError, "incompatible"):
                    fresh.validate()
            finally:
                fresh.close()

    def test_window_and_exact_science_mismatches_reject(self):
        pair = self.fixture.claim["profiles"][0]["pairs"][0]
        record = json.loads((self.fixture.root / pair["baseline"]["path"]).read_text())
        process_link = record["trial"]["rawArtifacts"]["process"]
        process = json.loads((self.fixture.root / process_link["path"]).read_text())
        process["measurementWindow"]["warmupSteps"] += 1
        record["trial"]["rawArtifacts"]["process"] = write_json(self.fixture.root, process_link["path"], process)
        pair["baseline"] = write_json(self.fixture.root, pair["baseline"]["path"], record)
        with self.assertRaisesRegex(common.C6V2PolicyError, "incompatible"):
            self.fixture.validate()
        fresh = PairedFixture()
        try:
            def alter_science(record):
                record["trial"]["science"]["fields"]["p"] = fresh.bytes("baseline/changed-p.bin", b"changed science")
            fresh.rewrite_baseline(0, alter_science)
            with self.assertRaisesRegex(common.C6V2PolicyError, "exact science"):
                fresh.validate()
        finally:
            fresh.close()

    def test_baseline_source_inventory_cannot_be_forged_or_drift_across_trials(self):
        def drift(record):
            for snapshot in (record["snapshotBefore"], record["snapshotAfter"]):
                files = snapshot["source"]["inventory"]["files"]
                files[-1]["sha256"] = "d" * 64
                snapshot["source"]["inventory"]["sha256"] = hashlib.sha256(
                    common.release_manifest.canonical_bytes(files)).hexdigest()
        self.fixture.rewrite_baseline(1, drift)
        with self.assertRaisesRegex(common.C6V2PolicyError, "cohort drift"):
            self.fixture.validate()
        fresh = PairedFixture()
        try:
            def missing_member(record):
                for snapshot in (record["snapshotBefore"], record["snapshotAfter"]):
                    snapshot["source"]["inventory"]["files"].pop()
            fresh.rewrite_baseline(0, missing_member)
            with self.assertRaisesRegex(common.C6V2PolicyError, "inventory seal"):
                fresh.validate()
        finally:
            fresh.close()

    def test_current_trial_and_measurement_payload_cannot_detach(self):
        self.fixture.profile["trials"][0]["avgStepMs"] += 1
        with self.assertRaisesRegex(common.C6V2PolicyError, "detached candidate"):
            self.fixture.validate()

    def test_computed_small_gain_is_rejected_without_new_thresholds(self):
        for index in range(7):
            candidate_time = self.fixture.profile["trials"][index]["avgStepMs"]
            self.fixture.rewrite_baseline(index, lambda record: record["trial"].update(avgStepMs=candidate_time * 1.01))
        with self.assertRaisesRegex(common.C6V2PolicyError, ">2%"):
            self.fixture.validate()

    def test_finite_positive_inputs_cannot_round_to_an_impossible_complete_time_reduction(self):
        pair = self.fixture.claim["profiles"][0]["pairs"][0]
        record = json.loads((self.fixture.root / pair["candidate"]["path"]).read_text())
        record["trial"]["avgStepMs"] = 1e-300
        pair["candidate"] = self.fixture.write_measurement(record, pair["candidate"]["path"])
        self.fixture.profile["trials"][0] = {**record["trial"], "measurementArtifact": pair["candidate"]}
        self.fixture.rewrite_baseline(0, lambda value: value["trial"].update(avgStepMs=1e300))
        with self.assertRaisesRegex(common.C6V2PolicyError, "impossible"):
            self.fixture.validate()


if __name__ == "__main__":
    unittest.main()

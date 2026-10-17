#!/usr/bin/env python3
"""Synthetic activation and fail-closed mutations for C6's v2 draft gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification import validate_c6_threshold_candidate_v2 as c6  # noqa: E402
from scripts.qualification import collect_c6_evidence as evidence  # noqa: E402
from scripts.qualification.provenance import release_manifest  # noqa: E402


POLICY_PATH = ROOT / "tests/fixtures/reference/performance/threshold_policy_v2.json"
SHA_A = "a" * 64
SHA_B = "b" * 64


def write_json(root: Path, relative: str, value: dict) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": relative, "sha256": c6.digest(path)}


def write_bytes(root: Path, relative: str, value: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"path": relative, "sha256": c6.digest(path)}


class C6V2ThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.policy = copy.deepcopy(json.loads(POLICY_PATH.read_text(encoding="utf-8")))
        write_bytes(self.root, "mk/tests.mk", b"synthetic canonical declaration")
        write_bytes(self.root, "src/core.cpp", b"synthetic source")
        source = evidence.source_inventory(self.root, _roots=[Path("mk/tests.mk"), Path("src")])
        self.seal = evidence.source_seal(source)
        self.source_sha = self.seal["inventorySha256"]
        self.policy["canonicalSource"] = {key: value for key, value in self.seal.items() if key != "record"}
        self.policy["canonicalSource"]["inventoryArtifact"] = write_json(self.root, "identities/source.json", source)
        self.policy["historicalV1"] = write_json(
            self.root, "historical-v1.json",
            json.loads((ROOT / self.policy["historicalV1"]["path"]).read_text(encoding="utf-8")),
        )
        self.binary = self.root / "bin/nimbus_sim"
        write_bytes(self.root, "bin/nimbus_sim", b"synthetic test-only release binary")
        self.policy["releaseBinary"]["sha256"] = c6.digest(self.binary)
        self._build_complete_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def science(self, prefix: str) -> dict:
        checkpoint = write_bytes(self.root, f"{prefix}/checkpoint.bin", b"synthetic checkpoint")
        checkpoint["bytes"] = len(b"synthetic checkpoint")
        return {
            "checkpoint": checkpoint,
            "fields": {field: write_bytes(self.root, f"{prefix}/fields/{field}.bin", field.encode())
                       for field in c6.FIELDS},
            "ledgers": {ledger: write_bytes(self.root, f"{prefix}/ledgers/{ledger}", ledger.encode())
                        for ledger in c6.LEDGERS},
        }

    def _build_complete_fixture(self) -> None:
        for row in self.policy["workloadClasses"]:
            identifier = row["id"]
            profiles = []
            limit_profiles = []
            for mode in row["requiredModes"]:
                grid = row["grid"] or [12, 12, 16]
                config_path = row["config"] or f"configs/{identifier}-{mode}.yaml"
                config = write_bytes(self.root, config_path, b"synthetic config")
                device_record = (evidence.identity_record({"name": "synthetic GPU", "driver": "synthetic-driver",
                                  "backendVersion": "test"}, evidence.DEVICE_SCHEMA) if mode in c6.GPU_MODES else None)
                device = evidence.validate_identity(device_record, evidence.DEVICE_SCHEMA) if device_record else None
                host_record = evidence.identity_record({"os": "test-os", "architecture": "test-arch",
                              "cpuModel": "test-cpu", "logicalThreads": 8, "ramBytes": 10_000_000,
                              "hostname": "test-host"}, evidence.HOST_SCHEMA)
                host = evidence.validate_identity(host_record, evidence.HOST_SCHEMA)
                snapshot = {"schema": evidence.SNAPSHOT_SCHEMA,
                            "binary": release_manifest.file_record(self.binary, self.root),
                            "config": release_manifest.file_record(self.root / config_path, self.root),
                            "source": self.seal["record"], "host": host_record, "device": device_record}
                window = {"warmupSteps": 20, "measuredSteps": 60, "timeoutSeconds": 20}
                times = [10.0 + index * .01 for index in range(7)]
                trials = []
                reference = self.science(f"science/{identifier}-{mode}/reference")
                for number, timing in enumerate(times, start=1):
                    validation = {
                        "violations": 0, "exitStatus": 0, "skipped": 0, "trialIndex": number,
                        "binarySha256": self.policy["releaseBinary"]["sha256"],
                        "sourceInventorySha256": self.source_sha, "configSha256": config["sha256"],
                    }
                    trials.append({
                        "index": number, "status": "completed", "skipped": False, "fallbackUsed": False,
                        "binarySha256": self.policy["releaseBinary"]["sha256"],
                        "sourceInventorySha256": self.source_sha, "configSha256": config["sha256"],
                        "hostFingerprintSha256": host["fingerprintSha256"],
                        "deviceFingerprintSha256": device["fingerprintSha256"] if device else None,
                        "mode": mode, "grid": grid,
                        "avgStepMs": timing, "stepWallSeconds": 1.0,
                        "namedWallFraction": .97,
                        "peakHostBytes": 1_000_000,
                        "peakDeviceBytes": 2_000_000 if device else None,
                        "peakReaderHostBytes": 500_000 if mode == "cpu_writer_studio_reader" else None,
                        "readerBytesDecoded": 5_000 if mode == "cpu_writer_studio_reader" else None,
                        "validation": validation,
                        "validationArtifact": write_json(self.root, f"validation/{identifier}-{mode}-{number}.json", validation),
                        "science": self.science(f"science/{identifier}-{mode}/trial{number}"),
                        "rawLog": write_bytes(self.root, f"logs/{identifier}-{mode}-{number}.log", b"synthetic log"),
                    })
                    trial = trials[-1]
                    identity = evidence.trial_identity(snapshot, mode, grid)
                    trial["identityBefore"] = identity
                    trial["identityAfter"] = identity
                    process = {"schema": evidence.PROCESS_SCHEMA, "trialIndex": number,
                               "exitStatus": 0, "timedOut": False, "fallbackUsed": False,
                               "identity": identity, "measurementWindow": window,
                               "measurements": {key: trial.get(key) for key in evidence.MEASURED_FIELDS},
                               "rawLog": trial["rawLog"]}
                    trial["rawArtifacts"] = {"process": write_json(
                        self.root, f"process/{identifier}-{mode}-{number}.json", process)}
                    trial["measurementArtifact"] = write_json(
                        self.root, f"measurements/{identifier}-{mode}-{number}.json",
                        evidence.measurement_record(trial, snapshot, snapshot))
                median = statistics.median(times)
                profiles.append({
                    "mode": mode, "binarySha256": self.policy["releaseBinary"]["sha256"],
                    "sourceInventorySha256": self.source_sha, "config": config, "grid": grid,
                    "fullFieldBytes": grid[0] * grid[1] * grid[2] * 4,
                    "host": host, "hostArtifact": write_json(self.root, f"identities/{identifier}-{mode}-host.json", host_record),
                    "device": device, "deviceArtifact": write_json(self.root, f"identities/{identifier}-{mode}-device.json",
                                                               device_record) if device_record else None,
                    "referenceScience": reference, "measurementWindow": window,
                    "trials": trials,
                    "aggregate": {"medianStepMs": median, "madStepMs": .02,
                                  "cv": statistics.pstdev(times) / statistics.mean(times),
                                  "maxPeakHostBytes": 1_000_000,
                                  "maxPeakDeviceBytes": 2_000_000 if device else None,
                                  "maxPeakReaderHostBytes": 500_000 if mode == "cpu_writer_studio_reader" else None,
                                  "minNamedWallFraction": .97},
                })
                limit_profiles.append({
                    "mode": mode, "binarySha256": self.policy["releaseBinary"]["sha256"],
                    "sourceInventorySha256": self.source_sha, "configSha256": config["sha256"],
                    "hostFingerprintSha256": host["fingerprintSha256"],
                    "deviceFingerprintSha256": device["fingerprintSha256"] if device else None,
                    "grid": grid, "maxMedianStepMs": 20, "maxCv": .1,
                    "maxPeakHostBytes": 2_000_000,
                    "maxPeakDeviceBytes": 3_000_000 if device else None,
                    "maxPeakReaderHostBytes": 600_000 if mode == "cpu_writer_studio_reader" else None,
                })
            receipt = {
                "schema": c6.RECEIPT_SCHEMA, "status": "complete", "workloadId": identifier,
                "binarySha256": self.policy["releaseBinary"]["sha256"],
                "sourceInventorySha256": self.source_sha, "operator": "synthetic-operator",
                "profiles": profiles, "repairClaim": None,
            }
            row["receipt"] = write_json(self.root, f"receipts/{identifier}.json", receipt)
            review_record = {
                "schema": "nimbus.c6-independent-limit-review.v1", "workloadId": identifier,
                "reviewer": "synthetic-independent-reviewer", "decision": "approved",
                "binarySha256": self.policy["releaseBinary"]["sha256"],
                "sourceInventorySha256": self.source_sha,
                "limitProfilesSha256": hashlib.sha256(release_manifest.canonical_bytes(limit_profiles)).hexdigest(),
            }
            limits = {
                "schema": c6.LIMIT_SCHEMA, "status": "reviewed", "workloadId": identifier,
                "receiptSha256": row["receipt"]["sha256"],
                "review": {"reviewer": "synthetic-independent-reviewer", "decision": "approved",
                           "record": write_json(self.root, f"reviews/{identifier}.json", review_record)},
                "profiles": limit_profiles,
            }
            row["limits"] = write_json(self.root, f"limits/{identifier}.json", limits)
            row["missing"] = []
        self.policy["status"] = "reviewed_active"
        self.reseal_activation()

    def reseal_activation(self) -> None:
        unsigned = copy.deepcopy(self.policy)
        unsigned["activationReview"] = None
        review = {
            "schema": c6.REVIEW_SCHEMA, "decision": "approved", "reviewer": "synthetic-reviewer",
            "candidateUnsignedSha256": hashlib.sha256(release_manifest.canonical_bytes(unsigned)).hexdigest(),
            "binarySha256": self.policy["releaseBinary"]["sha256"],
            "sourceInventorySha256": self.source_sha,
        }
        self.policy["activationReview"] = write_json(self.root, "activation.json", review)

    def read_receipt(self, row_index: int = 0) -> dict:
        return json.loads((self.root / self.policy["workloadClasses"][row_index]["receipt"]["path"]).read_text())

    def reseal_receipt(self, receipt: dict, row_index: int = 0) -> None:
        row = self.policy["workloadClasses"][row_index]
        row["receipt"] = write_json(self.root, row["receipt"]["path"], receipt)
        limits = json.loads((self.root / row["limits"]["path"]).read_text())
        limits["receiptSha256"] = row["receipt"]["sha256"]
        row["limits"] = write_json(self.root, row["limits"]["path"], limits)
        self.reseal_activation()

    def read_limits(self, row_index: int = 0) -> dict:
        return json.loads((self.root / self.policy["workloadClasses"][row_index]["limits"]["path"]).read_text())

    def reseal_limits(self, limits: dict, row_index: int = 0) -> None:
        row = self.policy["workloadClasses"][row_index]
        review = json.loads((self.root / limits["review"]["record"]["path"]).read_text())
        review["limitProfilesSha256"] = hashlib.sha256(release_manifest.canonical_bytes(limits["profiles"])).hexdigest()
        limits["review"]["record"] = write_json(self.root, limits["review"]["record"]["path"], review)
        row["limits"] = write_json(self.root, row["limits"]["path"], limits)
        self.reseal_activation()

    def reseal_measurement(self, trial: dict) -> None:
        link = trial["measurementArtifact"]
        record = json.loads((self.root / link["path"]).read_text())
        record["trial"] = {key: value for key, value in trial.items() if key != "measurementArtifact"}
        trial["measurementArtifact"] = write_json(self.root, link["path"], record)

    def validate(self) -> dict:
        return c6._validate_policy(self.policy, self.root, self.binary, source_seal=self.seal)

    def test_actual_repository_policy_is_draft(self) -> None:
        actual = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(actual["status"], "draft_not_activated")
        self.assertIsNone(actual["activationReview"])
        self.assertTrue(all(row["receipt"] is None and row["limits"] is None for row in actual["workloadClasses"]))
        if actual["releaseBinary"]["sha256"] != c6.digest(ROOT / "bin/nimbus_sim"):
            with self.assertRaisesRegex(c6.C6V2PolicyError, "stale release binary"):
                c6.validate_policy(actual, ROOT, ROOT / "bin/nimbus_sim")
        else:
            self.assertFalse(c6.validate_policy(actual, ROOT, ROOT / "bin/nimbus_sim")["releaseReady"])

    def test_synthetic_complete_activation_requires_ten_receipts_and_terrain_three_modes(self) -> None:
        result = self.validate()
        self.assertTrue(result["releaseReady"])
        self.assertEqual(len(result["validatedWorkloads"]), 10)
        self.assertEqual(len(result["validatedWorkloads"]["retained_terrain_postimplementation"]), 3)

    def test_public_validator_completes_final_evidence_seal(self) -> None:
        with mock.patch.object(c6, "canonical_source_seal", return_value=self.seal):
            self.assertTrue(c6.validate_policy(self.policy, self.root, self.binary)["releaseReady"])

    def test_collector_rejects_raw_artifact_mutation_before_sealing(self) -> None:
        trial = self.read_receipt()["profiles"][0]["trials"][0]
        evidence.validate_trial_artifacts(trial, self.root)
        (self.root / trial["rawLog"]["path"]).write_bytes(b"changed before collector seal")
        with self.assertRaisesRegex(evidence.C6EvidenceError, "stale raw evidence"):
            evidence.validate_trial_artifacts(trial, self.root)

    def test_stale_binary_and_source_fail(self) -> None:
        self.binary.write_bytes(b"changed")
        with self.assertRaisesRegex(c6.C6V2PolicyError, "stale release binary"):
            self.validate()
        self.binary.write_bytes(b"synthetic test-only release binary")
        self.seal["inventorySha256"] = SHA_B
        with self.assertRaisesRegex(c6.C6V2PolicyError, "stale canonical source inventory"):
            self.validate()

    def test_stale_report_hash_fails(self) -> None:
        path = self.root / self.policy["workloadClasses"][0]["receipt"]["path"]
        path.write_text("{}")
        with self.assertRaisesRegex(c6.C6V2PolicyError, "stale .* receipt report"):
            self.validate()

    def test_less_than_seven_or_skipped_trial_fails(self) -> None:
        receipt = self.read_receipt()
        receipt["profiles"][0]["trials"].pop()
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "fewer than seven"):
            self.validate()
        receipt = self.read_receipt()
        receipt["profiles"][0]["trials"].append(copy.deepcopy(receipt["profiles"][0]["trials"][-1]))
        receipt["profiles"][0]["trials"][-1]["index"] = 7
        receipt["profiles"][0]["trials"][0]["skipped"] = True
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "skipped"):
            self.validate()

    def test_named_wall_and_validation_fail(self) -> None:
        receipt = self.read_receipt()
        receipt["profiles"][0]["trials"][0]["namedWallFraction"] = .949
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "95% named wall"):
            self.validate()
        receipt = self.read_receipt()
        receipt["profiles"][0]["trials"][0]["namedWallFraction"] = .97
        receipt["profiles"][0]["trials"][0]["validation"]["violations"] = 1
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "nonzero validation"):
            self.validate()

    def test_exact_checkpoint_eight_fields_and_six_ledgers_fail(self) -> None:
        for mutation in ("checkpoint", "fields", "ledgers"):
            with self.subTest(mutation=mutation):
                receipt = self.read_receipt()
                science = receipt["profiles"][0]["trials"][0]["science"]
                if mutation == "checkpoint":
                    science["checkpoint"]["sha256"] = SHA_B
                elif mutation == "fields":
                    science["fields"].pop("p")
                else:
                    science["ledgers"].pop("budget.csv")
                self.reseal_receipt(receipt)
                with self.assertRaisesRegex(c6.C6V2PolicyError, "checkpoint|exact"):
                    self.validate()
                receipt = self.read_receipt()
                receipt["profiles"][0]["trials"][0]["science"] = self.science(f"science/restore-{mutation}")
                self.reseal_receipt(receipt)

    def test_wrong_size_and_backend_fail(self) -> None:
        receipt = self.read_receipt()
        receipt["profiles"][0]["grid"] = [1, 1, 1]
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "wrong size"):
            self.validate()
        receipt = self.read_receipt()
        receipt["profiles"][0]["grid"] = [24, 24, 20]
        receipt["profiles"][0]["mode"] = "metal"
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "wrong size/backend applicability"):
            self.validate()

    def test_policy_cannot_redefine_retained_size_or_config(self) -> None:
        row = self.policy["workloadClasses"][0]
        row["grid"] = [1, 1, 1]
        row["fullFieldBytes"] = 4
        with self.assertRaisesRegex(c6.C6V2PolicyError, "differs from retained inventory"):
            self.validate()

    def test_missing_terrain_backend_fails(self) -> None:
        receipt = self.read_receipt(9)
        receipt["profiles"].pop()
        self.reseal_receipt(receipt, 9)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "wrong size/backend applicability"):
            self.validate()

    def test_host_device_and_config_identity_fail(self) -> None:
        receipt = self.read_receipt(4)
        trial = receipt["profiles"][0]["trials"][0]
        trial["deviceFingerprintSha256"] = SHA_A
        self.reseal_receipt(receipt, 4)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "trial identity drift"):
            self.validate()
        receipt = self.read_receipt(0)
        receipt["profiles"][0]["config"]["sha256"] = SHA_B
        self.reseal_receipt(receipt, 0)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "stale .* config"):
            self.validate()

    def test_peak_variance_and_limit_applicability_fail(self) -> None:
        receipt = self.read_receipt(4)
        receipt["profiles"][0]["trials"][0]["peakDeviceBytes"] = None
        self.reseal_receipt(receipt, 4)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "device peak"):
            self.validate()
        receipt = self.read_receipt(0)
        receipt["profiles"][0]["aggregate"]["cv"] = 0
        self.reseal_receipt(receipt, 0)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "variance or peak aggregate"):
            self.validate()
        receipt["profiles"][0]["aggregate"]["cv"] = statistics.pstdev([10 + i * .01 for i in range(7)]) / statistics.mean([10 + i * .01 for i in range(7)])
        self.reseal_receipt(receipt, 0)
        limits = self.read_limits(0)
        limits["profiles"][0]["grid"] = [32, 32, 24]
        self.reseal_limits(limits, 0)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "limit applicability mismatch"):
            self.validate()

    def test_unreviewed_or_placeholder_limit_fails(self) -> None:
        limits = self.read_limits()
        limits["review"]["decision"] = "pending"
        self.reseal_limits(limits)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "independent limit review"):
            self.validate()
        limits["review"]["decision"] = "approved"
        limits["profiles"][0]["maxMedianStepMs"] = 0
        self.reseal_limits(limits)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "placeholder limit"):
            self.validate()

    def test_review_record_binds_exact_limit_values(self) -> None:
        row = self.policy["workloadClasses"][0]
        limits = self.read_limits()
        limits["profiles"][0]["maxMedianStepMs"] = 30
        row["limits"] = write_json(self.root, row["limits"]["path"], limits)
        self.reseal_activation()
        with self.assertRaisesRegex(c6.C6V2PolicyError, "limit review does not bind exact limits"):
            self.validate()

    def test_validation_artifact_and_trial_science_files_are_bound(self) -> None:
        receipt = self.read_receipt()
        validation_path = receipt["profiles"][0]["trials"][0]["validationArtifact"]["path"]
        (self.root / validation_path).write_text("{}")
        with self.assertRaisesRegex(c6.C6V2PolicyError, "stale .* validation report"):
            self.validate()
        receipt = self.read_receipt()
        write_json(self.root, validation_path, receipt["profiles"][0]["trials"][0]["validation"])
        science_path = receipt["profiles"][0]["trials"][0]["science"]["fields"]["p"]["path"]
        (self.root / science_path).write_bytes(b"mutated field")
        with self.assertRaisesRegex(c6.C6V2PolicyError, "stale .* p"):
            self.validate()

    def test_threshold_exceedance_and_stale_activation_fail(self) -> None:
        limits = self.read_limits()
        limits["profiles"][0]["maxMedianStepMs"] = 1
        self.reseal_limits(limits)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "threshold exceeded"):
            self.validate()
        limits["profiles"][0]["maxMedianStepMs"] = 20
        self.reseal_limits(limits)
        self.policy["workloadClasses"][0]["missing"] = ["not really complete"]
        with self.assertRaisesRegex(c6.C6V2PolicyError, "still declares missing"):
            self.validate()
        self.policy["workloadClasses"][0]["missing"] = []
        self.policy["activationReview"]["sha256"] = SHA_B
        with self.assertRaisesRegex(c6.C6V2PolicyError, "stale activation review report"):
            self.validate()

    def test_rejected_speedup_cannot_be_promoted(self) -> None:
        receipt = self.read_receipt()
        receipt["repairClaim"] = {"kind": "paired_time_gain", "pairedGainPercent": [1.9] * 7}
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "2% and >3 MAD"):
            self.validate()
        receipt["repairClaim"]["pairedGainPercent"] = [0, 1, 2, 3, 4, 5, 6]
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "2% and >3 MAD"):
            self.validate()

    def test_full_source_inventory_cannot_omit_a_member(self) -> None:
        source = copy.deepcopy(self.seal["record"])
        source["inventory"]["files"].pop()
        self.policy["canonicalSource"]["inventoryArtifact"] = write_json(self.root, "identities/source.json", source)
        self.reseal_activation()
        with self.assertRaisesRegex(c6.C6V2PolicyError, "incomplete or stale canonical source inventory artifact"):
            self.validate()

    def test_new_source_member_invalidates_receipts(self) -> None:
        write_bytes(self.root, "src/new.metal", b"new shader")
        self.seal = evidence.source_seal(evidence.source_inventory(self.root, _roots=[Path("mk/tests.mk"), Path("src")]))
        with self.assertRaisesRegex(c6.C6V2PolicyError, "stale canonical source inventory"):
            self.validate()

    def test_forged_host_or_device_fingerprint_fails(self) -> None:
        for row_index, kind in ((0, "host"), (4, "device")):
            with self.subTest(kind=kind):
                receipt = self.read_receipt(row_index)
                profile = receipt["profiles"][0]
                link = profile[f"{kind}Artifact"]
                record = json.loads((self.root / link["path"]).read_text())
                record["fingerprintSha256"] = SHA_A
                profile[kind]["fingerprintSha256"] = SHA_A
                profile[f"{kind}Artifact"] = write_json(self.root, link["path"], record)
                self.reseal_receipt(receipt, row_index)
                with self.assertRaisesRegex(c6.C6V2PolicyError, "hardware fingerprint does not match identity"):
                    self.validate()
                original = evidence.identity_record(record["identity"], record["schema"])
                profile[kind] = evidence.validate_identity(original, record["schema"])
                profile[f"{kind}Artifact"] = write_json(self.root, link["path"], original)
                self.reseal_receipt(receipt, row_index)

    def test_stale_host_artifact_fails(self) -> None:
        receipt = self.read_receipt()
        (self.root / receipt["profiles"][0]["hostArtifact"]["path"]).write_text("{}")
        with self.assertRaisesRegex(c6.C6V2PolicyError, "stale .* host report"):
            self.validate()

    def test_inline_metric_cannot_change_without_measurement_artifact(self) -> None:
        receipt = self.read_receipt()
        receipt["profiles"][0]["trials"][0]["avgStepMs"] += .1
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "raw measurement artifact disagrees"):
            self.validate()

    def test_raw_process_metric_and_raw_log_mutations_fail(self) -> None:
        receipt = self.read_receipt()
        trial = receipt["profiles"][0]["trials"][0]
        link = trial["rawArtifacts"]["process"]
        process = json.loads((self.root / link["path"]).read_text())
        process["measurements"]["peakHostBytes"] += 1
        trial["rawArtifacts"]["process"] = write_json(self.root, link["path"], process)
        self.reseal_measurement(trial)
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "process measurement artifact disagrees"):
            self.validate()
        process["measurements"]["peakHostBytes"] -= 1
        trial["rawArtifacts"]["process"] = write_json(self.root, link["path"], process)
        self.reseal_measurement(trial)
        self.reseal_receipt(receipt)
        (self.root / trial["rawLog"]["path"]).write_bytes(b"tampered log")
        with self.assertRaisesRegex(c6.C6V2PolicyError, "stale .* trial log"):
            self.validate()

    def test_before_after_snapshots_cannot_hide_host_drift(self) -> None:
        receipt = self.read_receipt()
        trial = receipt["profiles"][0]["trials"][0]
        link = trial["measurementArtifact"]
        measurement = json.loads((self.root / link["path"]).read_text())
        measurement["snapshotAfter"]["host"]["identity"]["ramBytes"] += 1
        trial["measurementArtifact"] = write_json(self.root, link["path"], measurement)
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "collection snapshot drift"):
            self.validate()

    def test_trial_cannot_reuse_another_trials_raw_log(self) -> None:
        receipt = self.read_receipt()
        trials = receipt["profiles"][0]["trials"]
        trial = trials[1]
        trial["rawLog"] = trials[0]["rawLog"]
        link = trial["rawArtifacts"]["process"]
        process = json.loads((self.root / link["path"]).read_text())
        process["rawLog"] = trial["rawLog"]
        trial["rawArtifacts"]["process"] = write_json(self.root, link["path"], process)
        self.reseal_measurement(trial)
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "reused raw artifact across trials"):
            self.validate()

    def test_future_science_payload_cannot_serve_as_earlier_raw_log(self) -> None:
        receipt = self.read_receipt()
        trials = receipt["profiles"][0]["trials"]
        trial = trials[0]
        trial["rawLog"] = trials[1]["science"]["fields"]["qv"]
        link = trial["rawArtifacts"]["process"]
        process = json.loads((self.root / link["path"]).read_text())
        process["rawLog"] = trial["rawLog"]
        trial["rawArtifacts"]["process"] = write_json(self.root, link["path"], process)
        self.reseal_measurement(trial)
        self.reseal_receipt(receipt)
        with self.assertRaisesRegex(c6.C6V2PolicyError, "science artifact reused as earlier raw artifact"):
            self.validate()

    def test_raw_file_mutation_after_read_is_caught_by_final_seal(self) -> None:
        trial = self.read_receipt()["profiles"][0]["trials"][0]
        path = self.root / trial["rawLog"]["path"]
        original_validator = c6._validate_policy

        def mutate_after_validation(*args, **kwargs):
            result = original_validator(*args, **kwargs)
            path.write_bytes(b"mutated after read")
            return result

        with mock.patch.object(c6, "canonical_source_seal", return_value=self.seal), \
                mock.patch.object(c6, "_validate_policy", side_effect=mutate_after_validation):
            with self.assertRaisesRegex(c6.C6V2PolicyError, "stale final-seal evidence"):
                c6.validate_policy(self.policy, self.root, self.binary)

    def test_source_mutation_during_validation_fails_final_inventory(self) -> None:
        changed = copy.deepcopy(self.seal)
        changed["inventorySha256"] = SHA_B
        with mock.patch.object(c6, "canonical_source_seal", side_effect=[self.seal, changed]):
            with self.assertRaisesRegex(c6.C6V2PolicyError, "canonical source drift during validation"):
                c6.validate_policy(self.policy, self.root, self.binary)


if __name__ == "__main__":
    unittest.main()

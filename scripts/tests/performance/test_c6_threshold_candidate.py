#!/usr/bin/env python3
"""Fail-closed mutations for the C6 draft release-threshold policy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification import validate_c6_threshold_candidate as c6


POLICY_PATH = ROOT / "tests/fixtures/reference/performance/threshold_policy_v1.json"
AB_PATH = ROOT / "tests/fixtures/reference/performance/observations/cpu_milbrandt_cold_gamma_ab.json"


class C6ThresholdCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        cls.report = json.loads(AB_PATH.read_text(encoding="utf-8"))
        cls.observation = next(row for row in cls.policy["observations"] if row["kind"] == "paired_ab")

    def copy_report(self) -> dict:
        return copy.deepcopy(self.report)

    def test_current_policy_is_valid_draft_not_release_ready(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["releaseBinary"]["sha256"] = c6.sha256(ROOT / "bin/nimbus_sim")
        result = c6.validate_policy(policy, ROOT, ROOT / "bin/nimbus_sim")
        self.assertFalse(result["releaseReady"])
        self.assertEqual(len(result["missingWorkloadClasses"]), len(c6.REQUIRED_WORKLOADS))

    def test_stale_binary_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "different-binary"
            binary.write_bytes(b"stale")
            with self.assertRaisesRegex(c6.C6PolicyError, "stale release binary"):
                c6.validate_release_binary(self.policy, binary)

    def test_fewer_than_seven_pairs_fails(self) -> None:
        report = self.copy_report()
        report["pairs"] = report["pairs"][:6]
        report["contract"]["trials"] = 6
        with self.assertRaisesRegex(c6.C6PolicyError, "fewer than seven"):
            c6.validate_paired_ab(report, self.observation, self.policy["acceptance"])

    def test_less_than_95_percent_named_wall_fails(self) -> None:
        report = self.copy_report()
        report["pairs"][0]["candidate"]["named_wall_fraction"] = 0.949
        with self.assertRaisesRegex(c6.C6PolicyError, "named wall coverage"):
            c6.validate_paired_ab(report, self.observation, self.policy["acceptance"])

    def test_missing_exact_science_fails(self) -> None:
        for mutation in ("pair_flag", "checkpoint", "field", "ledger"):
            with self.subTest(mutation=mutation):
                report = self.copy_report()
                candidate = report["pairs"][0]["candidate"]
                if mutation == "pair_flag":
                    report["pairs"][0]["exact_state_fields_and_ledgers"] = False
                elif mutation == "checkpoint":
                    candidate["checkpoint_sha256"] = ""
                elif mutation == "field":
                    candidate["field_sha256"].pop("p")
                else:
                    candidate["ledger_sha256"].pop("budget.csv")
                with self.assertRaisesRegex(c6.C6PolicyError, "exact"):
                    c6.validate_paired_ab(report, self.observation, self.policy["acceptance"])

    def with_gains(self, gains: list[float], disposition: str) -> tuple[dict, dict]:
        report = self.copy_report()
        observation = copy.deepcopy(self.observation)
        for pair, gain in zip(report["pairs"], gains):
            baseline_ms = pair["baseline"]["avg_step_ms"]
            pair["candidate"]["avg_step_ms"] = baseline_ms * (1.0 - gain / 100.0)
            pair["speedup_percent"] = gain
        median = statistics.median(gains)
        mad = statistics.median(abs(gain - median) for gain in gains)
        report["aggregate"]["paired_speedups_percent"] = gains
        report["aggregate"]["median_speedup_percent"] = median
        report["aggregate"]["paired_mad_percent"] = mad
        report["aggregate"]["three_mad_percent"] = 3.0 * mad
        report["aggregate"]["time_gate_passed"] = disposition == "time_gain_accepted"
        report["disposition"] = disposition
        observation["disposition"] = disposition
        return report, observation

    def test_below_two_percent_cannot_be_promoted(self) -> None:
        report, observation = self.with_gains([1.9] * 7, "time_gain_accepted")
        with self.assertRaisesRegex(c6.C6PolicyError, "time gate"):
            c6.validate_paired_ab(report, observation, self.policy["acceptance"])

    def test_below_three_mad_cannot_be_promoted(self) -> None:
        report, observation = self.with_gains([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "time_gain_accepted")
        with self.assertRaisesRegex(c6.C6PolicyError, "time gate"):
            c6.validate_paired_ab(report, observation, self.policy["acceptance"])

    def test_valid_strict_gain_can_pass_candidate_gate(self) -> None:
        report, observation = self.with_gains([3.2] * 7, "time_gain_accepted")
        result = c6.validate_paired_ab(report, observation, self.policy["acceptance"])
        self.assertTrue(result["timeGainAccepted"])

    def test_memory_alternative_and_outside_cost_fail_closed(self) -> None:
        self.assertTrue(c6.qualifies_memory_gain(46080, 26730496, 1, 0, False))
        self.assertTrue(c6.qualifies_memory_gain(46080, 26730496, 0, 534610, False))
        self.assertFalse(c6.qualifies_memory_gain(46080, 26730496, 0, 534609, False))
        self.assertFalse(c6.qualifies_memory_gain(46080, 26730496, 1, 0, True))

    def test_missing_workload_or_stale_report_fails(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["releaseBinary"]["sha256"] = c6.sha256(ROOT / "bin/nimbus_sim")
        policy["workloadClasses"].pop()
        with self.assertRaisesRegex(c6.C6PolicyError, "workload classes"):
            c6.validate_policy(policy, ROOT, ROOT / "bin/nimbus_sim")
        policy = copy.deepcopy(self.policy)
        policy["releaseBinary"]["sha256"] = c6.sha256(ROOT / "bin/nimbus_sim")
        policy["observations"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(c6.C6PolicyError, "stale C6 evidence"):
            c6.validate_policy(policy, ROOT, ROOT / "bin/nimbus_sim")

    def test_unvalidated_receipt_and_threshold_placeholders_do_not_clear_work(self) -> None:
        policy = copy.deepcopy(self.policy)
        row = policy["workloadClasses"][0]
        row["receipt"] = {"unvalidated": True}
        row["thresholds"] = {"unvalidated": True}
        missing = c6.validate_workloads(policy["workloadClasses"], ROOT)
        self.assertIn(row["id"], missing)


if __name__ == "__main__":
    unittest.main()

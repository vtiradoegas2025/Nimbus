#!/usr/bin/env python3
"""Numeric boundary and coverage mutations for the planner preflight probe."""

from __future__ import annotations

from copy import deepcopy
import unittest

from scripts.qualification.probe_bunkers_left_planner_threshold import (
    VALUES, validate_results,
)


class BunkersLeftPlannerThresholdTests(unittest.TestCase):
    def test_complete_monotone_boundary_passes(self) -> None:
        rows = [{"ny": ny, "status": "pass" if ny >= 180 else "rejected"}
                for ny in VALUES]
        self.assertEqual(validate_results(rows), 180)

    def test_missing_and_reordered_values_reject(self) -> None:
        rows = [{"ny": ny, "status": "pass" if ny >= 180 else "rejected"}
                for ny in VALUES]
        with self.assertRaisesRegex(RuntimeError, "omitted, duplicated or misordered"):
            validate_results(rows[:-1])
        mutant = deepcopy(rows)
        mutant[1], mutant[2] = mutant[2], mutant[1]
        with self.assertRaisesRegex(RuntimeError, "omitted, duplicated or misordered"):
            validate_results(mutant)

    def test_false_success_and_nonmonotone_science_reject(self) -> None:
        rows = [{"ny": ny, "status": "pass" if ny >= 180 else "rejected"}
                for ny in VALUES]
        mutant = deepcopy(rows)
        mutant[0]["status"] = "pass"
        with self.assertRaisesRegex(RuntimeError, "both threshold sides"):
            validate_results(mutant)
        mutant = deepcopy(rows)
        mutant[12]["status"] = "rejected"
        with self.assertRaisesRegex(RuntimeError, "nonmonotone"):
            validate_results(mutant)


if __name__ == "__main__":
    unittest.main()

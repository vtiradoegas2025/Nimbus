#!/usr/bin/env python3
"""Reverse-coverage checks for first-party YAML source classification."""

from __future__ import annotations

import unittest

from scripts.qualification.survey_config_source_axes import summarize


AXES = {"output.format": {"allowed": ["npy_3d", "netcdf", "zfp_low"],
                          "effectiveCounts": {"npy_3d": 2},
                          "reviewedAliasDispositions": {}},
        "grid.staggering": {"allowed": ["c_grid", "collocated"],
                            "effectiveCounts": {"c_grid": 2},
                            "reviewedAliasDispositions": {}}}


class ConfigSourceAxisSurveyTests(unittest.TestCase):
    def test_new_choice_is_distinct_from_still_unrepresented_choice(self) -> None:
        rows = [{"id": "source:extra.yaml", "strictLoader": "canonicalized",
                 "effectiveAxisValues": {"output.format": "netcdf", "grid.staggering": "c_grid"},
                 "explicitAxisValues": {"output.format": "netcdf"}}]
        result = summarize(rows, {"source:maintained.yaml"}, AXES)
        formats = result["axisRows"][1]
        self.assertEqual(formats["newlyRepresentedNonAliasChoices"], ["netcdf"])
        self.assertEqual(formats["remainingNonAliasChoices"], ["zfp_low"])

    def test_rejected_source_does_not_cover_a_physical_choice(self) -> None:
        rows = [{"id": "source:negative.yaml", "strictLoader": "rejected",
                 "explicitAxisValues": {"output.format": "zfp_low"}}]
        result = summarize(rows, set(), AXES)
        formats = result["axisRows"][1]
        self.assertEqual(formats["rejectedSourceExplicitValueCounts"], {"zfp_low": 1})
        self.assertEqual(formats["remainingNonAliasChoices"], ["netcdf", "zfp_low"])

    def test_duplicate_and_maintained_source_reject(self) -> None:
        row = {"id": "source:duplicate.yaml", "strictLoader": "rejected",
               "explicitAxisValues": {}}
        with self.assertRaisesRegex(RuntimeError, "duplicated or absorbed"):
            summarize([row, row], set(), AXES)
        with self.assertRaisesRegex(RuntimeError, "duplicated or absorbed"):
            summarize([row], {row["id"]}, AXES)


if __name__ == "__main__":
    unittest.main()

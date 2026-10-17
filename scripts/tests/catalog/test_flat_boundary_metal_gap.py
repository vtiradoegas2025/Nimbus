#!/usr/bin/env python3
"""Source-reviewed structural/public Metal boundary partition checks."""

from __future__ import annotations

import unittest

from scripts.qualification.analyze_flat_boundary_metal_gap import category
from scripts.qualification.enumerate_flat_boundary_family import enumerate_rows


def selected(laterals: tuple[str, ...], bottom: str, top: str) -> dict:
    return {"lateral": dict(zip(("west", "east", "south", "north"), laterals)),
            "bottom": bottom, "top": top}


class FlatBoundaryMetalGapTests(unittest.TestCase):
    def test_canonical_and_mixed_native_are_distinct_public_classes(self) -> None:
        self.assertEqual(category(selected(("rigid",) * 4, "rigid_slip", "rigid")),
                         "canonical_native_and_public")
        self.assertEqual(category(selected(("rigid_slip",) * 4, "rigid_noslip", "open")),
                         "structural_native_but_public_guarded")

    def test_neighbor_boundary_does_not_inherit_native_flag(self) -> None:
        self.assertEqual(category(selected(("rigid", "rigid_slip", "rigid_slip", "rigid_slip"),
                                           "rigid_noslip", "open")),
                         "not_structural_native_explicit_host_transport")
        self.assertEqual(category(selected(("rigid_slip",) * 4, "rigid_slip", "open")),
                         "not_structural_native_explicit_host_transport")

    def test_complete_fixed_family_has_exact_reviewed_partition(self) -> None:
        counts = {"canonical_native_and_public": 0,
                  "structural_native_but_public_guarded": 0,
                  "not_structural_native_explicit_host_transport": 0}
        for row in enumerate_rows():
            counts[category({"lateral": row["lateral"], "bottom": row["bottom"],
                             "top": row["top"]})] += 1
        self.assertEqual(counts, {"canonical_native_and_public": 1,
                                  "structural_native_but_public_guarded": 1,
                                  "not_structural_native_explicit_host_transport": 1534})


if __name__ == "__main__":
    unittest.main()

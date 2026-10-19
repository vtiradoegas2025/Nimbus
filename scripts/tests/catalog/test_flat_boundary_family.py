#!/usr/bin/env python3
"""Stable-ID and conditional-class accounting for the flat boundary inventory."""

from __future__ import annotations

import unittest

from scripts.qualification.enumerate_flat_boundary_family import (
    BASE, BOTTOM, FACES, LATERAL, TOP, enumerate_rows, evidence_classes, family_id,
    physical_id, variant,
)


class FlatBoundaryFamilyTests(unittest.TestCase):
    def test_complete_nonperiodic_categorical_set_and_physical_closures(self) -> None:
        rows = enumerate_rows()
        self.assertEqual(len(rows), len(LATERAL) ** len(FACES) * len(BOTTOM) * len(TOP))
        self.assertEqual(len({row["id"] for row in rows}), 1536)
        self.assertEqual(len({row["physicalBoundaryClosureId"] for row in rows}), 1024)

    def test_rigid_with_sponge_is_not_a_public_alias_when_damping_is_off(self) -> None:
        laterals = ("rigid", "radiative", "zero_gradient", "rigid_slip")
        self.assertNotEqual(family_id(laterals, "rigid_slip", "rigid"),
                            family_id(laterals, "rigid_slip", "rigid_with_sponge"))
        self.assertEqual(physical_id(laterals, "rigid_slip", "rigid"),
                         physical_id(laterals, "rigid_slip", "rigid_with_sponge"))
        self.assertNotEqual(physical_id(laterals, "rigid_slip", "open"),
                            physical_id(laterals, "rigid_slip", "rigid"))

    def test_variant_preserves_fixed_solver_and_changes_exact_boundary_keys(self) -> None:
        source = BASE.read_text()
        output = variant(source, ("radiative", "rigid_slip", "zero_gradient", "rigid"),
                         "rigid_noslip", "open", "gpu")
        self.assertIn("    west: radiative\n", output)
        self.assertIn("    east: rigid_slip\n", output)
        self.assertIn("  bottom: rigid_noslip\n", output)
        self.assertIn("  top: open\n", output)
        self.assertIn("    backend: gpu\n", output)
        self.assertIn("  scheme: tvd\n", output)
        with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
            variant(source.replace("    west: rigid\n", "    west: periodic\n"),
                    ("radiative", "rigid", "rigid", "rigid"), "rigid_slip", "rigid", "cpu")

    def test_mixed_case_requires_corner_and_radiative_history_evidence(self) -> None:
        classes = evidence_classes(("rigid", "rigid_slip", "radiative", "zero_gradient"),
                                   "rigid_noslip", "rigid_with_sponge")
        self.assertIn("mixed_lateral_corner", classes)
        self.assertIn("lateral_radiative_history", classes)
        self.assertIn("bottom_no_slip", classes)
        self.assertIn("separate_top_sponge_inactive", classes)
        self.assertNotIn("top_open_flux", classes)


if __name__ == "__main__":
    unittest.main()

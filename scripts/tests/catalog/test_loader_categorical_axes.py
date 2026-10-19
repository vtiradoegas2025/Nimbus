#!/usr/bin/env python3
"""Mutation checks for strict-loader categorical-axis discovery."""

from __future__ import annotations

import unittest

from scripts.qualification.audit_loader_categorical_axes import alias_disposition, literal_axes


class LoaderCategoricalAxisTests(unittest.TestCase):
    def test_added_selector_changes_the_reconciled_axis_set(self) -> None:
        one = 'take_enum("first.choice", value, parse_first, "alpha, beta");'
        two = one + '\ntake_enum("second.choice", value, parse_second, "none, active");'
        self.assertEqual(len(literal_axes(one, minimum_axes=1, include_special=False)), 1)
        self.assertEqual(len(literal_axes(two, minimum_axes=1, include_special=False)), 2)

    def test_duplicate_conflicting_domain_rejects(self) -> None:
        source = ('take_enum("same.choice", value, parse_choice, "alpha, beta");\n'
                  'take_enum("same.choice", value, parse_choice, "alpha, gamma");')
        with self.assertRaisesRegex(RuntimeError, "conflicting domains"):
            literal_axes(source, minimum_axes=1, include_special=False)

    def test_malformed_allowed_value_rejects(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            literal_axes('take_enum("bad.choice", value, parse_choice, "alpha, bad value");',
                         minimum_axes=1, include_special=False)

    def test_coordinate_and_staggering_domains_cannot_disappear(self) -> None:
        source = ('take_coordinate_system("grid.coordinate_system", grid.coordinate_system);\n'
                  'take_stagger_type("grid.staggering", grid.staggering);\n'
                  'expected: cartesian\nexpected one of: collocated, c_grid')
        axes = literal_axes(source, minimum_axes=2)
        self.assertEqual(axes["grid.coordinate_system"]["allowed"], ["cartesian"])
        self.assertEqual(axes["grid.staggering"]["allowed"], ["collocated", "c_grid"])
        with self.assertRaisesRegex(RuntimeError, "special categorical domain changed"):
            literal_axes(source.replace("collocated, c_grid", "c_grid"), minimum_axes=2)

    def test_legacy_open_is_conditional_not_a_new_physical_branch(self) -> None:
        disposition = alias_disposition("boundary_conditions.lateral.west", "open")
        self.assertEqual(disposition["targetByStaggering"],
                         {"c_grid": "radiative", "collocated": "zero_gradient"})
        self.assertEqual(alias_disposition("extensions.compute.backend", "vk")["physicalBackend"],
                         "vulkan")
        self.assertIsNone(alias_disposition("boundary_conditions.top", "open"))


if __name__ == "__main__":
    unittest.main()

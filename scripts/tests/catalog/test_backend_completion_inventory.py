#!/usr/bin/env python3
"""Mutation checks for retained-inventory declaration accounting."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.qualification.backend_completion_inventory import (
    explicit_yaml_scalars,
    obligations,
    selector_tokens,
    unique,
)


class InventoryIntegrityTests(unittest.TestCase):
    def test_duplicate_identifier_rejects(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicate retained identifiers"):
            unique([{"id": "kept"}, {"id": "kept"}])

    def test_missing_plan_obligation_rejects(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "register coverage drifted"):
            obligations("| SE-001 | retained |\n", "SE", 2)

    def test_duplicate_plan_obligation_rejects(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicate retained identifiers"):
            obligations("| CAT-001 | retained |\n| CAT-001 | duplicate |\n", "CAT", 1)

    def test_duplicate_explicit_config_key_rejects(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus-inventory-mutant-") as temporary:
            path = Path(temporary) / "duplicate.yaml"
            path.write_text("time:\n  dt_large: 0.1\n  dt_large: 0.2\n")
            with self.assertRaisesRegex(RuntimeError, "duplicate explicit YAML scalar"):
                explicit_yaml_scalars(path)

    def test_distinct_nested_config_keys_preserve_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus-inventory-nested-") as temporary:
            path = Path(temporary) / "nested.yaml"
            path.write_text("time:\n  dt_large: 0.1\noutput:\n  format: netcdf\n")
            self.assertEqual(explicit_yaml_scalars(path),
                             {"time.dt_large": "0.1", "output.format": "netcdf"})

    def test_product_set_aliases_are_individual_tokens(self) -> None:
        self.assertEqual(selector_tokens({"[]", "[reflectivity_dbz, radial_velocity]"},
                                         "product_set_v1"),
                         ({"reflectivity_dbz", "radial_velocity"}, set()))

    def test_malformed_product_set_requires_review(self) -> None:
        self.assertEqual(selector_tokens({"[reflectivity_dbz,, velocity]"},
                                         "product_set_v1"),
                         (set(), {"[reflectivity_dbz,, velocity]"}))


if __name__ == "__main__":
    unittest.main()

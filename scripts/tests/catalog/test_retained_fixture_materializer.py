#!/usr/bin/env python3
"""Bounded conditional-selector and inactive-branch checks for I1 materialization."""

from __future__ import annotations

import unittest

from scripts.qualification.materialize_retained_fixtures import (
    conditional_activity,
    ordered_product_ids,
    physical_backend,
    scalar_identity,
)


CATALOG = {"modules": [{
    "id": "advection", "selectionPath": "advection.scheme",
    "studioSelection": {"mode": "form"},
    "schemes": [{"id": "none", "aliases": []},
                {"id": "tvd", "aliases": []},
                {"id": "weno5", "aliases": ["weno"]}],
}]}


class RetainedFixtureMaterializerTests(unittest.TestCase):
    def test_auto_cgrid_resolves_to_weno5(self) -> None:
        selected, unresolved = scalar_identity(
            CATALOG, {"advection.scheme": "auto"},
            {"advection.scheme": "auto", "grid.staggering": "c_grid"})
        self.assertEqual(selected["advection.scheme"]["canonicalId"], "weno5")
        self.assertEqual(unresolved, [])

    def test_auto_collocated_resolves_to_tvd(self) -> None:
        selected, unresolved = scalar_identity(
            CATALOG, {}, {"advection.scheme": "auto", "grid.staggering": "collocated"})
        self.assertEqual(selected["advection.scheme"]["canonicalId"], "tvd")
        self.assertEqual(unresolved, [])

    def test_auto_unknown_staggering_remains_unresolved(self) -> None:
        selected, unresolved = scalar_identity(
            CATALOG, {}, {"advection.scheme": "auto", "grid.staggering": "unknown"})
        self.assertNotIn("canonicalId", selected["advection.scheme"])
        self.assertEqual(unresolved, ["advection.scheme"])

    def test_inactive_nested_and_tracking_settings_do_not_define_a_family(self) -> None:
        inactive = conditional_activity({
            "domain_motion.lifetime_planning.enabled": "false",
            "domain_motion.lifetime_planning.action": "warn",
            "domain_motion.nesting.enabled": "false",
            "domain_motion.nesting.feedback": "two_way",
            "domain_motion.tracking.enabled": "false",
            "domain_motion.tracking.interval_s": "5",
        })
        self.assertEqual(inactive["domain_motion.nesting.feedback"], "nesting is disabled")
        self.assertEqual(inactive["domain_motion.lifetime_planning.action"],
                         "lifetime planning is disabled")
        self.assertEqual(inactive["domain_motion.tracking.interval_s"], "tracking is disabled")
        self.assertNotIn("domain_motion.nesting.enabled", inactive)
        self.assertNotIn("domain_motion.lifetime_planning.enabled", inactive)
        self.assertNotIn("domain_motion.tracking.enabled", inactive)

    def test_generic_gpu_is_the_vulkan_alias(self) -> None:
        self.assertEqual(physical_backend("gpu"), "vulkan")
        self.assertEqual(physical_backend("vulkan"), "vulkan")
        self.assertEqual(physical_backend("metal"), "metal")
        with self.assertRaisesRegex(RuntimeError, "unknown compute backend"):
            physical_backend("any_accelerator")

    def test_polarimetric_products_keep_distinct_ids_under_one_operator(self) -> None:
        radar = {"modules": [{
            "selectionPath": "radar.products", "studioSelection": {"mode": "product_set_v1"},
            "schemes": [{"id": "reflectivity", "aliases": ["reflectivity_dbz"]},
                        {"id": "velocity", "aliases": ["radial_velocity"]},
                        {"id": "zdr", "aliases": ["kdp", "rhohv"]}],
        }]}
        selected, unresolved = scalar_identity(
            radar, {}, {"radar.products": "[zdr, kdp, rhohv]"})
        self.assertEqual(selected["radar.products"]["productIds"], ["zdr", "kdp", "rhohv"])
        self.assertEqual(selected["radar.products"]["descriptorOwnerIds"], ["zdr"] * 3)
        self.assertEqual(unresolved, [])

    def test_duplicate_product_id_rejects(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicate product"):
            ordered_product_ids("[zdr, zdr]")

    def test_unmatched_product_alias_rejects(self) -> None:
        radar = {"modules": [{
            "selectionPath": "radar.products", "studioSelection": {"mode": "product_set_v1"},
            "schemes": [{"id": "reflectivity", "aliases": ["reflectivity_dbz"]}],
        }]}
        with self.assertRaisesRegex(RuntimeError, "no unique catalog descriptor owner"):
            scalar_identity(radar, {}, {"radar.products": "[unregistered_product]"})


if __name__ == "__main__":
    unittest.main()

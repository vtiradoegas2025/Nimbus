"""Mutation checks for generated retained structural-record provenance."""

import copy
import unittest

from scripts.qualification import join_retained_conditional_combinations as join


def fixture():
    sources = []
    for source_id, owner, config, membership, terrain in (
            ("source:a", "one", "config:1", "retained_maintained_execution_fixture", False),
            ("source:b", "one", "config:2", "provisional_canonical_first_party_pending_review", False),
            ("source:alias", "one", "config:2", "provisional_canonical_first_party_pending_review", False),
            ("source:c", "two", "config:3", "provisional_canonical_first_party_pending_review", True)):
        signature = {"scheme": owner, "terrain": terrain}
        sources.append({"id": source_id, "canonicalStructuralId": join.stable_id(
            "conditional-combination", signature), "structuralSignature": signature,
            "exactConfigId": config, "fullProductMembership": membership,
            "terrainDeferred": terrain})
    return sources, join.structural_groups(sources)


def validate(groups, sources):
    validator = getattr(join, "validate_structural_provenance", join.validate_structural_records)
    validator(groups, sources)


class RetainedStructuralProvenanceTests(unittest.TestCase):
    def test_generated_metadata_matches_source_owner_and_alias_deduplication(self):
        sources, groups = fixture()
        validate(groups, sources)
        one = next(group for group in groups if group["signature"]["scheme"] == "one")
        two = next(group for group in groups if group["signature"]["scheme"] == "two")
        self.assertEqual(one["sourceIntentCount"], 3)
        self.assertEqual(one["exactConfigCount"], 2)
        self.assertEqual(one["exactConfigIds"], ["config:1", "config:2"])
        self.assertTrue(one["fullProductMember"])
        self.assertFalse(two["fullProductMember"])
        self.assertEqual(one["proposedBetaMembership"], "undecided_requires_product_scope_join")

    def test_exact_config_orphan_duplicate_deletion_and_order_mutations_reject(self):
        sources, groups = fixture()
        index = next(index for index, group in enumerate(groups) if group["signature"]["scheme"] == "one")
        for value in (["config:orphan"], ["config:1"], ["config:1", "config:1", "config:2"],
                      ["config:2", "config:1"], ["config:1", "config:2", "config:3"]):
            changed = copy.deepcopy(groups)
            changed[index]["exactConfigIds"] = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                validate(changed, sources)

    def test_source_and_exact_config_counts_cannot_be_forged(self):
        sources, groups = fixture()
        for key in ("sourceIntentCount", "exactConfigCount"):
            for value in (0, 99, True, "3"):
                changed = copy.deepcopy(groups)
                changed[0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    validate(changed, sources)

    def test_provisional_group_cannot_promote_retained_membership(self):
        sources, groups = fixture()
        index = next(index for index, group in enumerate(groups) if group["signature"]["scheme"] == "two")
        for key, value in (("fullProductMember", True), ("fullProductMember", 0),
                           ("membershipClasses", ["retained_maintained_execution_fixture"]),
                           ("membershipClasses", [])):
            changed = copy.deepcopy(groups)
            changed[index][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate(changed, sources)

    def test_retained_group_cannot_hide_source_membership(self):
        sources, groups = fixture()
        index = next(index for index, group in enumerate(groups) if group["signature"]["scheme"] == "one")
        for key, value in (("fullProductMember", False), ("fullProductMember", 1),
                           ("membershipClasses", ["retained_maintained_execution_fixture"])):
            changed = copy.deepcopy(groups)
            changed[index][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate(changed, sources)

    def test_terrain_provenance_and_source_order_are_bound(self):
        sources, groups = fixture()
        for index in range(len(groups)):
            for value in (not groups[index]["terrainDeferred"], None, 0, "false"):
                changed = copy.deepcopy(groups)
                changed[index]["terrainDeferred"] = value
                with self.subTest(index=index, value=value), self.assertRaises(RuntimeError):
                    validate(changed, sources)
        changed = copy.deepcopy(groups)
        index = next(index for index, group in enumerate(groups) if len(group["sourceIds"]) > 1)
        changed[index]["sourceIds"].reverse()
        with self.assertRaises(RuntimeError):
            validate(changed, sources)

    def test_missing_generated_metadata_rejects(self):
        sources, groups = fixture()
        for key in ("exactConfigIds", "sourceIntentCount", "exactConfigCount", "membershipClasses",
                    "fullProductMember", "terrainDeferred"):
            changed = copy.deepcopy(groups)
            del changed[0][key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate(changed, sources)

    def test_source_metadata_change_cannot_silently_rebind_existing_groups(self):
        sources, groups = fixture()
        for key, value in (("exactConfigId", "config:orphan"),
                           ("fullProductMembership", "retained_unreviewed_product_choice"),
                           ("fullProductMembership", "retained_current_public_profile_fixture"),
                           ("terrainDeferred", True), ("terrainDeferred", 0)):
            changed = copy.deepcopy(sources)
            changed[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                validate(groups, changed)

    def test_minimal_signature_validator_contract_remains_available(self):
        signature = {"scheme": "minimal"}
        ident = join.stable_id("conditional-combination", signature)
        join.validate_structural_records([{"id": ident, "signature": signature, "sourceIds": ["source:a"]}],
            [{"id": "source:a", "canonicalStructuralId": ident, "structuralSignature": signature}])


if __name__ == "__main__":
    unittest.main()

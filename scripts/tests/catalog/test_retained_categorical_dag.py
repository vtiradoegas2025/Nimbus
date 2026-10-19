from __future__ import annotations

import copy
import unittest

from scripts.qualification import retained_categorical_dag as dag


class ConditionalDagContractTests(unittest.TestCase):
    def test_six_outcomes_and_authoritative_negative_reasons(self) -> None:
        self.assertEqual(len(dag.OutcomeState), 6)
        for state in dag.OutcomeState:
            if state == dag.OutcomeState.STRUCTURAL_CANDIDATE:
                with self.assertRaisesRegex(ValueError, "negative retained intent"):
                    dag.negative_intent("x", "a", {}, state, "reason", "source")
                continue
            task = "retained-task" if state == dag.OutcomeState.RETAINED_GUARDED_OR_UNIMPLEMENTED else ""
            record = dag.negative_intent("x", state.value, {}, state,
                                         "source-reviewed reason", "owner:function", task)
            self.assertTrue(record["id"].startswith("raw-categorical-intent:"))
            self.assertEqual(record["outcome"], state.value)
            with self.assertRaisesRegex(ValueError, "disposition authority"):
                dag.negative_intent("x", state.value, {}, state, "", "", task)
        with self.assertRaisesRegex(ValueError, "disposition authority"):
            dag.negative_intent("x", "true", {},
                                dag.OutcomeState.RETAINED_GUARDED_OR_UNIMPLEMENTED,
                                "guarded", "owner:function")

    def test_conditional_axis_elides_inactive_owner_and_keeps_alias(self) -> None:
        axes = (
            dag.Axis("scheme", (dag.Choice("none", ("none",), "loader"),
                                dag.Choice("milbrandt", ("milbrandt",), "loader")), "loader"),
            dag.Axis("hail", (dag.Choice("false", ("false",), "loader"),
                              dag.Choice("true", ("true", "<omitted>"), "config")),
                     "loader", (dag.Condition("scheme", "milbrandt"),),
                     "inactive outside Milbrandt"),
        )
        leaves = dag.enumerate_assignments(axes)
        self.assertEqual(len(leaves), 3)
        self.assertNotIn("hail", leaves[0]["selected"])
        self.assertEqual(leaves[0]["inactive"][0]["reason"],
                         "inactive outside Milbrandt")
        self.assertEqual(dag.alias_links(axes)[0]["raw"], "<omitted>")

    def test_duplicate_alias_and_unordered_dependency_fail_closed(self) -> None:
        duplicate = dag.Axis("x", (dag.Choice("a", ("a", "alias"), "source"),
                                   dag.Choice("b", ("b", "alias"), "source")), "source")
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            dag.enumerate_assignments((duplicate,))
        late = dag.Axis("x", (dag.Choice("a", ("a",), "source"),),
                        "source", (dag.Condition("missing", "a"),), "inactive")
        with self.assertRaisesRegex(ValueError, "activation dependency"):
            dag.enumerate_assignments((late,))

    def test_non_candidate_leaf_requires_reason_and_authority(self) -> None:
        signature = {"axis": "a"}
        accepted = dag.leaf(signature, {}, [], dag.OutcomeState.STRUCTURAL_CANDIDATE,
                            "structural_only")
        self.assertEqual(accepted["id"], dag.stable_id("conditional-combination", signature))
        with self.assertRaisesRegex(ValueError, "rejection authority"):
            dag.leaf(signature, {}, [], dag.OutcomeState.POST_LOAD_REJECTED,
                     "postload_only")

    def test_disjoint_shard_member_count_and_digest_mutations(self) -> None:
        leaves = [{"id": "conditional-combination:" + str(index), "value": index}
                  for index in range(12)]
        shards = dag.partition(leaves, 2)
        self.assertEqual(sum(row["count"] for row in shards), 12)
        dag.verify_partition(leaves, shards)
        skipped = copy.deepcopy(shards[:1])
        with self.assertRaisesRegex(ValueError, "skipped"):
            dag.verify_partition(leaves, skipped, 2)
        deleted = copy.deepcopy(shards)
        target = next(row for row in deleted if row["records"])
        target["records"].pop()
        target["ids"] = [row["id"] for row in target["records"]]
        target["count"] = len(target["records"])
        target["recordsSha256"] = dag.digest(target["records"])
        with self.assertRaisesRegex(ValueError, "deleted or orphaned"):
            dag.verify_partition(leaves, deleted)
        duplicate = copy.deepcopy(shards)
        target = next(row for row in duplicate if row["records"])
        target["records"].append(copy.deepcopy(target["records"][0]))
        target["records"].sort(key=lambda row: row["id"])
        target["ids"] = [row["id"] for row in target["records"]]
        target["count"] = len(target["records"])
        target["recordsSha256"] = dag.digest(target["records"])
        with self.assertRaisesRegex(ValueError, "membership"):
            dag.verify_partition(leaves, duplicate)
        drift = copy.deepcopy(shards)
        drift[0]["recordsSha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest drifted"):
            dag.verify_partition(leaves, drift)


if __name__ == "__main__":
    unittest.main()

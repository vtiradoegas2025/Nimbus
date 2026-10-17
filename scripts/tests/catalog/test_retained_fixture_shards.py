#!/usr/bin/env python3
"""Mutations that must prevent a complete retained-fixture shard claim."""

from __future__ import annotations

from copy import deepcopy
import unittest

from scripts.qualification.merge_retained_fixture_shards import validate_shard_headers


def headers() -> list[dict]:
    return [{"schema": "nimbus.retained-fixture-combination-shard.v1",
             "status": "complete", "shardIndex": index, "shardCount": 2,
             "sourceCount": 4, "inputs": {"binary": {"sha256": "current"}},
             "catalogSha256": "catalog"} for index in range(2)]


class RetainedFixtureShardMutationTests(unittest.TestCase):
    def test_complete_headers_pass(self) -> None:
        self.assertEqual(validate_shard_headers(headers())["shardCount"], 2)

    def test_skipped_shard_rejects(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "skipped or extra shard"):
            validate_shard_headers(headers()[:1])

    def test_duplicate_shard_rejects(self) -> None:
        mutant = headers()
        mutant[1]["shardIndex"] = 0
        with self.assertRaisesRegex(RuntimeError, "missing or duplicated"):
            validate_shard_headers(mutant)

    def test_stale_binary_identity_rejects(self) -> None:
        mutant = deepcopy(headers())
        mutant[1]["inputs"]["binary"]["sha256"] = "stale"
        with self.assertRaisesRegex(RuntimeError, "source identities differ"):
            validate_shard_headers(mutant)

    def test_partial_shard_rejects(self) -> None:
        mutant = headers()
        mutant[1]["status"] = "partial"
        with self.assertRaisesRegex(RuntimeError, "schema or source identities differ"):
            validate_shard_headers(mutant)


if __name__ == "__main__":
    unittest.main()

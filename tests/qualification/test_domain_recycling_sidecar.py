from __future__ import annotations

import copy
import unittest

from scripts.qualification.domain_motion.qualify_domain_recycling import (
    require_swath_brick_hierarchy,
)


def valid_hierarchy() -> dict:
    return {
        "schema": "nimbus.field-brick-hierarchy",
        "schemaVersion": 2,
        "field": "p",
        "publicationGeneration": "a" * 32,
        "sourceShape": [4, 16, 24],
        "sourceIndexOffset": [0, 0, 0],
        "sourceDtype": "float32",
        "sampleLocation": "cell_center",
        "cullingSafe": True,
        "levels": [{
            "level": 0,
            "stride": [1, 1, 1],
            "shape": [4, 16, 24],
            "source": {"kind": "primary_field"},
            "bricks": [{
                "contentSha256": "b" * 64,
                "scalarRange": {"finiteCount": 1536, "nonFiniteCount": 0},
                "physicalBoundsM": {
                    "lower": [0, 0, 0],
                    "upper": [4000, 16000, 12000],
                },
            }],
        }],
    }


class DomainRecyclingSidecarTests(unittest.TestCase):
    def test_accepts_exact_flat_lod_sidecar(self) -> None:
        require_swath_brick_hierarchy(valid_hierarchy(), "p", "test")

    def test_rejects_legacy_schema_and_wrong_field(self) -> None:
        legacy = valid_hierarchy()
        legacy["schemaVersion"] = 1
        with self.assertRaisesRegex(RuntimeError, "hierarchy schema"):
            require_swath_brick_hierarchy(legacy, "p", "test")
        wrong_field = valid_hierarchy()
        wrong_field["field"] = "theta"
        with self.assertRaisesRegex(RuntimeError, "brick field"):
            require_swath_brick_hierarchy(wrong_field, "p", "test")

    def test_rejects_incomplete_level_and_content_identity(self) -> None:
        wrong_source = valid_hierarchy()
        wrong_source["levels"][0]["source"] = {"kind": "npy_lod"}
        with self.assertRaisesRegex(RuntimeError, "primary brick level"):
            require_swath_brick_hierarchy(wrong_source, "p", "test")
        missing_identity = copy.deepcopy(valid_hierarchy())
        missing_identity["levels"][0]["bricks"][0]["contentSha256"] = "short"
        with self.assertRaisesRegex(RuntimeError, "content identity"):
            require_swath_brick_hierarchy(missing_identity, "p", "test")


if __name__ == "__main__":
    unittest.main()

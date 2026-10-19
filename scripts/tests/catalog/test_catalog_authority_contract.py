from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import unittest

from scripts.tests.catalog.test_catalog_authority import fingerprint, validated_readiness


ROOT = Path(__file__).resolve().parents[3]


class CatalogReadinessSealTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = json.loads(
            subprocess.check_output(
                [str(ROOT / "bin/nimbus_sim"), "--scheme-catalog=json"],
                cwd=ROOT,
                text=True,
            )
        )

    def test_state_mutation_changes_readiness_fingerprint(self) -> None:
        mutated = copy.deepcopy(self.catalog)
        original = fingerprint(validated_readiness(mutated))
        item = mutated["readiness"]["items"][0]
        item["state"] = "engineering_incomplete"
        self.assertNotEqual(original, fingerprint(validated_readiness(mutated)))

    def test_evidence_mutation_changes_readiness_fingerprint(self) -> None:
        mutated = copy.deepcopy(self.catalog)
        original = fingerprint(validated_readiness(mutated))
        item = mutated["readiness"]["items"][0]
        item["boundedEvidence"] = ""
        self.assertNotEqual(original, fingerprint(validated_readiness(mutated)))

    def test_missing_registry_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.catalog)
        del mutated["readiness"]
        with self.assertRaisesRegex(RuntimeError, "readiness registry"):
            validated_readiness(mutated)


if __name__ == "__main__":
    unittest.main()

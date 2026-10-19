from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/tests/catalog/test_native_full_preflight.py"
SPEC = importlib.util.spec_from_file_location("native_full_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class NativeFullPreflightTests(unittest.TestCase):
    def evidence(self) -> dict[str, object]:
        return json.loads((ROOT / "reports/pre_beta/native_full_preflight_evidence_v1.json").read_text())

    def test_current_catalog_is_exact(self) -> None:
        PREFLIGHT.validate_catalog(self.evidence()["catalog"])

    def test_current_evidence_verifies(self) -> None:
        PREFLIGHT.verify_evidence(self.evidence(), PREFLIGHT.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_preflight_rejection_mutation_fails_closed(self) -> None:
        evidence = self.evidence()
        evidence["catalog"]["summary"]["preflightRejected"] = 1
        with self.assertRaisesRegex(RuntimeError, "summary drifted"):
            PREFLIGHT.verify_evidence(evidence, PREFLIGHT.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_check_count_mutation_fails_closed(self) -> None:
        evidence = self.evidence()
        evidence["catalog"]["checks"]["projection"] -= 1
        with self.assertRaisesRegex(RuntimeError, "check counts drifted"):
            PREFLIGHT.verify_evidence(evidence, PREFLIGHT.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_partition_mutation_fails_closed(self) -> None:
        evidence = self.evidence()
        evidence["catalog"]["partitions"]["c_grid.cpu.exploratory"] -= 1
        with self.assertRaisesRegex(RuntimeError, "partitions drifted"):
            PREFLIGHT.verify_evidence(evidence, PREFLIGHT.sha256_file(ROOT / "bin/nimbus_sim"))


if __name__ == "__main__":
    unittest.main()

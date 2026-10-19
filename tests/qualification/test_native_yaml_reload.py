from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/tests/catalog/qualify_native_yaml_reload.py"
SPEC = importlib.util.spec_from_file_location("native_yaml_reload", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RELOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELOAD)


class NativeYamlReloadTests(unittest.TestCase):
    def evidence(self) -> dict[str, object]:
        return json.loads((ROOT / "reports/pre_beta/native_yaml_reload_v1.json").read_text())

    def test_current_evidence_verifies(self) -> None:
        RELOAD.verify_evidence(self.evidence(), RELOAD.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_summary_mutation_fails_closed(self) -> None:
        evidence = self.evidence()
        evidence["summary"]["passed"] -= 1
        with self.assertRaisesRegex(RuntimeError, "summary drifted"):
            RELOAD.verify_evidence(evidence, RELOAD.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_representative_mutation_fails_closed(self) -> None:
        evidence = self.evidence()
        evidence["representativeResults"][0]["status"] = "skip"
        with self.assertRaisesRegex(RuntimeError, "not passing"):
            RELOAD.verify_evidence(evidence, RELOAD.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_aggregate_mutation_fails_closed(self) -> None:
        evidence = self.evidence()
        evidence["representativeResults"][0]["yamlSha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "aggregate drifted"):
            RELOAD.verify_evidence(evidence, RELOAD.sha256_file(ROOT / "bin/nimbus_sim"))


if __name__ == "__main__":
    unittest.main()

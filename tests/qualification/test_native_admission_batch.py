from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/tests/catalog/test_native_admission_batch.py"
SPEC = importlib.util.spec_from_file_location("native_admission", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADMISSION)


class NativeAdmissionBatchTests(unittest.TestCase):
    def evidence(self) -> dict[str, object]:
        return json.loads((ROOT / "reports/pre_beta/native_admission_evidence_v1.json").read_text())

    def test_current_catalog_is_exact(self) -> None:
        ADMISSION.validate_catalog(self.evidence()["catalog"])

    def test_current_evidence_verifies(self) -> None:
        ADMISSION.verify_evidence(self.evidence(), ADMISSION.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_admitted_hash_mutation_fails_closed(self) -> None:
        evidence = self.evidence()
        evidence["catalog"]["admittedTupleSha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "admitted tuple identity drifted"):
            ADMISSION.verify_evidence(evidence, ADMISSION.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_partition_mutation_fails_closed(self) -> None:
        evidence = self.evidence()
        evidence["catalog"]["partitions"]["c_grid.cpu.qualified"] -= 1
        with self.assertRaisesRegex(RuntimeError, "partitions drifted"):
            ADMISSION.verify_evidence(evidence, ADMISSION.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_rejection_mutation_fails_closed(self) -> None:
        evidence = self.evidence()
        evidence["catalog"]["rejections"][0]["count"] += 1
        with self.assertRaisesRegex(RuntimeError, "rejection counts drifted"):
            ADMISSION.verify_evidence(evidence, ADMISSION.sha256_file(ROOT / "bin/nimbus_sim"))


if __name__ == "__main__":
    unittest.main()

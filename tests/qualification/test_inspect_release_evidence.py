from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.qualification.provenance.inspect_release_evidence import (
    inspect_manifest,
    inspect_receipt,
    sha256_file,
)
from scripts.qualification.provenance.record_gate_result import SCHEMA


class InspectReleaseEvidenceTests(unittest.TestCase):
    def test_receipt_classifies_current_stale_and_missing_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "gate.log"
            log.write_text("PASS\n", encoding="utf-8")
            receipt = root / "gate.json"
            payload = {
                "schema": SCHEMA,
                "target": "test-catalog-authority",
                "binarySha256": "a" * 64,
                "status": "pass",
                "log": "gate.log",
                "logSha256": sha256_file(log),
            }
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                inspect_receipt(
                    receipt, "test-catalog-authority", "a" * 64, root,
                )["classification"],
                "current",
            )
            self.assertEqual(
                inspect_receipt(
                    receipt, "test-catalog-authority", "b" * 64, root,
                )["classification"],
                "stale",
            )
            self.assertEqual(
                inspect_receipt(
                    root / "missing.json", "test-catalog-authority", "a" * 64,
                    root,
                )["classification"],
                "missing",
            )

    def test_receipt_rejects_a_changed_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "gate.log"
            log.write_text("PASS\n", encoding="utf-8")
            receipt = root / "gate.json"
            receipt.write_text(json.dumps({
                "schema": SCHEMA,
                "target": "qualify-io",
                "binarySha256": "c" * 64,
                "status": "pass",
                "log": "gate.log",
                "logSha256": sha256_file(log),
            }), encoding="utf-8")
            log.write_text("changed\n", encoding="utf-8")
            inspected = inspect_receipt(receipt, "qualify-io", "c" * 64, root)
            self.assertEqual(inspected["classification"], "invalid")
            self.assertIn("log identity", str(inspected["reason"]))

    def test_manifest_reports_stale_before_full_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "candidate.json"
            manifest.write_text(json.dumps({
                "schema": "nimbus.release_provenance",
                "version": 2,
                "binary": {"sha256": "d" * 64},
            }), encoding="utf-8")
            inspected = inspect_manifest(manifest, "e" * 64, root)
            self.assertEqual(inspected["classification"], "stale")


if __name__ == "__main__":
    unittest.main()

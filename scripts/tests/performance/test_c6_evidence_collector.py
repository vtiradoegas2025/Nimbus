#!/usr/bin/env python3
"""Collection identity and failure-atomicity checks with temporary evidence."""

from __future__ import annotations

import copy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.qualification import collect_c6_evidence as evidence
from scripts.qualification.provenance import release_manifest


class C6EvidenceCollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        for relative, payload in (
            ("mk/tests.mk", b"canonical source declaration"), ("src/physics.cpp", b"physics"),
            ("src/physics.metal", b"metal"), ("src/physics.comp", b"vulkan"),
            ("bin/nimbus_sim", b"binary"), ("configs/bounded.yaml", b"config"),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        self.roots = [Path("mk"), Path("src")]
        self.host = evidence.identity_record({
            "os": "test-os", "architecture": "test-architecture", "cpuModel": "test-cpu",
            "logicalThreads": 8, "ramBytes": 64_000_000, "hostname": "test-host",
        }, evidence.HOST_SCHEMA)
        self.device = evidence.identity_record({"name": "test-GPU", "driver": "test-driver",
                                                "backendVersion": "test-version"}, evidence.DEVICE_SCHEMA)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def snapshot(self) -> dict:
        with mock.patch.object(evidence, "capture_host", return_value=self.host):
            return evidence.capture_snapshot(self.root, self.root / "bin/nimbus_sim",
                                             self.root / "configs/bounded.yaml", device=self.device,
                                             _roots=self.roots)

    def trial(self, snapshot: dict) -> dict:
        return {**evidence.trial_identity(snapshot, "metal", [12, 12, 12]), "index": 1}

    def test_complete_inventory_covers_non_cpp_sources(self) -> None:
        snapshot = self.snapshot()
        files = snapshot["source"]["inventory"]["files"]
        self.assertEqual([record["path"] for record in files],
                         ["mk/tests.mk", "src/physics.comp", "src/physics.cpp", "src/physics.metal"])
        self.assertEqual(snapshot["source"]["inventory"]["sha256"], evidence.fingerprint(files))
        self.assertEqual(evidence.source_seal(snapshot["source"])["fileCount"], 4)

    def test_stable_snapshots_seal_and_do_not_mutate_input(self) -> None:
        before = self.snapshot()
        trial = self.trial(before)
        original = copy.deepcopy(trial)
        measurement = evidence.measurement_record(trial, before, self.snapshot())
        self.assertEqual(trial, original)
        self.assertEqual(measurement["trial"]["identityBefore"], measurement["trial"]["identityAfter"])
        self.assertEqual(measurement["snapshotBefore"], measurement["snapshotAfter"])

    def test_binary_config_shader_and_new_source_drift_reject(self) -> None:
        for relative, key in (("bin/nimbus_sim", "binary"), ("configs/bounded.yaml", "config"),
                              ("src/physics.metal", "source"), ("src/new.comp", "source")):
            with self.subTest(relative=relative):
                before = self.snapshot()
                path = self.root / relative
                prior = path.read_bytes() if path.exists() else None
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(evidence.C6EvidenceError, f"{key} drift"):
                    evidence.measurement_record(self.trial(before), before, self.snapshot())
                if prior is None:
                    path.unlink()
                else:
                    path.write_bytes(prior)

    def test_host_and_device_drift_reject(self) -> None:
        before = self.snapshot()
        self.assertEqual(before["device"]["evidenceKind"], "supplied_identity_artifact")
        for key in ("host", "device"):
            with self.subTest(key=key):
                after = copy.deepcopy(before)
                after[key]["fingerprintSha256"] = "a" * 64
                with self.assertRaisesRegex(evidence.C6EvidenceError, f"{key} drift"):
                    evidence.measurement_record(self.trial(before), before, after)

    def test_trial_cannot_relabel_snapshot_identity(self) -> None:
        snapshot = self.snapshot()
        trial = self.trial(snapshot)
        trial["sourceInventorySha256"] = "a" * 64
        with self.assertRaisesRegex(evidence.C6EvidenceError, "trial identity differs from snapshot"):
            evidence.measurement_record(trial, snapshot, snapshot)
        trial = self.trial(snapshot)
        trial["identityBefore"] = {**evidence.trial_identity(snapshot, "metal", [12, 12, 12]),
                                   "binarySha256": "b" * 64}
        with self.assertRaisesRegex(evidence.C6EvidenceError, "supplied trial before/after identity differs"):
            evidence.measurement_record(trial, snapshot, snapshot)

    def test_forged_fingerprint_and_fractional_host_capacity_reject(self) -> None:
        host = copy.deepcopy(self.host)
        host["fingerprintSha256"] = "a" * 64
        with self.assertRaisesRegex(evidence.C6EvidenceError, "fingerprint does not match identity"):
            evidence.validate_identity(host, evidence.HOST_SCHEMA)
        host["identity"]["logicalThreads"] = 1.5
        host["fingerprintSha256"] = evidence.fingerprint(host["identity"])
        with self.assertRaisesRegex(evidence.C6EvidenceError, "invalid host capacity"):
            evidence.validate_identity(host, evidence.HOST_SCHEMA)

    def test_generated_cache_guard_preserves_file(self) -> None:
        path = self.root / "src/__pycache__/physics.pyc"
        path.parent.mkdir()
        path.write_bytes(b"cache to preserve")
        with self.assertRaisesRegex(release_manifest.ManifestError, "generated cache"):
            self.snapshot()
        self.assertEqual(path.read_bytes(), b"cache to preserve")

    def test_source_symlink_rejects_without_removing_link(self) -> None:
        path = self.root / "src/link.cpp"
        path.symlink_to(self.root / "src/physics.cpp")
        with self.assertRaisesRegex(release_manifest.ManifestError, "symbolic link"):
            self.snapshot()
        self.assertTrue(path.is_symlink())

    def test_write_new_json_preserves_existing_evidence(self) -> None:
        path = self.root / "reports/snapshot.json"
        evidence.write_new_json(path, {"original": True})
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            evidence.write_new_json(path, {"original": False})
        self.assertEqual(path.read_bytes(), original)

    def test_ordinary_cli_imports_cannot_create_source_root_bytecode(self) -> None:
        package = self.root / "scripts/qualification"
        provenance = package / "provenance"
        provenance.mkdir(parents=True)
        (provenance / "release_manifest.py").write_text("class ManifestError(Exception):\n    pass\n")
        (provenance / "probe_backend_release_state.py").write_text(
            "def canonical_release_source_roots():\n    return []\n")
        (package / "validate_c6_threshold_candidate.py").write_text(
            "FIELDS = set()\nLEDGERS = set()\nREQUIRED_WORKLOADS = set()\n"
            "def qualifies_memory_gain(*args):\n    return False\n")
        for name in ("collect_c6_evidence.py", "validate_c6_threshold_candidate_v2.py"):
            with self.subTest(entrypoint=name):
                path = package / name
                shutil.copyfile(evidence.ROOT / "scripts/qualification" / name, path)
                result = subprocess.run([sys.executable, "-E", str(path), "--help"], cwd=self.root,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        check=False, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(list(package.rglob("__pycache__")))


if __name__ == "__main__":
    unittest.main()

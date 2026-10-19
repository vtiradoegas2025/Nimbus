#!/usr/bin/env python3
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("thompson_fetch", ROOT / "scripts/dependencies/fetch_thompson_tables.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class Provisioning(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / "installed"
        self.destination.mkdir()

    def bundle(self, members=None):
        files = {"collection.ntctbl": b"canonical collection", "cold.ntcold": b"canonical cold"}
        manifest = {"schema": "nimbus.thompson-runtime-tables", "version": 1,
                    "files": [{"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                              for name, data in files.items()]}
        archive = self.root / "tables.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, data, kind in members or [(name, data, tarfile.REGTYPE) for name, data in files.items()]:
                entry = tarfile.TarInfo(name)
                entry.type = kind
                entry.size = len(data) if kind == tarfile.REGTYPE else 0
                tar.addfile(entry, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
        manifest["archive"] = {"name": archive.name, "bytes": archive.stat().st_size,
                               "sha256": fetch.file_hash(archive), "url": "https://example.invalid/tables.tar.gz"}
        return manifest, archive, files

    def test_exact_install_and_offline_reuse(self):
        manifest, archive, files = self.bundle()
        self.assertTrue(fetch.provision(self.destination, manifest, archive))
        self.assertEqual({path.name: path.read_bytes() for path in self.destination.iterdir()}, files)
        archive.unlink()
        self.assertFalse(fetch.provision(self.destination, manifest))

    def test_wrong_digest_preserves_installed_files(self):
        manifest, archive, _ = self.bundle()
        (self.destination / "collection.ntctbl").write_bytes(b"previous")
        manifest["archive"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            fetch.provision(self.destination, manifest, archive)
        self.assertEqual({p.name: p.read_bytes() for p in self.destination.iterdir()}, {"collection.ntctbl": b"previous"})

    def test_all_members_validate_before_any_publication(self):
        manifest, archive, _ = self.bundle([
            ("collection.ntctbl", b"canonical collection", tarfile.REGTYPE),
            ("cold.ntcold", b"wrong contents", tarfile.REGTYPE),
        ])
        (self.destination / "collection.ntctbl").write_bytes(b"previous")
        with self.assertRaises(ValueError):
            fetch.provision(self.destination, manifest, archive)
        self.assertEqual({p.name: p.read_bytes() for p in self.destination.iterdir()}, {"collection.ntctbl": b"previous"})

    def test_missing_duplicate_traversal_and_link_members_reject(self):
        variants = [
            [("collection.ntctbl", b"canonical collection", tarfile.REGTYPE)],
            [("collection.ntctbl", b"canonical collection", tarfile.REGTYPE)] * 2,
            [("../escape", b"bad", tarfile.REGTYPE)],
            [("collection.ntctbl", b"", tarfile.SYMTYPE)],
        ]
        for members in variants:
            with self.subTest(members=members):
                manifest, archive, _ = self.bundle(members)
                with self.assertRaises(ValueError):
                    fetch.provision(self.destination, manifest, archive)
                self.assertEqual(list(self.destination.iterdir()), [])
                self.assertFalse((self.root / "escape").exists())

    def test_download_and_manifest_sizes_are_bounded(self):
        target = self.root / "bounded"
        for data in (b"short", b"far too long"):
            with self.assertRaises(ValueError):
                fetch.copy_bounded(io.BytesIO(data), target, 8)
        manifest, _, _ = self.bundle()
        invalid = copy.deepcopy(manifest)
        invalid["files"][0]["bytes"] = 513 * 1024 * 1024
        with self.assertRaises(ValueError):
            fetch.validate_manifest(invalid)
        invalid = copy.deepcopy(manifest)
        invalid["files"][0]["name"] = "../escape"
        with self.assertRaises(ValueError):
            fetch.validate_manifest(invalid)

    def test_published_table_identities_match_the_existing_runtime_headers(self):
        manifest = json.loads(fetch.MANIFEST.read_text())
        fetch.validate_manifest(manifest)
        headers = {"ccn_activation.ntact": "activation_table.hpp", "collection_v1.ntctbl": "collection_tables.hpp",
                   "cold_process_v1.ntcold": "cold_process_tables.hpp"}
        entries = {entry["name"]: entry for entry in manifest["files"]}
        for name, header in headers.items():
            source = (ROOT / "src/microphysics/schemes/thompson" / header).read_text()
            self.assertIn(entries[name]["sha256"], source)


if __name__ == "__main__":
    unittest.main()

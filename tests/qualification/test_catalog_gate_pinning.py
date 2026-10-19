from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from scripts.tests.catalog.qualify_catalog_physical_backends import (
    TARGETS as PHYSICAL_TARGETS,
    copy_atomic,
    sha256_file,
)
from scripts.tests.catalog.qualify_catalog_restart_threading import (
    CLASS_SCRIPTS,
    ROOT,
)
sys.path.insert(0, str(ROOT / "scripts" / "qualification" / "nested_domains"))
from scripts.qualification.nested_domains.qualify_moving_nested_vulkan_runtime import (
    require_public_binary_identity,
)


class CatalogGatePinningTests(unittest.TestCase):
    def test_physical_snapshot_is_byte_exact_and_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.write_bytes(b"exact simulator bytes")
            source.chmod(0o755)
            destination.write_bytes(b"stale")
            copy_atomic(source, destination)
            self.assertEqual(sha256_file(destination), sha256_file(source))
            self.assertTrue(destination.stat().st_mode & 0o100)

    def test_physical_target_inventory_is_unique_and_complete(self) -> None:
        self.assertEqual(len(PHYSICAL_TARGETS), 17)
        self.assertEqual(len(set(PHYSICAL_TARGETS)), len(PHYSICAL_TARGETS))
        self.assertIn("qualify-moving-nested-vulkan-public-runtime", PHYSICAL_TARGETS)
        self.assertIn("qualify-moving-nested-metal-public-runtime-smoke", PHYSICAL_TARGETS)
        self.assertIn("qualify-catalog-backend-operation-observations", PHYSICAL_TARGETS)

    def test_restart_threading_inventory_is_present_and_complete(self) -> None:
        self.assertIn("scripts/tests/science/test_earth_fixed_terrain_motion_ready.py", CLASS_SCRIPTS)
        self.assertEqual(len(set(CLASS_SCRIPTS)), len(CLASS_SCRIPTS))
        self.assertTrue(all((ROOT / relative).is_file() for relative in CLASS_SCRIPTS))

    def test_public_runtime_snapshot_requires_exact_production_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            production = root / "production"
            snapshot = root / "snapshot"
            production.write_bytes(b"same")
            snapshot.write_bytes(b"same")
            require_public_binary_identity(snapshot, production)
            snapshot.write_bytes(b"different")
            with self.assertRaisesRegex(
                RuntimeError, "bytes identical to the production simulator",
            ):
                require_public_binary_identity(snapshot, production)


if __name__ == "__main__":
    unittest.main()

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.qualification.catalog_inventory import (
    ROOT, configured_backend, maintained_cases, inventory_counts, restart_class_scripts,
)


class CatalogInventoryTests(unittest.TestCase):
    def test_native_operation_requirements_cover_every_maintained_accelerator_case(self):
        path = ROOT / "tests/fixtures/reference/catalog/backend_tuple_operation_requirements_v1.json"
        requirements = json.loads(path.read_text())["records"]
        expected = {case.relative_to(ROOT).as_posix(): case
                    for group in maintained_cases() for case in group if configured_backend(case) != "cpu"}
        self.assertEqual(sorted(record["config"] for record in requirements), sorted(expected))
        for record in requirements:
            with self.subTest(config=record["config"]):
                config = expected[record["config"]]
                self.assertEqual(record["backend"], configured_backend(config))
                self.assertEqual(record["configSha256"], hashlib.sha256(config.read_bytes()).hexdigest())

    def test_current_manifest_is_complete(self):
        counts = inventory_counts()
        self.assertEqual(counts["catalogCases"], counts["cpuCases"]+counts["acceleratorCases"])
        self.assertIn("scripts/tests/science/test_earth_fixed_terrain_motion_ready.py", restart_class_scripts())

    def test_unreviewed_fixture_or_backend_change_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root/"tests/fixtures/configs/qualification/a.yaml"
            fixture.parent.mkdir(parents=True)
            preset = root/"configs/1000meterResolution/b.yaml"
            preset.parent.mkdir(parents=True)
            fixture.write_text("    mode: qualified\n    backend: cpu\n")
            preset.write_text("    backend: cpu\n")
            manifest = root/"inventory.json"
            manifest.write_text(json.dumps({"schema": "nimbus.maintained-execution-inventory.v1",
                "qualified": [{"path": str(fixture.relative_to(root)), "backend": "cpu"}],
                "representatives": [{"path": str(preset.relative_to(root)), "backend": "cpu"}]}))
            maintained_cases(root, manifest)
            fixture.write_text("    mode: qualified\n    backend: vulkan\n")
            with self.assertRaises(RuntimeError):
                maintained_cases(root, manifest)
            fixture.write_text("    mode: qualified\n    backend: cpu\n")
            unexpected = fixture.parent/"unexpected.yaml"
            unexpected.write_text(fixture.read_text())
            with self.assertRaises(RuntimeError):
                maintained_cases(root, manifest)
            unexpected.unlink()
            fixture.unlink()
            with self.assertRaises(RuntimeError):
                maintained_cases(root, manifest)


if __name__ == "__main__":
    unittest.main()

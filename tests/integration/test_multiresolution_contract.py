import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "qualification" / "output"))

from multiresolution_contract import (  # noqa: E402
    MultiresolutionContractError,
    inspect_durable_hierarchies,
)


class MultiresolutionContractTests(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        run = root / "run"
        step = run / "step_000000"
        step.mkdir(parents=True)
        (step / "theta.npy").write_bytes(b"primary")
        lod = b"lossless-lod"
        (step / "theta.lod1.npy").write_bytes(lod)
        hierarchy = {
            "schema": "nimbus.field-brick-hierarchy",
            "schemaVersion": 2,
            "field": "theta",
            "publicationGeneration": "a" * 32,
            "sourceShape": [4, 4, 4],
            "sourceIndexOffset": [0, 0, 0],
            "sourceDtype": "float32",
            "sampleLocation": "cell_center",
            "extremaSemantics": "stored_values",
            "extremaAbsolutePadding": 0,
            "cullingSafe": True,
            "levels": [
                {
                    "level": 0,
                    "shape": [4, 4, 4],
                    "stride": [1, 1, 1],
                    "source": {"kind": "primary_field"},
                    "bricks": [{"contentSha256": "b" * 64}],
                },
                {
                    "level": 1,
                    "shape": [2, 2, 2],
                    "stride": [2, 2, 2],
                    "source": {
                        "kind": "npy_lod",
                        "path": "theta.lod1.npy",
                        "contentSha256": hashlib.sha256(lod).hexdigest(),
                        "aggregation": "recursive_cell_mean_2x2x2",
                    },
                    "bricks": [{"contentSha256": "c" * 64}],
                },
            ],
        }
        hierarchy_bytes = json.dumps(hierarchy).encode()
        (step / "theta.bricks.json").write_bytes(hierarchy_bytes)
        manifest = {
            "publication_generation": "a" * 32,
            "data_product": {"storage_format": "npy_3d"},
            "published_fields": ["theta"],
            "fields": [{
                "field_id": "theta",
                "file_pattern": "theta.npy",
                "brick_hierarchy": {
                    "schema_version": 2,
                    "path": "theta.bricks.json",
                    "content_sha256": hashlib.sha256(hierarchy_bytes).hexdigest(),
                },
            }],
        }
        (step / "manifest.json").write_text(json.dumps(manifest))
        return run

    def test_valid_hierarchy_and_lod_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = inspect_durable_hierarchies(
                self.fixture(Path(temporary)),
                expected_format="npy_3d",
                require_lod=True,
            )
            self.assertEqual(report["validatedLods"], 1)

    def test_changed_lod_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.fixture(Path(temporary))
            (run / "step_000000" / "theta.lod1.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(MultiresolutionContractError, "LOD 1 checksum"):
                inspect_durable_hierarchies(
                    run, expected_format="npy_3d", require_lod=True
                )

    def test_lod_cannot_be_advertised_as_a_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.fixture(Path(temporary))
            manifest_path = run / "step_000000" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["published_fields"].append("theta.lod1")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(MultiresolutionContractError, "LOD sidecars"):
                inspect_durable_hierarchies(
                    run, expected_format="npy_3d", require_lod=True
                )

    def test_pending_hierarchy_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.fixture(Path(temporary))
            manifest_path = run / "step_000000" / "manifest.json"
            manifest_path.write_text(
                manifest_path.read_text() + "nimbus-pending-field-brick-sha256:theta"
            )
            with self.assertRaisesRegex(MultiresolutionContractError, "pending hierarchy token"):
                inspect_durable_hierarchies(
                    run, expected_format="npy_3d", require_lod=True
                )


if __name__ == "__main__":
    unittest.main()

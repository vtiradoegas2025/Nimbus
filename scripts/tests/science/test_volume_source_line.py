from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "science/audit_volume_line.py"
SPEC = importlib.util.spec_from_file_location("volume_source_line", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SourceLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "theta.npy"

    def tearDown(self):
        self.temp.cleanup()

    def test_constant_and_affine_lines_have_no_second_difference(self):
        self.assertEqual(AUDIT.line_metrics(np.ones(64))["range"], 0)
        affine = AUDIT.line_metrics(np.arange(64) * 0.5 + 300)
        self.assertEqual(affine["neighbor_difference_rms"], 0.5)
        self.assertEqual(affine["second_difference_rms"], 0)

    def test_alternating_line_has_known_difference_amplitude(self):
        m = AUDIT.line_metrics(300 + 2 * (-1.0) ** np.arange(64))
        self.assertEqual(m["range"], 4)
        self.assertEqual(m["standard_deviation"], 2)
        self.assertEqual(m["neighbor_difference_rms"], 4)
        self.assertEqual(m["second_difference_rms"], 8)

    def test_invalid_samples_are_not_filled_or_bridged(self):
        m = AUDIT.line_metrics(np.array([300, np.nan, np.inf, -np.inf, 301]))
        self.assertEqual([m[k] for k in ("nan_count", "positive_infinity_count", "negative_infinity_count")], [1, 1, 1])
        self.assertIsNone(m["second_difference_rms"])
        self.assertIsNone(m["range"])

    def test_axis_permutation_and_float64_values_survive_exactly(self):
        a = 300 + np.arange(12 * 9 * 5).reshape(12, 9, 5) * 1e-10
        expected = a[2:9, 3, 4].tolist()
        for order in ("xyz", "zyx", "yxz"):
            np.save(self.path, np.asfortranarray(a.transpose(tuple("xyz".index(c) for c in order))))
            before = hashlib.sha256(self.path.read_bytes()).hexdigest()
            result = AUDIT.audit_line(self.path, order, (2, 3, 4), "x", 7)
            self.assertEqual(result["values"], expected)
            self.assertEqual(result["shape_xyz"], (12, 9, 5))
            self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_float32_and_non_x_axes(self):
        a = np.arange(10 * 9 * 8, dtype=np.float32).reshape(10, 9, 8)
        np.save(self.path, a)
        self.assertEqual(AUDIT.audit_line(self.path, "xyz", (2, 1, 3), "y", 5)["values"], a[2, 1:6, 3].tolist())
        self.assertEqual(AUDIT.audit_line(self.path, "xyz", (2, 1, 3), "z", 5)["values"], a[2, 1, 3:8].tolist())

    def test_invalid_requests_reject(self):
        np.save(self.path, np.ones((8, 8, 8)))
        for order, start, length in (("xxz", (0, 0, 0), 4), ("xyz", (-1, 0, 0), 4),
                                     ("xyz", (6, 0, 0), 4), ("xyz", (0, 0, 0), 4097)):
            with self.assertRaises(ValueError):
                AUDIT.audit_line(self.path, order, start, "x", length)

    def test_invalid_storage_rejects(self):
        for a in (np.ones((8, 8)), np.ones((8, 8, 8), dtype=np.int32)):
            np.save(self.path, a)
            with self.assertRaises(ValueError):
                AUDIT.audit_line(self.path, "xyz", (0, 0, 0), "x", 4)

    def test_publication_axes_and_halos_are_respected(self):
        np.save(self.path, np.ones((8, 8, 8)))
        manifest = {"grid": {"axis_order": list("xyz"), "valid_index_range": {"lower": [1, 1, 0], "upper": [6, 6, 7]}}}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        AUDIT.audit_line(self.path, "xyz", (1, 2, 3), "x", 6)
        for order, start in (("xyz", (0, 2, 3)), ("zyx", (1, 2, 3))):
            with self.assertRaises(ValueError):
                AUDIT.audit_line(self.path, order, start, "x", 4)
        manifest["publication_state"] = "writing"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            AUDIT.audit_line(self.path, "xyz", (1, 2, 3), "x", 4)

    def test_cli_preserves_existing_evidence_and_publishes_no_invalid_batch(self):
        np.save(self.path, np.ones((8, 8, 8)))
        output = self.root / "audit.json"
        command = [sys.executable, str(SCRIPT), str(self.path), "--axis-order", "xyz", "--start", "1", "2", "3", "--length", "4", "--output", str(output)]
        subprocess.run(command, check=True, capture_output=True)
        original = output.read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(output.read_bytes(), original)
        output.unlink()
        command.insert(3, str(self.root / "missing.npy"))
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertFalse(output.exists())

    def test_malformed_domain_metadata_rejects(self):
        np.save(self.path, np.ones((8, 8, 8)))
        for grid in ({"dimensions": {"x": 7, "y": 8, "z": 8}},
                     {"valid_index_range": {"lower": [0], "upper": [7]}},
                     {"valid_index_range": {"lower": [0, 0, 0], "upper": [8, 7, 7]}}):
            (self.root / "manifest.json").write_text(json.dumps({"grid": grid}))
            with self.assertRaises(ValueError):
                AUDIT.audit_line(self.path, "xyz", (1, 2, 3), "x", 4)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("audit_volume_plane", Path(__file__).resolve().parents[2] / "science/audit_volume_plane.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourcePlaneAuditTests(unittest.TestCase):
    def test_constant_and_affine_have_no_detrended_short_scale_energy(self):
        x, y = np.indices((32, 32))
        for values in (np.full((32, 32), 300), 300 + 2 * x + 3 * y):
            metrics = MODULE.plane_metrics(values)
            self.assertEqual(metrics["detrended_hann_spectrum"]["energy"], 0)

    def test_checkerboard_is_resolved_at_the_sampling_limit(self):
        x, y = np.indices((64, 64))
        metrics = MODULE.plane_metrics(np.where((x + y) % 2, -2, 2))
        self.assertGreater(metrics["detrended_hann_spectrum"]["near_nyquist_energy_fraction"], 0.999)
        self.assertAlmostEqual(metrics["checkerboard_projection_rms"], 2)

    def test_smooth_wave_and_diagonal_four_cell_wave_are_distinguished(self):
        x, y = np.indices((64, 64))
        smooth = MODULE.plane_metrics(np.cos(2 * np.pi * (x + y) / 16))
        fine = MODULE.plane_metrics(np.cos(2 * np.pi * (x + y) / 4))
        self.assertLess(smooth["detrended_hann_spectrum"]["high_axis_frequency_energy_fraction"], 1e-5)
        self.assertGreater(fine["detrended_hann_spectrum"]["high_axis_frequency_energy_fraction"], 0.8)

    def test_missing_values_are_gaps_without_spectral_inpainting(self):
        values = np.zeros((8, 8)); values[2, 3] = np.inf
        metrics = MODULE.plane_metrics(values)
        self.assertEqual(metrics["finite_samples"], 63)
        self.assertIsNone(metrics["raw_spectrum"])

    def test_layout_permutation_and_float64_values_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theta.npy"
            original = np.arange(8 * 9 * 10, dtype=np.float64).reshape(8, 9, 10) + 1e-9
            np.save(path, np.asfortranarray(original.transpose(2, 1, 0)))
            result = MODULE.audit_plane(path, "zyx", (0, 0, 4), "xy", (8, 9))
            np.testing.assert_array_equal(result["values"], original[:, :, 4])
            self.assertEqual(result["dtype"], "<f8")

    def test_published_domain_and_incomplete_outputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theta.npy"; np.save(path, np.ones((10, 10, 10), dtype=np.float32))
            manifest = {"publication_state": "incomplete"}; metadata = path.parent / "manifest.json"
            metadata.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): MODULE.audit_plane(path, "xyz", (0, 0, 0), "xy", (8, 8))
            metadata.write_text(json.dumps({"grid": {"valid_index_range": {"lower": [1, 1, 0], "upper": [8, 8, 9]}}}))
            with self.assertRaises(ValueError): MODULE.audit_plane(path, "xyz", (0, 0, 0), "xy", (8, 8))
            self.assertEqual(MODULE.audit_plane(path, "xyz", (1, 1, 0), "xy", (8, 8))["metrics"]["samples"], 64)


if __name__ == "__main__":
    unittest.main()

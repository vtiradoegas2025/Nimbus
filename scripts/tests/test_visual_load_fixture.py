import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('fixture', ROOT / 'scripts/dev/visualization/generate_visual_load_fixture.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class VisualLoadFixtureTest(unittest.TestCase):
    def test_resolved_fields_are_finite_coherent_and_chunk_independent(self):
        shape = (36, 28, 20)
        whole = fixture.fields(shape, (50, 50, 100), 0, shape[0], 0)
        later = fixture.fields(shape, (50, 50, 100), 0, shape[0], 5)
        self.assertEqual(set(whole), set(fixture.VOLUMES + fixture.COLUMNS))
        for name, array in whole.items():
            self.assertEqual(array.shape, shape if name in fixture.VOLUMES else shape[:2])
            self.assertTrue(np.isfinite(array).all(), name)
            chunked = np.concatenate([fixture.fields(shape, (50, 50, 100), first, min(first + 7, shape[0]), 0)[name]
                                      for first in range(0, shape[0], 7)], axis=0)
            np.testing.assert_array_equal(array, chunked, err_msg=name)
        np.testing.assert_array_equal(whole['cloud_condensate'], whole['cloud_liquid_condensate'] + whole['cloud_frozen_condensate'])
        np.testing.assert_array_equal(whole['total_condensate'], whole['cloud_condensate'] + whole['qr'])
        self.assertFalse(np.array_equal(whole['cloud_condensate'], later['cloud_condensate']))

    def test_full_native_byte_count(self):
        shape = (640, 640, 180)
        self.assertEqual(len(fixture.header(shape)), 80)
        self.assertEqual(len(fixture.VOLUMES) * (80 + np.prod(shape) * 4)
                         + len(fixture.COLUMNS) * (80 + np.prod(shape[:2]) * 4), 5332994640)

    def test_cloud_and_precipitation_advect_together_at_saved_wind_translation(self):
        shape = (40, 36, 32)
        first = fixture.fields(shape, (60, 25, 100), 0, 39, 0)
        next_frame = fixture.fields(shape, (60, 25, 100), 1, 40, 5)
        for field in ('cloud_condensate', 'qr', 'w', 'theta_prime'):
            np.testing.assert_array_equal(first[field][:, :-1], next_frame[field][:, 1:], err_msg=field)

    def test_storm_has_low_base_deep_tower_anvil_and_precipitation(self):
        values = fixture.fields((80, 80, 90), (400, 400, 200), 0, 80, 0)
        cloud = values['cloud_condensate']
        self.assertGreater(np.count_nonzero(cloud[:, :, 4] > .0001), 0)
        self.assertGreater(np.count_nonzero(cloud[:, :, 25] > .0001), 10)
        self.assertGreater(np.count_nonzero(cloud[:, :, 57] > .0001), np.count_nonzero(cloud[:, :, 25] > .0001) * 3)
        self.assertGreater(np.count_nonzero(cloud[:, :, 66] > .0001), 0)
        self.assertGreater(float(np.max(values['qr'][:, :, :20])), .001)
        self.assertEqual(float(np.max(cloud[:, :, -1])), 0)

    def test_npy_publication_hashes_and_non_overwrite(self):
        fields = fixture.VOLUMES + fixture.COLUMNS
        reference = {'schema': 'nimbus.step', 'manifest_version': 7, 'published_fields': list(fields),
                     'fields': [{'field_id': field} for field in fields]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'fixture'
            fixture.generate(output, (12, 10, 8), (50, 50, 100), 2, 5, reference, 3)
            receipt = json.loads((output / 'fixture_receipt.json').read_text())
            self.assertTrue(receipt['specification']['not_a_simulation'])
            self.assertEqual(receipt['total_field_bytes'], sum(path.stat().st_size for path in output.glob('step_*/*.npy')))
            for frame in receipt['frames']:
                for name, info in frame['fields'].items():
                    path = output / f'step_{frame["step"]:06}' / f'{name}.npy'
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), info['sha256'])
                    array = np.load(path, allow_pickle=False)
                    self.assertEqual(array.dtype, np.dtype('<f4'))
                    self.assertEqual(array.shape, (12, 10, 8) if name in fixture.VOLUMES else (12, 10))
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                fixture.generate(output, (12, 10, 8), (50, 50, 100), 2, 5, reference, 3)


if __name__ == '__main__':
    unittest.main()

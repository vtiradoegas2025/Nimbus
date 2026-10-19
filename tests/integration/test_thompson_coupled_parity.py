import importlib
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/tests/science'))
parity = importlib.import_module('test_thompson_accelerator_ready')


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


class CoupledThompsonParity(unittest.TestCase):
    def evaluate(self, changes):
        reference = {field: [f32(1)] for field in parity.thompson.STATE_FIELDS}
        reference.update({field: [f32(0)] for field in ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh')})
        reference['qv'] = [f32(.008)]
        reference['qc'] = [f32(.004)]
        candidate = {field: list(values) for field, values in reference.items()}
        for field, change in changes.items():
            candidate[field][0] = f32(candidate[field][0] + change)

        def values(path):
            data = reference if path.parts[0] == 'reference' else candidate
            return data[path.stem]

        with patch.object(parity.thompson, 'final_frame', lambda path: path), \
             patch.object(parity.thompson, 'npy_float_values', values):
            return parity.require_parity(Path('reference'), Path('candidate'))

    def test_conserved_phase_exchange(self):
        result = self.evaluate({'qv': -5e-8, 'qc': 5e-8})
        self.assertLessEqual(result['total_water']['maxAbsoluteError'], result['total_water']['absoluteLimit'])

    def test_joint_water_drift_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'total-water parity'):
            self.evaluate({'qv': 3e-8, 'qc': 3e-8})

    def test_negative_water_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'invalid coupled'):
            self.evaluate({'qh': -1e-10})

    def test_nonfinite_water_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'invalid coupled'):
            self.evaluate({'qv': float('nan')})


if __name__ == '__main__':
    unittest.main()

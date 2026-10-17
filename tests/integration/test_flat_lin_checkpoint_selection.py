import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.qualification.accelerators.qualify_flat_lin_storm import checkpoint_at_time


class CheckpointSelectionTests(unittest.TestCase):
    def make_checkpoint(self, root, name, time):
        directory = root / 'checkpoints'
        directory.mkdir(exist_ok=True)
        checkpoint = directory / (name + '.nimbus')
        checkpoint.touch()
        Path(str(checkpoint) + '.json').write_text(json.dumps({'simulation_time_s': time}))
        return checkpoint

    def test_decimal_timestep_roundoff_selects_the_saved_midpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_checkpoint(root, 'midpoint', sum([0.1] * 20))
            self.make_checkpoint(root, 'final', sum([0.1] * 30))
            self.assertEqual(checkpoint_at_time(root, 2.0), expected)

    def test_wrong_time_or_nonfinite_metadata_cannot_supply_a_midpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, value in enumerate((2.1, 2.0 + 100 * math.ulp(2.0), float('nan'), float('inf'), None, '2.0', True)):
                self.make_checkpoint(root, str(index), value)
            with self.assertRaises(RuntimeError):
                checkpoint_at_time(root, 2.0)

    def test_multiple_nearby_checkpoints_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root, 'exact', 2.0)
            self.make_checkpoint(root, 'neighbor', math.nextafter(2.0, 3.0))
            with self.assertRaises(RuntimeError):
                checkpoint_at_time(root, 2.0)


if __name__ == '__main__':
    unittest.main()

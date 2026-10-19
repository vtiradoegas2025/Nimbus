import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class SavedVisualReferenceTest(unittest.TestCase):
    def test_source_values_metadata_and_frame_ids_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            output = Path(directory) / 'copy'
            for step in (21, 22, 23):
                frame = source / f'step_{step:06}'
                frame.mkdir(parents=True)
                for field in ('cloud_condensate', 'qr', 'w'):
                    np.save(frame / (field + '.npy'), np.arange(120, dtype=np.float32).reshape(4, 5, 6) + step)
                (frame / 'manifest.json').write_text(json.dumps(dict(step_index=step, simulation_time_s=step*60,
                    grid=dict(nr=4, nth=5, nz=6, dr_m=100, dtheta_rad=100, dz_m=100))))
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            before = {str(path.relative_to(source)): sha(path) for path in source.rglob('*') if path.is_file()}
            command = [sys.executable, str(ROOT / 'scripts/dev/visualization/prepare_saved_visual_reference.py'),
                       '--source', str(source), '--output', str(output), '--steps', '21', '22', '23']
            subprocess.run(command, check=True, capture_output=True)
            for relative, digest in before.items():
                self.assertEqual(sha(source / relative), digest)
                self.assertEqual(sha(output / relative), digest)
            spec = json.loads((output / 'visual_load_fixture.json').read_text())
            self.assertFalse(spec['synthetic'])
            self.assertFalse(spec['values_modified'])
            self.assertEqual(spec['frame_indices'], [21, 22, 23])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            verifier = [sys.executable, str(ROOT / 'scripts/dev/visualization/verify_visual_load_fixture.py'), str(output)]
            subprocess.run(verifier, check=True, capture_output=True)
            (output / 'step_000022/manifest.json').write_text('{}')
            self.assertNotEqual(subprocess.run(verifier, capture_output=True).returncode, 0)


if __name__ == '__main__': unittest.main()

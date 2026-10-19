import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('evidence', Path(__file__).resolve().parents[2] / 'tests/ui/qualify_file_load.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class VisualLoadEvidenceTest(unittest.TestCase):
    def report(self):
        names = ['native-history', 'single-volume', 'single-supercell', 'quad-volume', 'quad-supercell', 'playback-sync', 'steady-export']
        return dict(profile='file-load', status='passed', checks=[dict(name=name, passed=True) for name in names],
                    fixture=dict(specification=dict(not_a_simulation=True, frames=4, shape=[640, 640, 180], bytes_per_frame=5332994640)),
                    steps=[{}] * 4, scenes=[dict(name=name, captures=[dict(pngDataUri='data:image/png;base64,a',
                      gpu=dict(renderer='NVIDIA RTX 5090'), evidence=dict(descriptor=dict(fidelityMode='scientific')))] * panes)
                      for name, panes in [('single-volume', 1), ('single-supercell', 1), ('quad-volume', 4), ('quad-supercell', 4)]])

    def test_complete_measured_scopes(self):
        self.assertEqual(evidence.validate(self.report(), 'NVIDIA')['status'], 'passed')

    def test_incomplete_or_wrong_backend_is_rejected(self):
        for change in [lambda r: r['checks'].pop(), lambda r: r['checks'][0].update(passed=False),
                       lambda r: r['scenes'][-1]['captures'].pop(), lambda r: r['steps'].pop(),
                       lambda r: r['scenes'][0]['captures'][0]['evidence']['descriptor'].update(fidelityMode='presentation'),
                       lambda r: r['scenes'][0]['captures'][0]['gpu'].update(renderer='llvmpipe')]:
            report = copy.deepcopy(self.report())
            change(report)
            with self.assertRaises(ValueError): evidence.validate(report, 'NVIDIA')


if __name__ == '__main__': unittest.main()

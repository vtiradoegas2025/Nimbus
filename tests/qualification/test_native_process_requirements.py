from pathlib import Path
import tempfile
import unittest

from scripts.tests.catalog.qualify_catalog_bounded_runs import require_native_process_chain


class NativeProcessRequirementsTests(unittest.TestCase):
    def check(self, scheme, counts, ratio=2):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'case.yaml'
            config.write_text('time:\n  dt_large: 0.5\n  adaptive_dt: false\n'
                              'domain_motion:\n  nesting:\n    enabled: true\n'
                              f'    parent_time_step_ratio: {ratio}\n'
                              f'microphysics:\n  scheme: {scheme}\n')
            require_native_process_chain(config, counts)

    def milbrandt(self):
        return {f'microphysics/milbrandt_{process}.comp': count
                for process, count in {'cloud_adjustment': 6, 'warm_rain': 6,
                                       'cold': 6, 'melting': 6, 'sedimentation': 30}.items()}

    def test_observing_a_partial_chain_cannot_bless_a_missing_native_owner(self):
        self.check('milbrandt', self.milbrandt())
        for name in self.milbrandt():
            with self.subTest(missing=name):
                incomplete = self.milbrandt()
                del incomplete[name]
                with self.assertRaises(RuntimeError):
                    self.check('milbrandt', incomplete)

    def test_one_missing_species_or_timestep_rejects(self):
        for name in self.milbrandt():
            incomplete = self.milbrandt()
            incomplete[name] -= 1
            with self.assertRaises(RuntimeError):
                self.check('milbrandt', incomplete)

    def test_wrong_family_and_vacuous_counts_reject(self):
        with self.assertRaises(RuntimeError):
            self.check('thompson', self.milbrandt())
        self.check('none', {'dynamics/slow_apply.comp': 1})
        with self.assertRaises(RuntimeError):
            self.check('none', self.milbrandt())
        for value in (0, -1, True):
            invalid = {name: value for name in self.milbrandt()}
            with self.assertRaises(RuntimeError):
                self.check('milbrandt', invalid)

    def test_consistently_skipping_an_entire_physics_step_rejects(self):
        incomplete = {name: value * 5 // 6 for name, value in self.milbrandt().items()}
        with self.assertRaises(RuntimeError):
            self.check('milbrandt', incomplete)

    def test_thompson_feedback_completion_replay_retains_the_full_chain(self):
        counts = {f'microphysics/thompson_{name}.comp': 8
                  for name in ('cloud_adjustment', 'warm_rain', 'cloud_phase', 'freezing', 'deposition')}
        counts['microphysics/thompson_moment_sedimentation.comp'] = 40
        counts['microphysics/thompson_completion.comp'] = 24
        self.check('thompson', counts, ratio=3)
        counts['microphysics/thompson_completion.comp'] = 7
        with self.assertRaises(RuntimeError):
            self.check('thompson', counts, ratio=3)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Regression tests for acoustic boundary-flux sign and quadrature."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

# Register the shared checker module under the import name used by tools.
from tests.integration import test_hydrostatic_qualification as _hydrostatic


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "tests" / "science" / "agrid" / "boundaries" / "check_agrid_boundary_pulse.py"
SPEC = importlib.util.spec_from_file_location("check_agrid_boundary_pulse", PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def field(values: dict[tuple[int, int, int], float]) -> checker.Float32Array:
    shape = (4, 3, 2)
    data = [0.0] * 24
    for (i, j, k), value in values.items():
        data[(i * shape[1] + j) * shape[2] + k] = value
    return checker.Float32Array(shape, tuple(data))


def frame(time_s: float) -> checker.Frame:
    return checker.Frame(
        Path("synthetic"), time_s, (4, 3, 2), 20.0, 10.0, 5.0,
        frozenset({"p", "u"}), (1, 1, 0), (2, 1, 1))


class AGridBoundaryPulseFluxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = {(1, 0): (100.0, 1.0, 10.0),
                     (1, 1): (100.0, 1.0, 10.0)}

    def test_outward_normals_give_the_correct_west_east_sign(self) -> None:
        fields = {
            "p": field({(0, 1, 0): 101.0, (0, 1, 1): 101.0,
                        (3, 1, 0): 102.0, (3, 1, 1): 102.0}),
            "u": field({(0, 1, 0): -4.0, (0, 1, 1): -4.0,
                        (3, 1, 0): 3.0, (3, 1, 1): 3.0}),
        }
        # Each level contributes east 2*3 plus west -1*(-4) = 10 W/m2.
        self.assertEqual(
            checker.outward_acoustic_power_w(frame(0.0), fields, self.base),
            1000.0)

    def test_trapezoid_integrates_every_published_interval(self) -> None:
        fields = {
            "p": field({(3, 1, 0): 102.0, (3, 1, 1): 102.0}),
            "u": field({(3, 1, 0): 3.0, (3, 1, 1): 3.0}),
        }
        history = [(frame(0.0), fields), (frame(2.0), fields),
                   (frame(5.0), fields)]
        # 2 levels * 6 W/m2 * 50 m2 = 600 W for five seconds.
        self.assertEqual(
            checker.integrated_outward_acoustic_energy_j(history, self.base),
            3000.0)


if __name__ == "__main__":
    unittest.main()

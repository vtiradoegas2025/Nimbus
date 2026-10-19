#!/usr/bin/env python3
"""Regression tests for the A-grid acoustic-substep sweep checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from tests.integration.test_hydrostatic_qualification import flat_index, write_run


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "tests" / "science" / "agrid" / "acoustic" / "check_agrid_acoustic_sweep.py"
SPEC = importlib.util.spec_from_file_location("check_agrid_acoustic_sweep", PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


class AGridAcousticSweepTests(unittest.TestCase):
    def test_identical_strict_holds_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_agrid_acoustic_") as temp:
            root = Path(temp)
            runs = {count: root / f"run-{count}" for count in (3, 6, 12)}
            for path in runs.values():
                write_run(path)
            result = checker.qualify_sweep(runs)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["substep_counts"], [3, 6, 12])

    def test_material_final_state_sensitivity_fails(self) -> None:
        def perturb_w(fields: dict[str, list[float]]) -> None:
            fields["w"][flat_index(1, 1, 1)] = 2.0e-4

        with tempfile.TemporaryDirectory(prefix="nimbus_agrid_acoustic_") as temp:
            root = Path(temp)
            runs = {count: root / f"run-{count}" for count in (3, 6, 12)}
            write_run(runs[3], perturb_w)
            write_run(runs[6])
            write_run(runs[12])
            with self.assertRaisesRegex(checker.QualificationError, "3-vs-12 w"):
                checker.qualify_sweep(runs)


if __name__ == "__main__":
    unittest.main()

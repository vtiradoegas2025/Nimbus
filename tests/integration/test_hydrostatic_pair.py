#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "tests" / "science" / "hydrostatic" / "check_hydrostatic_pair.py"
SPEC = importlib.util.spec_from_file_location("check_hydrostatic_pair", PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def result(water: float, shape: list[int] | None = None) -> dict:
    return {
        "status": "PASS",
        "grid_shape": shape or [32, 32, 32],
        "duration_s": 3600.0,
        "inventories": {"initial": {"water": water}},
        "metrics": {"max_wind_ms": 0.0},
    }


class HydrostaticPairTests(unittest.TestCase):
    def test_distinct_matched_pair_passes(self) -> None:
        checked = checker.check_pair(result(1.0), result(200.0))
        self.assertEqual(checked["status"], "PASS")
        self.assertEqual(checked["moist_to_dry_water_ratio"], 200.0)

    def test_insufficient_moisture_separation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "100x"):
            checker.check_pair(result(2.0), result(100.0))

    def test_grid_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "grid shapes differ"):
            checker.check_pair(result(1.0), result(200.0, [16, 16, 16]))


if __name__ == "__main__":
    unittest.main()

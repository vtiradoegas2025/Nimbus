#!/usr/bin/env python3
"""Regression tests for the one-hour hydrostatic qualification checker."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / "tests" / "science" / "common" / "check_hydrostatic_hold.py"
SPEC = importlib.util.spec_from_file_location("check_hydrostatic_hold", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


SHAPE = (4, 4, 3)
DX_M = 1000.0
DY_M = 1200.0
DZ_M = 500.0
GENERATION = "0123456789abcdef0123456789abcdef"


def flat_index(i: int, j: int, k: int) -> int:
    _, ny, nz = SHAPE
    return (i * ny + j) * nz + k


def write_npy_float32(path: Path, values: list[float]) -> None:
    header = repr({"descr": "<f4", "fortran_order": False, "shape": SHAPE})
    padding = (-(10 + len(header) + 1)) % 16
    raw_header = (header + " " * padding + "\n").encode("latin1")
    path.write_bytes(
        b"\x93NUMPY"
        + bytes((1, 0))
        + struct.pack("<H", len(raw_header))
        + raw_header
        + b"".join(struct.pack("<f", value) for value in values)
    )


def base_fields() -> dict[str, list[float]]:
    count = math.prod(SHAPE)
    fields = {
        "u": [0.0] * count,
        "v": [0.0] * count,
        "w": [0.0] * count,
        "rho": [1.0] * count,
        "p": [0.0] * count,
        "theta": [0.0] * count,
        "qv": [0.001] * count,
    }
    kappa = (
        checker.GAS_CONSTANT_DRY_AIR_JKGK
        / checker.SPECIFIC_HEAT_CP_JKGK
    )
    for i in range(SHAPE[0]):
        for j in range(SHAPE[1]):
            for k in range(SHAPE[2]):
                index = flat_index(i, j, k)
                pressure = (
                    checker.REFERENCE_PRESSURE_PA
                    - checker.GRAVITY_MS2 * DZ_M * (float(k) + 0.5)
                )
                temperature = pressure / checker.GAS_CONSTANT_DRY_AIR_JKGK
                theta = temperature * (
                    checker.REFERENCE_PRESSURE_PA / pressure
                ) ** kappa
                fields["p"][index] = pressure
                fields["theta"][index] = theta
    return fields


def step_manifest(step_name: str, step_index: int, time_s: float, generation: str) -> dict[str, object]:
    return {
        "schema": "nimbus.step",
        "manifest_version": checker.OUTPUT_MANIFEST_VERSION,
        "publication_state": "complete",
        "publication_generation": generation,
        "step_index": step_index,
        "simulation_time_s": time_s,
        "step_dir": step_name,
        "grid": {
            "coordinate_system": "cartesian",
            "axis_order": ["x", "y", "z"],
            "dimensions": {"x": SHAPE[0], "y": SHAPE[1], "z": SHAPE[2]},
            "spacing_m": {"x": DX_M, "y": DY_M, "z": DZ_M},
            "native_model_staggering": "c_grid",
            "publication_staggering": "cell_centered",
            "scalar_origin_m": [0.5 * DX_M, 0.5 * DY_M, 0.5 * DZ_M],
            "physical_bounds_m": {
                "lower": [DX_M, DY_M, 0.0],
                "upper": [(SHAPE[0] - 1) * DX_M, (SHAPE[1] - 1) * DY_M, SHAPE[2] * DZ_M],
            },
            "valid_index_range": {
                "lower": [1, 1, 0],
                "upper": [SHAPE[0] - 2, SHAPE[1] - 2, SHAPE[2] - 1],
            },
            "halo_width": {"lower": [1, 1, 0], "upper": [1, 1, 0]},
        },
        "published_fields": list(checker.REQUIRED_FIELDS),
    }


def write_run(
    root: Path,
    mutate_final: Callable[[dict[str, list[float]]], None] | None = None,
    *,
    summary_overrides: dict[str, object] | None = None,
    final_generation: str = GENERATION,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "nimbus_output.json").write_text(
        json.dumps(
            {
                "schema": "nimbus.output",
                "manifest_version": checker.OUTPUT_MANIFEST_VERSION,
                "publication_generation": GENERATION,
                "publication_state": "ready",
                "frame_completion": "atomic_step_manifest",
            }
        ),
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "context": "run_summary",
        "guard_mode": "strict",
        "guard_fail_on": "both",
        "guard_scope": "required_only",
        "report_count": 2,
        "failed_report_count": 0,
        "total_violations": 0,
        "total_nonfinite": 0,
        "total_bounds": 0,
        "total_sanitized_nonfinite": 0,
        "total_sanitized_bounds": 0,
        "failed": False,
    }
    if summary_overrides:
        summary.update(summary_overrides)
    (root / "validation_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    initial = base_fields()
    final = {name: values.copy() for name, values in initial.items()}
    if mutate_final:
        mutate_final(final)
    for step_name, step_index, time_s, generation, fields in (
        ("step_000000", 0, 0.0, GENERATION, initial),
        ("step_000001", 1, 3600.0, final_generation, final),
    ):
        step = root / step_name
        step.mkdir()
        (step / "manifest.json").write_text(
            json.dumps(step_manifest(step_name, step_index, time_s, generation)),
            encoding="utf-8",
        )
        for name, values in fields.items():
            write_npy_float32(step / f"{name}.npy", values)


class HydrostaticQualificationTests(unittest.TestCase):
    def test_balanced_atomic_strict_run_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_hydrostatic_check_") as temp:
            run_dir = Path(temp)
            write_run(run_dir)
            result = checker.qualify_run(run_dir)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["duration_s"], 3600.0)
            self.assertEqual(result["physical_cell_count"], 12)

    def test_cgrid_scalar_ghosts_do_not_contaminate_physical_metrics(self) -> None:
        def poison_ghosts(fields: dict[str, list[float]]) -> None:
            for i in range(SHAPE[0]):
                for j in range(SHAPE[1]):
                    if i not in (0, SHAPE[0] - 1) and j not in (0, SHAPE[1] - 1):
                        continue
                    for k in range(SHAPE[2]):
                        index = flat_index(i, j, k)
                        fields["rho"][index] = 20.0
                        fields["p"][index] = 180000.0
                        fields["theta"][index] = 800.0
                        fields["qv"][index] = 0.5

        with tempfile.TemporaryDirectory(prefix="nimbus_hydrostatic_check_") as temp:
            run_dir = Path(temp)
            write_run(run_dir, poison_ghosts)
            self.assertEqual(checker.qualify_run(run_dir)["status"], "PASS")

    def test_output_velocities_are_already_cell_centered_for_energy(self) -> None:
        stationary = base_fields()
        moving = {name: values.copy() for name, values in stationary.items()}
        for i in range(SHAPE[0]):
            for j in range(SHAPE[1]):
                for k in range(SHAPE[2]):
                    index = flat_index(i, j, k)
                    moving["u"][index] = 3.0
                    moving["v"][index] = 4.0
                    moving["w"][index] = 12.0

        def arrays(raw: dict[str, list[float]]) -> dict[str, checker.Float32Array]:
            return {
                name: checker.Float32Array(shape=SHAPE, values=tuple(values))
                for name, values in raw.items()
            }

        energy_stationary = checker.inventories(arrays(stationary), DZ_M)["total_energy"]
        energy_moving = checker.inventories(arrays(moving), DZ_M)["total_energy"]
        physical_cells = (SHAPE[0] - 2) * (SHAPE[1] - 2) * SHAPE[2]
        expected_kinetic = physical_cells * 0.5 * (3.0**2 + 4.0**2 + 12.0**2)
        self.assertAlmostEqual(energy_moving - energy_stationary, expected_kinetic)

    def test_excess_wind_fails_the_balance_threshold(self) -> None:
        def inject_wind(fields: dict[str, list[float]]) -> None:
            fields["w"][flat_index(1, 1, 1)] = 0.01

        with tempfile.TemporaryDirectory(prefix="nimbus_hydrostatic_check_") as temp:
            run_dir = Path(temp)
            write_run(run_dir, inject_wind)
            with self.assertRaisesRegex(checker.QualificationError, "max_wind_ms"):
                checker.qualify_run(run_dir)

    def test_nonfinite_payload_fails_closed(self) -> None:
        def inject_nan(fields: dict[str, list[float]]) -> None:
            fields["rho"][flat_index(1, 1, 1)] = math.nan

        with tempfile.TemporaryDirectory(prefix="nimbus_hydrostatic_check_") as temp:
            run_dir = Path(temp)
            write_run(run_dir, inject_nan)
            with self.assertRaisesRegex(checker.QualificationError, "NaN or infinity"):
                checker.qualify_run(run_dir)

    def test_strict_validation_violation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_hydrostatic_check_") as temp:
            run_dir = Path(temp)
            write_run(
                run_dir,
                summary_overrides={"total_violations": 1, "total_bounds": 1},
            )
            with self.assertRaisesRegex(checker.QualificationError, "total_violations"):
                checker.qualify_run(run_dir)

    def test_frame_generation_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_hydrostatic_check_") as temp:
            run_dir = Path(temp)
            write_run(run_dir, final_generation="fedcba9876543210fedcba9876543210")
            with self.assertRaisesRegex(checker.QualificationError, "generation mismatch"):
                checker.qualify_run(run_dir)

    def test_frame_step_index_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus_hydrostatic_check_") as temp:
            run_dir = Path(temp)
            write_run(run_dir)
            manifest_path = run_dir / "step_000001" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["step_index"] = 7
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(checker.QualificationError, "step_index"):
                checker.qualify_run(run_dir)


if __name__ == "__main__":
    unittest.main()

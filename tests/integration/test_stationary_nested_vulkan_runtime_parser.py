#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "qualification" / "nested_domains" / "qualify_stationary_nested_vulkan_runtime.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_stationary_nested_vulkan_runtime", PATH
)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def kessler_log(**overrides: float | int) -> str:
    values: dict[str, float | int] = {
        "parent_relative": 1.0e-12,
        "parent_imbalance": 1.0e-3,
        "parent_baseline": 1.0e10,
        "parent_activity": 2.0e5,
        "parent_water_repair": 0.0,
        "parent_water_repair_count": 0,
        "parent_precipitation": 1.0,
        "parent_heat": 1.0e11,
        "parent_theta_repair": 0.0,
        "parent_theta_repair_count": 0,
        "parent_energy_residual": 1.0e-5,
        "parent_energy_activity": 1.0e12,
        "child_relative": 1.0e-12,
        "child_imbalance": -5.0e-4,
        "child_baseline": 1.0e9,
        "child_activity": 1.0e5,
        "child_water_repair": 0.0,
        "child_water_repair_count": 0,
        "child_precipitation": 0.5,
        "child_heat": 5.0e10,
        "child_theta_repair": 0.0,
        "child_theta_repair_count": 0,
        "child_energy_residual": -1.0e-5,
        "child_energy_activity": 5.0e11,
    }
    values.update(overrides)
    return "\n".join(
        (
            "[NESTING DEVICE MICROPHYSICS] "
            "kessler_pointwise_dispatches=24, "
            "kessler_sedimentation_dispatches=24, "
            "microphysics_apply_dispatches=0",
            "[NESTING DEVICE MICROPHYSICS LEVELS] "
            "parent_pointwise=8, parent_sedimentation=8, parent_apply=0, "
            "child_pointwise=16, child_sedimentation=16, child_apply=0",
            "[NESTING MICROPHYSICS] "
            f"parent_active=1, parent_relative_imbalance={values['parent_relative']}, "
            f"parent_imbalance_kg={values['parent_imbalance']}, "
            f"parent_baseline_water_kg={values['parent_baseline']}, "
            f"parent_water_activity_kg={values['parent_activity']}, "
            f"parent_material_water_repair_kg={values['parent_water_repair']}, "
            "parent_material_water_repair_count="
            f"{values['parent_water_repair_count']}, "
            f"parent_surface_precipitation_kg={values['parent_precipitation']}, "
            f"parent_heat_j={values['parent_heat']}, "
            f"parent_theta_repair_j={values['parent_theta_repair']}, "
            f"parent_theta_repair_count={values['parent_theta_repair_count']}, "
            "parent_moist_energy_complete=1, "
            f"parent_moist_energy_residual_j={values['parent_energy_residual']}, "
            f"parent_energy_activity_j={values['parent_energy_activity']}, "
            f"child_active=1, child_relative_imbalance={values['child_relative']}, "
            f"child_imbalance_kg={values['child_imbalance']}, "
            f"child_baseline_water_kg={values['child_baseline']}, "
            f"child_water_activity_kg={values['child_activity']}, "
            f"child_material_water_repair_kg={values['child_water_repair']}, "
            "child_material_water_repair_count="
            f"{values['child_water_repair_count']}, "
            f"child_surface_precipitation_kg={values['child_precipitation']}, "
            f"child_heat_j={values['child_heat']}, "
            f"child_theta_repair_j={values['child_theta_repair']}, "
            f"child_theta_repair_count={values['child_theta_repair_count']}, "
            "child_moist_energy_complete=1, "
            f"child_moist_energy_residual_j={values['child_energy_residual']}, "
            f"child_energy_activity_j={values['child_energy_activity']}",
        )
    )


def smagorinsky_log(**overrides: float | int) -> str:
    values: dict[str, float | int] = {
        "parent_work": -100.0,
        "parent_stress_steps": 8,
        "parent_boundary_work": 20.0,
        "parent_dissipation": 120.0,
        "parent_stress_residual": 1.0e-12,
        "child_work": -50.0,
        "child_stress_steps": 16,
        "child_boundary_work": 10.0,
        "child_dissipation": 60.0,
        "child_stress_residual": -1.0e-12,
    }
    values.update(overrides)
    return "\n".join(
        (
            "[NESTING DEVICE] smagorinsky_coefficients_dispatches=24, "
            "smagorinsky_tendencies_dispatches=24, sgs_apply_dispatches=24",
            "[NESTING DEVICE LEVELS] parent_coefficients=8, "
            "parent_tendencies=8, parent_sgs_apply=8, "
            "child_coefficients=16, child_tendencies=16, child_sgs_apply=16",
            "[NESTING SGS] parent_work_steps=8, "
            f"parent_committed_work_j={values['parent_work']}, "
            f"parent_stress_steps={values['parent_stress_steps']}, "
            f"parent_open_boundary_work_j={values['parent_boundary_work']}, "
            f"parent_irreversible_dissipation_j={values['parent_dissipation']}, "
            f"parent_stress_residual_j={values['parent_stress_residual']}, "
            "child_work_steps=16, "
            f"child_committed_work_j={values['child_work']}, "
            f"child_stress_steps={values['child_stress_steps']}, "
            f"child_open_boundary_work_j={values['child_boundary_work']}, "
            f"child_irreversible_dissipation_j={values['child_dissipation']}, "
            f"child_stress_residual_j={values['child_stress_residual']}",
        )
    )


def radiation_log(**overrides: float | int | str) -> str:
    values: dict[str, float | int | str] = {
        "parent_owner": "host",
        "parent_last": 4.0,
        "parent_next": 6.0,
        "parent_steps": 12,
        "parent_held": -4.0e6,
        "parent_abs": 8.0e6,
        "parent_nonzero": 4096,
        "parent_nonfinite": 0,
        "parent_last_expected": -2.0e6,
        "parent_last_applied": -2.0e6,
        "parent_expected": -2.4e7,
        "parent_applied": -2.4e7,
        "parent_residual": 0.0,
        "child_owner": "host",
        "child_last": 4.0,
        "child_next": 6.0,
        "child_steps": 24,
        "child_held": -1.0e6,
        "child_abs": 2.0e6,
        "child_nonzero": 1500,
        "child_nonfinite": 0,
        "child_last_expected": -2.5e5,
        "child_last_applied": -2.5e5,
        "child_expected": -6.0e6,
        "child_applied": -6.0e6,
        "child_residual": 0.0,
    }
    values.update(overrides)
    return (
        "[NESTING RADIATION] "
        f"parent_owner={values['parent_owner']}, "
        f"parent_last_update_time_s={values['parent_last']}, "
        f"parent_next_update_time_s={values['parent_next']}, "
        f"parent_applied_steps={values['parent_steps']}, "
        f"parent_held_power_w={values['parent_held']}, "
        f"parent_held_abs_power_w={values['parent_abs']}, "
        f"parent_nonzero_cells={values['parent_nonzero']}, "
        f"parent_nonfinite_cells={values['parent_nonfinite']}, "
        f"parent_last_expected_energy_j={values['parent_last_expected']}, "
        f"parent_last_applied_energy_j={values['parent_last_applied']}, "
        f"parent_expected_energy_j={values['parent_expected']}, "
        f"parent_applied_energy_j={values['parent_applied']}, "
        f"parent_energy_residual_j={values['parent_residual']}, "
        f"child_owner={values['child_owner']}, "
        f"child_last_update_time_s={values['child_last']}, "
        f"child_next_update_time_s={values['child_next']}, "
        f"child_applied_steps={values['child_steps']}, "
        f"child_held_power_w={values['child_held']}, "
        f"child_held_abs_power_w={values['child_abs']}, "
        f"child_nonzero_cells={values['child_nonzero']}, "
        f"child_nonfinite_cells={values['child_nonfinite']}, "
        f"child_last_expected_energy_j={values['child_last_expected']}, "
        f"child_last_applied_energy_j={values['child_last_applied']}, "
        f"child_expected_energy_j={values['child_expected']}, "
        f"child_applied_energy_j={values['child_applied']}, "
        f"child_energy_residual_j={values['child_residual']}"
    )


class StationaryNestedVulkanRuntimeParserTests(unittest.TestCase):
    def test_valid_kessler_and_smagorinsky_evidence_passes(self) -> None:
        qualification.require_kessler_log(kessler_log(), "valid")
        qualification.require_smagorinsky_log(smagorinsky_log(), "valid")

    def test_zero_or_crumb_heat_and_fallout_fail(self) -> None:
        invalid = (
            {"parent_heat": 0.0},
            {"child_heat": 1.0e7},
            {"parent_precipitation": 0.0},
            {"child_precipitation": 1.0e-5},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(RuntimeError):
                    qualification.require_kessler_log(
                        kessler_log(**overrides), "inactive"
                    )

    def test_activity_scale_rejects_residual_hidden_by_baseline(self) -> None:
        # 0.02 kg / 1e10 kg passes the baseline-relative criterion, while
        # 0.02 kg / 1000 kg exceeds the active-process tolerance.
        with self.assertRaises(RuntimeError):
            qualification.require_kessler_log(
                kessler_log(
                    parent_relative=2.0e-12,
                    parent_imbalance=2.0e-2,
                    parent_activity=1.0e3,
                ),
                "diluted residual",
            )

    def test_material_water_or_theta_repair_fails(self) -> None:
        invalid = (
            {
                "child_water_repair": 1.0e-3,
                "child_water_repair_count": 1,
            },
            {
                "parent_theta_repair": 1.0,
                "parent_theta_repair_count": 1,
            },
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(RuntimeError):
                    qualification.require_kessler_log(
                        kessler_log(**overrides), "repaired"
                    )

    def test_bad_reservoir_energy_closure_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            qualification.require_kessler_log(
                kessler_log(
                    parent_energy_residual=2.0,
                    parent_energy_activity=1.0e12,
                ),
                "energy residual",
            )

    def test_nonpositive_sgs_dissipation_fails(self) -> None:
        for dissipation in (0.0, -1.0):
            with self.subTest(dissipation=dissipation):
                with self.assertRaises(RuntimeError):
                    qualification.require_smagorinsky_log(
                        smagorinsky_log(child_dissipation=dissipation),
                        "non-dissipative",
                    )

    def test_bad_sgs_stress_residual_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            qualification.require_smagorinsky_log(
                smagorinsky_log(parent_stress_residual=1.0e-4),
                "stress residual",
            )

    def test_valid_host_owned_radiation_evidence_passes(self) -> None:
        qualification.radiation_row(radiation_log(), "valid", 4.0, 6.0)

    def test_radiation_rejects_nonhost_or_vacuous_evidence(self) -> None:
        invalid = (
            {"parent_owner": "device"},
            {"parent_abs": 0.0},
            {"child_nonzero": 0},
            {"child_nonfinite": 1},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(RuntimeError):
                    qualification.radiation_row(
                        radiation_log(**overrides), "invalid", 4.0, 6.0
                    )

    def test_radiation_rejects_wrong_cadence_phase(self) -> None:
        with self.assertRaises(RuntimeError):
            qualification.radiation_row(
                radiation_log(parent_next=8.0), "phase", 4.0, 6.0
            )

    def test_radiation_rejects_step_and_cumulative_energy_failures(self) -> None:
        invalid = (
            {"parent_last_applied": -1.0e6},
            {
                "child_applied": -5.0e6,
                "child_residual": 1.0e6,
            },
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(RuntimeError):
                    qualification.radiation_row(
                        radiation_log(**overrides), "energy", 4.0, 6.0
                    )

    def test_radiation_ledger_parity_rejects_large_difference(self) -> None:
        cpu = qualification.radiation_row(
            radiation_log(), "cpu", 4.0, 6.0
        )
        vulkan = qualification.radiation_row(
            radiation_log(parent_held=-3.0e6), "vulkan", 4.0, 6.0
        )
        with self.assertRaises(RuntimeError):
            qualification.require_radiation_ledger_parity(
                cpu, vulkan, "parity"
            )


if __name__ == "__main__":
    unittest.main()

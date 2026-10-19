"""Independent checks for the SCI-157 completed-step energy composition."""

from __future__ import annotations

import math
from typing import Callable


SCALARS = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh")
OWNER_FIELDS = (
    "acoustic_committed_change_j",
    "slow_dynamics_committed_change_j",
    "scalar_transport_committed_change_j",
    "thermodynamic_sources_committed_change_j",
    "sgs_committed_change_j",
    "diffusion_committed_change_j",
    "eos_projection_committed_change_j",
    "sponge_committed_change_j",
    "final_boundary_committed_change_j",
)
RESERVOIR_FIELDS = (
    "dry_internal_change_j",
    "water_sensible_change_j",
    "phase_reference_change_j",
    "carrier_kinetic_change_j",
    "gravitational_change_j",
    "subgrid_tke_change_j",
)
ENDPOINT_COLUMNS = (
    "d_moist_v2_dry_internal_j",
    "d_moist_v2_water_sensible_j",
    "d_moist_v2_phase_reference_j",
    "d_moist_v2_resolved_kinetic_j",
    "d_moist_v2_gravitational_j",
    "d_moist_v2_subgrid_tke_j",
)
INVENTORY_NAMES = (
    "dry_internal_j",
    "water_sensible_j",
    "phase_reference_j",
    "resolved_kinetic_j",
    "gravitational_j",
    "subgrid_tke_j",
)


def _relative(left: float, right: float) -> float:
    return abs(left - right) / max(1.0, abs(left), abs(right))


def _subtraction_spacing(
    authored: float,
    endpoint_change: float,
    before_inventory: float,
    after_inventory: float,
) -> float:
    # Both currencies are FP64. The endpoint is the difference of two large
    # independently reduced inventories, so its meaningful floor is their
    # representational spacing, not the magnitude of a potentially tiny
    # physical change.
    return (
        math.ulp(before_inventory)
        + math.ulp(after_inventory)
        + math.ulp(authored)
        + math.ulp(endpoint_change)
    )


def _subtraction_ulp_error(
    authored: float,
    endpoint_change: float,
    before_inventory: float,
    after_inventory: float,
) -> float:
    return abs(authored - endpoint_change) / _subtraction_spacing(
        authored, endpoint_change, before_inventory, after_inventory)


def _primitive_step_owners(row: dict[str, str]) -> dict[str, float]:
    return {
        "acoustic_committed_change_j":
            float(row["acoustic_v1_committed_inventory_change_j"]),
        "slow_dynamics_committed_change_j":
            float(row[
                "slow_momentum_energy_step_committed_kinetic_change_j"])
            + float(row[
                "slow_momentum_energy_step_committed_pressure_energy_change_j"]),
        "scalar_transport_committed_change_j":
            sum(float(row[
                f"advection_{scalar}_energy_step_committed_state_change_j"])
                for scalar in SCALARS),
        "thermodynamic_sources_committed_change_j":
            float(row["microphysics_moist_committed_energy_change_j"]),
        "sgs_committed_change_j":
            float(row[
                "sgs_moist_energy_step_committed_energy_change_j"]),
        "diffusion_committed_change_j":
            float(row[
                "diffusion_moist_energy_step_committed_energy_change_j"]),
        "eos_projection_committed_change_j":
            float(row[
                "eos_pressure_energy_step_committed_energy_change_j"]),
        "sponge_committed_change_j":
            sum(float(row[
                f"sponge_energy_step_{name}_committed_energy_change_j"])
                for name in ("upper", "w_only", "lateral")),
        "final_boundary_committed_change_j":
            float(row[
                "final_boundary_energy_step_committed_energy_change_j"]),
    }


def completed_step_energy_identities(
    sources: list[dict[str, str]],
    energy_rows: list[dict[str, str]],
    expected_counts: list[int],
    require: Callable[[bool, str], None],
    maximum_relative_residual: float = 2.0e-12,
    maximum_endpoint_ulp_error: float = 64.0,
) -> dict[str, float]:
    total_rows = [row for row in energy_rows
                  if row["stage"] == "total_step"]
    require(len(total_rows) == len(sources),
            "basis-v2 endpoint ledger lacks one total_step row per source row")
    require([int(row["completed_step_energy_count"]) for row in sources]
            == expected_counts,
            "completed-step energy sequence is discontinuous")

    owner_identity = 0.0
    reservoir_identity = 0.0
    primitive_identity = 0.0
    endpoint_step_identity = 0.0
    endpoint_cumulative_identity = 0.0
    endpoint_step_ulp_error = 0.0
    endpoint_cumulative_ulp_error = 0.0
    endpoint_step_ulp_location = "none"
    endpoint_cumulative_ulp_location = "none"
    running_owners = {field: 0.0 for field in OWNER_FIELDS}
    cumulative_endpoint_spacing = {
        field: 0.0 for field in RESERVOIR_FIELDS}
    cumulative_total_spacing = 0.0
    initial_inventories: dict[str, float] | None = None
    initial_total_inventory = 0.0

    for row_index, (source, endpoint) in enumerate(zip(sources, total_rows)):
        step_count = expected_counts[row_index]
        require(
            int(source["completed_step_energy_report_schema_version"]) == 1
            and int(source["completed_step_energy_basis_version"]) == 2
            and int(source["completed_step_energy_step_complete"]) == 1
            and int(source["completed_step_energy_cumulative_complete"]) == 1,
            "completed-step energy currency is partial or changed")
        primitive = _primitive_step_owners(source)
        for field in OWNER_FIELDS:
            step_value = float(
                source[f"completed_step_energy_step_{field}"])
            cumulative_value = float(
                source[f"completed_step_energy_cumulative_{field}"])
            running_owners[field] += primitive[field]
            primitive_identity = max(
                primitive_identity,
                _relative(step_value, primitive[field]),
                _relative(cumulative_value, running_owners[field]))

        for cadence in ("step", "cumulative"):
            prefix = f"completed_step_energy_{cadence}_"
            owners = sum(float(source[prefix + field])
                         for field in OWNER_FIELDS)
            reservoirs = sum(float(source[prefix + field])
                             for field in RESERVOIR_FIELDS)
            committed = float(source[prefix + "committed_owner_sum_j"])
            closure = float(source[prefix + "closure_residual_j"])
            owner_identity = max(
                owner_identity, _relative(committed, owners))
            reservoir_identity = max(
                reservoir_identity,
                _relative(committed, reservoirs + closure))

        endpoint_values = [
            float(endpoint[column]) for column in ENDPOINT_COLUMNS]
        endpoint_total = float(endpoint["d_moist_v2_total_j"])
        before_inventories = {
            field: float(endpoint[f"moist_v2_before_{name}"])
            for field, name in zip(RESERVOIR_FIELDS, INVENTORY_NAMES)
        }
        after_inventories = {
            field: float(endpoint[f"moist_v2_after_{name}"])
            for field, name in zip(RESERVOIR_FIELDS, INVENTORY_NAMES)
        }
        if initial_inventories is None:
            initial_inventories = before_inventories
            initial_total_inventory = float(
                endpoint["moist_v2_before_total_j"])
        for field, endpoint_value in zip(
                RESERVOIR_FIELDS, endpoint_values):
            step_value = float(
                source[f"completed_step_energy_step_{field}"])
            cumulative_endpoint = (
                after_inventories[field] - initial_inventories[field])
            cumulative_value = float(
                source[f"completed_step_energy_cumulative_{field}"])
            endpoint_step_identity = max(
                endpoint_step_identity,
                _relative(step_value, endpoint_value))
            endpoint_cumulative_identity = max(
                endpoint_cumulative_identity,
                _relative(cumulative_value, cumulative_endpoint))
            step_ulp_error = _subtraction_ulp_error(
                step_value, endpoint_value,
                before_inventories[field], after_inventories[field])
            if step_ulp_error > endpoint_step_ulp_error:
                endpoint_step_ulp_error = step_ulp_error
                endpoint_step_ulp_location = f"step {step_count} {field}"
            cumulative_endpoint_spacing[field] += _subtraction_spacing(
                step_value, endpoint_value,
                before_inventories[field], after_inventories[field])
            direct_cumulative_spacing = _subtraction_spacing(
                cumulative_value, cumulative_endpoint,
                initial_inventories[field], after_inventories[field])
            cumulative_ulp_error = (
                abs(cumulative_value - cumulative_endpoint)
                / max(cumulative_endpoint_spacing[field],
                      direct_cumulative_spacing))
            if cumulative_ulp_error > endpoint_cumulative_ulp_error:
                endpoint_cumulative_ulp_error = cumulative_ulp_error
                endpoint_cumulative_ulp_location = (
                    f"step {step_count} {field}")
        step_total = float(
            source["completed_step_energy_step_committed_owner_sum_j"])
        cumulative_total = float(
            source["completed_step_energy_cumulative_committed_owner_sum_j"])
        before_total = float(endpoint["moist_v2_before_total_j"])
        after_total = float(endpoint["moist_v2_after_total_j"])
        cumulative_endpoint_total = after_total - initial_total_inventory
        endpoint_step_identity = max(
            endpoint_step_identity,
            _relative(step_total, endpoint_total))
        endpoint_cumulative_identity = max(
            endpoint_cumulative_identity,
            _relative(cumulative_total, cumulative_endpoint_total))
        total_step_ulp_error = _subtraction_ulp_error(
            step_total, endpoint_total, before_total, after_total)
        if total_step_ulp_error > endpoint_step_ulp_error:
            endpoint_step_ulp_error = total_step_ulp_error
            endpoint_step_ulp_location = f"step {step_count} total"
        cumulative_total_spacing += _subtraction_spacing(
            step_total, endpoint_total, before_total, after_total)
        direct_cumulative_total_spacing = _subtraction_spacing(
            cumulative_total, cumulative_endpoint_total,
            initial_total_inventory, after_total)
        total_cumulative_ulp_error = (
            abs(cumulative_total - cumulative_endpoint_total)
            / max(cumulative_total_spacing,
                  direct_cumulative_total_spacing))
        if total_cumulative_ulp_error > endpoint_cumulative_ulp_error:
            endpoint_cumulative_ulp_error = total_cumulative_ulp_error
            endpoint_cumulative_ulp_location = f"step {step_count} total"

    algebraic_maximum = max(
        owner_identity, reservoir_identity, primitive_identity)
    endpoint_ulp_maximum = max(
        endpoint_step_ulp_error, endpoint_cumulative_ulp_error)
    require(math.isfinite(algebraic_maximum)
            and algebraic_maximum <= maximum_relative_residual,
            "completed-step composition disagrees with a primitive owner")
    require(math.isfinite(endpoint_ulp_maximum)
            and endpoint_ulp_maximum <= maximum_endpoint_ulp_error,
            "completed-step composition exceeds the FP64 cancellation bound "
            "of the independent basis-v2 endpoint inventory "
            f"({endpoint_ulp_maximum:.17g} > "
            f"{maximum_endpoint_ulp_error:.17g} ulps; "
            f"step={endpoint_step_ulp_error:.17g} at "
            f"{endpoint_step_ulp_location}, "
            f"cumulative={endpoint_cumulative_ulp_error:.17g} at "
            f"{endpoint_cumulative_ulp_location})")
    return {
        "maximum_completed_step_owner_identity_relative_residual":
            owner_identity,
        "maximum_completed_step_reservoir_identity_relative_residual":
            reservoir_identity,
        "maximum_completed_step_primitive_identity_relative_residual":
            primitive_identity,
        "maximum_completed_step_endpoint_relative_residual":
            endpoint_step_identity,
        "maximum_cumulative_completed_step_endpoint_relative_residual":
            endpoint_cumulative_identity,
        "maximum_completed_step_endpoint_ulp_error":
            endpoint_step_ulp_error,
        "maximum_cumulative_completed_step_endpoint_ulp_error":
            endpoint_cumulative_ulp_error,
    }

/**
 * @file completed_step_energy_report.hpp
 * @brief Non-double-counted basis-v2 composition of one completed timestep.
 *
 * The owner fields are the authoritative committed inventory changes from
 * SCI-141 through SCI-156. The reservoir fields independently compose those
 * same reports by physical basis-v2 reservoir. PBL momentum is already part
 * of the slow-momentum commit and is deliberately not a separate owner.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>

namespace tmv
{

struct CompletedStepEnergyReportV1
{
    static constexpr std::uint32_t report_schema_version = 3;
    static constexpr std::uint32_t energy_basis_version = 2;

    double acoustic_committed_change_j = 0.0;
    double slow_dynamics_committed_change_j = 0.0;
    double scalar_transport_committed_change_j = 0.0;
    double thermodynamic_sources_committed_change_j = 0.0;
    double sgs_committed_change_j = 0.0;
    double diffusion_committed_change_j = 0.0;
    double eos_projection_committed_change_j = 0.0;
    double sponge_committed_change_j = 0.0;
    double final_boundary_committed_change_j = 0.0;
    double unsplit_rk_committed_change_j = 0.0;
    double coordinate_remap_committed_change_j = 0.0;
    double committed_owner_sum_j = 0.0;

    double dry_internal_change_j = 0.0;
    double water_sensible_change_j = 0.0;
    double phase_reference_change_j = 0.0;
    double carrier_kinetic_change_j = 0.0;
    double gravitational_change_j = 0.0;
    double subgrid_tke_change_j = 0.0;
    double closure_residual_j = 0.0;

    bool complete = false;
};

inline double completed_step_owner_sum(
    const CompletedStepEnergyReportV1& report)
{
    return report.acoustic_committed_change_j
        + report.slow_dynamics_committed_change_j
        + report.scalar_transport_committed_change_j
        + report.thermodynamic_sources_committed_change_j
        + report.sgs_committed_change_j
        + report.diffusion_committed_change_j
        + report.eos_projection_committed_change_j
        + report.sponge_committed_change_j
        + report.final_boundary_committed_change_j
        + report.unsplit_rk_committed_change_j
        + report.coordinate_remap_committed_change_j;
}

inline double completed_step_reservoir_sum(
    const CompletedStepEnergyReportV1& report)
{
    return report.dry_internal_change_j
        + report.water_sensible_change_j
        + report.phase_reference_change_j
        + report.carrier_kinetic_change_j
        + report.gravitational_change_j
        + report.subgrid_tke_change_j;
}

/** Validate report shape, finiteness, and exact bookkeeping identities only.
 * This does not assert that the physical closure residual is acceptably small.
 */
inline bool structurally_valid_completed_step_energy_report(
    const CompletedStepEnergyReportV1& report)
{
    const double values[] = {
        report.acoustic_committed_change_j,
        report.slow_dynamics_committed_change_j,
        report.scalar_transport_committed_change_j,
        report.thermodynamic_sources_committed_change_j,
        report.sgs_committed_change_j,
        report.diffusion_committed_change_j,
        report.eos_projection_committed_change_j,
        report.sponge_committed_change_j,
        report.final_boundary_committed_change_j,
        report.unsplit_rk_committed_change_j,
        report.coordinate_remap_committed_change_j,
        report.committed_owner_sum_j,
        report.dry_internal_change_j,
        report.water_sensible_change_j,
        report.phase_reference_change_j,
        report.carrier_kinetic_change_j,
        report.gravitational_change_j,
        report.subgrid_tke_change_j,
        report.closure_residual_j,
    };
    if (!std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return std::isfinite(value); }))
    {
        return false;
    }
    if (!report.complete)
    {
        return std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return value == 0.0; });
    }
    return report.committed_owner_sum_j
            == completed_step_owner_sum(report)
        && report.closure_residual_j
            == report.committed_owner_sum_j
                - completed_step_reservoir_sum(report);
}

// Compatibility name for checkpoint formats and older callers. New
// qualification code should say explicitly whether it needs structural
// validity or bounded physical closure.
inline bool valid_completed_step_energy_report(
    const CompletedStepEnergyReportV1& report)
{
    return structurally_valid_completed_step_energy_report(report);
}

inline double completed_step_energy_closure_scale(
    const CompletedStepEnergyReportV1& report)
{
    return std::max({1.0,
                     std::abs(report.committed_owner_sum_j),
                     std::abs(completed_step_reservoir_sum(report))});
}

inline double completed_step_energy_normalized_closure_residual(
    const CompletedStepEnergyReportV1& report)
{
    return std::abs(report.closure_residual_j) /
        completed_step_energy_closure_scale(report);
}

/** Require both the structural identity and an explicit physical tolerance. */
inline bool completed_step_energy_is_physically_closed(
    const CompletedStepEnergyReportV1& report,
    double relative_tolerance,
    double absolute_tolerance_j = 0.0)
{
    if (!structurally_valid_completed_step_energy_report(report) ||
        !report.complete || !std::isfinite(relative_tolerance) ||
        !std::isfinite(absolute_tolerance_j) || relative_tolerance < 0.0 ||
        absolute_tolerance_j < 0.0)
    {
        return false;
    }
    const double tolerance = std::max(
        absolute_tolerance_j,
        relative_tolerance * completed_step_energy_closure_scale(report));
    return std::abs(report.closure_residual_j) <= tolerance;
}

inline void accumulate_completed_step_energy(
    CompletedStepEnergyReportV1& destination,
    const CompletedStepEnergyReportV1& source)
{
    const bool destination_had_report = destination.complete;
    destination.acoustic_committed_change_j +=
        source.acoustic_committed_change_j;
    destination.slow_dynamics_committed_change_j +=
        source.slow_dynamics_committed_change_j;
    destination.scalar_transport_committed_change_j +=
        source.scalar_transport_committed_change_j;
    destination.thermodynamic_sources_committed_change_j +=
        source.thermodynamic_sources_committed_change_j;
    destination.sgs_committed_change_j += source.sgs_committed_change_j;
    destination.diffusion_committed_change_j +=
        source.diffusion_committed_change_j;
    destination.eos_projection_committed_change_j +=
        source.eos_projection_committed_change_j;
    destination.sponge_committed_change_j +=
        source.sponge_committed_change_j;
    destination.final_boundary_committed_change_j +=
        source.final_boundary_committed_change_j;
    destination.unsplit_rk_committed_change_j +=
        source.unsplit_rk_committed_change_j;
    destination.coordinate_remap_committed_change_j +=
        source.coordinate_remap_committed_change_j;
    destination.dry_internal_change_j += source.dry_internal_change_j;
    destination.water_sensible_change_j +=
        source.water_sensible_change_j;
    destination.phase_reference_change_j +=
        source.phase_reference_change_j;
    destination.carrier_kinetic_change_j +=
        source.carrier_kinetic_change_j;
    destination.gravitational_change_j +=
        source.gravitational_change_j;
    destination.subgrid_tke_change_j += source.subgrid_tke_change_j;
    destination.committed_owner_sum_j =
        completed_step_owner_sum(destination);
    destination.closure_residual_j =
        destination.committed_owner_sum_j
        - completed_step_reservoir_sum(destination);
    destination.complete =
        source.complete
        && (!destination_had_report || destination.complete);
}

} // namespace tmv

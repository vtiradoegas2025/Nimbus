/**
 * @file final_boundary_energy_report.hpp
 * @brief Exact basis-v2 attribution for the completed-step boundary closure.
 *
 * This report is intentionally separate from acoustic-substep attribution.
 * The final closure runs after EOS and sponge commits. On the C-grid it can
 * constrain physical wall/lid faces and the diagnosed terrain ground
 * velocity; on the collocated A-grid it changes excluded halo storage only.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>

namespace tmv
{

struct FinalBoundaryEnergyReportV1
{
    static constexpr std::uint32_t report_schema_version = 1;
    static constexpr std::uint32_t energy_basis_version = 2;

    double constraint_committed_work_j = 0.0;
    double dry_internal_change_j = 0.0;
    double water_sensible_change_j = 0.0;
    double phase_reference_change_j = 0.0;
    double carrier_kinetic_change_j = 0.0;
    double gravitational_change_j = 0.0;
    double subgrid_tke_change_j = 0.0;
    double committed_energy_change_j = 0.0;
    double closure_residual_j = 0.0;

    bool applied = false;
    bool complete = false;
};

inline bool valid_final_boundary_energy_report(
    const FinalBoundaryEnergyReportV1& report)
{
    const double values[] = {
        report.constraint_committed_work_j,
        report.dry_internal_change_j,
        report.water_sensible_change_j,
        report.phase_reference_change_j,
        report.carrier_kinetic_change_j,
        report.gravitational_change_j,
        report.subgrid_tke_change_j,
        report.committed_energy_change_j,
        report.closure_residual_j,
    };
    if (!std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return std::isfinite(value); }))
    {
        return false;
    }
    const bool all_zero = std::all_of(
        std::begin(values), std::end(values),
        [](double value) { return value == 0.0; });
    if (!report.complete)
        return all_zero && !report.applied;
    const double reservoir_sum =
        report.dry_internal_change_j
        + report.water_sensible_change_j
        + report.phase_reference_change_j
        + report.carrier_kinetic_change_j
        + report.gravitational_change_j
        + report.subgrid_tke_change_j;
    return report.applied
        && report.constraint_committed_work_j
            == report.carrier_kinetic_change_j
        && report.closure_residual_j
            == report.committed_energy_change_j - reservoir_sum;
}

inline void accumulate_final_boundary_energy(
    FinalBoundaryEnergyReportV1& destination,
    const FinalBoundaryEnergyReportV1& source)
{
    const bool destination_had_application = destination.applied;
    destination.constraint_committed_work_j +=
        source.constraint_committed_work_j;
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
    destination.committed_energy_change_j +=
        source.committed_energy_change_j;
    destination.closure_residual_j =
        destination.committed_energy_change_j
        - (destination.dry_internal_change_j
           + destination.water_sensible_change_j
           + destination.phase_reference_change_j
           + destination.carrier_kinetic_change_j
           + destination.gravitational_change_j
           + destination.subgrid_tke_change_j);
    destination.applied = destination.applied || source.applied;
    destination.complete =
        source.complete
        && (!destination_had_application || destination.complete);
}

} // namespace tmv

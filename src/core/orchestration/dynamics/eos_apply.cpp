#include "dynamics/eos_pressure.hpp"

#include "core/field/field_bounds.hpp"
#include "core/field/field_pool.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/deterministic_reduction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace tmv::dynamics_eos
{
namespace
{

bool field_matches_domain(const SimulationStateView& state,
                          const Field3D& field)
{
    return field.size_r() == state.grid.nr
        && field.size_th() == state.grid.nth
        && field.size_z() == state.grid.nz;
}

bool complete_moist_energy_state(const SimulationStateView& state)
{
    const auto& f = state.fields;
    return field_matches_domain(state, f.u)
        && field_matches_domain(state, f.v)
        && field_matches_domain(state, f.w)
        && field_matches_domain(state, f.rho)
        && field_matches_domain(state, f.p)
        && field_matches_domain(state, f.theta)
        && field_matches_domain(state, f.qv)
        && field_matches_domain(state, f.qc)
        && field_matches_domain(state, f.qr)
        && field_matches_domain(state, f.qi)
        && field_matches_domain(state, f.qs)
        && field_matches_domain(state, f.qg)
        && field_matches_domain(state, f.qh)
        && field_matches_domain(state, f.tke);
}

} // namespace

PressureProjectionEnergyReportV1 apply_pressure_projection(
    const SimulationStateView& state)
{
    auto& fields = state.fields;
    if (!field_matches_domain(state, fields.rho)
        || !field_matches_domain(state, fields.theta)
        || !field_matches_domain(state, fields.p))
    {
        throw std::invalid_argument(
            "EOS pressure projection requires complete rho/theta/p fields");
    }

    const bool attribute = complete_moist_energy_state(state);
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(state);
    if (attribute)
        tmv::physical_grid_metrics::validate(state);

    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double c_l =
        physical_constants::specific_heat_liquid_water_jkgk;
    constexpr double c_i =
        physical_constants::specific_heat_ice_jkgk;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;

    auto pressure_guard = FieldPool::instance().scoped_acquire_for_overwrite(
        state.grid.nr, state.grid.nth, state.grid.nz);
    Field3D& candidate_pressure = pressure_guard.field;

    const auto sums = tmv::diagnostics::fixed_slab_reduce<3>(
        0, state.grid.nr,
        [&](int i, std::array<double, 3>& partial)
        {
            for (int j = 0; j < state.grid.nth; ++j)
                for (int k = 0; k < state.grid.nz; ++k)
                {
                    const float rho = fields.rho[i][j][k];
                    const float theta = fields.theta[i][j][k];
                    const double diagnosed = pressure_fp64(rho, theta);
                    if (!std::isfinite(rho) || rho <= 0.0f
                        || !std::isfinite(theta) || theta <= 0.0f
                        || !std::isfinite(diagnosed) || diagnosed <= 0.0
                        || !std::isfinite(static_cast<float>(diagnosed)))
                    {
                        partial[2] += 1.0;
                        continue;
                    }

                    const float pressure_old = fields.p[i][j][k];
                    const float pressure_new = clamp_pressure_pa(
                        static_cast<float>(diagnosed));
                    if (!std::isfinite(pressure_new)
                        || pressure_new <= 0.0f)
                    {
                        partial[2] += 1.0;
                        continue;
                    }
                    candidate_pressure[i][j][k] = pressure_new;

                    if (!attribute
                        || i < domain.i_begin || i >= domain.i_end
                        || j < domain.j_begin || j >= domain.j_end
                        || k < domain.k_begin || k >= domain.k_end)
                    {
                        continue;
                    }

                    const double qv = fields.qv[i][j][k];
                    const double liquid =
                        static_cast<double>(fields.qc[i][j][k])
                        + static_cast<double>(fields.qr[i][j][k]);
                    const double ice =
                        static_cast<double>(fields.qi[i][j][k])
                        + static_cast<double>(fields.qs[i][j][k])
                        + static_cast<double>(fields.qg[i][j][k])
                        + static_cast<double>(fields.qh[i][j][k]);
                    const double dry_mass =
                        mass_geometry.scalar_mass_kg(fields.rho, i, j, k);
                    const double temperature_old =
                        static_cast<double>(theta) * std::pow(
                            static_cast<double>(pressure_old)
                                / physical_constants::reference_pressure_pa,
                            kappa);
                    const double temperature_new =
                        static_cast<double>(theta) * std::pow(
                            static_cast<double>(pressure_new)
                                / physical_constants::reference_pressure_pa,
                            kappa);
                    const double temperature_change =
                        temperature_new - temperature_old;
                    const double dry_change =
                        dry_mass * cv_d * temperature_change;
                    const double water_change = dry_mass * temperature_change
                        * (qv * cv_v + liquid * c_l + ice * c_i);
                    const double values[] = {
                        static_cast<double>(pressure_old),
                        static_cast<double>(pressure_new),
                        qv, liquid, ice, dry_mass,
                        temperature_old, temperature_new,
                        dry_change, water_change,
                    };
                    if (pressure_old <= 0.0f || qv < 0.0
                        || liquid < 0.0 || ice < 0.0 || dry_mass <= 0.0
                        || !std::all_of(
                            std::begin(values), std::end(values),
                            [](double value) {
                                return std::isfinite(value);
                            }))
                    {
                        partial[2] += 1.0;
                        continue;
                    }
                    partial[0] += dry_change;
                    partial[1] += water_change;
                }
        });

    if (sums[2] != 0.0)
    {
        throw std::runtime_error(
            "EOS pressure projection rejected invalid physical state");
    }

    std::copy_n(candidate_pressure.data(), candidate_pressure.size(), fields.p.data());

    PressureProjectionEnergyReportV1 report;
    if (!attribute)
        return report;
    report.dry_internal_change_j = sums[0];
    report.water_sensible_change_j = sums[1];
    report.committed_energy_change_j =
        report.dry_internal_change_j
        + report.water_sensible_change_j
        + report.phase_reference_change_j
        + report.carrier_kinetic_change_j
        + report.gravitational_change_j
        + report.subgrid_tke_change_j;
    report.closure_residual_j =
        report.committed_energy_change_j
        - (report.dry_internal_change_j
           + report.water_sensible_change_j
           + report.phase_reference_change_j
           + report.carrier_kinetic_change_j
           + report.gravitational_change_j
           + report.subgrid_tke_change_j);
    report.complete = true;
    return report;
}

void accumulate_pressure_projection_energy(
    PressureProjectionEnergyReportV1& destination,
    const PressureProjectionEnergyReportV1& source)
{
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
    destination.complete = destination.complete || source.complete;
}

} // namespace tmv::dynamics_eos

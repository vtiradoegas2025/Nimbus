// State-aware diffusion tendency computation and guarded application.

#include "core/orchestration/physics/diffusion_step.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/diffusion/linear_height_profile.hpp"
#include "numerics/numerics_base.hpp"
#include "terrain/conservative_transport.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

inline bool field_matches_domain(const SimulationStateView& state,
                                 const Field3D& field)
{
    return field.size_r() == state.grid.nr
        && field.size_th() == state.grid.nth
        && field.size_z() == state.grid.nz;
}

void copy_field_reusing_storage(const SimulationStateView& state,
                                Field3D& destination,
                                const Field3D& source)
{
    if (!field_matches_domain(state, destination))
        destination.resize(
            state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    std::copy(
        source.data(), source.data() + source.size(),
        destination.data());
}

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool diffusion_applies_to_momentum(const DiffusionConfig& config)
{
    const std::string apply_to = lower_copy(config.apply_to);
    // v1 config vocabulary is "scalars" / "scalars_and_momentum" (config_v1.cpp);
    // accept legacy "momentum"/"all" too. The previous check only matched
    // "momentum"/"all", which the v1 loader never emits -- so momentum diffusion
    // was dead via v1 configs.
    return apply_to == "scalars_and_momentum" || apply_to == "momentum" || apply_to == "all";
}

bool diffusion_applies_to_scalars(const DiffusionConfig& config)
{
    const std::string apply_to = lower_copy(config.apply_to);
    return apply_to == "scalars" || apply_to == "scalars_and_momentum" || apply_to == "all";
}

bool diffusion_runtime_has_work(const DiffusionRuntimeState& runtime)
{
    if (!std::isfinite(runtime.config.K_h) || runtime.config.K_h < 0.0 ||
        !std::isfinite(runtime.config.K_v) || runtime.config.K_v < 0.0)
        throw std::invalid_argument(
            "diffusion coefficients must be non-negative and finite");
    if (!runtime.scheme)
    {
        if (runtime.config.K_h > 0.0 || runtime.config.K_v > 0.0 ||
            runtime.config.use_variable_K)
            throw std::runtime_error(
                "active diffusion configuration has no initialized scheme");
        return false;
    }
    const double K_h = runtime.config.K_h;
    const double K_v = runtime.config.K_v;
    if (K_h <= 0.0 && K_v <= 0.0 && !runtime.config.use_variable_K)
    {
        return false;
    }

    const bool momentum = diffusion_applies_to_momentum(runtime.config);
    const bool scalars = diffusion_applies_to_scalars(runtime.config);
    if (!momentum && !scalars)
        throw std::invalid_argument(
            "diffusion apply_to has an unknown target");
    return true;
}

bool complete_moist_energy_state(const SimulationStateView& state)
{
    const Field3D* required[] = {
        &state.fields.rho, &state.fields.p, &state.fields.theta,
        &state.fields.u, &state.fields.v, &state.fields.w,
        &state.fields.qv, &state.fields.qc, &state.fields.qr,
        &state.fields.qi, &state.fields.qs, &state.fields.qg,
        &state.fields.qh, &state.fields.tke,
    };
    return std::all_of(
        std::begin(required), std::end(required),
        [&](const Field3D* field)
        {
            return field_matches_domain(state, *field);
        });
}

double diffusion_total_water(
    const SimulationStateView& state, int i, int j, int k)
{
    const auto& f = state.fields;
    const double values[] = {
        static_cast<double>(f.qv[i][j][k]),
        static_cast<double>(f.qc[i][j][k]),
        static_cast<double>(f.qr[i][j][k]),
        static_cast<double>(f.qi[i][j][k]),
        static_cast<double>(f.qs[i][j][k]),
        static_cast<double>(f.qg[i][j][k]),
        static_cast<double>(f.qh[i][j][k]),
    };
    double total = 0.0;
    for (double value : values)
    {
        if (!std::isfinite(value) || value < 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        total += value;
    }
    return total;
}

double diffusion_specific_kinetic_v2(
    const SimulationStateView& state,
    const Field3D& u, const Field3D& v, const Field3D& w,
    const TerrainMetrics3D* terrain,
    const PhysicalCellDomain& domain,
    int i, int j, int k)
{
    if (!state.grid.geometry.staggered)
    {
        const double uk = static_cast<double>(u[i][j][k]);
        const double vk = static_cast<double>(v[i][j][k]);
        const double wk = static_cast<double>(w[i][j][k]);
        return 0.5 * (uk * uk + vk * vk + wk * wk);
    }
    const double u_w = static_cast<double>(u[i - 1][j][k]);
    const double u_e = static_cast<double>(u[i][j][k]);
    const double v_s = static_cast<double>(v[i][j - 1][k]);
    const double v_n = static_cast<double>(v[i][j][k]);
    double w_b = k == domain.k_begin
        ? 0.0 : static_cast<double>(w[i][j][k - 1]);
    if (terrain != nullptr && k == domain.k_begin)
    {
        const double frame_u = state.domain_frame != nullptr
            ? state.domain_frame->velocity_ms.u_ms : 0.0;
        const double frame_v = state.domain_frame != nullptr
            ? state.domain_frame->velocity_ms.v_ms : 0.0;
        w_b = (0.5 * (u_w + u_e) - frame_u)
                * (-terrain->Gx_face(i, j, domain.k_begin))
            + (0.5 * (v_s + v_n) - frame_v)
                * (-terrain->Gy_face(i, j, domain.k_begin));
    }
    const double w_t = static_cast<double>(w[i][j][k]);
    return 0.25 * (
        u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n
        + w_b * w_b + w_t * w_t);
}

bool diffusion_scalar_energy_change(
    const SimulationStateView& simulation,
    const Field3D& committed_u,
    const Field3D& committed_v,
    const Field3D& committed_w,
    const Field3D& committed_theta,
    const Field3D& committed_qv,
    const TerrainMetrics3D* terrain,
    const PhysicalCellDomain& domain,
    const tmv::physical_grid_metrics::ControlVolumeMassView& mass_geometry,
    int i, int j, int k,
    std::array<double, 6>& change)
{
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
    constexpr double t_ref =
        physical_constants::freezing_temperature_k;
    constexpr double vapor_reference =
        physical_constants::latent_heat_vaporization_jkg
        - physical_constants::gas_constant_water_vapor_jkgk * t_ref;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    const auto& f = simulation.fields;
    const double dry_mass =
        mass_geometry.scalar_mass_kg(f.rho, i, j, k);
    const double pressure = f.p[i][j][k];
    const double exner = std::pow(
        pressure / physical_constants::reference_pressure_pa, kappa);
    const double theta_old = f.theta[i][j][k];
    const double theta_new = committed_theta[i][j][k];
    const double qv_old = f.qv[i][j][k];
    const double qv_new = committed_qv[i][j][k];
    const double liquid =
        static_cast<double>(f.qc[i][j][k])
        + static_cast<double>(f.qr[i][j][k]);
    const double ice =
        static_cast<double>(f.qi[i][j][k])
        + static_cast<double>(f.qs[i][j][k])
        + static_cast<double>(f.qg[i][j][k])
        + static_cast<double>(f.qh[i][j][k]);
    const double temperature_old = theta_old * exner;
    const double temperature_new = theta_new * exner;
    const double specific_kinetic =
        diffusion_specific_kinetic_v2(
            simulation, committed_u, committed_v, committed_w,
            terrain, domain, i, j, k);
    const double z =
        tmv::physical_grid_metrics::scalar_height(simulation, i, j, k);
    change[0] = dry_mass * cv_d
        * (temperature_new - temperature_old);
    change[1] = dry_mass * (
        (temperature_new - t_ref)
            * (qv_new * cv_v + liquid * c_l + ice * c_i)
        - (temperature_old - t_ref)
            * (qv_old * cv_v + liquid * c_l + ice * c_i));
    change[2] = dry_mass
        * (qv_new - qv_old) * vapor_reference;
    change[3] = dry_mass
        * (qv_new - qv_old) * specific_kinetic;
    change[4] = dry_mass
        * (qv_new - qv_old)
        * physical_constants::gravity_ms2 * z;
    change[5] = 0.0;
    const double values[] = {
        dry_mass, pressure, exner, theta_old, theta_new,
        qv_old, qv_new, liquid, ice,
        temperature_old, temperature_new, specific_kinetic, z,
        change[0], change[1], change[2], change[3], change[4],
    };
    return dry_mass > 0.0 && pressure > 0.0
        && theta_old > 0.0 && theta_new > 0.0
        && qv_old >= 0.0 && qv_new >= 0.0
        && liquid >= 0.0 && ice >= 0.0
        && std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return std::isfinite(value); });
}

void log_runtime_diffusion_path_once(const DiffusionRuntimeState& runtime)
{
    static bool logged = false;
    if (logged)
    {
        return;
    }
    logged = true;

    if (!runtime.scheme)
    {
        tmv::log_info("[DIFFUSION] Runtime path: disabled (no diffusion scheme).");
        return;
    }

    tmv::log_info("[DIFFUSION] Runtime path: src/numerics/diffusion/",
                  runtime.scheme->name(),
                  " (apply_to=", runtime.config.apply_to,
                  ", K_h=", runtime.config.K_h,
                  ", K_v=", runtime.config.K_v,
                  ", coefficient_source=", runtime.config.coefficient_source,
                  ", K_v_surface=", runtime.config.K_v_surface,
                  ", K_v_top=", runtime.config.K_v_top,
                  ")");
}

GridMetrics grid_metrics_for(const SimulationStateView& state)
{
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::active_terrain_metrics(state);
    if (state.process_grid_metrics && terrain_metrics == nullptr &&
        state.process_grid_metrics->dz.size() == static_cast<std::size_t>(state.grid.nz))
        return *state.process_grid_metrics;

    GridMetrics grid;
    grid.dx = state.grid.dr;
    grid.dy = representative_dy(
        state.grid.geometry.coord, state.grid.dr, state.grid.dtheta);
    grid.staggering = state.grid.geometry.staggered
        ? StaggerType::CGrid : StaggerType::Collocated;
    grid.dz.assign(static_cast<std::size_t>(state.grid.nz), state.grid.dz);
    if (terrain_metrics != nullptr)
    {
        tmv::physical_grid_metrics::validate(state);
        grid.terrain_metrics_active = true;
        grid.terrain_metrics = terrain_metrics;
        grid.terrain_topography = &state.terrain_runtime->topography;
        for (int k = 0; k < state.grid.nz; ++k)
        {
            double minimum = std::numeric_limits<double>::infinity();
            for (int i = 0; i < state.grid.nr; ++i)
                for (int j = 0; j < state.grid.nth; ++j)
                    minimum = std::min(
                        minimum,
                        terrain_metrics->z_face(i, j, k + 1) -
                            terrain_metrics->z_face(i, j, k));
            if (!std::isfinite(minimum) || minimum <= 0.0)
                throw std::invalid_argument(
                    "terrain diffusion layer thickness must be positive and finite");
            grid.dz[static_cast<std::size_t>(k)] = minimum;
        }
    }
    grid.z_int.resize(static_cast<std::size_t>(state.grid.nz + 1));
    grid.z_int[0] = 0.0;
    for (int k = 1; k <= state.grid.nz; ++k)
    {
        grid.z_int[static_cast<std::size_t>(k)] =
            grid.z_int[static_cast<std::size_t>(k - 1)] +
            grid.dz[static_cast<std::size_t>(k - 1)];
    }
    return grid;
}

DiffusionMomentumWorkReport attribute_diffusion_momentum_work(
    const SimulationStateView& simulation,
    const DiffusionStateView& state,
    const DiffusionTendencies& tendencies,
    const Field3D& committed_u,
    const Field3D& committed_v,
    const Field3D& committed_w,
    const Field3D& committed_theta,
    const Field3D& committed_qv,
    bool do_momentum,
    bool do_scalars,
    double dt_dynamics)
{
    DiffusionMomentumWorkReport report;
    if (!complete_moist_energy_state(simulation))
        return report;
    tmv::physical_grid_metrics::validate(simulation);
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(
        simulation);
    const PhysicalCellDomain domain =
        physical_scalar_domain(simulation.grid);
    const TerrainMetrics3D* terrain =
        tmv::physical_grid_metrics::active_terrain_metrics(simulation);
    const bool terrain_faces_complete = terrain == nullptr
        || (terrain->Gx_face.size_r() == simulation.grid.nr
            && terrain->Gx_face.size_th() == simulation.grid.nth
            && terrain->Gx_face.size_z() == simulation.grid.nz + 1
            && terrain->Gy_face.size_r() == simulation.grid.nr
            && terrain->Gy_face.size_th() == simulation.grid.nth
            && terrain->Gy_face.size_z() == simulation.grid.nz + 1);
    if (!terrain_faces_complete)
        throw std::invalid_argument(
            "mapped diffusion moist-energy attribution requires complete "
            "terrain face metrics");
    const auto carrier_mass = [&](int i, int j, int k)
    {
        return mass_geometry.scalar_mass_kg(
            simulation.fields.rho, i, j, k)
            * (1.0 + diffusion_total_water(simulation, i, j, k));
    };

    const auto sums = tmv::diagnostics::fixed_slab_reduce<12>(
        domain.i_begin, domain.i_end,
        [&](int i, std::array<double, 12>& partial)
        {
            const auto momentum =
                [&](double mass, double old_value, double new_value,
                    double rate)
                {
                    const double delta = rate * dt_dynamics;
                    const double values[] = {
                        mass, old_value, new_value, delta,
                    };
                    if (mass <= 0.0 || std::any_of(
                            std::begin(values), std::end(values),
                            [](double value) {
                                return !std::isfinite(value);
                            }))
                    {
                        partial[11] += 1.0;
                        return;
                    }
                    partial[0] += mass * old_value * delta;
                    partial[1] += 0.5 * mass * delta * delta;
                    partial[2] += 0.5 * mass
                        * (new_value * new_value - old_value * old_value);
                };
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double cell_carrier_mass = carrier_mass(i, j, k);
                    if (do_momentum)
                    {
                        if (!state.c_grid)
                        {
                            momentum(
                                cell_carrier_mass,
                                simulation.fields.u[i][j][k],
                                committed_u[i][j][k],
                                tendencies.dudt_diff[i][j][k]);
                            momentum(
                                cell_carrier_mass,
                                simulation.fields.v[i][j][k],
                                committed_v[i][j][k],
                                tendencies.dvdt_diff[i][j][k]);
                            momentum(
                                cell_carrier_mass,
                                simulation.fields.w[i][j][k],
                                committed_w[i][j][k],
                                tendencies.dwdt_diff[i][j][k]);
                        }
                        else
                        {
                            // Mirror the basis-v2 endpoint definition
                            // cell-for-cell. Each scalar cell owns one half of
                            // each adjacent stored velocity face. This remains
                            // exact when periodic seam slots temporarily
                            // differ before the final boundary closure.
                            const double half_mass =
                                0.5 * cell_carrier_mass;
                            momentum(
                                half_mass,
                                simulation.fields.u[i - 1][j][k],
                                committed_u[i - 1][j][k],
                                tendencies.dudt_diff[i - 1][j][k]);
                            momentum(
                                half_mass, simulation.fields.u[i][j][k],
                                committed_u[i][j][k],
                                tendencies.dudt_diff[i][j][k]);
                            momentum(
                                half_mass,
                                simulation.fields.v[i][j - 1][k],
                                committed_v[i][j - 1][k],
                                tendencies.dvdt_diff[i][j - 1][k]);
                            momentum(
                                half_mass, simulation.fields.v[i][j][k],
                                committed_v[i][j][k],
                                tendencies.dvdt_diff[i][j][k]);
                            if (k > domain.k_begin)
                            {
                                momentum(
                                    half_mass,
                                    simulation.fields.w[i][j][k - 1],
                                    committed_w[i][j][k - 1],
                                    tendencies.dwdt_diff[i][j][k - 1]);
                            }
                            momentum(
                                half_mass, simulation.fields.w[i][j][k],
                                committed_w[i][j][k],
                                tendencies.dwdt_diff[i][j][k]);

                            if (terrain != nullptr && k == domain.k_begin)
                            {
                                const double slope_x =
                                    -terrain->Gx_face(
                                        i, j, domain.k_begin);
                                const double slope_y =
                                    -terrain->Gy_face(
                                        i, j, domain.k_begin);
                                const double frame_u =
                                    simulation.domain_frame != nullptr
                                    ? simulation.domain_frame->velocity_ms.u_ms
                                    : 0.0;
                                const double frame_v =
                                    simulation.domain_frame != nullptr
                                    ? simulation.domain_frame->velocity_ms.v_ms
                                    : 0.0;
                                const double old_ground =
                                    (0.5 * (
                                        simulation.fields.u[i - 1][j][k]
                                        + simulation.fields.u[i][j][k])
                                        - frame_u)
                                        * slope_x
                                    + (0.5 * (
                                        simulation.fields.v[i][j - 1][k]
                                        + simulation.fields.v[i][j][k])
                                        - frame_v)
                                        * slope_y;
                                const double new_ground =
                                    (0.5 * (
                                        committed_u[i - 1][j][k]
                                        + committed_u[i][j][k])
                                        - frame_u)
                                        * slope_x
                                    + (0.5 * (
                                        committed_v[i][j - 1][k]
                                        + committed_v[i][j][k])
                                        - frame_v)
                                        * slope_y;
                                const double ground_rate =
                                    0.5 * (
                                        tendencies.dudt_diff[i - 1][j][k]
                                        + tendencies.dudt_diff[i][j][k])
                                        * slope_x
                                    + 0.5 * (
                                        tendencies.dvdt_diff[i][j - 1][k]
                                        + tendencies.dvdt_diff[i][j][k])
                                        * slope_y;
                                momentum(
                                    half_mass, old_ground, new_ground,
                                    ground_rate);
                            }
                        }
                    }

                    if (!do_scalars)
                        continue;
                    std::array<double, 6> scalar_change{};
                    if (!diffusion_scalar_energy_change(
                            simulation,
                            committed_u, committed_v, committed_w,
                            committed_theta, committed_qv,
                            terrain, domain, mass_geometry,
                            i, j, k, scalar_change))
                    {
                        partial[11] += 1.0;
                        continue;
                    }
                    if (do_momentum)
                    {
                        // The simultaneous qv/momentum carrier change must use
                        // one before/after reduction; separately reduced
                        // partitions do not reproduce the endpoint inventory.
                        const double dry_mass =
                            mass_geometry.scalar_mass_kg(
                                simulation.fields.rho, i, j, k);
                        const double old_water =
                            diffusion_total_water(simulation, i, j, k);
                        const double new_water = old_water
                            + static_cast<double>(committed_qv[i][j][k])
                            - static_cast<double>(
                                simulation.fields.qv[i][j][k]);
                        const double old_kinetic =
                            diffusion_specific_kinetic_v2(
                                simulation,
                                simulation.fields.u,
                                simulation.fields.v,
                                simulation.fields.w,
                                terrain, domain, i, j, k);
                        const double new_kinetic =
                            diffusion_specific_kinetic_v2(
                                simulation,
                                committed_u, committed_v, committed_w,
                                terrain, domain, i, j, k);
                        const double old_carrier =
                            dry_mass * (1.0 + old_water) * old_kinetic;
                        const double new_carrier =
                            dry_mass * (1.0 + new_water) * new_kinetic;
                        if (!std::isfinite(old_carrier)
                            || !std::isfinite(new_carrier)
                            || new_water < 0.0)
                        {
                            partial[11] += 1.0;
                            continue;
                        }
                        partial[3] += old_carrier;
                        partial[4] += new_carrier;
                    }
                    for (std::size_t reservoir = 0;
                         reservoir < scalar_change.size(); ++reservoir)
                    {
                        partial[5 + reservoir] += scalar_change[reservoir];
                    }
                }
        });
    if (sums[11] != 0.0)
        throw std::runtime_error(
            "diffusion energy attribution rejected invalid state");
    report.requested_linear_work_j = sums[0];
    report.finite_step_correction_j = sums[1];
    report.committed_work_j = sums[2];
    report.storage_rounding_j = sums[2] - sums[0] - sums[1];
    report.moist_dry_internal_change_j = sums[5];
    report.moist_water_sensible_change_j = sums[6];
    report.moist_phase_reference_change_j = sums[7];
    report.moist_carrier_kinetic_change_j = do_momentum
        ? (sums[4] - sums[3]) - report.committed_work_j
        : sums[8];
    report.moist_gravitational_change_j = sums[9];
    report.moist_subgrid_tke_change_j = sums[10];
    report.moist_committed_energy_change_j =
        report.committed_work_j
        + report.moist_dry_internal_change_j
        + report.moist_water_sensible_change_j
        + report.moist_phase_reference_change_j
        + report.moist_carrier_kinetic_change_j
        + report.moist_gravitational_change_j
        + report.moist_subgrid_tke_change_j;
    report.moist_energy_closure_residual_j =
        report.moist_committed_energy_change_j
        - (report.committed_work_j
           + report.moist_dry_internal_change_j
           + report.moist_water_sensible_change_j
           + report.moist_phase_reference_change_j
           + report.moist_carrier_kinetic_change_j
           + report.moist_gravitational_change_j
           + report.moist_subgrid_tke_change_j);
    report.moist_energy_complete = true;
    if (!std::isfinite(report.requested_linear_work_j)
        || !std::isfinite(report.finite_step_correction_j)
        || !std::isfinite(report.storage_rounding_j)
        || !std::isfinite(report.committed_work_j)
        || !std::isfinite(report.moist_committed_energy_change_j)
        || !std::isfinite(report.moist_energy_closure_residual_j)
        || report.finite_step_correction_j < 0.0)
        throw std::runtime_error(
            "diffusion energy attribution produced an invalid ledger");
    return report;
}

void accumulate_diffusion_energy_work(
    DiffusionMomentumWorkReport& destination,
    const DiffusionMomentumWorkReport& source)
{
    destination.requested_linear_work_j += source.requested_linear_work_j;
    destination.finite_step_correction_j +=
        source.finite_step_correction_j;
    destination.storage_rounding_j += source.storage_rounding_j;
    destination.committed_work_j += source.committed_work_j;
    destination.moist_dry_internal_change_j +=
        source.moist_dry_internal_change_j;
    destination.moist_water_sensible_change_j +=
        source.moist_water_sensible_change_j;
    destination.moist_phase_reference_change_j +=
        source.moist_phase_reference_change_j;
    destination.moist_carrier_kinetic_change_j +=
        source.moist_carrier_kinetic_change_j;
    destination.moist_gravitational_change_j +=
        source.moist_gravitational_change_j;
    destination.moist_subgrid_tke_change_j +=
        source.moist_subgrid_tke_change_j;
    destination.moist_committed_energy_change_j +=
        source.moist_committed_energy_change_j;
    const double sum =
        destination.committed_work_j
        + destination.moist_dry_internal_change_j
        + destination.moist_water_sensible_change_j
        + destination.moist_phase_reference_change_j
        + destination.moist_carrier_kinetic_change_j
        + destination.moist_gravitational_change_j
        + destination.moist_subgrid_tke_change_j;
    destination.moist_energy_closure_residual_j =
        destination.moist_committed_energy_change_j - sum;
    destination.moist_energy_complete =
        destination.moist_energy_complete || source.moist_energy_complete;
}

}  // namespace

bool runtime_diffusion_enabled(const SimulationStateView& simulation)
{
    const DiffusionRuntimeState& runtime = simulation.diffusion_runtime
        ? *simulation.diffusion_runtime : tmv::require_level_runtime(simulation.diffusion_runtime, "diffusion_runtime");
    return diffusion_runtime_has_work(runtime);
}

void apply_runtime_diffusion(const SimulationStateView& simulation,
                             double dt_dynamics)
{
    DiffusionRuntimeState& runtime = simulation.diffusion_runtime
        ? *simulation.diffusion_runtime : tmv::require_level_runtime(simulation.diffusion_runtime, "diffusion_runtime");
    if (!diffusion_runtime_has_work(runtime))
    {
        return;
    }

    log_runtime_diffusion_path_once(runtime);

    const bool do_momentum = diffusion_applies_to_momentum(runtime.config);
    const bool do_scalars = diffusion_applies_to_scalars(runtime.config);

    if (!std::isfinite(dt_dynamics) || dt_dynamics <= 0.0)
        throw std::invalid_argument(
            "diffusion dynamics timestep must be positive and finite");
    const auto require_state_shape = [&](const Field3D& field,
                                         const char* name)
    {
        if (!field_matches_domain(simulation, field))
            throw std::invalid_argument(
                std::string("diffusion state field shape mismatch: ") + name);
    };
    require_state_shape(simulation.fields.rho, "rho");
    if (do_momentum)
    {
        require_state_shape(simulation.fields.u, "u");
        require_state_shape(simulation.fields.v, "v");
        require_state_shape(simulation.fields.w, "w");
    }
    if (do_scalars)
    {
        require_state_shape(simulation.fields.theta, "theta");
        require_state_shape(simulation.fields.qv, "qv");
    }

    DiffusionConfig runtime_cfg = runtime.config;
    runtime_cfg.dt_diffusion = dt_dynamics;

    const GridMetrics grid = grid_metrics_for(simulation);
    if (!std::isfinite(grid.dx) || grid.dx <= 0.0 ||
        !std::isfinite(grid.dy) || grid.dy <= 0.0 ||
        grid.dz.size() != static_cast<std::size_t>(simulation.grid.nz))
        throw std::invalid_argument(
            "diffusion grid metrics are missing or invalid");
    for (double layer_dz : grid.dz)
    {
        if (!std::isfinite(layer_dz) || layer_dz <= 0.0)
            throw std::invalid_argument(
                "diffusion layer thickness must be positive and finite");
    }
    DiffusionStateView state{};
    tmv::bind_flow_execution(state, simulation);
    state.grid = &grid;
    state.rho = &simulation.fields.rho;
    state.c_grid = simulation.grid.geometry.staggered;
    state.periodic_x =
        tmv::require_run_configuration(simulation).lateral_bc_west == tmv::init::config::LateralBC::Periodic
        && tmv::require_run_configuration(simulation).lateral_bc_east == tmv::init::config::LateralBC::Periodic;
    state.periodic_y =
        tmv::require_run_configuration(simulation).lateral_bc_south == tmv::init::config::LateralBC::Periodic
        && tmv::require_run_configuration(simulation).lateral_bc_north == tmv::init::config::LateralBC::Periodic;
    const Field3D* variable_k = tmv::prescribed_vertical_diffusivity(
        runtime, simulation.grid.nr, simulation.grid.nth, grid);
    state.K_momentum = variable_k;
    state.K_scalar = variable_k;
    if (do_momentum)
    {
        state.u = &simulation.fields.u;
        state.v = &simulation.fields.v;
        state.w = &simulation.fields.w;
    }
    if (do_scalars)
    {
        state.theta = &simulation.fields.theta;
        state.qv = &simulation.fields.qv;
    }

    DiffusionTendencies candidate_tendencies;
    DiffusionDiagnostics candidate_diagnostics;
    runtime.scheme->compute_diffusion_tendencies(
        runtime_cfg, state, candidate_tendencies, &candidate_diagnostics);

    const bool shape_ok =
        field_matches_domain(simulation, candidate_tendencies.dudt_diff) &&
        field_matches_domain(simulation, candidate_tendencies.dvdt_diff) &&
        field_matches_domain(simulation, candidate_tendencies.dwdt_diff) &&
        field_matches_domain(simulation, candidate_tendencies.dthetadt_diff) &&
        field_matches_domain(simulation, candidate_tendencies.dqvdt_diff);
    if (!shape_ok)
        throw std::runtime_error(
            "diffusion tendency buffers have invalid shape");

    Field3D& candidate_u = runtime.scratch_u;
    Field3D& candidate_v = runtime.scratch_v;
    Field3D& candidate_w = runtime.scratch_w;
    Field3D& candidate_theta = runtime.scratch_theta;
    Field3D& candidate_qv = runtime.scratch_qv;
    if (do_momentum)
    {
        copy_field_reusing_storage(
            simulation, candidate_u, simulation.fields.u);
        copy_field_reusing_storage(
            simulation, candidate_v, simulation.fields.v);
        copy_field_reusing_storage(
            simulation, candidate_w, simulation.fields.w);
    }
    if (do_scalars)
    {
        copy_field_reusing_storage(
            simulation, candidate_theta, simulation.fields.theta);
        copy_field_reusing_storage(
            simulation, candidate_qv, simulation.fields.qv);
    }

    int invalid_updates = 0;
    const bool inline_scalar_attribution =
        do_scalars && !do_momentum
        && complete_moist_energy_state(simulation);
    const PhysicalCellDomain inline_domain =
        physical_scalar_domain(simulation.grid);
    const TerrainMetrics3D* inline_terrain =
        tmv::physical_grid_metrics::active_terrain_metrics(simulation);
    const tmv::physical_grid_metrics::ControlVolumeMassView
        inline_mass_geometry(simulation);
    std::array<double, 7> inline_scalar_energy{};
    if (inline_scalar_attribution)
    {
        tmv::physical_grid_metrics::validate(simulation);
        const bool terrain_faces_complete = inline_terrain == nullptr
            || (inline_terrain->Gx_face.size_r() == simulation.grid.nr
                && inline_terrain->Gx_face.size_th() == simulation.grid.nth
                && inline_terrain->Gx_face.size_z() == simulation.grid.nz + 1
                && inline_terrain->Gy_face.size_r() == simulation.grid.nr
                && inline_terrain->Gy_face.size_th() == simulation.grid.nth
                && inline_terrain->Gy_face.size_z() == simulation.grid.nz + 1);
        if (!terrain_faces_complete)
            throw std::invalid_argument(
                "mapped diffusion moist-energy attribution requires "
                "complete terrain face metrics");
    }

    if (do_momentum)
    {
        #pragma omp parallel for collapse(2) reduction(+:invalid_updates)
        for (int i = 0; i < simulation.grid.nr; ++i)
        {
            for (int j = 0; j < simulation.grid.nth; ++j)
            {
                for (int k = 0; k < simulation.grid.nz; ++k)
                {
                    const double dudt = candidate_tendencies.dudt_diff[i][j][k];
                    const double dvdt = candidate_tendencies.dvdt_diff[i][j][k];
                    const double dwdt = candidate_tendencies.dwdt_diff[i][j][k];

                    double u_new = static_cast<double>(simulation.fields.u[i][j][k]) + dudt * dt_dynamics;
                    double v_new = static_cast<double>(simulation.fields.v[i][j][k]) + dvdt * dt_dynamics;
                    double w_new = static_cast<double>(simulation.fields.w[i][j][k]) + dwdt * dt_dynamics;

                    const float u_value = static_cast<float>(u_new);
                    const float v_value = static_cast<float>(v_new);
                    const float w_value = static_cast<float>(w_new);
                    if (!std::isfinite(dudt) || !std::isfinite(dvdt) ||
                        !std::isfinite(dwdt) || !std::isfinite(u_value) ||
                        !std::isfinite(v_value) || !std::isfinite(w_value))
                    {
                        ++invalid_updates;
                        continue;
                    }
                    candidate_u[i][j][k] = clamp_wind_horizontal_ms(u_value);
                    candidate_v[i][j][k] = clamp_wind_horizontal_ms(v_value);
                    candidate_w[i][j][k] = clamp_wind_vertical_ms(w_value);
                }
            }
        }
    }

    if (do_scalars)
    {
        const auto commit_scalar_cell = [&](int i, int j, int k)
        {
            const double dthetadt =
                candidate_tendencies.dthetadt_diff[i][j][k];
            const double dqvdt =
                candidate_tendencies.dqvdt_diff[i][j][k];
            const double theta_old =
                static_cast<double>(simulation.fields.theta[i][j][k]);
            const double theta_new = theta_old + dthetadt * dt_dynamics;
            const double qv_new =
                static_cast<double>(simulation.fields.qv[i][j][k])
                + dqvdt * dt_dynamics;
            const float theta_value = static_cast<float>(theta_new);
            const float qv_value = static_cast<float>(qv_new);
            if (!std::isfinite(dthetadt) || !std::isfinite(dqvdt)
                || !std::isfinite(theta_value) || !std::isfinite(qv_value))
            {
                return false;
            }
            candidate_theta[i][j][k] = clamp_theta_k(theta_value);
            candidate_qv[i][j][k] = clamp_qv_kgkg(qv_value);
            return true;
        };
        if (inline_scalar_attribution)
        {
            inline_scalar_energy =
                tmv::diagnostics::fixed_slab_reduce<7>(
                    0, simulation.grid.nr,
                    [&](int i, std::array<double, 7>& partial)
                    {
                        for (int j = 0; j < simulation.grid.nth; ++j)
                            for (int k = 0; k < simulation.grid.nz; ++k)
                            {
                                if (!commit_scalar_cell(i, j, k))
                                {
                                    partial[6] += 1.0;
                                    continue;
                                }
                                if (i < inline_domain.i_begin
                                    || i >= inline_domain.i_end
                                    || j < inline_domain.j_begin
                                    || j >= inline_domain.j_end
                                    || k < inline_domain.k_begin
                                    || k >= inline_domain.k_end)
                                    continue;
                                std::array<double, 6> change{};
                                if (!diffusion_scalar_energy_change(
                                        simulation,
                                        simulation.fields.u,
                                        simulation.fields.v,
                                        simulation.fields.w,
                                        candidate_theta, candidate_qv,
                                        inline_terrain, inline_domain,
                                        inline_mass_geometry,
                                        i, j, k, change))
                                {
                                    partial[6] += 1.0;
                                    continue;
                                }
                                for (std::size_t reservoir = 0;
                                     reservoir < change.size(); ++reservoir)
                                    partial[reservoir] += change[reservoir];
                            }
                    });
            invalid_updates +=
                static_cast<int>(inline_scalar_energy[6]);
        }
        else
        {
            #pragma omp parallel for collapse(2) reduction(+:invalid_updates)
            for (int i = 0; i < simulation.grid.nr; ++i)
                for (int j = 0; j < simulation.grid.nth; ++j)
                    for (int k = 0; k < simulation.grid.nz; ++k)
                    {
                        if (!commit_scalar_cell(i, j, k))
                            ++invalid_updates;
                    }
        }
    }

    if (invalid_updates > 0)
    {
        tmv::record_numerical_nonfinite_incidents(
            static_cast<std::size_t>(invalid_updates));
        throw std::runtime_error(
            "diffusion update produced non-finite tendencies or state");
    }
    if (!std::isfinite(candidate_diagnostics.max_diffusion_number) ||
        candidate_diagnostics.max_diffusion_number < 0.0)
        throw std::runtime_error(
            "diffusion scheme returned an invalid stability diagnostic");
    if (lower_copy(runtime.config.scheme_id) == "explicit" &&
        candidate_diagnostics.max_diffusion_number > 1.0)
    {
        throw std::runtime_error(
            "explicit diffusion stability limit exceeded; reduce dt, K_h, or K_v");
    }

    runtime.last_momentum_work = {};
    const Field3D& committed_u =
        do_momentum ? candidate_u : simulation.fields.u;
    const Field3D& committed_v =
        do_momentum ? candidate_v : simulation.fields.v;
    const Field3D& committed_w =
        do_momentum ? candidate_w : simulation.fields.w;
    const Field3D& committed_theta =
        do_scalars ? candidate_theta : simulation.fields.theta;
    const Field3D& committed_qv =
        do_scalars ? candidate_qv : simulation.fields.qv;
    DiffusionMomentumWorkReport report;
    if (inline_scalar_attribution)
    {
        report.moist_dry_internal_change_j = inline_scalar_energy[0];
        report.moist_water_sensible_change_j = inline_scalar_energy[1];
        report.moist_phase_reference_change_j = inline_scalar_energy[2];
        report.moist_carrier_kinetic_change_j = inline_scalar_energy[3];
        report.moist_gravitational_change_j = inline_scalar_energy[4];
        report.moist_subgrid_tke_change_j = inline_scalar_energy[5];
        report.moist_committed_energy_change_j =
            inline_scalar_energy[0] + inline_scalar_energy[1]
            + inline_scalar_energy[2] + inline_scalar_energy[3]
            + inline_scalar_energy[4] + inline_scalar_energy[5];
        report.moist_energy_closure_residual_j =
            report.moist_committed_energy_change_j
            - (inline_scalar_energy[0] + inline_scalar_energy[1]
               + inline_scalar_energy[2] + inline_scalar_energy[3]
               + inline_scalar_energy[4] + inline_scalar_energy[5]);
        report.moist_energy_complete = true;
    }
    else
    {
        report = attribute_diffusion_momentum_work(
            simulation, state, candidate_tendencies,
            committed_u, committed_v, committed_w,
            committed_theta, committed_qv,
            do_momentum, do_scalars, dt_dynamics);
    }
    runtime.last_momentum_work = report;
    if (do_momentum)
    {
        runtime.cumulative_requested_momentum_work_j +=
            report.requested_linear_work_j;
        runtime.cumulative_finite_step_momentum_correction_j +=
            report.finite_step_correction_j;
        runtime.cumulative_momentum_storage_rounding_j +=
            report.storage_rounding_j;
        runtime.cumulative_committed_momentum_work_j +=
            report.committed_work_j;
        ++runtime.momentum_work_step_count;
    }
    if (report.moist_energy_complete)
    {
        accumulate_diffusion_energy_work(
            runtime.cumulative_moist_energy_work, report);
        ++runtime.moist_energy_step_count;
    }

    if (do_momentum)
    {
        std::swap(simulation.fields.u, candidate_u);
        std::swap(simulation.fields.v, candidate_v);
        std::swap(simulation.fields.w, candidate_w);
    }
    if (do_scalars)
    {
        std::swap(simulation.fields.theta, candidate_theta);
        std::swap(simulation.fields.qv, candidate_qv);
    }
    runtime.tendencies = std::move(candidate_tendencies);
    runtime.diagnostics = std::move(candidate_diagnostics);
}

void apply_runtime_diffusion(double dt_dynamics)
{
    apply_runtime_diffusion(global_simulation_state(), dt_dynamics);
}

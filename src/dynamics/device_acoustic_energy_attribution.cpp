#include "dynamics/device_acoustic_energy_attribution.hpp"

#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/infra/domain_motion.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "dynamics/acoustic_energy_report.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace tmv::device_acoustic_energy
{
namespace
{

double cell_volume(const SimulationStateView& state, int k)
{
    const auto& grid = state.grid;
    const double lower = grid.geometry.z_faces.empty()
        ? static_cast<double>(k) * grid.dz
        : grid.geometry.z_faces[static_cast<std::size_t>(k)];
    const double upper = grid.geometry.z_faces.empty()
        ? static_cast<double>(k + 1) * grid.dz
        : grid.geometry.z_faces[static_cast<std::size_t>(k + 1)];
    return grid.dr * grid.dtheta * (upper - lower);
}

double scalar_height(const SimulationStateView& state, int k)
{
    return state.grid.geometry.z.empty()
        ? (static_cast<double>(k) + 0.5) * state.grid.dz
        : state.grid.geometry.z[static_cast<std::size_t>(k)];
}

double mixture_mass(
    const SimulationStateView& state, double density,
    int i, int j, int k)
{
    const auto& f = state.fields;
    const double total_water =
        static_cast<double>(f.qv[i][j][k])
        + static_cast<double>(f.qc[i][j][k])
        + static_cast<double>(f.qr[i][j][k])
        + static_cast<double>(f.qi[i][j][k])
        + static_cast<double>(f.qs[i][j][k])
        + static_cast<double>(f.qg[i][j][k])
        + static_cast<double>(f.qh[i][j][k]);
    return density * (1.0 + total_water) * cell_volume(state, k);
}

std::array<double, 6> reservoirs(
    const SimulationStateView& state,
    const Field3D& velocity_u,
    const Field3D& velocity_v,
    const Field3D& velocity_w,
    double density,
    double pressure,
    int i, int j, int k)
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
    constexpr double ice_reference =
        -physical_constants::latent_heat_fusion_jkg;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;

    const auto& f = state.fields;
    const double vapor = static_cast<double>(f.qv[i][j][k]);
    const double liquid =
        static_cast<double>(f.qc[i][j][k])
        + static_cast<double>(f.qr[i][j][k]);
    const double ice =
        static_cast<double>(f.qi[i][j][k])
        + static_cast<double>(f.qs[i][j][k])
        + static_cast<double>(f.qg[i][j][k])
        + static_cast<double>(f.qh[i][j][k]);
    const double dry_mass = density * cell_volume(state, k);
    const double moist_mass = dry_mass * (1.0 + vapor + liquid + ice);
    const double temperature =
        static_cast<double>(f.theta[i][j][k])
        * std::pow(
            pressure / physical_constants::reference_pressure_pa, kappa);
    const double delta_temperature = temperature - t_ref;
    const double u_w = static_cast<double>(velocity_u[i - 1][j][k]);
    const double u_e = static_cast<double>(velocity_u[i][j][k]);
    const double v_s = static_cast<double>(velocity_v[i][j - 1][k]);
    const double v_n = static_cast<double>(velocity_v[i][j][k]);
    const double w_b = k == 0
        ? 0.0 : static_cast<double>(velocity_w[i][j][k - 1]);
    const double w_t = static_cast<double>(velocity_w[i][j][k]);

    return {{
        dry_mass * cv_d * delta_temperature,
        dry_mass * delta_temperature
            * (vapor * cv_v + liquid * c_l + ice * c_i),
        dry_mass * (vapor * vapor_reference + ice * ice_reference),
        0.25 * moist_mass
            * (u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n
               + w_b * w_b + w_t * w_t),
        moist_mass * physical_constants::gravity_ms2
            * scalar_height(state, k),
        dry_mass * static_cast<double>(f.tke[i][j][k]),
    }};
}

void require_shape(
    const Field3D& field, const SimulationStateView& state,
    const char* name)
{
    if (field.size_r() != state.grid.nr
        || field.size_th() != state.grid.nth
        || field.size_z() != state.grid.nz)
    {
        throw std::invalid_argument(
            std::string("flat device acoustic attribution ") + name
            + " shape mismatch");
    }
}

} // namespace

void attribute_flat_cgrid_fused_commit(
    const SimulationStateView& state,
    const Field3D& u_before,
    const Field3D& v_before,
    const Field3D& w_before,
    const Field3D& rho_before,
    const Field3D& p_before,
    float dt_small,
    float dx,
    float dy,
    float dz,
    CartesianCGridAcousticEnergyReportV1& report)
{
    if (state.grid.geometry.coord != CoordinateSystem::Cartesian
        || !state.grid.geometry.staggered
        || tmv::physical_grid_metrics::terrain_is_active(state)
        || state.grid.nr < 3 || state.grid.nth < 3 || state.grid.nz < 2
        || !(dt_small > 0.0f) || !(dx > 0.0f)
        || !(dy > 0.0f) || !(dz > 0.0f)
        || dx != static_cast<float>(state.grid.dr)
        || dy != static_cast<float>(state.grid.dtheta)
        || dz != static_cast<float>(state.grid.dz)
        || static_cast<int>(state.base.p0.size()) < state.grid.nz)
    {
        throw std::invalid_argument(
            "flat device acoustic attribution received invalid geometry");
    }
    tmv::physical_grid_metrics::validate(state);
    const auto& centers = state.grid.geometry.z;
    const auto& faces = state.grid.geometry.z_faces;
    const double geometry_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max(state.grid.dz, 1.0);
    for (int k = 0; k < state.grid.nz; ++k)
    {
        const double expected_center =
            (static_cast<double>(k) + 0.5) * state.grid.dz;
        const double expected_lower =
            static_cast<double>(k) * state.grid.dz;
        const double expected_upper =
            static_cast<double>(k + 1) * state.grid.dz;
        const double center = centers.empty()
            ? expected_center : centers[static_cast<std::size_t>(k)];
        const double lower = faces.empty()
            ? expected_lower : faces[static_cast<std::size_t>(k)];
        const double upper = faces.empty()
            ? expected_upper : faces[static_cast<std::size_t>(k + 1)];
        if (std::abs(center - expected_center) > geometry_tolerance
            || std::abs(lower - expected_lower) > geometry_tolerance
            || std::abs(upper - expected_upper) > geometry_tolerance)
        {
            throw std::invalid_argument(
                "flat device acoustic attribution requires uniform vertical "
                "geometry matching the shader spacing");
        }
    }
    const Field3D* required_fields[] = {
        &state.fields.u, &state.fields.v, &state.fields.w,
        &state.fields.rho, &state.fields.p, &state.fields.theta,
        &state.fields.qv, &state.fields.qc, &state.fields.qr,
        &state.fields.qi, &state.fields.qs, &state.fields.qg,
        &state.fields.qh, &state.fields.tke,
        &u_before, &v_before, &w_before, &rho_before, &p_before,
    };
    const char* required_names[] = {
        "u", "v", "w", "rho", "p", "theta",
        "qv", "qc", "qr", "qi", "qs", "qg", "qh", "tke",
        "u_before", "v_before", "w_before", "rho_before", "p_before",
    };
    for (std::size_t n = 0; n < std::size(required_fields); ++n)
        require_shape(*required_fields[n], state, required_names[n]);

    const double report_values[] = {
        report.continuity_eos_dry_internal_change_j,
        report.continuity_eos_water_sensible_change_j,
        report.continuity_eos_phase_reference_change_j,
        report.continuity_carrier_kinetic_change_j,
        report.continuity_gravitational_change_j,
        report.continuity_subgrid_tke_change_j,
        report.continuity_carrier_requested_change_j,
        report.pressure_gradient_requested_linear_work_j,
        report.pressure_gradient_finite_step_correction_j,
        report.pressure_gradient_storage_rounding_j,
        report.pressure_gradient_committed_work_j,
        report.committed_inventory_change_j,
    };
    if (std::any_of(
            std::begin(report_values), std::end(report_values),
            [](double value) { return !std::isfinite(value); })
        || report.pressure_gradient_finite_step_correction_j < 0.0
        || report.complete != (report.substep_count > 0))
    {
        throw std::invalid_argument(
            "flat device acoustic attribution received an invalid report");
    }

    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const float inv_dx = 1.0f / dx;
    const float inv_dy = 1.0f / dy;
    const float inv_dz = 1.0f / dz;
    const auto sums = diagnostics::fixed_slab_reduce<12>(
        domain.i_begin, domain.i_end,
        [&](int i, std::array<double, 12>& partial)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const auto before = reservoirs(
                        state, u_before, v_before, w_before,
                        static_cast<double>(rho_before[i][j][k]),
                        static_cast<double>(p_before[i][j][k]),
                        i, j, k);
                    const auto after_pressure = reservoirs(
                        state, u_before, v_before, w_before,
                        static_cast<double>(state.fields.rho[i][j][k]),
                        static_cast<double>(state.fields.p[i][j][k]),
                        i, j, k);
                    bool valid = true;
                    for (std::size_t n = 0; n < before.size(); ++n)
                    {
                        valid = valid && std::isfinite(before[n])
                            && std::isfinite(after_pressure[n]);
                        partial[n] += after_pressure[n] - before[n];
                    }
                    const float density_request = requested_density_change(
                        rho_before, u_before, v_before, w_before,
                        state.grid.nr, state.grid.nth, state.grid.nz,
                        i, j, k, dt_small, dx, dy, dz,
                        state.domain_frame ? static_cast<float>(state.domain_frame->velocity_ms.u_ms) : 0.0f,
                        state.domain_frame ? static_cast<float>(state.domain_frame->velocity_ms.v_ms) : 0.0f);
                    const double volume = cell_volume(state, k);
                    const double dry_mass_before =
                        static_cast<double>(rho_before[i][j][k]) * volume;
                    const double carrier_coefficient =
                        before[3] / dry_mass_before;
                    const double carrier_request =
                        static_cast<double>(density_request)
                        * volume * carrier_coefficient;
                    valid = valid
                        && std::isfinite(carrier_request)
                        && dry_mass_before > 0.0;
                    if (valid)
                        partial[6] += carrier_request;
                    else
                        partial[7] += 1.0;

                    const bool momentum_active =
                        k <= state.grid.nz - 2;
                    if (!momentum_active)
                        continue;
                    const auto safe_rho = [](float value)
                    {
                        return std::isfinite(value) && value > 1.0e-6f
                            ? value : 1.0f;
                    };
                    const auto finite_or_zero = [](float value)
                    {
                        return std::isfinite(value) ? value : 0.0f;
                    };
                    const auto& rho = state.fields.rho;
                    const auto& pressure = state.fields.p;
                    const float rho_x =
                        0.5f * (rho[i][j][k] + rho[i + 1][j][k]);
                    const float rho_y =
                        0.5f * (rho[i][j][k] + rho[i][j + 1][k]);
                    const float rho_z =
                        0.5f * (rho[i][j][k] + rho[i][j][k + 1]);
                    const float du = finite_or_zero(
                        -((pressure[i + 1][j][k] - pressure[i][j][k])
                          * inv_dx)
                        / safe_rho(rho_x));
                    const float dv = finite_or_zero(
                        -((pressure[i][j + 1][k] - pressure[i][j][k])
                          * inv_dy)
                        / safe_rho(rho_y));
                    const float base_gradient = static_cast<float>(
                        (state.base.p0[static_cast<std::size_t>(k + 1)]
                         - state.base.p0[static_cast<std::size_t>(k)])
                        / state.grid.dz);
                    const float dw = finite_or_zero(
                        -(((pressure[i][j][k + 1] - pressure[i][j][k])
                           * inv_dz) - base_gradient)
                        / safe_rho(rho_z));
                    const double delta_u = static_cast<double>(du * dt_small);
                    const double delta_v = static_cast<double>(dv * dt_small);
                    const double delta_w = static_cast<double>(dw * dt_small);
                    const double center_mass = mixture_mass(
                        state, static_cast<double>(rho[i][j][k]),
                        i, j, k);
                    double u_mass = 0.5 * center_mass;
                    if (i + 1 < domain.i_end)
                        u_mass += 0.5 * mixture_mass(
                            state, static_cast<double>(rho[i + 1][j][k]),
                            i + 1, j, k);
                    double v_mass = 0.5 * center_mass;
                    if (j + 1 < domain.j_end)
                        v_mass += 0.5 * mixture_mass(
                            state, static_cast<double>(rho[i][j + 1][k]),
                            i, j + 1, k);
                    const double w_mass = 0.5 * (
                        center_mass + mixture_mass(
                            state, static_cast<double>(rho[i][j][k + 1]),
                            i, j, k + 1));
                    const double old_u =
                        static_cast<double>(u_before[i][j][k]);
                    const double old_v =
                        static_cast<double>(v_before[i][j][k]);
                    const double old_w =
                        static_cast<double>(w_before[i][j][k]);
                    const double stored_u =
                        static_cast<double>(state.fields.u[i][j][k]);
                    const double stored_v =
                        static_cast<double>(state.fields.v[i][j][k]);
                    const double stored_w =
                        static_cast<double>(state.fields.w[i][j][k]);
                    partial[8] +=
                        u_mass * old_u * delta_u
                        + v_mass * old_v * delta_v
                        + w_mass * old_w * delta_w;
                    partial[9] += 0.5 * (
                        u_mass * delta_u * delta_u
                        + v_mass * delta_v * delta_v
                        + w_mass * delta_w * delta_w);
                    partial[10] += 0.5 * (
                        u_mass * (stored_u * stored_u - old_u * old_u)
                        + v_mass * (stored_v * stored_v - old_v * old_v)
                        + w_mass * (stored_w * stored_w - old_w * old_w));
                    const double momentum_values[] = {
                        u_mass, v_mass, w_mass, old_u, old_v, old_w,
                        delta_u, delta_v, delta_w,
                        stored_u, stored_v, stored_w,
                    };
                    if (std::any_of(
                            std::begin(momentum_values),
                            std::end(momentum_values),
                            [](double value) {
                                return !std::isfinite(value);
                            })
                        || u_mass <= 0.0 || v_mass <= 0.0 || w_mass <= 0.0)
                    {
                        partial[11] += 1.0;
                    }
                }
        });
    if (sums[7] != 0.0 || sums[11] != 0.0
        || std::any_of(
            sums.begin(), sums.end(),
            [](double value) { return !std::isfinite(value); })
        || sums[9] < 0.0)
    {
        throw std::runtime_error(
            "flat device acoustic attribution produced invalid basis-v2 state");
    }

    const double storage_rounding = sums[10] - sums[8] - sums[9];
    const double inventory =
        std::accumulate(sums.begin(), sums.begin() + 6, 0.0)
        + sums[10];
    CartesianCGridAcousticEnergyReportV1 next = report;
    next.continuity_eos_dry_internal_change_j += sums[0];
    next.continuity_eos_water_sensible_change_j += sums[1];
    next.continuity_eos_phase_reference_change_j += sums[2];
    next.continuity_carrier_kinetic_change_j += sums[3];
    next.continuity_gravitational_change_j += sums[4];
    next.continuity_subgrid_tke_change_j += sums[5];
    next.continuity_carrier_requested_change_j += sums[6];
    next.pressure_gradient_requested_linear_work_j += sums[8];
    next.pressure_gradient_finite_step_correction_j += sums[9];
    next.pressure_gradient_storage_rounding_j += storage_rounding;
    next.pressure_gradient_committed_work_j += sums[10];
    next.committed_inventory_change_j += inventory;
    ++next.substep_count;
    next.complete = true;
    report = next;
}

} // namespace tmv::device_acoustic_energy

#include "core/orchestration/dynamics/sgs_apply.hpp"
#include "core/orchestration/dynamics/sgs_apply_internal.hpp"

#include "compute/compute_kernel_template.hpp"
#include "compute/parity_check.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/runtime_config.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "numerics/runtime.hpp"
#include "terrain/conservative_transport.hpp"
#include "turbulence/base/eddy_viscosity.hpp"
#include "turbulence/runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace
{

bool gpu_sgs_apply_enabled(const SimulationStateView& execution_state)
{
    return tmv::require_run_configuration(execution_state).gpu_apply_sgs;
}

bool native_sgs_apply_required(const SimulationStateView& state)
{
    const auto* configuration = state.configuration;
    const auto* compute = state.compute_runtime;
    return configuration != nullptr && compute != nullptr &&
        configuration->execution_mode ==
            tmv::init::config::ExecutionMode::Qualified &&
        compute->active_kind != ComputeBackendKind::Cpu &&
        !compute->fallback_active && !compute->config.allow_fallback &&
        state.turbulence_runtime != nullptr &&
        (state.turbulence_runtime->config.scheme_id == "smagorinsky" ||
         state.turbulence_runtime->config.scheme_id == "tke") &&
        configuration->gpu_apply_sgs;
}

bool open_lateral_boundary(tmv::init::config::LateralBC boundary)
{
    using Boundary = tmv::init::config::LateralBC;
    return boundary == Boundary::Radiative
        || boundary == Boundary::ZeroGradient
        || boundary == Boundary::LegacyOpen;
}

bool has_open_lateral_boundary(const SimulationStateView& execution_state)
{
    return open_lateral_boundary(tmv::require_run_configuration(execution_state).lateral_bc_west)
        || open_lateral_boundary(tmv::require_run_configuration(execution_state).lateral_bc_east)
        || open_lateral_boundary(tmv::require_run_configuration(execution_state).lateral_bc_south)
        || open_lateral_boundary(tmv::require_run_configuration(execution_state).lateral_bc_north);
}

bool has_open_top_boundary(const SimulationStateView& execution_state)
{
    return tmv::require_run_configuration(execution_state).top_bc == tmv::init::config::TopBC::Open;
}

bool has_compute_context(const SimulationStateView& state)
{
    return state.compute_runtime != nullptr && state.configuration != nullptr;
}

bool field_matches_domain(const SimulationStateView& state, const Field3D& field)
{
    return field.size_r() == state.grid.nr &&
           field.size_th() == state.grid.nth &&
           field.size_z() == state.grid.nz;
}

TurbulenceRuntimeState& turbulence_runtime(const SimulationStateView& state)
{
    return state.turbulence_runtime
        ? *state.turbulence_runtime : tmv::require_level_runtime(state.turbulence_runtime, "turbulence_runtime");
}

bool active_sgs_work_ledger(const SimulationStateView& state)
{
    const auto& runtime = turbulence_runtime(state);
    return runtime.scheme && runtime.scheme->name() != "none";
}

double realized_directional_stress_boundary_power_w(
    const SimulationStateView& state)
{
    const auto& runtime = turbulence_runtime(state);
    const Field3D* viscosity = runtime.scheme
        ? runtime.scheme->realized_eddy_viscosity() : nullptr;
    if (viscosity == nullptr || !field_matches_domain(state, *viscosity)
        || tmv::physical_grid_metrics::terrain_is_active(state))
        return std::numeric_limits<double>::quiet_NaN();

    // The flat collocated operator closes every non-periodic exterior face
    // with zero traction. There is no closure-plane velocity in its stencil,
    // so its independently evaluated exterior power is exactly zero.
    if (!state.grid.geometry.staggered)
        return 0.0;

    const int nx = state.grid.nr;
    const int ny = state.grid.nth;
    const int nz = state.grid.nz;
    const bool use_h = runtime.config.mode != "vertical_only";
    const bool use_v = runtime.config.mode != "horizontal_only";
    const bool open_west = open_lateral_boundary(tmv::require_run_configuration(state).lateral_bc_west);
    const bool open_east = open_lateral_boundary(tmv::require_run_configuration(state).lateral_bc_east);
    const bool open_south = open_lateral_boundary(tmv::require_run_configuration(state).lateral_bc_south);
    const bool open_north = open_lateral_boundary(tmv::require_run_configuration(state).lateral_bc_north);
    const bool wrap_x =
        tmv::require_run_configuration(state).lateral_bc_west == tmv::init::config::LateralBC::Periodic
        || tmv::require_run_configuration(state).lateral_bc_east == tmv::init::config::LateralBC::Periodic;
    const bool wrap_y =
        tmv::require_run_configuration(state).lateral_bc_south == tmv::init::config::LateralBC::Periodic
        || tmv::require_run_configuration(state).lateral_bc_north == tmv::init::config::LateralBC::Periodic;
    const auto& u = state.fields.u;
    const auto& v = state.fields.v;
    const auto& w = state.fields.w;
    const auto& rho = state.fields.rho;
    const double dx = state.grid.dr;
    const double dy = state.grid.dtheta;
    const auto dz_cell = [&](int i, int j, int k)
    {
        return tmv::physical_grid_metrics::layer_thickness(state, i, j, k);
    };
    const auto dz_center = [&](int i, int j, int k)
    {
        return tmv::physical_grid_metrics::scalar_height(state, i, j, k + 1)
            - tmv::physical_grid_metrics::scalar_height(state, i, j, k);
    };
    const auto rho_c = [&](int i, int j, int k)
    { return static_cast<double>(rho(i, j, k)); };
    const auto mu_c = [&](int i, int j, int k)
    {
        return rho_c(i, j, k)
            * static_cast<double>((*viscosity)(i, j, k));
    };
    const auto mu_xy = [&](int i, int j, int k)
    {
        return 0.25 * (mu_c(i, j, k) + mu_c(i + 1, j, k)
                     + mu_c(i, j + 1, k) + mu_c(i + 1, j + 1, k));
    };
    const auto mu_xz = [&](int i, int j, int k)
    {
        const double lower = 0.5 * (mu_c(i, j, k) + mu_c(i + 1, j, k));
        const double upper = 0.5 * (
            mu_c(i, j, k + 1) + mu_c(i + 1, j, k + 1));
        const double dz0 = dz_cell(i, j, k);
        const double dz1 = dz_cell(i, j, k + 1);
        return (dz1 * lower + dz0 * upper) / (dz0 + dz1);
    };
    const auto mu_yz = [&](int i, int j, int k)
    {
        const double lower = 0.5 * (mu_c(i, j, k) + mu_c(i, j + 1, k));
        const double upper = 0.5 * (
            mu_c(i, j, k + 1) + mu_c(i, j + 1, k + 1));
        const double dz0 = dz_cell(i, j, k);
        const double dz1 = dz_cell(i, j, k + 1);
        return (dz1 * lower + dz0 * upper) / (dz0 + dz1);
    };
    const auto t12 = [&](int i, int j, int k)
    {
        return mu_xy(i, j, k) * (
            (static_cast<double>(u(i, j + 1, k)) - u(i, j, k)) / dy
          + (static_cast<double>(v(i + 1, j, k)) - v(i, j, k)) / dx);
    };
    const auto t31 = [&](int i, int j, int k)
    {
        double gradient =
            (static_cast<double>(w(i + 1, j, k)) - w(i, j, k)) / dx;
        if (use_v)
            gradient += (static_cast<double>(u(i, j, k + 1))
                       - u(i, j, k)) / dz_center(i, j, k);
        return mu_xz(i, j, k) * gradient;
    };
    const auto t32 = [&](int i, int j, int k)
    {
        double gradient =
            (static_cast<double>(w(i, j + 1, k)) - w(i, j, k)) / dy;
        if (use_v)
            gradient += (static_cast<double>(v(i, j, k + 1))
                       - v(i, j, k)) / dz_center(i, j, k);
        return mu_yz(i, j, k) * gradient;
    };

    double boundary_power = 0.0;
    if ((open_west || open_east) && use_h)
    {
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz - 1; ++k)
            {
                const double area = dy * dz_cell(1, j, k);
                if (open_west)
                {
                    const double west = 2.0 * mu_c(1, j, k)
                        * (static_cast<double>(u(1, j, k))
                           - u(0, j, k)) / dx;
                    boundary_power -= area
                        * static_cast<double>(u(0, j, k)) * west;
                }
                if (open_east)
                {
                    const double east = 2.0 * mu_c(nx - 2, j, k)
                        * (static_cast<double>(u(nx - 2, j, k))
                           - u(nx - 3, j, k)) / dx;
                    boundary_power += area
                        * static_cast<double>(u(nx - 2, j, k)) * east;
                }
            }
        const int j_end = wrap_y ? ny - 1 : ny - 2;
        for (int j = 1; j < j_end; ++j)
            for (int k = 0; k < nz - 1; ++k)
            {
                const double area = dy * dz_cell(1, j, k);
                if (open_west)
                    boundary_power -= area
                        * static_cast<double>(v(0, j, k))
                        * t12(0, j, k);
                if (open_east)
                    boundary_power += area
                        * static_cast<double>(v(nx - 1, j, k))
                        * t12(nx - 2, j, k);
            }
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz - 1; ++k)
            {
                const double area = dy * dz_center(1, j, k);
                if (open_west)
                    boundary_power -= area
                        * static_cast<double>(w(0, j, k))
                        * t31(0, j, k);
                if (open_east)
                    boundary_power += area
                        * static_cast<double>(w(nx - 1, j, k))
                        * t31(nx - 2, j, k);
            }
    }
    if ((open_south || open_north) && use_h)
    {
        for (int i = 1; i < nx - 1; ++i)
            for (int k = 0; k < nz - 1; ++k)
            {
                const double area = dx * dz_cell(i, 1, k);
                if (open_south)
                {
                    const double south = 2.0 * mu_c(i, 1, k)
                        * (static_cast<double>(v(i, 1, k))
                           - v(i, 0, k)) / dy;
                    boundary_power -= area
                        * static_cast<double>(v(i, 0, k)) * south;
                }
                if (open_north)
                {
                    const double north = 2.0 * mu_c(i, ny - 2, k)
                        * (static_cast<double>(v(i, ny - 2, k))
                           - v(i, ny - 3, k)) / dy;
                    boundary_power += area
                        * static_cast<double>(v(i, ny - 2, k)) * north;
                }
            }
        const int i_end = wrap_x ? nx - 1 : nx - 2;
        for (int i = 1; i < i_end; ++i)
            for (int k = 0; k < nz - 1; ++k)
            {
                const double area = dx * dz_cell(i, 1, k);
                if (open_south)
                    boundary_power -= area
                        * static_cast<double>(u(i, 0, k))
                        * t12(i, 0, k);
                if (open_north)
                    boundary_power += area
                        * static_cast<double>(u(i, ny - 1, k))
                        * t12(i, ny - 2, k);
            }
        for (int i = 1; i < nx - 1; ++i)
            for (int k = 0; k < nz - 1; ++k)
            {
                const double area = dx * dz_center(i, 1, k);
                if (open_south)
                    boundary_power -= area
                        * static_cast<double>(w(i, 0, k))
                        * t32(i, 0, k);
                if (open_north)
                    boundary_power += area
                        * static_cast<double>(w(i, ny - 1, k))
                        * t32(i, ny - 2, k);
            }
    }

    if (use_v && has_open_top_boundary(state))
    {
        const int k = nz - 2;
        const int i_u_end = wrap_x ? nx - 1 : nx - 2;
        for (int i = 1; i < i_u_end; ++i)
            for (int j = 1; j < ny - 1; ++j)
            {
                double gradient = (static_cast<double>(u(i, j, k + 1))
                                  - u(i, j, k)) / dz_center(i, j, k);
                if (use_h)
                    gradient += (static_cast<double>(w(i + 1, j, k))
                               - w(i, j, k)) / dx;
                boundary_power += dx * dy
                    * static_cast<double>(u(i, j, k + 1))
                    * mu_xz(i, j, k) * gradient;
            }
        const int j_v_end = wrap_y ? ny - 1 : ny - 2;
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < j_v_end; ++j)
            {
                double gradient = (static_cast<double>(v(i, j, k + 1))
                                  - v(i, j, k)) / dz_center(i, j, k);
                if (use_h)
                    gradient += (static_cast<double>(w(i, j + 1, k))
                               - w(i, j, k)) / dy;
                boundary_power += dx * dy
                    * static_cast<double>(v(i, j, k + 1))
                    * mu_yz(i, j, k) * gradient;
            }
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
            {
                const double top_gradient =
                    (static_cast<double>(w(i, j, nz - 1))
                     - w(i, j, nz - 2)) / dz_cell(i, j, nz - 1);
                boundary_power += dx * dy
                    * static_cast<double>(w(i, j, nz - 1))
                    * 2.0 * mu_c(i, j, nz - 1) * top_gradient;
            }
    }
    return boundary_power;
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

double cell_total_water(const SimulationStateView& state,
                        int i, int j, int k,
                        bool complete_moist_state)
{
    if (!complete_moist_state)
        return 0.0;
    const auto& fields = state.fields;
    const double water[] = {
        static_cast<double>(fields.qv[i][j][k]),
        static_cast<double>(fields.qc[i][j][k]),
        static_cast<double>(fields.qr[i][j][k]),
        static_cast<double>(fields.qi[i][j][k]),
        static_cast<double>(fields.qs[i][j][k]),
        static_cast<double>(fields.qg[i][j][k]),
        static_cast<double>(fields.qh[i][j][k]),
    };
    double total = 0.0;
    for (double value : water)
    {
        if (!std::isfinite(value) || value < 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        total += value;
    }
    return total;
}

double cell_carrier_mass_kg(
    const SimulationStateView& state,
    const tmv::physical_grid_metrics::ControlVolumeMassView& geometry,
    int i, int j, int k,
    bool complete_moist_state)
{
    const double total_water =
        cell_total_water(state, i, j, k, complete_moist_state);
    const double mass =
        geometry.scalar_mass_kg(state.fields.rho, i, j, k)
        * (1.0 + total_water);
    return std::isfinite(mass) && mass > 0.0
        ? mass : std::numeric_limits<double>::quiet_NaN();
}

double basis_v2_specific_kinetic_energy(
    const SimulationStateView& state,
    const TerrainMetrics3D* terrain,
    const PhysicalCellDomain& domain,
    int i, int j, int k)
{
    const auto& fields = state.fields;
    if (!state.grid.geometry.staggered)
    {
        const double u = static_cast<double>(fields.u[i][j][k]);
        const double v = static_cast<double>(fields.v[i][j][k]);
        const double w = static_cast<double>(fields.w[i][j][k]);
        return 0.5 * (u * u + v * v + w * w);
    }
    const double u_w = static_cast<double>(fields.u[i - 1][j][k]);
    const double u_e = static_cast<double>(fields.u[i][j][k]);
    const double v_s = static_cast<double>(fields.v[i][j - 1][k]);
    const double v_n = static_cast<double>(fields.v[i][j][k]);
    double w_b = k == domain.k_begin
        ? 0.0 : static_cast<double>(fields.w[i][j][k - 1]);
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
    const double w_t = static_cast<double>(fields.w[i][j][k]);
    return 0.25 * (
        u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n
        + w_b * w_b + w_t * w_t);
}

TurbulenceMomentumWorkReport apply_sgs_tendencies_cpu(
    const SimulationStateView& state,
    double dt_dynamics,
    const TurbulenceTendencies& turb_tend,
    bool attribute_work)
{
    const auto& grid = state.grid;
    auto& fields = state.fields;
    auto& tendency_fields = state.tendencies;
    const auto& base = state.base;
    const bool has_qv = !fields.qv.empty();
    const bool update_tke =
        field_matches_domain(state, fields.tke) &&
        field_matches_domain(state, tendency_fields.dtke_dt_pbl) &&
        field_matches_domain(state, turb_tend.dtkedt_sgs);
    const bool attribute_moist_energy =
        attribute_work && complete_moist_energy_state(state) && update_tke;
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    if (attribute_work)
    {
        if (!std::isfinite(dt_dynamics) || dt_dynamics <= 0.0 ||
            !field_matches_domain(state, fields.rho) ||
            !std::isfinite(grid.dr) || grid.dr <= 0.0 ||
            !std::isfinite(grid.dtheta) || grid.dtheta <= 0.0 ||
            !std::isfinite(grid.dz) || grid.dz <= 0.0)
        {
            throw std::invalid_argument(
                "active SGS work attribution requires a finite positive "
                "Cartesian physical-grid state and timestep");
        }
        if (grid.nr < 3 || grid.nth < 3 || grid.nz < 2)
        {
            throw std::invalid_argument(
                "active SGS work attribution requires at least a 3x3x2 grid");
        }
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    if (!std::isfinite(static_cast<double>(
                            fields.rho(i, j, k)))
                        || fields.rho(i, j, k) <= 0.0f)
                        throw std::invalid_argument(
                            "SGS work attribution requires positive finite "
                            "physical live dry density");
        tmv::physical_grid_metrics::validate(state);
    }

    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(state);
    const TerrainMetrics3D* terrain =
        tmv::physical_grid_metrics::active_terrain_metrics(state);
    const int update_i_begin = grid.geometry.staggered ? 0 : domain.i_begin;
    const int update_i_end = grid.geometry.staggered ? grid.nr : domain.i_end;
    const int update_j_begin = grid.geometry.staggered ? 0 : domain.j_begin;
    const int update_j_end = grid.geometry.staggered ? grid.nth : domain.j_end;
    const int update_k_begin = grid.geometry.staggered ? 0 : domain.k_begin;
    const int update_k_end = grid.geometry.staggered ? grid.nz : domain.k_end;
    const bool terrain_ground_available = terrain != nullptr
        && terrain->Gx_face.size_r() == grid.nr
        && terrain->Gx_face.size_th() == grid.nth
        && terrain->Gx_face.size_z() == grid.nz + 1
        && terrain->Gy_face.size_r() == grid.nr
        && terrain->Gy_face.size_th() == grid.nth
        && terrain->Gy_face.size_z() == grid.nz + 1;
    if (attribute_moist_energy && terrain != nullptr
        && !terrain_ground_available)
    {
        throw std::invalid_argument(
            "mapped SGS moist-energy attribution requires complete "
            "terrain face metrics");
    }

    // The mapped lower face is implicit. Attribute its exact change before
    // mutating horizontal velocities so all four neighbouring samples are
    // from one operator time level. This is surface-only work.
    std::array<double, 4> ground_work{};
    if (attribute_work && terrain_ground_available)
    {
        ground_work = tmv::diagnostics::fixed_slab_reduce<4>(
            domain.i_begin, domain.i_end,
            [&](int i, std::array<double, 4>& partial)
            {
                const int k = domain.k_begin;
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    const double slope_x = -terrain->Gx_face(i, j, k);
                    const double slope_y = -terrain->Gy_face(i, j, k);
                    const auto stored_after =
                        [&](const Field3D& velocity,
                            const Field3D& tendency,
                            int ii, int jj)
                        {
                            const double old =
                                static_cast<double>(velocity[ii][jj][k]);
                            double delta = dt_dynamics * static_cast<double>(
                                tendency[ii][jj][k]);
                            if (primary_state_repairs_enabled
                                && !std::isfinite(delta))
                                delta = 0.0;
                            double next = old + delta;
                            if (primary_state_repairs_enabled
                                && !std::isfinite(next))
                                next = 0.0;
                            return static_cast<double>(
                                clamp_wind_horizontal_ms(
                                    static_cast<float>(next)));
                        };
                    const auto raw_delta =
                        [&](const Field3D& tendency, int ii, int jj)
                        {
                            double delta = dt_dynamics
                                * static_cast<double>(
                                    tendency[ii][jj][k]);
                            if (primary_state_repairs_enabled
                                && !std::isfinite(delta))
                                delta = 0.0;
                            return delta;
                        };
                    const double u_w =
                        static_cast<double>(fields.u[i - 1][j][k]);
                    const double u_e =
                        static_cast<double>(fields.u[i][j][k]);
                    const double v_s =
                        static_cast<double>(fields.v[i][j - 1][k]);
                    const double v_n =
                        static_cast<double>(fields.v[i][j][k]);
                    const double du_w =
                        raw_delta(turb_tend.dudt_sgs, i - 1, j);
                    const double du_e =
                        raw_delta(turb_tend.dudt_sgs, i, j);
                    const double dv_s =
                        raw_delta(turb_tend.dvdt_sgs, i, j - 1);
                    const double dv_n =
                        raw_delta(turb_tend.dvdt_sgs, i, j);
                    const double frame_u = state.domain_frame != nullptr
                        ? state.domain_frame->velocity_ms.u_ms : 0.0;
                    const double frame_v = state.domain_frame != nullptr
                        ? state.domain_frame->velocity_ms.v_ms : 0.0;
                    const double old_ground =
                        (0.5 * (u_w + u_e) - frame_u) * slope_x
                        + (0.5 * (v_s + v_n) - frame_v) * slope_y;
                    const double raw_delta_ground =
                        0.5 * (du_w + du_e) * slope_x
                        + 0.5 * (dv_s + dv_n) * slope_y;
                    const double new_ground =
                        (0.5 * (
                            stored_after(
                                fields.u, turb_tend.dudt_sgs,
                                i - 1, j)
                            + stored_after(
                                fields.u, turb_tend.dudt_sgs,
                                i, j)) - frame_u) * slope_x
                        + (0.5 * (
                            stored_after(
                                fields.v, turb_tend.dvdt_sgs,
                                i, j - 1)
                            + stored_after(
                                fields.v, turb_tend.dvdt_sgs,
                                i, j)) - frame_v) * slope_y;
                    const double face_mass = 0.5 * cell_carrier_mass_kg(
                        state, mass_geometry, i, j, k,
                        attribute_moist_energy);
                    partial[0] +=
                        face_mass * old_ground * raw_delta_ground;
                    partial[1] += 0.5 * face_mass
                        * raw_delta_ground * raw_delta_ground;
                    partial[2] += 0.5 * face_mass
                        * (new_ground * new_ground
                           - old_ground * old_ground);
                }
            });
    }

    // Phase one updates momentum. Keeping scalars unchanged until this pass
    // ends makes every adjacent-face mass and every later carrier coefficient
    // a race-free, single-time-level quantity.
    const auto momentum_work = tmv::diagnostics::fixed_slab_reduce<4>(
        update_i_begin, update_i_end,
        [&](int i, std::array<double, 4>& partial)
        {
            for (int j = update_j_begin; j < update_j_end; ++j)
                for (int k = update_k_begin; k < update_k_end; ++k)
            {
                double dudt_sgs = static_cast<double>(turb_tend.dudt_sgs[i][j][k]);
                double dvdt_sgs = static_cast<double>(turb_tend.dvdt_sgs[i][j][k]);
                double dwdt_sgs = static_cast<double>(turb_tend.dwdt_sgs[i][j][k]);
                double dthetadt_sgs =
                    static_cast<double>(turb_tend.dthetadt_sgs[i][j][k]);

                if (primary_state_repairs_enabled && !std::isfinite(dudt_sgs)) dudt_sgs = 0.0;
                if (primary_state_repairs_enabled && !std::isfinite(dvdt_sgs)) dvdt_sgs = 0.0;
                if (primary_state_repairs_enabled && !std::isfinite(dwdt_sgs)) dwdt_sgs = 0.0;
                if (primary_state_repairs_enabled && !std::isfinite(dthetadt_sgs)) dthetadt_sgs = 0.0;

                const double u_old = static_cast<double>(fields.u[i][j][k]);
                const double v_old = static_cast<double>(fields.v[i][j][k]);
                const double w_old = static_cast<double>(fields.w[i][j][k]);
                const double du = dudt_sgs * dt_dynamics;
                const double dv = dvdt_sgs * dt_dynamics;
                const double dw = dwdt_sgs * dt_dynamics;
                double u_new = u_old + du;
                double v_new = v_old + dv;
                double w_new = w_old + dw;
                if (primary_state_repairs_enabled && !std::isfinite(u_new)) u_new = 0.0;
                if (primary_state_repairs_enabled && !std::isfinite(v_new)) v_new = 0.0;
                if (primary_state_repairs_enabled && !std::isfinite(w_new)) w_new = 0.0;
                fields.u[i][j][k] =
                    clamp_wind_horizontal_ms(static_cast<float>(u_new));
                fields.v[i][j][k] =
                    clamp_wind_horizontal_ms(static_cast<float>(v_new));
                fields.w[i][j][k] =
                    clamp_wind_vertical_ms(static_cast<float>(w_new));

                if (attribute_work)
                {
                    const double mass_here =
                        i >= domain.i_begin && i < domain.i_end &&
                        j >= domain.j_begin && j < domain.j_end &&
                        k >= domain.k_begin && k < domain.k_end
                            ? cell_carrier_mass_kg(state, mass_geometry, i, j, k,
                                                   attribute_moist_energy)
                            : 0.0;
                    const auto accumulate =
                        [&](double mass, double old_velocity, double delta,
                            double stored_velocity)
                        {
                            if (!std::isfinite(mass) || mass <= 0.0)
                            {
                                partial[3] += 1.0;
                                return;
                            }
                            partial[0] += mass * old_velocity * delta;
                            partial[1] += 0.5 * mass * delta * delta;
                            partial[2] += 0.5 * mass
                                * (stored_velocity * stored_velocity
                                   - old_velocity * old_velocity);
                        };
                    if (!grid.geometry.staggered
                        && i >= domain.i_begin && i < domain.i_end
                        && j >= domain.j_begin && j < domain.j_end
                        && k >= domain.k_begin && k < domain.k_end)
                    {
                        const double mass = mass_here;
                        accumulate(mass, u_old, du,
                                   static_cast<double>(fields.u[i][j][k]));
                        accumulate(mass, v_old, dv,
                                   static_cast<double>(fields.v[i][j][k]));
                        accumulate(mass, w_old, dw,
                                   static_cast<double>(fields.w[i][j][k]));
                    }
                    else if (grid.geometry.staggered
                        && j >= domain.j_begin && j < domain.j_end
                        && k >= domain.k_begin && k < domain.k_end
                        && i >= domain.i_begin - 1
                        && i < domain.i_end)
                    {
                        double mass = 0.0;
                        if (i >= domain.i_begin)
                            mass += 0.5 * mass_here;
                        if (i + 1 < domain.i_end)
                            mass += 0.5 * cell_carrier_mass_kg(
                                state, mass_geometry, i + 1, j, k,
                                attribute_moist_energy);
                        accumulate(
                            mass, u_old, du,
                            static_cast<double>(fields.u[i][j][k]));
                    }
                    if (grid.geometry.staggered
                        && i >= domain.i_begin && i < domain.i_end
                        && k >= domain.k_begin && k < domain.k_end
                        && j >= domain.j_begin - 1
                        && j < domain.j_end)
                    {
                        double mass = 0.0;
                        if (j >= domain.j_begin)
                            mass += 0.5 * mass_here;
                        if (j + 1 < domain.j_end)
                            mass += 0.5 * cell_carrier_mass_kg(
                                state, mass_geometry, i, j + 1, k,
                                attribute_moist_energy);
                        accumulate(
                            mass, v_old, dv,
                            static_cast<double>(fields.v[i][j][k]));
                    }
                    if (grid.geometry.staggered
                        && i >= domain.i_begin && i < domain.i_end
                        && j >= domain.j_begin && j < domain.j_end
                        && k >= domain.k_begin && k < domain.k_end)
                    {
                        double mass = 0.5 * mass_here;
                        if (k + 1 < domain.k_end)
                            mass += 0.5 * cell_carrier_mass_kg(
                                state, mass_geometry, i, j, k + 1,
                                attribute_moist_energy);
                        accumulate(
                            mass, w_old, dw,
                            static_cast<double>(fields.w[i][j][k]));
                    }
                }
            }
        });
    if (attribute_work && momentum_work[3] != 0.0)
        throw std::runtime_error(
            "SGS momentum-work attribution rejected invalid carrier mass");

    // Phase two commits scalars and evaluates their exact basis-v2 reservoir
    // changes using the now-stable final velocity field.
    const auto scalar_energy = tmv::diagnostics::fixed_slab_reduce<7>(
        update_i_begin, update_i_end,
        [&](int i, std::array<double, 7>& partial)
        {
            for (int j = update_j_begin; j < update_j_end; ++j)
                for (int k = update_k_begin; k < update_k_end; ++k)
                {
                const double theta_old =
                    static_cast<double>(fields.theta[i][j][k]);
                double dthetadt_sgs =
                    static_cast<double>(turb_tend.dthetadt_sgs[i][j][k]);
                if (primary_state_repairs_enabled
                    && !std::isfinite(dthetadt_sgs))
                    dthetadt_sgs = 0.0;
                double theta_new = theta_old + dthetadt_sgs * dt_dynamics;
                if (primary_state_repairs_enabled && !std::isfinite(theta_new))
                {
                    const std::size_t ks = static_cast<std::size_t>(k);
                    theta_new = std::isfinite(theta_old) ? theta_old :
                        (ks < base.theta0.size() ? base.theta0[ks] : theta0);
                }
                fields.theta[i][j][k] =
                    clamp_theta_k(static_cast<float>(theta_new));

                const double qv_old = has_qv
                    ? static_cast<double>(fields.qv[i][j][k]) : 0.0;
                if (has_qv)
                {
                    double dqvdt_sgs =
                        static_cast<double>(turb_tend.dqvdt_sgs[i][j][k]);
                    if (primary_state_repairs_enabled && !std::isfinite(dqvdt_sgs)) dqvdt_sgs = 0.0;
                    double qv_new = static_cast<double>(fields.qv[i][j][k]) +
                                    dqvdt_sgs * dt_dynamics;
                    if (primary_state_repairs_enabled && !std::isfinite(qv_new))
                    {
                        qv_new = 0.0;
                    }
                    fields.qv[i][j][k] =
                        clamp_qv_kgkg(static_cast<float>(qv_new));
                }

                const double tke_old = update_tke
                    ? static_cast<double>(fields.tke[i][j][k]) : 0.0;
                if (update_tke)
                {
                    double dtkedt_sgs =
                        static_cast<double>(turb_tend.dtkedt_sgs[i][j][k]);
                    double dtkedt_pbl =
                        static_cast<double>(
                            tendency_fields.dtke_dt_pbl[i][j][k]);
                    if (primary_state_repairs_enabled && !std::isfinite(dtkedt_sgs)) dtkedt_sgs = 0.0;
                    if (primary_state_repairs_enabled && !std::isfinite(dtkedt_pbl)) dtkedt_pbl = 0.0;
                    double tke_new = static_cast<double>(fields.tke[i][j][k]) +
                                     (dtkedt_sgs + dtkedt_pbl) * dt_dynamics;
                    if (primary_state_repairs_enabled && !std::isfinite(tke_new))
                    {
                        tke_new = std::isfinite(tke_old) ? tke_old : 0.0;
                    }
                    fields.tke[i][j][k] = static_cast<float>(
                        primary_state_repairs_enabled
                            ? std::max(0.0, tke_new) : tke_new);
                }

                if (!attribute_moist_energy
                    || i < domain.i_begin || i >= domain.i_end
                    || j < domain.j_begin || j >= domain.j_end
                    || k < domain.k_begin || k >= domain.k_end)
                    continue;

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
                    - physical_constants::gas_constant_water_vapor_jkgk
                        * t_ref;
                constexpr double kappa =
                    physical_constants::gas_constant_dry_air_jkgk
                    / physical_constants::specific_heat_cp_jkgk;
                const double pressure =
                    static_cast<double>(fields.p[i][j][k]);
                const double exner = std::pow(
                    pressure / physical_constants::reference_pressure_pa,
                    kappa);
                const double temperature_old = theta_old * exner;
                const double temperature_new =
                    static_cast<double>(fields.theta[i][j][k]) * exner;
                const double qv_new =
                    static_cast<double>(fields.qv[i][j][k]);
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
                const double specific_kinetic =
                    basis_v2_specific_kinetic_energy(
                        state,
                        terrain_ground_available ? terrain : nullptr,
                        domain, i, j, k);
                const double z =
                    tmv::physical_grid_metrics::scalar_height(
                        state, i, j, k);
                const double tke_new =
                    static_cast<double>(fields.tke[i][j][k]);
                const double dry_internal = dry_mass * cv_d
                    * (temperature_new - temperature_old);
                const double water_sensible = dry_mass * (
                    (temperature_new - t_ref)
                        * (qv_new * cv_v + liquid * c_l + ice * c_i)
                    - (temperature_old - t_ref)
                        * (qv_old * cv_v + liquid * c_l + ice * c_i));
                const double phase_reference =
                    dry_mass * (qv_new - qv_old) * vapor_reference;
                const double carrier_kinetic =
                    dry_mass * (qv_new - qv_old) * specific_kinetic;
                const double gravitational =
                    dry_mass * (qv_new - qv_old)
                    * physical_constants::gravity_ms2 * z;
                const double subgrid_tke =
                    dry_mass * (tke_new - tke_old);
                const double values[] = {
                    pressure, exner, temperature_old, temperature_new,
                    qv_old, qv_new, liquid, ice, dry_mass,
                    specific_kinetic, z, tke_old, tke_new,
                    dry_internal, water_sensible, phase_reference,
                    carrier_kinetic, gravitational, subgrid_tke,
                };
                if (std::any_of(
                        std::begin(values), std::end(values),
                        [](double value) { return !std::isfinite(value); })
                    || pressure <= 0.0 || exner <= 0.0
                    || temperature_old <= 0.0 || temperature_new <= 0.0
                    || qv_old < 0.0 || qv_new < 0.0
                    || liquid < 0.0 || ice < 0.0
                    || dry_mass <= 0.0 || tke_old < 0.0 || tke_new < 0.0)
                {
                    partial[6] += 1.0;
                    continue;
                }
                partial[0] += dry_internal;
                partial[1] += water_sensible;
                partial[2] += phase_reference;
                partial[3] += carrier_kinetic;
                partial[4] += gravitational;
                partial[5] += subgrid_tke;
            }
        });
    if (attribute_moist_energy && scalar_energy[6] != 0.0)
        throw std::runtime_error(
            "SGS moist-energy attribution rejected invalid state");

    TurbulenceMomentumWorkReport report;
    if (attribute_work)
    {
        report.requested_linear_work_j =
            momentum_work[0] + ground_work[0];
        report.finite_step_correction_j =
            momentum_work[1] + ground_work[1];
        report.committed_work_j =
            momentum_work[2] + ground_work[2];
        report.storage_rounding_j =
            report.committed_work_j - report.requested_linear_work_j
            - report.finite_step_correction_j;
        report.moist_dry_internal_change_j = scalar_energy[0];
        report.moist_water_sensible_change_j = scalar_energy[1];
        report.moist_phase_reference_change_j = scalar_energy[2];
        report.moist_carrier_kinetic_change_j = scalar_energy[3];
        report.moist_gravitational_change_j = scalar_energy[4];
        report.moist_subgrid_tke_change_j = scalar_energy[5];
        report.moist_energy_complete = attribute_moist_energy;
        if (attribute_moist_energy)
        {
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
        }
        if (!std::isfinite(report.requested_linear_work_j) ||
            !std::isfinite(report.finite_step_correction_j) ||
            !std::isfinite(report.storage_rounding_j) ||
            !std::isfinite(report.committed_work_j) ||
            report.finite_step_correction_j < 0.0)
        {
            throw std::runtime_error(
                "SGS momentum-work attribution produced an invalid ledger");
        }
    }
    return report;
}

struct DeviceSgsApplyAttributionContext
{
    const SimulationStateView* state = nullptr;
    const TurbulenceTendencies* tendencies = nullptr;
    double dt_dynamics = 0.0;
    bool attribute_work = false;
    TurbulenceMomentumWorkReport report;
};

TurbulenceMomentumWorkReport attribute_completed_sgs_commit(const SimulationStateView&  state,
                                                            double                      dt_dynamics,
                                                            const TurbulenceTendencies& turb_tend,
                                                            bool                        attribute_work,
                                                            const float*                u_committed,
                                                            const float*                v_committed,
                                                            const float*                w_committed,
                                                            const float*                theta_committed,
                                                            const float*                qv_committed,
                                                            const float*                tke_committed)
{
    const float* committed[] = {
        u_committed,
        v_committed,
        w_committed,
        theta_committed,
        qv_committed,
        tke_committed,
    };
    if (std::any_of(std::begin(committed), std::end(committed), [](const float* values) { return values == nullptr; }))
    {
        throw std::invalid_argument("device SGS attribution requires six committed fields");
    }

    const auto&       grid            = state.grid;
    const auto&       fields          = state.fields;
    const auto&       tendency_fields = state.tendencies;
    const std::size_t cell_count      = fields.u.size();
    const auto        linear_index    = [&](int i, int j, int k)
    {
        return (static_cast<std::size_t>(i) * static_cast<std::size_t>(grid.nth) + static_cast<std::size_t>(j)) *
                   static_cast<std::size_t>(grid.nz) +
               static_cast<std::size_t>(k);
    };
    const auto final_value = [&](const float* values, int i, int j, int k)
    { return static_cast<double>(values[linear_index(i, j, k)]); };

    // Warn/Strict writers must expose raw candidates. Reject an invalid
    // device result before copy-back, while the authoritative host state is
    // still untouched.
    if (!primary_state_repairs_enabled)
    {
        for (std::size_t n = 0; n < cell_count; ++n)
        {
            const float u     = u_committed[n];
            const float v     = v_committed[n];
            const float w     = w_committed[n];
            const float theta = theta_committed[n];
            const float qv    = qv_committed[n];
            const float tke   = tke_committed[n];
            if (!std::isfinite(u) || std::fabs(u) > wind_horizontal_abs_max_ms || !std::isfinite(v) ||
                std::fabs(v) > wind_horizontal_abs_max_ms || !std::isfinite(w) ||
                std::fabs(w) > wind_vertical_abs_max_ms || !std::isfinite(theta) || theta < theta_min_k ||
                theta > theta_max_k || !std::isfinite(qv) || qv < 0.0f || qv > qv_max_kgkg || !std::isfinite(tke) ||
                tke < 0.0f)
            {
                throw std::runtime_error("raw device SGS commit violated primary-state bounds");
            }
        }
    }
    if (!attribute_work)
        return {};

    if (!std::isfinite(dt_dynamics) || dt_dynamics <= 0.0 || grid.geometry.coord != CoordinateSystem::Cartesian ||
        !field_matches_domain(state, fields.rho) || !std::isfinite(grid.dr) || grid.dr <= 0.0 ||
        !std::isfinite(grid.dtheta) || grid.dtheta <= 0.0 || !std::isfinite(grid.dz) || grid.dz <= 0.0 || grid.nr < 3 ||
        grid.nth < 3 || grid.nz < 2)
    {
        throw std::invalid_argument("device SGS work attribution requires a finite positive "
                                    "Cartesian state and timestep");
    }
    for (std::size_t n = 0; n < fields.rho.size(); ++n)
    {
        const double density = static_cast<double>(fields.rho.data()[n]);
        if (!std::isfinite(density) || density <= 0.0)
            throw std::invalid_argument("device SGS attribution requires positive finite density");
    }
    tmv::physical_grid_metrics::validate(state);

    const bool               update_tke             = field_matches_domain(state, fields.tke) &&
                                                      field_matches_domain(state, tendency_fields.dtke_dt_pbl) &&
                                                      field_matches_domain(state, turb_tend.dtkedt_sgs);
    const bool               attribute_moist_energy = complete_moist_energy_state(state) && update_tke;
    const PhysicalCellDomain domain                 = physical_scalar_domain(grid);
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(state);
    const TerrainMetrics3D* terrain = tmv::physical_grid_metrics::active_terrain_metrics(state);
    if (terrain != nullptr && !grid.geometry.staggered)
        throw std::invalid_argument("native mapped collocated SGS attribution is unavailable");
    const bool terrain_ground_available =
        terrain != nullptr && terrain->Gx_face.size_r() == grid.nr && terrain->Gx_face.size_th() == grid.nth &&
        terrain->Gx_face.size_z() == grid.nz + 1 && terrain->Gy_face.size_r() == grid.nr &&
        terrain->Gy_face.size_th() == grid.nth && terrain->Gy_face.size_z() == grid.nz + 1;
    if (attribute_moist_energy && terrain != nullptr && !terrain_ground_available)
    {
        throw std::invalid_argument("mapped device SGS attribution requires complete terrain metrics");
    }
    const auto requested_delta = [&](const Field3D& tendency, int i, int j, int k)
    {
        double rate = static_cast<double>(tendency[i][j][k]);
        if (primary_state_repairs_enabled && !std::isfinite(rate))
            rate = 0.0;
        return rate * dt_dynamics;
    };

    std::array<double, 4> ground_work{};
    if (terrain_ground_available)
    {
        ground_work = tmv::diagnostics::fixed_slab_reduce<4>(
            domain.i_begin,
            domain.i_end,
            [&](int i, std::array<double, 4>& partial)
            {
                const int k = domain.k_begin;
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    const double slope_x = -terrain->Gx_face(i, j, k);
                    const double slope_y = -terrain->Gy_face(i, j, k);
                    const double u_w     = fields.u[i - 1][j][k];
                    const double u_e     = fields.u[i][j][k];
                    const double v_s     = fields.v[i][j - 1][k];
                    const double v_n     = fields.v[i][j][k];
                    const double du_w    = requested_delta(turb_tend.dudt_sgs, i - 1, j, k);
                    const double du_e    = requested_delta(turb_tend.dudt_sgs, i, j, k);
                    const double dv_s    = requested_delta(turb_tend.dvdt_sgs, i, j - 1, k);
                    const double dv_n    = requested_delta(turb_tend.dvdt_sgs, i, j, k);
                    const double frame_u = state.domain_frame != nullptr ? state.domain_frame->velocity_ms.u_ms : 0.0;
                    const double frame_v = state.domain_frame != nullptr ? state.domain_frame->velocity_ms.v_ms : 0.0;
                    const double old_ground =
                        (0.5 * (u_w + u_e) - frame_u) * slope_x + (0.5 * (v_s + v_n) - frame_v) * slope_y;
                    const double delta_ground = 0.5 * (du_w + du_e) * slope_x + 0.5 * (dv_s + dv_n) * slope_y;
                    const double new_ground =
                        (0.5 * (final_value(u_committed, i - 1, j, k) + final_value(u_committed, i, j, k)) - frame_u) *
                            slope_x +
                        (0.5 * (final_value(v_committed, i, j - 1, k) + final_value(v_committed, i, j, k)) - frame_v) *
                            slope_y;
                    const double face_mass =
                        0.5 * cell_carrier_mass_kg(state, mass_geometry, i, j, k, attribute_moist_energy);
                    partial[0] += face_mass * old_ground * delta_ground;
                    partial[1] += 0.5 * face_mass * delta_ground * delta_ground;
                    partial[2] += 0.5 * face_mass * (new_ground * new_ground - old_ground * old_ground);
                }
            });
    }

    const auto momentum_work = tmv::diagnostics::fixed_slab_reduce<4>(
        grid.geometry.staggered ? 0 : domain.i_begin,
        grid.geometry.staggered ? grid.nr : domain.i_end,
        [&](int i, std::array<double, 4>& partial)
        {
            const auto accumulate = [&](double mass, double old_velocity, double delta, double stored_velocity)
            {
                if (!std::isfinite(mass) || mass <= 0.0)
                {
                    partial[3] += 1.0;
                    return;
                }
                partial[0] += mass * old_velocity * delta;
                partial[1] += 0.5 * mass * delta * delta;
                partial[2] += 0.5 * mass * (stored_velocity * stored_velocity - old_velocity * old_velocity);
            };
            for (int j = 0; j < grid.nth; ++j)
                for (int k = 0; k < grid.nz; ++k)
                {
                    const double du = requested_delta(turb_tend.dudt_sgs, i, j, k);
                    const double dv = requested_delta(turb_tend.dvdt_sgs, i, j, k);
                    const double dw = requested_delta(turb_tend.dwdt_sgs, i, j, k);
                    if (!grid.geometry.staggered)
                    {
                        if (i >= domain.i_begin && i < domain.i_end && j >= domain.j_begin && j < domain.j_end &&
                            k >= domain.k_begin && k < domain.k_end)
                        {
                            const double mass =
                                cell_carrier_mass_kg(state, mass_geometry, i, j, k, attribute_moist_energy);
                            accumulate(mass, fields.u[i][j][k], du, final_value(u_committed, i, j, k));
                            accumulate(mass, fields.v[i][j][k], dv, final_value(v_committed, i, j, k));
                            accumulate(mass, fields.w[i][j][k], dw, final_value(w_committed, i, j, k));
                        }
                        continue;
                    }
                    if (j >= domain.j_begin && j < domain.j_end && k >= domain.k_begin && k < domain.k_end &&
                        i >= domain.i_begin - 1 && i < domain.i_end)
                    {
                        double mass = 0.0;
                        if (i >= domain.i_begin)
                            mass += 0.5 * cell_carrier_mass_kg(state, mass_geometry, i, j, k, attribute_moist_energy);
                        if (i + 1 < domain.i_end)
                            mass +=
                                0.5 * cell_carrier_mass_kg(state, mass_geometry, i + 1, j, k, attribute_moist_energy);
                        accumulate(mass, fields.u[i][j][k], du, final_value(u_committed, i, j, k));
                    }
                    if (i >= domain.i_begin && i < domain.i_end && k >= domain.k_begin && k < domain.k_end &&
                        j >= domain.j_begin - 1 && j < domain.j_end)
                    {
                        double mass = 0.0;
                        if (j >= domain.j_begin)
                            mass += 0.5 * cell_carrier_mass_kg(state, mass_geometry, i, j, k, attribute_moist_energy);
                        if (j + 1 < domain.j_end)
                            mass +=
                                0.5 * cell_carrier_mass_kg(state, mass_geometry, i, j + 1, k, attribute_moist_energy);
                        accumulate(mass, fields.v[i][j][k], dv, final_value(v_committed, i, j, k));
                    }
                    if (i >= domain.i_begin && i < domain.i_end && j >= domain.j_begin && j < domain.j_end &&
                        k >= domain.k_begin && k < domain.k_end)
                    {
                        double mass = 0.5 * cell_carrier_mass_kg(state, mass_geometry, i, j, k, attribute_moist_energy);
                        if (k + 1 < domain.k_end)
                            mass +=
                                0.5 * cell_carrier_mass_kg(state, mass_geometry, i, j, k + 1, attribute_moist_energy);
                        accumulate(mass, fields.w[i][j][k], dw, final_value(w_committed, i, j, k));
                    }
                }
        });
    if (momentum_work[3] != 0.0)
        throw std::runtime_error("device SGS attribution rejected invalid carrier mass");

    const auto committed_specific_kinetic = [&](int i, int j, int k)
    {
        if (!grid.geometry.staggered)
        {
            const double u = final_value(u_committed, i, j, k);
            const double v = final_value(v_committed, i, j, k);
            const double w = final_value(w_committed, i, j, k);
            return 0.5 * (u * u + v * v + w * w);
        }
        const double u_w = final_value(u_committed, i - 1, j, k);
        const double u_e = final_value(u_committed, i, j, k);
        const double v_s = final_value(v_committed, i, j - 1, k);
        const double v_n = final_value(v_committed, i, j, k);
        double       w_b = k == domain.k_begin ? 0.0 : final_value(w_committed, i, j, k - 1);
        if (terrain_ground_available && k == domain.k_begin)
        {
            const double frame_u = state.domain_frame != nullptr ? state.domain_frame->velocity_ms.u_ms : 0.0;
            const double frame_v = state.domain_frame != nullptr ? state.domain_frame->velocity_ms.v_ms : 0.0;
            w_b                  = (0.5 * (u_w + u_e) - frame_u) * (-terrain->Gx_face(i, j, domain.k_begin)) +
                                   (0.5 * (v_s + v_n) - frame_v) * (-terrain->Gy_face(i, j, domain.k_begin));
        }
        const double w_t = final_value(w_committed, i, j, k);
        return 0.25 * (u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n + w_b * w_b + w_t * w_t);
    };

    std::array<double, 7> scalar_energy{};
    if (attribute_moist_energy)
    {
        scalar_energy = tmv::diagnostics::fixed_slab_reduce<7>(
            grid.geometry.staggered ? 0 : domain.i_begin,
            grid.geometry.staggered ? grid.nr : domain.i_end,
            [&](int i, std::array<double, 7>& partial)
            {
                for (int j = 0; j < grid.nth; ++j)
                    for (int k = 0; k < grid.nz; ++k)
                    {
                        // Preserve the CPU owner's exact fixed-slab partition
                        // and merge order. Restricting the reduction range to
                        // physical cells changes binary64 association even
                        // when every per-cell term is identical.
                        if (i < domain.i_begin || i >= domain.i_end || j < domain.j_begin || j >= domain.j_end ||
                            k < domain.k_begin || k >= domain.k_end)
                            continue;
                        constexpr double cv_d =
                            physical_constants::specific_heat_cp_jkgk - physical_constants::gas_constant_dry_air_jkgk;
                        constexpr double cv_v            = physical_constants::specific_heat_water_vapor_cp_jkgk -
                                                           physical_constants::gas_constant_water_vapor_jkgk;
                        constexpr double c_l             = physical_constants::specific_heat_liquid_water_jkgk;
                        constexpr double c_i             = physical_constants::specific_heat_ice_jkgk;
                        constexpr double t_ref           = physical_constants::freezing_temperature_k;
                        constexpr double vapor_reference = physical_constants::latent_heat_vaporization_jkg -
                                                           physical_constants::gas_constant_water_vapor_jkgk * t_ref;
                        constexpr double kappa =
                            physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;
                        const double theta_old = fields.theta[i][j][k];
                        const double theta_new = final_value(theta_committed, i, j, k);
                        const double qv_old    = fields.qv[i][j][k];
                        const double qv_new    = final_value(qv_committed, i, j, k);
                        const double tke_old   = fields.tke[i][j][k];
                        const double tke_new   = final_value(tke_committed, i, j, k);
                        const double pressure  = fields.p[i][j][k];
                        const double exner     = std::pow(pressure / physical_constants::reference_pressure_pa, kappa);
                        const double temperature_old = theta_old * exner;
                        const double temperature_new = theta_new * exner;
                        const double liquid =
                            static_cast<double>(fields.qc[i][j][k]) + static_cast<double>(fields.qr[i][j][k]);
                        const double ice =
                            static_cast<double>(fields.qi[i][j][k]) + static_cast<double>(fields.qs[i][j][k]) +
                            static_cast<double>(fields.qg[i][j][k]) + static_cast<double>(fields.qh[i][j][k]);
                        const double dry_mass         = mass_geometry.scalar_mass_kg(fields.rho, i, j, k);
                        const double specific_kinetic = committed_specific_kinetic(i, j, k);
                        const double z                = tmv::physical_grid_metrics::scalar_height(state, i, j, k);
                        const double dry_internal     = dry_mass * cv_d * (temperature_new - temperature_old);
                        const double water_sensible =
                            dry_mass * ((temperature_new - t_ref) * (qv_new * cv_v + liquid * c_l + ice * c_i) -
                                        (temperature_old - t_ref) * (qv_old * cv_v + liquid * c_l + ice * c_i));
                        const double phase_reference = dry_mass * (qv_new - qv_old) * vapor_reference;
                        const double carrier_kinetic = dry_mass * (qv_new - qv_old) * specific_kinetic;
                        const double gravitational = dry_mass * (qv_new - qv_old) * physical_constants::gravity_ms2 * z;
                        const double subgrid_tke   = dry_mass * (tke_new - tke_old);
                        const double values[]      = {
                            pressure,
                            exner,
                            temperature_old,
                            temperature_new,
                            qv_old,
                            qv_new,
                            liquid,
                            ice,
                            dry_mass,
                            specific_kinetic,
                            z,
                            tke_old,
                            tke_new,
                            dry_internal,
                            water_sensible,
                            phase_reference,
                            carrier_kinetic,
                            gravitational,
                            subgrid_tke,
                        };
                        if (std::any_of(std::begin(values),
                                        std::end(values),
                                        [](double value) { return !std::isfinite(value); }) ||
                            pressure <= 0.0 || exner <= 0.0 || temperature_old <= 0.0 || temperature_new <= 0.0 ||
                            qv_old < 0.0 || qv_new < 0.0 || liquid < 0.0 || ice < 0.0 || dry_mass <= 0.0 ||
                            tke_old < 0.0 || tke_new < 0.0)
                        {
                            partial[6] += 1.0;
                            continue;
                        }
                        partial[0] += dry_internal;
                        partial[1] += water_sensible;
                        partial[2] += phase_reference;
                        partial[3] += carrier_kinetic;
                        partial[4] += gravitational;
                        partial[5] += subgrid_tke;
                    }
            });
        if (scalar_energy[6] != 0.0)
            throw std::runtime_error("device SGS moist-energy attribution rejected invalid state");
    }

    TurbulenceMomentumWorkReport report;
    report.requested_linear_work_j  = momentum_work[0] + ground_work[0];
    report.finite_step_correction_j = momentum_work[1] + ground_work[1];
    report.committed_work_j         = momentum_work[2] + ground_work[2];
    report.storage_rounding_j =
        report.committed_work_j - report.requested_linear_work_j - report.finite_step_correction_j;
    report.moist_dry_internal_change_j    = scalar_energy[0];
    report.moist_water_sensible_change_j  = scalar_energy[1];
    report.moist_phase_reference_change_j = scalar_energy[2];
    report.moist_carrier_kinetic_change_j = scalar_energy[3];
    report.moist_gravitational_change_j   = scalar_energy[4];
    report.moist_subgrid_tke_change_j     = scalar_energy[5];
    report.moist_energy_complete          = attribute_moist_energy;
    if (attribute_moist_energy)
    {
        report.moist_committed_energy_change_j =
            report.committed_work_j + report.moist_dry_internal_change_j + report.moist_water_sensible_change_j +
            report.moist_phase_reference_change_j + report.moist_carrier_kinetic_change_j +
            report.moist_gravitational_change_j + report.moist_subgrid_tke_change_j;
        report.moist_energy_closure_residual_j =
            report.moist_committed_energy_change_j -
            (report.committed_work_j + report.moist_dry_internal_change_j + report.moist_water_sensible_change_j +
             report.moist_phase_reference_change_j + report.moist_carrier_kinetic_change_j +
             report.moist_gravitational_change_j + report.moist_subgrid_tke_change_j);
    }
    if (!std::isfinite(report.requested_linear_work_j) || !std::isfinite(report.finite_step_correction_j) ||
        !std::isfinite(report.storage_rounding_j) || !std::isfinite(report.committed_work_j) ||
        report.finite_step_correction_j < 0.0)
    {
        throw std::runtime_error("device SGS attribution produced an invalid ledger");
    }
    return report;
}

void observe_completed_sgs_commit(void*        raw_context,
                                  const float* u_committed,
                                  const float* v_committed,
                                  const float* w_committed,
                                  const float* theta_committed,
                                  const float* qv_committed,
                                  const float* tke_committed)
{
    if (raw_context == nullptr)
        throw std::invalid_argument("device SGS observer context is null");
    auto& context = *static_cast<DeviceSgsApplyAttributionContext*>(raw_context);
    if (context.state == nullptr || context.tendencies == nullptr)
        throw std::invalid_argument("device SGS observer context is incomplete");
    context.report = attribute_completed_sgs_commit(
        *context.state, context.dt_dynamics, *context.tendencies,
        context.attribute_work,
        u_committed, v_committed, w_committed,
        theta_committed, qv_committed, tke_committed);
}

bool sgs_gpu_apply_ready(const SimulationStateView& state,
                         const TurbulenceTendencies& turb_tend,
                         std::size_t cell_count)
{
    return gpu_sgs_apply_enabled(state)
        && has_compute_context(state)
        && cell_count > 0
        && !state.fields.qv.empty() && !state.fields.tke.empty()
        && state.fields.tke.size() == cell_count
        && turb_tend.dudt_sgs.size() == cell_count
        && turb_tend.dvdt_sgs.size() == cell_count
        && turb_tend.dwdt_sgs.size() == cell_count
        && turb_tend.dthetadt_sgs.size() == cell_count
        && turb_tend.dqvdt_sgs.size() == cell_count
        && turb_tend.dtkedt_sgs.size() == cell_count
        && state.tendencies.dtke_dt_pbl.size() == cell_count;
}

bool try_apply_sgs_tendencies_gpu(const SimulationStateView& state,
                                  double dt_dynamics,
                                  const TurbulenceTendencies& turb_tend,
                                  TurbulenceMomentumWorkReport* report)
{
    auto& fields = state.fields;
    const auto& grid = state.grid;
    const std::size_t cell_count = fields.u.size();
    const bool attribute_work = active_sgs_work_ledger(state);
    if (!sgs_gpu_apply_ready(state, turb_tend, cell_count))
    {
        return false;
    }

    DeviceSgsApplyAttributionContext attribution_context;
    attribution_context.state = &state;
    attribution_context.tendencies = &turb_tend;
    attribution_context.dt_dynamics = dt_dynamics;
    attribution_context.attribute_work = attribute_work;

    const auto try_gpu = [&]() {
        return dispatch_sgs_apply_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"),
            fields.u.data(), fields.v.data(), fields.w.data(),
            fields.theta.data(), fields.qv.data(), fields.tke.data(),
            turb_tend.dudt_sgs.data(), turb_tend.dvdt_sgs.data(),
            turb_tend.dwdt_sgs.data(), turb_tend.dthetadt_sgs.data(),
            turb_tend.dqvdt_sgs.data(), turb_tend.dtkedt_sgs.data(),
            state.tendencies.dtke_dt_pbl.data(),
            grid.nr, grid.nth, grid.nz,
            true, true,
            static_cast<float>(dt_dynamics),
            static_cast<float>(theta0), theta_min_k, theta_max_k,
            qv_max_kgkg,
            wind_horizontal_abs_max_ms, wind_vertical_abs_max_ms,
            turb_tend.resident_source_unchanged,
            primary_state_repairs_enabled,
            &observe_completed_sgs_commit,
            &attribution_context,
            !grid.geometry.staggered);
    };

    if (!nimbus::parity::sample("sgs_apply"))
    {
        const bool dispatched = try_gpu();
        if (dispatched && report != nullptr)
            *report = attribution_context.report;
        return dispatched;
    }

    Field3D pre_u = fields.u, pre_v = fields.v, pre_w = fields.w;
    Field3D pre_theta = fields.theta, pre_qv = fields.qv;
    Field3D pre_tke = fields.tke;
    if (!try_gpu())
    {
        return false;
    }

    Field3D gpu_u = fields.u, gpu_v = fields.v, gpu_w = fields.w;
    Field3D gpu_theta = fields.theta, gpu_qv = fields.qv;
    Field3D gpu_tke = fields.tke;
    const std::size_t bytes = cell_count * sizeof(float);
    std::memcpy(fields.u.data(), pre_u.data(), bytes);
    std::memcpy(fields.v.data(), pre_v.data(), bytes);
    std::memcpy(fields.w.data(), pre_w.data(), bytes);
    std::memcpy(fields.theta.data(), pre_theta.data(), bytes);
    std::memcpy(fields.qv.data(), pre_qv.data(), bytes);
    std::memcpy(fields.tke.data(), pre_tke.data(), bytes);

    (void)apply_sgs_tendencies_cpu(
        state, dt_dynamics, turb_tend, false);

    using nimbus::parity::Tier;
    const char* kn = "sgs_apply";
    nimbus::parity::check_field(kn, "u",
        fields.u.data(), gpu_u.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "v",
        fields.v.data(), gpu_v.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "w",
        fields.w.data(), gpu_w.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "theta",
        fields.theta.data(), gpu_theta.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "qv",
        fields.qv.data(), gpu_qv.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "tke",
        fields.tke.data(), gpu_tke.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);

    std::memcpy(fields.u.data(), gpu_u.data(), bytes);
    std::memcpy(fields.v.data(), gpu_v.data(), bytes);
    std::memcpy(fields.w.data(), gpu_w.data(), bytes);
    std::memcpy(fields.theta.data(), gpu_theta.data(), bytes);
    std::memcpy(fields.qv.data(), gpu_qv.data(), bytes);
    std::memcpy(fields.tke.data(), gpu_tke.data(), bytes);
    if (report != nullptr)
        *report = attribution_context.report;
    return true;
}

} // namespace

SgsExplicitStability compute_sgs_explicit_stability(
    const SimulationStateView& state,
    const TurbulenceTendencies& turb_tend)
{
    const auto& grid = state.grid;
    const auto& fields = state.fields;
    const bool has_qv = !fields.qv.empty();
    const bool has_tke = !fields.tke.empty();
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    const bool dimensions_valid = grid.geometry.staggered
        ? grid.nr >= 3 && grid.nth >= 3 && grid.nz >= 2
        : grid.nr >= 5 && grid.nth >= 5 && grid.nz >= 5;
    if (!dimensions_valid ||
        !std::isfinite(grid.dr) || grid.dr <= 0.0 ||
        !std::isfinite(grid.dtheta) || grid.dtheta <= 0.0 ||
        !std::isfinite(grid.dz) || grid.dz <= 0.0 ||
        !field_matches_domain(state, fields.rho) ||
        !field_matches_domain(state, fields.u) ||
        !field_matches_domain(state, fields.v) ||
        !field_matches_domain(state, fields.w) ||
        !field_matches_domain(state, fields.theta) ||
        !field_matches_domain(state, turb_tend.dudt_sgs) ||
        !field_matches_domain(state, turb_tend.dvdt_sgs) ||
        !field_matches_domain(state, turb_tend.dwdt_sgs) ||
        !field_matches_domain(state, turb_tend.dthetadt_sgs) ||
        (has_qv && (!field_matches_domain(state, fields.qv) ||
                    !field_matches_domain(state, turb_tend.dqvdt_sgs))) ||
        (has_tke && (!field_matches_domain(state, fields.tke) ||
                     !field_matches_domain(state, turb_tend.dtkedt_sgs) ||
                     !field_matches_domain(state,
                                           state.tendencies.dtke_dt_pbl))))
    {
        throw std::invalid_argument(
            "SGS stability evaluation requires an exact Cartesian physical-grid "
            "state and complete realized tendencies");
    }
    tmv::physical_grid_metrics::validate(state);
    const auto j_extent = static_cast<std::uint64_t>(domain.j_end - domain.j_begin);
    const auto k_extent = static_cast<std::uint64_t>(domain.k_end - domain.k_begin);
    const auto cells = static_cast<std::uint64_t>(domain.i_end - domain.i_begin) * j_extent * k_extent;
    // The minimum (cell, category) key preserves serial fail-first diagnostics.
    constexpr auto no_invalid_input = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t first_invalid_code = no_invalid_input;
    #pragma omp parallel for collapse(2) schedule(static) reduction(min:first_invalid_code) if(cells >= 65536)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double values[] = {
                    static_cast<double>(fields.rho(i, j, k)),
                    static_cast<double>(fields.u(i, j, k)),
                    static_cast<double>(fields.v(i, j, k)),
                    static_cast<double>(fields.w(i, j, k)),
                    static_cast<double>(fields.theta(i, j, k)),
                    static_cast<double>(turb_tend.dudt_sgs(i, j, k)),
                    static_cast<double>(turb_tend.dvdt_sgs(i, j, k)),
                    static_cast<double>(turb_tend.dwdt_sgs(i, j, k)),
                    static_cast<double>(turb_tend.dthetadt_sgs(i, j, k))};
                bool finite = true;
                for (double value : values) finite &= std::isfinite(value);
                const bool moisture_bad = has_qv
                    && (!std::isfinite(static_cast<double>(fields.qv(i, j, k)))
                        || fields.qv(i, j, k) < 0.0f
                        || !std::isfinite(static_cast<double>(
                            turb_tend.dqvdt_sgs(i, j, k))));
                const bool tke_bad = has_tke
                    && (!std::isfinite(static_cast<double>(fields.tke(i, j, k)))
                        || fields.tke(i, j, k) < 0.0f
                        || !std::isfinite(static_cast<double>(
                            turb_tend.dtkedt_sgs(i, j, k)))
                        || !std::isfinite(static_cast<double>(
                            state.tendencies.dtke_dt_pbl(i, j, k))));
                const std::uint64_t category = !finite ? 0u : values[0] <= 0.0 ? 1u
                    : values[4] <= 0.0 ? 2u : moisture_bad ? 3u : tke_bad ? 4u : 5u;
                const auto ordinal = (static_cast<std::uint64_t>(i - domain.i_begin) * j_extent +
                    static_cast<std::uint64_t>(j - domain.j_begin)) * k_extent +
                    static_cast<std::uint64_t>(k - domain.k_begin);
                const auto code = category < 5u ? ordinal * 8u + category : no_invalid_input;
                first_invalid_code = std::min(first_invalid_code, code);
            }
    if (first_invalid_code != no_invalid_input)
    {
        static const char* const errors[] = {
            "SGS stability evaluation requires finite physical state and tendencies",
            "SGS stability evaluation requires positive live density",
            "SGS stability evaluation requires positive theta",
            "SGS stability evaluation requires finite non-negative moisture and a finite tendency",
            "SGS stability evaluation requires finite non-negative TKE and finite tendencies"};
        throw std::invalid_argument(errors[first_invalid_code % 8u]);
    }

    const bool wrap_x =
        tmv::require_run_configuration(state).lateral_bc_west == tmv::init::config::LateralBC::Periodic ||
        tmv::require_run_configuration(state).lateral_bc_east == tmv::init::config::LateralBC::Periodic;
    const bool wrap_y =
        tmv::require_run_configuration(state).lateral_bc_south == tmv::init::config::LateralBC::Periodic ||
        tmv::require_run_configuration(state).lateral_bc_north == tmv::init::config::LateralBC::Periodic;
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(state);

    double linear_power = 0.0;
    double acceleration_norm = 0.0;
    double theta_min = std::numeric_limits<double>::infinity();
    double theta_max = -std::numeric_limits<double>::infinity();
    double qv_min = std::numeric_limits<double>::infinity();
    double qv_max = -std::numeric_limits<double>::infinity();

    const bool open_lateral = has_open_lateral_boundary(state);
    const int extrema_i_begin = grid.geometry.staggered && open_lateral
        ? 0 : domain.i_begin;
    const int extrema_i_end = grid.geometry.staggered && open_lateral
        ? grid.nr : domain.i_end;
    const int extrema_j_begin = grid.geometry.staggered && open_lateral
        ? 0 : domain.j_begin;
    const int extrema_j_end = grid.geometry.staggered && open_lateral
        ? grid.nth : domain.j_end;
    const auto accumulate_momentum =
        [&](int i, int j, int k, double& power, double& norm)
    {
        if (!grid.geometry.staggered)
        {
            if (i < domain.i_begin || i >= domain.i_end
                || j < domain.j_begin || j >= domain.j_end
                || k < domain.k_begin || k >= domain.k_end)
                return;
            const double mass = mass_geometry.scalar_mass_kg(
                fields.rho, i, j, k);
            const double velocities[] = {
                fields.u(i, j, k), fields.v(i, j, k), fields.w(i, j, k)};
            const double accelerations[] = {
                turb_tend.dudt_sgs(i, j, k),
                turb_tend.dvdt_sgs(i, j, k),
                turb_tend.dwdt_sgs(i, j, k)};
            for (int component = 0; component < 3; ++component)
            {
                power += mass * velocities[component]
                    * accelerations[component];
                norm += mass * accelerations[component]
                    * accelerations[component];
            }
            return;
        }
        if (i == 0 || i >= grid.nr - 1
            || j == 0 || j >= grid.nth - 1
            || k >= grid.nz - 1)
            return;
        if (wrap_x || i <= grid.nr - 3)
        {
            const int i1 = wrap_x && i >= grid.nr - 2 ? 1 : i + 1;
            const double mass = mass_geometry.u_dual_mass_kg(
                fields.rho, i, j, k, i1);
            const double velocity =
                static_cast<double>(fields.u(i, j, k));
            const double acceleration =
                static_cast<double>(turb_tend.dudt_sgs(i, j, k));
            power += mass * velocity * acceleration;
            norm += mass * acceleration * acceleration;
        }
        if (wrap_y || j < grid.nth - 2)
        {
            const int j1 = wrap_y && j >= grid.nth - 2 ? 1 : j + 1;
            const double mass = mass_geometry.v_dual_mass_kg(
                fields.rho, i, j, k, j1);
            const double velocity =
                static_cast<double>(fields.v(i, j, k));
            const double acceleration =
                static_cast<double>(turb_tend.dvdt_sgs(i, j, k));
            power += mass * velocity * acceleration;
            norm += mass * acceleration * acceleration;
        }
        const double mass = mass_geometry.w_dual_mass_kg(
            fields.rho, i, j, k);
        const double velocity = static_cast<double>(fields.w(i, j, k));
        const double acceleration =
            static_cast<double>(turb_tend.dwdt_sgs(i, j, k));
        power += mass * velocity * acceleration;
        norm += mass * acceleration * acceleration;
    };

    if (tmv::diagnostics::deterministic_reductions_enabled())
    {
        const int slab_count = extrema_i_end - extrema_i_begin;
        auto& partials = tmv::diagnostics::fixed_slab_scratch<6>();
        partials.assign(
            static_cast<std::size_t>(slab_count),
            std::array<double, 6>{
                0.0, 0.0,
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity()});

        #pragma omp parallel for schedule(static) if(static_cast<std::size_t>(grid.nr) * grid.nth * grid.nz >= 32768)
        for (int slab = 0; slab < slab_count; ++slab)
        {
            const int i = extrema_i_begin + slab;
            auto& partial = partials[static_cast<std::size_t>(slab)];
            for (int j = extrema_j_begin; j < extrema_j_end; ++j)
            {
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    accumulate_momentum(
                        i, j, k, partial[0], partial[1]);
                    const double theta_here =
                        static_cast<double>(fields.theta(i, j, k));
                    partial[2] = std::min(partial[2], theta_here);
                    partial[3] = std::max(partial[3], theta_here);
                    if (has_qv)
                    {
                        const double qv_here =
                            static_cast<double>(fields.qv(i, j, k));
                        partial[4] = std::min(partial[4], qv_here);
                        partial[5] = std::max(partial[5], qv_here);
                    }
                }
            }
        }

        tmv::diagnostics::CompensatedSum power_sum;
        tmv::diagnostics::CompensatedSum norm_sum;
        for (const auto& partial : partials)
        {
            power_sum.add(partial[0]);
            norm_sum.add(partial[1]);
            theta_min = std::min(theta_min, partial[2]);
            theta_max = std::max(theta_max, partial[3]);
            qv_min = std::min(qv_min, partial[4]);
            qv_max = std::max(qv_max, partial[5]);
        }
        linear_power = power_sum.value();
        acceleration_norm = norm_sum.value();
    }
    else
    {
        #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(grid.nr) * grid.nth * grid.nz >= 32768) reduction(+:linear_power,acceleration_norm) reduction(min:theta_min,qv_min) reduction(max:theta_max,qv_max)
        for (int i = extrema_i_begin; i < extrema_i_end; ++i)
        {
            for (int j = extrema_j_begin; j < extrema_j_end; ++j)
            {
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double theta_here =
                        static_cast<double>(fields.theta(i, j, k));
                    theta_min = std::min(theta_min, theta_here);
                    theta_max = std::max(theta_max, theta_here);
                    if (has_qv)
                    {
                        const double qv_here =
                            static_cast<double>(fields.qv(i, j, k));
                        qv_min = std::min(qv_min, qv_here);
                        qv_max = std::max(qv_max, qv_here);
                    }

                    accumulate_momentum(
                        i, j, k, linear_power, acceleration_norm);
                }
            }
        }
    }

    if (!std::isfinite(linear_power) || !std::isfinite(acceleration_norm) ||
        acceleration_norm < 0.0)
    {
        throw std::runtime_error(
            "SGS momentum stability reduction produced an invalid result");
    }

    SgsExplicitStability result;
    constexpr double safety = 0.9;
    if (acceleration_norm > 0.0)
    {
        const TurbulenceRuntimeState& runtime = turbulence_runtime(state);
        const bool has_open_stress_boundary = open_lateral
            || (has_open_top_boundary(state)
                && runtime.config.mode != "horizontal_only");
        const bool current_realized_operator =
            &turb_tend == &runtime.held_tendencies
            && runtime.realized_transport_tendency_current;
        const double directional_boundary_power =
            has_open_stress_boundary && current_realized_operator
            ? (tmv::physical_grid_metrics::terrain_is_active(state)
                ? turb_tend.mapped_boundary_power_w
                : realized_directional_stress_boundary_power_w(state))
            : std::numeric_limits<double>::quiet_NaN();
        // Exact discrete summation by parts on the advanced control volumes:
        // <u,L(u)> = B - D. B is evaluated independently from the realized
        // boundary contract; the remainder is the operator-identical
        // irreversible loss.
        const double directional_dissipation_power =
            std::isfinite(directional_boundary_power)
            ? directional_boundary_power - linear_power
            : std::numeric_limits<double>::quiet_NaN();
        if (has_open_stress_boundary
            && std::isfinite(directional_dissipation_power)
            && directional_dissipation_power > 0.0
            && std::isfinite(directional_boundary_power))
        {
            // Bound the positive forward-Euler quadratic correction by the
            // exact irreversible remainder, while allowing resolved traction
            // work to enter or leave the open control volume.
            result.momentum_irreversible_dissipation_w =
                directional_dissipation_power;
            result.momentum_boundary_work_w = directional_boundary_power;
            result.momentum_stress_closure_residual_w = linear_power
                + directional_dissipation_power
                - directional_boundary_power;
            result.open_boundary_work_accounted = true;
            if (!std::isfinite(result.momentum_boundary_work_w)
                || !std::isfinite(
                    result.momentum_stress_closure_residual_w))
                throw std::runtime_error(
                    "realized SGS boundary-work attribution is invalid");
            result.momentum_energy_timestep_s = safety
                * (2.0 * directional_dissipation_power
                   / acceleration_norm);
        }
        else if (linear_power < 0.0)
        {
            result.momentum_irreversible_dissipation_w = -linear_power;
            result.momentum_stress_closure_residual_w =
                linear_power + result.momentum_irreversible_dissipation_w;
            result.momentum_energy_timestep_s =
                safety * (-2.0 * linear_power / acceleration_norm);
        }
        else
        {
            throw std::runtime_error(
                "realized SGS momentum tendency is not dissipative: "
                "linear_power_w=" + std::to_string(linear_power)
                + ", acceleration_norm="
                + std::to_string(acceleration_norm)
                + ", directional_dissipation_power_w="
                + std::to_string(directional_dissipation_power));
        }
    }

    double theta_limit = std::numeric_limits<double>::infinity();
    double qv_limit = std::numeric_limits<double>::infinity();
    double tke_limit = std::numeric_limits<double>::infinity();
    const bool mapped_tensor_diffusion =
        tmv::physical_grid_metrics::terrain_is_active(state);
    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(grid.nr) * grid.nth * grid.nz >= 32768) reduction(min:theta_limit,qv_limit,tke_limit)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double theta_here =
                    static_cast<double>(fields.theta(i, j, k));
                const double theta_rate = static_cast<double>(
                    turb_tend.dthetadt_sgs(i, j, k));
                const double theta_lower = mapped_tensor_diffusion
                    ? static_cast<double>(theta_min_k) : theta_min;
                const double theta_upper = mapped_tensor_diffusion
                    ? static_cast<double>(theta_max_k) : theta_max;
                if (theta_rate > 0.0)
                    theta_limit = std::min(
                        theta_limit, (theta_upper - theta_here) / theta_rate);
                else if (theta_rate < 0.0)
                    theta_limit = std::min(
                        theta_limit, (theta_here - theta_lower) / -theta_rate);

                if (has_qv)
                {
                    const double qv_here =
                        static_cast<double>(fields.qv(i, j, k));
                    const double qv_rate = static_cast<double>(
                        turb_tend.dqvdt_sgs(i, j, k));
                    const double qv_lower = mapped_tensor_diffusion
                        ? 0.0 : qv_min;
                    const double qv_upper = mapped_tensor_diffusion
                        ? static_cast<double>(qv_max_kgkg) : qv_max;
                    if (qv_rate > 0.0)
                        qv_limit = std::min(
                            qv_limit, (qv_upper - qv_here) / qv_rate);
                    else if (qv_rate < 0.0)
                        qv_limit = std::min(
                            qv_limit, (qv_here - qv_lower) / -qv_rate);
                }

                if (has_tke)
                {
                    const double tke_here =
                        static_cast<double>(fields.tke(i, j, k));
                    const double tke_rate =
                        static_cast<double>(turb_tend.dtkedt_sgs(i, j, k)) +
                        static_cast<double>(
                            state.tendencies.dtke_dt_pbl(i, j, k));
                    if (tke_rate < 0.0)
                        tke_limit = std::min(
                            tke_limit, tke_here / -tke_rate);
                }
            }
        }
    }

    const auto admit_limit = [](double raw, const char* label)
    {
        // Some OpenMP runtimes materialize the min-reduction identity as the
        // largest finite double rather than +infinity.
        if (std::isinf(raw) ||
            raw >= 0.5 * std::numeric_limits<double>::max())
            return std::numeric_limits<double>::infinity();
        if (!std::isfinite(raw) || raw <= 0.0)
            throw std::runtime_error(
                std::string("realized SGS tendency has no positive ") +
                label + " timestep");
        return safety * raw;
    };
    result.theta_monotonic_timestep_s =
        admit_limit(theta_limit, "theta-monotonic");
    result.qv_monotonic_timestep_s =
        admit_limit(qv_limit, "moisture-monotonic");
    if (mapped_tensor_diffusion)
    {
        // The mapped horizontal tensor contains non-orthogonal cross fluxes,
        // so its conservative centered stencil is not an M-matrix and cannot
        // promise the flat-grid discrete maximum principle. Admit it by the
        // exact forward-Euler scalar-energy root plus the physical bounds
        // above. Subtracting theta0 avoids cancellation against the large
        // constant thermodynamic reference; conservative fluxes annihilate
        // that constant exactly.
        const std::array<double, 4> scalar_energy =
            tmv::diagnostics::fixed_slab_reduce<4>(
                domain.i_begin, domain.i_end,
                [&](int i, std::array<double, 4>& partial)
                {
                    for (int j = domain.j_begin; j < domain.j_end; ++j)
                        for (int k = domain.k_begin; k < domain.k_end; ++k)
                        {
                            const double mass = mass_geometry.scalar_mass_kg(
                                fields.rho, i, j, k);
                            const double theta_rate = static_cast<double>(
                                turb_tend.dthetadt_sgs(i, j, k));
                            partial[0] += mass
                                * (static_cast<double>(fields.theta(i, j, k))
                                   - theta0)
                                * theta_rate;
                            partial[1] += mass * theta_rate * theta_rate;
                            if (has_qv)
                            {
                                const double qv_rate = static_cast<double>(
                                    turb_tend.dqvdt_sgs(i, j, k));
                                partial[2] += mass
                                    * static_cast<double>(fields.qv(i, j, k))
                                    * qv_rate;
                                partial[3] += mass * qv_rate * qv_rate;
                            }
                        }
                });
        const auto energy_limit = [&](double power, double norm,
                                      const char* label)
        {
            if (!std::isfinite(power) || !std::isfinite(norm) || norm < 0.0)
                throw std::runtime_error(
                    std::string("mapped SGS ") + label
                    + " energy reduction produced an invalid result");
            if (norm == 0.0)
                return std::numeric_limits<double>::infinity();
            if (!(power < 0.0))
                throw std::runtime_error(
                    std::string("mapped SGS ") + label
                    + " tendency is not energy dissipative");
            const double limit = safety * (-2.0 * power / norm);
            if (!std::isfinite(limit) || limit <= 0.0)
                throw std::runtime_error(
                    std::string("mapped SGS ") + label
                    + " energy timestep is invalid");
            return limit;
        };
        result.theta_energy_timestep_s = energy_limit(
            scalar_energy[0], scalar_energy[1], "theta");
        if (has_qv)
            result.qv_energy_timestep_s = energy_limit(
                scalar_energy[2], scalar_energy[3], "moisture");
    }
    result.tke_positivity_timestep_s =
        admit_limit(tke_limit, "TKE-positive");
    result.maximum_timestep_s = std::min({
        result.momentum_energy_timestep_s,
        result.theta_monotonic_timestep_s,
        result.qv_monotonic_timestep_s,
        result.theta_energy_timestep_s,
        result.qv_energy_timestep_s,
        result.tke_positivity_timestep_s});
    if (!(result.maximum_timestep_s > 0.0) ||
        std::isnan(result.maximum_timestep_s))
        throw std::runtime_error(
            "realized SGS explicit stability limit is invalid");
    return result;
}

TurbulenceMomentumWorkReport apply_admitted_sgs_substep(
    const SimulationStateView& state,
    double dt_dynamics,
    const TurbulenceTendencies& turb_tend)
{
    if (!std::isfinite(dt_dynamics) || dt_dynamics <= 0.0 ||
        state.grid.nr <= 0 || state.grid.nth <= 0 || state.grid.nz <= 0 ||
        !field_matches_domain(state, state.fields.u) ||
        !field_matches_domain(state, state.fields.v) ||
        !field_matches_domain(state, state.fields.w) ||
        !field_matches_domain(state, state.fields.theta) ||
        !field_matches_domain(state, turb_tend.dudt_sgs) ||
        !field_matches_domain(state, turb_tend.dvdt_sgs) ||
        !field_matches_domain(state, turb_tend.dwdt_sgs) ||
        !field_matches_domain(state, turb_tend.dthetadt_sgs) ||
        (!state.fields.qv.empty() &&
         (!field_matches_domain(state, state.fields.qv) ||
          !field_matches_domain(state, turb_tend.dqvdt_sgs))) ||
        (!state.fields.tke.empty() &&
         (!field_matches_domain(state, state.fields.tke) ||
          !field_matches_domain(state, turb_tend.dtkedt_sgs) ||
          !field_matches_domain(state, state.tendencies.dtke_dt_pbl))))
    {
        throw std::invalid_argument(
            "SGS apply requires a positive finite timestep and exact shapes "
            "for every participating state and tendency field");
    }
    const bool attribute_work = active_sgs_work_ledger(state);
    TurbulenceMomentumWorkReport report;
    if (!try_apply_sgs_tendencies_gpu(
            state, dt_dynamics, turb_tend, &report))
    {
        if (native_sgs_apply_required(state))
        {
            throw std::runtime_error(
                "qualified fallback-off accelerator SGS application dispatch failed");
        }
        report = apply_sgs_tendencies_cpu(
            state, dt_dynamics, turb_tend, attribute_work);
    }
    // Every admitted substep commits a new nonlinear state.  In particular,
    // the CPU work-attribution path updates these host arrays after the fused
    // Smagorinsky producer has retained its input fields on the device.  Do
    // not let the next coefficient refresh mistake those pre-commit device
    // values for the authoritative post-commit state.  This preserves reuse
    // within one Smagorinsky -> SGS-apply pair while requiring the upload that
    // is physically necessary before the next nonlinear substep.
    invalidate_resident_field_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), "u");
    invalidate_resident_field_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), "v");
    invalidate_resident_field_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), "w");
    invalidate_resident_field_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), "theta");
    invalidate_resident_field_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), "qv");
    invalidate_resident_field_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), "tke");
    return report;
}

TurbulenceMomentumWorkReport apply_sgs_tendencies(
    const SimulationStateView& state,
    double dt_dynamics,
    const TurbulenceTendencies& turb_tend)
{
    const auto& runtime = turbulence_runtime(state);
    const bool requires_explicit_admission = runtime.scheme &&
        (runtime.scheme->name() == "smagorinsky" ||
         runtime.scheme->name() == "tke");
    if (requires_explicit_admission)
    {
        const double limit_s =
            compute_sgs_explicit_stability(state, turb_tend)
                .maximum_timestep_s;
        const double tolerance_s = 1.0e-12 *
            std::max({1.0, dt_dynamics,
                      std::isfinite(limit_s) ? limit_s : 1.0});
        if (dt_dynamics > limit_s + tolerance_s)
            throw std::runtime_error(
                "SGS apply timestep exceeds the realized explicit "
                "stability limit");
    }
    TurbulenceMomentumWorkReport report =
        apply_admitted_sgs_substep(state, dt_dynamics, turb_tend);
    if (active_sgs_work_ledger(state))
        commit_sgs_momentum_work(state, report);
    return report;
}

void accumulate_sgs_energy_work(
    TurbulenceMomentumWorkReport& destination,
    const TurbulenceMomentumWorkReport& source)
{
    const bool destination_empty =
        destination.requested_linear_work_j == 0.0
        && destination.finite_step_correction_j == 0.0
        && destination.storage_rounding_j == 0.0
        && destination.committed_work_j == 0.0
        && destination.moist_committed_energy_change_j == 0.0
        && !destination.moist_energy_complete;
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
    const double reservoir_sum =
        destination.committed_work_j
        + destination.moist_dry_internal_change_j
        + destination.moist_water_sensible_change_j
        + destination.moist_phase_reference_change_j
        + destination.moist_carrier_kinetic_change_j
        + destination.moist_gravitational_change_j
        + destination.moist_subgrid_tke_change_j;
    destination.moist_energy_closure_residual_j =
        destination.moist_committed_energy_change_j - reservoir_sum;
    destination.moist_energy_complete = destination_empty
        ? source.moist_energy_complete
        : destination.moist_energy_complete && source.moist_energy_complete;
}

void commit_sgs_momentum_work(
    const SimulationStateView& state,
    const TurbulenceMomentumWorkReport& report)
{
    const double moist_values[] = {
        report.moist_dry_internal_change_j,
        report.moist_water_sensible_change_j,
        report.moist_phase_reference_change_j,
        report.moist_carrier_kinetic_change_j,
        report.moist_gravitational_change_j,
        report.moist_subgrid_tke_change_j,
        report.moist_committed_energy_change_j,
        report.moist_energy_closure_residual_j,
    };
    const double moist_reservoir_sum =
        report.committed_work_j
        + report.moist_dry_internal_change_j
        + report.moist_water_sensible_change_j
        + report.moist_phase_reference_change_j
        + report.moist_carrier_kinetic_change_j
        + report.moist_gravitational_change_j
        + report.moist_subgrid_tke_change_j;
    const bool moist_all_zero = std::all_of(
        std::begin(moist_values), std::end(moist_values),
        [](double value) { return value == 0.0; });
    if (!std::isfinite(report.requested_linear_work_j) ||
        !std::isfinite(report.finite_step_correction_j) ||
        !std::isfinite(report.storage_rounding_j) ||
        !std::isfinite(report.committed_work_j) ||
        report.finite_step_correction_j < 0.0 ||
        !std::all_of(
            std::begin(moist_values), std::end(moist_values),
            [](double value) { return std::isfinite(value); }) ||
        (report.moist_energy_complete &&
         report.moist_energy_closure_residual_j
             != report.moist_committed_energy_change_j
                - moist_reservoir_sum) ||
        (!report.moist_energy_complete && !moist_all_zero))
    {
        throw std::invalid_argument(
            "cannot commit an invalid SGS energy-work report");
    }
    TurbulenceRuntimeState& runtime = turbulence_runtime(state);
    runtime.last_momentum_work = report;
    accumulate_sgs_energy_work(runtime.cumulative_moist_energy_work, report);
    runtime.cumulative_requested_momentum_work_j +=
        report.requested_linear_work_j;
    runtime.cumulative_finite_step_momentum_correction_j +=
        report.finite_step_correction_j;
    runtime.cumulative_momentum_storage_rounding_j +=
        report.storage_rounding_j;
    runtime.cumulative_committed_momentum_work_j += report.committed_work_j;
    ++runtime.momentum_work_step_count;
}

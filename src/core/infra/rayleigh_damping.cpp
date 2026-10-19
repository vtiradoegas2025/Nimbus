// Rayleigh sponges relax velocity toward the environmental wind and w = 0;
// density and pressure are intentionally not relaxed (science audit C6).
// The smooth onset follows
//   alpha(d) = alpha_max sin^2[(pi/2)(1 - d / sponge_width)],
// which limits reflection at the sponge's inner edge (Klemp-Lilly 1978).

#include "core/infra/rayleigh_damping.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/field/field_bounds.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "init/hodograph/wind_profile.hpp"
#include "terrain/conservative_transport.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declaration: defined in nimbus_sim.cpp
void compute_wind_profile(const WindProfile& profile, double z, double& u, double& v);

namespace
{

struct RayleighConfig
{
    // Note: alpha_max and z_damp are now sourced from global_damping_alpha
    // and global_damping_z_d (Phase C3). The legacy depth_fraction=0.3
    // semantics still apply when global_damping_z_d is the sentinel -1
    // (resolved at init time to z_top * 0.7). LOAD-BEARING knob: dropping
    // alpha to 0 catastrophically destabilises cgrid_tornado_genesis
    // (max_u 180 by step 180, all near-lid k=22); the top sponge is the
    // primary absorber for the lid-BC residual (linear face-averaged p
    // extrapolation at the rigid lid leaks ~(1/12)*alpha^3 perturbation
    // per acoustic substep — see Phase D.5 in CyclostrophicConservativePlan.md).
    double depth_fraction = 0.3;
};

struct LateralSpongeConfig
{
    int    width_points    = 20;   // sponge width in grid points on each face
    double alpha_max       = 0.05; // maximum damping rate (1/s). LOAD-BEARING.
                                   // Phase D closed the discrete cyclostrophic balance
                                   // residual via trapezoidal centrifugal + matching
                                   // dp_cyclo IC. Phase D.5 diagnostic identified the
                                   // remaining secondary-circulation source as the lid
                                   // BC. Phase D.5b/c tried six fixes (sponge tuning,
                                   // gamma-damped, cubic Taylor, exp-ratio, r^2*u*v
                                   // conservative dv/dt, IC+BC coupled) -- all REJECTED.
                                   // The OLD lid BC's dynamic rho_face provides implicit
                                   // dissipation that's hard to replicate. Sponge bump
                                   // remains load-bearing pending bigger-scope work
                                   // (one-way absorbing BC, or accepting alpha=0.05).
                                   // See plan section 8 for the catalog of attempts.
};

RayleighConfig g_rayleigh_config;
LateralSpongeConfig g_lateral_config;
struct RayleighRuntimeView
{
    bool& upper_initialized;
    bool& lateral_initialized;
    std::vector<double>& alpha;
    std::vector<double>& u_ref;
    std::vector<double>& v_ref;
    std::vector<double>& upper_w_alpha;
    std::vector<double>& w_only_alpha;
    std::vector<double>& lateral_alpha;
};

RayleighRuntimeView global_rayleigh_runtime_view(const SimulationStateView& execution_state)
{
    return {tmv::require_level_runtime(execution_state.rayleigh_runtime, "rayleigh_runtime").upper_initialized,
            tmv::require_level_runtime(execution_state.rayleigh_runtime, "rayleigh_runtime").lateral_initialized,
            tmv::require_level_runtime(execution_state.rayleigh_runtime, "rayleigh_runtime").alpha,
            tmv::require_level_runtime(execution_state.rayleigh_runtime, "rayleigh_runtime").u_ref,
            tmv::require_level_runtime(execution_state.rayleigh_runtime, "rayleigh_runtime").v_ref,
            tmv::require_level_runtime(execution_state.rayleigh_runtime, "rayleigh_runtime").upper_w_alpha,
            tmv::require_level_runtime(execution_state.rayleigh_runtime, "rayleigh_runtime").w_only_alpha,
            tmv::require_level_runtime(execution_state.rayleigh_runtime, "rayleigh_runtime").lateral_alpha};
}

RayleighRuntimeView rayleigh_runtime_view(RayleighDampingRuntimeState& runtime)
{
    return {runtime.upper_initialized, runtime.lateral_initialized,
            runtime.alpha, runtime.u_ref, runtime.v_ref,
            runtime.upper_w_alpha,
            runtime.w_only_alpha, runtime.lateral_alpha};
}

RayleighRuntimeView rayleigh_runtime_for(const SimulationStateView& state)
{
    return state.rayleigh_runtime
        ? rayleigh_runtime_view(*state.rayleigh_runtime)
        : global_rayleigh_runtime_view(state);
}

bool field_matches_domain(const SimulationStateView& state,
                          const Field3D& field)
{
    return field.size_r() == state.grid.nr
        && field.size_th() == state.grid.nth
        && field.size_z() == state.grid.nz;
}

bool complete_sponge_energy_state(const SimulationStateView& state)
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

tmv::init::config::DampingMode damping_mode_for(
    const SimulationStateView& state) noexcept
{
    return state.configuration
        ? state.configuration->damping_mode
        : tmv::require_run_configuration(state).damping_mode;
}

double damping_relaxation_factor(
    const SimulationStateView& state, double alpha, double dt)
{
    if (!std::isfinite(alpha) || alpha < 0.0
        || !std::isfinite(dt) || dt <= 0.0)
        throw std::invalid_argument(
            "Rayleigh relaxation requires finite non-negative alpha and "
            "positive finite dt");
    const double scaled = alpha * dt;
    if (!std::isfinite(scaled))
        throw std::invalid_argument(
            "Rayleigh relaxation coefficient overflowed");
    if (damping_mode_for(state)
        == tmv::init::config::DampingMode::Implicit)
        return scaled / (1.0 + scaled);
    // Preserve the established exact-decay writer for explicit mode. This is
    // the historical Nimbus equation and remains byte-compatible.
    return -std::expm1(-scaled);
}

double sponge_total_water(const SimulationStateView& state,
                          int i, int j, int k)
{
    const auto& f = state.fields;
    const double water[] = {
        static_cast<double>(f.qv[i][j][k]),
        static_cast<double>(f.qc[i][j][k]),
        static_cast<double>(f.qr[i][j][k]),
        static_cast<double>(f.qi[i][j][k]),
        static_cast<double>(f.qs[i][j][k]),
        static_cast<double>(f.qg[i][j][k]),
        static_cast<double>(f.qh[i][j][k]),
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

double sponge_cell_carrier_mass(
    const SimulationStateView& state,
    const tmv::physical_grid_metrics::ControlVolumeMassView& geometry,
    int i, int j, int k)
{
    const double total_water = sponge_total_water(state, i, j, k);
    const double mass =
        geometry.scalar_mass_kg(state.fields.rho, i, j, k)
        * (1.0 + total_water);
    return std::isfinite(mass) && mass > 0.0
        ? mass : std::numeric_limits<double>::quiet_NaN();
}

std::array<double, 3> sponge_velocity_masses(
    const SimulationStateView& state,
    const tmv::physical_grid_metrics::ControlVolumeMassView& geometry,
    const PhysicalCellDomain& domain,
    int i, int j, int k)
{
    std::array<double, 3> result{};
    if (!state.grid.geometry.staggered)
    {
        if (i < domain.i_begin || i >= domain.i_end
            || j < domain.j_begin || j >= domain.j_end
            || k < domain.k_begin || k >= domain.k_end)
            return result;
        const double mass =
            sponge_cell_carrier_mass(state, geometry, i, j, k);
        result.fill(mass);
        return result;
    }

    const bool periodic_x =
        tmv::require_run_configuration(state).lateral_bc_west == tmv::init::config::LateralBC::Periodic
        || tmv::require_run_configuration(state).lateral_bc_east == tmv::init::config::LateralBC::Periodic;
    const bool periodic_y =
        tmv::require_run_configuration(state).lateral_bc_south == tmv::init::config::LateralBC::Periodic
        || tmv::require_run_configuration(state).lateral_bc_north == tmv::init::config::LateralBC::Periodic;
    const bool physical_i =
        i >= domain.i_begin && i < domain.i_end;
    const bool physical_j =
        j >= domain.j_begin && j < domain.j_end;
    const bool physical_k =
        k >= domain.k_begin && k < domain.k_end;
    const double center_mass =
        physical_i && physical_j && physical_k
            ? sponge_cell_carrier_mass(state, geometry, i, j, k)
            : 0.0;

    if (physical_j && physical_k
        && i >= (periodic_x ? domain.i_begin : domain.i_begin - 1)
        && i < domain.i_end)
    {
        if (physical_i)
            result[0] += 0.5 * center_mass;
        if (i + 1 < domain.i_end)
            result[0] += 0.5 * sponge_cell_carrier_mass(
                state, geometry, i + 1, j, k);
        else if (periodic_x)
            result[0] += 0.5 * sponge_cell_carrier_mass(
                state, geometry, domain.i_begin, j, k);
    }
    if (physical_i && physical_k
        && j >= (periodic_y ? domain.j_begin : domain.j_begin - 1)
        && j < domain.j_end)
    {
        if (physical_j)
            result[1] += 0.5 * center_mass;
        if (j + 1 < domain.j_end)
            result[1] += 0.5 * sponge_cell_carrier_mass(
                state, geometry, i, j + 1, k);
        else if (periodic_y)
            result[1] += 0.5 * sponge_cell_carrier_mass(
                state, geometry, i, domain.j_begin, k);
    }
    if (physical_i && physical_j && physical_k)
    {
        result[2] = 0.5 * center_mass;
        if (k + 1 < domain.k_end)
            result[2] += 0.5 * sponge_cell_carrier_mass(
                state, geometry, i, j, k + 1);
    }
    return result;
}

void attribute_momentum_change(
    std::array<double, 7>& partial,
    double mass, double old_velocity,
    double requested_delta, double stored_velocity)
{
    if (mass == 0.0)
        return;
    const double values[] = {
        mass, old_velocity, requested_delta, stored_velocity,
    };
    if (mass < 0.0 || std::any_of(
            std::begin(values), std::end(values),
            [](double value) { return !std::isfinite(value); }))
    {
        partial[5] += 1.0;
        return;
    }
    partial[0] += mass * old_velocity * requested_delta;
    partial[1] += 0.5 * mass * requested_delta * requested_delta;
    partial[2] += 0.5 * mass
        * (stored_velocity * stored_velocity
           - old_velocity * old_velocity);
}

void attribute_theta_change(
    const SimulationStateView& state,
    const tmv::physical_grid_metrics::ControlVolumeMassView& geometry,
    const PhysicalCellDomain& domain,
    std::array<double, 7>& partial,
    int i, int j, int k,
    double theta_old, double theta_stored)
{
    if (i < domain.i_begin || i >= domain.i_end
        || j < domain.j_begin || j >= domain.j_end
        || k < domain.k_begin || k >= domain.k_end)
        return;

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

    const auto& f = state.fields;
    const double pressure = static_cast<double>(f.p[i][j][k]);
    const double qv = static_cast<double>(f.qv[i][j][k]);
    const double liquid = static_cast<double>(f.qc[i][j][k])
        + static_cast<double>(f.qr[i][j][k]);
    const double ice = static_cast<double>(f.qi[i][j][k])
        + static_cast<double>(f.qs[i][j][k])
        + static_cast<double>(f.qg[i][j][k])
        + static_cast<double>(f.qh[i][j][k]);
    const double dry_mass = geometry.scalar_mass_kg(f.rho, i, j, k);
    const double exner = std::pow(
        pressure / physical_constants::reference_pressure_pa, kappa);
    const double temperature_change =
        (theta_stored - theta_old) * exner;
    const double values[] = {
        pressure, qv, liquid, ice, dry_mass, exner,
        theta_old, theta_stored, temperature_change,
    };
    if (pressure <= 0.0 || qv < 0.0 || liquid < 0.0 || ice < 0.0
        || dry_mass <= 0.0
        || std::any_of(
            std::begin(values), std::end(values),
            [](double value) { return !std::isfinite(value); }))
    {
        partial[5] += 1.0;
        return;
    }
    partial[3] += dry_mass * cv_d * temperature_change;
    partial[4] += dry_mass * temperature_change
        * (qv * cv_v + liquid * c_l + ice * c_i);
}

tmv::SpongePassEnergyReportV1 finish_sponge_pass(
    const std::array<double, 7>& sums,
    bool attribute, bool applied)
{
    tmv::SpongePassEnergyReportV1 report;
    report.applied = applied;
    if (!attribute)
        return report;
    if (sums[5] != 0.0)
        throw std::runtime_error(
            "sponge energy attribution rejected invalid physical state");
    report.momentum_requested_linear_work_j = sums[0];
    report.momentum_finite_step_correction_j = sums[1];
    report.carrier_kinetic_change_j = sums[2];
    report.momentum_storage_rounding_j =
        sums[2] - sums[0] - sums[1];
    report.dry_internal_change_j = sums[3];
    report.water_sensible_change_j = sums[4];
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

// Smell #8 closure: w-only upper sponge gate. Read every call so YAML /
// runtime_config flips between reset_rayleigh_damping() calls are respected;
// tests rely on the every-call read of `global_w_only_top_sponge`.
// modularity cleanup 6a removed the NIMBUS_W_ONLY_TOP_SPONGE override; the config
// field damping.w_only_top_sponge is now the single source of truth.
bool w_only_top_sponge_enabled(const SimulationStateView& execution_state)
{
    return tmv::require_run_configuration(execution_state).w_only_top_sponge;
}

void ensure_initialized(const SimulationStateView& state)
{
    RayleighRuntimeView runtime = rayleigh_runtime_for(state);
    const auto& grid = state.grid;
    if (runtime.upper_initialized &&
        static_cast<int>(runtime.alpha.size()) == grid.nz &&
        static_cast<int>(runtime.upper_w_alpha.size()) == grid.nz &&
        static_cast<int>(runtime.w_only_alpha.size()) == grid.nz)
    {
        return;
    }

    runtime.alpha.assign(static_cast<size_t>(grid.nz), 0.0);
    runtime.u_ref.assign(static_cast<size_t>(grid.nz), 0.0);
    runtime.v_ref.assign(static_cast<size_t>(grid.nz), 0.0);
    runtime.upper_w_alpha.assign(static_cast<size_t>(grid.nz), 0.0);
    runtime.w_only_alpha.assign(static_cast<size_t>(grid.nz), 0.0);

    const double z_top = grid.geometry.z_faces.empty()
        ? static_cast<double>(grid.nz) * grid.dz
        : grid.geometry.z_faces.back();

    // C3: z_d sourced from global_damping_z_d. Sentinel <= 0 means "use
    // the legacy depth_fraction=0.3 default" (preserves v0 behavior).
    double z_damp = tmv::require_run_configuration(state).damping_z_d;
    if (z_damp <= 0.0)
    {
        z_damp = z_top * (1.0 - g_rayleigh_config.depth_fraction);
    }
    const double sponge_depth = z_top - z_damp;

    const double alpha_max = tmv::require_run_configuration(state).damping_alpha;
    const double pi_half   = 3.14159265358979323846 * 0.5;
    const auto alpha_at = [&](double height)
    {
        if (sponge_depth <= 0.0 || height <= z_damp)
            return 0.0;
        const double frac = std::min(
            1.0, (height - z_damp) / sponge_depth);
        const double s = std::sin(pi_half * frac);
        return alpha_max * s * s;
    };

    for (int k = 0; k < grid.nz; ++k)
    {
        const double z = static_cast<std::size_t>(k) < grid.geometry.z.size()
            ? grid.geometry.z[static_cast<std::size_t>(k)]
            : tmv::grid_coordinates::scalar_center(k, grid.dz);

        // Reference wind from the environmental hodograph
        compute_wind_profile(tmv::require_run_configuration(state).wind_profile, z,
                             runtime.u_ref[static_cast<size_t>(k)],
                             runtime.v_ref[static_cast<size_t>(k)]);

        runtime.alpha[static_cast<size_t>(k)] = alpha_at(z);
        const double z_w =
            grid.geometry.staggered
                ? (grid.geometry.z_faces.empty()
                    ? static_cast<double>(k + 1) * grid.dz
                    : grid.geometry.z_faces[static_cast<std::size_t>(k + 1)])
                : z;
        runtime.upper_w_alpha[static_cast<size_t>(k)] = alpha_at(z_w);
    }

    // W-only sponge profile. Indexed by distance from the lid in grid points:
    // d = 0 at k = NZ-1, d = N-1 at k = NZ-N. Sin^2 ramp from 0 (sponge
    // inner edge) to w_only_alpha_max (at the lid). Profile values left at
    // 0.0 outside the top w_only_depth_cells. Independent of `sponge_depth`
    // so the w-only path is usable even when the legacy z-fraction sponge
    // is disabled. NZ < 2 is a degenerate domain — skip safely.
    const int w_only_depth = std::min(std::max(tmv::require_run_configuration(state).w_only_depth_cells, 0), grid.nz);
    const double w_only_alpha_max = std::max(tmv::require_run_configuration(state).w_only_alpha_max, 0.0);
    if (w_only_depth > 0 && w_only_alpha_max > 0.0)
    {
        for (int d = 0; d < w_only_depth; ++d)
        {
            const int    k    = grid.nz - 1 - d;
            const double frac = (w_only_depth > 1)
                ? 1.0 - static_cast<double>(d) / static_cast<double>(w_only_depth - 1)
                : 1.0;
            const double s = std::sin(pi_half * frac);
            runtime.w_only_alpha[static_cast<size_t>(k)] = w_only_alpha_max * s * s;
        }
    }

    // Profile construction is also used to pack the fused backend input when
    // upper damping is disabled. Do not report an inactive precomputed
    // profile as applied physics.
    if (tmv::require_run_configuration(state).damping_enabled && sponge_depth > 0.0 && alpha_max > 0.0)
    {
        const int first_damped = static_cast<int>(std::ceil(z_damp / grid.dz));
        tmv::log_info("[RAYLEIGH] Sponge layer active: z_damp=", z_damp,
                      "m (k>=", first_damped, "), alpha_max=", alpha_max,
                      " 1/s, depth=", sponge_depth, "m");
    }
    if (tmv::require_run_configuration(state).damping_enabled && w_only_top_sponge_enabled(state) &&
        w_only_depth > 0 && w_only_alpha_max > 0.0)
    {
        tmv::log_info("[RAYLEIGH] W-only top sponge active: depth_cells=",
                      w_only_depth, " (k>=", grid.nz - w_only_depth,
                      "), alpha_max=", w_only_alpha_max, " 1/s");
    }

    runtime.upper_initialized = true;
}

} // namespace

tmv::SpongeEnergyReportV1 attribute_state_relaxation_cartesian_device_commit(
    const SimulationStateView& state,
    float dt_step,
    const float* profiles,
    int profiles_len,
    int lateral_width,
    bool do_upper,
    bool do_w_only,
    bool do_lateral,
    const float* u_committed,
    const float* v_committed,
    const float* w_committed,
    const float* theta_committed)
{
    const auto& grid = state.grid;
    const auto& fields = state.fields;
    if (!complete_sponge_energy_state(state)
        || tmv::physical_grid_metrics::active_terrain_metrics(state) != nullptr
        || !std::isfinite(dt_step) || dt_step <= 0.0f
        || profiles == nullptr || profiles_len < 7 * grid.nz
        || lateral_width < 0 || lateral_width > grid.nz
        || u_committed == nullptr || v_committed == nullptr
        || w_committed == nullptr || theta_committed == nullptr)
    {
        throw std::invalid_argument(
            "device sponge attribution requires a complete flat Cartesian "
            "state, shader profile block, and committed outputs");
    }
    tmv::physical_grid_metrics::validate(state);
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(
        state);
    const auto index = [&](int i, int j, int k)
    {
        return (static_cast<std::size_t>(i)
                    * static_cast<std::size_t>(grid.nth)
                + static_cast<std::size_t>(j))
                * static_cast<std::size_t>(grid.nz)
            + static_cast<std::size_t>(k);
    };
    const float dt_safe = std::max(dt_step, 1.0e-12f);
    const auto factor = [&](float alpha)
    {
        return 1.0f - std::exp(-alpha * dt_safe);
    };

    const auto sums = tmv::diagnostics::fixed_slab_reduce<21>(
        0, grid.nr,
        [&](int i, std::array<double, 21>& partial)
        {
            std::array<double, 7> upper{};
            std::array<double, 7> w_only{};
            std::array<double, 7> lateral{};
            for (int j = 0; j < grid.nth; ++j)
                for (int k = 0; k < grid.nz; ++k)
                {
                    const std::size_t c = index(i, j, k);
                    const float u0 = fields.u.data()[c];
                    const float v0 = fields.v.data()[c];
                    const float w0 = fields.w.data()[c];
                    const float theta0_old = fields.theta.data()[c];
                    const float u_ref = profiles[2 * grid.nz + k];
                    const float v_ref = profiles[3 * grid.nz + k];
                    const float theta_ref = profiles[4 * grid.nz + k];

                    const int distance = std::min(
                        std::min(i, grid.nr - 1 - i),
                        std::min(j, grid.nth - 1 - j));
                    const bool lateral_active =
                        do_lateral && lateral_width > 0
                        && distance < lateral_width
                        && profiles[5 * grid.nz + distance] > 0.0f;

                    float u_upper = u0;
                    float v_upper = v0;
                    float w_upper = w0;
                    float theta_upper = theta0_old;
                    const float alpha =
                        do_upper ? profiles[k] : 0.0f;
                    const float alpha_upper_w =
                        do_upper ? profiles[6 * grid.nz + k] : 0.0f;
                    const bool upper_horizontal_active = alpha > 0.0f;
                    const bool upper_w_active = alpha_upper_w > 0.0f;
                    if (upper_horizontal_active || upper_w_active)
                    {
                        const float f = factor(alpha);
                        const float fw = factor(alpha_upper_w);
                        const float du = f * (u_ref - u_upper);
                        const float dv = f * (v_ref - v_upper);
                        const float dw = fw * (0.0f - w_upper);
                        const float dtheta =
                            f * (theta_ref - theta_upper);
                        u_upper += du;
                        v_upper += dv;
                        w_upper += dw;
                        theta_upper += dtheta;
                        const auto masses = sponge_velocity_masses(
                            state, mass_geometry, domain, i, j, k);
                        if (upper_horizontal_active)
                        {
                            attribute_momentum_change(
                                upper, masses[0], u0, du,
                                lateral_active
                                    ? u_upper
                                    : static_cast<double>(
                                        u_committed[c]));
                            attribute_momentum_change(
                                upper, masses[1], v0, dv,
                                lateral_active
                                    ? v_upper
                                    : static_cast<double>(
                                        v_committed[c]));
                            attribute_theta_change(
                                state, mass_geometry, domain, upper,
                                i, j, k, theta0_old,
                                lateral_active
                                    ? theta_upper
                                    : theta_committed[c]);
                        }
                        if (upper_w_active)
                        {
                            const bool later_w_pass =
                                (do_w_only
                                 && profiles[grid.nz + k] > 0.0f)
                                || lateral_active;
                            attribute_momentum_change(
                                upper, masses[2], w0, dw,
                                later_w_pass
                                    ? w_upper
                                    : static_cast<double>(
                                        w_committed[c]));
                        }
                        upper[6] += 1.0;
                    }

                    float w_after_w_only = w_upper;
                    const float alpha_w_only =
                        do_w_only ? profiles[grid.nz + k] : 0.0f;
                    if (alpha_w_only > 0.0f)
                    {
                        const float dw =
                            factor(alpha_w_only)
                            * (0.0f - w_after_w_only);
                        w_after_w_only += dw;
                        const auto masses = sponge_velocity_masses(
                            state, mass_geometry, domain, i, j, k);
                        attribute_momentum_change(
                            w_only, masses[2], w_upper, dw,
                            lateral_active
                                ? w_after_w_only
                                : static_cast<double>(w_committed[c]));
                        w_only[6] += 1.0;
                    }

                    if (lateral_active)
                    {
                        const float f = factor(
                            profiles[5 * grid.nz + distance]);
                        const float du = f * (u_ref - u_upper);
                        const float dv = f * (v_ref - v_upper);
                        const float dw =
                            f * (0.0f - w_after_w_only);
                        const auto masses = sponge_velocity_masses(
                            state, mass_geometry, domain, i, j, k);
                        attribute_momentum_change(
                            lateral, masses[0], u_upper, du,
                            u_committed[c]);
                        attribute_momentum_change(
                            lateral, masses[1], v_upper, dv,
                            v_committed[c]);
                        attribute_momentum_change(
                            lateral, masses[2], w_after_w_only, dw,
                            w_committed[c]);
                        attribute_theta_change(
                            state, mass_geometry, domain, lateral,
                            i, j, k, theta_upper,
                            theta_committed[c]);
                        lateral[6] += 1.0;
                    }
                }
            for (std::size_t n = 0; n < 7; ++n)
            {
                partial[n] += upper[n];
                partial[7 + n] += w_only[n];
                partial[14 + n] += lateral[n];
            }
        });

    std::array<double, 7> upper{};
    std::array<double, 7> w_only{};
    std::array<double, 7> lateral{};
    for (std::size_t n = 0; n < 7; ++n)
    {
        upper[n] = sums[n];
        w_only[n] = sums[7 + n];
        lateral[n] = sums[14 + n];
    }
    tmv::SpongeEnergyReportV1 report;
    report.upper = finish_sponge_pass(
        upper, true, upper[6] > 0.0);
    report.w_only = finish_sponge_pass(
        w_only, true, w_only[6] > 0.0);
    report.lateral = finish_sponge_pass(
        lateral, true, lateral[6] > 0.0);
    report.complete = report.upper.complete
        && report.w_only.complete && report.lateral.complete;
    return report;
}

tmv::SpongeEnergyReportV1 apply_rayleigh_damping(
    const SimulationStateView& state, double dt_damp)
{
    tmv::SpongeEnergyReportV1 report;
    const bool attribute = complete_sponge_energy_state(state);
    const auto complete_noop = [&]()
    {
        report.upper.complete = attribute;
        report.w_only.complete = attribute;
        report.lateral.complete = attribute;
        report.complete = attribute;
        return report;
    };

    // C3: hard short-circuit when YAML disables the sponge.
    if (!tmv::require_run_configuration(state).damping_enabled)
    {
        return complete_noop();
    }
    if (!std::isfinite(dt_damp) || dt_damp <= 0.0)
        throw std::invalid_argument(
            "Rayleigh damping timestep must be positive and finite");

    ensure_initialized(state);
    RayleighRuntimeView runtime = rayleigh_runtime_for(state);

    const auto& grid = state.grid;
    auto& fields = state.fields;
    const auto& base = state.base;
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::active_terrain_metrics(state);

    if (grid.nr <= 0 || grid.nth <= 0 || grid.nz <= 0)
    {
        return complete_noop();
    }
    if (static_cast<int>(runtime.alpha.size()) != grid.nz ||
        static_cast<int>(runtime.upper_w_alpha.size()) != grid.nz ||
        static_cast<int>(runtime.w_only_alpha.size()) != grid.nz)
    {
        return complete_noop();
    }
    if (attribute)
        tmv::physical_grid_metrics::validate(state);
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(
        state);

    double terrain_z_top = 0.0;
    double terrain_z_damp = 0.0;
    double terrain_sponge_depth = 0.0;
    if (terrain_metrics != nullptr)
    {
        tmv::physical_grid_metrics::validate(state);
        if (base.theta0.size() != static_cast<std::size_t>(grid.nz))
            throw std::invalid_argument(
                "terrain Rayleigh damping requires a complete theta reference profile");
        double min_top = std::numeric_limits<double>::infinity();
        double max_top = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < grid.nr; ++i)
            for (int j = 0; j < grid.nth; ++j)
            {
                const double top = terrain_metrics->z_face(i, j, grid.nz);
                min_top = std::min(min_top, top);
                max_top = std::max(max_top, top);
            }
        const double top_tolerance = 1.0e-10 * std::max(1.0, std::abs(max_top));
        if (!std::isfinite(min_top) || !std::isfinite(max_top) ||
            max_top - min_top > top_tolerance)
            throw std::invalid_argument(
                "terrain Rayleigh damping requires a flat physical model top");
        terrain_z_top = 0.5 * (min_top + max_top);
        terrain_z_damp = tmv::require_run_configuration(state).damping_z_d > 0.0
            ? tmv::require_run_configuration(state).damping_z_d
            : terrain_z_top * (1.0 - g_rayleigh_config.depth_fraction);
        terrain_sponge_depth = terrain_z_top - terrain_z_damp;
    }

    const double dt_safe = dt_damp;
    constexpr double pi_half = 3.14159265358979323846 * 0.5;
    const auto terrain_alpha_at = [&](double physical_z)
    {
        if (terrain_sponge_depth <= 0.0 || physical_z <= terrain_z_damp)
            return 0.0;
        const double fraction = std::min(
            1.0, (physical_z - terrain_z_damp) / terrain_sponge_depth);
        const double sine = std::sin(pi_half * fraction);
        return tmv::require_run_configuration(state).damping_alpha * sine * sine;
    };
    // A mapped C-grid's lower w face is diagnosed from the horizontal
    // velocities. Attribute its exact upper-sponge change before those
    // velocities are mutated. This is a surface-only reduction.
    std::array<double, 7> upper_ground{};
    if (attribute && terrain_metrics != nullptr
        && grid.geometry.staggered)
    {
        if (terrain_metrics->Gx_face.size_r() != grid.nr
            || terrain_metrics->Gx_face.size_th() != grid.nth
            || terrain_metrics->Gx_face.size_z() != grid.nz + 1
            || terrain_metrics->Gy_face.size_r() != grid.nr
            || terrain_metrics->Gy_face.size_th() != grid.nth
            || terrain_metrics->Gy_face.size_z() != grid.nz + 1)
            throw std::invalid_argument(
                "terrain sponge energy attribution requires complete "
                "terrain face metrics");
        upper_ground = tmv::diagnostics::fixed_slab_reduce<7>(
            domain.i_begin, domain.i_end,
            [&](int i, std::array<double, 7>& partial)
            {
                const int k = domain.k_begin;
                const auto predicted_u = [&](int ii, int jj)
                {
                    const int east = std::min(ii + 1, grid.nr - 1);
                    const double z_scalar =
                        terrain_metrics->z(ii, jj, k);
                    const double z_u = 0.5 * (
                        z_scalar + terrain_metrics->z(east, jj, k));
                    const double factor = damping_relaxation_factor(
                        state, terrain_alpha_at(z_u), dt_safe);
                    double u_ref = 0.0;
                    double ignored = 0.0;
                    compute_wind_profile(
                        tmv::require_run_configuration(state).wind_profile, z_u, u_ref, ignored);
                    const float old = fields.u[ii][jj][k];
                    const double delta =
                        factor * (u_ref - static_cast<double>(old));
                    const float stored =
                        old + static_cast<float>(delta);
                    return std::array<double, 3>{
                        static_cast<double>(old), delta,
                        static_cast<double>(stored)};
                };
                const auto predicted_v = [&](int ii, int jj)
                {
                    const int north = std::min(jj + 1, grid.nth - 1);
                    const double z_scalar =
                        terrain_metrics->z(ii, jj, k);
                    const double z_v = 0.5 * (
                        z_scalar + terrain_metrics->z(ii, north, k));
                    const double factor = damping_relaxation_factor(
                        state, terrain_alpha_at(z_v), dt_safe);
                    double ignored = 0.0;
                    double v_ref = 0.0;
                    compute_wind_profile(
                        tmv::require_run_configuration(state).wind_profile, z_v, ignored, v_ref);
                    const float old = fields.v[ii][jj][k];
                    const double delta =
                        factor * (v_ref - static_cast<double>(old));
                    const float stored =
                        old + static_cast<float>(delta);
                    return std::array<double, 3>{
                        static_cast<double>(old), delta,
                        static_cast<double>(stored)};
                };
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    const auto u_w = predicted_u(i - 1, j);
                    const auto u_e = predicted_u(i, j);
                    const auto v_s = predicted_v(i, j - 1);
                    const auto v_n = predicted_v(i, j);
                    const double slope_x =
                        -terrain_metrics->Gx_face(i, j, k);
                    const double slope_y =
                        -terrain_metrics->Gy_face(i, j, k);
                    const double frame_u = state.domain_frame != nullptr
                        ? state.domain_frame->velocity_ms.u_ms : 0.0;
                    const double frame_v = state.domain_frame != nullptr
                        ? state.domain_frame->velocity_ms.v_ms : 0.0;
                    const double old_ground =
                        (0.5 * (u_w[0] + u_e[0]) - frame_u) * slope_x
                        + (0.5 * (v_s[0] + v_n[0]) - frame_v) * slope_y;
                    const double delta_ground =
                        0.5 * (u_w[1] + u_e[1]) * slope_x
                        + 0.5 * (v_s[1] + v_n[1]) * slope_y;
                    const double stored_ground =
                        (0.5 * (u_w[2] + u_e[2]) - frame_u) * slope_x
                        + (0.5 * (v_s[2] + v_n[2]) - frame_v) * slope_y;
                    attribute_momentum_change(
                        partial,
                        0.5 * sponge_cell_carrier_mass(
                            state, mass_geometry, i, j, k),
                        old_ground, delta_ground, stored_ground);
                }
            });
    }

    auto upper_sums = tmv::diagnostics::fixed_slab_reduce<7>(
        0, grid.nr,
        [&](int i, std::array<double, 7>& partial)
        {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                if (terrain_metrics != nullptr)
                {
                    const int east = std::min(i + 1, grid.nr - 1);
                    const int north = std::min(j + 1, grid.nth - 1);
                    const double z_scalar = terrain_metrics->z(i, j, k);
                    const double z_u = 0.5 *
                        (z_scalar + terrain_metrics->z(east, j, k));
                    const double z_v = 0.5 *
                        (z_scalar + terrain_metrics->z(i, north, k));
                    const double z_w = terrain_metrics->z_face(i, j, k + 1);
                    const double alpha_u = terrain_alpha_at(z_u);
                    const double alpha_v = terrain_alpha_at(z_v);
                    const double alpha_w = terrain_alpha_at(z_w);
                    const double alpha_theta = terrain_alpha_at(z_scalar);

                    double u_ref = 0.0, ignored_v = 0.0;
                    double ignored_u = 0.0, v_ref = 0.0;
                    compute_wind_profile(tmv::require_run_configuration(state).wind_profile, z_u,
                                         u_ref, ignored_v);
                    compute_wind_profile(tmv::require_run_configuration(state).wind_profile, z_v,
                                         ignored_u, v_ref);
                    const double theta_ref =
                        terrain_transport::reference_profile_value_fp32(
                            base.theta0, z_scalar, grid.dz);
                    const double factor_u = damping_relaxation_factor(
                        state, alpha_u, dt_safe);
                    const double factor_v = damping_relaxation_factor(
                        state, alpha_v, dt_safe);
                    const double factor_w = damping_relaxation_factor(
                        state, alpha_w, dt_safe);
                    const double factor_theta = damping_relaxation_factor(
                        state, alpha_theta, dt_safe);
                    const double u_old = fields.u[i][j][k];
                    const double v_old = fields.v[i][j][k];
                    const double w_old = fields.w[i][j][k];
                    const double theta_old = fields.theta[i][j][k];
                    const double delta_u = factor_u * (u_ref - u_old);
                    const double delta_v = factor_v * (v_ref - v_old);
                    const double delta_w = factor_w * (0.0 - w_old);
                    fields.u[i][j][k] += static_cast<float>(factor_u *
                        (u_ref - u_old));
                    fields.v[i][j][k] += static_cast<float>(factor_v *
                        (v_ref - v_old));
                    fields.w[i][j][k] += static_cast<float>(factor_w *
                        (0.0 - w_old));
                    fields.theta[i][j][k] += static_cast<float>(factor_theta *
                        (theta_ref - theta_old));
                    if (attribute)
                    {
                        const auto velocity_masses =
                            sponge_velocity_masses(
                                state, mass_geometry, domain, i, j, k);
                        attribute_momentum_change(
                            partial, velocity_masses[0],
                            u_old, delta_u, fields.u[i][j][k]);
                        attribute_momentum_change(
                            partial, velocity_masses[1],
                            v_old, delta_v, fields.v[i][j][k]);
                        attribute_momentum_change(
                            partial, velocity_masses[2],
                            w_old, delta_w, fields.w[i][j][k]);
                        attribute_theta_change(
                            state, mass_geometry, domain, partial, i, j, k,
                            theta_old, fields.theta[i][j][k]);
                    }
                    if (factor_u > 0.0 || factor_v > 0.0
                        || factor_w > 0.0 || factor_theta > 0.0)
                        partial[6] += 1.0;
                    continue;
                }

                const double alpha = runtime.alpha[static_cast<size_t>(k)];
                const double alpha_w =
                    runtime.upper_w_alpha[static_cast<std::size_t>(k)];
                if (alpha <= 0.0 && alpha_w <= 0.0)
                {
                    continue;
                }

                const double factor = damping_relaxation_factor(
                    state, alpha, dt_safe);
                const double factor_w =
                    grid.geometry.staggered
                        ? damping_relaxation_factor(
                            state, alpha_w, dt_safe)
                        : factor;

                const double u_ref = runtime.u_ref[static_cast<size_t>(k)];
                const double v_ref = runtime.v_ref[static_cast<size_t>(k)];
                const double w_ref = 0.0;
                // Relax theta toward the per-level base-state profile, not the
                // scalar reference theta0. Using theta0 was a latent bug: at any
                // height where the IC theta differs from 300 K (e.g. the upper
                // troposphere ~370 K), the sponge dragged theta toward 300 K
                // every step, producing a cold anomaly that drove negative
                // buoyancy, vertical motion, and ultimately a runaway 2dz mode
                // that saturated against the [200,700] theta clamps.
                const double theta_ref = (static_cast<size_t>(k) < base.theta0.size())
                                             ? base.theta0[static_cast<size_t>(k)]
                                             : theta0;
                const double u_old = fields.u[i][j][k];
                const double v_old = fields.v[i][j][k];
                const double w_old = fields.w[i][j][k];
                const double theta_old = fields.theta[i][j][k];
                const double delta_u = factor * (u_ref - u_old);
                const double delta_v = factor * (v_ref - v_old);
                const double delta_w = factor_w * (w_ref - w_old);

                fields.u[i][j][k] += static_cast<float>(
                    delta_u);
                fields.v[i][j][k] += static_cast<float>(
                    delta_v);
                fields.w[i][j][k] += static_cast<float>(
                    delta_w);

                // Relax theta only among the thermodynamic fields
                // (SCIENCE.md C6): relaxing rho is an artificial mass
                // source/sink, and on the Cartesian path p is diagnostic
                // from the full EOS anyway - the lateral sponge already
                // excludes both for the same reason. rho evolves by
                // continuity; p follows from the EOS rediagnosis.
                fields.theta[i][j][k] += static_cast<float>(
                    factor * (theta_ref - theta_old));
                if (attribute)
                {
                    const auto velocity_masses =
                        sponge_velocity_masses(
                            state, mass_geometry, domain, i, j, k);
                    attribute_momentum_change(
                        partial, velocity_masses[0],
                        u_old, delta_u, fields.u[i][j][k]);
                    attribute_momentum_change(
                        partial, velocity_masses[1],
                        v_old, delta_v, fields.v[i][j][k]);
                    attribute_momentum_change(
                        partial, velocity_masses[2],
                        w_old, delta_w, fields.w[i][j][k]);
                    attribute_theta_change(
                        state, mass_geometry, domain, partial, i, j, k,
                        theta_old, fields.theta[i][j][k]);
                }
                partial[6] += 1.0;
            }
        }
    });
    for (std::size_t n = 0; n < upper_sums.size(); ++n)
        upper_sums[n] += upper_ground[n];
    report.upper = finish_sponge_pass(
        upper_sums, attribute, upper_sums[6] > 0.0);

    // Smell #8 closure: additive w-only absorber in the top w_only_depth_cells.
    // The legacy all-field sponge above relaxes the full prognostic state
    // toward base across the broad sponge depth (depth_fraction=0.3 by
    // default). The w-only absorber adds a strong, narrow w damping right
    // at the lid -- it targets the upward-propagating acoustic / steady
    // secondary-circulation w-signal driven by the lid-BC O(alpha^3)
    // residual without further damping any horizontal velocity or
    // thermodynamic field. The closure goal is to make the legacy alpha
    // small enough (target: 0.02, non-load-bearing) while the w-only
    // absorber holds the lid stability budget. See
    // docs/CyclostrophicConservativePlan.md §8.
    if (!w_only_top_sponge_enabled(state))
    {
        report.w_only.complete = attribute;
        report.lateral.complete = attribute;
        report.complete = attribute;
        return report;
    }
    const auto w_only_sums = tmv::diagnostics::fixed_slab_reduce<7>(
        0, grid.nr,
        [&](int i, std::array<double, 7>& partial)
        {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                const double alpha_w = runtime.w_only_alpha[static_cast<size_t>(k)];
                if (alpha_w <= 0.0)
                {
                    continue;
                }
                const double factor_w = damping_relaxation_factor(
                    state, alpha_w, dt_safe);
                const double w_old = fields.w[i][j][k];
                const double delta_w = factor_w * (0.0 - w_old);
                fields.w[i][j][k] += static_cast<float>(
                    delta_w);
                if (attribute)
                {
                    const auto velocity_masses =
                        sponge_velocity_masses(
                            state, mass_geometry, domain, i, j, k);
                    attribute_momentum_change(
                        partial, velocity_masses[2],
                        w_old, delta_w, fields.w[i][j][k]);
                }
                partial[6] += 1.0;
            }
        }
    });
    report.w_only = finish_sponge_pass(
        w_only_sums, attribute, w_only_sums[6] > 0.0);
    report.lateral.complete = attribute;
    report.complete = attribute;
    return report;
}

tmv::SpongeEnergyReportV1 apply_rayleigh_damping(double dt_damp)
{
    return apply_rayleigh_damping(global_simulation_state(), dt_damp);
}

void reset_rayleigh_damping()
{
    return reset_rayleigh_damping(global_simulation_state());
}

void reset_rayleigh_damping(const SimulationStateView& execution_state)
{
    tmv::require_level_runtime(execution_state.rayleigh_runtime, "rayleigh_runtime") = RayleighDampingRuntimeState{};
}

void reset_rayleigh_damping(RayleighDampingRuntimeState& runtime)
{
    runtime = RayleighDampingRuntimeState{};
}

// ---------------------------------------------------------------------------
// Lateral Rayleigh sponge (Cartesian grids only)
// ---------------------------------------------------------------------------

namespace
{

void ensure_lateral_initialized(const SimulationStateView& execution_state, RayleighRuntimeView& runtime)
{
    if (runtime.lateral_initialized)
    {
        return;
    }

    // Determine effective sponge width.
    //
    // Applies on all four lateral faces (x/y); the configured width is taken
    // as-is.
    //
    // SCIENCE.md M18: width and strength come from YAML
    // (damping.lateral_width / damping.lateral_alpha); width 0 or
    // alpha 0 disables the band. The struct defaults remain the
    // fallback for code paths that never load a config.
    g_lateral_config.width_points = tmv::require_run_configuration(execution_state).lateral_sponge_width;
    g_lateral_config.alpha_max    = tmv::require_run_configuration(execution_state).lateral_sponge_alpha;
    const int W = (g_lateral_config.alpha_max > 0.0)
        ? g_lateral_config.width_points : 0;
    if (W <= 0)
    {
        runtime.lateral_initialized = true;
        return;
    }

    // Precompute sine-squared damping profile indexed by distance from boundary
    // (in grid points). Index 0 = boundary face, index W-1 = sponge inner edge.
    runtime.lateral_alpha.assign(static_cast<size_t>(W), 0.0);
    const double pi_half = 3.14159265358979323846 * 0.5;
    for (int d = 0; d < W; ++d)
    {
        const double frac = 1.0 - static_cast<double>(d) / static_cast<double>(W);
        const double s = std::sin(pi_half * frac);
        runtime.lateral_alpha[static_cast<size_t>(d)] = g_lateral_config.alpha_max * s * s;
    }

    tmv::log_info("[RAYLEIGH] Lateral sponge active: width=", W,
                  " points (all 4 lateral faces), alpha_max=",
                  g_lateral_config.alpha_max, " 1/s");

    runtime.lateral_initialized = true;
}

} // namespace

tmv::SpongeEnergyReportV1 apply_lateral_damping(
    const SimulationStateView& state, double dt_damp)
{
    tmv::SpongeEnergyReportV1 report;
    const bool attribute = complete_sponge_energy_state(state);
    report.upper.complete = attribute;
    report.w_only.complete = attribute;
    const auto complete_noop = [&]()
    {
        report.lateral.complete = attribute;
        report.complete = attribute;
        return report;
    };

    // Lateral sponge master toggle (modularity cleanup 6a: was the
    // NIMBUS_DISABLE_LATERAL_SPONGE env knob, originally a Smell #7 experiment
    // probing the long-time instability at i ~ 24..25, k = 0..2 on
    // cgrid_hydrostatic + cgrid_tornado_genesis).
    // Parent boundary absorption and child interface relaxation are separate
    // source owners. A nested child already receives its boundary forcing in
    // conserved form, so applying the parent lateral sponge to the same band
    // would double-relax momentum and obscure the interface ledger.
    if (!tmv::require_run_configuration(state).lateral_sponge_enabled ||
        global_simulation_storage.nested_execution.child_execution_active)
    {
        return complete_noop();
    }
    if (!std::isfinite(dt_damp) || dt_damp <= 0.0)
        throw std::invalid_argument(
            "lateral damping timestep must be positive and finite");
    RayleighRuntimeView runtime = rayleigh_runtime_for(state);
    ensure_lateral_initialized(state, runtime);

    if (runtime.lateral_alpha.empty())
    {
        return complete_noop();
    }

    const int W = static_cast<int>(runtime.lateral_alpha.size());
    const double dt_safe = dt_damp;

    // Ensure vertical reference winds are available (shared with upper sponge)
    ensure_initialized(state);

    const auto& grid = state.grid;
    auto& fields = state.fields;
    const auto& base = state.base;
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::active_terrain_metrics(state);
    if (terrain_metrics != nullptr)
    {
        tmv::physical_grid_metrics::validate(state);
        if (base.theta0.size() != static_cast<std::size_t>(grid.nz))
            throw std::invalid_argument(
                "terrain lateral damping requires a complete theta reference profile");
    }
    if (attribute)
        tmv::physical_grid_metrics::validate(state);
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(
        state);
    const auto lateral_factor_at = [&](int i, int j)
    {
        const int dist_i = std::min(i, grid.nr - 1 - i);
        const int dist_j = std::min(j, grid.nth - 1 - j);
        const int dist = std::min(dist_i, dist_j);
        if (dist < 0 || dist >= W)
            return 0.0;
        return damping_relaxation_factor(
            state,
            runtime.lateral_alpha[static_cast<std::size_t>(dist)],
            dt_safe);
    };

    std::array<double, 7> lateral_ground{};
    if (attribute && terrain_metrics != nullptr
        && grid.geometry.staggered)
    {
        if (terrain_metrics->Gx_face.size_r() != grid.nr
            || terrain_metrics->Gx_face.size_th() != grid.nth
            || terrain_metrics->Gx_face.size_z() != grid.nz + 1
            || terrain_metrics->Gy_face.size_r() != grid.nr
            || terrain_metrics->Gy_face.size_th() != grid.nth
            || terrain_metrics->Gy_face.size_z() != grid.nz + 1)
            throw std::invalid_argument(
                "terrain lateral sponge energy attribution requires "
                "complete terrain face metrics");
        lateral_ground = tmv::diagnostics::fixed_slab_reduce<7>(
            domain.i_begin, domain.i_end,
            [&](int i, std::array<double, 7>& partial)
            {
                const int k = domain.k_begin;
                const auto predicted_u = [&](int ii, int jj)
                {
                    const double factor = lateral_factor_at(ii, jj);
                    const int east = std::min(ii + 1, grid.nr - 1);
                    const double z_scalar =
                        terrain_metrics->z(ii, jj, k);
                    const double z_u = 0.5 * (
                        z_scalar + terrain_metrics->z(east, jj, k));
                    double u_ref = 0.0;
                    double ignored = 0.0;
                    compute_wind_profile(
                        tmv::require_run_configuration(state).wind_profile, z_u, u_ref, ignored);
                    const float old = fields.u[ii][jj][k];
                    const double delta =
                        factor * (u_ref - static_cast<double>(old));
                    const float stored =
                        old + static_cast<float>(delta);
                    return std::array<double, 3>{
                        static_cast<double>(old), delta,
                        static_cast<double>(stored)};
                };
                const auto predicted_v = [&](int ii, int jj)
                {
                    const double factor = lateral_factor_at(ii, jj);
                    const int north = std::min(jj + 1, grid.nth - 1);
                    const double z_scalar =
                        terrain_metrics->z(ii, jj, k);
                    const double z_v = 0.5 * (
                        z_scalar + terrain_metrics->z(ii, north, k));
                    double ignored = 0.0;
                    double v_ref = 0.0;
                    compute_wind_profile(
                        tmv::require_run_configuration(state).wind_profile, z_v, ignored, v_ref);
                    const float old = fields.v[ii][jj][k];
                    const double delta =
                        factor * (v_ref - static_cast<double>(old));
                    const float stored =
                        old + static_cast<float>(delta);
                    return std::array<double, 3>{
                        static_cast<double>(old), delta,
                        static_cast<double>(stored)};
                };
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    const auto u_w = predicted_u(i - 1, j);
                    const auto u_e = predicted_u(i, j);
                    const auto v_s = predicted_v(i, j - 1);
                    const auto v_n = predicted_v(i, j);
                    const double slope_x =
                        -terrain_metrics->Gx_face(i, j, k);
                    const double slope_y =
                        -terrain_metrics->Gy_face(i, j, k);
                    const double frame_u = state.domain_frame != nullptr
                        ? state.domain_frame->velocity_ms.u_ms : 0.0;
                    const double frame_v = state.domain_frame != nullptr
                        ? state.domain_frame->velocity_ms.v_ms : 0.0;
                    const double old_ground =
                        (0.5 * (u_w[0] + u_e[0]) - frame_u) * slope_x
                        + (0.5 * (v_s[0] + v_n[0]) - frame_v) * slope_y;
                    const double delta_ground =
                        0.5 * (u_w[1] + u_e[1]) * slope_x
                        + 0.5 * (v_s[1] + v_n[1]) * slope_y;
                    const double stored_ground =
                        (0.5 * (u_w[2] + u_e[2]) - frame_u) * slope_x
                        + (0.5 * (v_s[2] + v_n[2]) - frame_v) * slope_y;
                    attribute_momentum_change(
                        partial,
                        0.5 * sponge_cell_carrier_mass(
                            state, mass_geometry, i, j, k),
                        old_ground, delta_ground, stored_ground);
                }
            });
    }

    auto lateral_sums = tmv::diagnostics::fixed_slab_reduce<7>(
        0, grid.nr,
        [&](int i, std::array<double, 7>& partial)
        {
        for (int j = 0; j < grid.nth; ++j)
        {
            // Distance from the sponge target boundary (in grid points): the
            // nearest of the four lateral faces -- all four reflect.
            const int dist_i = std::min(i, grid.nr - 1 - i);
            const int dist_j = std::min(j, grid.nth - 1 - j);
            const int dist = std::min(dist_i, dist_j);

            if (dist >= W)
            {
                continue;
            }

            const double alpha = runtime.lateral_alpha[static_cast<size_t>(dist)];
            const double factor = damping_relaxation_factor(
                state, alpha, dt_safe);

            for (int k = 0; k < grid.nz; ++k)
            {
                double u_ref = 0.0;
                double v_ref = 0.0;
                double theta_ref = theta0;
                if (terrain_metrics != nullptr)
                {
                    // Sample every reference at the physical location of its
                    // staggered prognostic.  Reusing level-k flat references
                    // over elevated boundary columns would impose spurious
                    // shear and thermal anomalies inside the lateral sponge.
                    const int east = std::min(i + 1, grid.nr - 1);
                    const int north = std::min(j + 1, grid.nth - 1);
                    const double z_scalar = terrain_metrics->z(i, j, k);
                    const double z_u = 0.5 *
                        (z_scalar + terrain_metrics->z(east, j, k));
                    const double z_v = 0.5 *
                        (z_scalar + terrain_metrics->z(i, north, k));
                    double ignored = 0.0;
                    compute_wind_profile(tmv::require_run_configuration(state).wind_profile, z_u,
                                         u_ref, ignored);
                    compute_wind_profile(tmv::require_run_configuration(state).wind_profile, z_v,
                                         ignored, v_ref);
                    theta_ref =
                        terrain_transport::reference_profile_value_fp32(
                            base.theta0, z_scalar, grid.dz);
                }
                else
                {
                    u_ref =
                        (static_cast<size_t>(k) < runtime.u_ref.size())
                            ? runtime.u_ref[static_cast<size_t>(k)] : 0.0;
                    v_ref =
                        (static_cast<size_t>(k) < runtime.v_ref.size())
                            ? runtime.v_ref[static_cast<size_t>(k)] : 0.0;
                    theta_ref =
                        (static_cast<size_t>(k) < base.theta0.size())
                            ? base.theta0[static_cast<size_t>(k)] : theta0;
                }
                // Relax theta toward the per-level base-state profile (same
                // reasoning as apply_rayleigh_damping above): using the scalar
                // theta0 drags the upper troposphere down toward 300 K and
                // creates a cold anomaly.
                const double u_old = fields.u[i][j][k];
                const double v_old = fields.v[i][j][k];
                const double w_old = fields.w[i][j][k];
                const double theta_old = fields.theta[i][j][k];
                const double delta_u = factor * (u_ref - u_old);
                const double delta_v = factor * (v_ref - v_old);
                const double delta_w = factor * (0.0 - w_old);
                fields.u[i][j][k] += static_cast<float>(
                    delta_u);
                fields.v[i][j][k] += static_cast<float>(
                    delta_v);
                fields.w[i][j][k] += static_cast<float>(
                    delta_w);
                fields.theta[i][j][k] += static_cast<float>(
                    factor * (theta_ref - theta_old));
                if (attribute)
                {
                    const auto velocity_masses =
                        sponge_velocity_masses(
                            state, mass_geometry, domain, i, j, k);
                    attribute_momentum_change(
                        partial, velocity_masses[0],
                        u_old, delta_u, fields.u[i][j][k]);
                    attribute_momentum_change(
                        partial, velocity_masses[1],
                        v_old, delta_v, fields.v[i][j][k]);
                    attribute_momentum_change(
                        partial, velocity_masses[2],
                        w_old, delta_w, fields.w[i][j][k]);
                    attribute_theta_change(
                        state, mass_geometry, domain, partial, i, j, k,
                        theta_old, fields.theta[i][j][k]);
                }
                partial[6] += 1.0;

                // Intentionally do NOT sponge rho or p here. The EOS recompute
                // (update_density_from_eos in the dynamics orchestrator) keeps
                // rho consistent with (p, theta), and sponging rho/p in the
                // calm-fixture limit was producing a ~1e-9/step drift in
                // total_mass on cgrid_hydrostatic because the sponge fights
                // EOS roundoff at the boundary. CM1/WRF-style lateral
                // damping operates only on velocities (and optionally theta);
                // this matches standard practice.
            }
        }
    });
    for (std::size_t n = 0; n < lateral_sums.size(); ++n)
        lateral_sums[n] += lateral_ground[n];
    report.lateral = finish_sponge_pass(
        lateral_sums, attribute, lateral_sums[6] > 0.0);
    report.complete = attribute;
    return report;
}

tmv::SpongeEnergyReportV1 apply_lateral_damping(double dt_damp)
{
    return apply_lateral_damping(global_simulation_state(), dt_damp);
}

// ── Profile accessors for the fused GPU state relaxation ──────────────
// (GPU audit Phase 3.5). Each accessor runs the same lazy
// initialization as its apply_* function so the profiles are always
// current with NZ / the runtime config.

const std::vector<double>& rayleigh_alpha_profile(const SimulationStateView& state)
{
    ensure_initialized(state);
    return state.rayleigh_runtime
        ? state.rayleigh_runtime->alpha
        : tmv::require_level_runtime(state.rayleigh_runtime, "rayleigh_runtime").alpha;
}

const std::vector<double>& rayleigh_alpha_profile()
{
    return rayleigh_alpha_profile(global_simulation_state());
}

const std::vector<double>& rayleigh_w_only_alpha_profile(const SimulationStateView& state)
{
    ensure_initialized(state);
    return state.rayleigh_runtime
        ? state.rayleigh_runtime->w_only_alpha
        : tmv::require_level_runtime(state.rayleigh_runtime, "rayleigh_runtime").w_only_alpha;
}

const std::vector<double>& rayleigh_w_only_alpha_profile()
{
    return rayleigh_w_only_alpha_profile(global_simulation_state());
}

const std::vector<double>& rayleigh_upper_w_alpha_profile(
    const SimulationStateView& state)
{
    ensure_initialized(state);
    return state.rayleigh_runtime
        ? state.rayleigh_runtime->upper_w_alpha
        : tmv::require_level_runtime(state.rayleigh_runtime, "rayleigh_runtime").upper_w_alpha;
}

const std::vector<double>& rayleigh_upper_w_alpha_profile()
{
    return rayleigh_upper_w_alpha_profile(global_simulation_state());
}

const std::vector<double>& rayleigh_u_ref_profile(const SimulationStateView& state)
{
    ensure_initialized(state);
    return state.rayleigh_runtime
        ? state.rayleigh_runtime->u_ref
        : tmv::require_level_runtime(state.rayleigh_runtime, "rayleigh_runtime").u_ref;
}

const std::vector<double>& rayleigh_u_ref_profile()
{
    return rayleigh_u_ref_profile(global_simulation_state());
}

const std::vector<double>& rayleigh_v_ref_profile(const SimulationStateView& state)
{
    ensure_initialized(state);
    return state.rayleigh_runtime
        ? state.rayleigh_runtime->v_ref
        : tmv::require_level_runtime(state.rayleigh_runtime, "rayleigh_runtime").v_ref;
}

const std::vector<double>& rayleigh_v_ref_profile()
{
    return rayleigh_v_ref_profile(global_simulation_state());
}

const std::vector<double>& rayleigh_lateral_alpha_profile(const SimulationStateView& state)
{
    RayleighRuntimeView runtime = rayleigh_runtime_for(state);
    ensure_lateral_initialized(state, runtime);
    return state.rayleigh_runtime
        ? state.rayleigh_runtime->lateral_alpha
        : tmv::require_level_runtime(state.rayleigh_runtime, "rayleigh_runtime").lateral_alpha;
}

const std::vector<double>& rayleigh_lateral_alpha_profile()
{
    return rayleigh_lateral_alpha_profile(global_simulation_state());
}

bool rayleigh_upper_sponge_enabled()
{
    return rayleigh_upper_sponge_enabled(global_simulation_state());
}

bool rayleigh_upper_sponge_enabled(const SimulationStateView& execution_state)
{
    return tmv::require_run_configuration(execution_state).damping_enabled;
}

bool rayleigh_w_only_sponge_enabled()
{
    return rayleigh_w_only_sponge_enabled(global_simulation_state());
}

bool rayleigh_w_only_sponge_enabled(const SimulationStateView& execution_state)
{
    return tmv::require_run_configuration(execution_state).damping_enabled && w_only_top_sponge_enabled(execution_state);
}

bool rayleigh_lateral_sponge_enabled()
{
    return rayleigh_lateral_sponge_enabled(global_simulation_state());
}

bool rayleigh_lateral_sponge_enabled(const SimulationStateView& execution_state)
{
    if (!tmv::require_run_configuration(execution_state).lateral_sponge_enabled ||
        global_simulation_storage.nested_execution.child_execution_active)
    {
        return false;
    }
    RayleighRuntimeView runtime = global_rayleigh_runtime_view(execution_state);
    ensure_lateral_initialized(execution_state, runtime);
    return !tmv::require_level_runtime(execution_state.rayleigh_runtime, "rayleigh_runtime").lateral_alpha.empty();
}

void validate_rayleigh_runtime_for_flat_horizontal_recycling(
    const SimulationStateView& state)
{
    if (state.rayleigh_runtime == nullptr || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "Rayleigh D5 validation requires bound Rayleigh and terrain runtimes");
    const auto& live = *state.rayleigh_runtime;
    const auto require_empty = [](const std::vector<double>& values,
                                  const char* name)
    {
        if (!values.empty())
            throw std::invalid_argument(
                std::string("uninitialized Rayleigh ") + name
                + " unexpectedly owns profile values");
    };
    if (!live.upper_initialized)
    {
        require_empty(live.alpha, "alpha");
        require_empty(live.u_ref, "u_ref");
        require_empty(live.v_ref, "v_ref");
        require_empty(live.upper_w_alpha, "upper_w_alpha");
        require_empty(live.w_only_alpha, "w_only_alpha");
    }
    if (!live.lateral_initialized)
        require_empty(live.lateral_alpha, "lateral_alpha");
    if (!live.upper_initialized && !live.lateral_initialized)
        return;
    if (state.terrain_runtime->config.scheme_id != "none")
        throw std::invalid_argument(
            "active Rayleigh terrain recycling requires a column-resolved damping owner");

    RayleighDampingRuntimeState expected;
    SimulationStateView expected_state = state;
    expected_state.rayleigh_runtime = &expected;
    if (live.upper_initialized)
    {
        ensure_initialized(expected_state);
        if (live.alpha != expected.alpha
            || live.u_ref != expected.u_ref
            || live.v_ref != expected.v_ref
            || live.w_only_alpha != expected.w_only_alpha)
            throw std::invalid_argument(
                "initialized Rayleigh upper/reference profile disagrees with the live vertical-grid configuration");
        // Runtime-section checkpoints before the upper-w field was added are
        // intentionally lazy: an empty field forces ensure_initialized() to
        // rebuild all five profiles before use. Any nonempty mismatch is not
        // a valid lazy state.
        if (!live.upper_w_alpha.empty()
            && live.upper_w_alpha != expected.upper_w_alpha)
            throw std::invalid_argument(
                "initialized Rayleigh upper-w profile disagrees with the live vertical-grid configuration");
    }
    if (live.lateral_initialized)
    {
        RayleighRuntimeView expected_view =
            rayleigh_runtime_view(expected);
        ensure_lateral_initialized(state, expected_view);
        if (live.lateral_alpha != expected.lateral_alpha)
            throw std::invalid_argument(
                "initialized Rayleigh lateral profile disagrees with the live damping configuration");
    }
}

const char* rayleigh_horizontal_recycling_algorithm_id() noexcept
{
    return "validated_translation_invariant_rayleigh_profiles_v1";
}

namespace tmv
{

void accumulate_sponge_pass_energy(
    SpongePassEnergyReportV1& destination,
    const SpongePassEnergyReportV1& source)
{
    destination.momentum_requested_linear_work_j +=
        source.momentum_requested_linear_work_j;
    destination.momentum_finite_step_correction_j +=
        source.momentum_finite_step_correction_j;
    destination.dry_internal_change_j += source.dry_internal_change_j;
    destination.water_sensible_change_j +=
        source.water_sensible_change_j;
    destination.phase_reference_change_j +=
        source.phase_reference_change_j;
    destination.carrier_kinetic_change_j +=
        source.carrier_kinetic_change_j;
    destination.gravitational_change_j +=
        source.gravitational_change_j;
    destination.subgrid_tke_change_j +=
        source.subgrid_tke_change_j;
    destination.committed_energy_change_j +=
        source.committed_energy_change_j;
    destination.momentum_storage_rounding_j =
        destination.carrier_kinetic_change_j
        - destination.momentum_requested_linear_work_j
        - destination.momentum_finite_step_correction_j;
    destination.closure_residual_j =
        destination.committed_energy_change_j
        - (destination.dry_internal_change_j
           + destination.water_sensible_change_j
           + destination.phase_reference_change_j
           + destination.carrier_kinetic_change_j
           + destination.gravitational_change_j
           + destination.subgrid_tke_change_j);
    destination.applied = destination.applied || source.applied;
    destination.complete = destination.complete || source.complete;
}

void accumulate_sponge_energy(
    SpongeEnergyReportV1& destination,
    const SpongeEnergyReportV1& source)
{
    accumulate_sponge_pass_energy(destination.upper, source.upper);
    accumulate_sponge_pass_energy(destination.w_only, source.w_only);
    accumulate_sponge_pass_energy(destination.lateral, source.lateral);
    destination.complete = destination.complete || source.complete;
}

} // namespace tmv

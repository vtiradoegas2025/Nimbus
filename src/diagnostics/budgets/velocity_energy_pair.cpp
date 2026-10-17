#include "diagnostics/energy_budget.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
// Preserve the rounding boundary between cell contributions and slab sums.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
std::array<double, 2> cell_kinetic_energies(
    const PrognosticFieldsView& f,
    const tmv::physical_grid_metrics::ControlVolumeMassView& geometry,
    int i, int j, int k)
{
    const double liquid = static_cast<double>(f.qc[i][j][k]) + static_cast<double>(f.qr[i][j][k]);
    const double ice = static_cast<double>(f.qi[i][j][k]) + static_cast<double>(f.qs[i][j][k])
        + static_cast<double>(f.qg[i][j][k]) + static_cast<double>(f.qh[i][j][k]);
    const double water = static_cast<double>(f.qv[i][j][k]) + liquid + ice;
    const double dry_mass = static_cast<double>(f.rho[i][j][k]) * geometry.scalar_volume_m3(i, j, k);
    const double mass = dry_mass * (1.0 + water);
    const double u_w = f.u[i - 1][j][k], u_e = f.u[i][j][k];
    const double v_s = f.v[i][j - 1][k], v_n = f.v[i][j][k];
    const double w_b = k == 0 ? 0.0 : static_cast<double>(f.w[i][j][k - 1]);
    const double w_t = f.w[i][j][k];
    const double u = 0.5 * (u_w + u_e), v = 0.5 * (v_s + v_n), w = 0.5 * (w_b + w_t);
    return {0.25 * mass * (u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n + w_b * w_b + w_t * w_t),
            0.5 * dry_mass * (u * u + v * v + w * w)};
}
}

std::array<double, 2> compute_flat_cgrid_velocity_energy_pair_deterministic(
    const SimulationStateView& state, Field3D& before_u, Field3D& before_v, Field3D& before_w)
{
    if (state.grid.geometry.coord != CoordinateSystem::Cartesian || !state.grid.geometry.staggered ||
        tmv::physical_grid_metrics::terrain_is_active(state) || state.grid.nr < 3 || state.grid.nth < 3 || state.grid.nz < 2)
        throw std::invalid_argument("velocity energy pair requires a flat Cartesian C-grid");
    const auto& f = state.fields;
    for (const auto* field : {&f.u, &f.v, &f.w, &before_u, &before_v, &before_w})
        if (field->size_r() != state.grid.nr || field->size_th() != state.grid.nth || field->size_z() != state.grid.nz)
            throw std::invalid_argument("velocity energy pair field shape mismatch");

    SimulationStateView prior{state.grid,
        {f.rho, f.p, before_u, before_w, before_v, f.tracer, f.theta, f.qv, f.qc, f.qr, f.qi, f.qs, f.qh, f.qg, f.tke},
        state.base, state.tendencies, state.coriolis_f, state.time_s, state.surface};
    prior.terrain_runtime = state.terrain_runtime;
    prior.domain_frame = state.domain_frame;
    const auto before = compute_energy_diagnostic_snapshot_v1_deterministic(prior);
    return {before.moist_mixture_v2.resolved_kinetic,
            compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(state, before)};
}

double compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(
    const SimulationStateView& state, const EnergyDiagnosticSnapshotV1& before)
{
    if (state.grid.geometry.coord != CoordinateSystem::Cartesian || !state.grid.geometry.staggered ||
        tmv::physical_grid_metrics::terrain_is_active(state) || state.grid.nr < 3 || state.grid.nth < 3 || state.grid.nz < 2)
        throw std::invalid_argument("velocity energy completion requires a flat Cartesian C-grid");
    const auto& f = state.fields;
    for (const auto* field : {&f.rho, &f.p, &f.theta, &f.u, &f.v, &f.w,
            &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke})
        if (field->size_r() != state.grid.nr || field->size_th() != state.grid.nth || field->size_z() != state.grid.nz)
            throw std::invalid_argument("velocity energy completion field shape mismatch");
    const auto domain = physical_scalar_domain(state.grid);
    const tmv::physical_grid_metrics::ControlVolumeMassView geometry(state);
    static_assert(std::numeric_limits<float>::digits == 24 && std::numeric_limits<float>::max_exponent == 128);
    // Binary32 inputs, <=2^31 cells, these volumes and heights keep every
    // complete/legacy inventory below 2^900, including summation roundoff.
    // Larger geometry retains the full evaluator's exact overflow decisions.
    bool bounded = f.rho.size() <= (std::uint64_t{1} << 31);
    for (int k = domain.k_begin; k < domain.k_end && bounded; ++k)
        bounded = geometry.scalar_volume_m3(domain.i_begin, domain.j_begin, k) <= 1e100 &&
            std::abs(tmv::physical_grid_metrics::scalar_height(state, domain.i_begin, domain.j_begin, k)) <= 1e50;
    if (!bounded)
        return compute_moist_mixture_energy_v2_deterministic(state).resolved_kinetic;
    const auto sums = tmv::diagnostics::fixed_slab_reduce<3>(domain.i_begin, domain.i_end,
        [&](int i, std::array<double, 3>& partial)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double velocities[]{f.u[i - 1][j][k], f.u[i][j][k], f.v[i][j - 1][k], f.v[i][j][k],
                        k == 0 ? 0.0 : static_cast<double>(f.w[i][j][k - 1]), f.w[i][j][k]};
                    if (std::any_of(std::begin(velocities), std::end(velocities), [](double v) { return !std::isfinite(v); }))
                    {
                        partial[2] += 1.0;
                        continue;
                    }
                    const auto values = cell_kinetic_energies(f, geometry, i, j, k);
                    partial[0] += values[0];
                    partial[1] += values[1];
                }
        });
    if (sums[2] != 0.0)
        throw std::runtime_error("velocity energy pair rejected non-finite velocities");
    const auto& b = before.moist_mixture_v2;
    const double total = b.dry_internal + b.water_sensible + b.phase_reference + sums[0] + b.gravitational + b.subgrid_tke;
    const double legacy_total = sums[1] + before.legacy_proxy.internal + before.legacy_proxy.potential + before.legacy_proxy.latent;
    if (!std::isfinite(sums[0]) || !std::isfinite(sums[1]) || !std::isfinite(total) || !std::isfinite(legacy_total))
        throw std::overflow_error("velocity energy pair produced a non-finite inventory");
    return sums[0];
}

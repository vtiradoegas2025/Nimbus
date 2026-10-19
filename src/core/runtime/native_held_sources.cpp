#include "core/runtime/native_held_sources.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "compute/dispatch/terrain.hpp"
#include "compute/held_sources_contract.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace tmv
{
namespace
{
void pack(double value, float& high, float& low)
{
    high = static_cast<float>(value);
    low  = static_cast<float>(value - high);
    if (!std::isfinite(high) || !std::isfinite(low) || (value != 0 && high == 0))
        throw std::overflow_error("held-source geometry exceeds compensated storage");
}
} // namespace
ReconstructedHeldSources reconstruct_native_held_sources(const SimulationStateView& state,
                                                         const HeldSourceInputs&    source,
                                                         TerrainComputeKernels&     kernels)
{
    physical_grid_metrics::validate(state);
    if (state.grid.geometry.coord != CoordinateSystem::Cartesian || !state.configuration ||
        !kernels.supports_held_sources())
        throw std::invalid_argument("native held-source reconstruction requires Cartesian state and its kernel");
    const bool pbl = source.theta_rate != nullptr, rad = source.radiation_power != nullptr,
               cloud = source.liquid_rate != nullptr;
    if (!pbl && !rad)
        throw std::invalid_argument("native held-source reconstruction has no owner");
    if (bool(source.u_force) != pbl || bool(source.v_force) != pbl || bool(source.vapor_rate) != pbl ||
        bool(source.ice_rate) != cloud || (cloud && !pbl))
        throw std::invalid_argument("native held-source ownership is incomplete");
    const int nx = state.grid.nr, ny = state.grid.nth, nz = state.grid.nz;
    if (std::size_t(nx) > std::size_t(std::numeric_limits<int>::max()) / HeldSourcesDispatch::input_stride /
                              std::size_t(ny) / std::size_t(nz))
        throw std::overflow_error("native held sources exceed dispatch bounds");
    const auto cells = std::size_t(nx) * ny * nz;
    const auto shape = [&](const Field3D& f)
    {
        if (f.size_r() != nx || f.size_th() != ny || f.size_z() != nz || f.size() != cells)
            throw std::invalid_argument("held-source field shape is invalid");
    };
    shape(state.fields.rho);
    if (rad)
        shape(state.fields.p);
    const std::array<const Field3D*, 7> fields{source.u_force,
                                               source.v_force,
                                               source.theta_rate,
                                               source.vapor_rate,
                                               source.liquid_rate,
                                               source.ice_rate,
                                               source.radiation_power};
    for (auto* field : fields)
        if (field)
            shape(*field);
    HeldSourcesDispatch d;
    auto&               p      = d.constants;
    const auto          domain = physical_scalar_domain(state.grid);
    p.nx                       = nx;
    p.ny                       = ny;
    p.nz                       = nz;
    p.i_begin                  = domain.i_begin;
    p.i_end                    = domain.i_end;
    p.j_begin                  = domain.j_begin;
    p.j_end                    = domain.j_end;
    p.k_begin                  = domain.k_begin;
    p.k_end                    = domain.k_end;
    p.staggered                = state.grid.geometry.staggered;
    p.pbl_enabled              = pbl;
    p.radiation_enabled        = rad;
    p.cloud_enabled            = cloud;
    using L                    = init::config::LateralBC;
    p.periodic_x =
        state.configuration->lateral_bc_west == L::Periodic || state.configuration->lateral_bc_east == L::Periodic;
    p.periodic_y =
        state.configuration->lateral_bc_south == L::Periodic || state.configuration->lateral_bc_north == L::Periodic;
    pack(physical_constants::specific_heat_cp_jkgk, p.cp_high, p.cp_low);
    pack(physical_constants::reference_pressure_pa, p.p0_high, p.p0_low);
    pack(physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk,
         p.kappa_high,
         p.kappa_low);
    std::vector<float> input(cells * HeldSourcesDispatch::input_stride),
        output(cells * HeldSourcesDispatch::output_stride);
    for (std::size_t f = 0; f < fields.size(); ++f)
    {
        if (!fields[f])
            continue;
        for (std::size_t n = 0; n < cells; ++n)
        {
            if (f < 6 && !nimbus::compute::held_sources::owned(d, n))
                continue;
            const float value = fields[f]->data()[n];
            if (!std::isfinite(value))
                throw std::invalid_argument("held source is not finite on its owned cells");
            input[HeldSourcesDispatch::input_stride * n + 4 + f] = value;
        }
    }
    const physical_grid_metrics::ControlVolumeMassView geometry(state);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const auto n = (std::size_t(i) * ny + j) * nz + k;
                if (!rad && !nimbus::compute::held_sources::owned(d, n))
                    continue;
                input[11 * n] = state.fields.rho(i, j, k);
                if (rad)
                    input[11 * n + 1] = state.fields.p(i, j, k);
                pack(geometry.scalar_volume_m3(i, j, k), input[11 * n + 2], input[11 * n + 3]);
            }
    d.input  = input.data();
    d.output = output.data();
    if (!nimbus::compute::held_sources::validate(d))
        throw std::invalid_argument("native held-source input contract failed");
    if (!kernels.dispatch_held_sources(d) || !nimbus::compute::held_sources::finite_candidate(output.data(), d))
        throw std::runtime_error("native held-source reconstruction failed");
    ReconstructedHeldSources      result;
    const std::array<Field3D*, 7> targets{
        &result.du, &result.dv, &result.dtheta_pbl, &result.dqv, &result.dqc, &result.dqi, &result.dtheta_rad};
    for (std::size_t f = 0; f < targets.size(); ++f)
    {
        if ((f < 6 && !pbl) || (f == 6 && !rad))
            continue;
        targets[f]->resize(nx, ny, nz, 0);
        for (std::size_t n = 0; n < cells; ++n)
            targets[f]->data()[n] = output[7 * n + f];
    }
    return result;
}
void reconstruct_native_held_sources(SimulationStateView& state, TerrainComputeKernels& kernels)
{
    HeldSourceInputs sources;
    if (state.boundary_layer_runtime && state.boundary_layer_runtime->scheme &&
        state.boundary_layer_runtime->scheme->name() != "none")
    {
        auto& p            = *state.boundary_layer_runtime;
        sources.u_force    = &p.held_u_force_n;
        sources.v_force    = &p.held_v_force_n;
        sources.theta_rate = &p.held_theta_inventory_rate_kgk_s;
        sources.vapor_rate = &p.held_vapor_mass_rate_kg_s;
        if (!p.config.clear_sky_only)
        {
            sources.liquid_rate = &p.held_cloud_liquid_mass_rate_kg_s;
            sources.ice_rate    = &p.held_cloud_ice_mass_rate_kg_s;
            if (!state.tendencies.dqc_dt_pbl || !state.tendencies.dqi_dt_pbl)
                throw std::invalid_argument("native cloudy PBL sources require cloud tendencies");
        }
    }
    if (state.radiation_runtime && !state.radiation_runtime->held_cell_power_w.empty())
        sources.radiation_power = &state.radiation_runtime->held_cell_power_w;
    if (!sources.theta_rate && !sources.radiation_power)
        return;
    auto result = reconstruct_native_held_sources(state, sources, kernels);
    if (sources.theta_rate)
    {
        state.tendencies.du_dt_pbl     = std::move(result.du);
        state.tendencies.dv_dt_pbl     = std::move(result.dv);
        state.tendencies.dtheta_dt_pbl = std::move(result.dtheta_pbl);
        state.tendencies.dqv_dt_pbl    = std::move(result.dqv);
        if (state.tendencies.dqc_dt_pbl)
            *state.tendencies.dqc_dt_pbl = std::move(result.dqc);
        if (state.tendencies.dqi_dt_pbl)
            *state.tendencies.dqi_dt_pbl = std::move(result.dqi);
    }
    if (sources.radiation_power)
        state.tendencies.dtheta_dt_rad = std::move(result.dtheta_rad);
}
} // namespace tmv

#include "chaos/runtime_internal.hpp"

#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "util/string_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv::chaos::runtime_detail
{

namespace
{

ChaosRuntimeView runtime_view(ChaosRuntimeState& runtime)
{
    return {runtime.scheme, runtime.config, runtime.diagnostics,
            runtime.noise_elapsed_s, runtime.base_du_dt_pbl,
            runtime.base_dv_dt_pbl, runtime.base_dtheta_dt_pbl,
            runtime.base_dqv_dt_pbl, runtime.base_dtke_dt_pbl,
            runtime.base_sfc_ustar, runtime.base_sfc_tau_u,
            runtime.base_sfc_tau_v, runtime.base_sfc_sensible_heat_flux,
            runtime.base_sfc_latent_heat_flux,
            runtime.base_sfc_moisture_flux,
            runtime.pbl_base_tendencies_valid,
            runtime.grid_metrics_scratch};
}

double spacing_from_coordinates(const std::vector<double>& coords,
                                double configured_spacing)
{
    if (coords.size() >= 2)
    {
        const double spacing = coords[1] - coords[0];
        if (std::isfinite(spacing) && spacing > 0.0)
        {
            return spacing;
        }
        throw std::invalid_argument(
            "Chaos grid coordinates must have positive finite spacing");
    }
    if (!std::isfinite(configured_spacing) || configured_spacing <= 0.0)
    {
        throw std::invalid_argument(
            "Chaos configured grid spacing must be positive and finite");
    }
    return configured_spacing;
}

} // namespace

ChaosRuntimeView chaos_runtime_for(const SimulationStateView& state)
{
    return runtime_view(tmv::require_level_runtime(state.chaos_runtime, "chaos runtime"));
}

std::string normalize_tendency_block_name(const std::string& block_raw)
{
    std::string block = tmv::strutil::trim_and_lower(block_raw);
    if (block == "pbl" || block == "boundary_layer" ||
        block == "boundarylayer" || block == "bl")
    {
        return "pbl";
    }
    if (block == "micro" || block == "microphysics")
    {
        return "microphysics";
    }
    if (block == "diffusion" || block == "turbulence" ||
        block == "sgs" || block == "subgrid")
    {
        return "diffusion";
    }
    return block;
}

void invalidate_pbl_base_tendencies(ChaosRuntimeView& runtime)
{
    runtime.base_du_dt_pbl = Field3D();
    runtime.base_dv_dt_pbl = Field3D();
    runtime.base_dtheta_dt_pbl = Field3D();
    runtime.base_dqv_dt_pbl = Field3D();
    runtime.base_dtke_dt_pbl = Field3D();
    runtime.base_sfc_ustar.clear();
    runtime.base_sfc_tau_u.clear();
    runtime.base_sfc_tau_v.clear();
    runtime.base_sfc_sensible_heat_flux.clear();
    runtime.base_sfc_latent_heat_flux.clear();
    runtime.base_sfc_moisture_flux.clear();
    runtime.pbl_base_tendencies_valid = false;
}

const GridMetrics& grid_metrics_from_state(
    const SimulationStateView& state)
{
    const TerrainMetrics3D* terrain_metrics =
        physical_grid_metrics::active_terrain_metrics(state);
    ChaosRuntimeView runtime = chaos_runtime_for(state);

    const auto* owned_grid = state.process_grid_metrics;
    if (terrain_metrics == nullptr && owned_grid &&
        std::isfinite(owned_grid->dx) && std::isfinite(owned_grid->dy) &&
        owned_grid->dx > 0.0 && owned_grid->dy > 0.0 &&
        owned_grid->dz.size() == static_cast<std::size_t>(state.grid.nz))
        return *owned_grid;

    GridMetrics& grid = runtime.grid_metrics_scratch;
    grid.dx = spacing_from_coordinates(state.grid.geometry.r, state.grid.dr);
    grid.dy = spacing_from_coordinates(
        state.grid.geometry.theta, state.grid.dtheta);
    grid.staggering = state.grid.geometry.staggered
        ? StaggerType::CGrid : StaggerType::Collocated;
    if (!std::isfinite(state.grid.dz) || state.grid.dz <= 0.0)
    {
        throw std::invalid_argument("Chaos vertical grid spacing must be positive and finite");
    }
    grid.dz.assign(static_cast<std::size_t>(state.grid.nz), state.grid.dz);
    grid.terrain_metrics_active = false;
    grid.terrain_metrics = nullptr;
    grid.terrain_topography = nullptr;
    if (terrain_metrics != nullptr)
    {
        physical_grid_metrics::validate(state);
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
                    "terrain chaos layer thickness must be positive and finite");
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

} // namespace tmv::chaos::runtime_detail

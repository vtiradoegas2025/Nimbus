#include "core/runtime/native_earth_fixed_transport.hpp"
#include "core/runtime/simulation_state.hpp"
#include "compute/dispatch/terrain.hpp"
#include "dynamics/native_moving_cgrid_transport.hpp"
#include "terrain/native_conservative_remap.hpp"
#include <stdexcept>

namespace tmv
{
PreparedNativeEarthFixedTransport prepare_native_earth_fixed_transport(const SimulationStateView& state,
                                                                       double                     end_time_s,
                                                                       TerrainComputeKernels&     kernels)
{
    if (!state.terrain_runtime || !state.domain_frame || !state.grid.geometry.staggered)
        throw std::invalid_argument("native Earth transport requires mapped C-grid state and a domain frame");
    if (!kernels.supports_earth_geometry() || !kernels.supports_conservative_remap() ||
        !kernels.supports_moving_momentum())
        throw std::runtime_error("native Earth transport requires the complete geometry and transport kernel set");
    PreparedNativeEarthFixedTransport result;
    result.motion = prepare_earth_fixed_terrain_motion(
        *state.terrain_runtime, *state.domain_frame, state.time_s, end_time_s, kernels);
    result.mass_flux = terrain_native::moving_mesh_mass_flux(kernels, state.fields.rho, result.motion.swept_volumes);
    result.dry_mass  = terrain_native::remap_inventory(kernels,
                                                       state.fields.rho,
                                                       result.density,
                                                       state.fields.rho,
                                                       state.fields.rho,
                                                       state.terrain_runtime->metrics,
                                                       result.motion.terrain.geometry.metrics,
                                                       result.motion.swept_volumes,
                                                       false,
                                                       terrain_native::RemapLowerBound::Positive);
    result.momentum  = prepare_moving_cgrid_momentum_transport(
        state, result.motion.terrain.geometry.metrics, result.density, result.mass_flux, kernels, true);
    result.ground_targets = prepare_moving_ground_targets(
        state.terrain_runtime->metrics, state.fields.rho, state.fields.u, state.fields.v, result.mass_flux, kernels);
    result.projected_u    = result.momentum.u;
    result.projected_v    = result.momentum.v;
    result.top_projection = project_moving_top_momentum(
        result.motion.terrain.geometry.metrics, result.density, result.projected_u, result.projected_v, kernels);
    return result;
}
PreparedNativeEarthFixedEnergy prepare_native_earth_fixed_energy(const SimulationStorageState&            state,
                                                                 const PreparedNativeEarthFixedTransport& transport,
                                                                 TerrainComputeKernels&                   kernels)
{
    if (state.time_s != transport.motion.begin_time_s ||
        !can_commit_earth_fixed_terrain(transport.motion.terrain, state.runtime.terrain))
        throw std::invalid_argument("native moving-energy preparation has stale terrain or state ownership");
    const auto initial = moving_energy_density(state.fields, state.runtime.terrain.metrics, kernels);
    PreparedNativeEarthFixedEnergy result;
    result.transport = terrain_native::remap_inventory(kernels,
                                                       initial,
                                                       result.target_density,
                                                       state.runtime.terrain.metrics,
                                                       transport.motion.terrain.geometry.metrics,
                                                       transport.motion.swept_volumes);
    return result;
}
} // namespace tmv

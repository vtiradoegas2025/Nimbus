#pragma once

#include "dynamics/moving_cgrid_transport.hpp"
#include "dynamics/moving_boundary.hpp"
#include "dynamics/moving_energy.hpp"
#include "terrain/earth_fixed_geometry.hpp"

struct SimulationStorageState;

namespace tmv
{
struct PreparedNativeEarthFixedTransport
{
    PreparedEarthFixedTerrainMotion     motion;
    TerrainContravariantFluxes          mass_flux;
    Field3D                             density;
    MovingMeshInventoryUpdate           dry_mass;
    MovingCGridMomentumTransport        momentum;
    MovingGroundTargets                 ground_targets;
    Field3D                             projected_u, projected_v;
    std::array<MovingBoundaryLedger, 2> top_projection;
};

struct PreparedNativeEarthFixedEnergy
{
    std::vector<double>       target_density;
    MovingMeshInventoryUpdate transport;
};
PreparedNativeEarthFixedEnergy prepare_native_earth_fixed_energy(const SimulationStorageState&            state,
                                                                 const PreparedNativeEarthFixedTransport& transport,
                                                                 TerrainComputeKernels&                   kernels);

PreparedNativeEarthFixedTransport prepare_native_earth_fixed_transport(const SimulationStateView& state,
                                                                       double                     end_time_s,
                                                                       TerrainComputeKernels&     kernels);
} // namespace tmv

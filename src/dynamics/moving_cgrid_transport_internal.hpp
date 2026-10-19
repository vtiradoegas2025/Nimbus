#pragma once
#include "dynamics/moving_cgrid_transport.hpp"

namespace tmv::moving_transport_detail
{
const TerrainMetrics3D& validate(const SimulationStateView&        state,
                                 const TerrainMetrics3D&           after,
                                 const Field3D&                    density_after,
                                 const TerrainContravariantFluxes& flux);
}

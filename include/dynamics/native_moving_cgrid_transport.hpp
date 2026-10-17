#pragma once
#include "dynamics/moving_cgrid_transport.hpp"
#include "terrain/earth_fixed_geometry.hpp"

namespace tmv::terrain_native
{
TerrainContravariantFluxes
moving_mesh_mass_flux(TerrainComputeKernels& kernels, const Field3D& density, const TerrainMeshSweptVolumes& sweep);
}

#pragma once

#include "compute/dispatch/terrain.hpp"
#include "terrain/earth_fixed_geometry.hpp"

namespace tmv::terrain_native
{
enum class RemapLowerBound
{
    Unrestricted,
    Nonnegative,
    Positive
};

MovingMeshInventoryUpdate remap_inventory(TerrainComputeKernels&         kernels,
                                          const Field3D&                 before,
                                          Field3D&                       after,
                                          const Field3D&                 density_begin,
                                          const Field3D&                 density_end,
                                          const TerrainMetrics3D&        geometry_begin,
                                          const TerrainMetrics3D&        geometry_end,
                                          const TerrainMeshSweptVolumes& sweep,
                                          bool                           mixing_ratio,
                                          RemapLowerBound                lower_bound = RemapLowerBound::Unrestricted);

MovingMeshInventoryUpdate remap_inventory(TerrainComputeKernels&         kernels,
                                          const std::vector<double>&     before,
                                          std::vector<double>&           after,
                                          const TerrainMetrics3D&        geometry_begin,
                                          const TerrainMetrics3D&        geometry_end,
                                          const TerrainMeshSweptVolumes& sweep,
                                          RemapLowerBound                lower_bound = RemapLowerBound::Unrestricted);
} // namespace tmv::terrain_native

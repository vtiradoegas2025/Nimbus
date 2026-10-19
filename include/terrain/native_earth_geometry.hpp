#pragma once
#include "terrain/earth_fixed_geometry.hpp"
class TerrainComputeKernels;

namespace tmv::terrain_native
{
EarthFixedTerrainGeometry        build_earth_geometry(TerrainComputeKernels&               kernels,
                                                      const TerrainConfig&                 config,
                                                      const EarthFixedTerrainBuildRequest& request);
EarthFixedVolumeGeometryInterval build_earth_geometry_interval(TerrainComputeKernels&               kernels,
                                                               const TerrainConfig&                 config,
                                                               const EarthFixedTerrainBuildRequest& begin,
                                                               domain::HorizontalPoint              end_origin_earth_m,
                                                               double                               interval_s,
                                                               const TerrainMetrics3D* expected_begin = nullptr);
} // namespace tmv::terrain_native

#pragma once
#include "terrain/terrain_base.hpp"
struct PrognosticFieldStorage;
struct BaseStateStorage;
class TerrainComputeKernels;
namespace tmv
{
void fill_moving_environment(PrognosticFieldStorage& fields,
                             const BaseStateStorage& base,
                             const TerrainMetrics3D& geometry);
void fill_moving_environment(PrognosticFieldStorage& fields,
                             const BaseStateStorage& base,
                             const TerrainMetrics3D& geometry,
                             TerrainComputeKernels&  kernels);
} // namespace tmv

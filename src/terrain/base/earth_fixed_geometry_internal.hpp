#pragma once
#include "terrain/earth_fixed_geometry.hpp"

namespace tmv::earth_fixed_detail
{
void require_build_contract(const TerrainConfig& config, const EarthFixedTerrainBuildRequest& request);

PreparedEarthFixedTerrainCommit prepare_earth_fixed_terrain_identity(const TerrainRuntimeState&           live,
                                                                     const EarthFixedTerrainBuildRequest& request);

struct EarthFixedMotionIdentity
{
    PreparedEarthFixedTerrainMotion prepared;
    EarthFixedTerrainBuildRequest   begin;
    domain::HorizontalPoint         end;
};
EarthFixedMotionIdentity prepare_earth_fixed_motion_identity(const TerrainRuntimeState&           live,
                                                             const domain::HorizontalDomainFrame& frame,
                                                             double                               begin_time_s,
                                                             double                               end_time_s);
} // namespace tmv::earth_fixed_detail

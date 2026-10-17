#pragma once

#include "terrain/terrain_base.hpp"

namespace tmv
{
struct EarthFixedTerrainBuildRequest;
extern TerrainRuntimeState& global_terrain_runtime_state;

// Compatibility bindings into the process-owned runtime.
extern std::unique_ptr<TerrainSchemeBase>& terrain_scheme;
extern TerrainConfig& global_terrain_config;
extern Topography2D& global_topography;
extern TerrainMetrics3D& global_terrain_metrics;

void initialize_terrain(
    const std::string& scheme_name,
    const TerrainConfig& cfg = TerrainConfig{});
void initialize_earth_fixed_terrain(
    const std::string& scheme_name,
    const TerrainConfig& cfg,
    const EarthFixedTerrainBuildRequest& request);
void build_terrain_fields();
}

using tmv::global_terrain_runtime_state;
using tmv::terrain_scheme;
using tmv::global_terrain_config;
using tmv::global_topography;
using tmv::global_terrain_metrics;
using tmv::initialize_terrain;
using tmv::initialize_earth_fixed_terrain;
using tmv::build_terrain_fields;

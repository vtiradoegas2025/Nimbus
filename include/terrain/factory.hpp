#pragma once

// Public terrain factory surface.

#include "terrain/scheme_descriptor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tmv
{
class TerrainSchemeBase;

std::unique_ptr<TerrainSchemeBase> create_terrain_scheme(const std::string& scheme_name);
std::vector<std::string> get_available_terrain_schemes();
} // namespace tmv

using tmv::create_terrain_scheme;
using tmv::get_available_terrain_schemes;
using tmv::get_terrain_scheme_descriptors;

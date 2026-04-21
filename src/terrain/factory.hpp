/**
 * @file factory.hpp
 * @brief Declarations for the terrain module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the terrain runtime and scheme implementations.
 * This file is part of the src/terrain subsystem.
 */

#pragma once
#include <memory>
#include <string>
#include "terrain/terrain_base.hpp"

class BellScheme;
class ScharScheme;
class NoneScheme;

/**
 * @brief Creates a terrain scheme by configured name.
 */
std::unique_ptr<TerrainSchemeBase> create_terrain_scheme(const std::string& scheme_name);

/**
 * @brief Returns names of available terrain schemes.
 */
std::vector<std::string> get_available_terrain_schemes();

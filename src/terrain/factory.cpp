/**
 * @file factory.cpp
 * @brief Implementation for the terrain module.
 *
 * Provides executable logic for the terrain runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/terrain subsystem.
 */

#include "factory.hpp"
#include "schemes/bell/bell.hpp"
#include "schemes/schar/schar.hpp"
#include "schemes/none.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<TerrainSchemeBase> registry({
    {"none",  [] { return std::make_unique<NoneScheme>(); }},
    {"bell",  [] { return std::make_unique<BellScheme>(); }},
    {"schar", [] { return std::make_unique<ScharScheme>(); }},
}, {
    {"flat",   "none"},
    {"schaer", "schar"},
});
}

/**
 * @brief Creates a terrain scheme instance from configured id.
 */
std::unique_ptr<TerrainSchemeBase> create_terrain_scheme(const std::string& scheme_name)
{
    return registry.create("terrain", scheme_name);
}

/**
 * @brief Gets the available terrain schemes.
 */
std::vector<std::string> get_available_terrain_schemes()
{
    return registry.available_ids();
}

/**
 * @file factory.cpp
 * @brief Implementation for the dynamics module.
 *
 * Provides executable logic for the dynamics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/dynamics subsystem.
 */

#include "factory.hpp"
#include "schemes/supercell/supercell.hpp"
#include "schemes/tornado/tornado.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<DynamicsScheme> registry({
    {"supercell", [] { return std::make_unique<SupercellScheme>(); }},
    {"tornado",   [] { return std::make_unique<TornadoScheme>(); }},
}, {
    {"mesocyclone",  "supercell"},
    {"axisymmetric", "tornado"},
});
}

/**
 * @brief Creates a dynamics scheme instance from its configured id.
 */
std::unique_ptr<DynamicsScheme> create_dynamics_scheme(const std::string& scheme_name)
{
    return registry.create("dynamics", scheme_name);
}

/**
 * @brief Returns supported dynamics scheme ids.
 */
std::vector<std::string> get_available_dynamics_schemes()
{
    return registry.available_ids();
}

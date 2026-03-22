/**
 * @file factory.cpp
 * @brief Implementation for the numerics module.
 *
 * Provides executable logic for the numerics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/numerics subsystem.
 */

#include "factory.hpp"
#include "schemes/tvd/tvd.hpp"
#include "schemes/weno5/weno5.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<AdvectionSchemeBase> registry({
    {"tvd",   [] { return std::make_unique<TVDScheme>(); }},
    {"weno5", [] { return std::make_unique<WENO5Scheme>(); }},
});
}

/**
 * @brief Creates the advection scheme.
 */
std::unique_ptr<AdvectionSchemeBase> create_advection_scheme(const std::string& scheme_name)
{
    return registry.create("advection", scheme_name);
}

std::vector<std::string> get_available_advection_schemes()
{
    return registry.available_ids();
}

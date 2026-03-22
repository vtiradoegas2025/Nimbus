/**
 * @file factory.cpp
 * @brief Implementation for the numerics module.
 *
 * Provides executable logic for the numerics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/numerics subsystem.
 */

#include "factory.hpp"
#include "schemes/rk3/rk3.hpp"
#include "schemes/rk4/rk4.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<TimeSteppingSchemeBase> registry({
    {"rk3", [] { return std::make_unique<RK3Scheme>(); }},
    {"rk4", [] { return std::make_unique<RK4Scheme>(); }},
}, {
    {"ssprk3", "rk3"},
});
}

/**
 * @brief Creates the time stepping scheme.
 */
std::unique_ptr<TimeSteppingSchemeBase> create_time_stepping_scheme(const std::string& scheme_name)
{
    return registry.create("time stepping", scheme_name);
}

/**
 * @brief Gets the available time stepping schemes.
 */
std::vector<std::string> get_available_time_stepping_schemes()
{
    return registry.available_ids();
}

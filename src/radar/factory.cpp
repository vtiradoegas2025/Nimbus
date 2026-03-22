/**
 * @file factory.cpp
 * @brief Implementation for the radar module.
 *
 * Provides executable logic for the radar runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radar subsystem.
 */

#include "factory.hpp"
#include "schemes/reflectivity/reflectivity.hpp"
#include "schemes/velocity/velocity.hpp"
#include "schemes/zdr/zdr.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<RadarSchemeBase> registry({
    {"reflectivity", [] { return std::make_unique<ReflectivityScheme>(); }},
    {"velocity",     [] { return std::make_unique<VelocityScheme>(); }},
    {"zdr",          [] { return std::make_unique<ZDRScheme>(); }},
}, {
    {"dbz",             "reflectivity"},
    {"ref",             "reflectivity"},
    {"z",               "reflectivity"},
    {"vel",             "velocity"},
    {"vr",              "velocity"},
    {"radial_velocity", "velocity"},
});
}

/**
 * @brief Creates the radar scheme.
 */
std::unique_ptr<RadarSchemeBase> create_radar_scheme(const std::string& scheme_name)
{
    return registry.create("radar", scheme_name);
}

/**
 * @brief Gets the available radar schemes.
 */
std::vector<std::string> get_available_radar_schemes()
{
    return registry.available_ids();
}

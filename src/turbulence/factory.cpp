/**
 * @file factory.cpp
 * @brief Implementation for the turbulence module.
 *
 * Provides executable logic for the turbulence runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/turbulence subsystem.
 */

#include "factory.hpp"
#include "schemes/smagorinsky/smagorinsky.hpp"
#include "schemes/tke/tke.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<TurbulenceSchemeBase> registry({
    {"smagorinsky", [] { return std::make_unique<SmagorinskyScheme>(); }},
    {"tke",         [] { return std::make_unique<TKEScheme>(); }},
}, {
    {"smag",              "smagorinsky"},
    {"smagorinsky_lilly", "smagorinsky"},
    {"smagorinsky-lilly", "smagorinsky"},
    {"1.5",               "tke"},
    {"1.5order",          "tke"},
    {"1.5-order",         "tke"},
});
}

/**
 * @brief Creates a turbulence scheme instance from configured id.
 */
std::unique_ptr<TurbulenceSchemeBase> create_turbulence_scheme(const std::string& scheme_name)
{
    return registry.create("turbulence", scheme_name);
}

/**
 * @brief Gets the available turbulence schemes.
 */
std::vector<std::string> get_available_turbulence_schemes()
{
    return registry.available_ids();
}

/**
 * @file factory.cpp
 * @brief Implementation for the radiation module.
 *
 * Provides executable logic for the radiation runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radiation subsystem.
 */

#include "factory.hpp"
#include "schemes/simple_grey/simple_grey.hpp"
#include "util/scheme_factory.hpp"

#include <stdexcept>

namespace
{
/**
 * @brief Registry for implemented radiation schemes.
 *
 * The rrtmg scheme is recognized for compatibility but not registered
 * here — it is handled explicitly in create_radiation_scheme() to
 * provide a distinct error message.
 */
const tmv::SchemeRegistry<RadiationSchemeBase> registry({
    {"simple_grey", [] { return std::make_unique<SimpleGreyScheme>(); }},
}, {
    {"simple-grey", "simple_grey"},
    {"simplegray",  "simple_grey"},
    {"simplegrey",  "simple_grey"},
    {"gray",        "simple_grey"},
    {"grey",        "simple_grey"},
});

const std::vector<std::string> k_rrtmg_aliases = {
    "rrtmg", "rrtmg_lw", "rrtmg_sw", "rrtmg_lw_sw", "rrtmgp", "rrtmgp_lw_sw"
};

bool is_rrtmg_alias(const std::string& normalized)
{
    for (const auto& alias : k_rrtmg_aliases)
    {
        if (alias == normalized) return true;
    }
    return false;
}
}

std::string canonicalize_radiation_scheme_id(const std::string& scheme_name)
{
    const std::string resolved = registry.resolve(scheme_name);
    if (is_rrtmg_alias(resolved))
    {
        return "rrtmg";
    }
    return resolved;
}

bool is_radiation_scheme_implemented(const std::string& scheme_name)
{
    return canonicalize_radiation_scheme_id(scheme_name) == "simple_grey";
}

/**
 * @brief Creates the radiation scheme.
 */
std::unique_ptr<RadiationSchemeBase> create_radiation_scheme(const std::string& scheme_name)
{
    const std::string canonical = canonicalize_radiation_scheme_id(scheme_name);

    if (is_rrtmg_alias(canonical))
    {
        throw std::runtime_error(
            "Radiation scheme 'rrtmg' is recognized for platform-fidelity compatibility "
            "but is not yet implemented in this build. Use 'simple_grey' for runtime execution.");
    }

    return registry.create("radiation", scheme_name);
}

/**
 * @brief Gets the available radiation schemes.
 */
std::vector<std::string> get_available_radiation_schemes()
{
    return registry.available_ids();
}

/**
 * @brief Gets implemented and planned radiation scheme identifiers.
 */
std::vector<std::string> get_known_radiation_schemes()
{
    return {"simple_grey", "rrtmg"};
}

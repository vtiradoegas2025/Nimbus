/**
 * @file factory.cpp
 * @brief Implementation for the numerics module.
 *
 * Provides executable logic for the numerics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/numerics subsystem.
 */

#include "factory.hpp"
#include "schemes/explicit/explicit.hpp"
#include "schemes/implicit/implicit.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<DiffusionSchemeBase> registry({
    {"explicit", [] { return std::make_unique<ExplicitDiffusionScheme>(); }},
    {"implicit", [] { return std::make_unique<ImplicitDiffusionScheme>(); }},
});
}

/**
 * @brief Creates the diffusion scheme.
 */
std::unique_ptr<DiffusionSchemeBase> create_diffusion_scheme(const std::string& scheme_name)
{
    return registry.create("diffusion", scheme_name);
}

/**
 * @brief Gets the available diffusion schemes.
 */
std::vector<std::string> get_available_diffusion_schemes()
{
    return registry.available_ids();
}

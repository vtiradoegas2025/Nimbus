/**
 * @file factory.cpp
 * @brief Implementation for the microphysics module.
 *
 * Provides executable logic for the microphysics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/microphysics subsystem.
 */

#include "factory.hpp"
#include "schemes/kessler/kessler.hpp"
#include "schemes/lin/lin.hpp"
#include "schemes/thompson/thompson.hpp"
#include "schemes/milbrandt/milbrandt.hpp"
#include "schemes/none/none.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<MicrophysicsScheme> registry({
    {"kessler",   [] { return std::make_unique<KesslerScheme>(); }},
    {"lin",       [] { return std::make_unique<LinScheme>(); }},
    {"thompson",  [] { return std::make_unique<ThompsonScheme>(); }},
    {"milbrandt", [] { return std::make_unique<MilbrandtScheme>(); }},
    {"none",      [] { return std::make_unique<microphysics::NoneScheme>(); }},
});
}

/**
 * @brief Creates a microphysics scheme instance from its configured id.
 */
std::unique_ptr<MicrophysicsScheme> create_microphysics_scheme(const std::string& scheme_name)
{
    return registry.create("microphysics", scheme_name);
}

/**
 * @brief Returns supported microphysics scheme ids.
 */
std::vector<std::string> get_available_schemes()
{
    return registry.available_ids();
}

/**
 * @file factory.cpp
 * @brief Implementation for the soundings module.
 *
 * Provides executable logic for the soundings runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/soundings subsystem.
 */

#include "factory.hpp"
#include "schemes/sharpy/sharpy_sounding.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<SoundingScheme> registry({
    {"sharpy", [] { return std::make_unique<SharpySoundingScheme>(); }},
});
}

/**
 * @brief Creates the SHARPY sounding scheme.
 */
std::unique_ptr<SoundingScheme> create_sharpy_sounding_scheme()
{
    return std::make_unique<SharpySoundingScheme>();
}

std::unique_ptr<SoundingScheme> create_sounding_scheme(const std::string& scheme_id)
{
    const std::string normalized = tmv::strutil::trim_and_lower(scheme_id);

    if (normalized == "none" || normalized.empty())
    {
        return nullptr;
    }

    return registry.create("sounding", scheme_id);
}

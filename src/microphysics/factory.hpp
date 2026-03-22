/**
 * @file factory.hpp
 * @brief Declarations for the microphysics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the microphysics runtime and scheme implementations.
 * This file is part of the src/microphysics subsystem.
 */

#pragma once
#include <memory>
#include <string>
#include <vector>
#include "physics/microphysics_base.hpp"

class KesslerScheme;
class LinScheme;
class ThompsonScheme;
class MilbrandtScheme;

/**
 * @brief Creates a microphysics scheme by configured name.
 */
std::unique_ptr<MicrophysicsScheme> create_microphysics_scheme(const std::string& scheme_name);

/**
 * @brief Returns names of available microphysics schemes.
 */
std::vector<std::string> get_available_schemes();

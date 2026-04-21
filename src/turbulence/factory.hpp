/**
 * @file factory.hpp
 * @brief Declarations for the turbulence module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the turbulence runtime and scheme implementations.
 * This file is part of the src/turbulence subsystem.
 */

#pragma once
#include <memory>
#include <string>
#include "turbulence/turbulence_base.hpp"

class SmagorinskyScheme;
class TKEScheme;

/**
 * @brief Creates a turbulence scheme by configured name.
 */
std::unique_ptr<TurbulenceSchemeBase> create_turbulence_scheme(const std::string& scheme_name);

/**
 * @brief Returns names of available turbulence schemes.
 */
std::vector<std::string> get_available_turbulence_schemes();

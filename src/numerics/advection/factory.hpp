/**
 * @file factory.hpp
 * @brief Declarations for the numerics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the numerics runtime and scheme implementations.
 * This file is part of the src/numerics subsystem.
 */

#pragma once
#include <memory>
#include <string>
#include "numerics/advection/advection_base.hpp"

class TVDScheme;
class WENO5Scheme;

/**
 * @brief Creates an advection scheme by configured name.
 */
std::unique_ptr<AdvectionSchemeBase> create_advection_scheme(const std::string& scheme_name);

/**
 * @brief Returns names of available advection schemes.
 */
std::vector<std::string> get_available_advection_schemes();

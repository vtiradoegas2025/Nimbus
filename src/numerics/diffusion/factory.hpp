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
#include "numerics/diffusion_base.hpp"

class ExplicitDiffusionScheme;
class ImplicitDiffusionScheme;

/**
 * @brief Creates a diffusion scheme by configured name.
 */
std::unique_ptr<DiffusionSchemeBase> create_diffusion_scheme(const std::string& scheme_name);

/**
 * @brief Returns names of available diffusion schemes.
 */
std::vector<std::string> get_available_diffusion_schemes();

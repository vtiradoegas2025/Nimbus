/**
 * @file factory.hpp
 * @brief Declarations for the boundary_layer module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the boundary_layer runtime and scheme implementations.
 * This file is part of the src/boundary_layer subsystem.
 */

#pragma once
#include <memory>
#include <string>
#include "physics/boundary_layer_base.hpp"

class SlabScheme;
class YSUScheme;
class MYNNScheme;

/**
 * @brief Creates a boundary-layer scheme by configured name.
 */
std::unique_ptr<BoundaryLayerSchemeBase> create_boundary_layer_scheme(const std::string& scheme_name);

/**
 * @brief Returns names of available boundary-layer schemes.
 */
std::vector<std::string> get_available_boundary_layer_schemes();

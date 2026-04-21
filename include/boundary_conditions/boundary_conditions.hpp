/**
 * @file boundary_conditions.hpp
 * @brief Factory and scheme interface for coordinate-specific boundary conditions.
 *
 * Each coordinate system has its own BoundaryConditionScheme implementation:
 *   - CartesianBCScheme  (src/core/boundary_conditions_cartesian.cpp)
 *   - CylindricalBCScheme (src/core/boundary_conditions_cylindrical.cpp)
 *
 * The active scheme is selected at startup by `init_boundary_conditions()`
 * based on `global_coordinate_system`.
 */

#pragma once

#include "boundary_conditions/boundary_conditions_base.hpp"
#include <memory>
#include <string>

/// Create a Cartesian boundary condition scheme.
std::unique_ptr<BoundaryConditionScheme> create_cartesian_bc_scheme();

/// Create a cylindrical boundary condition scheme.
std::unique_ptr<BoundaryConditionScheme> create_cylindrical_bc_scheme();

// Convenience free functions for tests and legacy call sites.
// These create a temporary scheme and apply — no long-lived state needed.
inline void apply_cartesian_boundary_conditions()
{
    static auto scheme = create_cartesian_bc_scheme();
    scheme->apply_full();
}

inline void apply_cylindrical_boundary_conditions()
{
    static auto scheme = create_cylindrical_bc_scheme();
    scheme->apply_full();
}

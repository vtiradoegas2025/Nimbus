/**
 * @file boundary_conditions.hpp
 * @brief Factory and scheme interface for coordinate-specific boundary conditions.
 *
 * Available BoundaryConditionScheme implementations:
 *   - CartesianBCScheme         (boundary_conditions_cartesian.cpp)
 *   - CylindricalBCScheme       (boundary_conditions_cylindrical.cpp)
 *   - CylindricalCGridBCScheme  (boundary_conditions_cylindrical_cgrid.cpp)
 *
 * The dispatcher `create_boundary_condition_scheme(coord, stagger)` owns the
 * selection logic. Callers (dynamics orchestration, tests) ask for a scheme
 * by (coordinate system, stagger type) without needing to know which
 * concrete class to construct.
 */

#pragma once

#include "boundary_conditions/boundary_conditions_base.hpp"
#include "core/infra/coordinate_system.hpp"
#include <memory>

/// Create a Cartesian boundary condition scheme.
std::unique_ptr<BoundaryConditionScheme> create_cartesian_bc_scheme();

/// Create a collocated cylindrical boundary condition scheme.
std::unique_ptr<BoundaryConditionScheme> create_cylindrical_bc_scheme();

/// Create an Arakawa C-grid (staggered) cylindrical boundary condition scheme.
/// See boundary_conditions_cylindrical_cgrid.cpp for the field-placement and
/// BC convention. Phase C.2.
std::unique_ptr<BoundaryConditionScheme> create_cylindrical_cgrid_bc_scheme();

/// Dispatcher: creates the BC scheme matching the requested coordinate system
/// and stagger arrangement. Throws std::runtime_error on an unsupported
/// combination (e.g. Cartesian + CGrid is not implemented).
std::unique_ptr<BoundaryConditionScheme>
create_boundary_condition_scheme(CoordinateSystem coord, StaggerType stagger);

// Convenience free functions for tests and legacy call sites.
// These create a temporary scheme and apply -- no long-lived state needed.
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

inline void apply_cylindrical_cgrid_boundary_conditions()
{
    static auto scheme = create_cylindrical_cgrid_bc_scheme();
    scheme->apply_full();
}

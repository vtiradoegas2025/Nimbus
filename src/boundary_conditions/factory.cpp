/**
 * @file factory.cpp
 * @brief Boundary-condition scheme factory.
 *
 * Owns the dispatch from (CoordinateSystem, StaggerType) to a concrete
 * BoundaryConditionScheme implementation. Keeping this logic out of
 * dynamics.cpp preserves the "dynamics orchestrates timestepping" /
 * "boundary_conditions selects and applies BCs" separation of concerns.
 *
 * Cartesian + C-grid is not yet implemented (Phase C is cylindrical-only);
 * an unsupported combination throws so callers fail loudly at startup
 * rather than silently falling back to the wrong stencil.
 */

#include "boundary_conditions/boundary_conditions.hpp"

#include <stdexcept>

std::unique_ptr<BoundaryConditionScheme>
create_boundary_condition_scheme(CoordinateSystem coord, StaggerType stagger)
{
    if (coord == CoordinateSystem::Cartesian)
    {
        if (stagger == StaggerType::CGrid)
        {
            throw std::runtime_error(
                "Cartesian C-grid boundary conditions are not implemented. "
                "Phase C of the coordinate-backend plan adds C-grid support "
                "for the cylindrical coordinate system only.");
        }
        return create_cartesian_bc_scheme();
    }

    if (stagger == StaggerType::CGrid)
        return create_cylindrical_cgrid_bc_scheme();

    return create_cylindrical_bc_scheme();
}

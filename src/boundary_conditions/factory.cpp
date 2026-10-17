#include "boundary_conditions/factory.hpp"

#include <stdexcept>
#include "core/runtime/simulation_state.hpp"

namespace tmv
{

std::unique_ptr<BoundaryConditionScheme> create_boundary_condition_scheme(const SimulationStateView& state)
{
    if (state.grid.geometry.coord != CoordinateSystem::Cartesian)
        throw std::invalid_argument("unsupported boundary-condition coordinate system");
    return state.grid.geometry.staggered ? create_cartesian_cgrid_bc_scheme(state) : create_cartesian_bc_scheme(state);
}

std::unique_ptr<BoundaryConditionScheme>
create_boundary_condition_scheme(CoordinateSystem coord, StaggerType stagger)
{
    if (coord != CoordinateSystem::Cartesian)
        throw std::invalid_argument("unsupported boundary-condition coordinate system");

    switch (stagger)
    {
    case StaggerType::Collocated:
        return create_cartesian_bc_scheme();
    case StaggerType::CGrid:
        return create_cartesian_cgrid_bc_scheme();
    }

    throw std::invalid_argument("unsupported boundary-condition staggering");
}

} // namespace tmv

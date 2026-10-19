#pragma once

#include "boundary_conditions/boundary_conditions_base.hpp"
#include "core/infra/coordinate_system.hpp"

#include <memory>

struct SimulationStateView;

namespace tmv
{

std::unique_ptr<BoundaryConditionScheme> create_cartesian_bc_scheme(const SimulationStateView& state);
std::unique_ptr<BoundaryConditionScheme> create_cartesian_bc_scheme();
std::unique_ptr<BoundaryConditionScheme> create_cartesian_cgrid_bc_scheme(const SimulationStateView& state);
std::unique_ptr<BoundaryConditionScheme> create_cartesian_cgrid_bc_scheme();

std::unique_ptr<BoundaryConditionScheme> create_boundary_condition_scheme(const SimulationStateView& state);

// Select the Cartesian implementation matching the grid staggering.
std::unique_ptr<BoundaryConditionScheme>
create_boundary_condition_scheme(CoordinateSystem coord, StaggerType stagger);

} // namespace tmv

using tmv::create_cartesian_bc_scheme;
using tmv::create_cartesian_cgrid_bc_scheme;
using tmv::create_boundary_condition_scheme;

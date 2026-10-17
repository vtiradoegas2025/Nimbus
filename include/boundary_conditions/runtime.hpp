#pragma once

#include "boundary_conditions/boundary_conditions_base.hpp"

#include <memory>

struct SimulationStateView;

namespace tmv
{

extern std::unique_ptr<BoundaryConditionScheme>& bc_scheme;
std::unique_ptr<BoundaryConditionScheme>& boundary_scheme_for(const SimulationStateView& state);

// Apply the selected full closure, then validate the primary state bounds.
void apply_boundary_conditions(const SimulationStateView& state);
void apply_boundary_conditions();

void rebuild_microphysics_boundary_conditions(const SimulationStateView& state);

void rebuild_boundary_conditions_after_domain_recycling(
    const SimulationStateView& state);
void rebuild_boundary_conditions_after_domain_recycling();

} // namespace tmv

using tmv::apply_boundary_conditions;
using tmv::rebuild_boundary_conditions_after_domain_recycling;
using tmv::bc_scheme;

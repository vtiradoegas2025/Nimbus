#include "core/infra/nested_execution.hpp"

#include "boundary_layer/runtime.hpp"
#include "chaos/runtime.hpp"
#include "dynamics/dynamics_base.hpp"
#include "numerics/runtime.hpp"
#include "radiation/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{

void production_child_step(const SimulationStateView& child_state,
                           double                     child_step_s,
                           double                     child_start_time_s,
                           void*                      context)
{
    const double admitted_step_s = tmv::choose_runtime_timestep(child_state);
    const double tolerance_s     = 1.0e-12 * std::max({1.0, std::abs(child_step_s), std::abs(admitted_step_s)});
    if (!std::isfinite(admitted_step_s) || admitted_step_s + tolerance_s < child_step_s)
    {
        throw std::runtime_error("nested child interval exceeds its live stability limit");
    }

    tmv::step_radiation(child_state, child_start_time_s);
    tmv::step_boundary_layer(child_state, child_start_time_s);
    tmv::chaos::step_chaos_noise(child_state, child_step_s);
    auto& storage = *static_cast<SimulationStorageState*>(context);
    auto  bracket = storage.nested_execution.parent_time_bracket;
    tmv::domain::advance_child_substep(storage.nested_execution.relationship, bracket);
    synchronize_nested_child_chaos_clock(
        storage, tmv::domain::nested_child_time_s(storage.nested_execution.relationship, bracket));
    tmv::chaos::apply_chaos_tendencies(child_state);
    tmv::step_dynamics_new(child_state, child_step_s, child_start_time_s);
}

} // namespace

void advance_nested_child_dynamics_substep(SimulationStorageState&                      storage,
                                           const PrognosticFieldStorage&                previous_parent_fields,
                                           const tmv::domain::NestedRelaxationConfigV1& relaxation,
                                           NestedChildSubstepReport&                    report,
                                           const NestedParentMicrophysicsSnapshotV1*    previous_parent_microphysics)
{
    require_nested_child_level_runtime_ready(storage);
    advance_nested_child_substep(storage,
                                 previous_parent_fields,
                                 relaxation,
                                 production_child_step,
                                 &storage,
                                 report,
                                 previous_parent_microphysics);
}

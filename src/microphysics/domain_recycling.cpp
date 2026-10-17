#include "microphysics/runtime.hpp"

#include "core/runtime/simulation_state.hpp"

namespace tmv
{

void invalidate_microphysics_transients_after_domain_recycling(
    const SimulationStateView& state)
{
    MicrophysicsRuntimeState& runtime = state.microphysics_runtime
        ? *state.microphysics_runtime : global_microphysics_runtime_state;
    Field3D* const tendencies[] = {
        &runtime.tendencies.dtheta_dt, &runtime.tendencies.dqv_dt,
        &runtime.tendencies.dqc_dt, &runtime.tendencies.dqr_dt,
        &runtime.tendencies.dqi_dt, &runtime.tendencies.dqs_dt,
        &runtime.tendencies.dqg_dt, &runtime.tendencies.dqh_dt};
    for (Field3D* field : tendencies)
        if (!field->empty()) field->fill(0.0f);
    if (runtime.scheme)
        runtime.scheme->invalidate_transient_state_after_domain_recycling();
    invalidate_microphysics_reaction_snapshot(runtime);
}

const char* microphysics_recycling_invalidation_algorithm_id() noexcept
{
    return "preserve_prognostics_zero_step_products_v1";
}

} // namespace tmv

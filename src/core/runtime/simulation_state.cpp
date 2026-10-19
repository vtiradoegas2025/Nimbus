#include "core/field/field_inventory.hpp"
#include "core/runtime/simulation_state.hpp"

#include "core/runtime/simulation_runtime.hpp"

void AtmosphericLevelRuntimeState::bind(SimulationStateView& state)
{
    state.boundary_condition_runtime = &boundary_conditions;
    state.radiation_runtime = &radiation;
    state.turbulence_runtime = &turbulence;
    state.boundary_layer_runtime = &boundary_layer;
    state.chaos_runtime = &chaos;
    state.rayleigh_runtime = &rayleigh;
    state.microphysics_runtime = &microphysics;
    state.terrain_runtime = &terrain;
    state.radar_runtime = &radar;
    state.advection_runtime = &advection;
    state.diffusion_runtime = &diffusion;
    state.time_stepping_runtime = &time_stepping;
    state.dynamics_runtime = &dynamics;
}

void SimulationRuntimeState::bind(SimulationStateView& state)
{
    AtmosphericLevelRuntimeState::bind(state);
    state.compute_runtime = &compute;
    state.parity_runtime = &parity;
    state.runtime_owner = this;
}

GridStateView make_grid_state_view(GridStorageState& grid)
{
    return {grid.nr, grid.nth, grid.nz, grid.dx, grid.dz, grid.dt, grid.dy,
            grid.geometry};
}

namespace
{

template <typename Storage>
SimulationStateView make_level_field_view(Storage& storage)
{
    SimulationStateView state{
        make_grid_state_view(storage.grid),
        std::apply([&](const auto&... member) {
            return PrognosticFieldsView{member.get(storage.fields)...};
        }, tmv::fields::prognostic),
        {storage.base.rho0, storage.base.p0, storage.base.qv0,
         storage.base.theta0, storage.base.u0, storage.base.v0},
        {storage.tendencies.dtheta_dt_rad,
         storage.tendencies.dtheta_dt_pbl,
         storage.tendencies.dqv_dt_pbl,
         storage.tendencies.du_dt_pbl,
         storage.tendencies.dv_dt_pbl,
         storage.tendencies.dtke_dt_pbl,
         &storage.tendencies.dqc_dt_pbl,
         &storage.tendencies.dqi_dt_pbl},
        storage.coriolis_f,
        storage.time_s,
        {&storage.surface.ustar, &storage.surface.tau_u,
         &storage.surface.tau_v, &storage.surface.sensible_heat_flux,
         &storage.surface.latent_heat_flux, &storage.surface.moisture_flux,
         &storage.surface.pbl_height,
         &storage.surface.precipitation_rate_mm_h,
         &storage.surface.accumulated_rainfall_mm,
        &storage.surface.accumulated_rainfall_residual_mm,
         &storage.surface.accumulated_rainfall_last_update_s}};
    state.diagnostics = &storage.diagnostics;
    state.process_grid_metrics = &storage.process_grid_metrics;
    // Derived pointers must follow the owner after a move, copy or rollback.
    if (storage.process_grid_metrics.terrain_metrics_active ||
        storage.process_grid_metrics.terrain_metrics != nullptr)
    {
        storage.process_grid_metrics.terrain_metrics = &storage.runtime.terrain.metrics;
        storage.process_grid_metrics.terrain_topography = &storage.runtime.terrain.topography;
    }
    return state;
}

} // namespace

SimulationStateView make_simulation_state_view(
    AtmosphericLevelStorage& storage,
    RuntimeConfigurationState& run_configuration,
    ComputeBackendRuntimeState* shared_compute,
    nimbus::parity::ParityRuntimeState* shared_parity)
{
    auto state = make_level_field_view(storage);
    storage.runtime.bind(state);
    state.configuration = &run_configuration;
    state.compute_runtime = shared_compute;
    state.parity_runtime = shared_parity;
    state.domain_frame = &storage.domain_frame;
    state.budget_level = tmv::BudgetLevel::Child;
    return state;
}

SimulationStateView make_simulation_state_view(SimulationStorageState& storage)
{
    auto state = make_level_field_view(storage);
    storage.runtime.bind(state);
    state.configuration = &storage.configuration;
    state.domain_frame = &storage.configuration.domain_frame;
    state.storage_owner = &storage;
    return state;
}

SimulationStateView global_simulation_state()
{
    return make_simulation_state_view(global_simulation_storage);
}

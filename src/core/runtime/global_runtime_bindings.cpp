#include "core/runtime/simulation_state.hpp"

// Legacy process-wide module handles. New code binds caller-owned runtime
// state through SimulationStateView; these references preserve old call sites.
ComputeBackendRuntimeState& global_compute_backend_runtime_state =
    global_simulation_storage.runtime.compute;
ComputeBackendConfig& global_compute_backend_config =
    global_compute_backend_runtime_state.config;
nimbus::parity::ParityRuntimeState& nimbus::parity::global_parity_runtime_state =
    global_simulation_storage.runtime.parity;
tmv::BoundaryConditionRuntimeState&
tmv::global_boundary_condition_runtime_state =
    global_simulation_storage.runtime.boundary_conditions;
tmv::RadiationRuntimeState& tmv::global_radiation_runtime_state =
    global_simulation_storage.runtime.radiation;
tmv::TurbulenceRuntimeState& tmv::global_turbulence_runtime_state =
    global_simulation_storage.runtime.turbulence;
tmv::BoundaryLayerRuntimeState& tmv::global_boundary_layer_runtime_state =
    global_simulation_storage.runtime.boundary_layer;
tmv::chaos::ChaosRuntimeState& tmv::chaos::global_chaos_runtime_state =
    global_simulation_storage.runtime.chaos;
RayleighDampingRuntimeState& global_rayleigh_runtime_state =
    global_simulation_storage.runtime.rayleigh;
tmv::MicrophysicsRuntimeState& tmv::global_microphysics_runtime_state =
    global_simulation_storage.runtime.microphysics;
tmv::TerrainRuntimeState& tmv::global_terrain_runtime_state =
    global_simulation_storage.runtime.terrain;
tmv::RadarRuntimeState& tmv::global_radar_runtime_state =
    global_simulation_storage.runtime.radar;
tmv::AdvectionRuntimeState& tmv::global_advection_runtime_state =
    global_simulation_storage.runtime.advection;
tmv::DiffusionRuntimeState& tmv::global_diffusion_runtime_state =
    global_simulation_storage.runtime.diffusion;
tmv::TimeSteppingRuntimeState& tmv::global_time_stepping_runtime_state =
    global_simulation_storage.runtime.time_stepping;
tmv::DynamicsRuntimeState& tmv::global_dynamics_runtime_state =
    global_simulation_storage.runtime.dynamics;
std::unique_ptr<tmv::DynamicsScheme>& tmv::dynamics_scheme =
    tmv::global_dynamics_runtime_state.scheme;

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "boundary_conditions/runtime_state.hpp"
#include "boundary_layer/runtime.hpp"
#include "chaos/runtime.hpp"
#include "compute/compute_backend.hpp"
#include "compute/parity_check.hpp"
#include "core/infra/domain_motion.hpp"
#include "core/infra/domain_recycling_provenance.hpp"
#include "core/infra/rayleigh_damping.hpp"
#include "diagnostics/storm_tracker.hpp"
#include "dynamics/runtime.hpp"
#include "microphysics/runtime.hpp"
#include "numerics/advection/advection.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "radiation/runtime.hpp"
#include "radar/runtime.hpp"
#include "terrain/runtime.hpp"
#include "turbulence/runtime.hpp"

struct SimulationStateView;

// Persistent module objects and caches that belong to one atmospheric level.
struct AtmosphericLevelRuntimeState
{
    tmv::BoundaryConditionRuntimeState     boundary_conditions;
    tmv::RadiationRuntimeState             radiation;
    tmv::TurbulenceRuntimeState            turbulence;
    tmv::BoundaryLayerRuntimeState         boundary_layer;
    tmv::chaos::ChaosRuntimeState          chaos;
    RayleighDampingRuntimeState            rayleigh;
    tmv::MicrophysicsRuntimeState          microphysics;
    tmv::TerrainRuntimeState               terrain;
    tmv::RadarRuntimeState                 radar;
    tmv::AdvectionRuntimeState             advection;
    tmv::DiffusionRuntimeState             diffusion;
    tmv::TimeSteppingRuntimeState          time_stepping;
    tmv::DynamicsRuntimeState              dynamics;

    void bind(SimulationStateView& state);
};

// Persistent objects that are shared by, or authoritative for, the whole run.
struct SimulationRuntimeState : AtmosphericLevelRuntimeState
{
    ComputeBackendRuntimeState             compute;
    nimbus::parity::ParityRuntimeState     parity;
    tmv::diagnostics::StormContinuityState storm_tracking;
    tmv::domain::StormWindowMovementState  storm_window_movement;
    bool                                   storm_selected_partition_available = false;
    std::size_t                            storm_selected_partition_nx = 0;
    std::size_t                            storm_selected_partition_ny = 0;
    std::vector<std::uint8_t>              storm_selected_partition;
    tmv::domain::DomainRecyclingTransactionProvenance
        domain_recycling_provenance;

    void bind(SimulationStateView& state);
};

// Deep in-memory checkpoint of persistent module state.
struct SimulationRuntimeSnapshot
{
    SimulationRuntimeState state;
};

inline SimulationRuntimeSnapshot snapshot_simulation_runtime(const SimulationRuntimeState& runtime)
{
    return {runtime};
}

inline void restore_simulation_runtime(SimulationRuntimeState& runtime, const SimulationRuntimeSnapshot& snapshot)
{
    // Copy first so an unsupported third-party scheme leaves the destination
    // unchanged instead of producing a partially restored runtime.
    SimulationRuntimeState restored(snapshot.state);
    runtime = std::move(restored);
}

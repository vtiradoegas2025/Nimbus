#pragma once
#include "core/runtime/held_source_remap.hpp"
class ComputeBackend;
namespace tmv
{
ExtendedRadiationFluxes extend_environment_columns(SimulationStorageState& state);
ExtendedRadiationFluxes extend_environment_columns(SimulationStorageState& state, ComputeBackend& backend);
} // namespace tmv

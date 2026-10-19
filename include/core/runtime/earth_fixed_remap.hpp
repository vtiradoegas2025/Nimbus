#pragma once

#include "core/infra/domain_motion.hpp"
#include "dynamics/completed_step_energy_report.hpp"
#include "terrain/earth_fixed_geometry.hpp"

#include <array>

struct SimulationStorageState;
struct ComputeBackendRuntimeState;

namespace tmv
{

struct EarthFixedRemapReport
{
    MovingMeshInventoryUpdate                dry_mass;
    std::array<MovingMeshInventoryUpdate, 7> water;
    std::array<double, 3>                    initial_momentum{};
    std::array<double, 3>                    final_momentum{};
    std::array<double, 3>                    outward_momentum{};
    std::array<double, 3>                    momentum_storage_rounding{};
    CompletedStepEnergyReportV1              energy;
    double                                   outward_energy_j          = 0.0;
    double                                   energy_storage_rounding_j = 0.0;
    unsigned                                 substeps                  = 0;
};

// Mutates a private timestep candidate; the caller owns atomic publication.
EarthFixedRemapReport remap_earth_fixed_window(SimulationStorageState&              state,
                                               const domain::HorizontalDomainFrame& frame,
                                               double                               begin_time_s,
                                               double                               end_time_s);

EarthFixedRemapReport remap_earth_fixed_window(SimulationStorageState&              state,
                                               const domain::HorizontalDomainFrame& frame,
                                               double                               begin_time_s,
                                               double                               end_time_s,
                                               ComputeBackendRuntimeState&          compute);

} // namespace tmv

#pragma once
#include <array>
#include <vector>
struct SimulationStorageState;
class TerrainComputeKernels;
namespace tmv
{
using ExtendedRadiationFluxes = std::array<std::vector<double>, 4>;
void remap_held_owners(SimulationStorageState&  state,
                       double                   shift_x,
                       double                   shift_y,
                       ExtendedRadiationFluxes& extended);
void remap_held_owners(SimulationStorageState&  state,
                       double                   shift_x,
                       double                   shift_y,
                       ExtendedRadiationFluxes& extended,
                       TerrainComputeKernels&   kernels);
} // namespace tmv

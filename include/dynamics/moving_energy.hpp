#pragma once
#include "terrain/terrain_base.hpp"
#include <vector>
struct PrognosticFieldStorage;
class TerrainComputeKernels;

namespace tmv
{
struct MovingEnergyStorageReport
{
    double requested_energy_j = 0;
    double stored_energy_j    = 0;
    double storage_rounding_j = 0;
};
std::vector<double>       moving_energy_density(const PrognosticFieldStorage& fields, const TerrainMetrics3D& geometry);
std::vector<double>       moving_energy_density(const PrognosticFieldStorage& fields,
                                                const TerrainMetrics3D&       geometry,
                                                TerrainComputeKernels&        kernels);
MovingEnergyStorageReport reconstruct_moving_energy(PrognosticFieldStorage&    fields,
                                                    const TerrainMetrics3D&    geometry,
                                                    const std::vector<double>& target_energy_density);
MovingEnergyStorageReport reconstruct_moving_energy(PrognosticFieldStorage&    fields,
                                                    const TerrainMetrics3D&    geometry,
                                                    const std::vector<double>& target_energy_density,
                                                    TerrainComputeKernels&     kernels);
} // namespace tmv

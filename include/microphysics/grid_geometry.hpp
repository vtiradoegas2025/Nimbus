#pragma once

struct SimulationStateView;

namespace tmv
{
class MicrophysicsScheme;
struct TerrainMetrics3D;

void bind_microphysics_grid_geometry(const SimulationStateView& state,
                                     MicrophysicsScheme&        scheme,
                                     const TerrainMetrics3D*    destination = nullptr);
void prepare_microphysics_prognostics(const SimulationStateView& state, MicrophysicsScheme& scheme);
void bind_microphysics_velocity(const SimulationStateView& state, MicrophysicsScheme& scheme);
} // namespace tmv

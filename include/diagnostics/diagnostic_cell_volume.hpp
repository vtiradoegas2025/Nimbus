#pragma once

#include "core/runtime/grid_state.hpp"
#include "core/infra/physical_grid_metrics.hpp"

/**
 * @file diagnostic_cell_volume.hpp
 * @brief Cell volume for the conservation diagnostics.
 *
 * Every budget integral (mass, energy, momentum, moisture, the
 * microphysics gate) must weight cells by their true volume. On the
 * flat Cartesian grid that is dx * dy * dz (the runtime aliases grid.dx ->
 * dr and grid.dy -> dtheta, both in meters). The SimulationStateView overload
 * uses each terrain column's exact physical interface separation.
 *
 * The legacy `i` (radial index) parameter is retained for call-site stability --
 * the budget and microphysics-gate loops pass it -- but the uniform
 * Cartesian volume does not depend on it.
 */
inline double diagnostic_cell_volume(const GridStateView& grid, [[maybe_unused]] int i)
{
    // dr = dx, dtheta = dy (meters): uniform Cartesian cell volume.
    return grid.dr * grid.dtheta * grid.dz;
}

/** Exact per-cell volume, including active terrain-following layer thickness. */
inline double diagnostic_cell_volume(const SimulationStateView& state,
                                     int i, int j, int k)
{
    return tmv::physical_grid_metrics::cell_volume(state, i, j, k);
}

inline double diagnostic_cell_volume([[maybe_unused]] int i)
{
    return diagnostic_cell_volume(
        GridStateView{NR, NTH, NZ, dr, dz, dt, dtheta, global_grid_geometry}, i);
}

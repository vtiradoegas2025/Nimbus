#pragma once

#include "core/infra/grid_geometry.hpp"

/**
 * @file grid_state.hpp
 * @brief Narrow declarations for process-wide grid dimensions and spacing.
 *
 * This is an intermediate adapter for finding 2: headers that only need grid
 * shape or spacing can include this instead of the full simulation API.
 */

extern int& NR;
extern int& NTH;
extern int& NZ;

extern double& dr;
extern double& dz;
extern double& dt;
extern double& dtheta;

extern GridGeometry& global_grid_geometry;

struct GridStateView
{
    int& nr;
    int& nth;
    int& nz;
    double& dr;
    double& dz;
    double& dt;
    double& dtheta;
    GridGeometry& geometry;
};

#pragma once

#include "turbulence/turbulence_base.hpp"

namespace tmv::collocated_eddy_viscosity
{

// Cartesian A-grid operators. Flat and mapped branches restrict every read and
// write to the PhysicalCellDomain; the six closure planes are never stencil
// state.
double directional_strain_sq(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i,
    int j,
    int k,
    const std::string& mode,
    bool periodic_x,
    bool periodic_y);

double flat_shear_transfer(const TurbulenceStateView& state,
                           const GridMetrics& grid, const Field3D& viscosity,
                           int i, int j, int k, const std::string& mode);

void momentum_tendencies(
    const TurbulenceConfig& config,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& viscosity,
    Field3D& du_dt,
    Field3D& dv_dt,
    Field3D& dw_dt,
    bool periodic_x,
    bool periodic_y);

void scalar_tendency(
    const TurbulenceConfig& config,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& diffusivity,
    const Field3D& scalar,
    Field3D& tendency,
    bool periodic_x,
    bool periodic_y);

} // namespace tmv::collocated_eddy_viscosity

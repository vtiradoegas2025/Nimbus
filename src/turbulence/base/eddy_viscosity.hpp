/**
 * @file eddy_viscosity.hpp
 * @brief Declarations for the turbulence module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the turbulence runtime and scheme implementations.
 * This file is part of the src/turbulence subsystem.
 */

#pragma once
#include <vector>
#include "turbulence/turbulence_base.hpp"
#include "core/field/field3d.hpp"

namespace eddy_viscosity 
{

/**
 * @brief Components and magnitude of the local 3D strain-rate tensor.
 */
struct StrainRate 
{
    double S11, S12, S13;
    double S21, S22, S23;
    double S31, S32, S33;
    double magnitude;
};

/**
 * @brief Computes the 3D strain rate tensor at a grid point.
 */
StrainRate compute_strain_rate_3d(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k
);

/**
 * @brief Computes the filter width (Smagorinsky mixing length).
 */
double compute_filter_width(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    int i, int j, int k
);

/**
 * @brief Computes the Smagorinsky viscosity.
 */
double compute_smagorinsky_viscosity(
    double Cs,
    double Delta,
    double strain_mag,
    double stability_factor = 1.0
);

/**
 * @brief Computes the eddy diffusivities from the eddy viscosity.
 */
void compute_eddy_diffusivities(
    double nu_t,
    double Pr_t,
    double Sc_t,
    double& K_theta,
    double& K_q,
    double& K_tke
);

/**
 * @brief Computes the scalar diffusion tendency.
 */
double compute_scalar_diffusion_tendency(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& K_field,
    const Field3D& phi,
    int i, int j, int k,
    int var_index
);

/**
 * @brief Computes the momentum diffusion tendencies.
 */
void compute_momentum_diffusion_tendencies(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& nu_t,
    Field3D& dudt_sgs,
    Field3D& dvdt_sgs,
    Field3D& dwdt_sgs
);

/**
 * @brief Computes the scalar diffusion tendencies.
 */
void compute_scalar_diffusion_tendencies(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& K_field,
    const Field3D& phi,
    Field3D& dphi_dt_sgs
);

/**
 * @brief Computes the stability correction.
 */
double stability_correction_ri(
    double Ri,
    double Ri_crit = 0.25
);

/**
 * @brief Computes the TKE mixing length.
 */
double compute_tke_mixing_length(
    double Delta,
    double e,
    double N,
    double c_l = 0.15
);

/**
 * @brief Computes the Brunt-Väisälä frequency.
 */
double compute_brunt_vaisala_frequency(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k
);

/**
 * @brief Legacy overload with fixed vertical spacing assumptions.
 */
double compute_brunt_vaisala_frequency(
    const TurbulenceStateView& state,
    int i, int j, int k
);

/**
 * @brief Applies the positivity limits.
 */
void apply_positivity_limits(
    Field3D& field,
    double min_value = 0.0,
    double max_value = 1e6
);

}

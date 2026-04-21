/**
 * @file surface_fluxes.hpp
 * @brief Declarations for the boundary_layer module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the boundary_layer runtime and scheme implementations.
 * This file is part of the src/boundary_layer subsystem.
 */

#pragma once
#include <vector>
#include "boundary_layer/boundary_layer_base.hpp"


namespace surface_fluxes 
{

/**
 * @brief Bulk aerodynamic surface flux outputs.
 */
struct BulkFluxes 
{
    double ustar = 0.0;
    double tau_u = 0.0;
    double tau_v = 0.0;
    double H = 0.0;
    double E = 0.0;
    double Cd = 0.0;
    double Ch = 0.0;
    double Ce = 0.0;
};

/**
 * @brief Computes surface fluxes using bulk transfer coefficients.
 */
BulkFluxes compute_bulk_fluxes(
    const SurfaceConfig& sfc_cfg,
    double u_sfc,
    double v_sfc,
    double theta_sfc,
    double qv_sfc,
    double theta_air,
    double qv_air,
    double p_air,
    double rho_air,
    double z_sfc,
    double min_ustar = 1.0e-4
);

/**
 * @brief Monin-Obukhov surface flux outputs and stability diagnostics.
 */
struct MoninObukhovFluxes 
{
    double ustar = 0.0;
    double tau_u = 0.0;
    double tau_v = 0.0;
    double H = 0.0;
    double E = 0.0;
    double L = 0.0;
    double zeta = 0.0;
    double Cd = 0.0;
    double Ch = 0.0;
    double Ce = 0.0;
};

/**
 * @brief Computes fluxes using Monin-Obukhov similarity theory.
 */
MoninObukhovFluxes compute_monin_obukhov_fluxes(
    const SurfaceConfig& sfc_cfg,
    double u_sfc,
    double v_sfc,
    double theta_sfc,
    double qv_sfc,
    double theta_air,
    double qv_air,
    double p_air,
    double rho_air,
    double z_sfc,
    double min_ustar = 1.0e-4
);

/**
 * @brief Dispatches the configured surface-flux formulation.
 */
BulkFluxes compute_surface_fluxes(
    const BoundaryLayerConfig& bl_cfg,
    const SurfaceConfig& sfc_cfg,
    double u_sfc,
    double v_sfc,
    double theta_sfc,
    double qv_sfc,
    double theta_air,
    double qv_air,
    double p_air,
    double rho_air,
    double z_sfc
);

/**
 * @brief Stability correction function for momentum.
 */
double psi_m(double zeta);
/**
 * @brief Stability correction function for heat and moisture.
 */
double psi_h(double zeta);

/**
 * @brief Computes bulk Richardson number between surface and first model level.
 */
double bulk_richardson_number(
    double theta_sfc,
    double theta_air,
    double u_sfc,
    double v_sfc,
    double z_sfc
);

/**
 * @brief Computes virtual potential temperature from theta and qv.
 */
double virtual_potential_temperature(double theta, double qv);

/**
 * @brief Converts theta tendency to temperature tendency along a profile.
 */
void convert_theta_tendency_to_temperature(
    const std::vector<double>& dthetadt,
    const std::vector<double>& theta,
    const std::vector<double>& p,
    std::vector<double>& dTdt
);
}

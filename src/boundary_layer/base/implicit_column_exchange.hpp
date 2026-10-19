#pragma once

#include "boundary_layer/base/surface_fluxes.hpp"
#include "boundary_layer/base/vertical_diffusion.hpp"
#include "core/infra/physical_constants.hpp"

namespace tmv::boundary_layer
{
inline void mix_column_surface_exchange(
    const BoundaryLayerColumnStateView& col,
    const std::vector<double>& K_m, const std::vector<double>& K_h,
    surface_fluxes::BulkFluxes& fluxes, double interval_s,
    BoundaryLayerTendencies& tend,
    const std::vector<double>* u_flux = nullptr,
    const std::vector<double>* v_flux = nullptr,
    const std::vector<double>* theta_flux = nullptr,
    const std::vector<double>* vapor_flux = nullptr)
{
    const std::size_t nz = col.theta.size();
    const double exner = std::pow(col.p[0] / physical_constants::reference_pressure_pa,
        physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk);
    const double theta_surface_flux = fluxes.H / (boundary_layer_constants::cp * exner);
    const double hx = col.surface_slope_x, hy = col.surface_slope_y;
    const double slope = std::hypot(hx, hy);
    const double cosine = slope > 0.0 ? hx / slope : 1.0;
    const double sine = slope > 0.0 ? hy / slope : 0.0;
    const double wind = std::max(0.1, std::sqrt(
        col.u_sfc * col.u_sfc + col.v_sfc * col.v_sfc
        + std::pow(hx * col.u_sfc + hy * col.v_sfc, 2)));
    const double exchange = std::sqrt(1.0 + slope * slope) * col.rho[0] * wind;
    const double drag = exchange * fluxes.Cd;
    std::vector<double> parallel(nz), perpendicular(nz),
        parallel_flux(nz + 1), perpendicular_flux(nz + 1),
        parallel_tendency, perpendicular_tendency;
    for (std::size_t k = 0; k < nz; ++k)
    {
        parallel[k] = cosine * col.u[k] + sine * col.v[k];
        perpendicular[k] = -sine * col.u[k] + cosine * col.v[k];
    }
    for (std::size_t face = 0; face <= nz; ++face)
    {
        parallel_flux[face] = cosine * (u_flux ? (*u_flux)[face] : 0.0)
            + sine * (v_flux ? (*v_flux)[face] : 0.0);
        perpendicular_flux[face] = -sine * (u_flux ? (*u_flux)[face] : 0.0)
            + cosine * (v_flux ? (*v_flux)[face] : 0.0);
    }
    double parallel_stress, perpendicular_stress, realized_theta_flux;
    namespace vd = boundary_layer::vdiff;
    // The terrain metric has drag eigenvalues 1+slope^2 and 1. Rotating
    // both momentum columns makes the implicit surface solve scalar.
    vd::diffuse_column_implicit(parallel, K_m, col.z_int, col.rho,
        cosine * fluxes.tau_u + sine * fluxes.tau_v, interval_s,
        parallel_tendency, &parallel_flux, drag * (1.0 + slope * slope),
        &parallel_stress);
    vd::diffuse_column_implicit(perpendicular, K_m, col.z_int, col.rho,
        -sine * fluxes.tau_u + cosine * fluxes.tau_v, interval_s,
        perpendicular_tendency, &perpendicular_flux, drag, &perpendicular_stress);
    tend.dudt_pbl.resize(nz);
    tend.dvdt_pbl.resize(nz);
    for (std::size_t k = 0; k < nz; ++k)
    {
        tend.dudt_pbl[k] = cosine * parallel_tendency[k] - sine * perpendicular_tendency[k];
        tend.dvdt_pbl[k] = sine * parallel_tendency[k] + cosine * perpendicular_tendency[k];
    }
    fluxes.tau_u = cosine * parallel_stress - sine * perpendicular_stress;
    fluxes.tau_v = sine * parallel_stress + cosine * perpendicular_stress;
    vd::diffuse_column_implicit(
        col.theta, K_h, col.z_int, col.rho, theta_surface_flux, interval_s,
        tend.dthetadt_pbl, theta_flux,
        exchange * fluxes.Ch, &realized_theta_flux);
    vd::diffuse_column_implicit(
        col.qv, K_h, col.z_int, col.rho, fluxes.E, interval_s,
        tend.dqvdt_pbl, vapor_flux,
        exchange * fluxes.Ce, &fluxes.E);
    fluxes.H = realized_theta_flux * boundary_layer_constants::cp * exner;
}
}

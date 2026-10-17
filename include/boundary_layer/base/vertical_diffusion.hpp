#pragma once

/**
 * @file vertical_diffusion.hpp
 * @brief Shared conservative vertical-diffusion operator for PBL schemes.
 *
 * One implementation of the vertical turbulent-mixing operator and its surface
 * (lower-boundary) flux condition, so every boundary-layer scheme injects the
 * surface flux and conserves a well-mixed scalar in exactly the same, correct
 * way. Previously each scheme rolled its own: YSU hardcoded a zero surface flux
 * and MYNN used a one-sided, non-conservative stencil.
 */

#include "boundary_layer/boundary_layer_base.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace tmv::boundary_layer
{
namespace vdiff
{

/**
 * @brief Conservative, mass-weighted vertical diffusion of one column variable
 *        with a prescribed surface turbulent mass flux.
 *
 * Solves the flux-divergence form
 *     rho dC/dt = -dJ/dz,   J = -rho K dC/dz   (J = upward turbulent mass flux),
 * with the bottom-interface flux set to the surface flux and zero flux through
 * the rigid lid. This is conservative (the column integral of rho*C changes only
 * by the surface flux) and a well-mixed column (uniform C) produces no tendency.
 *
 * @param profile        cell-centered values, size nz.
 * @param K              diffusivity at each interface, size nz+1 (K[i] at z_int[i]).
 * @param z_int          interface heights, size nz+1 (monotone increasing).
 * @param rho            cell-centered density, size nz.
 * @param sfc_mass_flux  upward surface turbulent mass flux rho*w'c' at the ground
 *                       (H/(cp*Exner_sfc) for theta, E for qv,
 *                       tau_u/tau_v for momentum); pass
 *                       0 when surface fluxes are disabled.
 * @param tend           output dC/dt, resized to nz.
 */
inline void diffuse_column_with_internal_flux(
    const BoundaryLayerProfileView& profile,
    const std::vector<double>& K,
    const BoundaryLayerProfileView& z_int,
    const BoundaryLayerProfileView& rho,
    double sfc_mass_flux,
    const std::vector<double>* internal_mass_flux,
    std::vector<double>& tend)
{
    const std::size_t nz = profile.size();
    tend.assign(nz, 0.0);
    if (nz == 0)
    {
        return;
    }
    if (K.size() != nz + 1 || z_int.size() != nz + 1 || rho.size() != nz)
    {
        throw std::invalid_argument(
            "vertical diffusion profile, density, interface, and K shapes disagree");
    }
    if (internal_mass_flux != nullptr && internal_mass_flux->size() != nz + 1)
    {
        throw std::invalid_argument(
            "vertical diffusion internal-flux shape disagrees with the column");
    }
    if (!std::isfinite(sfc_mass_flux))
    {
        throw std::invalid_argument(
            "vertical diffusion surface flux must be finite");
    }
    for (std::size_t face = 0; face <= nz; ++face)
    {
        if (!std::isfinite(z_int[face]) || !std::isfinite(K[face]) ||
            K[face] < 0.0)
        {
            throw std::invalid_argument(
                "vertical diffusion requires finite interfaces and finite non-negative diffusivity");
        }
        if (internal_mass_flux != nullptr
            && !std::isfinite((*internal_mass_flux)[face]))
        {
            throw std::invalid_argument(
                "vertical diffusion internal flux must be finite");
        }
        if (face > 0 && !(z_int[face] > z_int[face - 1]))
        {
            throw std::invalid_argument(
                "vertical diffusion interfaces must increase strictly");
        }
    }
    for (std::size_t k = 0; k < nz; ++k)
    {
        if (!std::isfinite(profile[k]) || !std::isfinite(rho[k]) ||
            !(rho[k] > 0.0))
        {
            throw std::invalid_argument(
                "vertical diffusion requires finite profiles and positive finite density");
        }
    }

    const double sfc = sfc_mass_flux;

    auto z_center = [&](std::size_t k) { return 0.5 * (z_int[k] + z_int[k + 1]); };

    for (std::size_t k = 0; k < nz; ++k)
    {
        const double dz_k = z_int[k + 1] - z_int[k];
        const double rho_k = rho[k];

        // Upper interface mass flux J_{k+1/2} = -rho K dC/dz; zero through the lid.
        double J_up = 0.0;
        if (k + 1 < nz)
        {
            const double rho_up = 0.5 * (rho[k] + rho[k + 1]);
            const double dz_c = z_center(k + 1) - z_center(k);
            J_up = -rho_up * K[k + 1] *
                   (profile[k + 1] - profile[k]) / dz_c;
            if (internal_mass_flux != nullptr)
                J_up += (*internal_mass_flux)[k + 1];
        }

        // Lower interface mass flux J_{k-1/2}; at the ground it IS the surface flux.
        double J_dn;
        if (k == 0)
        {
            J_dn = sfc;
        }
        else
        {
            const double rho_dn = 0.5 * (rho[k - 1] + rho[k]);
            const double dz_c = z_center(k) - z_center(k - 1);
            J_dn = -rho_dn * K[k] *
                   (profile[k] - profile[k - 1]) / dz_c;
            if (internal_mass_flux != nullptr)
                J_dn += (*internal_mass_flux)[k];
        }

        tend[k] = -(J_up - J_dn) / (rho_k * dz_k);
        if (!std::isfinite(tend[k]))
        {
            throw std::runtime_error(
                "vertical diffusion produced a non-finite tendency");
        }
    }
}

inline void diffuse_column(const BoundaryLayerProfileView& profile,
                           const std::vector<double>& K,
                           const BoundaryLayerProfileView& z_int,
                           const BoundaryLayerProfileView& rho,
                           double sfc_mass_flux,
                           std::vector<double>& tend)
{
    diffuse_column_with_internal_flux(
        profile, K, z_int, rho, sfc_mass_flux, nullptr, tend);
}

inline void diffuse_column_implicit(const BoundaryLayerProfileView& profile,
                                   const std::vector<double>& K,
                                   const BoundaryLayerProfileView& z_int,
                                   const BoundaryLayerProfileView& rho,
                                   double sfc_mass_flux, double interval_s,
                                   std::vector<double>& tend,
                                   const std::vector<double>* internal_mass_flux = nullptr,
                                   double surface_conductance = 0.0,
                                   double* realized_surface_flux = nullptr)
{
    if (!std::isfinite(interval_s) || interval_s <= 0.0)
        throw std::invalid_argument("implicit column mixing requires a positive finite interval");
    if (!std::isfinite(surface_conductance) || surface_conductance < 0.0)
        throw std::invalid_argument("implicit surface exchange requires non-negative finite conductance");
    diffuse_column_with_internal_flux(
        profile, K, z_int, rho, sfc_mass_flux, internal_mass_flux, tend);
    const std::size_t n = profile.size();
    if (n == 0) return;
    std::vector<double> mass(n), conductance(n + 1, 0.0), upper(n), increment(n);
    for (std::size_t k = 0; k < n; ++k)
    {
        mass[k] = rho[k] * (z_int[k + 1] - z_int[k]);
        if (k > 0)
            conductance[k] = 0.5 * (rho[k - 1] + rho[k]) * K[k]
                / (0.5 * (z_int[k + 1] - z_int[k - 1]));
    }
    // Solve increments to avoid subtracting nearly equal absolute theta
    // values. Publish shared face fluxes, so roundoff cannot create a source.
    for (std::size_t k = 0; k < n; ++k)
    {
        const double lower = -interval_s * conductance[k];
        const double above = -interval_s * conductance[k + 1];
        const double diagonal = mass[k] - lower - above
            + (k == 0 ? interval_s * surface_conductance : 0.0);
        const double pivot = diagonal - (k > 0 ? lower * upper[k - 1] : 0.0);
        const double rhs = interval_s * mass[k] * tend[k]
            - (k > 0 ? lower * increment[k - 1] : 0.0);
        if (!std::isfinite(pivot) || pivot <= 0.0 || !std::isfinite(rhs))
            throw std::runtime_error("implicit column mixing has an unrepresentable solve");
        upper[k] = above / pivot;
        increment[k] = rhs / pivot;
    }
    for (std::size_t k = n - 1; k > 0; --k)
        increment[k - 1] -= upper[k - 1] * increment[k];
    double lower_flux = sfc_mass_flux - surface_conductance * increment[0];
    if (realized_surface_flux) *realized_surface_flux = lower_flux;
    for (std::size_t k = 0; k < n; ++k)
    {
        const double upper_flux = k + 1 < n
            ? -conductance[k + 1] * ((profile[k + 1] - profile[k])
                + (increment[k + 1] - increment[k]))
                + (internal_mass_flux ? (*internal_mass_flux)[k + 1] : 0.0) : 0.0;
        tend[k] = (lower_flux - upper_flux) / mass[k];
        if (!std::isfinite(tend[k]))
            throw std::runtime_error("implicit column mixing produced a non-finite flux divergence");
        lower_flux = upper_flux;
    }
}

}  // namespace vdiff
}  // namespace tmv::boundary_layer

#ifndef NIMBUS_BOUNDARY_LAYER_NAMESPACE_ALIAS
#define NIMBUS_BOUNDARY_LAYER_NAMESPACE_ALIAS
namespace boundary_layer = tmv::boundary_layer;
#endif

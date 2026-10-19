#pragma once

/**
 * @file pbl_height.hpp
 * @brief Shared Richardson, potential-temperature, and TKE PBL-height diagnostics.
 */

#include "boundary_layer/base/surface_fluxes.hpp"   // virtual_potential_temperature
#include "core/infra/physical_constants.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace tmv::boundary_layer
{
namespace pbl
{

inline constexpr double theta_increase_threshold_k = 1.5;
inline constexpr double tke_threshold_floor_m2s2 = 0.01;

inline double clamp_height(double value, double min_h, double max_h)
{
    return std::max(min_h, std::min(max_h, value));
}

inline double interpolate_threshold_height(double lower_value,
                                           double upper_value,
                                           double threshold,
                                           double lower_height,
                                           double upper_height)
{
    if (!(upper_value > lower_value))
        return upper_height;
    const double fraction =
        (threshold - lower_value) / (upper_value - lower_value);
    return lower_height + std::clamp(fraction, 0.0, 1.0)
        * (upper_height - lower_height);
}

/**
 * @brief Diagnose the PBL top via the bulk-Richardson method
 *        (Vogelezang & Holtslag 1996; Troen & Mahrt 1986).
 *
 * Walks up the column and finds the first height where the bulk Richardson
 * number relative to the lowest model level
 *     Ri_b(z) = g (theta_v(z) - theta_v(z1)) (z - z1)
 *               / ( theta_v(z1) * |U(z) - U(z1)|^2 )
 * reaches Ri_crit, then linearly interpolates the crossing. Uses VIRTUAL
 * potential temperature and the bulk wind shear relative to the lowest level.
 *
 * @param theta   cell-centered potential temperature, size nz.
 * @param qv      cell-centered water-vapor mixing ratio, size nz.
 * @param u,v     cell-centered horizontal winds, size nz.
 * @param z_int   interface heights, size nz+1.
 * @param Ri_crit critical bulk Richardson number (~0.25).
 * @param min_h   floor on the returned depth.
 * @param max_h   cap on the returned depth (also returned if no crossing).
 */
inline double diagnose_pbl_height_ri(const BoundaryLayerProfileView& theta,
                                     const BoundaryLayerProfileView& qv,
                                     const BoundaryLayerProfileView& u,
                                     const BoundaryLayerProfileView& v,
                                     const BoundaryLayerProfileView& z_int,
                                     double Ri_crit,
                                     double min_h,
                                     double max_h)
{
    const std::size_t nz = theta.size();
    if (nz < 2 || qv.size() < nz || u.size() < nz || v.size() < nz ||
        z_int.size() < nz + 1)
    {
        return std::max(min_h, std::min(max_h, min_h));
    }

    auto z_center = [&](std::size_t k) { return 0.5 * (z_int[k] + z_int[k + 1]); };

    const double thetav1 =
        surface_fluxes::virtual_potential_temperature(theta[0], qv[0]);
    const double z1 = z_center(0);
    const double u1 = u[0];
    const double v1 = v[0];
    constexpr double k_min_shear2 = 0.01;  // (0.1 m/s)^2 floor on |dU|^2

    double Ri_prev = 0.0;   // Ri = 0 at the reference level
    double z_prev = z1;

    for (std::size_t k = 1; k < nz; ++k)
    {
        const double zk = z_center(k);
        const double thetavk =
            surface_fluxes::virtual_potential_temperature(theta[k], qv[k]);
        const double du = u[k] - u1;
        const double dv = v[k] - v1;
        const double shear2 = std::max(du * du + dv * dv, k_min_shear2);
        const double Ri = physical_constants::gravity_ms2 *
                          (thetavk - thetav1) * (zk - z1) / (thetav1 * shear2);

        if (std::isfinite(Ri) && Ri >= Ri_crit)
        {
            double h = zk;
            if (Ri > Ri_prev)  // interpolate the Ri_crit crossing
            {
                h = z_prev + (zk - z_prev) * (Ri_crit - Ri_prev) / (Ri - Ri_prev);
            }
            return std::max(min_h, std::min(max_h, h));
        }
        Ri_prev = Ri;
        z_prev = zk;
    }

    return max_h;
}

inline double diagnose_pbl_height_theta(
    const BoundaryLayerProfileView& theta,
    const BoundaryLayerProfileView& z_int,
    double threshold_k,
    double min_h,
    double max_h)
{
    const std::size_t nz = theta.size();
    if (nz < 2 || z_int.size() < nz + 1 || !std::isfinite(threshold_k)
        || threshold_k <= 0.0)
        return clamp_height(min_h, min_h, max_h);

    const auto center = [&](std::size_t k)
    {
        return 0.5 * (z_int[k] + z_int[k + 1]);
    };
    double minimum_theta = theta[0];
    std::size_t minimum_index = 0;
    for (std::size_t k = 1; k < nz && center(k) <= 200.0; ++k)
    {
        if (std::isfinite(theta[k]) && theta[k] < minimum_theta)
        {
            minimum_theta = theta[k];
            minimum_index = k;
        }
    }

    double previous_excess = 0.0;
    double previous_height = center(minimum_index);
    for (std::size_t k = minimum_index + 1; k < nz; ++k)
    {
        const double excess = theta[k] - minimum_theta;
        const double height = center(k);
        if (std::isfinite(excess) && excess >= threshold_k)
        {
            return clamp_height(
                interpolate_threshold_height(
                    previous_excess, excess, threshold_k,
                    previous_height, height),
                min_h, max_h);
        }
        previous_excess = excess;
        previous_height = height;
    }
    return max_h;
}

inline double tke_diagnostic_threshold_m2s2(
    const BoundaryLayerProfileView& tke)
{
    if (tke.empty() || !std::isfinite(tke[0]))
        return tke_threshold_floor_m2s2;
    return std::max(tke_threshold_floor_m2s2, 0.05 * std::max(tke[0], 0.0));
}

inline double diagnose_pbl_height_tke(
    const BoundaryLayerProfileView& tke,
    const BoundaryLayerProfileView& z_int,
    double threshold_m2s2,
    double min_h,
    double max_h)
{
    const std::size_t nz = tke.size();
    if (nz < 2 || z_int.size() < nz + 1
        || !std::isfinite(threshold_m2s2) || threshold_m2s2 <= 0.0)
        return clamp_height(min_h, min_h, max_h);

    const auto center = [&](std::size_t k)
    {
        return 0.5 * (z_int[k] + z_int[k + 1]);
    };
    if (tke[0] <= threshold_m2s2)
        return clamp_height(center(0), min_h, max_h);

    for (std::size_t k = 1; k < nz; ++k)
    {
        if (std::isfinite(tke[k]) && tke[k] <= threshold_m2s2)
        {
            const double lower = tke[k - 1];
            const double upper = tke[k];
            double height = center(k);
            if (std::isfinite(lower) && lower > upper)
            {
                const double fraction = std::clamp(
                    (lower - threshold_m2s2) / (lower - upper), 0.0, 1.0);
                height = center(k - 1)
                    + fraction * (center(k) - center(k - 1));
            }
            return clamp_height(height, min_h, max_h);
        }
    }
    return max_h;
}

}  // namespace pbl
}  // namespace tmv::boundary_layer

#ifndef NIMBUS_BOUNDARY_LAYER_NAMESPACE_ALIAS
#define NIMBUS_BOUNDARY_LAYER_NAMESPACE_ALIAS
namespace boundary_layer = tmv::boundary_layer;
#endif

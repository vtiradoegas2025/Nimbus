#pragma once

#include <algorithm>
#include <cmath>

#include "physics/terrain_base.hpp"
#include "physics/turbulence_base.hpp"

/**
 * @file grid_metric_utils.hpp
 * @brief Helpers for local metric spacing in terrain-following grids.
 *
 * Provides robust spacing utilities used by numerics and turbulence code.
 * Functions automatically fall back to Cartesian spacing when terrain
 * metrics are unavailable or invalid.
 */

namespace grid_metric
{
/**
 * @brief Returns a positive finite value or a fallback.
 */
inline double sanitize_positive(double value, double fallback = 1.0)
{
    if (!std::isfinite(value) || value <= 1.0e-9)
    {
        return fallback;
    }
    return value;
}

/**
 * @brief Checks whether terrain metrics are present and initialized.
 */
inline bool has_terrain_metrics(const GridMetrics& grid)
{
    if (!grid.terrain_metrics_active || grid.terrain_metrics == nullptr)
    {
        return false;
    }

    const TerrainMetrics3D& tm = *grid.terrain_metrics;
    return tm.z.size_r() > 0 && tm.z.size_th() > 0 && tm.z.size_z() > 0;
}

/**
 * @brief Computes local vertical spacing near a grid point.
 */
inline double local_dz(const GridMetrics& grid, int i, int j, int k, int nz_hint)
{
    const double fallback = (!grid.dz.empty() && k >= 0 && k < static_cast<int>(grid.dz.size()))
                                ? sanitize_positive(grid.dz[k], 1.0)
                                : 1.0;

    if (!has_terrain_metrics(grid))
    {
        return fallback;
    }

    const TerrainMetrics3D& tm = *grid.terrain_metrics;
    const int nr = tm.z.size_r();
    const int nth = tm.z.size_th();
    const int nz = (nz_hint > 0) ? std::min(nz_hint, tm.z.size_z()) : tm.z.size_z();

    if (i < 0 || i >= nr || j < 0 || j >= nth || k < 0 || k >= nz || nz < 2)
    {
        return fallback;
    }

    auto delta = [&](int k0, int k1) -> double
    {
        const double z0 = tm.z(i, j, k0);
        const double z1 = tm.z(i, j, k1);
        const double dz = std::abs(z1 - z0);
        return sanitize_positive(dz, fallback);
    };

    if (k == 0)
    {
        return delta(0, 1);
    }
    if (k == nz - 1)
    {
        return delta(nz - 2, nz - 1);
    }

    const double up = delta(k - 1, k);
    const double down = delta(k, k + 1);
    return sanitize_positive(0.5 * (up + down), fallback);
}

/**
 * @brief Computes centered vertical span for gradient stencils.
 */
inline double centered_dz_span(const GridMetrics& grid, int i, int j, int k, int nz_hint)
{
    const double fallback = 2.0 * local_dz(grid, i, j, k, nz_hint);

    if (!has_terrain_metrics(grid))
    {
        return fallback;
    }

    const TerrainMetrics3D& tm = *grid.terrain_metrics;
    const int nr = tm.z.size_r();
    const int nth = tm.z.size_th();
    const int nz = (nz_hint > 0) ? std::min(nz_hint, tm.z.size_z()) : tm.z.size_z();

    if (i < 0 || i >= nr || j < 0 || j >= nth || k <= 0 || k >= nz - 1)
    {
        return fallback;
    }

    const double z_up = tm.z(i, j, k + 1);
    const double z_down = tm.z(i, j, k - 1);
    return sanitize_positive(std::abs(z_up - z_down), fallback);
}

/**
 * @brief Computes local effective x-spacing including terrain stretching.
 */
inline double local_dx(const GridMetrics& grid, int i, int j, int k)
{
    const double base_dx = sanitize_positive(grid.dx, 1.0);

    if (!has_terrain_metrics(grid))
    {
        return base_dx;
    }

    const TerrainMetrics3D& tm = *grid.terrain_metrics;
    if (i < 0 || i >= tm.J.size_r() || j < 0 || j >= tm.J.size_th() || k < 0 || k >= tm.J.size_z())
    {
        return base_dx;
    }

    const double J = tm.J(i, j, k);
    const double mx = tm.mx(i, j, k);
    if (!std::isfinite(J) || !std::isfinite(mx))
    {
        return base_dx;
    }

    // Terrain metric relation: dz/dx = -mx * J.
    const double dzdx = -mx * J;
    const double stretch = std::sqrt(1.0 + dzdx * dzdx);
    return sanitize_positive(base_dx * stretch, base_dx);
}

/**
 * @brief Computes local effective y-spacing including terrain stretching.
 */
inline double local_dy(const GridMetrics& grid, int i, int j, int k)
{
    const double base_dy = sanitize_positive(grid.dy, 1.0);

    if (!has_terrain_metrics(grid))
    {
        return base_dy;
    }

    const TerrainMetrics3D& tm = *grid.terrain_metrics;
    if (i < 0 || i >= tm.J.size_r() || j < 0 || j >= tm.J.size_th() || k < 0 || k >= tm.J.size_z())
    {
        return base_dy;
    }

    const double J = tm.J(i, j, k);
    const double my = tm.my(i, j, k);
    if (!std::isfinite(J) || !std::isfinite(my))
    {
        return base_dy;
    }

    // Terrain metric relation: dz/dy = -my * J.
    const double dzdy = -my * J;
    const double stretch = std::sqrt(1.0 + dzdy * dzdy);
    return sanitize_positive(base_dy * stretch, base_dy);
}
} // namespace grid_metric

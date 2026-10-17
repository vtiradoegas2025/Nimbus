#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "terrain/terrain_base.hpp"
#include "turbulence/turbulence_base.hpp"

/**
 * @file grid_metric_utils.hpp
 * @brief Helpers for local metric spacing in terrain-following grids.
 *
 * Provides fail-closed spacing utilities used by numerics and turbulence.
 * Flat grids use their declared Cartesian spacing. Active terrain requires a
 * complete, finite metric field; corruption is never converted into a flat
 * grid or a one-metre cell.
 */

namespace grid_metric
{
inline void validate_vertical_metrics(const GridMetrics& grid,
                                      int nr, int nth, int nz)
{
    if (nr <= 0 || nth <= 0 || nz <= 0
        || grid.dz.size() != static_cast<std::size_t>(nz))
    {
        throw std::invalid_argument(
            "vertical metrics must match the non-empty field shape");
    }
    for (double spacing : grid.dz)
    {
        if (!std::isfinite(spacing) || spacing <= 0.0)
        {
            throw std::invalid_argument(
                "vertical metric spacing must be positive and finite");
        }
    }
    if (!grid.terrain_metrics_active)
    {
        return;
    }
    if (grid.terrain_metrics == nullptr)
    {
        throw std::invalid_argument(
            "active terrain metrics require a metric field");
    }
    const TerrainMetrics3D& metrics = *grid.terrain_metrics;
    if (metrics.z.size_r() != nr || metrics.z.size_th() != nth
        || metrics.z.size_z() != nz)
    {
        throw std::invalid_argument(
            "terrain vertical metrics do not match the field shape");
    }
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            double previous = 0.0;
            for (int k = 0; k < nz; ++k)
            {
                const double height = metrics.z(i, j, k);
                if (!std::isfinite(height)
                    || (k > 0 && height <= previous))
                {
                    throw std::invalid_argument(
                        "terrain scalar heights must be finite and strictly increasing");
                }
                previous = height;
            }
        }
    }
}

/**
 * @brief Returns a positive finite value or throws.
 */
inline double sanitize_positive(double value, [[maybe_unused]] double fallback = 1.0)
{
    if (!std::isfinite(value) || value <= 1.0e-9)
    {
        throw std::invalid_argument(
            "grid metric spacing must be positive and finite");
    }
    return value;
}

/**
 * @brief Checks whether terrain metrics are present and initialized.
 */
inline bool has_terrain_metrics(const GridMetrics& grid)
{
    if (!grid.terrain_metrics_active)
    {
        return false;
    }
    if (grid.terrain_metrics == nullptr)
        throw std::invalid_argument(
            "active terrain metrics require a metric field");

    const TerrainMetrics3D& tm = *grid.terrain_metrics;
    if (tm.z.size_r() <= 0 || tm.z.size_th() <= 0 || tm.z.size_z() <= 0)
        throw std::invalid_argument(
            "active terrain metrics require non-empty scalar heights");
    return true;
}

/**
 * @brief Computes local vertical spacing near a grid point.
 */
inline double local_dz(const GridMetrics& grid, int i, int j, int k, int nz_hint)
{
    if (k < 0 || k >= static_cast<int>(grid.dz.size()))
        throw std::out_of_range(
            "vertical metric index is outside the declared dz field");
    const double fallback = sanitize_positive(grid.dz[k]);

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
        throw std::out_of_range(
            "terrain vertical metric index is outside the metric field");
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

/** Physical distance between scalar centers lower_k and lower_k+1. */
inline double vertical_center_distance(const GridMetrics& grid,
                                       int i, int j, int lower_k,
                                       int nz_hint)
{
    const int nz = nz_hint > 0 ? nz_hint
                               : static_cast<int>(grid.dz.size());
    if (lower_k < 0 || lower_k + 1 >= nz
        || grid.dz.size() != static_cast<std::size_t>(nz))
        throw std::out_of_range(
            "vertical center distance is outside the declared grid");
    if (!has_terrain_metrics(grid))
    {
        return sanitize_positive(0.5 * (
            grid.dz[static_cast<std::size_t>(lower_k)]
          + grid.dz[static_cast<std::size_t>(lower_k + 1)]));
    }
    const TerrainMetrics3D& tm = *grid.terrain_metrics;
    if (i < 0 || i >= tm.z.size_r() || j < 0 || j >= tm.z.size_th()
        || lower_k + 1 >= tm.z.size_z())
        throw std::out_of_range(
            "terrain center distance is outside the metric field");
    return sanitize_positive(
        tm.z(i, j, lower_k + 1) - tm.z(i, j, lower_k));
}

/** Physical thickness of scalar control volume k. */
inline double vertical_cell_thickness(const GridMetrics& grid,
                                      int i, int j, int k,
                                      int nz_hint)
{
    const int nz = nz_hint > 0 ? nz_hint
                               : static_cast<int>(grid.dz.size());
    if (k < 0 || k >= nz
        || grid.dz.size() != static_cast<std::size_t>(nz))
        throw std::out_of_range(
            "vertical cell thickness is outside the declared grid");
    if (!has_terrain_metrics(grid))
        return sanitize_positive(grid.dz[static_cast<std::size_t>(k)]);
    const TerrainMetrics3D& tm = *grid.terrain_metrics;
    if (tm.z_face.size_r() != tm.z.size_r()
        || tm.z_face.size_th() != tm.z.size_th()
        || tm.z_face.size_z() != tm.z.size_z() + 1
        || i < 0 || i >= tm.z_face.size_r()
        || j < 0 || j >= tm.z_face.size_th()
        || k + 1 >= tm.z_face.size_z())
        throw std::invalid_argument(
            "terrain cell thickness requires complete interface geometry");
    return sanitize_positive(
        tm.z_face(i, j, k + 1) - tm.z_face(i, j, k));
}

/**
 * @brief Computes centered vertical span for gradient stencils.
 */
inline double centered_dz_span(const GridMetrics& grid, int i, int j, int k, int nz_hint)
{
    const int nz = nz_hint > 0 ? nz_hint
                               : static_cast<int>(grid.dz.size());
    if (k < 0 || k >= nz)
        throw std::out_of_range(
            "centered vertical span is outside the declared grid");
    const double fallback = (k == 0)
        ? vertical_center_distance(grid, i, j, 0, nz)
        : (k == nz - 1)
            ? vertical_center_distance(grid, i, j, nz - 2, nz)
            : vertical_center_distance(grid, i, j, k - 1, nz)
              + vertical_center_distance(grid, i, j, k, nz);

    if (!has_terrain_metrics(grid))
    {
        return fallback;
    }

    const TerrainMetrics3D& tm = *grid.terrain_metrics;
    const int nr = tm.z.size_r();
    const int nth = tm.z.size_th();
    const int terrain_nz = (nz_hint > 0)
        ? std::min(nz_hint, tm.z.size_z()) : tm.z.size_z();

    if (i < 0 || i >= nr || j < 0 || j >= nth
        || k <= 0 || k >= terrain_nz - 1)
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
        throw std::out_of_range(
            "terrain x metric index is outside the metric field");
    }

    const double J = tm.J(i, j, k);
    const double mx = tm.mx(i, j, k);
    if (!std::isfinite(J) || !std::isfinite(mx))
    {
        throw std::invalid_argument(
            "terrain x metrics must be finite");
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
        throw std::out_of_range(
            "terrain y metric index is outside the metric field");
    }

    const double J = tm.J(i, j, k);
    const double my = tm.my(i, j, k);
    if (!std::isfinite(J) || !std::isfinite(my))
    {
        throw std::invalid_argument(
            "terrain y metrics must be finite");
    }

    // Terrain metric relation: dz/dy = -my * J.
    const double dzdy = -my * J;
    const double stretch = std::sqrt(1.0 + dzdy * dzdy);
    return sanitize_positive(base_dy * stretch, base_dy);
}
} // namespace grid_metric

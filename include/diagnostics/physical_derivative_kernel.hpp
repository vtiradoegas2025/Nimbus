#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>

#include "terrain/terrain_base.hpp"

/**
 * @file physical_derivative_kernel.hpp
 * @brief Cell-centred Cartesian gradients on flat or terrain-following grids.
 *
 * The horizontal grid is Cartesian and uniform in computational x/y.  On a
 * mapped grid, neighbouring scalar centres generally have different physical
 * heights.  A derivative at fixed computational height therefore contains a
 * coordinate-slope contribution.  These helpers remove that contribution:
 *
 *   d(phi)/dx|z = d(phi)/dx|zeta - d(z)/dx|zeta * d(phi)/dz
 *
 * and likewise in y.  Vertical derivatives use the actual scalar-centre
 * heights.  Three-point Lagrange derivatives are second-order in the interior
 * and at boundaries; two-point grids reduce to the exact secant.
 */
namespace tmv::diagnostics
{

struct PhysicalGradientPoint
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

namespace detail
{

inline double derivative_three_points(double evaluation,
                                      double x0, double x1, double x2,
                                      double f0, double f1, double f2) noexcept
{
    const double d0 = (x0 - x1) * (x0 - x2);
    const double d1 = (x1 - x0) * (x1 - x2);
    const double d2 = (x2 - x0) * (x2 - x1);
    if (!std::isfinite(evaluation) || !std::isfinite(x0)
        || !std::isfinite(x1) || !std::isfinite(x2)
        || !std::isfinite(f0) || !std::isfinite(f1)
        || !std::isfinite(f2) || d0 == 0.0 || d1 == 0.0 || d2 == 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return f0 * (2.0 * evaluation - x1 - x2) / d0
         + f1 * (2.0 * evaluation - x0 - x2) / d1
         + f2 * (2.0 * evaluation - x0 - x1) / d2;
}

template <typename CoordinateAt, typename ValueAt>
inline double derivative_at_index(int index, int count,
                                  const CoordinateAt& coordinate_at,
                                  const ValueAt& value_at) noexcept
{
    if (count <= 1)
        return 0.0;
    if (index < 0 || index >= count)
        return std::numeric_limits<double>::quiet_NaN();
    if (count == 2)
    {
        const double x0 = coordinate_at(0);
        const double x1 = coordinate_at(1);
        const double f0 = value_at(0);
        const double f1 = value_at(1);
        const double dx = x1 - x0;
        if (!std::isfinite(x0) || !std::isfinite(x1)
            || !std::isfinite(f0) || !std::isfinite(f1) || dx == 0.0)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return (f1 - f0) / dx;
    }

    const int first = index == 0 ? 0 : (index == count - 1 ? count - 3 : index - 1);
    const int second = first + 1;
    const int third = first + 2;
    return derivative_three_points(
        coordinate_at(index),
        coordinate_at(first), coordinate_at(second), coordinate_at(third),
        value_at(first), value_at(second), value_at(third));
}

} // namespace detail

inline void validate_physical_derivative_geometry(
    int nr, int nth, int nz, double dx, double dy, double dz,
    const TerrainMetrics3D* terrain, const std::vector<double>* flat_z = nullptr)
{
    if (nr <= 0 || nth <= 0 || nz <= 0
        || !std::isfinite(dx) || dx <= 0.0
        || !std::isfinite(dy) || dy <= 0.0
        || !std::isfinite(dz) || dz <= 0.0)
    {
        throw std::invalid_argument(
            "physical derivatives require a non-empty Cartesian grid and positive finite spacing");
    }
    if (terrain == nullptr)
    {
        if (flat_z)
        {
            if (flat_z->size() != static_cast<std::size_t>(nz))
                throw std::invalid_argument("physical derivative center heights must match the grid");
            double previous = -std::numeric_limits<double>::infinity();
            for (double height : *flat_z)
            {
                if (!std::isfinite(height) || height <= previous)
                    throw std::invalid_argument("physical derivative center heights must increase strictly");
                previous = height;
            }
        }
        return;
    }
    if (terrain->z.size_r() != nr || terrain->z.size_th() != nth
        || terrain->z.size_z() != nz)
    {
        throw std::invalid_argument(
            "physical derivative terrain heights must match the scalar grid");
    }
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            double previous = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < nz; ++k)
            {
                const double height = terrain->z(i, j, k);
                if (!std::isfinite(height) || !(height > previous))
                {
                    throw std::invalid_argument(
                        "physical derivative terrain columns require finite strictly increasing heights");
                }
                previous = height;
            }
        }
}

/**
 * Physical Cartesian gradient at a scalar centre. `value_at` must accept
 * `(i,j,k)` and return a value collocated with `terrain->z`.
 */
template <typename ValueAt>
inline PhysicalGradientPoint physical_gradient_at(
    const ValueAt& value_at,
    int i, int j, int k,
    int nr, int nth, int nz,
    double dx, double dy, double dz,
    const TerrainMetrics3D* terrain, const std::vector<double>* flat_z = nullptr) noexcept
{
    if (i < 0 || i >= nr || j < 0 || j >= nth || k < 0 || k >= nz)
    {
        const double invalid = std::numeric_limits<double>::quiet_NaN();
        return {invalid, invalid, invalid};
    }

    const auto height_at = [&](int ii, int jj, int kk) noexcept
    {
        return terrain != nullptr
            ? terrain->z(ii, jj, kk)
            : flat_z ? (*flat_z)[static_cast<std::size_t>(kk)] : (static_cast<double>(kk) + 0.5) * dz;
    };
    const auto x_coordinate = [&](int ii) noexcept
    {
        return static_cast<double>(ii) * dx;
    };
    const auto y_coordinate = [&](int jj) noexcept
    {
        return static_cast<double>(jj) * dy;
    };

    const double vertical = detail::derivative_at_index(
        k, nz,
        [&](int kk) noexcept { return height_at(i, j, kk); },
        [&](int kk) noexcept { return static_cast<double>(value_at(i, j, kk)); });
    const double x_at_zeta = detail::derivative_at_index(
        i, nr, x_coordinate,
        [&](int ii) noexcept { return static_cast<double>(value_at(ii, j, k)); });
    const double y_at_zeta = detail::derivative_at_index(
        j, nth, y_coordinate,
        [&](int jj) noexcept { return static_cast<double>(value_at(i, jj, k)); });

    if (terrain == nullptr)
        return {x_at_zeta, y_at_zeta, vertical};

    const double height_x_at_zeta = detail::derivative_at_index(
        i, nr, x_coordinate,
        [&](int ii) noexcept { return height_at(ii, j, k); });
    const double height_y_at_zeta = detail::derivative_at_index(
        j, nth, y_coordinate,
        [&](int jj) noexcept { return height_at(i, jj, k); });
    return {x_at_zeta - height_x_at_zeta * vertical,
            y_at_zeta - height_y_at_zeta * vertical,
            vertical};
}

inline double ertel_potential_vorticity_si(
    const PhysicalGradientPoint& theta_gradient,
    double relative_vorticity_x,
    double relative_vorticity_y,
    double relative_vorticity_z,
    double coriolis_vertical,
    double dry_air_density) noexcept
{
    if (!std::isfinite(theta_gradient.x)
        || !std::isfinite(theta_gradient.y)
        || !std::isfinite(theta_gradient.z)
        || !std::isfinite(relative_vorticity_x)
        || !std::isfinite(relative_vorticity_y)
        || !std::isfinite(relative_vorticity_z)
        || !std::isfinite(coriolis_vertical)
        || !std::isfinite(dry_air_density)
        || dry_air_density <= 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (relative_vorticity_x * theta_gradient.x
            + relative_vorticity_y * theta_gradient.y
            + (relative_vorticity_z + coriolis_vertical) * theta_gradient.z)
        / dry_air_density;
}

} // namespace tmv::diagnostics

#pragma once

#include <cstddef>

/**
 * @file grid_coordinates.hpp
 * @brief Authoritative Cartesian index-to-coordinate transforms.
 *
 * Physical domain faces are at integer multiples of the grid spacing. Scalar
 * values live at cell centers, one half-cell above/right of the corresponding
 * lower face. C-grid velocity storage uses the upper face of the same indexed
 * scalar cell: u[i] is on x-face i+1, v[j] on y-face j+1, and w[k] on z-face
 * k+1.
 */
namespace tmv::grid_coordinates
{

constexpr double face(int face_index, double spacing)
{
    return static_cast<double>(face_index) * spacing;
}

constexpr double lower_face(int scalar_index, double spacing)
{
    return face(scalar_index, spacing);
}

constexpr double scalar_center(int scalar_index, double spacing)
{
    return (static_cast<double>(scalar_index) + 0.5) * spacing;
}

constexpr double upper_face(int scalar_index, double spacing)
{
    return face(scalar_index + 1, spacing);
}

constexpr double physical_extent(std::size_t cell_count, double spacing)
{
    return static_cast<double>(cell_count) * spacing;
}

constexpr double scalar_fractional_index(double coordinate, double spacing)
{
    return coordinate / spacing - 0.5;
}

constexpr double upper_face_fractional_index(double coordinate, double spacing)
{
    return coordinate / spacing - 1.0;
}

}  // namespace tmv::grid_coordinates

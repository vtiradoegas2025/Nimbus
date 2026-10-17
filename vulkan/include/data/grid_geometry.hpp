/**
 * @file grid_geometry.hpp
 * @brief Physical grid metadata and volume-display geometry helpers.
 */

#pragma once

#include <array>
#include <string>

namespace oglcpp
{

/** @brief Cartesian cell spacing carried by a Nimbus step manifest. */
struct PhysicalGridSpacing
{
    double x_m = 0.0;
    double y_m = 0.0;
    double z_m = 0.0;

    [[nodiscard]] bool available() const;
};

/** @brief Inclusive scalar-cell ownership bounds in payload x/y/z indices. */
struct PhysicalIndexRange
{
    std::array<int, 3> lower{0, 0, 0};
    std::array<int, 3> upper{-1, -1, -1};

    [[nodiscard]] bool valid_for(int nx, int ny, int nz) const;
    [[nodiscard]] int nx() const { return upper[0] - lower[0] + 1; }
    [[nodiscard]] int ny() const { return upper[1] - lower[1] + 1; }
    [[nodiscard]] int nz() const { return upper[2] - lower[2] + 1; }
};

/** @brief Outcome of decoding grid metadata from a present step manifest. */
enum class GridManifestStatus
{
    available,
    unavailable,
    invalid,
};

/**
 * @brief Decode and validate Cartesian physical spacing from manifest JSON.
 *
 * Modern manifests use `dy_m`; legacy Cartesian manifests stored the same
 * metre value in `dtheta_rad`. Non-Cartesian manifests are reported as
 * unavailable because one uniform Cartesian AABB cannot represent them. A
 * v4+ manifest must also carry a self-consistent physical ownership range;
 * callers can use `valid_range` to crop storage-only closure cells.
 * Terrain-following v5 geometry fails closed because this uniform-grid viewer
 * cannot faithfully place a non-separable physical-height lattice.
 */
GridManifestStatus parse_grid_manifest(
    const std::string& json,
    int expected_nx,
    int expected_ny,
    int expected_nz,
    PhysicalGridSpacing& out,
    std::string& detail,
    PhysicalIndexRange* valid_range = nullptr);

/** @brief Compare two spacing triples with a small relative tolerance. */
[[nodiscard]] bool physical_grid_spacing_equal(
    const PhysicalGridSpacing& lhs,
    const PhysicalGridSpacing& rhs);

/** @brief Exact comparison of two validated inclusive ownership ranges. */
[[nodiscard]] bool physical_index_range_equal(
    const PhysicalIndexRange& lhs,
    const PhysicalIndexRange& rhs);

/**
 * @brief Convert dimensions and cell spacing to normalized world X/Y/Z scale.
 *
 * Texture X/Y/Z map to world X/Z/Y respectively. The larger horizontal
 * physical extent is normalized to one; vertical exaggeration is explicit.
 */
bool normalized_volume_world_scale(
    int nx,
    int ny,
    int nz,
    const PhysicalGridSpacing& spacing,
    float vertical_exaggeration,
    std::array<float, 3>& out_world_scale,
    std::string& error);

} // namespace oglcpp

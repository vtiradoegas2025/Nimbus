/**
 * @file marching_cubes_cpu.hpp
 * @brief CPU reference implementation of marching-cubes isosurface extraction.
 *
 * The verified CPU reference for S7.3 (Validation Plan §3.3). The S7.3 render
 * path is a GPU marching-cubes pipeline (classify + prefix-sum + emit); before
 * those shaders exist, this module produces the ground-truth triangle mesh the
 * GPU output will be parity-checked against, exactly as the dynamics shaders
 * are validated against CPU references in bin/test_vulkan_gpu_parity.
 *
 * Intentionally free of any Vulkan dependency so it can be unit-tested without
 * a device/loader. It draws its case tables from mc_tables.hpp (Lorensen-Cline
 * 1987; see that file for the variant rationale and its bounded
 * ambiguous-face limitation). Vertex normals are deliberately out of scope for
 * this first pass — they layer on with central differences once the surface
 * geometry is verified.
 */

#pragma once

#include <array>
#include <vector>

namespace vkcpp
{

/**
 * @brief One triangle of the extracted isosurface.
 *
 * v[k] is the k-th vertex as {x, y, z} in physical/world coordinates (the grid
 * origin sits at sample index (0, 0, 0); see marching_cubes()).
 */
struct Triangle
{
    std::array<std::array<float, 3>, 3> v;
};

/**
 * @brief Extract the @p iso isosurface of a scalar field by marching cubes.
 *
 * The field is sampled at @p nx * @p ny * @p nz grid points in row-major order
 * (index i + j*nx + k*nx*ny). Cubes span adjacent samples, so the routine
 * processes (nx-1) * (ny-1) * (nz-1) cells; the final sample plane in each
 * dimension bounds the last cell and never starts a new one (no clamping, no
 * wrap). Any dimension below 2 yields no cells and an empty result.
 *
 * A corner is classified "inside" when its sample is strictly below @p iso
 * (matching the Lorensen-Cline table polarity). Each crossed edge gets one
 * vertex by linear interpolation, t = (iso - a) / (b - a), clamped to [0, 1];
 * when the two corner samples are numerically indistinguishable the vertex
 * defaults to the edge midpoint (t = 0.5). Vertices are placed in physical
 * coordinates using the voxel spacing @p dx, @p dy, @p dz.
 *
 * @param field Row-major scalar samples; must hold nx*ny*nz values.
 * @param nx    Sample count along x (>= 2 to produce geometry).
 * @param ny    Sample count along y (>= 2 to produce geometry).
 * @param nz    Sample count along z (>= 2 to produce geometry).
 * @param dx    Physical voxel spacing along x.
 * @param dy    Physical voxel spacing along y.
 * @param dz    Physical voxel spacing along z.
 * @param iso   Isovalue (threshold) of the surface to extract.
 * @return Triangles of the isosurface; empty when no surface crosses the grid.
 */
std::vector<Triangle> marching_cubes(const float* field,
                                     int nx,
                                     int ny,
                                     int nz,
                                     float dx,
                                     float dy,
                                     float dz,
                                     float iso);

}  // namespace vkcpp

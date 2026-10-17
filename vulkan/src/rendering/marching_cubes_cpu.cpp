/**
 * @file marching_cubes_cpu.cpp
 * @brief Implementation of the CPU marching-cubes reference (see header).
 *
 * Standard table-driven Lorensen-Cline marching cubes. The corner numbering,
 * the 12 edge endpoint pairs, and the case-index bit assignment below are the
 * conventions the mc_tables.hpp tables are built for (Cory Gene Bloyd's
 * transcription on paulbourke.net/geometry/polygonise/); they must stay in
 * lock-step with those tables and with the eventual GPU classify/emit shaders.
 *
 * Cube corner layout (local index -> unit-cell offset), with the cell anchored
 * at sample (i, j, k):
 *
 *        4 ------- 5
 *       /|        /|        corner 0: (0,0,0)   corner 4: (0,1,0)
 *      7 ------- 6 |        corner 1: (1,0,0)   corner 5: (1,1,0)
 *      | 0 ------|-1        corner 2: (1,0,1)   corner 6: (1,1,1)
 *      |/        |/         corner 3: (0,0,1)   corner 7: (0,1,1)
 *      3 ------- 2
 */

#include "rendering/marching_cubes_cpu.hpp"

#include "rendering/mc_tables.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace vkcpp
{
namespace
{

/* Unit-cell offset of each cube corner, in (di, dj, dk) sample steps. */
constexpr int kCornerOffset[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1},
    {0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1},
};

/* The two corner endpoints of each of the 12 cube edges. */
constexpr int kEdgeCorner[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},  // bottom face (k -> k slice)
    {4, 5}, {5, 6}, {6, 7}, {7, 4},  // top face
    {0, 4}, {1, 5}, {2, 6}, {3, 7},  // vertical edges
};

/*
 * Below this the two corner samples on a crossed edge are numerically
 * indistinguishable, so the crossing location is undefined. We then place the
 * vertex at the edge midpoint (t = 0.5): the unique orientation-independent
 * choice, which keeps the output deterministic regardless of which endpoint is
 * treated as the edge origin. (For a genuinely crossed edge the corner samples
 * straddle iso, so this only triggers in true degeneracies.)
 */
constexpr float kVertexInterpEpsilon = 1e-7f;

}  // namespace

std::vector<Triangle> marching_cubes(const float* field,
                                     int nx,
                                     int ny,
                                     int nz,
                                     float dx,
                                     float dy,
                                     float dz,
                                     float iso)
{
    std::vector<Triangle> out;
    if (field == nullptr || nx < 2 || ny < 2 || nz < 2)
    {
        return out;
    }

    // Reserve for roughly one cell's worth of triangles across the largest
    // cross-section: an isosurface is a 2D sheet through a 3D grid, so its
    // triangle count scales with area, not volume. This amortizes growth
    // without over-allocating for large volumes.
    const std::size_t cross_section = static_cast<std::size_t>(
        std::max({nx * ny, ny * nz, nx * nz}));
    out.reserve(2 * cross_section);

    const auto idx = [nx, ny](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * nx +
               static_cast<std::size_t>(k) * nx * ny;
    };

    const float spacing[3] = {dx, dy, dz};

    for (int k = 0; k < nz - 1; ++k)
    {
        for (int j = 0; j < ny - 1; ++j)
        {
            for (int i = 0; i < nx - 1; ++i)
            {
                float corner_val[8];
                int corner_pos[8][3];
                int cube_index = 0;
                for (int c = 0; c < 8; ++c)
                {
                    const int ci = i + kCornerOffset[c][0];
                    const int cj = j + kCornerOffset[c][1];
                    const int ck = k + kCornerOffset[c][2];
                    corner_pos[c][0] = ci;
                    corner_pos[c][1] = cj;
                    corner_pos[c][2] = ck;
                    const float val = field[idx(ci, cj, ck)];
                    corner_val[c] = val;
                    if (val < iso)
                    {
                        cube_index |= (1 << c);
                    }
                }

                const int edges = edge_table[cube_index];
                if (edges == 0)
                {
                    continue;
                }

                // Interpolate a vertex on every crossed edge.
                std::array<std::array<float, 3>, 12> vert{};
                for (int e = 0; e < 12; ++e)
                {
                    if ((edges & (1 << e)) == 0)
                    {
                        continue;
                    }
                    const int a = kEdgeCorner[e][0];
                    const int b = kEdgeCorner[e][1];
                    const float va = corner_val[a];
                    const float vb = corner_val[b];
                    const float denom = vb - va;
                    float t = (std::fabs(denom) < kVertexInterpEpsilon)
                                  ? 0.5f
                                  : (iso - va) / denom;
                    t = std::clamp(t, 0.0f, 1.0f);
                    for (int d = 0; d < 3; ++d)
                    {
                        const float pa = static_cast<float>(corner_pos[a][d]);
                        const float pb = static_cast<float>(corner_pos[b][d]);
                        vert[e][d] = (pa + t * (pb - pa)) * spacing[d];
                    }
                }

                // Emit triangles for this case (edge-index triples, -1 ends).
                const int* tris = tri_table[cube_index];
                for (int t = 0; tris[t] != -1; t += 3)
                {
                    Triangle tri;
                    tri.v[0] = vert[tris[t]];
                    tri.v[1] = vert[tris[t + 1]];
                    tri.v[2] = vert[tris[t + 2]];
                    out.push_back(tri);
                }
            }
        }
    }

    return out;
}

}  // namespace vkcpp

/**
 * @file test_marching_cubes_cpu.cpp
 * @brief Unit tests for the S7.3 CPU marching-cubes reference.
 *
 * The §3.3 headline acceptance is that a synthetic sphere produces a correct
 * spherical isosurface. The assertions here pin the implementation to
 * *external truth* rather than to itself:
 *   - The case tables are checked against their own structural invariants and
 *     the well-known complement symmetry (complementing inside/outside leaves
 *     the crossed-edge set unchanged) — no MC algorithm involved.
 *   - The sphere and plane tests measure how far the emitted vertices stray
 *     from the analytic surface (R - ||v - c|| and the plane equation),
 *     quantities defined independently of the extraction code.
 *   - The 2x2x2 case checks the triangle count against the table prediction
 *     and the vertex positions against the cube edge midpoints derived from
 *     geometry alone.
 */
#include "catch2/catch.hpp"
#include "rendering/marching_cubes_cpu.hpp"
#include "rendering/mc_tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using vkcpp::marching_cubes;
using vkcpp::Triangle;

namespace
{

/** @brief Length of the leading non-(-1) run of a tri_table row. */
int tri_prefix_length(const int* row)
{
    int n = 0;
    while (n < 16 && row[n] != -1)
    {
        ++n;
    }
    return n;
}

double dist(const std::array<float, 3>& p, double cx, double cy, double cz)
{
    const double dx = p[0] - cx;
    const double dy = p[1] - cy;
    const double dz = p[2] - cz;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool vec_approx(const std::array<float, 3>& a, float x, float y, float z, float tol)
{
    return std::fabs(a[0] - x) <= tol && std::fabs(a[1] - y) <= tol &&
           std::fabs(a[2] - z) <= tol;
}

}  // namespace

TEST_CASE("Case tables satisfy structural invariants", "[vulkan][marching_cubes][tables]")
{
    for (int i = 0; i < 256; ++i)
    {
        INFO("case index " << i);

        // edge_table entries are 12-bit masks.
        CHECK(vkcpp::edge_table[i] >= 0);
        CHECK(vkcpp::edge_table[i] < (1 << 12));

        const int* row = vkcpp::tri_table[i];
        const int n = tri_prefix_length(row);

        // Triangles come in complete triples.
        CHECK(n % 3 == 0);
        CHECK(n <= 15);  // at most five triangles per cube

        // Every listed edge index is valid; everything past the run is -1.
        int used_mask = 0;
        for (int e = 0; e < n; ++e)
        {
            CHECK(row[e] >= 0);
            CHECK(row[e] < 12);
            used_mask |= (1 << row[e]);
        }
        for (int e = n; e < 16; ++e)
        {
            CHECK(row[e] == -1);
        }

        // Consistency: the edges referenced by the triangles are exactly the
        // edges the edge mask marks as crossed.
        CHECK(used_mask == vkcpp::edge_table[i]);

        // Complement symmetry: flipping every corner's inside/outside state
        // leaves the surface (hence the crossed-edge set) unchanged.
        CHECK(vkcpp::edge_table[i] == vkcpp::edge_table[0xFF ^ i]);
    }

    // The two trivial configurations produce no geometry.
    CHECK(vkcpp::edge_table[0x00] == 0);
    CHECK(vkcpp::edge_table[0xFF] == 0);
    CHECK(vkcpp::tri_table[0x00][0] == -1);
    CHECK(vkcpp::tri_table[0xFF][0] == -1);
}

TEST_CASE("Synthetic sphere vertices lie on the analytic sphere", "[vulkan][marching_cubes]")
{
    // 32^3 grid, unit spacing. Center and radius chosen so the sphere sits well
    // inside the box and is not coincident with sample points (avoids the
    // exactly-on-surface degeneracy).
    const int n = 32;
    const float dx = 1.0f, dy = 1.0f, dz = 1.0f;
    const double cx = 16.3, cy = 15.7, cz = 16.1;
    const double R = 9.5;

    std::vector<float> field(static_cast<std::size_t>(n) * n * n);
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                const double rx = i - cx, ry = j - cy, rz = k - cz;
                // f > 0 inside, f < 0 outside; the iso=0 surface is the sphere.
                field[static_cast<std::size_t>(i) + j * n + k * n * n] =
                    static_cast<float>(R * R - (rx * rx + ry * ry + rz * rz));
            }
        }
    }

    const auto tris = marching_cubes(field.data(), n, n, n, dx, dy, dz, 0.0f);
    REQUIRE(!tris.empty());

    const double maxd = std::max({dx, dy, dz});
    const double tol = 0.75 * maxd;  // §3.3-suggested, voxel-bounded tolerance

    double worst = 0.0;
    double min_x = 1e9, max_x = -1e9;
    for (const auto& t : tris)
    {
        for (const auto& v : t.v)
        {
            worst = std::max(worst, std::fabs(dist(v, cx, cy, cz) - R));
            min_x = std::min(min_x, static_cast<double>(v[0]));
            max_x = std::max(max_x, static_cast<double>(v[0]));
        }
    }

    INFO("triangles=" << tris.size() << " worst radial error=" << worst << " tol=" << tol);
    CHECK(worst <= tol);

    // The surface spans the full sphere: its x-extent reaches both poles to
    // within a voxel (a partial patch would fail this).
    CHECK(std::fabs(min_x - (cx - R)) <= maxd);
    CHECK(std::fabs(max_x - (cx + R)) <= maxd);
}

TEST_CASE("Synthetic plane produces a coplanar triangulation", "[vulkan][marching_cubes]")
{
    // Linear field f = n . x + d, so the iso=0 surface is exactly the plane
    // n . x + d = 0. Linear interpolation of a linear field is exact, so every
    // emitted vertex should satisfy the plane equation to round-off.
    const int n = 16;
    const float dx = 1.0f, dy = 1.0f, dz = 1.0f;
    const double nx_ = 1.0, ny_ = 2.0, nz_ = 3.0;  // non-axis-aligned normal
    const double cx = 7.5, cy = 7.5, cz = 7.5;
    const double d = -(nx_ * cx + ny_ * cy + nz_ * cz);
    const double nmag = std::sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_);

    std::vector<float> field(static_cast<std::size_t>(n) * n * n);
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                field[static_cast<std::size_t>(i) + j * n + k * n * n] =
                    static_cast<float>(nx_ * i + ny_ * j + nz_ * k + d);
            }
        }
    }

    const auto tris = marching_cubes(field.data(), n, n, n, dx, dy, dz, 0.0f);
    REQUIRE(!tris.empty());

    double worst = 0.0;
    for (const auto& t : tris)
    {
        for (const auto& v : t.v)
        {
            const double signed_dist =
                (nx_ * v[0] + ny_ * v[1] + nz_ * v[2] + d) / nmag;
            worst = std::max(worst, std::fabs(signed_dist));
        }
    }

    // Far inside the voxel size; the analytic plane is the external truth.
    INFO("triangles=" << tris.size() << " worst plane residual=" << worst);
    CHECK(worst <= 1e-3);
}

TEST_CASE("Degenerate volumes produce no triangles", "[vulkan][marching_cubes]")
{
    const int n = 6;
    std::vector<float> field(static_cast<std::size_t>(n) * n * n);

    // Entirely below iso -> every corner "inside" (case 0xFF) -> no surface.
    std::fill(field.begin(), field.end(), -5.0f);
    CHECK(marching_cubes(field.data(), n, n, n, 1.0f, 1.0f, 1.0f, 0.0f).empty());

    // Entirely above iso -> every corner "outside" (case 0x00) -> no surface.
    std::fill(field.begin(), field.end(), 5.0f);
    CHECK(marching_cubes(field.data(), n, n, n, 1.0f, 1.0f, 1.0f, 0.0f).empty());

    // Sub-cube grids cannot form a cell.
    CHECK(marching_cubes(field.data(), 1, 5, 5, 1.0f, 1.0f, 1.0f, 0.0f).empty());
    CHECK(marching_cubes(nullptr, n, n, n, 1.0f, 1.0f, 1.0f, 0.0f).empty());
}

TEST_CASE("2x2x2 single-corner cell matches the table and edge geometry", "[vulkan][marching_cubes]")
{
    // One cube. Corner 0 (sample index 0) is the only one below iso, so the
    // case index is exactly bit 0 == 1. The table predicts one triangle on the
    // three edges incident to corner 0 (edges 0, 3, 8).
    std::array<float, 8> field{};
    field.fill(1.0f);
    field[0] = -1.0f;  // sample (0,0,0) == cube corner 0

    const auto tris = marching_cubes(field.data(), 2, 2, 2, 1.0f, 1.0f, 1.0f, 0.0f);

    // Exact triangle count == the table prediction for case 1.
    const int predicted = tri_prefix_length(vkcpp::tri_table[1]) / 3;
    REQUIRE(predicted == 1);
    REQUIRE(tris.size() == static_cast<std::size_t>(predicted));

    // With symmetric +/-1 corners and iso 0 each crossing is the edge
    // midpoint. Edges incident to corner 0: 0 (0->1), 3 (3->0), 8 (0->4).
    // Their midpoints, derived from cube geometry alone:
    const std::array<std::array<float, 3>, 3> expected = {{
        {0.5f, 0.0f, 0.0f},  // edge 0 midpoint
        {0.0f, 0.0f, 0.5f},  // edge 3 midpoint
        {0.0f, 0.5f, 0.0f},  // edge 8 midpoint
    }};

    for (const auto& want : expected)
    {
        bool found = false;
        for (const auto& got : tris[0].v)
        {
            if (vec_approx(got, want[0], want[1], want[2], 1e-6f))
            {
                found = true;
                break;
            }
        }
        INFO("expected vertex (" << want[0] << ", " << want[1] << ", " << want[2] << ")");
        CHECK(found);
    }
}

TEST_CASE("Extraction is deterministic", "[vulkan][marching_cubes]")
{
    const int n = 20;
    const double cx = 10.2, cy = 9.8, cz = 10.4, R = 6.3;
    std::vector<float> field(static_cast<std::size_t>(n) * n * n);
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                const double rx = i - cx, ry = j - cy, rz = k - cz;
                field[static_cast<std::size_t>(i) + j * n + k * n * n] =
                    static_cast<float>(R * R - (rx * rx + ry * ry + rz * rz));
            }
        }
    }

    const auto a = marching_cubes(field.data(), n, n, n, 0.5f, 0.5f, 0.5f, 0.0f);
    const auto b = marching_cubes(field.data(), n, n, n, 0.5f, 0.5f, 0.5f, 0.0f);

    REQUIRE(a.size() == b.size());
    REQUIRE(!a.empty());
    bool identical = true;
    for (std::size_t t = 0; t < a.size(); ++t)
    {
        for (int p = 0; p < 3; ++p)
        {
            for (int c = 0; c < 3; ++c)
            {
                // Same input, same code path: bit-identical floats expected.
                if (a[t].v[p][c] != b[t].v[p][c])
                {
                    identical = false;
                }
            }
        }
    }
    CHECK(identical);
}

#include "catch2/catch.hpp"

#include "core/field/field3d.hpp"
#include "core/output/stagger_interpolation.hpp"
#include "core/runtime/simulation.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

void setup_global_grid(int nr, int nth, int nz)
{
    NR     = nr;
    NTH    = nth;
    NZ     = nz;
    dr     = 250.0;
    dz     = 250.0;
    dt     = 0.1;
    dtheta = 400.0;
}

} // namespace

// ============================================================================
// Gate 1 -- Cartesian x interpolation: interior + west boundary.
// ============================================================================
TEST_CASE("interpolate_u_face_to_center: interior mean and one-sided west boundary", "[output][cgrid][c8][stagger]")
{
    setup_global_grid(/*nr=*/8, /*nth=*/4, /*nz=*/6);

    Field3D u_face;
    u_face.resize(NR, NTH, NZ, 0.0f);
    // Linear ramp in i so the centered average has a closed form:
    // u_face[i] = i + 0.5  =>  u_center[i] = i  for i >= 1.
    // u_center[0] = u_face[0] = 0.5 (the maintained west boundary face).
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                u_face[i][j][k] = static_cast<float>(i) + 0.5f;

    Field3D u_center;
    interpolate_u_face_to_center(u_face, u_center);

    REQUIRE(u_center.size_r() == NR);
    REQUIRE(u_center.size_th() == NTH);
    REQUIRE(u_center.size_z() == NZ);

    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
        {
            INFO("west cell at (j=" << j << ", k=" << k << "): u_center[0] = " << u_center[0][j][k]);
            REQUIRE(u_center[0][j][k] == Approx(0.5f).margin(1.0e-6));

            for (int i = 1; i < NR; ++i)
            {
                INFO("interior cell at (i=" << i << ", j=" << j << ", k=" << k
                                            << "): u_center = " << u_center[i][j][k]);
                REQUIRE(u_center[i][j][k] == Approx(static_cast<float>(i)).margin(1.0e-6));
            }
        }
}

// ============================================================================
// Gate 2 -- Cartesian y interpolation: no cross-domain wrap.
// ============================================================================
TEST_CASE("interpolate_v_face_to_center: physical y edge never wraps north to south", "[output][cgrid][c8][stagger]")
{
    setup_global_grid(/*nr=*/4, /*nth=*/8, /*nz=*/4);

    Field3D v_face;
    v_face.resize(NR, NTH, NZ, 0.0f);
    // A large north-edge sentinel makes any accidental wrap obvious.
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                v_face[i][j][k] = static_cast<float>(j);
    for (int i = 0; i < NR; ++i)
        for (int k = 0; k < NZ; ++k)
            v_face[i][NTH - 1][k] = 1000.0f;

    Field3D v_center;
    interpolate_v_face_to_center(v_face, v_center);

    for (int i = 0; i < NR; ++i)
        for (int k = 0; k < NZ; ++k)
        {
            for (int j = 1; j < NTH - 1; ++j)
            {
                INFO("non-wrap cell at (i=" << i << ", j=" << j << ", k=" << k
                                            << "): v_center = " << v_center[i][j][k]);
                REQUIRE(v_center[i][j][k] == Approx(static_cast<float>(j) - 0.5f).margin(1.0e-6));
            }
            INFO("south boundary at (i=" << i << ", j=0, k=" << k << "): v_center = " << v_center[i][0][k]);
            REQUIRE(v_center[i][0][k] == Approx(0.0f).margin(1.0e-6));
            REQUIRE(v_center[i][NTH - 1][k] == Approx(0.5f * (static_cast<float>(NTH - 2) + 1000.0f)).margin(1.0e-6));
        }
}

// ============================================================================
// Gate 3 -- Vertical interpolation: interior + surface special case.
// ============================================================================
TEST_CASE("interpolate_w_face_to_center: interior arithmetic mean and surface half-cell",
          "[output][cgrid][c8][stagger]")
{
    setup_global_grid(/*nr=*/4, /*nth=*/4, /*nz=*/8);

    Field3D w_face;
    w_face.resize(NR, NTH, NZ, 0.0f);
    // w_face[k] = k + 0.5 -> w_center[k] = k for k >= 1, w_center[0] = 0.25.
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                w_face[i][j][k] = static_cast<float>(k) + 0.5f;

    Field3D w_center;
    interpolate_w_face_to_center(w_face, w_center);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            INFO("surface cell at (i=" << i << ", j=" << j << "): w_center[0] = " << w_center[i][j][0]);
            REQUIRE(w_center[i][j][0] == Approx(0.25f).margin(1.0e-6));

            for (int k = 1; k < NZ; ++k)
            {
                INFO("interior cell at (i=" << i << ", j=" << j << ", k=" << k
                                            << "): w_center = " << w_center[i][j][k]);
                REQUIRE(w_center[i][j][k] == Approx(static_cast<float>(k)).margin(1.0e-6));
            }
        }
}

TEST_CASE("interpolate_w_face_to_center: terrain surface uses physical lower-face velocity",
          "[output][cgrid][terrain][stagger]")
{
    setup_global_grid(/*nr=*/4, /*nth=*/5, /*nz=*/3);

    Field3D w_face;
    w_face.resize(NR, NTH, NZ, 0.0f);
    std::vector<double> ground_w(static_cast<std::size_t>(NR) * static_cast<std::size_t>(NTH));
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            const std::size_t index =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) + static_cast<std::size_t>(j);
            ground_w[index] = 2.0 + 0.25 * static_cast<double>(i) - 0.1 * static_cast<double>(j);
            w_face[i][j][0] = static_cast<float>(6.0 + i + j);
            w_face[i][j][1] = 10.0f;
            w_face[i][j][2] = 14.0f;
        }

    Field3D w_center;
    interpolate_w_face_to_center(w_face, w_center, ground_w);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            const std::size_t index =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) + static_cast<std::size_t>(j);
            CHECK(w_center[i][j][0] == Approx(0.5 * (ground_w[index] + w_face[i][j][0])));
            CHECK(w_center[i][j][1] == Approx(0.5f * (w_face[i][j][0] + w_face[i][j][1])));
            CHECK(w_center[i][j][2] == Approx(12.0f));
        }

    CHECK_THROWS_AS(interpolate_w_face_to_center(w_face, w_center, std::vector<double>(3)), std::invalid_argument);
    ground_w.front() = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(interpolate_w_face_to_center(w_face, w_center, ground_w), std::invalid_argument);
}

// ============================================================================
// Gate 4 -- Collocated equivalence: when face values are constant in i
// (or j, k), the interior arithmetic mean equals the face value, so an
// already-collocated input passes through unchanged at every interior
// cell. The physical west/south boundary faces remain one-sided, while the
// rigid surface still uses the implicit w=0 ground face.
// ============================================================================
TEST_CASE("interpolate_*: constant face fields produce constant centers in interior", "[output][cgrid][c8][stagger]")
{
    setup_global_grid(/*nr=*/6, /*nth=*/8, /*nz=*/6);

    const float U_VAL = 7.5f;
    const float V_VAL = -3.25f;
    const float W_VAL = 12.0f;

    Field3D u_face, v_face, w_face;
    u_face.resize(NR, NTH, NZ, U_VAL);
    v_face.resize(NR, NTH, NZ, V_VAL);
    w_face.resize(NR, NTH, NZ, W_VAL);

    Field3D u_center, v_center, w_center;
    interpolate_u_face_to_center(u_face, u_center);
    interpolate_v_face_to_center(v_face, v_center);
    interpolate_w_face_to_center(w_face, w_center);

    // Interior (i >= 1, k >= 1): the arithmetic mean of two equal
    // values is the same value.
    for (int i = 1; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 1; k < NZ; ++k)
            {
                REQUIRE(u_center[i][j][k] == Approx(U_VAL).margin(1.0e-6));
                REQUIRE(v_center[i][j][k] == Approx(V_VAL).margin(1.0e-6));
                REQUIRE(w_center[i][j][k] == Approx(W_VAL).margin(1.0e-6));
            }

    // West boundary uses its maintained face value; the surface averages
    // against the implicit rigid-ground w=0 face.
    for (int j = 0; j < NTH; ++j)
        for (int k = 1; k < NZ; ++k)
        {
            REQUIRE(u_center[0][j][k] == Approx(U_VAL).margin(1.0e-6));
        }
    for (int i = 1; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            REQUIRE(w_center[i][j][0] == Approx(0.5f * W_VAL).margin(1.0e-6));
        }
}

// ============================================================================
// Gate 5 -- A uniform Cartesian wind remains uniform at every output cell,
// including the physical west and south boundary planes.
// ============================================================================
TEST_CASE("interpolate_*: cgrid uniform Cartesian wind remains uniform", "[output][cgrid][c8][stagger]")
{
    const int NR_T  = 8;
    const int NTH_T = 12;
    const int NZ_T  = 6;
    setup_global_grid(NR_T, NTH_T, NZ_T);

    const double UX = 5.0;
    const double UY = 3.0;

    Field3D u_face, v_face;
    u_face.resize(NR, NTH, NZ, static_cast<float>(UX));
    v_face.resize(NR, NTH, NZ, static_cast<float>(UY));

    Field3D u_center, v_center;
    interpolate_u_face_to_center(u_face, u_center);
    interpolate_v_face_to_center(v_face, v_center);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                REQUIRE(u_center[i][j][k] == Approx(static_cast<float>(UX)).margin(1.0e-6));
                REQUIRE(v_center[i][j][k] == Approx(static_cast<float>(UY)).margin(1.0e-6));
            }
}

TEST_CASE("stagger interpolation derives dimensions from explicit source", "[output][cgrid][stagger][state]")
{
    setup_global_grid(9, 8, 7);
    Field3D u_face(2, 3, 4, 6.0f);
    Field3D u_center;

    interpolate_u_face_to_center(u_face, u_center);

    CHECK(u_center.size_r() == 2);
    CHECK(u_center.size_th() == 3);
    CHECK(u_center.size_z() == 4);
    CHECK(u_center[1][2][3] == Approx(6.0f));
}

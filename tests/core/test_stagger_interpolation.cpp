/**
 * @file test_stagger_interpolation.cpp
 * @brief Verification gates for Phase C.8 -- face-to-center velocity
 *        interpolation used by output paths.
 *
 * Phase C.8 of docs/CoordinateBackend_Plan.md.
 *
 * The stagger interpolation helpers convert C-grid face-staggered
 * velocity fields (u at r-face, v at theta-face, w at z-face) into
 * cell-center fields suitable for npy / shm output. The four gates
 * below cover the interior arithmetic, the axis special-case for u
 * (i=0), the periodic wrap for v, and the surface special-case for w
 * (k=0).
 */

#include "catch2/catch.hpp"

#include "core/field3d.hpp"
#include "core/output/stagger_interpolation.hpp"
#include "core/simulation.hpp"

#include <cmath>


namespace
{

constexpr double kPi = 3.14159265358979323846;

void setup_global_grid(int nr, int nth, int nz)
{
    NR     = nr;
    NTH    = nth;
    NZ     = nz;
    dr     = 250.0;
    dz     = 250.0;
    dt     = 0.1;
    dtheta = 2.0 * kPi / static_cast<double>(nth);
}

}  // namespace


// ============================================================================
// Gate 1 -- Radial interpolation: interior + axis special case.
// ============================================================================
TEST_CASE("interpolate_u_face_to_center: interior arithmetic mean and axis half-cell",
          "[output][cgrid][c8][stagger]")
{
    setup_global_grid(/*nr=*/8, /*nth=*/4, /*nz=*/6);

    Field3D u_face;
    u_face.resize(NR, NTH, NZ, 0.0f);
    // Linear ramp in i so the centered average has a closed form:
    // u_face[i] = i + 0.5  =>  u_center[i] = i  for i >= 1.
    // u_center[0] = 0.5 * u_face[0] = 0.25.
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
            INFO("axis cell at (j=" << j << ", k=" << k << "): u_center[0] = "
                 << u_center[0][j][k]);
            REQUIRE(u_center[0][j][k] == Approx(0.25f).margin(1.0e-6));

            for (int i = 1; i < NR; ++i)
            {
                INFO("interior cell at (i=" << i << ", j=" << j << ", k=" << k
                     << "): u_center = " << u_center[i][j][k]);
                REQUIRE(u_center[i][j][k] == Approx(static_cast<float>(i)).margin(1.0e-6));
            }
        }
}


// ============================================================================
// Gate 2 -- Azimuthal interpolation: periodic wrap.
// ============================================================================
TEST_CASE("interpolate_v_face_to_center: periodic theta-face arithmetic mean",
          "[output][cgrid][c8][stagger]")
{
    setup_global_grid(/*nr=*/4, /*nth=*/8, /*nz=*/4);

    Field3D v_face;
    v_face.resize(NR, NTH, NZ, 0.0f);
    // v_face stores values at theta-face (theta_{j+1/2}). Use a v_face[j] = j
    // ramp across the periodic theta direction. The theta_{j+1/2} averaging
    // gives v_center[j] = 0.5*(v_face[j-1] + v_face[j]) (periodic).
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                v_face[i][j][k] = static_cast<float>(j);

    Field3D v_center;
    interpolate_v_face_to_center(v_face, v_center);

    for (int i = 0; i < NR; ++i)
        for (int k = 0; k < NZ; ++k)
        {
            // Non-wrap cells: v_center[j] = 0.5*(j-1 + j) = j - 0.5
            for (int j = 1; j < NTH; ++j)
            {
                INFO("non-wrap cell at (i=" << i << ", j=" << j << ", k=" << k
                     << "): v_center = " << v_center[i][j][k]);
                REQUIRE(v_center[i][j][k] ==
                        Approx(static_cast<float>(j) - 0.5f).margin(1.0e-6));
            }
            // Wrap cell at j=0: v_center[0] = 0.5*(v_face[NTH-1] + v_face[0])
            //                              = 0.5*((NTH-1) + 0) = (NTH-1)/2
            INFO("wrap cell at (i=" << i << ", j=0, k=" << k << "): v_center = "
                 << v_center[i][0][k]);
            REQUIRE(v_center[i][0][k] ==
                    Approx(static_cast<float>(NTH - 1) * 0.5f).margin(1.0e-6));
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
            INFO("surface cell at (i=" << i << ", j=" << j << "): w_center[0] = "
                 << w_center[i][j][0]);
            REQUIRE(w_center[i][j][0] == Approx(0.25f).margin(1.0e-6));

            for (int k = 1; k < NZ; ++k)
            {
                INFO("interior cell at (i=" << i << ", j=" << j << ", k=" << k
                     << "): w_center = " << w_center[i][j][k]);
                REQUIRE(w_center[i][j][k] == Approx(static_cast<float>(k)).margin(1.0e-6));
            }
        }
}


// ============================================================================
// Gate 4 -- Collocated equivalence: when face values are constant in i
// (or j, k), the interior arithmetic mean equals the face value, so an
// already-collocated input passes through unchanged at every interior
// cell. The axis / surface half-cells differ (0.5 * face, not face),
// which is the documented C-grid convention; collocated configurations
// never invoke these helpers, but the test pins the expectation.
// ============================================================================
TEST_CASE("interpolate_*: constant face fields produce constant centers in interior",
          "[output][cgrid][c8][stagger]")
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

    // Axis / surface cells: 0.5 * face value (one-sided averaging
    // because the implicit ghost is the boundary BC value -- u=0 on
    // the singular axis, w=0 at the rigid surface).
    for (int j = 0; j < NTH; ++j)
        for (int k = 1; k < NZ; ++k)
        {
            REQUIRE(u_center[0][j][k] == Approx(0.5f * U_VAL).margin(1.0e-6));
        }
    for (int i = 1; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            REQUIRE(w_center[i][j][0] == Approx(0.5f * W_VAL).margin(1.0e-6));
        }
}


// ============================================================================
// Gate 5 -- C-grid uniform Cartesian wind: the interpolated cell-center
// velocity field reproduces the analytic projection on the cell-center
// theta values, demonstrating that the output velocities a downstream
// consumer receives are the right physical quantity.
//
// On the C-grid (Phase C.3 IC convention), a uniform Cartesian wind
// (ux, uy) produces:
//   u_face[i][j][k] = ux*cos(theta[j])         + uy*sin(theta[j])
//   v_face[i][j][k] = -ux*sin(theta_{j+1/2})   + uy*cos(theta_{j+1/2})
//
// After face-to-center interpolation the cell-center u retains its
// original cos/sin form (the radial averaging cancels because u_face
// is theta-dependent only). The cell-center v becomes the half-sum
// of v at theta_{j-1/2} and theta_{j+1/2}, which by the sum-to-product
// identity equals the cell-center projection scaled by cos(dtheta/2):
//   v_center[i][j][k] = cos(dtheta/2) * (-ux*sin(theta[j]) + uy*cos(theta[j]))
// (sin(A) + sin(B) = 2 sin((A+B)/2) cos((A-B)/2), and similarly for
// cos.) For dtheta = 2*pi/64, cos(dtheta/2) ~ 0.998795, so the
// interpolated value is within ~0.13 % of the analytic cell-center
// value -- close enough that a downstream user reading the npy file
// sees a smooth, near-uniform Cartesian wind.
// ============================================================================
TEST_CASE("interpolate_*: cgrid uniform Cartesian wind reproduces analytic projection",
          "[output][cgrid][c8][stagger]")
{
    const int    NR_T  = 8;
    const int    NTH_T = 64;
    const int    NZ_T  = 6;
    setup_global_grid(NR_T, NTH_T, NZ_T);

    const double UX = 5.0;
    const double UY = 3.0;

    Field3D u_face, v_face;
    u_face.resize(NR, NTH, NZ, 0.0f);
    v_face.resize(NR, NTH, NZ, 0.0f);
    for (int j = 0; j < NTH; ++j)
    {
        const double theta_c    = static_cast<double>(j) * dtheta;
        const double theta_face = theta_c + 0.5 * dtheta;
        const float u_val = static_cast<float>(UX * std::cos(theta_c)
                                             + UY * std::sin(theta_c));
        const float v_val = static_cast<float>(-UX * std::sin(theta_face)
                                              + UY * std::cos(theta_face));
        for (int i = 0; i < NR; ++i)
            for (int k = 0; k < NZ; ++k)
            {
                u_face[i][j][k] = u_val;
                v_face[i][j][k] = v_val;
            }
    }

    Field3D u_center, v_center;
    interpolate_u_face_to_center(u_face, u_center);
    interpolate_v_face_to_center(v_face, v_center);

    // u_face has no radial dependence, so u_center[i] = u_face[i] for i >= 1
    // (interior averaging of two equal values), and u_center[0] = 0.5*u_face[0].
    // The cell-center analytic value is u_x*cos(theta[j]) + u_y*sin(theta[j])
    // which is exactly u_face[j] for the i-uniform IC.
    for (int i = 1; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double theta_c = static_cast<double>(j) * dtheta;
                const double expected = UX * std::cos(theta_c) + UY * std::sin(theta_c);
                REQUIRE(u_center[i][j][k] ==
                        Approx(static_cast<float>(expected)).margin(1.0e-5));
            }

    // v_center has the cos(dtheta/2) attenuation discussed above.
    const double cos_half_dtheta = std::cos(0.5 * dtheta);
    double max_rel_err = 0.0;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double theta_c = static_cast<double>(j) * dtheta;
                const double expected_unattenuated =
                    -UX * std::sin(theta_c) + UY * std::cos(theta_c);
                const double expected = cos_half_dtheta * expected_unattenuated;
                const double observed = static_cast<double>(v_center[i][j][k]);
                if (std::abs(expected) > 1.0e-3)
                {
                    const double rel = std::abs(observed - expected) /
                                       std::abs(expected);
                    if (rel > max_rel_err) max_rel_err = rel;
                }
                // Per-element: tight relative tolerance (1e-5) plus a
                // small absolute floor for cells where expected ~ 0.
                REQUIRE(observed ==
                        Approx(static_cast<float>(expected))
                            .epsilon(1.0e-5).margin(1.0e-5));
            }

    INFO("Max relative error vs analytic v_center (NTH=64): " << max_rel_err);
    // The relative error here is float roundoff in the cos/sin
    // computation -- the v_center formula is bit-exact in double.
    // Setting the gate at 1e-5 captures float-precision drift while
    // catching any structural regression away from cos(dtheta/2)
    // attenuation.
    REQUIRE(max_rel_err < 1.0e-5);
}

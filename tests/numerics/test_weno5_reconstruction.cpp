/**
 * @file test_weno5_reconstruction.cpp
 * @brief Unit tests for the WENO5-JS cell-face reconstruction (science audit).
 *
 * Pins weno5_face_value (numerics/advection/weno5_face.hpp) - the upwind
 * Jiang & Shu (1996) WENO5 reconstruction used by the conservative Cartesian
 * C-grid transport and mirrored literally in the GPU flux shader. The audit
 * flagged that this verified-solid kernel had no reconstruction-level test.
 *
 * Stencil convention: weno5_face_value(a,b,c,d,e) reconstructs the value at
 * the face on the d-side of cell c (the i+1/2 face for c->d flow), from cells
 * (a,b,c,d,e) = (c-2,c-1,c,c+1,c+2).
 *
 * Coverage:
 *  - free-stream / constant preservation (interpolatory weights sum to 1);
 *  - exactness for low-degree polynomials in the finite-volume sense
 *    (linear: face = (c+d)/2 from cell centres; quadratic: each substencil is
 *    exact for cell averages, so the weighted result is exact regardless of
 *    the nonlinear weights);
 *  - a Jiang-Shu coefficient lock (an independent textbook reimplementation
 *    of the betas / ideal weights / eps must agree to round-off - this is the
 *    only case that exercises the smoothness-indicator coefficients, since the
 *    polynomial cases make all three candidates equal);
 *  - essentially-non-oscillatory behaviour at a discontinuity (the jump
 *    substencil is starved; no overshoot toward the jump);
 *  - the degenerate-input fallback to the upwind value.
 */

#include "catch2/catch.hpp"
#include "numerics/advection/weno5_face.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace
{

// Independent textbook Jiang & Shu (1996) WENO5 reconstruction, written from
// the canonical formulas (NOT copied from production). Used as a coefficient
// lock: production weno5_face_value must agree with this to round-off, so any
// drift in a beta coefficient, ideal weight, or epsilon breaks the test.
double weno5_js_reference(double a, double b, double c, double d, double e)
{
    // Field-relative eps (must mirror weno5_face_value after the science audit
    // fix): eps = eps_rel * max(stencil^2) + floor. The coefficient-lock cases
    // below are O(1) data, so eps stays ~1e-6 there - the lock still pins the
    // beta / ideal-weight / exponent constants.
    const double eps = 1.0e-6 * std::max({a * a, b * b, c * c, d * d, e * e}) + 1.0e-40;
    const double d0 = 0.1, d1 = 0.6, d2 = 0.3;

    const double p0 = (2.0 * a - 7.0 * b + 11.0 * c) / 6.0;
    const double p1 = (-1.0 * b + 5.0 * c + 2.0 * d) / 6.0;
    const double p2 = (2.0 * c + 5.0 * d - 1.0 * e) / 6.0;

    const double beta0 = (13.0 / 12.0) * (a - 2.0 * b + c) * (a - 2.0 * b + c)
                       + 0.25 * (a - 4.0 * b + 3.0 * c) * (a - 4.0 * b + 3.0 * c);
    const double beta1 = (13.0 / 12.0) * (b - 2.0 * c + d) * (b - 2.0 * c + d)
                       + 0.25 * (b - d) * (b - d);
    const double beta2 = (13.0 / 12.0) * (c - 2.0 * d + e) * (c - 2.0 * d + e)
                       + 0.25 * (3.0 * c - 4.0 * d + e) * (3.0 * c - 4.0 * d + e);

    const double al0 = d0 / ((eps + beta0) * (eps + beta0));
    const double al1 = d1 / ((eps + beta1) * (eps + beta1));
    const double al2 = d2 / ((eps + beta2) * (eps + beta2));
    const double s = al0 + al1 + al2;
    return (al0 * p0 + al1 * p1 + al2 * p2) / s;
}

}  // namespace

TEST_CASE("WENO5 preserves a constant field exactly (free-stream)",
          "[numerics][weno5][reconstruction]")
{
    REQUIRE(weno5_face_value(2.5, 2.5, 2.5, 2.5, 2.5) == Approx(2.5).margin(1e-13));
    REQUIRE(weno5_face_value(-3.0, -3.0, -3.0, -3.0, -3.0) == Approx(-3.0).margin(1e-13));
    REQUIRE(weno5_face_value(0.0, 0.0, 0.0, 0.0, 0.0) == Approx(0.0).margin(1e-13));
}

TEST_CASE("WENO5 reconstructs a linear field exactly (face = midpoint)",
          "[numerics][weno5][reconstruction]")
{
    // q[m] = a + b*m (cell centres). The face at c+1/2 sits halfway between
    // cell centres c and d, so the exact value is (c + d)/2 and every
    // candidate polynomial returns it.
    REQUIRE(weno5_face_value(1.0, 2.0, 3.0, 4.0, 5.0) == Approx(3.5).margin(1e-12));   // (3+4)/2
    REQUIRE(weno5_face_value(3.0, 5.0, 7.0, 9.0, 11.0) == Approx(8.0).margin(1e-12));  // (7+9)/2
    // Decreasing slope.
    REQUIRE(weno5_face_value(10.0, 8.0, 6.0, 4.0, 2.0) == Approx(5.0).margin(1e-12));  // (6+4)/2
}

TEST_CASE("WENO5 reconstructs a quadratic exactly from cell averages",
          "[numerics][weno5][reconstruction]")
{
    // Finite-volume exactness: each 3-cell substencil is exact for a
    // quadratic, so the weighted result is exact regardless of the nonlinear
    // weights. For f(x) = x^2 the cell average over the unit cell centred at m
    // is m^2 + 1/12; the reconstructed face POINT value at x = 2.5 is 6.25.
    const double t = 1.0 / 12.0;
    const double avg0 = 0.0 + t, avg1 = 1.0 + t, avg2 = 4.0 + t,
                 avg3 = 9.0 + t, avg4 = 16.0 + t;
    REQUIRE(weno5_face_value(avg0, avg1, avg2, avg3, avg4)
            == Approx(6.25).margin(1e-9));
}

TEST_CASE("WENO5 matches the Jiang-Shu formula to round-off (coefficient lock)",
          "[numerics][weno5][reconstruction]")
{
    // Non-polynomial inputs make the three candidates differ, so the nonlinear
    // weights (hence the beta coefficients, ideal weights, and eps) actually
    // matter. An independent textbook reimplementation must agree exactly.
    const double cases[][5] = {
        { 1.0,  0.5,  2.0,  1.5,  3.0},
        { 0.2,  1.7,  1.1,  2.9,  0.4},
        {-1.0,  2.0, -0.5,  3.0,  1.0},
        { 7.3,  7.1,  6.4,  2.0,  9.9},
    };
    for (const auto& q : cases)
    {
        const double got = weno5_face_value(q[0], q[1], q[2], q[3], q[4]);
        const double ref = weno5_js_reference(q[0], q[1], q[2], q[3], q[4]);
        REQUIRE(got == Approx(ref).epsilon(0.0).margin(1e-12));
    }
}

TEST_CASE("WENO5 reconstruction is scale-invariant (field-relative eps)",
          "[numerics][weno5][reconstruction]")
{
    // The science audit fix makes eps field-relative, so the nonlinear weights
    // depend only on the SHAPE of the stencil, not its magnitude:
    // weno5_face_value(s*q) == s * weno5_face_value(q). The old absolute
    // eps = 1e-6 broke this for moisture-scale fields (s ~ 1e-3): beta ~ s^2
    // fell to ~eps, blurring the weights toward the linear-upwind ideal weights
    // and killing the ENO property exactly at cloud edges. Non-polynomial
    // stencil so the weights genuinely matter.
    const double q[5] = {1.0, 0.5, 2.0, 1.5, 3.0};
    const double base = weno5_face_value(q[0], q[1], q[2], q[3], q[4]);
    for (double s : {1.0e3, 1.0e-3, 1.0e-6})
    {
        const double scaled =
            weno5_face_value(s * q[0], s * q[1], s * q[2], s * q[3], s * q[4]);
        REQUIRE(scaled == Approx(s * base).epsilon(1e-9));
    }
}

TEST_CASE("WENO5 is essentially non-oscillatory across a discontinuity",
          "[numerics][weno5][reconstruction]")
{
    // Upward step at the c->d face: the smooth upwind substencil (a,b,c all
    // equal) is selected and the jump substencils are starved, so the face
    // value stays at the upwind plateau - no overshoot toward the jump.
    const double up = weno5_face_value(5.0, 5.0, 5.0, 9.0, 9.0);
    REQUIRE(up == Approx(5.0).margin(1e-2));          // picks the upwind value
    REQUIRE(up >= 5.0 - 1e-9);                          // no undershoot below the plateau
    REQUIRE(up <= 9.0 + 1e-9);                          // bounded by the data range

    // The linear (ideal-weight) reconstruction overshoots toward the jump;
    // WENO5 must be markedly closer to the upwind value.
    const double d0 = 0.1, d1 = 0.6, d2 = 0.3;
    const double p0 = (2.0 * 5 - 7.0 * 5 + 11.0 * 5) / 6.0;
    const double p1 = (-1.0 * 5 + 5.0 * 5 + 2.0 * 9) / 6.0;
    const double p2 = (2.0 * 5 + 5.0 * 9 - 1.0 * 9) / 6.0;
    const double linear = d0 * p0 + d1 * p1 + d2 * p2;
    REQUIRE(linear > 6.0);                              // the optimal scheme overshoots
    REQUIRE(std::abs(up - 5.0) < std::abs(linear - 5.0));

    // Downward step is symmetric: face stays on the upwind (high) plateau.
    const double down = weno5_face_value(9.0, 9.0, 9.0, 5.0, 5.0);
    REQUIRE(down == Approx(9.0).margin(1e-2));
}

TEST_CASE("WENO5 falls back to the upwind value on non-finite input",
          "[numerics][weno5][reconstruction]")
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    // A non-finite stencil value makes the weights non-finite; the kernel
    // returns the upwind cell value c rather than propagating NaN/Inf.
    REQUIRE(weno5_face_value(nan, 1.0, 2.0, 3.0, 4.0) == Approx(2.0));
    REQUIRE(weno5_face_value(1.0, 2.0, 3.0, 4.0, inf) == Approx(3.0));
    REQUIRE(std::isfinite(weno5_face_value(nan, nan, 5.0, nan, nan)));
    REQUIRE(weno5_face_value(nan, nan, 5.0, nan, nan) == Approx(5.0));
}

TEST_CASE("FP32 WENO retains its nonlinear reconstruction across moisture and number scales", "[numerics][weno5][scale]")
{
    for (double scale : {1e-30, 1e-20, 1e-15, 1e-10, 1e-5, 1.0, 1e5, 1e10, 1e15, 1e20, 1e30})
    {
        CAPTURE(scale);
        const std::array<float, 5> q{float(.1 * scale), float(.6 * scale), float(.8 * scale), float(.7 * scale), float(.2 * scale)};
        const double expected = weno5_js_reference(q[0], q[1], q[2], q[3], q[4]);
        const float offset = weno5_face_offset_fp32(q[0], q[1], q[2], q[3], q[4]);
        CHECK(std::isfinite(offset));
        CHECK(double(q[2]) + offset == Approx(expected).margin(3e-7 * scale));
        CHECK(std::abs(offset) > 1e-4 * scale);
        CHECK(weno5_face_offset_fp32(q[2], q[2], q[2], q[2], q[2]) == 0);
    }
}

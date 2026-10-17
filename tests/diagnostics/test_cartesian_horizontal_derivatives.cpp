/**
 * @file test_cartesian_horizontal_derivatives.cpp
 * @brief Unit tests for the Cartesian flow-divergence / derivative helpers used
 *        by the headless diagnostic export on a Cartesian C-grid. Regression
 *        cover for the bug where cylindrical (1/r, periodic-theta) formulas were
 *        applied to a Cartesian box, corrupting `divergence` / `okubo_weiss` /
 *        `pressure_gradient_force_y` and blowing up at the i == 0 plane.
 */
#include "catch2/catch.hpp"
#include "diagnostics/cartesian_horizontal_derivatives.hpp"

#include <cmath>
#include <limits>

namespace
{
// Fill a Field3D from an analytic function of physical (x, y, z) = (i*dx, j*dy, k*dz).
template <typename Fn>
Field3D make_field(int NR, int NTH, int NZ, double dx, double dy, double dz, Fn fn)
{
    Field3D field(NR, NTH, NZ);
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                field(i, j, k) = static_cast<float>(fn(i * dx, j * dy, k * dz));
            }
        }
    }
    return field;
}
}  // namespace

TEST_CASE("cartesian ddx/ddy/ddz recover constant gradients everywhere", "[diagnostics][cartesian]")
{
    const int NR = 6, NTH = 5, NZ = 4;
    const double dx = 100.0, dy = 200.0, dz = 50.0;
    // u = 3x, v = -2y, w = 7z  ->  du/dx = 3, dv/dy = -2, dw/dz = 7.
    auto u = make_field(NR, NTH, NZ, dx, dy, dz, [](double x, double, double) { return 3.0 * x; });
    auto v = make_field(NR, NTH, NZ, dx, dy, dz, [](double, double y, double) { return -2.0 * y; });
    auto w = make_field(NR, NTH, NZ, dx, dy, dz, [](double, double, double z) { return 7.0 * z; });

    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                REQUIRE(tmv::cartesian_ddx(u, i, j, k, NR, dx) == Approx(3.0).margin(1e-3));
                REQUIRE(tmv::cartesian_ddy(v, i, j, k, NTH, dy) == Approx(-2.0).margin(1e-3));
                REQUIRE(tmv::cartesian_ddz(w, i, j, k, NZ, dz) == Approx(7.0).margin(1e-3));
            }
        }
    }
}

TEST_CASE("cartesian_flow_divergence recovers analytic divergence, including i==0",
          "[diagnostics][cartesian]")
{
    const int NR = 6, NTH = 5, NZ = 4;
    const double dx = 100.0, dy = 200.0, dz = 50.0;
    auto u = make_field(NR, NTH, NZ, dx, dy, dz, [](double x, double, double) { return 3.0 * x; });
    auto v = make_field(NR, NTH, NZ, dx, dy, dz, [](double, double y, double) { return -2.0 * y; });
    auto w = make_field(NR, NTH, NZ, dx, dy, dz, [](double, double, double z) { return 7.0 * z; });

    // Divergence = 3 - 2 + 7 = 8 everywhere.
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const float d =
                    tmv::cartesian_flow_divergence(i, j, k, NR, NTH, NZ, dx, dy, dz, u, v, w);
                REQUIRE(std::isfinite(d));
                REQUIRE(d == Approx(8.0).margin(1e-3));
            }
        }
    }

    // Regression: the old cylindrical form divided by r = i*dr = 0 at i == 0 and
    // produced NaN / garbage there. The Cartesian form must be finite and exact.
    for (int j = 0; j < NTH; ++j)
    {
        for (int k = 0; k < NZ; ++k)
        {
            const float d0 =
                tmv::cartesian_flow_divergence(0, j, k, NR, NTH, NZ, dx, dy, dz, u, v, w);
            REQUIRE(std::isfinite(d0));
            REQUIRE(d0 == Approx(8.0).margin(1e-3));
        }
    }
}

TEST_CASE("cartesian d2dy2 uses physical dy without periodic edge wrapping",
          "[diagnostics][cartesian][rectangular-grid]")
{
    const int NR = 3, NTH = 6, NZ = 3;
    const double dx = 100.0, dy = 250.0, dz = 50.0;
    auto quadratic = make_field(
        NR, NTH, NZ, dx, dy, dz,
        [](double, double y, double) { return 3.0 * y * y + 2.0 * y + 7.0; });

    for (int j = 0; j < NTH; ++j)
    {
        CAPTURE(j);
        REQUIRE(tmv::cartesian_d2dy2(quadratic, 1, j, 1, NTH, dy)
                == Approx(6.0).margin(1.0e-3));
    }
}

TEST_CASE("cartesian_flow_divergence is zero for non-divergent solid-body rotation",
          "[diagnostics][cartesian]")
{
    const int NR = 7, NTH = 7, NZ = 3;
    const double dx = 100.0, dy = 100.0, dz = 100.0;
    const double omega = 0.01;
    // u = -omega*y, v = omega*x, w = 0  ->  du/dx = dv/dy = 0  ->  divergence 0.
    auto u = make_field(NR, NTH, NZ, dx, dy, dz, [&](double, double y, double) { return -omega * y; });
    auto v = make_field(NR, NTH, NZ, dx, dy, dz, [&](double x, double, double) { return omega * x; });
    auto w = make_field(NR, NTH, NZ, dx, dy, dz, [](double, double, double) { return 0.0; });

    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                REQUIRE(tmv::cartesian_flow_divergence(i, j, k, NR, NTH, NZ, dx, dy, dz, u, v, w) ==
                        Approx(0.0).margin(1e-3));
            }
        }
    }
}

TEST_CASE("cartesian helpers reject bad spacing and propagate NaN", "[diagnostics][cartesian]")
{
    const int NR = 4, NTH = 4, NZ = 4;
    const double dx = 100.0, dy = 100.0, dz = 100.0;
    auto u = make_field(NR, NTH, NZ, dx, dy, dz, [](double x, double, double) { return x; });
    auto v = make_field(NR, NTH, NZ, dx, dy, dz, [](double, double y, double) { return y; });
    auto w = make_field(NR, NTH, NZ, dx, dy, dz, [](double, double, double) { return 0.0; });

    // Non-positive spacing -> NaN.
    REQUIRE(std::isnan(tmv::cartesian_ddx(u, 1, 1, 1, NR, 0.0)));
    REQUIRE(std::isnan(tmv::cartesian_ddy(v, 1, 1, 1, NTH, -5.0)));
    REQUIRE(std::isnan(tmv::cartesian_d2dy2(v, 1, 1, 1, NTH, 0.0)));
    REQUIRE(std::isnan(tmv::cartesian_ddz(w, 1, 1, 1, NZ, 0.0)));

    // A NaN neighbour poisons the divergence rather than silently reading 0.
    u(2, 1, 1) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(std::isnan(
        tmv::cartesian_flow_divergence(1, 1, 1, NR, NTH, NZ, dx, dy, dz, u, v, w)));
}

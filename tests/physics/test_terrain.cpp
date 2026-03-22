/**
 * @file test_terrain.cpp
 * @brief Unit tests for terrain schemes with analytical profile verification.
 *
 * Bell mountain (axisymmetric):
 *   h(x,y) = h0 * a^3 / (a^2 + x^2 + y^2)^1.5
 *   Peak: h(0,0) = h0
 *   Derivative: dh/dx = -3*h0*a^3*x / (a^2 + r^2)^2.5
 *
 * Bell mountain (2D ridge):
 *   h(x) = h0 / (1 + (x/a)^2)
 *   Peak: h(0) = h0
 */
#include "catch2/catch.hpp"
#include "physics/terrain_base.hpp"
#include "terrain/base/topography.hpp"
#include "core/simulation.hpp"

#include <cmath>

namespace
{

void setup_grid()
{
    NR = 16;
    NTH = 16;
    NZ = 16;
    dr = 1000.0;
    dz = 500.0;
    dtheta = 2.0 * 3.14159265358979323846 / NTH;
}

} // namespace

// ---- Factory tests ----

TEST_CASE("Terrain factory creates 'none' scheme", "[physics][terrain]")
{
    setup_grid();
    auto scheme = create_terrain_scheme("none");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Terrain factory creates 'bell' scheme", "[physics][terrain]")
{
    setup_grid();
    auto scheme = create_terrain_scheme("bell");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Terrain factory creates 'schar' scheme", "[physics][terrain]")
{
    setup_grid();
    auto scheme = create_terrain_scheme("schar");
    REQUIRE(scheme != nullptr);
}

// ---- Analytical topography function tests ----

TEST_CASE("Bell axisymmetric: peak height equals h0", "[physics][terrain][analytical]")
{
    // At r=0: h(0,0) = h0 * a^3 / (a^2)^1.5 = h0 * a^3 / a^3 = h0
    TerrainConfig::BellParams params;
    params.h0 = 500.0;
    params.a = 5000.0;
    params.axisymmetric = true;

    auto result = topography::eval_bell(0.0, 0.0, params);
    REQUIRE(result.h == Approx(500.0).epsilon(1e-12));
}

TEST_CASE("Bell axisymmetric: height decays with distance", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 1000.0;
    params.a = 5000.0;
    params.axisymmetric = true;

    // At r = a: h = h0 * a^3 / (a^2 + a^2)^1.5 = h0 / (2^1.5) = h0 / 2.828
    double expected_at_a = 1000.0 / std::pow(2.0, 1.5);
    auto result_at_a = topography::eval_bell(5000.0, 0.0, params);
    REQUIRE(result_at_a.h == Approx(expected_at_a).epsilon(1e-10));

    // At r = 2a: h = h0 * a^3 / (a^2 + 4a^2)^1.5 = h0 / 5^1.5
    double expected_at_2a = 1000.0 / std::pow(5.0, 1.5);
    auto result_at_2a = topography::eval_bell(10000.0, 0.0, params);
    REQUIRE(result_at_2a.h == Approx(expected_at_2a).epsilon(1e-10));

    // Monotone decay
    REQUIRE(result_at_a.h < 1000.0);
    REQUIRE(result_at_2a.h < result_at_a.h);
}

TEST_CASE("Bell axisymmetric: derivatives are zero at peak", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 500.0;
    params.a = 5000.0;
    params.axisymmetric = true;

    auto result = topography::eval_bell(0.0, 0.0, params);
    REQUIRE(result.hx == Approx(0.0).margin(1e-15));
    REQUIRE(result.hy == Approx(0.0).margin(1e-15));
}

TEST_CASE("Bell axisymmetric: derivative formula dh/dx = -3*h0*a^3*x / (a^2+r^2)^2.5", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 1000.0;
    params.a = 5000.0;
    params.axisymmetric = true;

    double x = 3000.0;
    double y = 4000.0;  // r = 5000 = a
    double r2 = x * x + y * y;
    double a = params.a;

    double expected_hx = -3.0 * params.h0 * a * a * a * x / std::pow(a * a + r2, 2.5);
    double expected_hy = -3.0 * params.h0 * a * a * a * y / std::pow(a * a + r2, 2.5);

    auto result = topography::eval_bell(x, y, params);
    REQUIRE(result.hx == Approx(expected_hx).epsilon(1e-10));
    REQUIRE(result.hy == Approx(expected_hy).epsilon(1e-10));
}

TEST_CASE("Bell axisymmetric: symmetry h(x,0) == h(0,x)", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 500.0;
    params.a = 5000.0;
    params.axisymmetric = true;

    auto h1 = topography::eval_bell(3000.0, 0.0, params);
    auto h2 = topography::eval_bell(0.0, 3000.0, params);
    REQUIRE(h1.h == Approx(h2.h).epsilon(1e-12));
}

TEST_CASE("Bell 2D ridge: peak height equals h0", "[physics][terrain][analytical]")
{
    // h(0) = h0 / (1 + 0) = h0
    TerrainConfig::BellParams params;
    params.h0 = 750.0;
    params.a = 5000.0;
    params.axisymmetric = false;

    auto result = topography::eval_bell(0.0, 0.0, params);
    REQUIRE(result.h == Approx(750.0).epsilon(1e-12));
}

TEST_CASE("Bell 2D ridge: h(a) = h0/2", "[physics][terrain][analytical]")
{
    // h(a) = h0 / (1 + 1) = h0/2
    TerrainConfig::BellParams params;
    params.h0 = 1000.0;
    params.a = 5000.0;
    params.axisymmetric = false;

    auto result = topography::eval_bell(5000.0, 0.0, params);
    REQUIRE(result.h == Approx(500.0).epsilon(1e-12));
}

TEST_CASE("Bell 2D ridge: y-invariant (hy = 0)", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 500.0;
    params.a = 5000.0;
    params.axisymmetric = false;

    auto result = topography::eval_bell(3000.0, 7000.0, params);
    REQUIRE(result.hy == 0.0);

    // Height depends only on x
    auto result2 = topography::eval_bell(3000.0, 0.0, params);
    REQUIRE(result.h == Approx(result2.h).epsilon(1e-12));
}

TEST_CASE("Bell 2D ridge: dh/dx = -2*h0*(x/a) / (a*(1+(x/a)^2)^2)", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 1000.0;
    params.a = 5000.0;
    params.axisymmetric = false;

    double x = 2500.0;
    double xa = x / params.a;
    double denom = 1.0 + xa * xa;
    double expected_hx = -2.0 * params.h0 * xa / (params.a * denom * denom);

    auto result = topography::eval_bell(x, 0.0, params);
    REQUIRE(result.hx == Approx(expected_hx).epsilon(1e-10));
}

TEST_CASE("Bell height is non-negative for positive h0", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 500.0;
    params.a = 5000.0;

    for (bool axi : {true, false})
    {
        params.axisymmetric = axi;
        for (double x = -20000.0; x <= 20000.0; x += 2000.0)
        {
            auto result = topography::eval_bell(x, 0.0, params);
            REQUIRE(result.h >= 0.0);
        }
    }
}

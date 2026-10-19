/**
 * @file test_staggered_derivatives.cpp
 * @brief Unit tests for StaggeredCartesianDerivatives and C-grid GridGeometry.
 *
 * Verifies the Cartesian C-grid staggered operators (grad/div on a staggered
 * mesh, flux-form continuity, upwind advection), the GridGeometry cell-center
 * arrays and spacing reciprocals, and the StaggerType parse/name round-trip.
 */

#include "catch2/catch.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/grid_geometry.hpp"
#include "core/runtime/simulation.hpp"
#include "numerics/derivatives/derivative_operators.hpp"

#include <cmath>

// =========================================================================
// StaggerType enum tests
// =========================================================================

TEST_CASE("StaggerType default is Collocated", "[core][stagger]")
{
    StaggerType st{};
    REQUIRE(st == StaggerType::Collocated);
    REQUIRE(static_cast<int>(st) == 0);
}

TEST_CASE("stagger_type_name returns canonical labels", "[core][stagger]")
{
    REQUIRE(std::string(stagger_type_name(StaggerType::Collocated)) == "collocated");
    REQUIRE(std::string(stagger_type_name(StaggerType::CGrid)) == "c_grid");
}

TEST_CASE("parse_stagger_type accepts canonical collocated labels", "[core][stagger]")
{
    StaggerType result = StaggerType::CGrid;
    REQUIRE(parse_stagger_type("collocated", result));
    REQUIRE(result == StaggerType::Collocated);

    result = StaggerType::CGrid;
    REQUIRE(parse_stagger_type("Collocated", result));
    REQUIRE(result == StaggerType::Collocated);

    result = StaggerType::CGrid;
    REQUIRE(parse_stagger_type("a_grid", result));
    REQUIRE(result == StaggerType::Collocated);
}

TEST_CASE("parse_stagger_type accepts canonical c_grid labels", "[core][stagger]")
{
    StaggerType result = StaggerType::Collocated;
    REQUIRE(parse_stagger_type("c_grid", result));
    REQUIRE(result == StaggerType::CGrid);

    result = StaggerType::Collocated;
    REQUIRE(parse_stagger_type("C_GRID", result));
    REQUIRE(result == StaggerType::CGrid);

    result = StaggerType::Collocated;
    REQUIRE(parse_stagger_type("cgrid", result));
    REQUIRE(result == StaggerType::CGrid);
}

TEST_CASE("parse_stagger_type rejects invalid input", "[core][stagger]")
{
    StaggerType result = StaggerType::CGrid;
    REQUIRE_FALSE(parse_stagger_type("", result));
    REQUIRE(result == StaggerType::CGrid);

    result = StaggerType::Collocated;
    REQUIRE_FALSE(parse_stagger_type("staggered", result));
    REQUIRE(result == StaggerType::Collocated);

    result = StaggerType::Collocated;
    REQUIRE_FALSE(parse_stagger_type("b_grid", result));
    REQUIRE(result == StaggerType::Collocated);
}

TEST_CASE("stagger_type_name labels round-trip through parse_stagger_type", "[core][stagger]")
{
    for (const auto st : {StaggerType::Collocated, StaggerType::CGrid})
    {
        const char* label = stagger_type_name(st);
        StaggerType parsed{};
        REQUIRE(parse_stagger_type(label, parsed));
        REQUIRE(parsed == st);
    }
}

// =========================================================================
// GridGeometry scalar-center and face-array tests
// =========================================================================

TEST_CASE("GridGeometry collocated mode records the collocated flag", "[core][grid_geometry]")
{
    GridGeometry geo;
    geo.initialize(10, 8, 20, 100.0, 50.0, 100.0,
                   CoordinateSystem::Cartesian, StaggerType::Collocated);

    REQUIRE(geo.staggered == false);
}

TEST_CASE("GridGeometry populates cell-center arrays and spacing reciprocals",
          "[core][grid_geometry]")
{
    const int nr = 10;
    const int nth = 8;
    const int nz = 20;
    const double dx_val = 100.0;
    const double dz_val = 50.0;
    const double dy_val = 100.0;

    GridGeometry geo;
    geo.initialize(nr, nth, nz, dx_val, dz_val, dy_val,
                   CoordinateSystem::Cartesian, StaggerType::CGrid);

    REQUIRE(geo.staggered == true);

    SECTION("cell-center arrays are half a cell above physical faces")
    {
        REQUIRE(geo.r[0] == Approx(0.5 * dx_val));
        REQUIRE(geo.r[5] == Approx(5.5 * dx_val));
        REQUIRE(geo.theta[0] == Approx(0.5 * dy_val));
        REQUIRE(geo.theta[3] == Approx(3.5 * dy_val));
        REQUIRE(geo.z[0] == Approx(0.5 * dz_val));
        REQUIRE(geo.z[10] == Approx(10.5 * dz_val));
    }

    SECTION("face arrays span the full physical domain")
    {
        REQUIRE(geo.r_faces.size() == static_cast<std::size_t>(nr + 1));
        REQUIRE(geo.theta_faces.size() == static_cast<std::size_t>(nth + 1));
        REQUIRE(geo.z_faces.size() == static_cast<std::size_t>(nz + 1));
        REQUIRE(geo.r_faces.front() == Approx(0.0));
        REQUIRE(geo.r_faces.back() == Approx(nr * dx_val));
        REQUIRE(geo.theta_faces.back() == Approx(nth * dy_val));
        REQUIRE(geo.z_faces.back() == Approx(nz * dz_val));
    }

    SECTION("spacing reciprocals match the supplied spacings")
    {
        REQUIRE(geo.inv_dr == Approx(1.0 / dx_val).epsilon(1e-14));
        REQUIRE(geo.inv_dz == Approx(1.0 / dz_val).epsilon(1e-14));
        REQUIRE(geo.inv_dtheta == Approx(1.0 / dy_val).epsilon(1e-14));
    }
}

// =========================================================================
// StaggeredCartesianDerivatives tests (Cartesian C-grid operator)
// =========================================================================

namespace
{

/// Cartesian C-grid fixture. u_x at x-face (i+1/2), u_y at y-face (j+1/2),
/// w at z-face (k+1/2), scalars at cell center.
struct CartCGridTestFixture
{
    static constexpr int NX = 8;
    static constexpr int NY = 8;
    static constexpr int NZ = 10;

    double dx_val = 100.0;
    double dy_val = 100.0;
    double dz_val = 50.0;

    Field3D make_field(float value = 0.0f) const
    {
        Field3D f;
        f.resize(NX, NY, NZ, value);
        return f;
    }

    StaggeredCartesianDerivatives make_deriv() const
    {
        return StaggeredCartesianDerivatives(dx_val, dy_val, dz_val, NX, NY, NZ);
    }
};

} // namespace

TEST_CASE("cart grad_x/y/z of linear scalar field returns constant slope",
          "[numerics][staggered_cartesian]")
{
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();

    const double ax = 3.0, ay = -2.0, az = 1.5;
    Field3D f = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                f[i][j][k] = static_cast<float>(ax * i * fix.dx_val
                                              + ay * j * fix.dy_val
                                              + az * k * fix.dz_val);

    for (int i = 0; i < fix.NX - 1; ++i)
        REQUIRE(stag.grad_x(f, i, 2, 3) == Approx(ax).epsilon(1e-5));
    for (int j = 0; j < fix.NY - 1; ++j)
        REQUIRE(stag.grad_y(f, 2, j, 3) == Approx(ay).epsilon(1e-5));
    for (int k = 0; k < fix.NZ - 1; ++k)
        REQUIRE(stag.grad_z(f, 2, 3, k) == Approx(az).epsilon(1e-5));
}

TEST_CASE("cart div_flux of uniform velocity is zero",
          "[numerics][staggered_cartesian]")
{
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();

    Field3D u_x = fix.make_field(0.7f);
    Field3D u_y = fix.make_field(-0.4f);
    Field3D w   = fix.make_field(0.3f);

    for (int i = 1; i < fix.NX - 1; ++i)
        for (int j = 1; j < fix.NY - 1; ++j)
            for (int k = 1; k < fix.NZ - 1; ++k)
                REQUIRE(std::abs(stag.div_flux(u_x, u_y, w, i, j, k)) < 1e-12);
}

TEST_CASE("cart div_flux of linear velocity matches analytic divergence",
          "[numerics][staggered_cartesian]")
{
    // u_x = a*x at x-faces, u_y = b*y at y-faces, w = c*z at z-faces.
    // div = a + b + c, exactly, for the adjacent-point staggered stencil.
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();

    const double a = 0.01, b = 0.02, c = 0.005;
    Field3D u_x = fix.make_field();
    Field3D u_y = fix.make_field();
    Field3D w   = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
            {
                // Face positions: x-face i at (i+1)*dx, y-face j at (j+1)*dy,
                // z-face k at (k+1)*dz (w[-1]=0 ground convention).
                u_x[i][j][k] = static_cast<float>(a * (i + 1) * fix.dx_val);
                u_y[i][j][k] = static_cast<float>(b * (j + 1) * fix.dy_val);
                w[i][j][k]   = static_cast<float>(c * (k + 1) * fix.dz_val);
            }

    const double expected = a + b + c;
    for (int i = 1; i < fix.NX - 1; ++i)
        for (int j = 1; j < fix.NY - 1; ++j)
            for (int k = 1; k < fix.NZ - 1; ++k)
                REQUIRE(stag.div_flux(u_x, u_y, w, i, j, k)
                        == Approx(expected).epsilon(1e-5));
}

TEST_CASE("cart div_flux_mass with zero velocity is zero",
          "[numerics][staggered_cartesian][conservation]")
{
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();

    Field3D rho = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                rho[i][j][k] = static_cast<float>(1.2 - 0.05 * k);

    Field3D zero = fix.make_field(0.0f);
    for (int i = 1; i < fix.NX; ++i)
        for (int j = 1; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                REQUIRE(stag.div_flux_mass(rho, zero, zero, zero, i, j, k) == 0.0);
}

TEST_CASE("cart div_flux_mass with uniform rho equals rho * div_flux",
          "[numerics][staggered_cartesian][conservation]")
{
    // For constant rho, div(rho u) = rho * div(u) at every interior cell.
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();

    const float rho_const = 1.3f;
    Field3D rho = fix.make_field(rho_const);

    Field3D u_x = fix.make_field();
    Field3D u_y = fix.make_field();
    Field3D w   = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
            {
                u_x[i][j][k] = 0.5f + 0.01f * i;
                u_y[i][j][k] = 0.2f - 0.02f * j;
                w[i][j][k]   = 0.1f - 0.005f * k;
            }

    for (int i = 1; i < fix.NX - 1; ++i)
        for (int j = 1; j < fix.NY - 1; ++j)
            for (int k = 1; k < fix.NZ - 1; ++k)
            {
                const double dm = stag.div_flux_mass(rho, u_x, u_y, w, i, j, k);
                const double dv = stag.div_flux(u_x, u_y, w, i, j, k);
                REQUIRE(dm == Approx(static_cast<double>(rho_const) * dv).epsilon(1e-10));
            }
}

TEST_CASE("cart staggered operator removes the checkerboard from the null space",
          "[numerics][staggered_cartesian][checkerboard]")
{
    // This is the crux of the Cartesian C-grid staggering. The collocated centered
    // 2-point operator annihilates the 2dx/2dy/2dz checkerboard field
    // phi[i][j][k] = (-1)^(i+j+k): (phi[i+1]-phi[i-1]) = 0. So the checkerboard
    // lives in the null space of CartesianDerivatives -> undamped computational
    // mode. The adjacent-point staggered grad_x/y/z does NOT annihilate it,
    // so the pressure-velocity coupling acts on it and it cannot grow unchecked.
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();
    CartesianDerivatives collocated(fix.dx_val, fix.dy_val, fix.dz_val);

    Field3D phi = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                phi[i][j][k] = ((i + j + k) % 2 == 0) ? 1.0f : -1.0f;

    for (int i = 1; i < fix.NX - 1; ++i)
        for (int j = 1; j < fix.NY - 1; ++j)
            for (int k = 1; k < fix.NZ - 1; ++k)
            {
                // Collocated centered FD is BLIND to the checkerboard.
                REQUIRE(collocated.di(phi, i, j, k) == Approx(0.0).margin(1e-12));
                REQUIRE(collocated.dj(phi, i, j, k) == Approx(0.0).margin(1e-12));
                REQUIRE(collocated.dk(phi, i, j, k) == Approx(0.0).margin(1e-12));

                // Staggered adjacent-point gradient SEES the checkerboard
                // (magnitude 2/dx, 2/dy, 2/dz on an alternating +/-1 field).
                REQUIRE(std::abs(stag.grad_x(phi, i, j, k)) == Approx(2.0 / fix.dx_val).epsilon(1e-6));
                REQUIRE(std::abs(stag.grad_y(phi, i, j, k)) == Approx(2.0 / fix.dy_val).epsilon(1e-6));
                REQUIRE(std::abs(stag.grad_z(phi, i, j, k)) == Approx(2.0 / fix.dz_val).epsilon(1e-6));
            }
}

TEST_CASE("cart interp center<->face round-trips on a linear field",
          "[numerics][staggered_cartesian]")
{
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();

    // Linear center field f[i] = i*dx. interp_to_x_face -> (i+0.5)*dx.
    Field3D f = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                f[i][j][k] = static_cast<float>(i * fix.dx_val);

    for (int i = 0; i < fix.NX - 1; ++i)
        REQUIRE(stag.interp_to_x_face(f, i, 2, 3)
                == Approx((i + 0.5) * fix.dx_val).epsilon(1e-5));

    // Face field g[i] = (i+0.5)*dx -> interp_from_x_face -> i*dx for i>=1.
    Field3D g = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                g[i][j][k] = static_cast<float>((i + 0.5) * fix.dx_val);

    for (int i = 1; i < fix.NX; ++i)
        REQUIRE(stag.interp_from_x_face(g, i, 2, 3)
                == Approx(i * fix.dx_val).epsilon(1e-5));
}

TEST_CASE("cart div_flux_z surface and lid short-circuits are finite",
          "[numerics][staggered_cartesian]")
{
    // k=0 (surface, w[-1]=0) and k=NZ-1 (lid, w[NZ-1]=0) must not OOB or NaN.
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();

    Field3D w = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ - 1; ++k)
                w[i][j][k] = 0.2f * static_cast<float>(k + 1);
    // w[NZ-1] = 0 (lid).

    for (int i = 1; i < fix.NX - 1; ++i)
        for (int j = 1; j < fix.NY - 1; ++j)
        {
            // Surface: (w[0] - 0)/dz.
            const double d_surf = stag.div_flux_z(w, i, j, 0);
            REQUIRE(std::isfinite(d_surf));
            REQUIRE(d_surf == Approx(static_cast<double>(w[i][j][0]) / fix.dz_val).epsilon(1e-6));
            // Lid: (0 - w[NZ-2])/dz.
            const double d_lid = stag.div_flux_z(w, i, j, fix.NZ - 1);
            REQUIRE(std::isfinite(d_lid));
            REQUIRE(d_lid == Approx(-static_cast<double>(w[i][j][fix.NZ - 2]) / fix.dz_val).epsilon(1e-6));
        }
}

// =========================================================================
// Upwind-biased momentum-advection derivatives (docs/Stability.md fix):
// odd-order upwind carries scale-selective dissipation that damps the 2dx
// mode centered advection leaves undamped. Order degrades 5 -> 3 -> 1 within
// 3 / 2 cells of the boundary. CartCGridTestFixture is NX=NY=8, NZ=10, so
// the achievable half-width min(idx, N-1-idx) is 1 at index 1, 2 at index 2,
// and 3 (5th-order) at indices 3..(N-4).
// =========================================================================

TEST_CASE("cart upwind_dx/dy/dz are exact on a linear field at every order and sign",
          "[numerics][staggered_cartesian][upwind]")
{
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();

    const double ax = 3.0, ay = -2.0, az = 1.5;
    Field3D f = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                f[i][j][k] = static_cast<float>(ax * i * fix.dx_val
                                              + ay * j * fix.dy_val
                                              + az * k * fix.dz_val);

    // Upwind FD is consistent: exact for a linear field, independent of the
    // advecting-velocity sign and of the boundary-degraded order.
    for (double a : {2.0, -2.0})
    {
        for (int i : {1, 2, 3})   // half = 1, 2, 3 -> order 1, 3, 5
            REQUIRE(stag.upwind_dx(f, a, i, 4, 5) == Approx(ax).epsilon(1e-4));
        for (int j : {1, 2, 3})
            REQUIRE(stag.upwind_dy(f, a, 4, j, 5) == Approx(ay).epsilon(1e-4));
        for (int k : {1, 2, 3})
            REQUIRE(stag.upwind_dz(f, a, 4, 5, k) == Approx(az).epsilon(1e-4));
    }
}

TEST_CASE("cart upwind advection damps the 2dx mode that centered advection ignores",
          "[numerics][staggered_cartesian][upwind][checkerboard]")
{
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();
    CartesianDerivatives collocated(fix.dx_val, fix.dy_val, fix.dz_val);

    // Field alternating in i only: f[i] = (-1)^i. Centered d/dx is blind
    // (f[i+1] - f[i-1] = 0); odd-order upwind sees it.
    Field3D f = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                f[i][j][k] = (i % 2 == 0) ? 1.0f : -1.0f;

    const int i5 = 4;          // half = min(4, 3) = 3 -> 5th order
    const int jc = 4, kc = 5;
    REQUIRE(f[i5][jc][kc] == Approx(1.0));   // i even -> +1

    // Centered FD annihilates the mode; 5th-order upwind gives 64/(60 dx).
    REQUIRE(collocated.di(f, i5, jc, kc) == Approx(0.0).margin(1e-12));
    const double mag = 64.0 / (60.0 * fix.dx_val);
    REQUIRE(stag.upwind_dx(f, +1.0, i5, jc, kc) == Approx(+mag).epsilon(1e-5));
    REQUIRE(stag.upwind_dx(f, -1.0, i5, jc, kc) == Approx(-mag).epsilon(1e-5));

    // The advective tendency -a*df/dx opposes the local value for BOTH wind
    // directions and at every order -> it damps the mode (the property that
    // removes the surface instability).
    for (int i_cell : {4, 2, 1})   // half = 3, 2, 1 -> order 5, 3, 1
    {
        const double fi = static_cast<double>(f[i_cell][jc][kc]);
        for (double a : {3.0, -3.0})
        {
            const double tend = -a * stag.upwind_dx(f, a, i_cell, jc, kc);
            REQUIRE(tend * fi < 0.0);
        }
    }
}

TEST_CASE("cart upwind derivatives return zero at the boundary face and never read OOB",
          "[numerics][staggered_cartesian][upwind]")
{
    CartCGridTestFixture fix;
    auto stag = fix.make_deriv();

    Field3D f = fix.make_field();
    for (int i = 0; i < fix.NX; ++i)
        for (int j = 0; j < fix.NY; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                f[i][j][k] = static_cast<float>(0.3 * i - 0.1 * j + 0.2 * k);

    // half <= 0 at a boundary index -> 0 (no neighbor on one side, no OOB).
    REQUIRE(stag.upwind_dx(f, 1.0, 0, 4, 5) == 0.0);
    REQUIRE(stag.upwind_dx(f, 1.0, fix.NX - 1, 4, 5) == 0.0);
    REQUIRE(stag.upwind_dy(f, 1.0, 4, 0, 5) == 0.0);
    REQUIRE(stag.upwind_dy(f, 1.0, 4, fix.NY - 1, 5) == 0.0);
    REQUIRE(stag.upwind_dz(f, 1.0, 4, 5, 0) == 0.0);
    REQUIRE(stag.upwind_dz(f, 1.0, 4, 5, fix.NZ - 1) == 0.0);

    // First/last interior index uses 1st-order and must stay in bounds + finite.
    for (double a : {1.0, -1.0})
    {
        REQUIRE(std::isfinite(stag.upwind_dx(f, a, 1, 4, 5)));
        REQUIRE(std::isfinite(stag.upwind_dx(f, a, fix.NX - 2, 4, 5)));
        REQUIRE(std::isfinite(stag.upwind_dz(f, a, 4, 5, 1)));
        REQUIRE(std::isfinite(stag.upwind_dz(f, a, 4, 5, fix.NZ - 2)));
    }
}

// =========================================================================
// Spatial convergence order of the production C-grid operators (science audit:
// "no test pins any convergence order of the production path"). The centered
// staggered gradient (PGF) and divergence (continuity) are 2nd-order; this
// measures the OBSERVED order on a smooth field under grid refinement - the
// refinement-clean signal (the time integrator is 1st-order Forward Euler,
// characterized separately in the legacy MMS results). Fields are fp32 (the
// production type), so refinement stops at N=64: the differencing amplifies
// fp32 storage noise by 1/dx, and beyond ~64 that rivals the O(dx^2) error.
// =========================================================================
namespace
{
constexpr double kConvL = 1.0;  // domain length

// L2 error of staggered grad_x vs the analytic derivative of sin(2 pi x / L)
// at resolution N (dx = L/N); grad_x lives at the x-face (i+1/2)*dx.
double grad_x_l2_error(int N)
{
    const double dx = kConvL / N;
    const int NY = 4, NZ = 4;
    StaggeredCartesianDerivatives stag(dx, dx, dx, N, NY, NZ);
    Field3D f;
    f.resize(N, NY, NZ, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k)
                f[i][j][k] = static_cast<float>(std::sin(2.0 * M_PI * i * dx / kConvL));
    double err2 = 0.0;
    int cnt = 0;
    for (int i = 1; i < N - 2; ++i)
    {
        const double got = stag.grad_x(f, i, 2, 2);
        const double xf = (i + 0.5) * dx;
        const double ana = (2.0 * M_PI / kConvL) * std::cos(2.0 * M_PI * xf / kConvL);
        err2 += (got - ana) * (got - ana);
        ++cnt;
    }
    return std::sqrt(err2 / cnt);
}

// L2 error of staggered div_flux vs the analytic divergence of u_x =
// sin(2 pi x / L) (u_y = w = 0); div lives at the cell center i*dx.
double div_flux_l2_error(int N)
{
    const double dx = kConvL / N;
    const int NY = 4, NZ = 4;
    StaggeredCartesianDerivatives stag(dx, dx, dx, N, NY, NZ);
    Field3D ux, zero;
    ux.resize(N, NY, NZ, 0.0f);
    zero.resize(N, NY, NZ, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k)
                ux[i][j][k] =
                    static_cast<float>(std::sin(2.0 * M_PI * (i + 0.5) * dx / kConvL));
    double err2 = 0.0;
    int cnt = 0;
    for (int i = 2; i < N - 2; ++i)
    {
        const double got = stag.div_flux(ux, zero, zero, i, 2, 2);
        const double xc = i * dx;
        const double ana = (2.0 * M_PI / kConvL) * std::cos(2.0 * M_PI * xc / kConvL);
        err2 += (got - ana) * (got - ana);
        ++cnt;
    }
    return std::sqrt(err2 / cnt);
}

double observed_order(double e_coarse, double e_fine)
{
    return std::log2(e_coarse / e_fine);
}
}  // namespace

TEST_CASE("cart C-grid gradient and divergence converge at 2nd order",
          "[numerics][staggered_cartesian][convergence]")
{
    // grad_x: the 2nd-order centered staggered gradient (pressure-gradient op).
    const double g16 = grad_x_l2_error(16);
    const double g32 = grad_x_l2_error(32);
    const double g64 = grad_x_l2_error(64);
    REQUIRE(g32 < g16);  // error decreases under refinement
    REQUIRE(observed_order(g16, g32) == Approx(2.0).margin(0.2));
    REQUIRE(observed_order(g32, g64) == Approx(2.0).margin(0.2));

    // div_flux: the 2nd-order centered staggered divergence (continuity op).
    const double d16 = div_flux_l2_error(16);
    const double d32 = div_flux_l2_error(32);
    const double d64 = div_flux_l2_error(64);
    REQUIRE(d32 < d16);
    REQUIRE(observed_order(d16, d32) == Approx(2.0).margin(0.2));
    REQUIRE(observed_order(d32, d64) == Approx(2.0).margin(0.2));
}

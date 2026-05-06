/**
 * @file test_staggered_derivatives.cpp
 * @brief Unit tests for StaggeredCylindricalDerivatives and C-grid GridGeometry.
 *
 * Phase C.1 verification: div_flux (uniform flow = 0, linear flow = analytical),
 * grad_r against known function, GridGeometry face array correctness,
 * StaggerType parse/name round-trip.
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
// GridGeometry face array tests
// =========================================================================

TEST_CASE("GridGeometry collocated mode has empty face arrays", "[core][grid_geometry]")
{
    GridGeometry geo;
    geo.initialize(10, 8, 20, 100.0, 50.0, 2.0 * M_PI / 8,
                   CoordinateSystem::Cylindrical, StaggerType::Collocated);

    REQUIRE(geo.staggered == false);
    REQUIRE(geo.r_face.empty());
    REQUIRE(geo.r_face_inv.empty());
    REQUIRE(geo.z_face.empty());
}

TEST_CASE("GridGeometry C-grid populates face arrays correctly", "[core][grid_geometry]")
{
    const int nr = 10;
    const int nth = 8;
    const int nz = 20;
    const double dr_val = 100.0;
    const double dz_val = 50.0;
    const double dtheta_val = 2.0 * M_PI / nth;

    GridGeometry geo;
    geo.initialize(nr, nth, nz, dr_val, dz_val, dtheta_val,
                   CoordinateSystem::Cylindrical, StaggerType::CGrid);

    REQUIRE(geo.staggered == true);
    REQUIRE(geo.r_face.size() == static_cast<size_t>(nr));
    REQUIRE(geo.r_face_inv.size() == static_cast<size_t>(nr));
    REQUIRE(geo.z_face.size() == static_cast<size_t>(nz));

    SECTION("r_face positions are at half-grid offsets")
    {
        for (int i = 0; i < nr; ++i)
        {
            const double expected = (i + 0.5) * dr_val;
            REQUIRE(geo.r_face[i] == Approx(expected).epsilon(1e-14));
            REQUIRE(geo.r_face_inv[i] == Approx(1.0 / expected).epsilon(1e-14));
        }
    }

    SECTION("z_face positions are at half-grid offsets")
    {
        for (int k = 0; k < nz; ++k)
        {
            const double expected = (k + 0.5) * dz_val;
            REQUIRE(geo.z_face[k] == Approx(expected).epsilon(1e-14));
        }
    }

    SECTION("cell-center arrays are still correct")
    {
        REQUIRE(geo.r[0] == Approx(0.0));
        REQUIRE(geo.r[5] == Approx(5.0 * dr_val));
        REQUIRE(geo.z[0] == Approx(0.0));
        REQUIRE(geo.z[10] == Approx(10.0 * dz_val));
        REQUIRE(geo.r_inv[0] == 0.0);
        REQUIRE(geo.r_inv[1] == Approx(1.0 / dr_val).epsilon(1e-14));
    }
}

// =========================================================================
// StaggeredCylindricalDerivatives tests
// =========================================================================

namespace
{

/// Helper: create a C-grid geometry and matching fields for testing.
struct CGridTestFixture
{
    static constexpr int NR  = 10;
    static constexpr int NTH = 8;
    static constexpr int NZ  = 10;

    double dr_val;
    double dz_val;
    double dtheta_val;
    GridGeometry geo;

    CGridTestFixture()
    {
        dr_val = 100.0;
        dz_val = 50.0;
        dtheta_val = 2.0 * M_PI / NTH;
        geo.initialize(NR, NTH, NZ, dr_val, dz_val, dtheta_val,
                       CoordinateSystem::Cylindrical, StaggerType::CGrid);
    }

    Field3D make_field(float value = 0.0f) const
    {
        Field3D f;
        f.resize(NR, NTH, NZ, value);
        return f;
    }
};

} // namespace

TEST_CASE("div_flux uniform radial flow", "[numerics][staggered_derivatives]")
{
    // Uniform u_r = C at all r-faces.  For incompressible flow in a cylinder,
    // d(r*u)/dr = C (constant), so (1/r) d(r*u)/dr = C/r != 0.
    // But a *truly* uniform face velocity gives non-zero divergence in
    // cylindrical coords. The only way to get zero radial divergence
    // everywhere is u_face[i] proportional to 1/r_face[i].
    //
    // Test: set u_face such that r_face[i]*u[i] = constant (incompressible
    // source-free flow). Then div_flux_r should be zero at all interior cells.

    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D u_face = fix.make_field(0.0f);

    const double flux_constant = 1000.0; // r*u = const
    for (int i = 0; i < fix.NR; ++i)
        for (int j = 0; j < fix.NTH; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                u_face[i][j][k] = static_cast<float>(flux_constant / fix.geo.r_face[i]);

    // Check div_flux_r at interior cells (i >= 1).
    // Tolerance accounts for float32 field storage: storing float(C/r_face)
    // introduces ~C*eps_f32 flux error. After differencing and dividing by
    // r*dr, residual is O(eps_f32 * C / dr^2) ~ 1e-7 * 1000 / 10000 ~ 1e-8.
    for (int i = 1; i < fix.NR - 1; ++i)
    {
        for (int j = 0; j < fix.NTH; ++j)
        {
            for (int k = 1; k < fix.NZ - 1; ++k)
            {
                const double div_r = stag.div_flux_r(u_face, i, j, k);
                REQUIRE(std::abs(div_r) < 1e-7);
            }
        }
    }
}

TEST_CASE("div_flux axis formula for uniform face velocity", "[numerics][staggered_derivatives]")
{
    // At i=0, div_r = 2*u[0]/dr.  Verify directly.
    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D u_face = fix.make_field(0.0f);

    const float u_val = 5.0f;
    for (int j = 0; j < fix.NTH; ++j)
        for (int k = 0; k < fix.NZ; ++k)
            u_face[0][j][k] = u_val;

    for (int j = 0; j < fix.NTH; ++j)
    {
        for (int k = 0; k < fix.NZ; ++k)
        {
            const double div_r = stag.div_flux_r(u_face, 0, j, k);
            const double expected = 2.0 * u_val / fix.dr_val;
            REQUIRE(div_r == Approx(expected).epsilon(1e-14));
        }
    }
}

TEST_CASE("div_flux linear radial flow matches analytical", "[numerics][staggered_derivatives]")
{
    // Set u_face[i] = alpha * r_face[i]  (linear in r).
    // Then r*u = alpha * r^2, so d(r*u)/dr = 2*alpha*r.
    // (1/r) d(r*u)/dr = 2*alpha.
    // Discrete: (r_face[i]*u[i] - r_face[i-1]*u[i-1]) / (r[i]*dr)
    //         = alpha * (r_face[i]^2 - r_face[i-1]^2) / (r[i]*dr)
    //         = alpha * ((i+0.5)^2 - (i-0.5)^2) * dr / (i*dr)
    //         = alpha * 2*i * dr / (i * dr) = 2*alpha.  Exact.

    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D u_face = fix.make_field(0.0f);

    const double alpha = 0.01;
    for (int i = 0; i < fix.NR; ++i)
        for (int j = 0; j < fix.NTH; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                u_face[i][j][k] = static_cast<float>(alpha * fix.geo.r_face[i]);

    // At axis (i=0): div_r = 2*u[0]/dr = 2*alpha*r_face[0]/dr
    //              = 2*alpha*0.5*dr/dr = alpha.
    // Wait -- that's NOT 2*alpha.  The axis formula is a special case.
    // The continuous limit at r=0 for u = alpha*r is:
    //   lim_{r->0} (1/r) d(r * alpha*r)/dr = lim (1/r)*2*alpha*r = 2*alpha.
    // The discrete axis formula: 2*u[0]/dr = 2*alpha*r_face[0]/dr
    //   = 2*alpha*(0.5*dr)/dr = alpha.
    // This is the discretization error at the axis cell (O(dr)).
    // For interior cells, the discrete result IS exactly 2*alpha.

    const double expected_interior = 2.0 * alpha;
    for (int i = 1; i < fix.NR - 1; ++i)
    {
        const double div_r = stag.div_flux_r(u_face, i, 0, 1);
        REQUIRE(div_r == Approx(expected_interior).epsilon(1e-10));
    }

    // Axis cell: discrete result is alpha (first-order axis truncation)
    const double div_axis = stag.div_flux_r(u_face, 0, 0, 1);
    const double expected_axis = 2.0 * alpha * fix.geo.r_face[0] / fix.dr_val;
    REQUIRE(div_axis == Approx(expected_axis).epsilon(1e-14));
}

TEST_CASE("div_flux total with zero velocity is zero", "[numerics][staggered_derivatives]")
{
    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D u_face = fix.make_field(0.0f);
    Field3D v_face = fix.make_field(0.0f);
    Field3D w_face = fix.make_field(0.0f);

    for (int i = 1; i < fix.NR - 1; ++i)
    {
        for (int j = 0; j < fix.NTH; ++j)
        {
            for (int k = 1; k < fix.NZ - 1; ++k)
            {
                const double div = stag.div_flux(u_face, v_face, w_face, i, j, k);
                REQUIRE(std::abs(div) < 1e-15);
            }
        }
    }
}

TEST_CASE("grad_r of linear scalar field", "[numerics][staggered_derivatives]")
{
    // f[i] = beta * r[i] = beta * i * dr.
    // grad_r at face (i+1/2) = (f[i+1] - f[i]) / dr = beta.

    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D f = fix.make_field(0.0f);
    const double beta = 3.0;
    for (int i = 0; i < fix.NR; ++i)
        for (int j = 0; j < fix.NTH; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                f[i][j][k] = static_cast<float>(beta * fix.geo.r[i]);

    for (int i = 0; i < fix.NR - 1; ++i)
    {
        const double gr = stag.grad_r(f, i, 0, 1);
        REQUIRE(gr == Approx(beta).epsilon(1e-6));
    }
}

TEST_CASE("grad_r of quadratic scalar field", "[numerics][staggered_derivatives]")
{
    // f[i] = r[i]^2.
    // grad_r at face (i+1/2) = (r[i+1]^2 - r[i]^2) / dr
    //                        = ((i+1)^2 - i^2) * dr = (2i+1) * dr.

    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D f = fix.make_field(0.0f);
    for (int i = 0; i < fix.NR; ++i)
        for (int j = 0; j < fix.NTH; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                f[i][j][k] = static_cast<float>(fix.geo.r[i] * fix.geo.r[i]);

    for (int i = 0; i < fix.NR - 1; ++i)
    {
        const double gr = stag.grad_r(f, i, 0, 1);
        const double expected = (2.0 * i + 1.0) * fix.dr_val;
        REQUIRE(gr == Approx(expected).epsilon(1e-6));
    }
}

TEST_CASE("grad_z of linear scalar field", "[numerics][staggered_derivatives]")
{
    // f[k] = gamma * z[k].
    // grad_z at face (k+1/2) = gamma.

    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D f = fix.make_field(0.0f);
    const double gamma = 2.5;
    for (int i = 0; i < fix.NR; ++i)
        for (int j = 0; j < fix.NTH; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                f[i][j][k] = static_cast<float>(gamma * fix.geo.z[k]);

    for (int k = 0; k < fix.NZ - 1; ++k)
    {
        const double gz = stag.grad_z(f, 1, 0, k);
        REQUIRE(gz == Approx(gamma).epsilon(1e-6));
    }
}

TEST_CASE("interp_to_r_face and interp_from_r_face round-trip", "[numerics][staggered_derivatives]")
{
    // For a linear field f[i] = i*dr, interp_to_r_face gives (i+0.5)*dr,
    // and interp_from_r_face of a linear face-field g[i] = (i+0.5)*dr
    // gives back i*dr (for i >= 1).

    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D f_center = fix.make_field(0.0f);
    for (int i = 0; i < fix.NR; ++i)
        for (int j = 0; j < fix.NTH; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                f_center[i][j][k] = static_cast<float>(fix.geo.r[i]);

    // Center -> face
    for (int i = 0; i < fix.NR - 1; ++i)
    {
        const double val = stag.interp_to_r_face(f_center, i, 0, 1);
        REQUIRE(val == Approx(fix.geo.r_face[i]).epsilon(1e-6));
    }

    // Build face field and go face -> center
    Field3D f_face = fix.make_field(0.0f);
    for (int i = 0; i < fix.NR; ++i)
        for (int j = 0; j < fix.NTH; ++j)
            for (int k = 0; k < fix.NZ; ++k)
                f_face[i][j][k] = static_cast<float>(fix.geo.r_face[i]);

    for (int i = 1; i < fix.NR; ++i)
    {
        const double val = stag.interp_from_r_face(f_face, i, 0, 1);
        REQUIRE(val == Approx(fix.geo.r[i]).epsilon(1e-6));
    }
}

TEST_CASE("interp_from_r_face axis uses half of f[0]", "[numerics][staggered_derivatives]")
{
    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D f_face = fix.make_field(0.0f);
    f_face[0][0][1] = 10.0f;

    const double val = stag.interp_from_r_face(f_face, 0, 0, 1);
    REQUIRE(val == Approx(5.0).epsilon(1e-14));
}

TEST_CASE("div_flux_theta periodic wrap", "[numerics][staggered_derivatives]")
{
    // Uniform v_theta at all theta-faces => div_flux_theta = 0
    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D v_face = fix.make_field(7.0f); // uniform

    for (int i = 1; i < fix.NR - 1; ++i)
    {
        for (int j = 0; j < fix.NTH; ++j)
        {
            const double div_th = stag.div_flux_theta(v_face, i, j, 1);
            REQUIRE(std::abs(div_th) < 1e-14);
        }
    }
}

TEST_CASE("div_flux_z with uniform w is zero", "[numerics][staggered_derivatives]")
{
    CGridTestFixture fix;
    StaggeredCylindricalDerivatives stag(fix.geo, fix.NTH);

    Field3D w_face = fix.make_field(3.0f); // uniform

    // Interior cells: w[k] - w[k-1] = 0 for k >= 1
    for (int k = 1; k < fix.NZ - 1; ++k)
    {
        const double div_z = stag.div_flux_z(w_face, 1, 0, k);
        REQUIRE(std::abs(div_z) < 1e-14);
    }
}

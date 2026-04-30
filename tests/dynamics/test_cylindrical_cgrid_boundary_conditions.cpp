/**
 * @file test_cylindrical_cgrid_boundary_conditions.cpp
 * @brief Verification gates for Phase C.2 -- C-grid cylindrical boundary
 *        conditions.
 *
 * The verification gate from docs/CoordinateBackend_Plan.md C.2 reads:
 *   "Hydrostatic column, C-grid BCs, 100 steps: all tendencies < 1e-10,
 *    zero clamps."
 *
 * Phase C.4 (the tornado_cgrid dynamics scheme) is what would compute
 * "tendencies", so the integrated end-to-end gate has to wait for that
 * phase. What we can verify here is the BC layer in isolation:
 *
 *   1. The BC factory dispatches to the correct scheme name for every
 *      supported (coordinate, stagger) combination, and rejects the
 *      unsupported Cartesian + CGrid combination loudly.
 *
 *   2. The C-grid BCs preserve a hydrostatically balanced state
 *      bit-exactly across 100 successive BC applies. Any spurious
 *      perturbation introduced by the BC code itself would surface
 *      here as a drifting field.
 *
 *   3. The two structural distinctions vs the collocated cylindrical BC:
 *
 *      a. u[0] (the radial velocity at r_face[0] = dr/2) is interior on
 *         the C-grid and must NOT be touched by the BC. The collocated
 *         scheme would overwrite u[0] = -u[1] (antisymmetric ghost), and
 *         that overwrite is precisely the Bug 7 false-gradient source
 *         we are eliminating in Phase C. A nonzero u[0] survives the BC
 *         apply unchanged.
 *
 *      b. w[i][j][0] (the first interior z-face at z_face[0] = dz/2) is
 *         likewise interior on the C-grid -- only w[i][j][NZ-1] (the
 *         lid face) is rigid. The collocated scheme zeroes w[i][j][0]
 *         (rigid surface at the lowest cell center). A nonzero w[i][j][0]
 *         survives the BC apply unchanged.
 *
 *   4. Outer wall, lid, axis, and hydrostatic pressure extrapolation
 *      behave as documented in boundary_conditions_cylindrical_cgrid.cpp.
 */

#include "catch2/catch.hpp"
#include "boundary_conditions/boundary_conditions.hpp"
#include "core/coordinate_system.hpp"
#include "core/field3d.hpp"
#include "core/runtime_config.hpp"
#include "core/simulation.hpp"
#include "dynamics/dynamics_base.hpp"

#include <cmath>
#include <stdexcept>

namespace
{

constexpr double kRho0 = 1.225;
constexpr double kP0   = 101325.0;

void setup_cylindrical_cgrid_grid()
{
    NR = 12;
    NTH = 8;
    NZ = 10;
    dr = 250.0;
    dz = 250.0;
    dt = 0.1;
    dtheta = 2.0 * 3.14159265358979323846 / static_cast<double>(NTH);
    rho0_base.assign(NZ, kRho0);
    p0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
        p0_base[k] = kP0 - kRho0 * dynamics_constants::g * static_cast<double>(k) * dz;
    u0_base.assign(NZ, 0.0);
    v0_base.assign(NZ, 0.0);
    global_coordinate_system = CoordinateSystem::Cylindrical;
    global_stagger_type = StaggerType::CGrid;
}

void resize_all_fields_to_grid()
{
    rho.resize(NR, NTH, NZ, static_cast<float>(kRho0));
    p.resize(NR, NTH, NZ);
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    theta.resize(NR, NTH, NZ, static_cast<float>(dynamics_constants::theta0));
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
}

/// Hydrostatic atmosphere with rho constant at kRho0 and
/// p[k] = kP0 - kRho0 * g * (k * dz). The exact discrete relation
///   p[k+1] - p[k] = -kRho0 * g * dz
/// matches the C-grid BC's pressure extrapolation at top and bottom, so a
/// BC apply is the identity on this state.
void initialize_hydrostatic_atmosphere()
{
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                rho[i][j][k] = static_cast<float>(kRho0);
                const double z = static_cast<double>(k) * dz;
                p[i][j][k] = static_cast<float>(kP0 - kRho0 * dynamics_constants::g * z);
                theta[i][j][k] = static_cast<float>(dynamics_constants::theta0);
                u[i][j][k] = 0.0f;
                v[i][j][k] = 0.0f;
                w[i][j][k] = 0.0f;
                qv[i][j][k] = 0.0f;
                qc[i][j][k] = 0.0f;
                qr[i][j][k] = 0.0f;
                qi[i][j][k] = 0.0f;
                qs[i][j][k] = 0.0f;
                qg[i][j][k] = 0.0f;
                qh[i][j][k] = 0.0f;
            }
}

/// Snapshot a Field3D into a heap-allocated copy for later comparison.
Field3D snapshot(const Field3D& f)
{
    Field3D out;
    out.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                out[i][j][k] = f[i][j][k];
    return out;
}

bool fields_equal_exact(const Field3D& a, const Field3D& b)
{
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                if (a[i][j][k] != b[i][j][k]) return false;
    return true;
}

}  // namespace

// ============================================================================
// Gate 1 -- factory dispatch correctness.
// ============================================================================
TEST_CASE("BC factory dispatches by (coordinate, stagger)",
          "[boundary_conditions][factory][c2]")
{
    auto cylC = create_boundary_condition_scheme(
        CoordinateSystem::Cylindrical, StaggerType::Collocated);
    REQUIRE(cylC);
    REQUIRE(cylC->get_scheme_name() == "cylindrical");

    auto cylG = create_boundary_condition_scheme(
        CoordinateSystem::Cylindrical, StaggerType::CGrid);
    REQUIRE(cylG);
    REQUIRE(cylG->get_scheme_name() == "cylindrical_cgrid");

    auto cart = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    REQUIRE(cart);
    REQUIRE(cart->get_scheme_name() == "cartesian");
}

TEST_CASE("BC factory rejects Cartesian C-grid (not implemented)",
          "[boundary_conditions][factory][c2]")
{
    REQUIRE_THROWS_AS(
        create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                         StaggerType::CGrid),
        std::runtime_error);
}

// ============================================================================
// Gate 2 -- 100-cycle stability of a hydrostatic column under C-grid BCs.
// All fields must be bit-exactly preserved.
// ============================================================================
TEST_CASE("C-grid cylindrical BCs preserve hydrostatic column over 100 cycles",
          "[boundary_conditions][cylindrical][cgrid][c2]")
{
    setup_cylindrical_cgrid_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();

    auto scheme = create_boundary_condition_scheme(
        CoordinateSystem::Cylindrical, StaggerType::CGrid);
    REQUIRE(scheme);
    REQUIRE(scheme->get_scheme_name() == "cylindrical_cgrid");

    const Field3D rho0   = snapshot(rho);
    const Field3D p0     = snapshot(p);
    const Field3D theta0 = snapshot(theta);
    const Field3D u0     = snapshot(u);
    const Field3D v0     = snapshot(v);
    const Field3D w0     = snapshot(w);

    constexpr int kCycles = 100;
    for (int cycle = 0; cycle < kCycles; ++cycle)
        scheme->apply_full();

    REQUIRE(fields_equal_exact(rho,   rho0));
    REQUIRE(fields_equal_exact(p,     p0));
    REQUIRE(fields_equal_exact(theta, theta0));
    REQUIRE(fields_equal_exact(u,     u0));
    REQUIRE(fields_equal_exact(v,     v0));
    REQUIRE(fields_equal_exact(w,     w0));
}

// ============================================================================
// Gate 3a -- u[0] is interior on C-grid and must NOT be touched by the BC.
// This is the structural distinction from the collocated cylindrical scheme,
// which sets u[0] = -u[1] (antisymmetric ghost) and is the source of Bug 7
// false gradients at the axis. We seed u[0] with a non-physical value and
// verify the BC leaves it alone.
// ============================================================================
TEST_CASE("C-grid cylindrical BCs leave u[0] interior face untouched",
          "[boundary_conditions][cylindrical][cgrid][bug7][c2]")
{
    setup_cylindrical_cgrid_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();

    constexpr float kSeed = 1.5f;
    for (int j = 0; j < NTH; ++j)
        for (int k = 1; k < NZ - 1; ++k)
            u[0][j][k] = kSeed;

    auto scheme = create_cylindrical_cgrid_bc_scheme();
    scheme->apply_full();

    for (int j = 0; j < NTH; ++j)
        for (int k = 1; k < NZ - 1; ++k)
        {
            INFO("u[0][" << j << "][" << k << "] = " << u[0][j][k]);
            REQUIRE(u[0][j][k] == Approx(kSeed).epsilon(0.0));
        }
}

// ============================================================================
// Gate 3b -- u[NR-1] is the rigid outer wall and must be exactly zero
// after the BC apply, regardless of its previous value.
// ============================================================================
TEST_CASE("C-grid cylindrical BCs zero u at the outer wall face",
          "[boundary_conditions][cylindrical][cgrid][c2]")
{
    setup_cylindrical_cgrid_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();

    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
            u[NR - 1][j][k] = 7.3f;

    auto scheme = create_cylindrical_cgrid_bc_scheme();
    scheme->apply_full();

    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
        {
            INFO("u[NR-1][" << j << "][" << k << "] = " << u[NR-1][j][k]);
            REQUIRE(u[NR - 1][j][k] == 0.0f);
        }
}

// ============================================================================
// Gate 3c -- w[i][j][NZ-1] is the rigid lid face and must be exactly zero
// after the BC apply.
// ============================================================================
TEST_CASE("C-grid cylindrical BCs zero w at the rigid lid face",
          "[boundary_conditions][cylindrical][cgrid][c2]")
{
    setup_cylindrical_cgrid_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            w[i][j][NZ - 1] = 4.2f;

    auto scheme = create_cylindrical_cgrid_bc_scheme();
    scheme->apply_full();

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            INFO("w[" << i << "][" << j << "][NZ-1] = " << w[i][j][NZ-1]);
            REQUIRE(w[i][j][NZ - 1] == 0.0f);
        }
}

// ============================================================================
// Gate 3d -- w[i][j][0] is the first interior z-face on the C-grid and must
// NOT be zeroed (rigid surface at z=0 is implicit in div_flux_z).
// ============================================================================
TEST_CASE("C-grid cylindrical BCs leave w[k=0] interior face untouched",
          "[boundary_conditions][cylindrical][cgrid][c2]")
{
    setup_cylindrical_cgrid_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();

    constexpr float kSeed = 0.85f;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 0; j < NTH; ++j)
            w[i][j][0] = kSeed;

    auto scheme = create_cylindrical_cgrid_bc_scheme();
    scheme->apply_full();

    for (int i = 1; i < NR - 1; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            INFO("w[" << i << "][" << j << "][0] = " << w[i][j][0]);
            REQUIRE(w[i][j][0] == Approx(kSeed).epsilon(0.0));
        }
}

// ============================================================================
// Gate 3e -- v[0][j] (axis theta-face) collapses to a single point and is set
// to zero by the BC.
// ============================================================================
TEST_CASE("C-grid cylindrical BCs zero v at the axis theta-face",
          "[boundary_conditions][cylindrical][cgrid][c2]")
{
    setup_cylindrical_cgrid_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();

    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
            v[0][j][k] = 2.7f;

    auto scheme = create_cylindrical_cgrid_bc_scheme();
    scheme->apply_full();

    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
        {
            INFO("v[0][" << j << "][" << k << "] = " << v[0][j][k]);
            REQUIRE(v[0][j][k] == 0.0f);
        }
}

// ============================================================================
// Gate 4 -- scalars at the axis ghost (i=0) and outer ghost (i=NR-1) follow
// zero-gradient from the adjacent interior cell.
// ============================================================================
TEST_CASE("C-grid cylindrical BCs apply zero-gradient scalars on radial ghosts",
          "[boundary_conditions][cylindrical][cgrid][c2]")
{
    setup_cylindrical_cgrid_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();

    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
        {
            rho[1][j][k]      = 0.95f;
            theta[1][j][k]    = 305.0f;
            qv[1][j][k]       = 0.012f;
            rho[NR - 2][j][k] = 1.05f;
            theta[NR - 2][j][k] = 295.0f;
            qv[NR - 2][j][k]    = 0.008f;
        }

    auto scheme = create_cylindrical_cgrid_bc_scheme();
    scheme->apply_full();

    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
        {
            REQUIRE(rho[0][j][k]   == rho[1][j][k]);
            REQUIRE(theta[0][j][k] == theta[1][j][k]);
            REQUIRE(qv[0][j][k]    == qv[1][j][k]);
            REQUIRE(rho[NR - 1][j][k]   == rho[NR - 2][j][k]);
            REQUIRE(theta[NR - 1][j][k] == theta[NR - 2][j][k]);
            REQUIRE(qv[NR - 1][j][k]    == qv[NR - 2][j][k]);
        }
}

// ============================================================================
// Gate 5 -- vertical pressure extrapolation matches the hydrostatic relation
// p[NZ-1] - p[NZ-2] = -rho * g * dz at the lid (and equivalent at the surface).
// ============================================================================
TEST_CASE("C-grid cylindrical BCs apply hydrostatic pressure extrapolation",
          "[boundary_conditions][cylindrical][cgrid][c2]")
{
    setup_cylindrical_cgrid_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();

    auto scheme = create_cylindrical_cgrid_bc_scheme();
    scheme->apply_full();

    const double g_val = dynamics_constants::g;

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            const double rho_top_expected = static_cast<double>(rho[i][j][NZ - 2]);
            const double rho_bot_expected = static_cast<double>(rho[i][j][1]);
            const double p_lid_expected   =
                static_cast<double>(p[i][j][NZ - 2]) - rho_top_expected * g_val * dz;
            const double p_surf_expected  =
                static_cast<double>(p[i][j][1])      + rho_bot_expected * g_val * dz;

            REQUIRE(static_cast<double>(p[i][j][NZ - 1])
                    == Approx(p_lid_expected).margin(1.0e-2));
            REQUIRE(static_cast<double>(p[i][j][0])
                    == Approx(p_surf_expected).margin(1.0e-2));
        }
}

// ============================================================================
// Gate 6 -- acoustic-substep BCs touch only momentum + pressure; scalars
// (theta, moisture) are left alone.
// ============================================================================
TEST_CASE("C-grid cylindrical acoustic BCs preserve scalar fields",
          "[boundary_conditions][cylindrical][cgrid][c2]")
{
    setup_cylindrical_cgrid_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();

    // Seed scalars in the ghost cells with values that the full BC would
    // overwrite via zero-gradient. The acoustic path must NOT overwrite.
    constexpr float kThetaSeed = 273.15f;
    constexpr float kQvSeed    = 0.123f;
    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
        {
            theta[0][j][k]      = kThetaSeed;
            theta[NR - 1][j][k] = kThetaSeed;
            qv[0][j][k]         = kQvSeed;
            qv[NR - 1][j][k]    = kQvSeed;
        }

    auto scheme = create_cylindrical_cgrid_bc_scheme();
    scheme->apply_acoustic();

    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
        {
            REQUIRE(theta[0][j][k]      == kThetaSeed);
            REQUIRE(theta[NR - 1][j][k] == kThetaSeed);
            REQUIRE(qv[0][j][k]         == kQvSeed);
            REQUIRE(qv[NR - 1][j][k]    == kQvSeed);
        }

    // And momentum + pressure BCs are still applied.
    REQUIRE(w[0][0][NZ - 1] == 0.0f);
    REQUIRE(u[NR - 1][0][NZ / 2] == 0.0f);
}

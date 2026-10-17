/**
 * @file test_cartesian_vertical_bc_yaml.cpp
 * @brief Phase C2 exit-criterion gate — `boundary_conditions.bottom` and
 *        `boundary_conditions.top` actually drive simulator behavior.
 *
 * Setup: small Cartesian grid, seed interior velocity to a known value,
 * apply BC under each (bottom × top) combination, inspect k=0 and k=NZ-1.
 *
 * Expected per setting:
 *   bottom=RigidNoSlip — u=0, v=0, w=0 at k=0
 *   bottom=RigidSlip   — u=u[1], v=v[1], w=0 at k=0 (legacy default)
 *   top=Rigid          — w=0, u/v zero-gradient at k=NZ-1 (legacy default)
 *   top=RigidWithSponge — same BC as Rigid (sponge layer itself is C3)
 *   top=Open           — no w constraint; u/v zero-gradient
 *
 * Default check: a fresh process leaves globals at the legacy values
 * (RigidSlip + Rigid) so v0 configs without a `schema:` stamp keep their
 * historical behavior.
 */

#include "catch2/catch.hpp"
#include "boundary_conditions/factory.hpp"
#include "core/field/field3d.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "dynamics/dynamics_base.hpp"
#include "terrain/base/topography.hpp"
#include "terrain/conservative_transport.hpp"

#include <cmath>

namespace
{

using tmv::init::config::BottomBC;
using tmv::init::config::TopBC;
using tmv::init::config::LateralBC;

constexpr int kNR = 6;
constexpr int kNTH = 6;
constexpr int kNZ = 6;

constexpr float kU_INTERIOR = 5.0f;
constexpr float kV_INTERIOR = 7.0f;
constexpr float kW_INTERIOR = 9.0f;
constexpr float kRHO_INTERIOR = 1.225f;
constexpr float kP_INTERIOR  = 100000.0f;

void resize_and_seed()
{
    NR = kNR; NTH = kNTH; NZ = kNZ;
    dr = 1000.0; dz = 500.0; dt = 0.1; dtheta = 0.0;
    rho0_base.assign(NZ, kRHO_INTERIOR);
    p0_base.assign(NZ, kP_INTERIOR);
    u0_base.assign(NZ, 0.0); v0_base.assign(NZ, 0.0);
    theta0_base.assign(NZ, 300.0);
    qv0_base.assign(NZ, 0.0);
    global_coordinate_system = CoordinateSystem::Cartesian;

    // Keep lateral BCs at ZeroGradient (the legacy behavior) so this test isolates
    // the vertical-BC behavior we're checking.
    global_lateral_bc_west = LateralBC::ZeroGradient;
    global_lateral_bc_east = LateralBC::ZeroGradient;
    global_lateral_bc_south = LateralBC::ZeroGradient;
    global_lateral_bc_north = LateralBC::ZeroGradient;

    rho.resize(NR, NTH, NZ, kRHO_INTERIOR);
    p.resize(NR, NTH, NZ, kP_INTERIOR);
    u.resize(NR, NTH, NZ, kU_INTERIOR);
    v.resize(NR, NTH, NZ, kV_INTERIOR);
    w.resize(NR, NTH, NZ, kW_INTERIOR);
    theta.resize(NR, NTH, NZ, 300.0f);
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
}

void reset_vertical_bcs()
{
    global_bottom_bc = BottomBC::RigidSlip;
    global_top_bc    = TopBC::Rigid;
}

}  // namespace

// ─────────────────────────────────────────────────────────────
// Bottom BC
// ─────────────────────────────────────────────────────────────

TEST_CASE("C2 bottom=RigidNoSlip zeroes all velocity at k=0",
          "[boundary_conditions][c2][vertical_bc]")
{
    resize_and_seed();
    global_bottom_bc = BottomBC::RigidNoSlip;
    global_top_bc    = TopBC::Rigid;

    auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                   StaggerType::Collocated);
    scheme->apply_full();

    const int i = 2, j = 3;
    REQUIRE(u[i][j][0] == Approx(0.0f));
    REQUIRE(v[i][j][0] == Approx(0.0f));
    REQUIRE(w[i][j][0] == Approx(0.0f));

    reset_vertical_bcs();
}

TEST_CASE("C2 bottom=RigidSlip leaves tangential u/v as zero-gradient at k=0",
          "[boundary_conditions][c2][vertical_bc]")
{
    resize_and_seed();
    global_bottom_bc = BottomBC::RigidSlip;
    global_top_bc    = TopBC::Rigid;

    auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                   StaggerType::Collocated);
    scheme->apply_full();

    const int i = 2, j = 3;
    REQUIRE(u[i][j][0] == Approx(kU_INTERIOR));   // copies u[1] = interior value
    REQUIRE(v[i][j][0] == Approx(kV_INTERIOR));
    REQUIRE(w[i][j][0] == Approx(0.0f));          // normal velocity always zero

    reset_vertical_bcs();
}

// ─────────────────────────────────────────────────────────────
// Top BC
// ─────────────────────────────────────────────────────────────

TEST_CASE("C2 top=Rigid zeroes w at k=NZ-1",
          "[boundary_conditions][c2][vertical_bc]")
{
    resize_and_seed();
    global_bottom_bc = BottomBC::RigidSlip;
    global_top_bc    = TopBC::Rigid;

    auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                   StaggerType::Collocated);
    scheme->apply_full();

    const int i = 1, j = 4;
    REQUIRE(w[i][j][kNZ - 1] == Approx(0.0f));
    REQUIRE(u[i][j][kNZ - 1] == Approx(kU_INTERIOR));  // zero-gradient
    REQUIRE(v[i][j][kNZ - 1] == Approx(kV_INTERIOR));

    reset_vertical_bcs();
}

TEST_CASE("C2 top=RigidWithSponge behaves identically to Rigid at the BC layer",
          "[boundary_conditions][c2][vertical_bc]")
{
    // The sponge itself is C3's deliverable — at the BC layer the two
    // top variants apply the same w=0 + zero-gradient horizontal pattern.
    resize_and_seed();
    global_bottom_bc = BottomBC::RigidSlip;
    global_top_bc    = TopBC::RigidWithSponge;

    auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                   StaggerType::Collocated);
    scheme->apply_full();

    const int i = 1, j = 4;
    REQUIRE(w[i][j][kNZ - 1] == Approx(0.0f));
    REQUIRE(u[i][j][kNZ - 1] == Approx(kU_INTERIOR));
    REQUIRE(v[i][j][kNZ - 1] == Approx(kV_INTERIOR));

    reset_vertical_bcs();
}

TEST_CASE("C2 top=Open leaves w at k=NZ-1 unconstrained",
          "[boundary_conditions][c2][vertical_bc]")
{
    resize_and_seed();
    global_bottom_bc = BottomBC::RigidSlip;
    global_top_bc    = TopBC::Open;

    auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                   StaggerType::Collocated);
    scheme->apply_full();

    const int i = 1, j = 4;
    // Open: w at the lid is NOT zeroed. The seeded interior value (kW_INTERIOR)
    // should survive untouched (the BC writes nothing to w[NZ-1]).
    REQUIRE(w[i][j][kNZ - 1] == Approx(kW_INTERIOR));
    REQUIRE(u[i][j][kNZ - 1] == Approx(kU_INTERIOR));
    REQUIRE(v[i][j][kNZ - 1] == Approx(kV_INTERIOR));

    reset_vertical_bcs();
}

// ─────────────────────────────────────────────────────────────
// Acoustic-substep dispatch
// ─────────────────────────────────────────────────────────────

TEST_CASE("C2 apply_acoustic respects vertical BC enums",
          "[boundary_conditions][c2][vertical_bc][acoustic]")
{
    resize_and_seed();
    global_bottom_bc = BottomBC::RigidNoSlip;
    global_top_bc    = TopBC::Open;

    auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                   StaggerType::Collocated);
    scheme->apply_acoustic(0.0);

    const int i = 2, j = 2;
    REQUIRE(u[i][j][0] == Approx(0.0f));          // noslip at bottom
    REQUIRE(v[i][j][0] == Approx(0.0f));
    REQUIRE(w[i][j][0] == Approx(0.0f));
    REQUIRE(w[i][j][kNZ - 1] == Approx(kW_INTERIOR));  // open top — w untouched

    reset_vertical_bcs();
}

// ─────────────────────────────────────────────────────────────
// Backward compat: a fresh process keeps the legacy defaults
// ─────────────────────────────────────────────────────────────

TEST_CASE("C2 globals default to legacy Cartesian behavior",
          "[boundary_conditions][c2][vertical_bc][defaults]")
{
    REQUIRE(global_bottom_bc == BottomBC::RigidSlip);
    REQUIRE(global_top_bc    == TopBC::Rigid);
}

// ─────────────────────────────────────────────────────────────
// C-grid scheme: the k=0 momentum level is prognostic
// ─────────────────────────────────────────────────────────────

TEST_CASE("CGrid bottom=RigidSlip leaves u/v at k=0 prognostic",
          "[boundary_conditions][cgrid][vertical_bc][m19]")
{
    // SCIENCE.md M19: the old k0 := k1 copy enslaved the lowest
    // momentum level and erased low-level shear. Free-slip is zero
    // vertical momentum flux at the ground, enforced in-kernel; the BC
    // must not overwrite the prognostic level.
    resize_and_seed();
    global_bottom_bc = BottomBC::RigidSlip;
    global_top_bc    = TopBC::Rigid;

    const int i = 2, j = 3;
    const float sheared_u = 7.5f;   // distinct from the interior value
    const float sheared_v = -3.25f;
    u[i][j][0] = sheared_u;
    v[i][j][0] = sheared_v;

    auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                   StaggerType::CGrid);
    scheme->apply_full();

    REQUIRE(u[i][j][0] == Approx(sheared_u));   // shear survives
    REQUIRE(v[i][j][0] == Approx(sheared_v));

    reset_vertical_bcs();
}

TEST_CASE("CGrid bottom=RigidNoSlip pins u/v at k=0",
          "[boundary_conditions][cgrid][vertical_bc][m19]")
{
    resize_and_seed();
    global_bottom_bc = BottomBC::RigidNoSlip;
    global_top_bc    = TopBC::Rigid;

    auto scheme = create_boundary_condition_scheme(CoordinateSystem::Cartesian,
                                                   StaggerType::CGrid);
    scheme->apply_full();

    const int i = 2, j = 3;
    REQUIRE(u[i][j][0] == Approx(0.0f));
    REQUIRE(v[i][j][0] == Approx(0.0f));

    reset_vertical_bcs();
}

TEST_CASE("CGrid terrain bottom distinguishes physical w from normal flow",
          "[boundary_conditions][cgrid][terrain][vertical_bc][TE-003]")
{
    resize_and_seed();
    dtheta = 900.0;
    global_top_bc = TopBC::Rigid;

    TerrainConfig cfg;
    cfg.coord_id = "btf";
    cfg.ztop = static_cast<double>(NZ) * dz;
    Topography2D topo;
    topography::initialize_topography(topo, NR, NTH);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            topo.h[i][j] = 120.0 + 18.0 * i + 7.0 * j;
    TerrainMetrics3D metrics;
    topography::initialize_metrics(metrics, NR, NTH, NZ);
    topography::build_conservative_face_metrics(
        cfg, topo, topography::build_zeta_interfaces(NZ, cfg.ztop),
        dr, dtheta, metrics);

    const int i = 2;
    const int j = 3;
    global_bottom_bc = BottomBC::RigidSlip;
    auto slip = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    slip->apply_full();
    const double slip_w = terrain_transport::cgrid_ground_vertical_velocity(
        metrics, u, v, i, j);
    REQUIRE(std::abs(slip_w) > 0.0);
    REQUIRE(slip_w - static_cast<double>(kU_INTERIOR)
                       * (-metrics.Gx_face(i, j, 0))
                   - static_cast<double>(kV_INTERIOR)
                       * (-metrics.Gy_face(i, j, 0))
            == Approx(0.0).margin(2.0e-15));

    global_bottom_bc = BottomBC::RigidNoSlip;
    auto no_slip = create_boundary_condition_scheme(
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    no_slip->apply_full();
    REQUIRE(terrain_transport::cgrid_ground_vertical_velocity(
                metrics, u, v, i, j)
            == 0.0);

    reset_vertical_bcs();
}

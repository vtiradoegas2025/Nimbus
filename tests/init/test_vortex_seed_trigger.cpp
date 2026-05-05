/**
 * @file test_vortex_seed_trigger.cpp
 * @brief Verifies VortexSeedTrigger imposes a Rankine vortex profile.
 *
 * The test_harness sets up a default 8x8x8 cylindrical grid with NR=NTH=NZ=8,
 * dr=dz=100m. v_max=10 m/s, r_max=200m, z_top=400m gives the trigger room
 * to act on cells [0..3] in z and a Rankine peak around the second-third
 * radial cell. Coverage:
 *
 *   - Cylindrical: v[i][j][k] receives the Rankine-shaped contribution
 *     for z <= z_top, with the inner-core scaling linearly with r and the
 *     outer skirt scaling with 1/r.
 *   - Above z_top: no contribution.
 *   - Cartesian: vortex axis at (cx, cy) projects v_theta onto (u, v).
 */

#include "catch2/catch.hpp"
#include "core/coordinate_system.hpp"
#include "core/field3d.hpp"
#include "core/grid_geometry.hpp"
#include "core/simulation.hpp"
#include "init/trigger/vortex_seed.hpp"

#include <cmath>

extern int NR;
extern int NTH;
extern int NZ;
extern Field3D u;
extern Field3D v;
extern GridGeometry global_grid_geometry;
extern CoordinateSystem global_coordinate_system;
extern StaggerType global_stagger_type;

namespace
{

void zero_winds()
{
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
}

}  // namespace

TEST_CASE("VortexSeedTrigger applies Rankine vortex on cylindrical grid",
          "[init][trigger][vortex_seed]")
{
    global_coordinate_system = CoordinateSystem::Cylindrical;
    zero_winds();

    tmv::init::VortexSeedParams p;
    p.r_max_m = 200.0;
    p.v_max_ms = 10.0;
    p.z_top_m = 400.0;
    tmv::init::VortexSeedTrigger trigger(p);
    trigger.apply();

    SECTION("v at axis (i=0, r~50m) follows linear inner-core")
    {
        // r[0] is the cell-center of the innermost cell. With dr=100,
        // typical centering is r[0] = dr/2 = 50 m. The Rankine profile is
        // v_theta = v_max * (r / r_max). Check the value matches at j=0.
        const double r0 = global_grid_geometry.r[0];
        const double expected = 10.0 * (r0 / 200.0);
        REQUIRE(v[0][0][0] == Approx(static_cast<float>(expected)).margin(1.0e-5));
    }

    SECTION("v outside r_max scales as 1/r")
    {
        // Find a cell where r > r_max. With dr=100 and 8 radial cells,
        // r values span roughly 50..750 m, so cells beyond r_max=200 exist.
        for (int i = 0; i < NR; ++i)
        {
            const double r = global_grid_geometry.r[i];
            if (r > 250.0)  // safely beyond r_max
            {
                const double expected = 10.0 * (200.0 / r);
                REQUIRE(v[i][0][0] == Approx(static_cast<float>(expected)).margin(1.0e-5));
                break;
            }
        }
    }

    SECTION("v above z_top is unchanged (zero, since we zeroed)")
    {
        // z[k] = k*dz = 0, 100, 200, 300, 400, 500, 600, 700.
        // z_top_m=400 means k=4 has z=400 (boundary, > z_top fails on
        // strict > comparison, so k=4 still gets a bump). But k=5 onwards
        // should be untouched.
        for (int k = 5; k < NZ; ++k)
        {
            REQUIRE(v[0][0][k] == 0.0f);
        }
    }

    SECTION("v is uniform in j (axisymmetric)")
    {
        for (int j = 1; j < NTH; ++j)
        {
            REQUIRE(v[0][j][0] == v[0][0][0]);
        }
    }
}

TEST_CASE("VortexSeedTrigger Cartesian projection",
          "[init][trigger][vortex_seed]")
{
    global_coordinate_system = CoordinateSystem::Cartesian;
    zero_winds();

    tmv::init::VortexSeedParams p;
    p.r_max_m = 200.0;
    p.v_max_ms = 10.0;
    p.z_top_m = 400.0;
    p.center_x_m = 0.0;
    p.center_y_m = 0.0;
    tmv::init::VortexSeedTrigger trigger(p);
    trigger.apply();

    // Cartesian harness uses geo.r[i] = i * dr (or similar) and geo.theta[j]
    // as the y coordinate (Phase A reuses dr for both dx and dy). The
    // tangential component produces u_x = -v_theta * dy / r, u_y = v_theta * dx / r.
    // At a point on the +x axis (dy = 0), u_x should be zero and u_y > 0.
    // Note: This test depends on harness grid layout; we just verify the
    // axisymmetric property u^2 + v^2 = v_theta^2 at every cell.
    const auto& geo = global_grid_geometry;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double dx = geo.r[i] - 0.0;
            const double dy = geo.theta[j] - 0.0;
            const double r = std::sqrt(dx * dx + dy * dy);
            if (r <= 0.0) continue;
            const double v_theta_expected =
                (r <= 200.0) ? 10.0 * (r / 200.0) : 10.0 * (200.0 / r);
            const double speed_sq = static_cast<double>(u[i][j][0]) * u[i][j][0]
                                  + static_cast<double>(v[i][j][0]) * v[i][j][0];
            REQUIRE(std::sqrt(speed_sq) ==
                    Approx(v_theta_expected).margin(1.0e-4));
        }
    }

    // Restore harness default for any subsequent tests in the same binary.
    global_coordinate_system = CoordinateSystem::Cylindrical;
}

TEST_CASE("VortexSeedTrigger describe",
          "[init][trigger][vortex_seed]")
{
    tmv::init::VortexSeedTrigger trigger({});
    REQUIRE(trigger.describe() == "vortex_seed/rankine");
}

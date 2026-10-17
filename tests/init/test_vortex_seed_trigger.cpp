/**
 * @file test_vortex_seed_trigger.cpp
 * @brief Verifies VortexSeedTrigger imposes a Rankine vortex profile.
 *
 * The model is Cartesian. The test_harness sets up a default 8x8x8 grid with
 * NR=NTH=NZ=8, dr=dz=100m. v_max=10 m/s, r_max=200m, z_top=400m gives the
 * trigger room to act below z_top with a Rankine peak around the second-third
 * radial cell. The trigger places a vortex axis at (cx, cy) and projects the
 * Rankine tangential velocity v_theta(r) onto the (u, v) wind components.
 * Coverage:
 *
 *   - The seeded wind speed sqrt(u^2 + v^2) equals v_theta(r) at every cell
 *     (the azimuthal projection preserves magnitude).
 *   - Cells above z_top receive no contribution.
 *   - describe() reports the canonical id.
 */

#include "catch2/catch.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/field/field3d.hpp"
#include "core/infra/grid_geometry.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "init/trigger/vortex_seed.hpp"

#include <cmath>
#include <vector>

extern int& NR;
extern int& NTH;
extern int& NZ;
extern Field3D& u;
extern Field3D& v;
extern GridGeometry& global_grid_geometry;
extern CoordinateSystem& global_coordinate_system;
extern StaggerType& global_stagger_type;

namespace
{

void zero_winds()
{
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
}

}  // namespace

TEST_CASE("VortexSeedTrigger projects Rankine vortex onto Cartesian winds",
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

    const auto& geo = global_grid_geometry;
    const PhysicalCellDomain domain = physical_scalar_domain(NR, NTH, NZ, StaggerType::Collocated);

    SECTION("seeded wind speed equals the Rankine v_theta at every cell")
    {
        // The harness lays the grid out with geo.r[i] as the x coordinate and
        // geo.theta[j] as the y coordinate. The trigger projects the azimuthal
        // v_theta onto (u, v): u_x = -v_theta*dy/r, u_y = +v_theta*dx/r, so the
        // seeded speed magnitude is v_theta(r) at every cell regardless of the
        // local azimuth.
        for (int i = domain.i_begin; i < domain.i_end; ++i)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                const double dx = geo.r[i] - 0.0;
                const double dy = geo.theta[j] - 0.0;
                const double r = std::sqrt(dx * dx + dy * dy);
                if (r <= 0.0) continue;
                const double v_theta_expected =
                    (r <= 200.0) ? 10.0 * (r / 200.0) : 10.0 * (200.0 / r);
                const int k = domain.k_begin;
                const double speed_sq = static_cast<double>(u[i][j][k]) * u[i][j][k]
                                      + static_cast<double>(v[i][j][k]) * v[i][j][k];
                REQUIRE(std::sqrt(speed_sq) ==
                        Approx(v_theta_expected).margin(1.0e-4));
            }
        }
    }

    SECTION("boundary-closure storage is not perturbed")
    {
        REQUIRE(u[0][domain.j_begin][domain.k_begin] == 0.0f);
        REQUIRE(v[domain.i_begin][0][domain.k_begin] == 0.0f);
        REQUIRE(u[domain.i_begin][domain.j_begin][0] == 0.0f);
        REQUIRE(v[NR - 1][domain.j_begin][domain.k_begin] == 0.0f);
        REQUIRE(u[domain.i_begin][NTH - 1][domain.k_begin] == 0.0f);
        REQUIRE(v[domain.i_begin][domain.j_begin][NZ - 1] == 0.0f);
    }

    SECTION("cells above z_top receive no contribution")
    {
        // The trigger skips any cell with geo.z[k] > z_top; those winds stay
        // at the zeroed value. With dz=100 and z_top=400 the upper cells
        // (z >= 500) are untouched.
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    if (geo.z[k] > 400.0)
                    {
                        REQUIRE(u[i][j][k] == 0.0f);
                        REQUIRE(v[i][j][k] == 0.0f);
                    }
                }
            }
        }
    }
}

TEST_CASE("VortexSeedTrigger describe",
          "[init][trigger][vortex_seed]")
{
    tmv::init::VortexSeedTrigger trigger({});
    REQUIRE(trigger.describe() == "vortex_seed/rankine");
}

TEST_CASE("VortexSeedTrigger rejects nonphysical parameters",
          "[init][trigger][vortex_seed][config-truth]")
{
    tmv::init::VortexSeedParams params;

    SECTION("non-positive radius of maximum wind")
    {
        params.r_max_m = 0.0;
        REQUIRE_THROWS_AS(tmv::init::VortexSeedTrigger(params), std::invalid_argument);
    }

    SECTION("negative vortex top")
    {
        params.z_top_m = -1.0;
        REQUIRE_THROWS_AS(tmv::init::VortexSeedTrigger(params), std::invalid_argument);
    }
}

TEST_CASE("VortexSeedTrigger can seed an explicit local state",
          "[init][trigger][vortex_seed][state]")
{
    NR = 4;
    NTH = 4;
    NZ = 4;
    dr = 100.0;
    dz = 100.0;
    dt = 1.0;
    dtheta = 100.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    CoordinateSystem::Cartesian);

    u.resize(NR, NTH, NZ, -999.0f);
    v.resize(NR, NTH, NZ, -999.0f);

    int local_nr = 4;
    int local_nth = 4;
    int local_nz = 4;
    double local_dr = 100.0;
    double local_dz = 100.0;
    double local_dt = 1.0;
    double local_dtheta = 100.0;
    GridGeometry local_geometry;
    local_geometry.initialize(local_nr, local_nth, local_nz,
                              local_dr, local_dz, local_dtheta,
                              CoordinateSystem::Cartesian);

    Field3D local_rho(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_p(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_u(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_w(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_v(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tracer(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_theta(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qv(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qc(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qr(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qi(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qs(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qh(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qg(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tke(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_rad(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dqv_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_du_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dv_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtke_dt_pbl(local_nr, local_nth, local_nz, 0.0f);

    std::vector<double> local_rho0;
    std::vector<double> local_p0;
    std::vector<double> local_qv0;
    std::vector<double> local_theta0;
    std::vector<double> local_u0;
    std::vector<double> local_v0;
    double local_coriolis = 0.0;
    double local_time = 0.0;

    SimulationStateView local_state{
        {local_nr, local_nth, local_nz, local_dr, local_dz, local_dt,
         local_dtheta, local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs,
         local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis,
        local_time};

    tmv::init::VortexSeedParams params;
    params.r_max_m = 200.0;
    params.v_max_ms = 10.0;
    params.z_top_m = 200.0;
    params.center_x_m = 0.0;
    params.center_y_m = 0.0;
    tmv::init::VortexSeedTrigger trigger(params);
    const tmv::init::TriggerSource& trigger_source = trigger;
    trigger_source.apply(local_state);

    const double x = local_geometry.r[1];
    const double y = local_geometry.theta[1];
    const double radius = std::hypot(x, y);
    const double vtheta = radius <= params.r_max_m
        ? params.v_max_ms * radius / params.r_max_m
        : params.v_max_ms * params.r_max_m / radius;
    REQUIRE(local_u[1][1][1]
            == Approx(-vtheta * y / radius).margin(1.0e-6));
    REQUIRE(local_v[1][1][1]
            == Approx(vtheta * x / radius).margin(1.0e-6));
    REQUIRE(local_v[1][1][0] == Approx(0.0f).margin(1.0e-6));
    REQUIRE(local_v[1][1][3] == Approx(0.0f).margin(1.0e-6));
    REQUIRE(v[1][1][1] == -999.0f);
}

/**
 * @file test_turbulence_topology.cpp
 * @brief SGS j-topology regression tests (SCIENCE.md M16).
 *
 * The diffusion Laplacian used to wrap j unconditionally - the SGS
 * tendencies mixed the north and south edges of every open-y Cartesian
 * domain. Both the diffusion and strain stencils now follow
 * eddy_viscosity::periodic_y_topology(): wrap only when y is
 * topologically periodic (Cartesian Periodic y-faces); otherwise the j
 * edge rows are skipped like the i edges.
 */

#include "catch2/catch.hpp"
#include "turbulence/turbulence_base.hpp"
#include "turbulence/base/eddy_viscosity.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/field/field3d.hpp"

#include <cmath>

namespace
{

constexpr int kN = 8;

struct TopologyFixture
{
    Field3D u, v, w, theta, rho;
    Field3D K, nu;
    TurbulenceStateView state;
    TurbulenceConfig cfg;
    GridMetrics grid;
    std::vector<double> rho0;

    TopologyFixture()
    {
        u.resize(kN, kN, kN, 0.0f);
        v.resize(kN, kN, kN, 0.0f);
        w.resize(kN, kN, kN, 0.0f);
        theta.resize(kN, kN, kN, 300.0f);
        rho.resize(kN, kN, kN, 1.0f);
        K.resize(kN, kN, kN, 10.0f);
        nu.resize(kN, kN, kN, 10.0f);
        state.u = &u; state.v = &v; state.w = &w; state.theta = &theta;
        state.rho = &rho;
        state.NR = kN; state.NTH = kN; state.NZ = kN;
        rho0.assign(kN, 1.0);
        state.rho0 = &rho0;
        cfg.mode = "3d";
        grid.dx = 100.0; grid.dy = 100.0;
        grid.dz.assign(kN, 100.0);
    }
};

void set_open_y()
{
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
}

void set_open_x()
{
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
}

void set_periodic_x()
{
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
}

void set_periodic_y()
{
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;
}

}  // namespace

TEST_CASE("Open-y SGS diffusion does not mix the north and south edges",
          "[turbulence][topology][m16]")
{
    TopologyFixture f;
    set_open_x();
    set_open_y();
    REQUIRE_FALSE(eddy_viscosity::periodic_y_topology());

    // Marker anomaly on the NORTH edge row only. Under the old
    // unconditional wrap, the j=0 (south) Laplacian read it as a
    // neighbor and produced a spurious tendency there.
    Field3D phi(kN, kN, kN, 300.0f);
    for (int i = 0; i < kN; ++i)
        for (int k = 0; k < kN; ++k)
            phi(i, kN - 1, k) = 400.0f;

    Field3D dphi;
    dphi.resize(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        f.cfg, f.state, f.grid, f.K, phi, dphi);

    const int i = 4, k = 4;
    // South edge row: skipped entirely on open y (like the i edges).
    REQUIRE(dphi(i, 0, k) == 0.0f);
    // North edge row likewise.
    REQUIRE(dphi(i, kN - 1, k) == 0.0f);
    // Interior row adjacent to the south edge: no contamination from
    // the north marker (the wrap would have injected (400-300)/dy^2
    // scaled terms here via j_prev = NTH-1 at j=0... and nonzero
    // diffusion exists at j = NTH-2 from the real marker neighbor).
    REQUIRE(dphi(i, 1, k) == 0.0f);
    REQUIRE(dphi(i, kN - 2, k) != 0.0f);

    set_open_y();  // restore defaults for subsequent tests
}

TEST_CASE("Periodic-y C-grid SGS wraps physical rows and leaves ghosts owned by BCs",
          "[turbulence][topology][m16]")
{
    TopologyFixture f;
    set_open_x();
    set_periodic_y();
    REQUIRE(eddy_viscosity::periodic_y_topology());

    Field3D phi(kN, kN, kN, 300.0f);
    // Mark the last PHYSICAL row. Keep both allocated ghosts deliberately
    // inconsistent: the SGS stencil must ignore them and wrap 1 <-> N-2.
    for (int i = 0; i < kN; ++i)
        for (int k = 0; k < kN; ++k)
            phi(i, kN - 2, k) = 400.0f;

    Field3D dphi;
    dphi.resize(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        f.cfg, f.state, f.grid, f.K, phi, dphi);

    const int i = 4, k = 4;
    REQUIRE(dphi(i, 1, k) != 0.0f);
    REQUIRE(dphi(i, 0, k) == 0.0f);
    REQUIRE(dphi(i, kN - 1, k) == 0.0f);

    set_open_y();
}

TEST_CASE("Periodic-x C-grid SGS wraps physical rows and ignores x ghosts",
          "[turbulence][topology][periodic-x]")
{
    TopologyFixture f;
    set_periodic_x();
    set_open_y();
    REQUIRE(eddy_viscosity::periodic_x_topology());

    Field3D phi(kN, kN, kN, 300.0f);
    for (int j = 0; j < kN; ++j)
        for (int k = 0; k < kN; ++k)
            phi(kN - 2, j, k) = 400.0f;
    // Deliberately poison storage-only closure rows. Periodic stencils own
    // the physical 1 <-> N-2 seam and must never obtain it from ghosts.
    for (int j = 0; j < kN; ++j)
        for (int k = 0; k < kN; ++k)
        {
            phi(0, j, k) = -900.0f;
            phi(kN - 1, j, k) = 1200.0f;
        }

    Field3D dphi(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        f.cfg, f.state, f.grid, f.K, phi, dphi);

    REQUIRE(dphi(1, 4, 4) != 0.0f);
    REQUIRE(dphi(0, 4, 4) == 0.0f);
    REQUIRE(dphi(kN - 1, 4, 4) == 0.0f);

    set_open_x();
}

TEST_CASE("Periodic-x SGS stress owns and conserves the seam u face",
          "[turbulence][topology][periodic-x][momentum]")
{
    TopologyFixture f;
    set_periodic_x();
    set_open_y();
    f.cfg.mode = "horizontal_only";
    for (int j = 1; j <= kN - 2; ++j)
        for (int k = 0; k < kN - 1; ++k)
            f.u(kN - 2, j, k) = 10.0f;

    Field3D du(kN, kN, kN, 0.0f), dv(kN, kN, kN, 0.0f),
            dw(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        f.cfg, f.state, f.grid, f.nu, du, dv, dw);

    REQUIRE(eddy_viscosity::compute_strain_sq_centered(
                f.state, f.grid, 1, 4, 4) > 0.0);
    REQUIRE(du(kN - 2, 4, 4) != 0.0f);
    double sum = 0.0;
    for (int i = 1; i <= kN - 2; ++i)
        sum += static_cast<double>(du(i, 4, 4));
    REQUIRE(std::abs(sum) < 1.0e-7);

    set_open_x();
}

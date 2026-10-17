/**
 * @file test_cartesian_boundary_conditions.cpp
 * @brief Verification gates for Phase A.3 — Cartesian boundary conditions.
 *
 * Verification gates for Cartesian boundary-condition ownership:
 *
 *   1. With Cartesian BCs and the warm-bubble IC from A.2's third test, run
 *      10 timesteps and confirm |w|_max stays within ±2 m/s. The Bug 7
 *      failure mode (cylindrical BCs feeding the radial-axis ghost cell a
 *      false gradient) blows up to O(100 m/s) within 10 steps; the 2 m/s
 *      bound is a clear pass/fail signal.
 *
 *   2. Mass conservation: Σρ over the domain is conserved to 1 part in 10⁴
 *      over 60 sim seconds with the hydrostatic equilibrium IC. Open lateral
 *      BCs are NOT mass-conservative when waves cross the boundary (the
 *      bubble case loses mass through the open faces by design); the
 *      conservation gate uses the equilibrium state where there is nothing
 *      to advect across the boundary, so a passing test means the BC code
 *      itself does not introduce spurious mass.
 *
 *   3. As a tighter Bug 7 stress test, run the equilibrium + uniform
 *      Cartesian wind state for 10 steps through the BC dispatcher and
 *      verify |w|_max stays at machine noise. This is the same configuration
 *      that produces a 17× stability gap on the cylindrical scheme.
 *
 * Test driver pattern: each test sets up the global Field3D state via the
 * test_harness symbols, instantiates a CartesianAGridScheme, and runs a manual
 * Euler timestep loop:
 *
 *     for step in 0..N:
 *         scheme.compute_momentum_tendencies(...)
 *         interior cells: state += tendency * dt
 *         apply_cartesian_boundary_conditions()
 *
 * The test does not call into step_dynamics_new because that path pulls in
 * microphysics, advection, turbulence, etc., which would mask whether the
 * BC code itself is the failure source.
 */

#include "catch2/catch.hpp"
#include "boundary_conditions/boundary_conditions.hpp"
#include "core/field/field3d.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "dynamics/schemes/klemp_wilhelmson_split/cartesian_agrid.hpp"
#include "dynamics/dynamics_base.hpp"

#include <algorithm>
#include <cmath>

namespace
{

// Sea-level air density and reference pressure for the synthetic atmosphere.
// Same constants as test_cartesian_dynamics.cpp so the bubble verification
// gate is reproduced bit-for-bit.
constexpr double kRho0 = 1.225;       // kg / m^3
constexpr double kP0   = 101325.0;    // Pa

void setup_cartesian_grid()
{
    NR = 16;
    NTH = 16;
    NZ = 16;
    dr = 1000.0;
    dz = 500.0;
    dt = 0.1;
    dtheta = 1000.0;
    rho0_base.assign(NZ, kRho0);
    p0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
    {
        p0_base[k] = kP0 - kRho0 * dynamics_constants::g * static_cast<double>(k) * dz;
    }
    u0_base.assign(NZ, 0.0);
    v0_base.assign(NZ, 0.0);
    theta0_base.assign(NZ, dynamics_constants::theta0);
    qv0_base.assign(NZ, 0.0);
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
    global_lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::Stationary;
    global_domain_frame = tmv::domain::HorizontalDomainFrame::stationary();
}

/**
 * @brief Resizes every prognostic field referenced by the BC function and
 *        the dynamics scheme to (NR, NTH, NZ), filled with sane defaults.
 *
 * The Cartesian BC function reads/writes u, v, w, rho, p, theta and
 * the seven moisture variables (qv .. qh). The CartesianAGridScheme momentum
 * routine touches u, v, w, rho, p, theta. We must allocate every one
 * of those before the first call or operator[] indexes into a 0×0×0 buffer.
 */
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

/**
 * @brief Sets every (i, j, k) cell of (rho, p) to the hydrostatic base
 *        state rho(z) = rho0, p(z) = p0 - rho0 * g * z. Centered differences
 *        of the linear p(z) reproduce -rho0*g exactly so the dynamics
 *        residual -dp/dz/rho - g is zero to machine precision.
 */
void initialize_hydrostatic_atmosphere()
{
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                rho[i][j][k] = static_cast<float>(kRho0);
                const double z = static_cast<double>(k) * dz;
                p[i][j][k] = static_cast<float>(kP0 - kRho0 * dynamics_constants::g * z);
                theta[i][j][k] = static_cast<float>(dynamics_constants::theta0);
            }
        }
    }
}

void zero_velocity_field()
{
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k]       = 0.0f;
                v[i][j][k] = 0.0f;
                w[i][j][k]       = 0.0f;
            }
        }
    }
}

void uniform_horizontal_wind(double ux, double uy)
{
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k]       = static_cast<float>(ux);
                v[i][j][k] = static_cast<float>(uy);
                w[i][j][k]       = 0.0f;
            }
        }
    }
}

/**
 * @brief Inserts a single-cell warm bubble at the geometric center.
 *
 * Uses the same modification pattern as test_cartesian_dynamics.cpp's third
 * test: only one cell is altered (theta increased by dtheta_bubble, rho
 * scaled to keep the ideal-gas relation consistent at the bubble center),
 * so the centered dp/dz stencil at the bubble center pulls from cells
 * (k-1) and (k+1) which are still at the unmodified hydrostatic value.
 * Analytic dw/dt at the center is g * dtheta / theta0 ≈ 0.0654 m/s^2.
 */
void insert_center_warm_bubble(double dtheta_bubble_k)
{
    const int i_c = NR / 2;
    const int j_c = NTH / 2;
    const int k_c = NZ / 2;
    const double new_theta = dynamics_constants::theta0 + dtheta_bubble_k;
    const double rho_warm  = kRho0 * (dynamics_constants::theta0 / new_theta);
    theta[i_c][j_c][k_c] = static_cast<float>(new_theta);
    rho[i_c][j_c][k_c]   = static_cast<float>(rho_warm);
}

struct TendencyBuffers
{
    Field3D du_x_dt;
    Field3D du_y_dt;
    Field3D dw_dt;
    Field3D drho_dt_out;
    Field3D dp_dt_out;

    void allocate()
    {
        du_x_dt.resize(NR, NTH, NZ, 0.0f);
        du_y_dt.resize(NR, NTH, NZ, 0.0f);
        dw_dt.resize(NR, NTH, NZ, 0.0f);
        drho_dt_out.resize(NR, NTH, NZ, 0.0f);
        dp_dt_out.resize(NR, NTH, NZ, 0.0f);
    }
};

/**
 * @brief Returns the largest absolute |w| over the entire domain (including
 *        boundary cells, so a runaway BC value would be visible).
 */
double max_abs_w_full_domain()
{
    double max_val = 0.0;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double v = std::abs(static_cast<double>(w[i][j][k]));
                if (v > max_val) max_val = v;
            }
        }
    }
    return max_val;
}

/**
 * @brief Returns the integral Σρ over the entire domain in float-summation
 *        order. We use double accumulation so the order-of-magnitude noise
 *        is at the FP64 epsilon level (~1e-16 of the sum), not the FP32
 *        epsilon level (~1e-7).
 */
double total_mass_full_domain()
{
    double sum = 0.0;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                sum += static_cast<double>(rho[i][j][k]);
            }
        }
    }
    return sum;
}

/**
 * @brief Drives one Cartesian timestep on the global state: tendency
 *        evaluation, Euler update on the interior, then BC apply.
 *
 * The Euler update intentionally writes to interior cells only (the strict
 * range [1..N-1) in each dimension). Boundary cells are then set by the
 * BC function from their interior neighbors. This mirrors the production
 * code path: the dynamics scheme computes tendencies on the interior, the
 * time integrator updates the interior, then `apply_boundary_conditions()`
 * fills in the ghost layer.
 */
void take_one_cartesian_dynamics_step(CartesianAGridScheme& scheme,
                                      TendencyBuffers& tb,
                                      double step_dt)
{
    scheme.compute_momentum_tendencies(
        u, v, w,
        rho, p, theta,
        step_dt,
        tb.du_x_dt, tb.du_y_dt, tb.dw_dt,
        tb.drho_dt_out, tb.dp_dt_out);

    for (int i = 1; i < NR - 1; ++i)
    {
        for (int j = 1; j < NTH - 1; ++j)
        {
            for (int k = 1; k < NZ - 1; ++k)
            {
                u[i][j][k]       += static_cast<float>(tb.du_x_dt[i][j][k]    * step_dt);
                v[i][j][k] += static_cast<float>(tb.du_y_dt[i][j][k]    * step_dt);
                w[i][j][k]       += static_cast<float>(tb.dw_dt[i][j][k]      * step_dt);
                rho[i][j][k]     += static_cast<float>(tb.drho_dt_out[i][j][k] * step_dt);
                p[i][j][k]       += static_cast<float>(tb.dp_dt_out[i][j][k]   * step_dt);
            }
        }
    }

    apply_cartesian_boundary_conditions();
}

}  // namespace

// ===========================================================================
// Gate 1: warm bubble + Cartesian BCs — |w|_max stays bounded for 10 steps.
// ===========================================================================
TEST_CASE("Cartesian BCs: warm bubble keeps |w| bounded for 10 timesteps",
          "[dynamics][cartesian][bug7][a3]")
{
    setup_cartesian_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();
    zero_velocity_field();
    insert_center_warm_bubble(/*dtheta_bubble_k=*/2.0);

    CartesianAGridScheme scheme;
    TendencyBuffers tb;
    tb.allocate();

    constexpr int    kSteps    = 10;
    constexpr double kStepDt   = 0.1;   // 1 sim second total
    constexpr double kBoundMps = 2.0;

    double max_w_observed = 0.0;
    for (int step = 0; step < kSteps; ++step)
    {
        take_one_cartesian_dynamics_step(scheme, tb, kStepDt);
        const double w_max_now = max_abs_w_full_domain();
        if (w_max_now > max_w_observed) max_w_observed = w_max_now;
    }

    INFO("max |w| observed over 10 steps = " << max_w_observed << " m/s");
    REQUIRE(max_w_observed <= kBoundMps);
    // The 2K bubble buoyancy is g*dtheta/theta0 ≈ 0.0654 m/s^2, so after
    // 1 sim second the linear-acceleration estimate is ~0.065 m/s. We expect
    // the observed max to be in this ballpark — well below 2 m/s — and a
    // failure here would mean either (a) Bug 7 has resurrected, or
    // (b) the BC code is feeding nonphysical values back into the dynamics.
    REQUIRE(max_w_observed < 0.5);
}

// ===========================================================================
// Gate 2: 60 sim seconds of equilibrium — Σρ conserved to 1 part in 10⁴.
// ===========================================================================
TEST_CASE("Cartesian BCs: hydrostatic equilibrium conserves mass over 60 s",
          "[dynamics][cartesian][a3]")
{
    setup_cartesian_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();
    zero_velocity_field();
    // No bubble — pure equilibrium. Open lateral BCs only let mass leave the
    // domain when there is something to advect across the boundary; an
    // equilibrium state has nothing to advect, so the conservation budget
    // is decided by the BC code itself, not by the physics.

    CartesianAGridScheme scheme;
    TendencyBuffers tb;
    tb.allocate();

    const double mass_initial = total_mass_full_domain();

    constexpr int    kSteps    = 600;
    constexpr double kStepDt   = 0.1;   // 60 sim seconds total
    constexpr double kRelTol   = 1.0e-4;

    for (int step = 0; step < kSteps; ++step)
    {
        take_one_cartesian_dynamics_step(scheme, tb, kStepDt);
    }

    const double mass_final = total_mass_full_domain();
    const double rel_drift  = std::abs(mass_final - mass_initial) / mass_initial;

    INFO("initial mass = " << mass_initial);
    INFO("final mass   = " << mass_final);
    INFO("relative drift = " << rel_drift);
    REQUIRE(rel_drift <= kRelTol);
}

// ===========================================================================
// Gate 3: equilibrium + uniform Cartesian wind — the Bug 7 stress test
// repeated through the BC dispatcher for 10 timesteps. The cylindrical BC
// would inject a spurious force at the i=0 axis on every step; the Cartesian
// BC must keep |w| at machine noise.
// ===========================================================================
TEST_CASE("Cartesian BCs: uniform wind preserves equilibrium for 10 timesteps",
          "[dynamics][cartesian][bug7][a3]")
{
    setup_cartesian_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();
    uniform_horizontal_wind(/*ux=*/10.0, /*uy=*/5.0);

    CartesianAGridScheme scheme;
    TendencyBuffers tb;
    tb.allocate();

    constexpr int    kSteps   = 10;
    constexpr double kStepDt  = 0.1;
    constexpr double kNoiseFloor = 1.0e-2;  // generous: noise is much lower

    double max_w_observed = 0.0;
    for (int step = 0; step < kSteps; ++step)
    {
        take_one_cartesian_dynamics_step(scheme, tb, kStepDt);
        const double w_max_now = max_abs_w_full_domain();
        if (w_max_now > max_w_observed) max_w_observed = w_max_now;
    }

    INFO("max |w| observed over 10 steps with uniform wind = "
         << max_w_observed << " m/s");
    REQUIRE(max_w_observed <= kNoiseFloor);
}

TEST_CASE("Cartesian A-grid lateral walls close grid-relative velocity",
          "[dynamics][cartesian][agrid][boundary][domain-motion][SCI-195]")
{
    setup_cartesian_grid();
    resize_all_fields_to_grid();
    initialize_hydrostatic_atmosphere();
    uniform_horizontal_wind(/*ux=*/9.0, /*uy=*/-4.0);
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::ConstantVelocity;
    global_domain_frame.velocity_ms = {6.5, -2.25};
    global_lateral_bc_west = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_east = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_south = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_north = tmv::init::config::LateralBC::RigidSlip;

    auto bc = tmv::create_cartesian_bc_scheme();
    bc->apply_full();
    const int j = NTH / 2;
    const int i = NR / 2;
    const int k = NZ / 2;
    REQUIRE(u[0][j][k] == 6.5f);
    REQUIRE(u[NR - 1][j][k] == 6.5f);
    REQUIRE(v[0][j][k] == -4.0f);
    REQUIRE(v[NR - 1][j][k] == -4.0f);
    REQUIRE(v[i][0][k] == -2.25f);
    REQUIRE(v[i][NTH - 1][k] == -2.25f);
    REQUIRE(u[i][0][k] == 9.0f);
    REQUIRE(u[i][NTH - 1][k] == 9.0f);

    global_lateral_bc_west = tmv::init::config::LateralBC::Rigid;
    global_lateral_bc_east = tmv::init::config::LateralBC::Rigid;
    global_lateral_bc_south = tmv::init::config::LateralBC::Rigid;
    global_lateral_bc_north = tmv::init::config::LateralBC::Rigid;
    bc->apply_acoustic(dt);
    REQUIRE(u[0][j][k] == 6.5f);
    REQUIRE(v[0][j][k] == -2.25f);
    REQUIRE(u[i][0][k] == 6.5f);
    REQUIRE(v[i][0][k] == -2.25f);

    // The coordinate window translates over an Earth-fixed lower surface;
    // lateral-wall velocity must never leak into no-slip surface physics.
    global_bottom_bc = tmv::init::config::BottomBC::RigidNoSlip;
    bc->apply_acoustic(dt);
    REQUIRE(u[i][j][0] == 0.0f);
    REQUIRE(v[i][j][0] == 0.0f);
}

TEST_CASE("Cartesian A-grid vertical ghosts preserve stored-base perturbations",
          "[dynamics][cartesian][agrid][boundary][well-balanced][AG-002]")
{
    setup_cartesian_grid();
    resize_all_fields_to_grid();

    for (int k = 0; k < NZ; ++k)
    {
        const double height_index = static_cast<double>(k);
        p0_base[k] = 100000.0 * std::exp(-0.11 * height_index);
        rho0_base[k] = 1.2 * std::exp(-0.08 * height_index);
        theta0_base[k] = 299.0 + 1.7 * height_index;
        qv0_base[k] = 0.014 * std::exp(-0.15 * height_index);
        u0_base[k] = 2.0 + 0.4 * height_index;
        v0_base[k] = -1.0 + 0.2 * height_index;
    }
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                p[i][j][k] = static_cast<float>(p0_base[k]);
                rho[i][j][k] = static_cast<float>(rho0_base[k]);
                theta[i][j][k] = static_cast<float>(theta0_base[k]);
                qv[i][j][k] = static_cast<float>(qv0_base[k]);
                u[i][j][k] = static_cast<float>(u0_base[k]);
                v[i][j][k] = static_cast<float>(v0_base[k]);
                w[i][j][k] = 0.0f;
            }

    const int i = NR / 2;
    const int j = NTH / 2;
    auto bc = tmv::create_cartesian_bc_scheme();
    bc->apply_acoustic(0.1);
    REQUIRE(p[i][j][0] == static_cast<float>(p0_base[0]));
    REQUIRE(p[i][j][NZ - 1] == static_cast<float>(p0_base[NZ - 1]));
    REQUIRE(rho[i][j][0] == static_cast<float>(rho0_base[0]));
    REQUIRE(rho[i][j][NZ - 1] == static_cast<float>(rho0_base[NZ - 1]));

    CartesianAGridScheme dynamics;
    TendencyBuffers rest;
    rest.allocate();
    dynamics.compute_fast_pressure_tendencies(
        u, v, w, rho, p, /*dt_small=*/0.1,
        rest.drho_dt_out, rest.dp_dt_out);
    dynamics.compute_fast_momentum_tendencies(
        u, v, w, rho, p, /*dt_small=*/0.1,
        rest.du_x_dt, rest.du_y_dt, rest.dw_dt);
    for (int ii = 1; ii < NR - 1; ++ii)
        for (int jj = 1; jj < NTH - 1; ++jj)
            for (int k = 1; k < NZ - 1; ++k)
            {
                REQUIRE(rest.drho_dt_out[ii][jj][k] == 0.0f);
                REQUIRE(rest.dp_dt_out[ii][jj][k] == 0.0f);
                REQUIRE(rest.du_x_dt[ii][jj][k] == 0.0f);
                REQUIRE(rest.du_y_dt[ii][jj][k] == 0.0f);
                REQUIRE(rest.dw_dt[ii][jj][k] == 0.0f);
            }

    p[i][j][1] += 17.0f;
    rho[i][j][1] += 0.03f;
    u[i][j][1] += 0.4f;
    v[i][j][1] -= 0.2f;
    p[i][j][NZ - 2] -= 11.0f;
    rho[i][j][NZ - 2] -= 0.02f;
    u[i][j][NZ - 2] -= 0.3f;
    v[i][j][NZ - 2] += 0.1f;

    bc->apply_acoustic(0.1);

    const auto perturbation = [](float value, const std::vector<double>& base,
                                 int k)
    {
        return static_cast<double>(value) - base[static_cast<std::size_t>(k)];
    };
    REQUIRE(perturbation(p[i][j][0], p0_base, 0)
            == Approx(perturbation(p[i][j][1], p0_base, 1)).margin(2.0e-3));
    REQUIRE(perturbation(rho[i][j][0], rho0_base, 0)
            == Approx(perturbation(rho[i][j][1], rho0_base, 1)).margin(2.0e-7));
    REQUIRE(perturbation(u[i][j][0], u0_base, 0)
            == Approx(perturbation(u[i][j][1], u0_base, 1)).margin(2.0e-7));
    REQUIRE(perturbation(v[i][j][0], v0_base, 0)
            == Approx(perturbation(v[i][j][1], v0_base, 1)).margin(2.0e-7));
    REQUIRE(perturbation(p[i][j][NZ - 1], p0_base, NZ - 1)
            == Approx(perturbation(p[i][j][NZ - 2], p0_base, NZ - 2)).margin(2.0e-3));
    REQUIRE(perturbation(rho[i][j][NZ - 1], rho0_base, NZ - 1)
            == Approx(perturbation(rho[i][j][NZ - 2], rho0_base, NZ - 2)).margin(2.0e-7));

    theta[i][j][1] += 0.6f;
    qv[i][j][1] += 0.001f;
    theta[i][j][NZ - 2] -= 0.4f;
    qv[i][j][NZ - 2] -= 0.0005f;
    bc->apply_full();
    REQUIRE(perturbation(theta[i][j][0], theta0_base, 0)
            == Approx(perturbation(theta[i][j][1], theta0_base, 1)).margin(2.0e-5));
    REQUIRE(perturbation(qv[i][j][0], qv0_base, 0)
            == Approx(perturbation(qv[i][j][1], qv0_base, 1)).margin(2.0e-9));
    REQUIRE(perturbation(theta[i][j][NZ - 1], theta0_base, NZ - 1)
            == Approx(perturbation(theta[i][j][NZ - 2], theta0_base, NZ - 2)).margin(2.0e-5));
    REQUIRE(perturbation(qv[i][j][NZ - 1], qv0_base, NZ - 1)
            == Approx(perturbation(qv[i][j][NZ - 2], qv0_base, NZ - 2)).margin(2.0e-9));
}

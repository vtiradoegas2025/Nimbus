/**
 * @file test_cartesian_boundary_conditions.cpp
 * @brief Verification gates for Phase A.3 — Cartesian boundary conditions.
 *
 * Verification gates from docs/CoordinateBackend_Plan.md §A.3:
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
 * test_harness symbols, instantiates a CartesianScheme, and runs a manual
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
#include "core/boundary_conditions.hpp"
#include "core/field3d.hpp"
#include "core/runtime_config.hpp"
#include "core/simulation.hpp"
#include "dynamics/schemes/cartesian/cartesian.hpp"
#include "physics/dynamics_base.hpp"

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
    dtheta = 0.0;
    rho0_base.assign(NZ, kRho0);
    p0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
    {
        p0_base[k] = kP0 - kRho0 * dynamics_constants::g * static_cast<double>(k) * dz;
    }
    global_coordinate_system = CoordinateSystem::Cartesian;
}

/**
 * @brief Resizes every prognostic field referenced by the BC function and
 *        the dynamics scheme to (NR, NTH, NZ), filled with sane defaults.
 *
 * The Cartesian BC function reads/writes u, v_theta, w, rho, p, theta and
 * the seven moisture variables (qv .. qh). The CartesianScheme momentum
 * routine touches u, v_theta, w, rho, p, theta. We must allocate every one
 * of those before the first call or operator[] indexes into a 0×0×0 buffer.
 */
void resize_all_fields_to_grid()
{
    rho.resize(NR, NTH, NZ, static_cast<float>(kRho0));
    p.resize(NR, NTH, NZ);
    u.resize(NR, NTH, NZ, 0.0f);
    v_theta.resize(NR, NTH, NZ, 0.0f);
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
                v_theta[i][j][k] = 0.0f;
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
                v_theta[i][j][k] = static_cast<float>(uy);
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
void take_one_cartesian_dynamics_step(CartesianScheme& scheme,
                                      TendencyBuffers& tb,
                                      double step_dt)
{
    scheme.compute_momentum_tendencies(
        u, v_theta, w,
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
                v_theta[i][j][k] += static_cast<float>(tb.du_y_dt[i][j][k]    * step_dt);
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

    CartesianScheme scheme;
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

    CartesianScheme scheme;
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

    CartesianScheme scheme;
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

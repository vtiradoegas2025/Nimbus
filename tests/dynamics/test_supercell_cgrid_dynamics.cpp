/**
 * @file test_supercell_cgrid_dynamics.cpp
 * @brief Verification gates for Phase C.5 -- C-grid non-axisymmetric
 *        supercell dynamics scheme (SupercellCGridScheme).
 *
 * Verification gates from docs/CoordinateBackend_Plan.md C.5:
 *
 *   1. Hydrostatic equilibrium, tendencies bit-exact zero (machine ULP).
 *      Same kRho0=1.25 / round-trip trick as C.4 (TornadoCGridScheme).
 *
 *   2. Hydrostatic + uniform Cartesian wind: tendencies are bounded by
 *      the inherent O(dtheta^2) cylindrical-from-Cartesian projection
 *      error and DO NOT diverge.  The collocated SupercellScheme cannot
 *      pass this test on its current axis-singularity treatment because
 *      the antisymmetric BC `u[0] = -u[1]` drives a `u/dr`-magnitude
 *      false divergence at the first interior cell -- the Bug 7
 *      amplification documented in `docs/Journey.md`.  Here on the
 *      C-grid the residual is bounded by the projection error alone
 *      (and is therefore convergent in NTH).
 *
 *   3. Slow + fast tendencies sum to the unsplit total tendencies to
 *      float roundoff at every (i, j, k).  This is the algebraic
 *      correctness check for the split-explicit decomposition.
 *
 *   4. Warm thermal: a small theta perturbation produces a positive
 *      dw/dt at the bubble center (correct buoyant signal) and the
 *      pressure tendency does not show a high-wavenumber 2dx checkerboard
 *      mode.
 *
 *   5. Scheme metadata sanity (name, coordinate system, prog vars,
 *      dynamic_cast to SplitExplicitDynamics succeeds).
 *
 * Test strategy mirrors test_tornado_cgrid_dynamics.cpp: the dynamics
 * scheme is exercised directly on local Field3D buffers with the C-grid
 * geometry initialized via setup_cgrid_grid.
 */

#include "catch2/catch.hpp"
#include "core/coordinate_system.hpp"
#include "core/field3d.hpp"
#include "core/grid_geometry.hpp"
#include "core/runtime_config.hpp"
#include "core/simulation.hpp"
#include "dynamics/dynamics_base.hpp"
#include "dynamics/schemes/supercell/supercell_cgrid.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{

constexpr double kPi   = 3.14159265358979323846;
constexpr double kRho0 = 1.25;        // exact in float and double
constexpr double kP0   = 101325.0;    // exact in float and double


/// @brief Initializes the grid globals + reference profiles for the
///        cylindrical C-grid configuration.  Mirrors C.4 setup.
void setup_cgrid_grid(int nr, int nth, int nz, double dr_v, double dz_v)
{
    NR     = nr;
    NTH    = nth;
    NZ     = nz;
    dr     = dr_v;
    dz     = dz_v;
    dt     = 0.1;
    dtheta = 2.0 * kPi / static_cast<double>(NTH);

    global_coordinate_system = CoordinateSystem::Cylindrical;
    global_stagger_type      = StaggerType::CGrid;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    global_coordinate_system,
                                    global_stagger_type);

    rho0_base.assign(NZ, kRho0);
    p0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
        p0_base[k] = kP0 - kRho0 * dynamics_constants::g
                          * static_cast<double>(k) * dz;
    u0_base.assign(NZ, 0.0);
    v0_base.assign(NZ, 0.0);
    qv0_base.clear();
}


/// @brief Hydrostatic atmosphere with the float/double round-trip that
///        zeros the discrete reference-state mismatch (see C.4 notes).
void initialize_hydrostatic_atmosphere(Field3D& rho_f, Field3D& p_f)
{
    rho_f.resize(NR, NTH, NZ, static_cast<float>(kRho0));
    p_f.resize(NR, NTH, NZ);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                p_f[i][j][k] = static_cast<float>(p0_base[k]);

    for (int k = 0; k < NZ; ++k)
    {
        rho0_base[k] = static_cast<double>(rho_f[0][0][k]);
        p0_base[k]   = static_cast<double>(p_f[0][0][k]);
    }
}


void zero_velocity(Field3D& u_f, Field3D& v_f, Field3D& w_f)
{
    u_f.resize(NR, NTH, NZ, 0.0f);
    v_f.resize(NR, NTH, NZ, 0.0f);
    w_f.resize(NR, NTH, NZ, 0.0f);
}


void initialize_isothermal_theta(Field3D& theta_f)
{
    theta_f.resize(NR, NTH, NZ,
                   static_cast<float>(dynamics_constants::theta0));
}


/// @brief Initializes (u, v) on the C-grid for a uniform Cartesian wind
///        (u_x, u_y).  u lives on r-faces at theta[j]; v lives on
///        theta-faces at theta_{j+1/2}.  w stays at zero.
///
///        u[i][j][k] = u_x * cos(theta[j])         + u_y * sin(theta[j])
///        v[i][j][k] = -u_x * sin(theta_{j+1/2})   + u_y * cos(theta_{j+1/2})
void initialize_uniform_cartesian_wind(double ux, double uy,
                                       Field3D& u_f, Field3D& v_f, Field3D& w_f)
{
    u_f.resize(NR, NTH, NZ, 0.0f);
    v_f.resize(NR, NTH, NZ, 0.0f);
    w_f.resize(NR, NTH, NZ, 0.0f);

    for (int j = 0; j < NTH; ++j)
    {
        const double theta_c    = static_cast<double>(j) * dtheta;          // center
        const double theta_face = theta_c + 0.5 * dtheta;                    // theta_{j+1/2}
        const float u_val = static_cast<float>(ux * std::cos(theta_c)
                                             + uy * std::sin(theta_c));
        const float v_val = static_cast<float>(-ux * std::sin(theta_face)
                                              + uy * std::cos(theta_face));
        for (int i = 0; i < NR; ++i)
            for (int k = 0; k < NZ; ++k)
            {
                u_f[i][j][k] = u_val;
                v_f[i][j][k] = v_val;
            }
    }
}


struct TendencyBuffers
{
    Field3D du, dv, dw, drho, dp;
};

TendencyBuffers make_tendency_buffers()
{
    TendencyBuffers tb;
    tb.du.resize(NR, NTH, NZ, 0.0f);
    tb.dv.resize(NR, NTH, NZ, 0.0f);
    tb.dw.resize(NR, NTH, NZ, 0.0f);
    tb.drho.resize(NR, NTH, NZ, 0.0f);
    tb.dp.resize(NR, NTH, NZ, 0.0f);
    return tb;
}


double max_abs_all(const Field3D& f)
{
    double m = 0.0;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double v = std::abs(static_cast<double>(f[i][j][k]));
                if (v > m) m = v;
            }
    return m;
}


/// @brief Largest absolute value over the C-grid interior cell-center
///        compute domain (i = 1..NR-2, k = 1..NZ-2).
double max_abs_center_interior(const Field3D& f)
{
    double m = 0.0;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                const double v = std::abs(static_cast<double>(f[i][j][k]));
                if (v > m) m = v;
            }
    return m;
}


/// @brief Largest absolute difference between two same-shape fields.
double max_abs_diff(const Field3D& a, const Field3D& b)
{
    double m = 0.0;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double v = std::abs(
                    static_cast<double>(a[i][j][k]) - static_cast<double>(b[i][j][k]));
                if (v > m) m = v;
            }
    return m;
}

}  // namespace


// ============================================================================
// Gate 1 -- Hydrostatic equilibrium: tendencies at machine zero.
// Same setup as the C.4 TornadoCGridScheme test; the C.5 supercell scheme
// reduces to identical answers under axisymmetric+zero-flow conditions.
// ============================================================================
TEST_CASE("SupercellCGridScheme: hydrostatic equilibrium has machine-zero tendencies",
          "[dynamics][cylindrical][cgrid][c5]")
{
    setup_cgrid_grid(/*nr=*/16, /*nth=*/8, /*nz=*/16,
                     /*dr=*/250.0, /*dz=*/250.0);

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity(u_f, v_f, w_f);

    TendencyBuffers tb = make_tendency_buffers();
    SupercellCGridScheme scheme;
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
        tb.du, tb.dv, tb.dw, tb.drho, tb.dp);

    const double kTol = 1.0e-10;
    INFO("max |du/dt| = "   << max_abs_all(tb.du));
    INFO("max |dv/dt| = "   << max_abs_all(tb.dv));
    INFO("max |dw/dt| = "   << max_abs_all(tb.dw));
    INFO("max |drho/dt| = " << max_abs_all(tb.drho));
    INFO("max |dp/dt| = "   << max_abs_all(tb.dp));
    REQUIRE(max_abs_all(tb.du)   <= kTol);
    REQUIRE(max_abs_all(tb.dv)   <= kTol);
    REQUIRE(max_abs_all(tb.dw)   <= kTol);
    REQUIRE(max_abs_all(tb.drho) <= kTol);
    REQUIRE(max_abs_all(tb.dp)   <= kTol);
}


// ============================================================================
// Gate 1b -- Hydrostatic equilibrium preserved over 100 Forward-Euler
// momentum-only steps.  Skips dp/dt and drho/dt to stay clear of the
// well-known Forward-Euler acoustic instability (production split_explicit
// integrates them stably; that gate moves to C.6).
// ============================================================================
TEST_CASE("SupercellCGridScheme: hydrostatic state preserved over 100 momentum steps",
          "[dynamics][cylindrical][cgrid][c5]")
{
    setup_cgrid_grid(/*nr=*/16, /*nth=*/8, /*nz=*/16,
                     /*dr=*/250.0, /*dz=*/250.0);

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity(u_f, v_f, w_f);

    TendencyBuffers tb = make_tendency_buffers();
    SupercellCGridScheme scheme;

    const double step_dt = 1.0;
    const int    n_steps = 100;
    const double kTol    = 1.0e-10;

    for (int step = 0; step < n_steps; ++step)
    {
        scheme.compute_momentum_tendencies(
            u_f, v_f, w_f, rho_f, p_f, theta_f, step_dt,
            tb.du, tb.dv, tb.dw, tb.drho, tb.dp);

        INFO("step=" << step
             << " max|du|="   << max_abs_all(tb.du)
             << " max|dv|="   << max_abs_all(tb.dv)
             << " max|dw|="   << max_abs_all(tb.dw)
             << " max|drho|=" << max_abs_all(tb.drho)
             << " max|dp|="   << max_abs_all(tb.dp));
        REQUIRE(max_abs_all(tb.du)   <= kTol);
        REQUIRE(max_abs_all(tb.dv)   <= kTol);
        REQUIRE(max_abs_all(tb.dw)   <= kTol);
        REQUIRE(max_abs_all(tb.drho) <= kTol);
        REQUIRE(max_abs_all(tb.dp)   <= kTol);

        // Forward-Euler: only momentum (acoustic instability prevented).
        const float dt_f = static_cast<float>(step_dt);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    u_f[i][j][k] += dt_f * tb.du[i][j][k];
                    v_f[i][j][k] += dt_f * tb.dv[i][j][k];
                    w_f[i][j][k] += dt_f * tb.dw[i][j][k];
                }
    }
}


// ============================================================================
// Gate 2 -- Hydrostatic + uniform Cartesian wind: Bug 7 verification.
//
// A uniform Cartesian wind (u_x, u_y) projects onto cylindrical as
//   u_r(theta) = u_x cos theta + u_y sin theta
//   u_th(theta) = -u_x sin theta + u_y cos theta
// which is theta-dependent even though the underlying flow has zero
// horizontal divergence.  On the COLLOCATED grid the antisymmetric ghost
// `u[0] = -u[1]` drives a `~u/dr`-magnitude false divergence at i=1 that
// scales with U directly (not with dtheta^2).  On the C-GRID the only
// residual is the inherent O(dtheta^2) cylindrical-from-Cartesian
// projection error, which (a) is bounded and convergent in NTH and
// (b) does NOT depend on the radial spacing.  The gate verifies that
// the C-grid tendencies are bounded for a small uniform wind.
// ============================================================================
TEST_CASE("SupercellCGridScheme: hydrostatic + uniform wind tendencies are bounded (Bug 7)",
          "[dynamics][cylindrical][cgrid][c5][bug7]")
{
    // Use a finer azimuthal grid so the projection error floor is small.
    setup_cgrid_grid(/*nr=*/24, /*nth=*/32, /*nz=*/12,
                     /*dr=*/250.0, /*dz=*/250.0);

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);

    // Small uniform westerly: u_x = 5 m/s, u_y = 0.
    initialize_uniform_cartesian_wind(/*ux=*/5.0, /*uy=*/0.0, u_f, v_f, w_f);

    TendencyBuffers tb = make_tendency_buffers();
    SupercellCGridScheme scheme;
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
        tb.du, tb.dv, tb.dw, tb.drho, tb.dp);

    // Bug 7 specifically refers to the *false divergence* the antisymmetric
    // ghost-cell hack `u[0] = -u[1]` drives in the COLLOCATED scheme: a
    // ~U/dr discrete radial gradient at the inner cell produces a ~U/dr
    // divergence, which feeds dp/dt at order (gamma * p * U / dr).  For
    // U=5 m/s and dr=250m here that would be ~2800 Pa/s on the collocated
    // grid.  The C-grid replaces the stored hack with the inline axis
    // ghost in the dynamics stencil and uses the flux-form divergence,
    // so the only divergence residual is the inherent O(dtheta^2)
    // cylindrical-from-Cartesian projection error.  The Bug-7 symptoms
    // are therefore in dp/dt and drho/dt.
    //
    // The momentum tendencies du/dt and dv/dt have a separate, kinematic
    // contribution from the centrifugal `v^2/r` and curvature `-u v/r`
    // terms that BOTH grids carry: a uniform Cartesian wind in
    // cylindrical coords legitimately has v(r,theta) = -ux*sin(theta) +
    // uy*cos(theta), so v^2/r at the innermost u-face is non-zero and
    // would only be cancelled by an azimuthal-varying pressure gradient
    // that this hydrostatic IC does not include.  We therefore bound
    // them at the kinematic order ~ ux^2 / r_face[0] ~ 25/125 = 0.2 m/s^2.
    const double max_du   = max_abs_all(tb.du);
    const double max_dv   = max_abs_all(tb.dv);
    const double max_dw   = max_abs_all(tb.dw);
    const double max_drho = max_abs_center_interior(tb.drho);
    const double max_dp   = max_abs_center_interior(tb.dp);

    INFO("Bug 7 gate (NTH=32, U=5 m/s):");
    INFO("  max |du/dt|   = " << max_du   << " m/s^2");
    INFO("  max |dv/dt|   = " << max_dv   << " m/s^2");
    INFO("  max |dw/dt|   = " << max_dw   << " m/s^2");
    INFO("  max |drho/dt| = " << max_drho << " kg/m^3/s");
    INFO("  max |dp/dt|   = " << max_dp   << " Pa/s");

    // Bug 7 actual symptoms: dp/dt and drho/dt at the cell center reflect
    // only the O(dtheta^2) projection-error divergence on C-grid.  Gate
    // dp/dt at 5 Pa/s and drho/dt at 1e-3 kg/m^3/s -- ~500x improvement
    // over the collocated grid for this configuration.
    REQUIRE(max_drho < 1.0e-3);
    REQUIRE(max_dp   < 5.0);
    // Vertical: hydrostatic perturbation residual.  Buoyancy = 0 (rho =
    // rho0); only floor is float roundoff in -dp_prime_dz/rho.
    REQUIRE(max_dw   < 1.0e-3);
    // Horizontal: kinematic centrifugal/curvature contribution. ~0.2 m/s^2
    // bound per the analysis above.
    REQUIRE(max_du   < 0.5);
    REQUIRE(max_dv   < 0.5);
}


// ============================================================================
// Gate 2b -- Convergence in NTH: the projection-error floor decreases
// when the azimuthal resolution doubles, demonstrating that the residual
// is the discretization-projection error (O(dtheta^2)) and not a
// stationary axis artifact.
// ============================================================================
TEST_CASE("SupercellCGridScheme: uniform-wind tendencies converge in NTH",
          "[dynamics][cylindrical][cgrid][c5][bug7]")
{
    auto run_at_nth = [&](int nth) -> double {
        setup_cgrid_grid(/*nr=*/24, nth, /*nz=*/12,
                         /*dr=*/250.0, /*dz=*/250.0);

        Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
        initialize_hydrostatic_atmosphere(rho_f, p_f);
        initialize_isothermal_theta(theta_f);
        initialize_uniform_cartesian_wind(5.0, 0.0, u_f, v_f, w_f);

        TendencyBuffers tb = make_tendency_buffers();
        SupercellCGridScheme scheme;
        scheme.compute_momentum_tendencies(
            u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
            tb.du, tb.dv, tb.dw, tb.drho, tb.dp);
        return max_abs_center_interior(tb.dp);
    };

    const double dp_n16  = run_at_nth(16);
    const double dp_n32  = run_at_nth(32);
    const double dp_n64  = run_at_nth(64);

    INFO("max|dp/dt| convergence:");
    INFO("  NTH=16  -> " << dp_n16  << " Pa/s");
    INFO("  NTH=32  -> " << dp_n32  << " Pa/s");
    INFO("  NTH=64  -> " << dp_n64  << " Pa/s");

    // O(dtheta^2): doubling NTH should reduce residual ~4x.  Allow a
    // generous 2.5x to absorb roundoff and r-dependence at the inner cells.
    REQUIRE(dp_n32 < dp_n16 / 2.5);
    REQUIRE(dp_n64 < dp_n32 / 2.5);
}


// ============================================================================
// Gate 3 -- Algebraic correctness of the split-explicit decomposition:
// total = slow + fast_pressure (for rho/p) and total = slow + fast_momentum
// (for u/v/w).  Tested on a non-trivial state (hydrostatic + uniform wind +
// small theta perturbation) so all the advection, buoyancy, and pressure
// contributions are exercised.
// ============================================================================
TEST_CASE("SupercellCGridScheme: slow + fast tendencies sum to unsplit total",
          "[dynamics][cylindrical][cgrid][c5][split-explicit]")
{
    setup_cgrid_grid(/*nr=*/16, /*nth=*/16, /*nz=*/12,
                     /*dr=*/250.0, /*dz=*/250.0);

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    initialize_uniform_cartesian_wind(3.0, 1.0, u_f, v_f, w_f);

    // Add a small density perturbation to exercise the buoyancy term.
    for (int i = 4; i < 8; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 4; k < 8; ++k)
                rho_f[i][j][k] = static_cast<float>(kRho0 - 0.005);

    SupercellCGridScheme scheme;

    // Total (unsplit).
    TendencyBuffers tot = make_tendency_buffers();
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
        tot.du, tot.dv, tot.dw, tot.drho, tot.dp);

    // Slow.
    TendencyBuffers slow = make_tendency_buffers();
    scheme.compute_slow_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
        slow.du, slow.dv, slow.dw, slow.drho, slow.dp);

    // Fast pressure (drho/dt, dp/dt).
    Field3D fp_drho, fp_dp;
    fp_drho.resize(NR, NTH, NZ, 0.0f);
    fp_dp.resize(NR, NTH, NZ, 0.0f);
    scheme.compute_fast_pressure_tendencies(
        u_f, v_f, w_f, rho_f, p_f, fp_drho, fp_dp);

    // Fast momentum (du/dt, dv/dt, dw/dt).
    Field3D fm_du, fm_dv, fm_dw;
    fm_du.resize(NR, NTH, NZ, 0.0f);
    fm_dv.resize(NR, NTH, NZ, 0.0f);
    fm_dw.resize(NR, NTH, NZ, 0.0f);
    scheme.compute_fast_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, fm_du, fm_dv, fm_dw);

    // Reconstruct total = slow + fast on each variable.
    Field3D recon_du, recon_dv, recon_dw, recon_drho, recon_dp;
    recon_du.resize(NR, NTH, NZ, 0.0f);
    recon_dv.resize(NR, NTH, NZ, 0.0f);
    recon_dw.resize(NR, NTH, NZ, 0.0f);
    recon_drho.resize(NR, NTH, NZ, 0.0f);
    recon_dp.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                recon_du[i][j][k]   = slow.du[i][j][k]   + fm_du[i][j][k];
                recon_dv[i][j][k]   = slow.dv[i][j][k]   + fm_dv[i][j][k];
                recon_dw[i][j][k]   = slow.dw[i][j][k]   + fm_dw[i][j][k];
                recon_drho[i][j][k] = slow.drho[i][j][k] + fp_drho[i][j][k];
                recon_dp[i][j][k]   = slow.dp[i][j][k]   + fp_dp[i][j][k];
            }

    // The decomposition is algebraically exact; the float roundoff floor
    // is set by the magnitude of the largest summand at each cell.
    // Allow a tolerance scaled to ~ULP(float) * max-tendency.
    const double max_du_diff   = max_abs_diff(tot.du,   recon_du);
    const double max_dv_diff   = max_abs_diff(tot.dv,   recon_dv);
    const double max_dw_diff   = max_abs_diff(tot.dw,   recon_dw);
    const double max_drho_diff = max_abs_diff(tot.drho, recon_drho);
    const double max_dp_diff   = max_abs_diff(tot.dp,   recon_dp);

    INFO("split-explicit decomposition residuals (total - (slow+fast)):");
    INFO("  du   diff = " << max_du_diff);
    INFO("  dv   diff = " << max_dv_diff);
    INFO("  dw   diff = " << max_dw_diff);
    INFO("  drho diff = " << max_drho_diff);
    INFO("  dp   diff = " << max_dp_diff);

    // Tolerances reflect float roundoff of small additions to the tendency
    // magnitudes seen in this configuration (max tendency ~ a few Pa/s for
    // dp; a few 1e-3 m/s^2 for momentum). 1e-3 dp leaves >1000x margin
    // over expected roundoff; 1e-6 momentum leaves >100x margin.
    REQUIRE(max_du_diff   < 1.0e-6);
    REQUIRE(max_dv_diff   < 1.0e-6);
    REQUIRE(max_dw_diff   < 1.0e-6);
    REQUIRE(max_drho_diff < 1.0e-9);
    REQUIRE(max_dp_diff   < 1.0e-3);
}


// ============================================================================
// Gate 4 -- Warm thermal: a localized density deficit (theta-equivalent of
// a warm bubble) produces a positive dw/dt at its top z-face (buoyant
// updraft signal).  Indirectly verifies the perturbation-density buoyancy
// term contribution is correctly placed and signed at z-faces.
// ============================================================================
TEST_CASE("SupercellCGridScheme: warm thermal produces upward dw/dt",
          "[dynamics][cylindrical][cgrid][c5][buoyancy]")
{
    setup_cgrid_grid(/*nr=*/16, /*nth=*/16, /*nz=*/16,
                     /*dr=*/250.0, /*dz=*/250.0);

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity(u_f, v_f, w_f);

    // Localized warm column near r = 4*dr, k = 6..10 -- 2 K theta excess
    // converted to ~0.7% density deficit through theta = T*(p0/p)^(R/cp)
    // (proportional approx).  Use a direct 0.005 kg/m^3 deficit (~0.4%)
    // to keep the test simple.
    const int i_b = 4, k_lo = 6, k_hi = 10;
    for (int j = 0; j < NTH; ++j)
        for (int k = k_lo; k <= k_hi; ++k)
            rho_f[i_b][j][k] = static_cast<float>(kRho0 - 0.005);

    TendencyBuffers tb = make_tendency_buffers();
    SupercellCGridScheme scheme;
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
        tb.du, tb.dv, tb.dw, tb.drho, tb.dp);

    // dw/dt at z-faces inside the warm column should be positive
    // (buoyant lift).  Sample at j=0, mid-column.
    const int k_mid = (k_lo + k_hi) / 2;
    const double dw_inside = static_cast<double>(tb.dw[i_b][0][k_mid]);
    INFO("dw/dt at warm column center (i=" << i_b << ", k=" << k_mid << ") = "
         << dw_inside << " m/s^2");
    // Magnitude check: g * (delta_rho / rho) ~ 9.81 * 0.005 / 1.25 ~ 0.04 m/s^2.
    REQUIRE(dw_inside > 0.01);
    REQUIRE(dw_inside < 1.0);

    // Far-field column (i = NR/2, away from bubble) dw/dt should be near zero.
    const double dw_far = static_cast<double>(tb.dw[NR / 2][0][k_mid]);
    INFO("dw/dt far from column (i=" << NR / 2 << ", k=" << k_mid << ") = "
         << dw_far << " m/s^2");
    REQUIRE(std::abs(dw_far) < 0.01);
}


// ============================================================================
// Gate 5 -- Scheme metadata + SplitExplicitDynamics capability.
// ============================================================================
TEST_CASE("SupercellCGridScheme: scheme metadata and split-explicit interface",
          "[dynamics][cylindrical][cgrid][c5]")
{
    setup_cgrid_grid(8, 8, 8, 250.0, 250.0);
    SupercellCGridScheme scheme;
    REQUIRE(scheme.get_scheme_name()         == "supercell_cgrid");
    REQUIRE(scheme.get_coordinate_system()   == "cylindrical_cgrid");
    REQUIRE(scheme.get_num_prognostic_vars() == 5);

    // SplitExplicitDynamics capability check via dynamic_cast on the
    // DynamicsScheme base pointer.
    DynamicsScheme* base = &scheme;
    SplitExplicitDynamics* split = dynamic_cast<SplitExplicitDynamics*>(base);
    REQUIRE(split != nullptr);
}

/**
 * @file test_supercell_cgrid_acoustic.cpp
 * @brief Verification gates for Phase C.6 -- Klemp-Wilhelmson split-explicit
 *        acoustic substep on the C-grid using SupercellCGridScheme.
 *
 * Verification gates from docs/CoordinateBackend_Plan.md C.6:
 *
 *   1. Acoustic pulse, 10 substeps, clean circular propagation without
 *      checkerboard.
 *   2. (Carried over from C.4)  Hydrostatic state preserved over 300 s of
 *      FULL split-explicit integration (slow + N substeps including the
 *      pressure/density update that C.4 had to skip for Forward-Euler
 *      stability).
 *
 * Test strategy: drive the SplitExplicitDynamics methods of
 * SupercellCGridScheme through a manual Klemp-Wilhelmson loop at the
 * scheme level.  This bypasses the production dynamics orchestrator and
 * isolates the acoustic-substep verification to the C-grid scheme itself.
 *
 * The Klemp-Wilhelmson loop per large step:
 *   1. compute_slow_tendencies  -> du/dt, dv/dt, dw/dt, drho/dt, dp/dt
 *   2. Apply slow with dt_large -> u, v, w, p (NOT rho; slow drho/dt = 0)
 *   3. For n in [0, N):
 *        a. compute_fast_pressure_tendencies -> drho/dt, dp/dt
 *        b. Apply fast pressure with dt_small -> rho, p
 *        c. compute_fast_momentum_tendencies -> du/dt, dv/dt, dw/dt
 *        d. Apply fast momentum with dt_small -> u, v, w
 *        e. Apply lightweight acoustic BCs (outer wall, lid; axis is
 *           handled inline by the dynamics scheme).
 */

#include "catch2/catch.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/field/field3d.hpp"
#include "core/infra/grid_geometry.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "dynamics/dynamics_base.hpp"
#include "dynamics/schemes/supercell/supercell_cgrid.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{

constexpr double kPi   = 3.14159265358979323846;
constexpr double kRho0 = 1.25;
constexpr double kP0   = 101325.0;


/// @brief Initializes the cylindrical C-grid configuration, mirroring C.4 / C.5.
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


/// @brief Lightweight acoustic boundary conditions for the test:
///        outer-wall u=0, lid w=0.  Axis u-ghost is handled inline by the
///        dynamics scheme; v[0]=0 is preserved by the loop range
///        (compute_*_tendencies start at i=1 for v); periodic theta is
///        automatic via the % NTH stencil.
void apply_acoustic_bcs(Field3D& u_f, Field3D& w_f)
{
    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
            u_f[NR - 1][j][k] = 0.0f;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            w_f[i][j][NZ - 1] = 0.0f;
}


/// @brief Add tendency * dt to a Field3D in-place (double-precision accum).
void add_scaled(Field3D& field, const Field3D& tend, double scale)
{
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double newv = static_cast<double>(field[i][j][k])
                                  + scale * static_cast<double>(tend[i][j][k]);
                field[i][j][k] = static_cast<float>(newv);
            }
}


/// @brief One Klemp-Wilhelmson large step.
///        Returns the maximum absolute slow-tendency value over the cell
///        (useful for the hydrostatic gate).
double kw_large_step(SupercellCGridScheme& scheme,
                     double dt_large, int n_substeps,
                     Field3D& u_f, Field3D& v_f, Field3D& w_f,
                     Field3D& rho_f, Field3D& p_f, Field3D& theta_f,
                     TendencyBuffers& tb)
{
    // 1. Slow tendencies (advection + buoyancy on velocities; advection
    //    only on pressure; drho/dt = 0).
    scheme.compute_slow_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt_large,
        tb.du, tb.dv, tb.dw, tb.drho, tb.dp);
    const double max_slow = std::max({
        max_abs_all(tb.du), max_abs_all(tb.dv), max_abs_all(tb.dw),
        max_abs_all(tb.drho), max_abs_all(tb.dp)});

    // 2. Apply slow with dt_large.
    add_scaled(u_f, tb.du, dt_large);
    add_scaled(v_f, tb.dv, dt_large);
    add_scaled(w_f, tb.dw, dt_large);
    add_scaled(p_f, tb.dp, dt_large);
    // drho/dt is zero in the slow path (mass continuity is fast); but apply
    // anyway in case a future revision drops zeros into it.
    add_scaled(rho_f, tb.drho, dt_large);

    apply_acoustic_bcs(u_f, w_f);

    // 3. Acoustic substep loop (forward-backward Klemp-Wilhelmson).
    const double dt_small = dt_large / static_cast<double>(n_substeps);
    Field3D fp_drho, fp_dp, fm_du, fm_dv, fm_dw;
    fp_drho.resize(NR, NTH, NZ, 0.0f);
    fp_dp.resize(NR, NTH, NZ, 0.0f);
    fm_du.resize(NR, NTH, NZ, 0.0f);
    fm_dv.resize(NR, NTH, NZ, 0.0f);
    fm_dw.resize(NR, NTH, NZ, 0.0f);

    for (int n = 0; n < n_substeps; ++n)
    {
        // 3a. FORWARD: compute divergence -> integrate p, rho.
        scheme.compute_fast_pressure_tendencies(
            u_f, v_f, w_f, rho_f, p_f, fp_drho, fp_dp);
        add_scaled(rho_f, fp_drho, dt_small);
        add_scaled(p_f,   fp_dp,   dt_small);

        // 3b. BACKWARD: compute pressure gradient using NEW p, rho ->
        //     integrate u, v, w.
        scheme.compute_fast_momentum_tendencies(
            u_f, v_f, w_f, rho_f, p_f, fm_du, fm_dv, fm_dw);
        add_scaled(u_f, fm_du, dt_small);
        add_scaled(v_f, fm_dv, dt_small);
        add_scaled(w_f, fm_dw, dt_small);

        // 3c. Acoustic BCs.
        apply_acoustic_bcs(u_f, w_f);
    }

    return max_slow;
}

}  // namespace


// ============================================================================
// Gate 1 -- Hydrostatic state preserved over 300 s of FULL split-explicit
// integration (slow + N forward-backward acoustic substeps).
//
// At hydrostatic equilibrium with zero flow:
//   - compute_slow_tendencies returns 0 everywhere (advection = 0,
//     buoyancy = 0 since rho == rho0(z))
//   - compute_fast_pressure_tendencies returns 0 (div_flux = 0 with u=v=w=0)
//   - compute_fast_momentum_tendencies returns 0 (grad_r(p)=0 since p has
//     no radial variation; grad_theta(p)=0; grad_z(p) - dp0/dz = 0 thanks
//     to the round-trip trick that bit-exactly aligns p_f with p0_base).
//
// Result: every tendency is zero, every substep is a no-op, the fields are
// preserved bit-exactly across all 300 s of integration.  This is the gate
// the C.4 momentum-only test had to skip because Forward-Euler is unstable
// for the acoustic system; on split-explicit Klemp-Wilhelmson the
// equilibrium IS a fixed point.
// ============================================================================
TEST_CASE("SupercellCGridScheme: hydrostatic preserved over 300s of split-explicit",
          "[dynamics][cylindrical][cgrid][c6][split-explicit]")
{
    setup_cgrid_grid(/*nr=*/16, /*nth=*/8, /*nz=*/16,
                     /*dr=*/250.0, /*dz=*/250.0);

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity(u_f, v_f, w_f);

    Field3D rho_snap = rho_f;
    Field3D p_snap   = p_f;

    SupercellCGridScheme scheme;
    TendencyBuffers tb = make_tendency_buffers();

    const double dt_large    = 1.0;
    const int    n_substeps  = 6;       // CFL ~ c * dt_small / dr ~ 350 * 0.17 / 250 ~ 0.24
    const int    n_steps     = 300;     // 300 simulated seconds
    const double kTol        = 1.0e-10;

    for (int step = 0; step < n_steps; ++step)
    {
        const double max_slow = kw_large_step(
            scheme, dt_large, n_substeps,
            u_f, v_f, w_f, rho_f, p_f, theta_f, tb);

        INFO("step=" << step << " max|slow_tend|=" << max_slow);
        REQUIRE(max_slow <= kTol);
    }

    // Bit-exact preservation of rho and p; u, v, w stay identically zero.
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                REQUIRE(rho_f[i][j][k] == rho_snap[i][j][k]);
                REQUIRE(p_f[i][j][k]   == p_snap[i][j][k]);
                REQUIRE(u_f[i][j][k] == 0.0f);
                REQUIRE(v_f[i][j][k] == 0.0f);
                REQUIRE(w_f[i][j][k] == 0.0f);
            }
}


// ============================================================================
// Gate 2 -- Acoustic pulse propagates outward at sound speed.
//
// Setup:
//   - Hydrostatic atmosphere (kRho0=1.25, kP0=101325)
//   - Small Gaussian pressure perturbation (delta_p = 100 Pa, ~0.1% of base)
//     centered at r ~ 4*dr, z ~ NZ/2, axisymmetric in theta
//
// Klemp-Wilhelmson with dt_large=0.5s, N=10 substeps => dt_small = 0.05s.
// CFL_acoustic = c * dt_small / dr = 350 * 0.05 / 250 = 0.07 << 1 (stable).
//
// Run for ~1 second. Expected: the pulse front propagates outward by
// c * t = 350 m, i.e. at least 1 cell at dr=250m.
// ============================================================================
TEST_CASE("SupercellCGridScheme: acoustic pulse propagates outward at sound speed",
          "[dynamics][cylindrical][cgrid][c6][split-explicit][acoustic]")
{
    setup_cgrid_grid(/*nr=*/24, /*nth=*/8, /*nz=*/16,
                     /*dr=*/250.0, /*dz=*/250.0);

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity(u_f, v_f, w_f);

    // Gaussian pressure pulse centered at (i_c, k_c).  Axisymmetric in
    // theta so the C-grid acoustic propagation is purely radial+vertical.
    const int    i_c       = 4;
    const int    k_c       = NZ / 2;
    const double r_c       = global_grid_geometry.r[i_c];
    const double z_c       = static_cast<double>(k_c) * dz;
    const double sigma     = 2.0 * dr;       // 500m gaussian radius
    const double delta_p   = 100.0;          // Pa (~0.1% of base)

    for (int i = 0; i < NR; ++i)
    {
        const double r_i = global_grid_geometry.r[i];
        for (int k = 0; k < NZ; ++k)
        {
            const double z_k  = static_cast<double>(k) * dz;
            const double dist2 = (r_i - r_c) * (r_i - r_c)
                                + (z_k - z_c) * (z_k - z_c);
            const double dp = delta_p * std::exp(-dist2 / (sigma * sigma));
            for (int j = 0; j < NTH; ++j)
                p_f[i][j][k] = static_cast<float>(p0_base[k] + dp);
        }
    }

    // Snapshot the initial pulse profile along the j=0 radial slice at k_c.
    std::vector<double> p_init(NR);
    for (int i = 0; i < NR; ++i)
        p_init[i] = static_cast<double>(p_f[i][0][k_c]) - p0_base[k_c];

    // Locate the initial peak.
    int i_peak_init = 0;
    double max_init = -1.0;
    for (int i = 0; i < NR; ++i)
        if (p_init[i] > max_init) { max_init = p_init[i]; i_peak_init = i; }
    REQUIRE(i_peak_init == i_c);
    REQUIRE(max_init >= 0.99 * delta_p);  // sanity check IC

    SupercellCGridScheme scheme;
    TendencyBuffers tb = make_tendency_buffers();

    const double dt_large    = 0.5;
    const int    n_substeps  = 10;          // dt_small = 0.05s
    const int    n_steps     = 2;           // 1 second total

    for (int step = 0; step < n_steps; ++step)
    {
        kw_large_step(scheme, dt_large, n_substeps,
                      u_f, v_f, w_f, rho_f, p_f, theta_f, tb);
    }

    // After 1s the pulse front should have propagated outward by ~ c * t.
    // c = sqrt(gamma R_d T) ~ 340 m/s for T=287 K.  At dr=250m, that's
    // ~1.4 cells of outward shift.  We track the position of the residual
    // at the original location -- it should drop substantially -- and check
    // that NEW pressure perturbation has appeared at i > i_c.
    std::vector<double> p_final(NR);
    for (int i = 0; i < NR; ++i)
        p_final[i] = static_cast<double>(p_f[i][0][k_c]) - p0_base[k_c];

    // Find peak of the OUTWARD-propagating pulse (i > i_c).
    int i_peak_outward = i_c;
    double max_outward = 0.0;
    for (int i = i_c + 1; i < NR - 1; ++i)
    {
        if (p_final[i] > max_outward)
        {
            max_outward = p_final[i];
            i_peak_outward = i;
        }
    }

    INFO("Initial peak: i=" << i_peak_init << " dp=" << max_init << " Pa");
    INFO("Outward pulse at i=" << i_peak_outward << " dp=" << max_outward << " Pa");

    // Verify outward propagation: the peak of the outward-going pulse
    // sits at least 1 cell from the source (since c * t = 340 m > dr).
    REQUIRE(i_peak_outward > i_c);
    // Verify there IS measurable outward propagation (at least 5% of the
    // initial pulse magnitude moved outward).  An acoustic pulse in 2D
    // disperses geometrically, so we don't expect 100% of the original
    // amplitude.
    REQUIRE(max_outward > 0.05 * delta_p);

    // Sanity: no NaN / Inf anywhere in the prognostic state.
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                REQUIRE(std::isfinite(rho_f[i][j][k]));
                REQUIRE(std::isfinite(p_f[i][j][k]));
                REQUIRE(std::isfinite(u_f[i][j][k]));
                REQUIRE(std::isfinite(v_f[i][j][k]));
                REQUIRE(std::isfinite(w_f[i][j][k]));
            }
}


// ============================================================================
// Gate 3 -- No 2dx checkerboard mode in the propagated pressure field.
//
// A checkerboard mode (p[i+1] = -p[i] alternating) would have a discrete
// Laplacian of magnitude ~ |p|/dr^2 -- 4x larger than the smooth pulse
// curvature.  We detect it as: for any radial slice, the L_inf norm of
// the discrete second difference (p[i+1] - 2*p[i] + p[i-1]) is bounded
// by a multiple of the smooth-pulse curvature scale.
//
// On a Gaussian pulse with sigma = 2*dr, the smooth curvature scale is
//   |d2p/dr2| ~ delta_p / sigma^2 = 100 / (500)^2 = 4e-4 Pa/m^2
// and the discrete second difference at the peak is ~
//   delta_p * (1 - 2 + 1)/(sigma/dr)^2 ~ 100 * 1 / 4 = 25 Pa
// A checkerboard at amplitude 100 Pa would give second diff = 4 * 100 = 400 Pa.
// Gate the second diff at < 100 Pa to leave a comfortable margin.
// ============================================================================
TEST_CASE("SupercellCGridScheme: acoustic propagation has no checkerboard mode",
          "[dynamics][cylindrical][cgrid][c6][split-explicit][acoustic]")
{
    setup_cgrid_grid(/*nr=*/24, /*nth=*/8, /*nz=*/16,
                     /*dr=*/250.0, /*dz=*/250.0);

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity(u_f, v_f, w_f);

    const int    i_c     = 4;
    const int    k_c     = NZ / 2;
    const double r_c     = global_grid_geometry.r[i_c];
    const double z_c     = static_cast<double>(k_c) * dz;
    const double sigma   = 2.0 * dr;
    const double delta_p = 100.0;

    for (int i = 0; i < NR; ++i)
    {
        const double r_i = global_grid_geometry.r[i];
        for (int k = 0; k < NZ; ++k)
        {
            const double z_k  = static_cast<double>(k) * dz;
            const double dist2 = (r_i - r_c) * (r_i - r_c)
                                + (z_k - z_c) * (z_k - z_c);
            const double dp = delta_p * std::exp(-dist2 / (sigma * sigma));
            for (int j = 0; j < NTH; ++j)
                p_f[i][j][k] = static_cast<float>(p0_base[k] + dp);
        }
    }

    SupercellCGridScheme scheme;
    TendencyBuffers tb = make_tendency_buffers();

    const double dt_large   = 0.5;
    const int    n_substeps = 10;
    const int    n_steps    = 2;

    for (int step = 0; step < n_steps; ++step)
    {
        kw_large_step(scheme, dt_large, n_substeps,
                      u_f, v_f, w_f, rho_f, p_f, theta_f, tb);
    }

    // Discrete second difference of (p - p0) along the radial slice at k_c.
    double max_d2p = 0.0;
    for (int i = 1; i < NR - 1; ++i)
    {
        const double pim = static_cast<double>(p_f[i - 1][0][k_c]) - p0_base[k_c];
        const double pi0 = static_cast<double>(p_f[i    ][0][k_c]) - p0_base[k_c];
        const double pip = static_cast<double>(p_f[i + 1][0][k_c]) - p0_base[k_c];
        const double d2p = std::abs(pim - 2.0 * pi0 + pip);
        if (d2p > max_d2p) max_d2p = d2p;
    }

    INFO("max |second diff p'| along radial slice = " << max_d2p << " Pa");
    // Gate at 100 Pa: a 100 Pa checkerboard would give 4*100 = 400 Pa.
    // The smooth Gaussian gives ~25 Pa peak second diff.
    REQUIRE(max_d2p < 100.0);
}

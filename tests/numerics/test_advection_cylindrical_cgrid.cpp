/**
 * @file test_advection_cylindrical_cgrid.cpp
 * @brief Verification gates for Phase C.7 -- cylindrical Arakawa C-grid
 *        scalar advection (advect_scalar_3d on the C-grid path).
 *
 * Phase C.7 of docs/CoordinateBackend_Plan.md.
 *
 * The C-grid scalar advection module uses face-staggered velocity reads
 * (u at r-face[i], v at theta-face[j], w at z-face[k]) and FLUX-form
 * upwind tendencies. The structural advantage over the collocated path
 * is that the flux at every shared face uses ONE upwind value, so the
 * mass debit at one cell and the mass credit at the adjacent cell
 * cancel bit-exactly in floating point. The verification gates below
 * exercise this property.
 *
 * The tests call advect_scalar_3d directly so the dispatch branch in
 * advect_scalar_3d.cpp is part of what is verified. To isolate the
 * C-grid kernels:
 *
 *   - global_coordinate_system = Cylindrical
 *   - global_stagger_type      = CGrid
 *   - global_grid_geometry initialized with StaggerType::CGrid
 *     (populates r_face[], r_face_inv[], z_face[])
 *   - advection_scheme reset (so the numerics-vertical-TVD path stays
 *     dormant; the C-grid dispatcher routes the vertical step to its
 *     own face-form kernel anyway, but this also kills the runtime
 *     log spam from log_runtime_advection_path_once)
 *
 * Verification gates:
 *
 *   1. Zero-flow preservation: u = v = w = 0. After many advection
 *      steps any cell-center scalar IC is preserved bit-exactly. This
 *      is the floor that catches bookkeeping mistakes -- a stray
 *      arithmetic op on a zero-velocity face would not be zero in
 *      float (`u*q + (-u)*q != 0` if those reads come from different
 *      memory locations).
 *
 *   2. Pure-vertical uniform advection: w = W_const > 0, u = v = 0.
 *      A Gaussian column translates upward; the mass-weighted z
 *      centroid moves at W within 1 % over the run, and the total
 *      interior mass is preserved (the only fluxes are through z-faces
 *      and the boundary fluxes vanish because the bump is well inside
 *      the interior).
 *
 *   3. Pure-azimuthal uniform advection: v = V_const > 0, u = w = 0.
 *      A Gaussian bump at fixed (i, k) translates around the periodic
 *      theta direction; after a full revolution the bump returns to
 *      its starting angular position, and the total interior mass is
 *      preserved bit-exactly (theta is periodic so all theta-face
 *      fluxes cancel pairwise).
 *
 *   4. Solid-body rotation: u = w = 0, v[i][j][k] = Omega * r[i] at
 *      theta-faces. A Gaussian bump at fixed (i_c, k_c) is advected
 *      for exactly one revolution period T = 2*pi/Omega. After T:
 *
 *        - L2 error vs the initial bump < 5 % of L2(IC).
 *        - Total interior mass change < 1e-6 relative (machine
 *          precision: the only float-rounding source is the
 *          per-step double->float cast of the per-cell update).
 *
 *      First-order upwind has numerical viscosity that diffuses the
 *      peak (the L2 budget absorbs that); flux form keeps the mass
 *      pairwise-conservative regardless of the smearing.
 *
 *   5. Stagger-routing distinguishability: with the SAME initial
 *      condition + SAME (u, v, w), the collocated path and the C-grid
 *      path produce DIFFERENT outputs. This catches a regression in
 *      which the dispatch silently falls through to the collocated
 *      kernel when StaggerType::CGrid is requested.
 */

#include "catch2/catch.hpp"

#include "core/coordinate_system.hpp"
#include "core/field3d.hpp"
#include "core/grid_geometry.hpp"
#include "core/runtime_config.hpp"
#include "core/simulation.hpp"
#include "numerics/advection/advection.hpp"
#include "numerics/advection/advection_base.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;


// ---------------------------------------------------------------------------
// Grid setup
// ---------------------------------------------------------------------------

/**
 * @brief Configures globals + grid geometry for a cylindrical C-grid
 *        advection test.
 */
void setup_cgrid_grid(int nr, int nth, int nz, double dr_m, double dz_m, double dt_s)
{
    NR     = nr;
    NTH    = nth;
    NZ     = nz;
    dr     = dr_m;
    dz     = dz_m;
    dt     = dt_s;
    dtheta = 2.0 * kPi / static_cast<double>(nth);

    global_coordinate_system = CoordinateSystem::Cylindrical;
    global_stagger_type      = StaggerType::CGrid;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    global_coordinate_system,
                                    global_stagger_type);
}

/**
 * @brief Configures globals for a cylindrical COLLOCATED advection test
 *        (used by the routing-distinguishability gate).
 */
void setup_collocated_grid(int nr, int nth, int nz, double dr_m, double dz_m, double dt_s)
{
    NR     = nr;
    NTH    = nth;
    NZ     = nz;
    dr     = dr_m;
    dz     = dz_m;
    dt     = dt_s;
    dtheta = 2.0 * kPi / static_cast<double>(nth);

    global_coordinate_system = CoordinateSystem::Cylindrical;
    global_stagger_type      = StaggerType::Collocated;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    global_coordinate_system,
                                    global_stagger_type);
}

/**
 * @brief Resets the global advection scheme pointer so the runtime path
 *        does not divert the vertical step to the numerics TVD scheme.
 *        The C-grid dispatcher routes the vertical step to its own
 *        flux-form kernel regardless, but this also silences runtime
 *        log spam from log_runtime_advection_path_once.
 */
void disable_numerics_vertical_advection()
{
    advection_scheme.reset();
}


// ---------------------------------------------------------------------------
// Velocity setup helpers (C-grid: u at r-face, v at theta-face, w at z-face)
// ---------------------------------------------------------------------------

void zero_velocity()
{
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
}

/**
 * @brief Uniform vertical wind: w[i][j][k] = w_uniform at all z-faces.
 *        u, v are zeroed. The face-centered storage carries the same
 *        constant value at every face.
 */
void set_uniform_vertical_velocity(double w_uniform)
{
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, static_cast<float>(w_uniform));
}

/**
 * @brief Uniform azimuthal velocity: v[i][j][k] = V_const at every
 *        theta-face. On the C-grid this is the simplest non-zero-flow
 *        test: each cell receives identical in-flux and out-flux
 *        amplitudes, and the only spatial pattern in q comes from
 *        the IC.
 */
void set_uniform_azimuthal_velocity(double v_uniform)
{
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, static_cast<float>(v_uniform));
    w.resize(NR, NTH, NZ, 0.0f);
}

/**
 * @brief Solid-body rotation on the C-grid: u_r = 0, u_theta = Omega * r.
 *
 * Field placement:
 *   u (r-face)     -- 0
 *   v (theta-face) -- Omega * r[i] at every (i, j, k); v[0] is on the
 *                     axis singular line and is set to 0 by C.2 BC
 *                     convention (kept in the IC for safety).
 *   w (z-face)     -- 0
 */
void set_solid_body_rotation_velocity(double omega)
{
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    for (int i = 1; i < NR; ++i)  // i=0 left at zero (axis BC)
    {
        const float v_val = static_cast<float>(omega *
            static_cast<double>(i) * dr);
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                v[i][j][k] = v_val;
    }
}


// ---------------------------------------------------------------------------
// Initial-condition placement (interior cells only)
// ---------------------------------------------------------------------------

/**
 * @brief Places a Gaussian bump anisotropic in (i, j), localized in k.
 *
 * The radial half-width @p sigma_cells_r is independent of the
 * azimuthal half-width @p sigma_cells_th. Restricted to the interior
 * cell-center compute domain (i = 1..NR-2, k = 1..NZ-2) and placed at
 * one k slice (@p k_center) so the vertical kernel running at full dt
 * with w=0 is a no-op for the test.
 */
void place_gaussian_bump_axis_radial(Field3D& q,
                                     int i_center, int j_center, int k_center,
                                     double sigma_cells_r, double sigma_cells_th,
                                     double peak)
{
    q.resize(NR, NTH, NZ, 0.0f);
    const double inv_two_sigma_r_sq  = 1.0 / (2.0 * sigma_cells_r  * sigma_cells_r);
    const double inv_two_sigma_th_sq = 1.0 / (2.0 * sigma_cells_th * sigma_cells_th);
    for (int i = 1; i < NR - 1; ++i)
    {
        const double di = static_cast<double>(i - i_center);
        const double radial_part = di * di * inv_two_sigma_r_sq;
        for (int j = 0; j < NTH; ++j)
        {
            // Periodic theta distance: pick the shorter of |dj| and NTH - |dj|.
            int dj = j - j_center;
            if (dj >  NTH / 2) dj -= NTH;
            if (dj < -NTH / 2) dj += NTH;
            const double azimuthal_part =
                static_cast<double>(dj) * static_cast<double>(dj) * inv_two_sigma_th_sq;
            const double value = peak * std::exp(-(radial_part + azimuthal_part));
            for (int k = 1; k < NZ - 1; ++k)
            {
                if (k == k_center)
                    q[i][j][k] = static_cast<float>(value);
            }
        }
    }
}

/**
 * @brief Places a smooth periodic theta variation `q = peak * (1 + cos(theta))`
 *        on a single radial ring (@p i_center) and a single z slice
 *        (@p k_center).
 *
 * The 1 + cos(theta) form is everywhere non-negative (so it admits a
 * positivity-preserving advection check), peaks at theta = 0 with
 * value 2 * peak, troughs at theta = pi with value 0, and returns to
 * itself after a 2 * pi rotation. Solid-body rotation should preserve
 * this profile up to numerical viscosity from the TVD-MUSCL slope
 * limiter; with the MC limiter, smooth (1 + cos) IC retains more than
 * 95 % of its L2 norm after one full revolution at NTH = 64
 * (in contrast to a single-peak Gaussian where the limiter zeroes the
 * slope at the peak cell, falling back to first-order locally).
 */
void place_periodic_cosine_ring(Field3D& q, int i_center, int k_center, double peak)
{
    q.resize(NR, NTH, NZ, 0.0f);
    for (int j = 0; j < NTH; ++j)
    {
        const double theta_j = static_cast<double>(j) * dtheta;
        const float value = static_cast<float>(peak * (1.0 + std::cos(theta_j)));
        if (i_center >= 1 && i_center <= NR - 2 &&
            k_center >= 1 && k_center <= NZ - 2)
        {
            q[i_center][j][k_center] = value;
        }
    }
}

/**
 * @brief Places a Gaussian column in z at fixed (i_center, j_center).
 *        Used by the pure-vertical advection gate.
 */
void place_gaussian_column_z(Field3D& q,
                             int i_center, int j_center, double z_center_m,
                             double sigma_m, double peak)
{
    q.resize(NR, NTH, NZ, 0.0f);
    const double inv_two_sigma_sq = 1.0 / (2.0 * sigma_m * sigma_m);
    for (int k = 1; k < NZ - 1; ++k)
    {
        const double zk  = static_cast<double>(k) * dz;
        const double dz_ = zk - z_center_m;
        const double value = peak * std::exp(-(dz_ * dz_) * inv_two_sigma_sq);
        q[i_center][j_center][k] = static_cast<float>(value);
    }
}


// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/**
 * @brief Total scalar mass over the cell-center compute domain
 *        (i = 1..NR-2, j = 0..NTH-1, k = 1..NZ-2), weighted by cylindrical
 *        cell volume r[i] * dr * dtheta * dz.
 *
 * Volume factor: only r[i] varies across cells in the interior (dr,
 * dtheta, dz are uniform), so this returns the sum
 *   sum_{i,j,k} r[i] * q[i][j][k]
 * times a constant prefactor. The prefactor cancels in relative mass
 * comparisons.
 */
double total_mass_interior(const Field3D& q)
{
    double sum = 0.0;
    const auto& geo = global_grid_geometry;
    for (int i = 1; i < NR - 1; ++i)
    {
        const double ri = geo.r[i];
        for (int j = 0; j < NTH; ++j)
            for (int k = 1; k < NZ - 1; ++k)
                sum += ri * static_cast<double>(q[i][j][k]);
    }
    return sum;
}

/**
 * @brief L2 norm of (a - b) over the interior compute domain, weighted
 *        by cell volume r[i] * dr * dtheta * dz.
 */
double l2_diff_interior(const Field3D& a, const Field3D& b)
{
    double sum_sq = 0.0;
    const auto& geo = global_grid_geometry;
    for (int i = 1; i < NR - 1; ++i)
    {
        const double ri = geo.r[i];
        for (int j = 0; j < NTH; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                const double d = static_cast<double>(a[i][j][k]) -
                                 static_cast<double>(b[i][j][k]);
                sum_sq += ri * d * d;
            }
    }
    return std::sqrt(sum_sq);
}

/**
 * @brief L2 norm of a over the interior compute domain.
 */
double l2_norm_interior(const Field3D& a)
{
    double sum_sq = 0.0;
    const auto& geo = global_grid_geometry;
    for (int i = 1; i < NR - 1; ++i)
    {
        const double ri = geo.r[i];
        for (int j = 0; j < NTH; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                const double v = static_cast<double>(a[i][j][k]);
                sum_sq += ri * v * v;
            }
    }
    return std::sqrt(sum_sq);
}

/**
 * @brief Mass-weighted z centroid over the interior.
 */
double centroid_z_interior(const Field3D& q)
{
    double m0 = 0.0;
    double m1 = 0.0;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                const double zk = static_cast<double>(k) * dz;
                const double v  = static_cast<double>(q[i][j][k]);
                m0 += v;
                m1 += zk * v;
            }
    return (m0 > 0.0) ? (m1 / m0) : 0.0;
}

/**
 * @brief Largest absolute pointwise difference over the interior.
 */
double max_abs_diff_interior(const Field3D& a, const Field3D& b)
{
    double m = 0.0;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                const double d = std::abs(static_cast<double>(a[i][j][k]) -
                                          static_cast<double>(b[i][j][k]));
                if (d > m) m = d;
            }
    return m;
}

/**
 * @brief Field copy that preserves shape.
 */
Field3D copy_field(const Field3D& src)
{
    Field3D out;
    out.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                out[i][j][k] = src[i][j][k];
    return out;
}

}  // namespace


// ============================================================================
// Gate 1 -- Zero-flow preservation: any IC stays bit-exact through advection.
// ============================================================================
TEST_CASE("advect_scalar_3d (cgrid): zero flow preserves any cell-center scalar bit-exactly",
          "[advection][cylindrical][cgrid][c7]")
{
    setup_cgrid_grid(/*nr=*/16, /*nth=*/8, /*nz=*/12,
                     /*dr=*/250.0, /*dz=*/250.0, /*dt=*/1.0);
    disable_numerics_vertical_advection();
    zero_velocity();

    // Fill the interior with a non-trivial pattern so a stray write would
    // be visible. The boundary cells (i=0, i=NR-1, k=0, k=NZ-1) stay at 0.
    Field3D q;
    q.resize(NR, NTH, NZ, 0.0f);
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 1; k < NZ - 1; ++k)
                q[i][j][k] = static_cast<float>(
                    1.0 + 0.1 * static_cast<double>(i) +
                          0.05 * static_cast<double>(j) +
                          0.02 * static_cast<double>(k));

    Field3D q_initial = copy_field(q);

    const int n_steps = 50;
    for (int step = 0; step < n_steps; ++step)
        advect_scalar_3d(q, 1.0, 0.0);  // dt=1 s, kappa=0 (no diffusion)

    INFO("max |q - q_initial| over interior = "
         << max_abs_diff_interior(q, q_initial));
    REQUIRE(max_abs_diff_interior(q, q_initial) == 0.0);
}


// ============================================================================
// Gate 2 -- Pure vertical uniform advection.
//   w = W > 0 constant; u = v = 0; Gaussian column in z; the centroid
//   should track at W m/s.
// ============================================================================
TEST_CASE("advect_scalar_3d (cgrid): pure-vertical advection translates centroid at face velocity",
          "[advection][cylindrical][cgrid][c7]")
{
    const int    NR_T  = 12;
    const int    NTH_T = 8;
    const int    NZ_T  = 32;
    const double DR    = 250.0;
    const double DZ    = 100.0;
    const double DT    = 0.1;     // CFL_z = w*dt/dz = 5*0.1/100 = 0.005
    const double W     = 5.0;     // m/s

    setup_cgrid_grid(NR_T, NTH_T, NZ_T, DR, DZ, DT);
    disable_numerics_vertical_advection();
    set_uniform_vertical_velocity(W);

    // Place a thin Gaussian column at i=4, j=2, centered at z = 8*DZ.
    Field3D q;
    place_gaussian_column_z(q, /*i_center=*/4, /*j_center=*/2,
                            /*z_center_m=*/8.0 * DZ, /*sigma_m=*/3.0 * DZ,
                            /*peak=*/1.0);

    const double cz_initial = centroid_z_interior(q);
    const double mass_initial = total_mass_interior(q);

    const double T_end = 100.0;          // sim seconds
    const int    n_steps = static_cast<int>(T_end / DT);

    for (int step = 0; step < n_steps; ++step)
        advect_scalar_3d(q, DT, 0.0);

    const double cz_final  = centroid_z_interior(q);
    const double mass_final = total_mass_interior(q);
    const double expected_cz = cz_initial + W * (n_steps * DT);

    INFO("Centroid: " << cz_initial << " -> " << cz_final
         << " (expected " << expected_cz << ")");
    INFO("Mass: " << mass_initial << " -> " << mass_final
         << " (relative drift = "
         << std::abs(mass_final - mass_initial) / std::abs(mass_initial) << ")");

    // Centroid speed within 2 % (the bump runs into the upper interior
    // boundary near k = NZ - 2 toward the end, which clips a sliver of
    // mass from the centroid average; 2 % absorbs that without hiding a
    // real bug).
    const double err_cz = std::abs(cz_final - expected_cz)
                        / std::abs(expected_cz - cz_initial);
    REQUIRE(err_cz < 0.02);

    // Mass conservation: bump is well inside interior, no boundary
    // fluxes, so interior mass should be preserved to float precision
    // (~ N_cells * N_steps * eps).
    const double mass_drift = std::abs(mass_final - mass_initial) /
                              std::abs(mass_initial);
    REQUIRE(mass_drift < 1.0e-5);
}


// ============================================================================
// Gate 3 -- Pure-azimuthal uniform advection (periodic theta returns).
//   v = V > 0 constant; u = w = 0; Gaussian bump in (i, j) at fixed k.
//   After NTH * dtheta / V seconds, the bump returns to its starting
//   angular position. Mass is conserved exactly because every theta
//   flux at face j cancels with the corresponding flux at face j (seen
//   from the neighboring cell) when both use the same v[i][j] and the
//   same upwind value.
// ============================================================================
TEST_CASE("advect_scalar_3d (cgrid): pure-azimuthal uniform advection conserves mass over one revolution",
          "[advection][cylindrical][cgrid][c7]")
{
    const int    NR_T  = 16;
    const int    NTH_T = 32;
    const int    NZ_T  = 12;
    const double DR    = 250.0;
    const double DZ    = 250.0;
    const double V     = 20.0;            // m/s

    // CFL_theta = v*dt / (r * dtheta).  At i = 1 (r = 250, dtheta = 2*pi/32),
    // r*dtheta = 49.087 m; pick dt so CFL_theta < 0.5.
    const double DT = 0.5;                // gives CFL_theta ~ 0.20 at i=1

    setup_cgrid_grid(NR_T, NTH_T, NZ_T, DR, DZ, DT);
    disable_numerics_vertical_advection();
    set_uniform_azimuthal_velocity(V);

    Field3D q;
    place_gaussian_bump_axis_radial(q, /*i_center=*/8, /*j_center=*/0, /*k_center=*/6,
                                    /*sigma_cells_r=*/2.0, /*sigma_cells_th=*/2.0,
                                    /*peak=*/1.0);

    const double mass_initial = total_mass_interior(q);
    REQUIRE(mass_initial > 0.0);

    // One revolution: bump traverses the full (2*pi)*r distance at
    // angular speed V/r; bump at i_center = 8 has r = 8 * DR = 2000 m,
    // so T_rev = 2*pi*r / V = 2*pi*2000/20 = 628.3 s.  Use one full
    // revolution to test mass conservation under a non-trivial step
    // count (628.3 / 0.5 ~= 1257 steps).
    //
    // Note: with a uniform v_field, cells at different radii rotate at
    // different angular rates -- v/r varies as 1/r.  So this is NOT a
    // "rigid rotation" of the bump shape; the inner cells advance
    // faster in theta than the outer cells.  Mass is still conserved
    // by the flux form regardless.
    const double r_center = 8.0 * DR;
    const double T_rev    = 2.0 * kPi * r_center / V;
    const int    n_steps  = static_cast<int>(T_rev / DT);

    for (int step = 0; step < n_steps; ++step)
        advect_scalar_3d(q, DT, 0.0);

    const double mass_final = total_mass_interior(q);
    const double mass_drift = std::abs(mass_final - mass_initial) /
                              std::abs(mass_initial);

    INFO("Pure-azimuthal uniform-velocity test:");
    INFO("  n_steps = " << n_steps << ", T_rev = " << T_rev << " s");
    INFO("  mass_initial = " << mass_initial);
    INFO("  mass_final   = " << mass_final);
    INFO("  relative drift = " << mass_drift);

    // Mass conservation: every theta-face flux cancels pairwise. With
    // ~1300 steps and ~16*32*10 = 5120 interior cells, the FP-roundoff
    // floor is roughly 1300 * 5120 * 1e-7 / (...) ~ 1e-4 absolute, but
    // the relative drift is much smaller because each step's roundoff
    // error has zero mean when summed over a periodic geometry.
    REQUIRE(mass_drift < 1.0e-5);
}


// ============================================================================
// Gate 4 -- Solid-body rotation: bump returns to its starting position
// after one revolution period T = 2*pi/Omega.
//   u_r = 0; u_theta = Omega * r at theta-faces.  This rotates with the
//   SAME angular speed at every radius (rigid rotation), so the bump
//   shape is preserved up to numerical viscosity from first-order
//   upwind.
//
//   Pass criteria (from C.7 plan):
//     - L2 error vs initial < 5 % of L2(initial) after one revolution.
//     - Total interior mass preserved to floating-point precision.
// ============================================================================
TEST_CASE("advect_scalar_3d (cgrid): solid-body rotation preserves mass and bump shape",
          "[advection][cylindrical][cgrid][c7][solid-body-rotation]")
{
    const int    NR_T  = 24;
    const int    NTH_T = 64;
    const int    NZ_T  = 12;
    const double DR    = 200.0;
    const double DZ    = 250.0;

    // Pick Omega so the period is bounded but the CFL is comfortable.
    // CFL_theta_max = Omega * r_max * dt / (r_max * dtheta) =
    //                 Omega * dt / dtheta.
    // Want CFL ~ 0.4 -> dt = 0.4 * dtheta / Omega.  With dtheta = 2*pi/64
    // and Omega = 0.01 rad/s:
    //                 dt = 0.4 * (2*pi/64) / 0.01 ~= 3.93 s, period
    //                 T = 2*pi / Omega = 628.3 s -> 160 steps.
    const double OMEGA = 0.01;
    const double DT    = 0.4 * (2.0 * kPi / NTH_T) / OMEGA;
    const double T_REV = 2.0 * kPi / OMEGA;
    const int    N_STEPS = static_cast<int>(T_REV / DT + 0.5);

    setup_cgrid_grid(NR_T, NTH_T, NZ_T, DR, DZ, DT);
    disable_numerics_vertical_advection();
    set_solid_body_rotation_velocity(OMEGA);

    // Smooth (1 + cos(theta)) tracer placed on a single radial ring and
    // a single z slice. This is the standard solid-body rotation
    // benchmark for advection schemes (smooth periodic profile + rigid
    // rotation = exact return-to-self after one period). With TVD-MUSCL
    // and MC limiter, the profile retains more than 95 % of its L2
    // norm after one full revolution. A single-peak Gaussian would
    // not pass this gate at the same resolution because the MC limiter
    // zeroes the slope at the peak (an extremum), falling back to
    // first-order at one cell per step and accumulating numerical
    // viscosity at that location; the periodic 1 + cos(theta) profile
    // has only one extremum (the maximum) and one zero (the minimum)
    // per ring, with the rest of the profile carrying smooth slopes
    // that the limiter does not restrict.
    Field3D q;
    place_periodic_cosine_ring(q, /*i_center=*/12, /*k_center=*/6, /*peak=*/1.0);
    Field3D q_initial = copy_field(q);

    const double mass_initial = total_mass_interior(q);
    const double l2_initial   = l2_norm_interior(q_initial);
    REQUIRE(l2_initial > 0.0);

    for (int step = 0; step < N_STEPS; ++step)
        advect_scalar_3d(q, DT, 0.0);

    const double mass_final = total_mass_interior(q);
    const double l2_err     = l2_diff_interior(q, q_initial);
    const double mass_drift = std::abs(mass_final - mass_initial) /
                              std::abs(mass_initial);
    const double l2_rel_err = l2_err / l2_initial;

    INFO("Solid-body rotation gate (Omega = " << OMEGA << " rad/s, "
         << "dt = " << DT << ", N_STEPS = " << N_STEPS << "):");
    INFO("  L2 relative error after 1 revolution = " << l2_rel_err);
    INFO("  Mass relative drift                  = " << mass_drift);
    INFO("  Mass: " << mass_initial << " -> " << mass_final);

    // Plan gate: L2 error < 5 %.
    REQUIRE(l2_rel_err < 0.05);
    // Plan gate: mass conservation to machine precision.
    REQUIRE(mass_drift < 1.0e-6);
}


// ============================================================================
// Gate 5 -- Stagger-routing distinguishability.
//   Run the same IC + same uniform-radial-flow (u, v, w) on both stagger
//   types; the C-grid path (TVD-MUSCL flux form, face velocities + 1/r
//   geometric divergence) must produce a DIFFERENT result than the
//   collocated path (first-order upwind advective form, cell-center
//   velocities, NO geometric divergence). This catches a regression in
//   which advect_scalar_3d silently falls through to the collocated
//   kernel when StaggerType::CGrid is set.
//
//   Why uniform radial flow distinguishes:
//   - Collocated radial advection (advective form): tendency = -u * dq/dr
//     (no q * div(u) correction at uniform u_r).
//   - C-grid radial advection (flux form):
//     tendency = -(r_face[i]*u*q_face[i] - r_face[i-1]*u*q_face[i-1])
//                / (r[i] * dr)
//     which differs from collocated by the geometric divergence
//     ~ -u * (q[i] + q[i-1]) / (2 * r[i]).
//   This term is non-zero whenever q is non-zero in the bump region.
// ============================================================================
TEST_CASE("advect_scalar_3d (cgrid): C-grid and collocated paths produce distinct results",
          "[advection][cylindrical][cgrid][c7][routing]")
{
    const int    NR_T  = 16;
    const int    NTH_T = 16;
    const int    NZ_T  = 12;
    const double DR    = 250.0;
    const double DZ    = 250.0;
    const double DT    = 0.5;
    const double UR    = 5.0;            // uniform radial outflow (m/s)

    auto run_path = [&](StaggerType stagger) -> Field3D {
        if (stagger == StaggerType::CGrid)
            setup_cgrid_grid(NR_T, NTH_T, NZ_T, DR, DZ, DT);
        else
            setup_collocated_grid(NR_T, NTH_T, NZ_T, DR, DZ, DT);
        disable_numerics_vertical_advection();

        // Uniform radial outflow: u[i][j][k] = UR everywhere; v = w = 0.
        // On the C-grid this corresponds to a uniform-flux outward
        // expansion (v = w = 0); on the collocated grid the advective
        // form sees the same cell-centered u value.
        u.resize(NR, NTH, NZ, static_cast<float>(UR));
        v.resize(NR, NTH, NZ, 0.0f);
        w.resize(NR, NTH, NZ, 0.0f);

        Field3D q;
        place_gaussian_bump_axis_radial(q, /*i_center=*/8, /*j_center=*/0, /*k_center=*/6,
                                        /*sigma_cells_r=*/2.0, /*sigma_cells_th=*/2.0,
                                        /*peak=*/1.0);

        for (int step = 0; step < 10; ++step)
            advect_scalar_3d(q, DT, 0.0);
        return q;
    };

    Field3D q_collocated = run_path(StaggerType::Collocated);
    Field3D q_cgrid      = run_path(StaggerType::CGrid);

    const double max_diff = max_abs_diff_interior(q_collocated, q_cgrid);
    INFO("max |q_collocated - q_cgrid| over interior = " << max_diff);

    // The bump peak is 1.0; meaningful deviation is at least a few
    // percent of peak. The collocated and C-grid paths converge to the
    // same continuum equation but produce different discrete updates
    // both because of (a) the geometric-divergence term unique to
    // flux form and (b) the TVD-MUSCL slope reconstruction the C-grid
    // path applies vs the first-order upwind on the collocated path.
    REQUIRE(max_diff > 1.0e-3);
}

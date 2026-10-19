/**
 * @file test_cartesian_initial_conditions.cpp
 * @brief Verification gates for Phase A.4 — Cartesian initial conditions.
 *
 * Verification gate for Cartesian initial-condition ownership:
 *
 *   Print init summary: u, v are constant in (x, y) at every level
 *   (matching the hodograph profile). w = 0 everywhere. Trigger bubble shows
 *   up as a localized Δθ patch where the config says it should.
 *
 * The plan's gate is "print and visually verify". This file converts the
 * three properties into checked assertions so the gate is repeatable and
 * regression-protected:
 *
 *   1. After apply_cartesian_wind_initialization():
 *        u[i][j][k]       == hodograph_u(z[k])  for every (i, j, k)
 *        v[i][j][k] == hodograph_v(z[k])  for every (i, j, k)
 *        w[i][j][k]       == 0                  for every (i, j, k)
 *      "constant in (x, y) at every level" is enforced as max-minus-min over
 *      every (i, j) for fixed k being exactly zero.
 *
 *   2. After apply_cartesian_bubble_initialization():
 *        theta at the configured (i_c, j_c, k_c) is bumped by the analytic
 *        Gaussian peak (bubble_dtheta * exp(0) = bubble_dtheta).
 *        Cells far outside the bubble radius are unchanged.
 *        Δθ falls off according to the documented Gaussian.
 *
 *   3. The wind init for a Cartesian uniform wind hodograph (u_x = 12,
 *      u_y = -4 at all levels) reproduces those exact constants on every
 *      cell — the test that the cylindrical scheme cannot pass per Bug 7.
 *
 * Test driver pattern: each test sets up the global Field3D state via the
 * test_harness symbols, sets `global_coordinate_system = Cartesian`,
 * configures the relevant globals (`global_wind_profile`, the bubble
 * globals), and calls the helper directly. There is no full `initialize()`
 * call here — that would link in advection, microphysics, radiation, etc.
 * The helpers under test ARE the production code path: `initialize()`
 * dispatches to them, so verifying the helpers verifies what the runtime
 * does in Cartesian mode.
 */

#include "catch2/catch.hpp"
#include "core/field/field3d.hpp"
#include "core/orchestration/dynamics/initial_conditions.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "terrain/base/topography.hpp"
#include "terrain/conservative_transport.hpp"
#include "terrain/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{

/**
 * @brief Local hodograph used by the test's compute_wind_profile stub.
 *
 * The values are deliberately non-zero in both components and vary with
 * height so that:
 *   - we can verify the helper actually uses z (not a constant)
 *   - we can verify the (u_x, u_y) split is preserved and not swapped
 *   - we can detect any spurious cos/sin projection (the cylindrical bug)
 */
constexpr double kSurfaceUx = 5.0;     // m/s at z = 0
constexpr double kSurfaceUy = -2.0;    // m/s at z = 0
constexpr double kShearUx   = 1.0e-3;  // m/s per m of altitude
constexpr double kShearUy   = 5.0e-4;

double expected_ux_at_height(double z)
{
    return kSurfaceUx + kShearUx * z;
}

double expected_uy_at_height(double z)
{
    return kSurfaceUy + kShearUy * z;
}

/**
 * @brief Sets up a small Cartesian grid in the test_harness globals.
 *
 * 16x16x16 with 1 km horizontal cells and 0.5 km vertical cells, matching
 * the A.2 / A.3 test grids. The grid is large enough that a 2 km bubble
 * fits inside the interior with room to spare for the falloff sphere.
 */
void setup_cartesian_grid()
{
    NR  = 16;
    NTH = 16;
    NZ  = 16;
    dr  = 1000.0;
    dz  = 500.0;
    dt  = 0.1;
    dtheta = dr;  // Cartesian: square cells (matches update_grid_resolution())
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    global_coordinate_system);
}

/**
 * @brief Resizes (u, v, w) to (NR, NTH, NZ) with sentinel values
 *        so any cell the helper fails to write is visible as a NaN/garbage
 *        rather than a coincidental zero.
 *
 * NaN sentinels would crash any min/max post-condition check, so we use
 * a large but finite "should never appear" value. The test then asserts
 * the helper overwrote every cell.
 */
void resize_velocity_with_sentinel()
{
    constexpr float kSentinel = -9999.0f;
    u.resize(NR, NTH, NZ, kSentinel);
    v.resize(NR, NTH, NZ, kSentinel);
    w.resize(NR, NTH, NZ, kSentinel);
}

/**
 * @brief Resizes theta to (NR, NTH, NZ) with a known base value so the
 *        bubble Δθ can be measured as theta[i][j][k] - kBaseTheta.
 */
constexpr float kBaseTheta = 300.0f;
void resize_theta_with_base_state()
{
    theta.resize(NR, NTH, NZ, kBaseTheta);
}

// Installs the analytic linear profile used by the production helper tests.
void install_test_hodograph()
{
    global_wind_profile.u_sfc = expected_ux_at_height(0.0);
    global_wind_profile.v_sfc = expected_uy_at_height(0.0);
    global_wind_profile.u_1km = expected_ux_at_height(1000.0);
    global_wind_profile.v_1km = expected_uy_at_height(1000.0);
    global_wind_profile.u_6km = expected_ux_at_height(6000.0);
    global_wind_profile.v_6km = expected_uy_at_height(6000.0);
}

/**
 * @brief Returns (max - min) of `field` over (i, j) at fixed k.
 *
 * Used to verify "u is constant in (x, y) at every level" — the gate
 * should be exactly zero (the helper writes the same float value at every
 * (i, j) for fixed k).
 */
double horizontal_spread_at_level(const Field3D& field, int k)
{
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double v = static_cast<double>(field[i][j][k]);
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
    }
    return hi - lo;
}

/**
 * @brief Returns the largest absolute value of `field` over the entire
 *        domain. Used to verify w == 0 in the wind-init gate.
 */
double max_abs_full_domain(const Field3D& field)
{
    double max_val = 0.0;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double v = std::abs(static_cast<double>(field[i][j][k]));
                if (v > max_val) max_val = v;
            }
        }
    }
    return max_val;
}

}  // namespace

// ===========================================================================
// Test stub for compute_wind_profile.
//
// Defined at file scope (NOT in an anonymous namespace) so the linker can
// resolve the forward declaration in src/core/initial_conditions_cartesian.cpp
// against this definition. The Makefile target for this test deliberately
// does NOT include src/core/nimbus_sim.cpp (which would otherwise provide
// the production definition), so this stub IS the linked symbol.
//
// The stub interpolates the supplied profile, making configuration ownership
// independently observable without linking the full runtime.
// ===========================================================================
void compute_wind_profile(const WindProfile& profile, double z, double& u, double& v)
{
    if (z <= 1000.0)
    {
        const double t = z / 1000.0;
        u = profile.u_sfc + t * (profile.u_1km - profile.u_sfc);
        v = profile.v_sfc + t * (profile.v_1km - profile.v_sfc);
        return;
    }
    const double t = (z - 1000.0) / 5000.0;
    u = profile.u_1km + t * (profile.u_6km - profile.u_1km);
    v = profile.v_1km + t * (profile.v_6km - profile.v_1km);
}

// ===========================================================================
// Gate 1: wind init produces (u_x(z), u_y(z), 0) and is constant in (x, y).
// ===========================================================================
TEST_CASE("Cartesian IC: wind init stores u_x, u_y per level with no projection",
          "[dynamics][cartesian][a4]")
{
    setup_cartesian_grid();
    resize_velocity_with_sentinel();
    install_test_hodograph();

    apply_cartesian_wind_initialization();

    // Every cell must be overwritten — no sentinel survivors.
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                INFO("cell (" << i << "," << j << "," << k << ")");
                REQUIRE(u[i][j][k]       != Approx(-9999.0f));
                REQUIRE(v[i][j][k] != Approx(-9999.0f));
                REQUIRE(w[i][j][k]       == 0.0f);
            }
        }
    }

    // Per-level horizontal uniformity: max - min over (i, j) at fixed k
    // must be exactly zero. The helper writes the same float per (i, j),
    // so any spread > 0 would mean a θ-dependent term leaked in.
    for (int k = 0; k < NZ; ++k)
    {
        INFO("level k=" << k);
        REQUIRE(horizontal_spread_at_level(u,       k) == 0.0);
        REQUIRE(horizontal_spread_at_level(v, k) == 0.0);
        REQUIRE(horizontal_spread_at_level(w,       k) == 0.0);
    }

    // Per-level value matches the expected (u_x(z), u_y(z)) from the
    // analytic hodograph. Tolerance is 1 ULP-ish (the helper does a
    // double->float cast at the end of the stencil chain).
    for (int k = 0; k < NZ; ++k)
    {
        const double z = global_grid_geometry.z[static_cast<std::size_t>(k)];
        const double expected_ux = expected_ux_at_height(z);
        const double expected_uy = expected_uy_at_height(z);
        INFO("level k=" << k << " (z=" << z << " m)");
        REQUIRE(static_cast<double>(u[0][0][k])
                == Approx(expected_ux).epsilon(1e-6));
        REQUIRE(static_cast<double>(v[0][0][k])
                == Approx(expected_uy).epsilon(1e-6));
    }

    // w must be identically zero everywhere. This is the gate that proves
    // the IC does not seed any spurious vertical motion.
    REQUIRE(max_abs_full_domain(w) == 0.0);
}

// ===========================================================================
// Gate 2: bubble init places a localized Δθ patch at the configured center.
// ===========================================================================
TEST_CASE("Cartesian IC: bubble init places localized dtheta at config center",
          "[dynamics][cartesian][a4]")
{
    setup_cartesian_grid();
    resize_theta_with_base_state();

    // Place a 2 km bubble at the geometric center of the test grid:
    //   x_c = 8 km (i = 8 of 16 with dr = 1 km)
    //   y_c = 8 km (j = 8 of 16)
    //   z_c = 4 km (k = 8 of 16 with dz = 0.5 km)
    constexpr int i_c = 8;
    constexpr int j_c = 8;
    constexpr int k_c = 8;
    const double bubble_radius_m = 2000.0;
    const double bubble_dtheta_k = 3.0;

    global_bubble_center_x_m = (static_cast<double>(i_c) + 0.5) * dr;  // 8500 m
    global_bubble_center_y_m = (static_cast<double>(j_c) + 0.5) * dr;  // 8500 m
    global_bubble_center_z_m = (static_cast<double>(k_c) + 0.5) * dz;  // 4250 m
    global_bubble_radius_m   = bubble_radius_m;
    global_bubble_dtheta_k   = bubble_dtheta_k;

    apply_cartesian_bubble_initialization();

    // The cell at the configured center should be bumped by exactly the
    // configured bubble_dtheta_k (Gaussian factor exp(0) = 1).
    const double dtheta_at_center =
        static_cast<double>(theta[i_c][j_c][k_c]) - static_cast<double>(kBaseTheta);
    INFO("dtheta at bubble center = " << dtheta_at_center << " K");
    REQUIRE(dtheta_at_center == Approx(bubble_dtheta_k).epsilon(1e-5));

    // Cells very far from the center (more than the radius) must be
    // unchanged. Pick (0, 0, 0) — it is √(8² + 8² + 4²) = ~12.5 km from
    // center, well outside the 2 km bubble.
    REQUIRE(static_cast<double>(theta[0][0][0]) == Approx(kBaseTheta).epsilon(1e-6));

    // Cells just inside the radius should be bumped by some positive Δθ
    // strictly less than the peak. Pick a cell on the x-axis exactly
    // 1 km away from the center: dist = 1 km, factor = exp(-(1/(2/3))²)
    // = exp(-2.25) ≈ 0.1054, so Δθ ≈ 0.316 K.
    const double dist_m = dr;  // 1000 m
    const double scaled = dist_m / (bubble_radius_m / 3.0);
    const double expected_factor = std::exp(-scaled * scaled);
    const double expected_dtheta_offset = bubble_dtheta_k * expected_factor;
    const double dtheta_offset_axis =
        static_cast<double>(theta[i_c + 1][j_c][k_c]) - static_cast<double>(kBaseTheta);
    INFO("dtheta at 1 km off-center = " << dtheta_offset_axis << " K (expected "
         << expected_dtheta_offset << " K)");
    REQUIRE(dtheta_offset_axis == Approx(expected_dtheta_offset).epsilon(1e-4));

    // Symmetry: at +x and -x of the same offset the bump must be identical.
    REQUIRE(static_cast<double>(theta[i_c + 1][j_c][k_c])
            == Approx(static_cast<double>(theta[i_c - 1][j_c][k_c])).epsilon(1e-6));
    REQUIRE(static_cast<double>(theta[i_c][j_c + 1][k_c])
            == Approx(static_cast<double>(theta[i_c][j_c - 1][k_c])).epsilon(1e-6));

    // Symmetry across the three axes: the cell offset by 1 km along x must
    // match the cell offset by 1 km along y (true Cartesian distance, not
    // a 2D ring or a radial-axis quirk).
    REQUIRE(static_cast<double>(theta[i_c + 1][j_c][k_c])
            == Approx(static_cast<double>(theta[i_c][j_c + 1][k_c])).epsilon(1e-6));
}

TEST_CASE("Cartesian IC: bubble cloud-water seed is localized and deterministic",
          "[dynamics][cartesian][initialization][microphysics]")
{
    setup_cartesian_grid();
    resize_theta_with_base_state();
    qc.resize(NR, NTH, NZ, 0.0f);

    constexpr int i_c = 8;
    constexpr int j_c = 8;
    constexpr int k_c = 8;
    RuntimeConfigurationState config;
    config.bubble_center_x_m = (static_cast<double>(i_c) + 0.5) * dr;
    config.bubble_center_y_m = (static_cast<double>(j_c) + 0.5) * dr;
    config.bubble_center_z_m = (static_cast<double>(k_c) + 0.5) * dz;
    config.bubble_radius_m = 2000.0;
    config.bubble_dtheta_k = 0.0;
    config.bubble_cloud_water_kgkg = 0.002;

    auto state = global_simulation_state();
    state.configuration = &config;
    apply_cartesian_bubble_initialization(state);

    REQUIRE(qc[i_c][j_c][k_c] == Approx(0.002f));
    const double expected_factor = std::exp(-2.25);
    REQUIRE(qc[i_c + 1][j_c][k_c]
            == Approx(0.002 * expected_factor).epsilon(1.0e-6));
    REQUIRE(qc[0][0][0] == 0.0f);
    REQUIRE(theta[i_c][j_c][k_c] == kBaseTheta);
}

// ===========================================================================
// Gate 3: wind init ignores `dtheta`.
//
// This is the IC-side analog of the cylindrical Bug 7 trap. If the helper
// were to read `dtheta` and project the hodograph onto (cos θ, sin θ) — as
// the cylindrical inline branch in equations.cpp does — then setting a
// non-zero `dtheta` would inject a per-j variation into u and v even
// though the underlying wind is uniform. The Cartesian helper must NOT
// consume `dtheta` at all, so the per-level spread must remain exactly
// zero regardless of what `dtheta` is set to.
//
// We deliberately load `dtheta = 2π / NTH` (the cylindrical default that
// runtime_config.cpp computes via update_grid_resolution()) to maximize
// the chance of catching a regression that accidentally re-introduces a
// projection step.
// ===========================================================================
TEST_CASE("Cartesian IC: wind init is invariant under non-zero dtheta",
          "[dynamics][cartesian][bug7][a4]")
{
    setup_cartesian_grid();
    resize_velocity_with_sentinel();
    install_test_hodograph();

    // Force a cylindrical-style dtheta. If the helper has any cos/sin
    // projection it would now produce j-dependent values.
    dtheta = 2.0 * 3.14159265358979323846 / static_cast<double>(NTH);

    apply_cartesian_wind_initialization();

    // Per-level horizontal spread must still be exactly zero.
    for (int k = 0; k < NZ; ++k)
    {
        INFO("level k=" << k << " (dtheta=" << dtheta << ")");
        REQUIRE(horizontal_spread_at_level(u,       k) == 0.0);
        REQUIRE(horizontal_spread_at_level(v, k) == 0.0);
    }

    // And the per-level value must still match the analytic hodograph
    // at every cell — no projection means no contamination from dtheta.
    for (int k = 0; k < NZ; ++k)
    {
        const double z = global_grid_geometry.z[static_cast<std::size_t>(k)];
        REQUIRE(static_cast<double>(u[NR / 2][NTH / 2][k])
                == Approx(expected_ux_at_height(z)).epsilon(1e-6));
        REQUIRE(static_cast<double>(v[NR / 2][NTH / 2][k])
                == Approx(expected_uy_at_height(z)).epsilon(1e-6));
    }

    REQUIRE(max_abs_full_domain(w) == 0.0);

    // Restore the dtheta the rest of the test file expects (zero) so any
    // following test cases see a clean slate.
    dtheta = 0.0;
}

TEST_CASE("Cartesian IC: wind init can operate on explicit local state",
          "[dynamics][cartesian][a4][state]")
{
    setup_cartesian_grid();
    resize_velocity_with_sentinel();
    install_test_hodograph();
    u[0][0][0] = 123.0f;

    int local_nr = 3;
    int local_nth = 4;
    int local_nz = 4;
    double local_dr = 250.0;
    double local_dz = 100.0;
    double local_dt = 0.1;
    double local_dtheta = 250.0;
    GridGeometry local_geometry;
    local_geometry.initialize(local_nr, local_nth, local_nz,
                              local_dr, local_dz, local_dtheta,
                              CoordinateSystem::Cartesian);

    Field3D local_rho(local_nr, local_nth, local_nz, 1.0f);
    Field3D local_p(local_nr, local_nth, local_nz, 100000.0f);
    Field3D local_u(local_nr, local_nth, local_nz, -9999.0f);
    Field3D local_w(local_nr, local_nth, local_nz, -9999.0f);
    Field3D local_v(local_nr, local_nth, local_nz, -9999.0f);
    Field3D local_tracer(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_theta(local_nr, local_nth, local_nz, 300.0f);
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

    std::vector<double> local_rho0(local_nz, 1.0);
    std::vector<double> local_p0(local_nz, 100000.0);
    std::vector<double> local_qv0(local_nz, 0.0);
    std::vector<double> local_theta0(local_nz, 300.0);
    std::vector<double> local_u0(local_nz, 0.0);
    std::vector<double> local_v0(local_nz, 0.0);
    double local_coriolis = 0.0;
    double local_time = 0.0;

    SimulationStateView local_state{
        {local_nr, local_nth, local_nz, local_dr, local_dz, local_dt, local_dtheta,
         local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs,
         local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis,
        local_time};

    RuntimeConfigurationState local_config;
    local_config.wind_profile = {20.0, -10.0, 22.0, -8.0, 32.0, 2.0};
    local_state.configuration = &local_config;

    apply_cartesian_wind_initialization(local_state);

    for (int k = 0; k < local_nz; ++k)
    {
        const double z = local_geometry.z[static_cast<std::size_t>(k)];
        double expected_u = 0.0;
        double expected_v = 0.0;
        compute_wind_profile(local_config.wind_profile, z,
                             expected_u, expected_v);
        REQUIRE(static_cast<double>(local_u[0][0][k])
                == Approx(expected_u).epsilon(1e-6));
        REQUIRE(static_cast<double>(local_v[0][0][k])
                == Approx(expected_v).epsilon(1e-6));
        REQUIRE(local_w[0][0][k] == 0.0f);
    }

    local_config.bubble_center_x_m = local_geometry.r[1];
    local_config.bubble_center_y_m = local_geometry.theta[1];
    local_config.bubble_center_z_m = local_geometry.z[1];
    local_config.bubble_radius_m = 100.0;
    local_config.bubble_dtheta_k = 4.0;
    apply_cartesian_bubble_initialization(local_state);
    REQUIRE(local_theta[1][1][1] == Approx(304.0f));
    REQUIRE(local_theta[0][1][1] == Approx(300.0f));
    REQUIRE(local_theta[1][0][1] == Approx(300.0f));
    REQUIRE(local_theta[1][1][0] == Approx(300.0f));
    REQUIRE(local_theta[local_nr - 1][1][1] == Approx(300.0f));
    REQUIRE(local_theta[1][local_nth - 1][1] == Approx(300.0f));
    REQUIRE(local_theta[1][1][local_nz - 1] == Approx(300.0f));
    REQUIRE(u[0][0][0] == Approx(123.0f));
}

TEST_CASE("Cartesian terrain IC samples wind and bubble at physical height",
          "[dynamics][cartesian][terrain][initialization][TE-004]")
{
    setup_cartesian_grid();
    resize_velocity_with_sentinel();
    resize_theta_with_base_state();
    p.resize(NR, NTH, NZ, 0.0f);
    rho.resize(NR, NTH, NZ, 0.0f);
    qv.resize(NR, NTH, NZ, 0.0f);
    install_test_hodograph();

    TerrainRuntimeState terrain;
    terrain.config.scheme_id = "bell";
    terrain.config.coord_id = "smoothed";
    terrain.config.ztop = static_cast<double>(NZ) * dz;
    terrain.config.smoothing_decay = 0.7;
    topography::initialize_topography(terrain.topography, NR, NTH);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            terrain.topography.h[i][j] = 300.0
                + 80.0 * std::sin(0.35 * i) * std::cos(0.27 * j);
    topography::initialize_metrics(terrain.metrics, NR, NTH, NZ);
    topography::build_conservative_face_metrics(
        terrain.config, terrain.topography,
        topography::build_zeta_interfaces(NZ, terrain.config.ztop),
        dr, dtheta, terrain.metrics);

    auto state = global_simulation_state();
    state.terrain_runtime = &terrain;
    std::vector<double> reference_pressure(static_cast<std::size_t>(NZ));
    std::vector<double> reference_theta(static_cast<std::size_t>(NZ), 300.0);
    std::vector<double> reference_qv(static_cast<std::size_t>(NZ));
    for (int level = 0; level < NZ; ++level)
    {
        const double height = (static_cast<double>(level) + 0.5) * dz;
        reference_pressure[static_cast<std::size_t>(level)] =
            100000.0 * std::exp(-height / 8200.0);
        reference_qv[static_cast<std::size_t>(level)] =
            0.014 * std::exp(-height / 2500.0);
    }
    apply_cartesian_terrain_thermodynamic_initialization(
        state, reference_pressure, reference_theta, reference_qv);
    apply_cartesian_wind_initialization(state);

    constexpr int i = 6;
    constexpr int j = 7;
    constexpr int k = 5;
    const double u_height = 0.5 * (terrain.metrics.z(i, j, k)
                                  + terrain.metrics.z(i + 1, j, k));
    const double v_height = 0.5 * (terrain.metrics.z(i, j, k)
                                  + terrain.metrics.z(i, j + 1, k));
    const double scalar_height = terrain.metrics.z(i, j, k);
    REQUIRE(static_cast<double>(p[i][j][k])
            == terrain_transport::reference_profile_value_fp32(
                reference_pressure, scalar_height, dz));
    REQUIRE(static_cast<double>(theta[i][j][k]) == 300.0);
    REQUIRE(static_cast<double>(rho[i][j][k])
            == terrain_transport::reference_density_fp32(
                reference_pressure, reference_theta, scalar_height, dz));
    REQUIRE(static_cast<double>(qv[i][j][k])
            == terrain_transport::reference_profile_value_fp32(
                reference_qv, scalar_height, dz));
    REQUIRE(static_cast<double>(u[i][j][k])
            == Approx(expected_ux_at_height(u_height)).margin(2.0e-6));
    REQUIRE(static_cast<double>(v[i][j][k])
            == Approx(expected_uy_at_height(v_height)).margin(2.0e-6));
    REQUIRE(w[i][j][k] == 0.0f);

    global_bubble_center_x_m = global_grid_geometry.r[i];
    global_bubble_center_y_m = global_grid_geometry.theta[j];
    global_bubble_center_z_m = terrain.metrics.z(i, j, k);
    global_bubble_radius_m = 100.0;
    global_bubble_dtheta_k = 4.0;
    apply_cartesian_bubble_initialization(state);
    REQUIRE(theta[i][j][k] == Approx(304.0f));
    REQUIRE(theta[i][j][k - 1] == Approx(300.0f));
    REQUIRE(theta[i][j][k + 1] == Approx(300.0f));
}

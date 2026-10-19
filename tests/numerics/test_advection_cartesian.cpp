/**
 * @file test_advection_cartesian.cpp
 * @brief Verification gates for Phase A.5 (Cartesian scalar advection).
 *
 * Regression coverage for the retained Cartesian advection operators.
 *
 * The Cartesian backend uses a parallel directional split implemented in
 * `src/advection/advection_cartesian.cpp` and dispatched from
 * `src/advection/advection.cpp::advect_scalar_3d` based on
 * `global_coordinate_system`. These tests exercise the dispatch
 * end-to-end (the test calls `advect_scalar_3d` directly, not the
 * helpers) so the branch in `advect_scalar_3d` is part of what is tested.
 *
 * The two tests below are the verification gates the plan asks for:
 *
 *   1. 1D x-direction Gaussian bump advected at 30 m/s for 100 sim
 *      seconds. Verify the centroid moves at 30 m/s ± 1 % and the total
 *      mass is conserved to 1 part in 10⁴.
 *
 *   2. 2D diagonal Gaussian bump advected at (u_x, u_y) = (20, 20) m/s
 *      for 100 sim seconds. Verify the (x, y) centroid arrives at the
 *      expected location and the bump's σ_x and σ_y stay close (no
 *      axis-aligned smearing).
 *
 * Both tests use a uniform velocity field, so first-order upwind has no
 * phase error per step (the centroid update is `+ u·dt` exactly). The
 * peak amplitude *will* decay (~5–10 %) from numerical viscosity — that
 * is expected first-order-upwind behavior, not a bug, and is intentionally
 * not asserted on. Phase B will upgrade both backends to TVD MUSCL on the
 * horizontal axes.
 *
 * Test linking architecture: this file links against
 *   - tests/test_harness.cpp (provides global Field3D state and config)
 *   - src/advection/advection.cpp (contains advect_scalar_3d + dispatch)
 *   - src/advection/advection_cartesian.cpp (the new helpers)
 *   - src/numerics/advection/ (TVD and WENO5 vertical schemes)
 *   - BACKEND_COMMON_SRCS (compute backend stubs for the GPU dispatchers
 *     that advection.cpp references — they all return false in this
 *     unit-test build, which is what we want)
 *
 * The test forces `advection_scheme = nullptr` so the legacy CPU vertical
 * kernel runs. With `w` set to zero, the vertical kernel is a no-op and
 * the test isolates the (x, y) directional split.
 */

#include "catch2/catch.hpp"

#include "compute/compute_backend.hpp"
#include "compute/compute_backend_factory.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/field/field3d.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/device_acoustic_energy_attribution.hpp"
#include "microphysics/schemes/milbrandt/milbrandt.hpp"
#include "numerics/advection/advection.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "terrain/terrain_base.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Grid setup helpers
// ---------------------------------------------------------------------------

/**
 * @brief Configures the global grid + coord-system globals for a Cartesian
 *        advection test. Cartesian `dtheta` stores the physical y spacing.
 */
void setup_cartesian_grid(int nr, int nth, int nz,
                          double dx_m, double dy_m, double dz_m, double dt_s)
{
    NR = nr;
    NTH = nth;
    NZ = nz;
    dr = dx_m;
    dz = dz_m;
    dt = dt_s;
    dtheta = dy_m;
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_grid_geometry.initialize(
        nr, nth, nz, dx_m, dz_m, dy_m,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::Stationary;
    global_domain_frame = tmv::domain::HorizontalDomainFrame::stationary();
}

TerrainMetrics3D make_sloping_transport_metrics(int nr, int nth, int nz,
                                                double dx_m, double dy_m,
                                                double dzeta_m)
{
    constexpr double jacobian = 1.2;
    constexpr double slope_x = 0.08;
    constexpr double slope_y = -0.05;
    TerrainMetrics3D metrics;
    metrics.dx = dx_m;
    metrics.dy = dy_m;
    metrics.dzeta = dzeta_m;
    metrics.z.resize(nr, nth, nz, 0.0);
    metrics.J.resize(nr, nth, nz, jacobian);
    metrics.mx.resize(nr, nth, nz, -slope_x / jacobian);
    metrics.my.resize(nr, nth, nz, -slope_y / jacobian);
    metrics.zeta.resize(nr, nth, nz, 0.0);
    metrics.z_face.resize(nr, nth, nz + 1, 0.0);
    metrics.Jx_face.resize(nr + 1, nth, nz, jacobian);
    metrics.Jy_face.resize(nr, nth + 1, nz, jacobian);
    metrics.Gx_face.resize(nr, nth, nz + 1, -slope_x);
    metrics.Gy_face.resize(nr, nth, nz + 1, -slope_y);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double ground = slope_x * i * dx_m + slope_y * j * dy_m;
            for (int k = 0; k <= nz; ++k)
                metrics.z_face(i, j, k) =
                    ground + jacobian * dzeta_m * k;
            for (int k = 0; k < nz; ++k)
            {
                metrics.z(i, j, k) =
                    ground + jacobian * dzeta_m * (k + 0.5);
                metrics.zeta(i, j, k) = dzeta_m * (k + 0.5);
            }
        }
    return metrics;
}

void prepare_sloping_transport_context(
    const TerrainMetrics3D& metrics, double dt_s,
    CartesianCGridTransportContext& context)
{
    context.begin(NR, NTH, NZ, dt_s, &metrics);
    context.capture_density_begin(rho);
    for (int i = 1; i < NR - 2; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 1; k < NZ - 1; ++k)
                context.mass_flux_x[i][j][k] = static_cast<float>(
                    0.22 + 0.01 * i - 0.004 * j + 0.002 * k);
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 2; ++j)
            for (int k = 1; k < NZ - 1; ++k)
                context.mass_flux_y[i][j][k] = static_cast<float>(
                    -0.12 + 0.003 * i + 0.006 * j - 0.001 * k);
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 1; k < NZ - 2; ++k)
                context.mass_flux_z[i][j][k] = static_cast<float>(
                    0.04 - 0.001 * i + 0.002 * j + 0.003 * k);
    context.record_external_accumulation(dt_s, 1);

    Field3D density_end = rho;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                const double mass_divergence =
                    ((static_cast<double>(context.mass_flux_x[i][j][k]) -
                      context.mass_flux_x[i - 1][j][k]) / metrics.dx +
                     (static_cast<double>(context.mass_flux_y[i][j][k]) -
                      context.mass_flux_y[i][j - 1][k]) / metrics.dy +
                     (static_cast<double>(context.mass_flux_z[i][j][k]) -
                      context.mass_flux_z[i][j][k - 1]) / metrics.dzeta) /
                    metrics.J(i, j, k);
                density_end[i][j][k] = static_cast<float>(
                    static_cast<double>(rho[i][j][k]) -
                    dt_s * mass_divergence);
            }
    rho = std::move(density_end);
    context.complete();
    REQUIRE(context.validate(NR, NTH, NZ, dt_s));
}

/**
 * @brief Resizes the global velocity fields and seeds them with a uniform
 *        (u_x, u_y, u_z): `u` carries u_x, `v` carries u_y, `w` carries u_z.
 */
void set_uniform_velocity(double ux, double uy, double uz)
{
    u.resize(NR, NTH, NZ, static_cast<float>(ux));
    v.resize(NR, NTH, NZ, static_cast<float>(uy));
    w.resize(NR, NTH, NZ, static_cast<float>(uz));
}

/**
 * @brief Resets the global advection_scheme pointer so that the legacy
 *        CPU vertical kernel runs (not the numerics TVD path).
 *
 * With `w = 0`, the legacy kernel computes `dq/dz = 0` for every cell and
 * does nothing. This isolates the (x, y) directional split for the test
 * — the only kernels exercised are the three Cartesian helpers plus the
 * dispatch in `advect_scalar_3d`.
 */
void disable_numerics_vertical_advection()
{
    advection_scheme.reset();
}

// ---------------------------------------------------------------------------
// Bump initial conditions
// ---------------------------------------------------------------------------

/**
 * @brief Places a 1D Gaussian bump along the x axis (constant in y) into
 *        a Field3D, restricted to the interior z range.
 *
 * Cell coordinates: x_i = i * dr, y_j = j * dtheta, z_k = k * dz.
 *
 * The bump uses the **standard** Gaussian form
 *   q(x) = peak * exp(−(x − x_c)² / (2 σ²))
 * so that the mass-weighted standard deviation of `q` equals `sigma_m`
 * exactly (in the continuum limit, ignoring discretization error).
 *
 * The bump is placed only on the interior z slices (k = 1..NZ − 2). The
 * k = 0 and k = NZ − 1 slices stay at zero. This matters because the
 * Cartesian x and y advection kernels only update interior k cells (the
 * z = 0 / z = top boundaries are pass-through), so leaving the bump on
 * the boundary slices would mean part of the bump's mass would never
 * advect, polluting the centroid measurement.
 */
void place_gaussian_bump_1d_x(Field3D& q, double x_center_m, double sigma_m, double peak)
{
    q.resize(NR, NTH, NZ, 0.0f);
    const double sigma_safe = std::max(sigma_m, 1.0e-12);
    const double inv_two_sigma_sq = 1.0 / (2.0 * sigma_safe * sigma_safe);
    const int k_lo = 1;
    const int k_hi = NZ - 1;  // exclusive
    for (int i = 0; i < NR; ++i)
    {
        const double x = static_cast<double>(i) * dr;
        const double xd = x - x_center_m;
        const double value = peak * std::exp(-(xd * xd) * inv_two_sigma_sq);
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = k_lo; k < k_hi; ++k)
            {
                q[i][j][k] = static_cast<float>(value);
            }
        }
    }
}

/**
 * @brief Places a 2D Gaussian bump in the (x, y) plane into a Field3D,
 *        restricted to the interior z range.
 *
 *   q(x, y) = peak * exp(−((x − x_c)² + (y − y_c)²) / (2 σ²))
 *
 * Same standard-form rationale as `place_gaussian_bump_1d_x`. Same
 * interior-k-only restriction.
 */
void place_gaussian_bump_2d(Field3D& q, double x_center_m, double y_center_m, double sigma_m, double peak)
{
    q.resize(NR, NTH, NZ, 0.0f);
    const double sigma_safe = std::max(sigma_m, 1.0e-12);
    const double inv_two_sigma_sq = 1.0 / (2.0 * sigma_safe * sigma_safe);
    const int k_lo = 1;
    const int k_hi = NZ - 1;  // exclusive
    for (int i = 0; i < NR; ++i)
    {
        const double x = static_cast<double>(i) * dr;
        const double xd = x - x_center_m;
        for (int j = 0; j < NTH; ++j)
        {
            const double y = static_cast<double>(j) * dtheta;
            const double yd = y - y_center_m;
            const double dist_sq = xd * xd + yd * yd;
            const double value = peak * std::exp(-dist_sq * inv_two_sigma_sq);
            for (int k = k_lo; k < k_hi; ++k)
            {
                q[i][j][k] = static_cast<float>(value);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostics — mass-weighted moments at a single z level
// ---------------------------------------------------------------------------

/**
 * @brief Sums q over the entire domain.
 *
 * Cell volume is uniform on a Cartesian grid in Phase A, so the unit-cell
 * volume factor cancels in mass conservation comparisons. This function
 * returns the raw sum.
 */
double total_mass(const Field3D& q)
{
    double sum = 0.0;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                sum += static_cast<double>(q[i][j][k]);
            }
        }
    }
    return sum;
}

/**
 * @brief Mass-weighted x-centroid: Σ x_i · q[i][j][k] / Σ q[i][j][k].
 */
double centroid_x(const Field3D& q)
{
    double m0 = 0.0;
    double m1 = 0.0;
    for (int i = 0; i < NR; ++i)
    {
        const double x = static_cast<double>(i) * dr;
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double v = static_cast<double>(q[i][j][k]);
                m0 += v;
                m1 += x * v;
            }
        }
    }
    return (m0 > 0.0) ? (m1 / m0) : 0.0;
}

/**
 * @brief Mass-weighted y-centroid: Σ y_j · q[i][j][k] / Σ q[i][j][k].
 */
double centroid_y(const Field3D& q)
{
    double m0 = 0.0;
    double m1 = 0.0;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double y = static_cast<double>(j) * dtheta;
            for (int k = 0; k < NZ; ++k)
            {
                const double v = static_cast<double>(q[i][j][k]);
                m0 += v;
                m1 += y * v;
            }
        }
    }
    return (m0 > 0.0) ? (m1 / m0) : 0.0;
}

/**
 * @brief Mass-weighted standard deviation in x:
 *        sqrt(Σ (x − cx)² · q / Σ q).
 */
double sigma_x(const Field3D& q, double cx)
{
    double m0 = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < NR; ++i)
    {
        const double x = static_cast<double>(i) * dr;
        const double dx = x - cx;
        const double dx2 = dx * dx;
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double v = static_cast<double>(q[i][j][k]);
                m0 += v;
                m2 += dx2 * v;
            }
        }
    }
    return (m0 > 0.0) ? std::sqrt(m2 / m0) : 0.0;
}

/**
 * @brief Mass-weighted standard deviation in y.
 */
double sigma_y(const Field3D& q, double cy)
{
    double m0 = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double y = static_cast<double>(j) * dtheta;
            const double dy = y - cy;
            const double dy2 = dy * dy;
            for (int k = 0; k < NZ; ++k)
            {
                const double v = static_cast<double>(q[i][j][k]);
                m0 += v;
                m2 += dy2 * v;
            }
        }
    }
    return (m0 > 0.0) ? std::sqrt(m2 / m0) : 0.0;
}

}  // namespace

// ---------------------------------------------------------------------------
// Gate 1: 1D x-direction advection at 30 m/s for 100 sim seconds
// ---------------------------------------------------------------------------

TEST_CASE("Cartesian advection: 1D x Gaussian bump translates at 30 m/s with mass conserved",
          "[numerics][advection][cartesian][manufactured][positive-direction]")
{
    // Grid: 200 cells × 100 m = 20 km domain. NTH = 5, NZ = 5 — small but
    // not degenerate, so the kernel's interior loops actually execute.
    const int nr = 200;
    const int nth = 5;
    const int nz = 5;
    const double dx_m = 100.0;
    const double dz_m = 100.0;

    // CFL = u·dt/dx = 30 · 1.5 / 100 = 0.45 — comfortable safety margin.
    const double dt_s = 1.5;
    const double total_sim_seconds = 100.5;  // 67 steps × 1.5 s
    const int n_steps = 67;

    setup_cartesian_grid(nr, nth, nz, dx_m, dx_m, dz_m, dt_s);

    // Velocity: 30 m/s in x, 0 in y and z. With v_y = w = 0, the y and z
    // sub-steps in the Strang split are no-ops, isolating the x kernel.
    const double u_x_ms = 30.0;
    set_uniform_velocity(u_x_ms, 0.0, 0.0);
    disable_numerics_vertical_advection();

    // Initial bump at x = 5 km, σ = 1 km, peak = 1.0. With dx = 100 m the
    // bump is resolved by ~10 cells — well-resolved for first-order upwind.
    const double sigma_m = 1000.0;
    const double initial_centroid_x = 5000.0;
    const double bump_peak = 1.0;
    Field3D q;
    place_gaussian_bump_1d_x(q, initial_centroid_x, sigma_m, bump_peak);

    const double initial_mass = total_mass(q);
    const double initial_cx = centroid_x(q);
    REQUIRE(initial_mass > 0.0);
    REQUIRE(initial_cx == Approx(initial_centroid_x).margin(1.0));  // sub-meter symmetry

    // Run the dispatch end-to-end. kappa = 0 disables diffusion so the
    // measurement is pure advection. The Cartesian branch in
    // advect_scalar_3d is the path under test.
    for (int step = 0; step < n_steps; ++step)
    {
        advect_scalar_3d(q, dt_s, /*kappa=*/0.0);
    }

    const double final_mass = total_mass(q);
    const double final_cx = centroid_x(q);
    const double expected_final_cx = initial_centroid_x + u_x_ms * total_sim_seconds;

    // ── Centroid drift gate ──
    // For first-order upwind with constant velocity the per-step centroid
    // update is exactly + u·dt (proven by direct sum manipulation), so the
    // tolerance can be very tight. We use ±1 % to give margin for FP and
    // for the y/z half-steps that touch the field even though they make
    // no physical change.
    const double centroid_error = std::abs(final_cx - expected_final_cx);
    const double centroid_relative = centroid_error / expected_final_cx;
    INFO("initial cx = " << initial_cx << ", expected = " << expected_final_cx
                         << ", final = " << final_cx
                         << ", abs error = " << centroid_error
                         << ", rel error = " << centroid_relative);
    REQUIRE(centroid_relative < 0.01);

    // ── Mass conservation gate ──
    // The bump remains far from the i = 0 and i = NR − 1 boundaries
    // (centered at 8 km on a 20 km domain), so no mass should leak out.
    // The Cartesian helpers conserve mass exactly in floating-point, so
    // the only drift is from boundary cells the kernel doesn't update —
    // those are bit-copied from the previous step, so they don't change
    // either. The conservation tolerance is set to 1e-4 per the plan.
    const double mass_error = std::abs(final_mass - initial_mass);
    const double mass_relative = mass_error / initial_mass;
    INFO("initial mass = " << initial_mass << ", final mass = " << final_mass
                           << ", rel error = " << mass_relative);
    REQUIRE(mass_relative < 1.0e-4);

    // ── Sanity: the field actually moved ──
    // If something is wrong with the dispatch and `advect_scalar_3d`
    // never invoked the Cartesian kernels, the bump would still sit at
    // 5 km. Confirm it has moved by at least 2.5 km (≈ 80 % of the
    // expected 3 km translation).
    REQUIRE((final_cx - initial_cx) > 2500.0);
}

// ---------------------------------------------------------------------------
// Gate 2: 2D diagonal advection at (20, 20) m/s for 100 sim seconds
// ---------------------------------------------------------------------------

TEST_CASE("Cartesian advection: 2D diagonal Gaussian bump translates without axis smearing",
          "[numerics][advection][cartesian][manufactured][positive-direction]")
{
    // Grid: 200 × 200 × 5 = 200 000 cells × ~10 fields × 4 B ≈ 8 MB.
    const int nr = 200;
    const int nth = 200;
    const int nz = 5;
    const double dx_m = 100.0;
    const double dz_m = 100.0;

    // CFL_x = CFL_y = 20 · 2.0 / 100 = 0.4 — well below the upwind 1.0 limit.
    const double dt_s = 2.0;
    const double total_sim_seconds = 100.0;  // 50 steps × 2.0 s
    const int n_steps = 50;

    setup_cartesian_grid(nr, nth, nz, dx_m, dx_m, dz_m, dt_s);

    // Velocity: (20, 20, 0) m/s — equal x and y components for the
    // diagonal symmetry test.
    const double u_x_ms = 20.0;
    const double u_y_ms = 20.0;
    set_uniform_velocity(u_x_ms, u_y_ms, 0.0);
    disable_numerics_vertical_advection();

    // Bump at (5, 5) km, σ = 1 km, peak = 1.0. The 2D Gaussian is
    // axisymmetric in (x, y) by construction, so σ_x and σ_y should
    // remain close after the directional split.
    const double sigma_m = 1000.0;
    const double initial_x_m = 5000.0;
    const double initial_y_m = 5000.0;
    const double bump_peak = 1.0;
    Field3D q;
    place_gaussian_bump_2d(q, initial_x_m, initial_y_m, sigma_m, bump_peak);

    const double initial_mass = total_mass(q);
    const double initial_cx = centroid_x(q);
    const double initial_cy = centroid_y(q);
    const double initial_sx = sigma_x(q, initial_cx);
    const double initial_sy = sigma_y(q, initial_cy);
    REQUIRE(initial_mass > 0.0);
    REQUIRE(initial_cx == Approx(initial_x_m).margin(1.0));
    REQUIRE(initial_cy == Approx(initial_y_m).margin(1.0));
    // The mass-weighted σ of a true Gaussian is exactly σ — confirm the
    // grid is fine enough that the discretization reproduces this within
    // 5 m (5 % of dx).
    REQUIRE(initial_sx == Approx(sigma_m).epsilon(0.005));
    REQUIRE(initial_sy == Approx(sigma_m).epsilon(0.005));
    // x/y symmetry should be exact at t = 0 by construction.
    REQUIRE(initial_sx == Approx(initial_sy).epsilon(1.0e-3));

    for (int step = 0; step < n_steps; ++step)
    {
        advect_scalar_3d(q, dt_s, /*kappa=*/0.0);
    }

    const double final_mass = total_mass(q);
    const double final_cx = centroid_x(q);
    const double final_cy = centroid_y(q);
    const double final_sx = sigma_x(q, final_cx);
    const double final_sy = sigma_y(q, final_cy);
    const double expected_final_x = initial_x_m + u_x_ms * total_sim_seconds;  // 7 km
    const double expected_final_y = initial_y_m + u_y_ms * total_sim_seconds;  // 7 km

    // ── Centroid arrival gates ──
    const double cx_relative = std::abs(final_cx - expected_final_x) / expected_final_x;
    const double cy_relative = std::abs(final_cy - expected_final_y) / expected_final_y;
    INFO("expected (cx, cy) = (" << expected_final_x << ", " << expected_final_y << ")"
                                 << ", final = (" << final_cx << ", " << final_cy << ")"
                                 << ", rel = (" << cx_relative << ", " << cy_relative << ")");
    REQUIRE(cx_relative < 0.01);
    REQUIRE(cy_relative < 0.01);

    // ── Mass conservation gate ──
    const double mass_relative = std::abs(final_mass - initial_mass) / initial_mass;
    INFO("initial mass = " << initial_mass << ", final mass = " << final_mass
                           << ", rel error = " << mass_relative);
    REQUIRE(mass_relative < 1.0e-4);

    // ── No-axis-smearing gate ──
    // First-order upwind has numerical diffusion in each axis. With the
    // (x, y) Strang split and equal CFL_x = CFL_y, the spreading should
    // be (nearly) the same in both directions, so σ_x / σ_y stays close
    // to 1. A directional bias would show up here as a ratio departing
    // from 1 — for example, if the y kernel were accidentally still using
    // the cylindrical (1/r) factor, σ_y would be different from σ_x at
    // every i ≠ 1.
    const double sigma_ratio = final_sx / final_sy;
    INFO("σ_x = " << final_sx << ", σ_y = " << final_sy << ", ratio = " << sigma_ratio);
    REQUIRE(sigma_ratio == Approx(1.0).epsilon(0.10));

    // Sanity: the bump should have grown from numerical viscosity but
    // not by a huge amount. σ should increase modestly (≤ 25 %).
    REQUIRE(final_sx > initial_sx);
    REQUIRE(final_sx < 1.25 * initial_sx);
    REQUIRE(final_sy > initial_sy);
    REQUIRE(final_sy < 1.25 * initial_sy);

    // Sanity: the field actually moved diagonally.
    REQUIRE((final_cx - initial_cx) > 1700.0);  // ≈ 85 % of 2 km expected
    REQUIRE((final_cy - initial_cy) > 1700.0);
}

// ---------------------------------------------------------------------------
// Gate 3: rectangular Cartesian cells use dtheta as physical y spacing
// ---------------------------------------------------------------------------

TEST_CASE("Cartesian advection: rectangular-grid y transport uses configured dy",
          "[numerics][advection][cartesian][manufactured][positive-direction][rectangular]")
{
    // Cartesian dtheta is dy in metres. A dx-for-dy regression moves this
    // bump 2.5 times too far in physical y on the deliberately rectangular
    // grid below.
    const int nr = 24;
    const int nth = 56;
    const int nz = 5;
    const double dx_m = 100.0;
    const double dy_m = 250.0;
    const double dz_m = 100.0;
    const double dt_s = 2.0;
    const int n_steps = 50;
    const double velocity_y = 20.0;
    const double y0 = 3500.0;

    setup_cartesian_grid(nr, nth, nz, dx_m, dy_m, dz_m, dt_s);
    set_uniform_velocity(0.0, velocity_y, 0.0);
    disable_numerics_vertical_advection();

    Field3D q;
    place_gaussian_bump_2d(q, 1200.0, y0, 750.0, 1.0);
    const double initial_y = centroid_y(q);
    for (int step = 0; step < n_steps; ++step)
        advect_scalar_3d(q, dt_s, /*kappa=*/0.0);

    const double final_y = centroid_y(q);
    const double expected_displacement = velocity_y * dt_s * n_steps;
    INFO("y centroid " << initial_y << " -> " << final_y
         << "; expected displacement " << expected_displacement);
    REQUIRE(final_y - initial_y == Approx(expected_displacement).epsilon(0.03));
}

TEST_CASE("transport-chained point diffusion is unavailable",
          "[numerics][advection][cartesian][diffusion][sci050]")
{
    setup_cartesian_grid(/*nr=*/5, /*nth=*/5, /*nz=*/5,
                         /*dx=*/100.0, /*dy=*/200.0, /*dz=*/400.0,
                         /*dt=*/1.0);

    Field3D src, dst;
    src.resize(NR, NTH, NZ, 0.0f);
    src[2][2][2] = 1.0f;
    REQUIRE_THROWS_AS(
        apply_diffusion_kernel_cartesian(src, dst, 1.0, 1.0),
        std::invalid_argument);
    REQUIRE_NOTHROW(
        apply_diffusion_kernel_cartesian(src, dst, 1.0, 0.0));
    for (std::size_t index = 0; index < src.size(); ++index)
        REQUIRE(dst.data()[index] == src.data()[index]);
    REQUIRE_THROWS_AS(
        advect_scalar_3d(src, 1.0, 0.01), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Gate 4: Cartesian C-grid conservative transport preserves a constant field
//         under a strongly divergent flow (free-stream / no spurious source)
// ---------------------------------------------------------------------------

TEST_CASE("Cartesian C-grid conservative advection: constant field preserved under divergent flow",
          "[numerics][advection][cartesian][cgrid]")
{
    // Regression gate for the flux-vs-advective bug. The old directional split
    // advected theta in flux form vertically (−∂(w·θ)/∂z) but advective form
    // horizontally, so a CONSTANT θ grew a spurious −θ·div(u) source wherever
    // the flow diverged — the 2Δx cloud-base dipole that ran θ to the
    // [200, 700] clamp. The conservative rho-weighted flux form must keep a
    // constant field exactly constant for ANY velocity field (free-stream
    // preservation), because rho*phi and rho_check carry the same mass fluxes.
    const int nr = 24, nth = 24, nz = 12;
    const double dx_m = 400.0, dz_m = 250.0, dt_s = 1.5;

    setup_cartesian_grid(nr, nth, nz, dx_m, dx_m, dz_m, dt_s);
    global_stagger_type = StaggerType::CGrid;  // route through the conservative C-grid kernel

    // Strongly divergent, height-sheared velocity field (div(u) ≠ 0 everywhere),
    // with a dw/dz > 0 updraft-like profile — exactly the configuration that
    // produced the spurious heating/cooling dipole under the old scheme.
    u.resize(nr, nth, nz, 0.0f);
    v.resize(nr, nth, nz, 0.0f);
    w.resize(nr, nth, nz, 0.0f);
    rho.resize(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                u[i][j][k]   = static_cast<float>(10.0 * std::sin(2.0 * kPi * i / nr));
                v[i][j][k]   = static_cast<float>( 8.0 * std::cos(2.0 * kPi * j / nth));
                w[i][j][k]   = static_cast<float>( 6.0 * (static_cast<double>(k) / nz));
                rho[i][j][k] = static_cast<float>(1.1 * std::exp(-(static_cast<double>(k) * dz_m) / 8000.0));
            }

    const float kConst = 300.0f;
    Field3D theta_field;
    theta_field.resize(nr, nth, nz, kConst);

    // Under the old flux-vertical scheme this constant field diverges by tens of
    // K within ~10 steps where dw/dz is large. The conservative scheme must not.
    for (int step = 0; step < 60; ++step)
    {
        advect_scalar_3d(theta_field, dt_s, /*kappa=*/0.0);
    }

    double max_abs_dev = 0.0;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                max_abs_dev = std::max(max_abs_dev,
                    std::abs(static_cast<double>(theta_field[i][j][k]) - static_cast<double>(kConst)));

    INFO("max |theta - 300| after 60 divergent-flow steps = " << max_abs_dev);
    // Free-stream is exact in real arithmetic; fp32 round-off over 60 steps is
    // well under 0.01 K. The old scheme would be off by tens of K here.
    REQUIRE(max_abs_dev < 0.01);

    // Restore the default stagger so later tests are unaffected.
    global_stagger_type = StaggerType::Collocated;
}

// ---------------------------------------------------------------------------
// Gate 5: Cartesian C-grid conservative transport actually advects (translation
//         at the flow speed + mass conservation) — proves the kernel is not a
//         no-op, complementing the free-stream gate above.
// ---------------------------------------------------------------------------

TEST_CASE("Cartesian C-grid conservative advection: bump translates at flow speed, mass conserved",
          "[numerics][advection][cartesian][cgrid]")
{
    // With uniform rho and uniform velocity, div(u) = 0 so rho_check = rho and
    // the conservative kernel reduces to standard MUSCL upwind advection. A
    // resolved Gaussian bump must translate at the flow speed and conserve its
    // (rho-weighted) mass. The 2D bump is centred away from the y-boundaries so
    // the kernel's interior-j update doesn't strand any appreciable mass.
    const int nr = 200, nth = 60, nz = 8;
    const double dx_m = 100.0, dz_m = 100.0, dt_s = 1.5;

    setup_cartesian_grid(nr, nth, nz, dx_m, dx_m, dz_m, dt_s);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(nr, nth, nz, 1.0f);            // uniform density -> rho_check == rho
    set_uniform_velocity(30.0, 0.0, 0.0);     // 30 m/s in x; CFL = 30*1.5/100 = 0.45

    const double sigma_m = 800.0;             // 8-cell bump, well resolved
    const double x0 = 5000.0, y0 = 3000.0;    // y0 = mid-domain (6 km), ~3.75 sigma from edges
    Field3D q;
    place_gaussian_bump_2d(q, x0, y0, sigma_m, 1.0);

    const double m0 = total_mass(q);
    const double cx0 = centroid_x(q);
    REQUIRE(m0 > 0.0);
    REQUIRE(cx0 == Approx(x0).margin(2.0));

    const int n_steps = 67;                   // 100.5 s
    for (int step = 0; step < n_steps; ++step)
        advect_scalar_3d(q, dt_s, /*kappa=*/0.0);

    const double m1 = total_mass(q);
    const double cx1 = centroid_x(q);
    const double expected_cx = x0 + 30.0 * (n_steps * dt_s);  // 8015 m

    INFO("centroid " << cx0 << " -> " << cx1 << " (expected " << expected_cx
         << "); mass " << m0 << " -> " << m1);
    REQUIRE((cx1 - cx0) > 2500.0);                                  // it moved
    REQUIRE(std::abs(cx1 - expected_cx) / expected_cx < 0.02);      // at ~the flow speed
    REQUIRE(std::abs(m1 - m0) / m0 < 1.0e-3);                       // mass conserved

    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("Cartesian C-grid conservative advection uses configured dy",
          "[numerics][advection][cartesian][cgrid][rectangular-grid]")
{
    const int nr = 32, nth = 64, nz = 8;
    const double dx_m = 100.0, dy_m = 250.0, dz_m = 100.0, dt_s = 2.0;
    const double velocity_y = 25.0;
    const int n_steps = 60;

    setup_cartesian_grid(nr, nth, nz, dx_m, dy_m, dz_m, dt_s);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(nr, nth, nz, 1.0f);
    set_uniform_velocity(0.0, velocity_y, 0.0);

    Field3D q;
    place_gaussian_bump_2d(q, /*x=*/1600.0, /*y=*/4000.0,
                           /*sigma=*/750.0, /*peak=*/1.0);
    const double mass_initial = total_mass(q);
    const double y_initial = centroid_y(q);

    for (int step = 0; step < n_steps; ++step)
        advect_scalar_3d(q, dt_s, /*kappa=*/0.0);

    const double y_final = centroid_y(q);
    const double expected_displacement = velocity_y * dt_s * n_steps;
    const double mass_final = total_mass(q);
    INFO("y centroid " << y_initial << " -> " << y_final
         << "; expected displacement " << expected_displacement);
    REQUIRE(y_final - y_initial == Approx(expected_displacement).epsilon(0.03));
    REQUIRE(std::abs(mass_final - mass_initial) / mass_initial < 1.0e-3);

    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("Cartesian C-grid transports a Gaussian in the negative diagonal direction",
          "[numerics][advection][cartesian][cgrid][manufactured][negative-direction]")
{
    const int nr = 112, nth = 112, nz = 8;
    const double spacing_m = 100.0, dt_s = 2.0;
    const double velocity_x = -20.0, velocity_y = -15.0;
    const int steps = 50;
    setup_cartesian_grid(
        nr, nth, nz, spacing_m, spacing_m, spacing_m, dt_s);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(nr, nth, nz, 1.0f);
    set_uniform_velocity(velocity_x, velocity_y, 0.0);

    Field3D q;
    place_gaussian_bump_2d(
        q, /*x=*/7500.0, /*y=*/7000.0, /*sigma=*/700.0, /*peak=*/1.0);
    const double mass_initial = total_mass(q);
    const double x_initial = centroid_x(q);
    const double y_initial = centroid_y(q);
    for (int step = 0; step < steps; ++step)
        advect_scalar_3d(q, dt_s, /*kappa=*/0.0);

    const double x_final = centroid_x(q);
    const double y_final = centroid_y(q);
    const double mass_final = total_mass(q);
    const double expected_dx = velocity_x * dt_s * steps;
    const double expected_dy = velocity_y * dt_s * steps;
    INFO("negative diagonal displacement observed=(" << x_final - x_initial
         << ", " << y_final - y_initial << ") expected=(" << expected_dx
         << ", " << expected_dy << ")");
    REQUIRE(x_final - x_initial == Approx(expected_dx).epsilon(0.03));
    REQUIRE(y_final - y_initial == Approx(expected_dy).epsilon(0.03));
    REQUIRE(std::abs(mass_final - mass_initial) / mass_initial < 1.0e-3);

    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("Cartesian C-grid production-order transport conserves retained-density rho*q",
          "[numerics][advection][cartesian][cgrid][conservation][SCI-025]")
{
    // Exercise the production order: continuity commits rho first, then this
    // public scalar-advection entry point updates q while retaining that rho.
    // Closed-domain conservation applies across the combined pair, not to the
    // intermediate post-continuity inventory by itself.
    const int nr = 18, nth = 8, nz = 4;
    const double dx_m = 100.0, dy_m = 120.0, dz_m = 100.0, dt_s = 1.0;

    setup_cartesian_grid(nr, nth, nz, dx_m, dy_m, dz_m, dt_s);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(nr, nth, nz, 1.0f);
    u.resize(nr, nth, nz, 0.0f);
    v.resize(nr, nth, nz, 0.0f);
    w.resize(nr, nth, nz, 0.0f);

    // u[i] is the i+1/2 face velocity.  The physical scalar cells are
    // i=1..nr-2, so u[0] and u[nr-2] are their wall-normal boundary faces.
    // This sinusoid is exactly zero on both faces but has substantial local
    // divergence in between.  q is phase-aligned with that divergence so the
    // SCI-025 recovery error cannot hide behind cancellation.
    const int face_span = nr - 2;
    for (int i = 0; i <= nr - 2; ++i)
    {
        const double phase = 2.0 * kPi * static_cast<double>(i) /
                             static_cast<double>(face_span);
        const float face_u = static_cast<float>(25.0 * std::sin(phase));
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                u[i][j][k] = face_u;
    }

    Field3D q(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
    {
        const double center_phase =
            2.0 * kPi * (static_cast<double>(i) - 0.5) /
            static_cast<double>(face_span);
        const float value = static_cast<float>(
            0.01 + 0.004 * std::cos(center_phase));
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                q[i][j][k] = value;
    }

    const auto retained_inventory = [&](const Field3D& density,
                                        const Field3D& scalar) {
        long double sum = 0.0L;
        for (int i = 1; i < nr - 1; ++i)
            for (int j = 1; j < nth - 1; ++j)
                for (int k = 0; k < nz; ++k)
                    sum += static_cast<long double>(density[i][j][k]) *
                           static_cast<long double>(scalar[i][j][k]);
        return sum;
    };

    // Exercise both moisture-like positive-definite transport and the
    // non-limited theta path.  A constant scalar additionally catches a
    // denominator-only repair that closes a nonuniform inventory by adding a
    // spurious q*div(rho*u) source.
    Field3D theta_like(nr, nth, nz, 300.0f);
    for (int i = 0; i < nr; ++i)
    {
        const double center_phase =
            2.0 * kPi * (static_cast<double>(i) - 0.5) /
            static_cast<double>(face_span);
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                theta_like[i][j][k] = static_cast<float>(
                    300.0 + 2.5 * std::sin(center_phase)
                    + 0.25 * static_cast<double>(k));
    }
    constexpr float constant_q = 0.012f;
    Field3D uniform_q(nr, nth, nz, constant_q);

    const auto relative_inventory_error = [](long double before,
                                             long double after) {
        return static_cast<double>(
            std::abs(after - before) /
            std::max(std::abs(before), 1.0e-30L));
    };

    constexpr int n_steps = 6;
    const int acoustic_counts[3] = {1, 3, 5};
    double max_q_error = 0.0;
    double max_theta_error = 0.0;
    double max_uniform_inventory_error = 0.0;
    double max_uniform_value_error = 0.0;
    double max_rho_change = 0.0;
    double minimum_density = std::numeric_limits<double>::infinity();

    for (int step = 0; step < n_steps; ++step)
    {
        const Field3D rho_before_dynamics = rho;
        const long double q_before =
            retained_inventory(rho_before_dynamics, q);
        const long double theta_before =
            retained_inventory(rho_before_dynamics, theta_like);
        const long double uniform_before =
            retained_inventory(rho_before_dynamics, uniform_q);

        const int acoustic_substeps = acoustic_counts[step % 3];
        const double dt_small = dt_s / acoustic_substeps;
        CartesianCGridTransportContext transport_context;
        transport_context.begin(nr, nth, nz, dt_s);

        // This is the low-dependency equivalent of the production acoustic
        // loop: sample each pre-continuity state into the shared flux history,
        // then commit that substep's density before the next sample.  Scalar
        // transport runs only after every acoustic density update is retained.
        for (int substep = 0; substep < acoustic_substeps; ++substep)
        {
            transport_context.accumulate(rho, u, v, w, dt_small);
            Field3D rho_next = rho;
            for (int i = 1; i < nr - 1; ++i)
                for (int j = 1; j < nth - 1; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        const auto mass_flux_x = [&](int face_i) {
                            return static_cast<float>(
                                0.5 * (static_cast<double>(
                                           rho[face_i][j][k]) +
                                       static_cast<double>(
                                           rho[face_i + 1][j][k])) *
                                static_cast<double>(u[face_i][j][k]));
                        };
                        const double mass_div =
                            (static_cast<double>(mass_flux_x(i)) -
                             static_cast<double>(mass_flux_x(i - 1))) /
                            dx_m;
                        rho_next[i][j][k] = static_cast<float>(
                            static_cast<double>(rho[i][j][k]) -
                            dt_small * mass_div);
                    }
            rho = std::move(rho_next);
        }
        transport_context.complete();
        REQUIRE(transport_context.validate(nr, nth, nz, dt_s));
        REQUIRE(transport_context.substep_count() == acoustic_substeps);

        const Field3D rho_after_dynamics = rho;
        advect_scalar_3d(q, dt_s, /*kappa=*/0.0,
                         /*positive_definite=*/true, &transport_context);
        advect_scalar_3d(theta_like, dt_s, /*kappa=*/0.0,
                         /*positive_definite=*/false, &transport_context);
        advect_scalar_3d(uniform_q, dt_s, /*kappa=*/0.0,
                         /*positive_definite=*/true, &transport_context);

        max_q_error = std::max(
            max_q_error,
            relative_inventory_error(q_before,
                                     retained_inventory(rho, q)));
        max_theta_error = std::max(
            max_theta_error,
            relative_inventory_error(theta_before,
                                     retained_inventory(rho, theta_like)));
        max_uniform_inventory_error = std::max(
            max_uniform_inventory_error,
            relative_inventory_error(
                uniform_before, retained_inventory(rho, uniform_q)));

        for (int i = 1; i < nr - 1; ++i)
            for (int j = 1; j < nth - 1; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    max_uniform_value_error = std::max(
                        max_uniform_value_error,
                        std::abs(static_cast<double>(uniform_q[i][j][k]) -
                                 static_cast<double>(constant_q)));
                    minimum_density = std::min(
                        minimum_density,
                        static_cast<double>(rho[i][j][k]));
                }
        for (std::size_t n = 0; n < rho.size(); ++n)
            max_rho_change = std::max(
                max_rho_change,
                std::abs(static_cast<double>(rho.data()[n]) -
                         static_cast<double>(rho_after_dynamics.data()[n])));
    }

    INFO("production-order multi-step retained-rho errors: q="
         << max_q_error << " theta=" << max_theta_error
         << " uniform_inventory=" << max_uniform_inventory_error
         << " uniform_value=" << max_uniform_value_error
         << " rho_mutation=" << max_rho_change
         << " minimum_density=" << minimum_density);

    // The scalar stage consumes the continuity history but never changes the
    // already committed density.
    global_stagger_type = StaggerType::Collocated;
    REQUIRE(minimum_density > static_cast<double>(density_min_kgm3));
    REQUIRE(max_rho_change == 0.0);
    REQUIRE(max_uniform_value_error < 1.0e-7);
    REQUIRE(max_uniform_inventory_error < 1.0e-7);
    REQUIRE(max_theta_error < 1.0e-7);
    REQUIRE(max_q_error < 1.0e-7);
}

TEST_CASE("Cartesian C-grid transport context validates a complete weighted flux history",
          "[numerics][advection][cartesian][cgrid][SCI-025][context]")
{
    setup_cartesian_grid(/*nr=*/4, /*nth=*/3, /*nz=*/2,
                         /*dx=*/100.0, /*dy=*/120.0, /*dz=*/80.0,
                         /*dt=*/2.0);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 1.0f);
    u.resize(NR, NTH, NZ, 2.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);

    CartesianCGridTransportContext context;
    context.begin(NR, NTH, NZ, /*dt_large=*/2.0);
    context.accumulate(rho, u, v, w, /*dt_small=*/0.5);
    REQUIRE(context.has_density_begin());
    REQUIRE(context.density_begin[1][1][0] == 1.0f);

    // A second acoustic state contributes 3/4 of the average. Its x mass
    // flux is rho_face*u = 2*4 = 8, versus 2 in the first state.
    std::fill(rho.data(), rho.data() + rho.size(), 2.0f);
    std::fill(u.data(), u.data() + u.size(), 4.0f);
    context.accumulate(rho, u, v, w, /*dt_small=*/1.5);
    // Later acoustic states contribute flux but must not replace the density
    // paired with the scalar entering the large step.
    REQUIRE(context.density_begin[1][1][0] == 1.0f);
    context.complete();

    std::string reason;
    REQUIRE(context.validate(NR, NTH, NZ, 2.0, &reason));
    REQUIRE(reason.empty());
    REQUIRE(context.substep_count() == 2);
    REQUIRE(context.accumulated_dt() == Approx(2.0));
    REQUIRE(context.mass_flux_x[1][1][0] == Approx(6.5f));
    REQUIRE(context.mass_flux_y[1][1][0] == 0.0f);
    REQUIRE(context.mass_flux_z[1][1][0] == 0.0f);
    REQUIRE_FALSE(context.validate(NR, NTH, NZ, 1.0, &reason));
    REQUIRE(reason.find("timestep") != std::string::npos);

    CartesianCGridTransportContext incomplete;
    incomplete.begin(NR, NTH, NZ, 2.0);
    incomplete.complete();
    REQUIRE_FALSE(incomplete.validate(NR, NTH, NZ, 2.0, &reason));
    REQUIRE_THROWS(incomplete.require_valid(NR, NTH, NZ, 2.0));

    CartesianCGridTransportContext invalid_external;
    invalid_external.begin(NR, NTH, NZ, 2.0);
    invalid_external.mass_flux_x[1][1][0] =
        std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS(invalid_external.record_external_accumulation(2.0, 2));
    invalid_external.complete();
    REQUIRE_FALSE(invalid_external.validate(NR, NTH, NZ, 2.0, &reason));
    REQUIRE(reason.find("non-finite") != std::string::npos);
}

TEST_CASE("Cartesian retained scalar fluxes are invariant under a constant translating frame",
          "[numerics][advection][cartesian][cgrid][domain-motion][SCI-195]")
{
    setup_cartesian_grid(/*nr=*/8, /*nth=*/7, /*nz=*/5,
                         /*dx=*/100.0, /*dy=*/120.0, /*dz=*/80.0,
                         /*dt=*/0.5);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 0.0f);
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                rho[i][j][k] = static_cast<float>(
                    0.95 + 0.012 * i - 0.005 * j + 0.003 * k);
                u[i][j][k] = static_cast<float>(
                    1.25 + 0.08 * i - 0.03 * j + 0.01 * k);
                v[i][j][k] = static_cast<float>(
                    -0.75 + 0.02 * i + 0.06 * j - 0.015 * k);
                w[i][j][k] = static_cast<float>(
                    0.04 * i - 0.025 * j + 0.03 * k);
            }

    CartesianCGridTransportContext stationary;
    stationary.begin(NR, NTH, NZ, dt);
    stationary.accumulate(rho, u, v, w, dt);
    stationary.complete();
    REQUIRE(stationary.validate(NR, NTH, NZ, dt));

    constexpr double frame_u = 6.5;
    constexpr double frame_v = -3.25;
    Field3D u_translated = u;
    Field3D v_translated = v;
    for (std::size_t index = 0; index < u.size(); ++index)
    {
        u_translated.data()[index] = static_cast<float>(
            static_cast<double>(u.data()[index]) + frame_u);
        v_translated.data()[index] = static_cast<float>(
            static_cast<double>(v.data()[index]) + frame_v);
    }

    CartesianCGridTransportContext translated;
    translated.begin(NR, NTH, NZ, dt, nullptr, frame_u, frame_v);
    translated.accumulate(rho, u_translated, v_translated, w, dt);
    translated.complete();
    REQUIRE(translated.validate(NR, NTH, NZ, dt));
    REQUIRE(translated.frame_u_ms() == frame_u);
    REQUIRE(translated.frame_v_ms() == frame_v);
    for (std::size_t index = 0; index < rho.size(); ++index)
    {
        REQUIRE(translated.density_begin.data()[index]
                == stationary.density_begin.data()[index]);
        REQUIRE(translated.mass_flux_x.data()[index]
                == Approx(stationary.mass_flux_x.data()[index]).margin(1.0e-6));
        REQUIRE(translated.mass_flux_y.data()[index]
                == Approx(stationary.mass_flux_y.data()[index]).margin(1.0e-6));
        REQUIRE(translated.mass_flux_z.data()[index]
                == stationary.mass_flux_z.data()[index]);
    }
}

TEST_CASE(
    "Cartesian transport accepts only bounded FP32 device timestep currency",
    "[numerics][advection][cartesian][cgrid][device-time][SCI-025]")
{
    setup_cartesian_grid(/*nr=*/4, /*nth=*/3, /*nz=*/2,
                         /*dx=*/100.0, /*dy=*/120.0, /*dz=*/80.0,
                         /*dt=*/2.0);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 1.0f);
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);

    constexpr int substeps = 6;
    const double device_dt =
        static_cast<double>(static_cast<float>(2.0 / substeps));

    CartesianCGridTransportContext separate_device;
    separate_device.begin(NR, NTH, NZ, 2.0);
    for (int n = 0; n < substeps; ++n)
    {
        separate_device.accumulate(
            rho, u, v, w, device_dt, nullptr, nullptr,
            tmv::AcousticMassFluxPrecision::NativeFloat);
    }
    separate_device.complete();
    REQUIRE(separate_device.accumulated_dt()
            == Approx(device_dt * substeps));
    REQUIRE(separate_device.validate(NR, NTH, NZ, 2.0));

    CartesianCGridTransportContext batched_device;
    batched_device.begin(NR, NTH, NZ, 2.0);
    batched_device.capture_density_begin(rho);
    batched_device.record_external_carrier_kinetic_transport(12.0, substeps);
    batched_device.record_external_accumulation(
        device_dt * substeps, substeps,
        /*fp32_time_currency=*/true);
    batched_device.complete();
    REQUIRE(batched_device.validate(NR, NTH, NZ, 2.0));
    REQUIRE(batched_device.carrier_kinetic_transport_complete());
    REQUIRE(batched_device.outward_carrier_kinetic_transport_j() == 12.0);

    CartesianCGridTransportContext missing_device_substep;
    missing_device_substep.begin(NR, NTH, NZ, 2.0);
    for (int n = 0; n < substeps - 1; ++n)
    {
        missing_device_substep.accumulate(
            rho, u, v, w, device_dt, nullptr, nullptr,
            tmv::AcousticMassFluxPrecision::NativeFloat);
    }
    missing_device_substep.complete();
    std::string reason;
    REQUIRE_FALSE(
        missing_device_substep.validate(NR, NTH, NZ, 2.0, &reason));
    REQUIRE(reason.find("do not cover") != std::string::npos);
}

TEST_CASE(
    "Cartesian retained context can mirror device FP32 acoustic mass flux",
    "[numerics][advection][cartesian][cgrid][device-energy][SCI-158]")
{
    constexpr int nr = 5;
    constexpr int nth = 5;
    constexpr int nz = 3;
    constexpr float dx = 10.0f;
    constexpr float dy = 20.0f;
    constexpr float dz = 5.0f;
    constexpr float dt_s = 0.25f;
    const float frame_u = GENERATE(0.0f, 5.25f);
    const float frame_v = frame_u == 0.0f ? 0.0f : -2.125f;
    setup_cartesian_grid(nr, nth, nz, dx, dy, dz, dt_s);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(nr, nth, nz, 0.0f);
    u.resize(nr, nth, nz, 0.0f);
    v.resize(nr, nth, nz, 0.0f);
    w.resize(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                rho[i][j][k] = static_cast<float>(
                    0.91 + 0.017 * i - 0.009 * j + 0.003 * k);
                u[i][j][k] = static_cast<float>(
                    frame_u - 1.3 + 0.11 * i + 0.07 * j - 0.02 * k);
                v[i][j][k] = static_cast<float>(
                    frame_v + 0.8 - 0.04 * i + 0.09 * j + 0.01 * k);
                w[i][j][k] = static_cast<float>(
                    -0.2 + 0.03 * i - 0.02 * j + 0.08 * k);
            }
    // The device FP32 helper mirrors the already boundary-closed acoustic
    // state. Preserve arbitrary interior values while making the rigid
    // physical faces admissible before retaining their fluxes.
    for (int j = 0; j < nth; ++j)
        for (int k = 0; k < nz; ++k)
        {
            u[0][j][k] = frame_u;
            u[nr - 2][j][k] = frame_u;
        }
    for (int i = 0; i < nr; ++i)
        for (int k = 0; k < nz; ++k)
        {
            v[i][0][k] = frame_v;
            v[i][nth - 2][k] = frame_v;
        }
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            w[i][j][nz - 1] = 0.0f;

    CartesianCGridTransportContext context;
    context.begin(nr, nth, nz, dt_s, nullptr, frame_u, frame_v);
    context.accumulate(
        rho, u, v, w, dt_s,
        nullptr, nullptr, tmv::AcousticMassFluxPrecision::NativeFloat);
    context.complete();
    REQUIRE(context.validate(nr, nth, nz, dt_s));

    constexpr int i = 2;
    constexpr int j = 2;
    constexpr int k = 1;
    REQUIRE(context.mass_flux_x[i][j][k]
            == tmv::device_acoustic_energy::x_face_mass_flux(
                rho, u, i, j, k, frame_u));
    REQUIRE(context.mass_flux_y[i][j][k]
            == tmv::device_acoustic_energy::y_face_mass_flux(
                rho, v, i, j, k, frame_v));
    REQUIRE(context.mass_flux_z[i][j][k]
            == tmv::device_acoustic_energy::z_face_mass_flux(
                rho, w, nz, i, j, k));

    const float inv_dx = 1.0f / dx;
    const float inv_dy = 1.0f / dy;
    const float inv_dz = 1.0f / dz;
    const float divm_x =
        (context.mass_flux_x[i][j][k]
         - context.mass_flux_x[i - 1][j][k]) * inv_dx;
    const float divm_y =
        (context.mass_flux_y[i][j][k]
         - context.mass_flux_y[i][j - 1][k]) * inv_dy;
    const float divm_z =
        (context.mass_flux_z[i][j][k]
         - context.mass_flux_z[i][j][k - 1]) * inv_dz;
    const float retained_drho = -(divm_x + divm_y + divm_z);
    const float retained_requested = retained_drho * dt_s;
    REQUIRE(retained_requested
            == tmv::device_acoustic_energy::requested_density_change(
                rho, u, v, w, nr, nth, nz, i, j, k,
                dt_s, dx, dy, dz, frame_u, frame_v));

    CartesianCGridTransportContext precise;
    precise.begin(nr, nth, nz, dt_s, nullptr, frame_u, frame_v);
    precise.accumulate(rho, u, v, w, dt_s, nullptr, nullptr,
                       tmv::AcousticMassFluxPrecision::NativeCompensated);
    precise.complete();
    REQUIRE(precise.validate(nr, nth, nz, dt_s));
    REQUIRE(precise.mass_flux_x[i][j][k] == static_cast<float>(
        tmv::device_acoustic_energy::x_face_mass_flux_precise(rho, u, i, j, k, frame_u)));
    REQUIRE(precise.mass_flux_y[i][j][k] == static_cast<float>(
        tmv::device_acoustic_energy::y_face_mass_flux_precise(rho, v, i, j, k, frame_v)));
    REQUIRE(precise.mass_flux_z[i][j][k] == static_cast<float>(
        tmv::device_acoustic_energy::z_face_mass_flux_precise(rho, w, nz, i, j, k)));

    CartesianCGridTransportContext invalid_precision;
    invalid_precision.begin(nr, nth, nz, dt_s);
    REQUIRE_THROWS_WITH(invalid_precision.accumulate(rho, u, v, w, dt_s, nullptr, nullptr,
        static_cast<tmv::AcousticMassFluxPrecision>(99)), Catch::Matchers::Contains("precision"));
    REQUIRE(invalid_precision.mass_flux_x[i][j][k] == 0.0f);

    global_stagger_type = StaggerType::Collocated;
    CartesianCGridTransportContext wrong_layout;
    wrong_layout.begin(nr, nth, nz, dt_s);
    REQUIRE_THROWS_AS(
        wrong_layout.accumulate(
            rho, u, v, w, dt_s,
            nullptr, nullptr, tmv::AcousticMassFluxPrecision::NativeFloat),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        wrong_layout.accumulate(rho, u, v, w, dt_s, nullptr, nullptr,
            tmv::AcousticMassFluxPrecision::NativeCompensated), std::invalid_argument);
    global_stagger_type = StaggerType::CGrid;
}

TEST_CASE("Collocated Cartesian continuity context closes walls and preserves a constant scalar",
          "[numerics][advection][cartesian][agrid][AG-003][context]")
{
    const int nr = 9, nth = 8, nz = 7;
    const double dt_s = 0.2;
    setup_cartesian_grid(nr, nth, nz,
                         /*dx=*/100.0, /*dy=*/120.0, /*dz=*/80.0, dt_s);
    global_stagger_type = StaggerType::Collocated;
    global_lateral_bc_west = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_east = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_south = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_north = tmv::init::config::LateralBC::RigidSlip;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;

    rho.resize(nr, nth, nz, 0.0f);
    u.resize(nr, nth, nz, 0.0f);
    v.resize(nr, nth, nz, 0.0f);
    w.resize(nr, nth, nz, 0.0f);
    Field3D pressure(nr, nth, nz, 90000.0f);
    const std::vector<double> base_pressure(
        static_cast<std::size_t>(nz), 90000.0);
    const tmv::cartesian_mass_flux::CollocatedPressureCoupling
        pressure_coupling{
            &pressure, &base_pressure, dt_s, dr, dtheta, dz};
    constexpr float constant_q = 0.0075f;
    Field3D scalar(nr, nth, nz, constant_q);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                rho[i][j][k] = static_cast<float>(
                    1.0 + 0.01 * i - 0.003 * j + 0.002 * k);
                u[i][j][k] = static_cast<float>(0.5 + 0.08 * i - 0.02 * j);
                v[i][j][k] = static_cast<float>(-0.3 + 0.04 * j + 0.01 * k);
                w[i][j][k] = static_cast<float>(0.03 * k - 0.01 * i);
            }

    CartesianCGridTransportContext context;
    context.begin(nr, nth, nz, dt_s);
    context.accumulate(rho, u, v, w, dt_s, &pressure_coupling);
    REQUIRE(context.staggering() == StaggerType::Collocated);
    for (int j = 0; j < nth; ++j)
        for (int k = 0; k < nz; ++k)
        {
            REQUIRE(context.mass_flux_x[0][j][k] == 0.0f);
            REQUIRE(context.mass_flux_x[nr - 2][j][k] == 0.0f);
        }
    for (int i = 0; i < nr; ++i)
        for (int k = 0; k < nz; ++k)
        {
            REQUIRE(context.mass_flux_y[i][0][k] == 0.0f);
            REQUIRE(context.mass_flux_y[i][nth - 2][k] == 0.0f);
        }
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            REQUIRE(context.mass_flux_z[i][j][0] == 0.0f);
            REQUIRE(context.mass_flux_z[i][j][nz - 2] == 0.0f);
        }

    const int fi = 3, fj = 4, fk = 3;
    const double expected_x = 0.25
        * (static_cast<double>(rho[fi][fj][fk]) + rho[fi + 1][fj][fk])
        * (static_cast<double>(u[fi][fj][fk]) + u[fi + 1][fj][fk]);
    REQUIRE(context.mass_flux_x[fi][fj][fk]
            == Approx(expected_x).margin(2.0e-7));

    long double inventory_before = 0.0L;
    Field3D rho_next = rho;
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                inventory_before += static_cast<long double>(rho[i][j][k])
                                  * constant_q;
                const double div =
                    (static_cast<double>(context.mass_flux_x[i][j][k])
                     - context.mass_flux_x[i - 1][j][k]) / dr
                  + (static_cast<double>(context.mass_flux_y[i][j][k])
                     - context.mass_flux_y[i][j - 1][k]) / dtheta
                  + (static_cast<double>(context.mass_flux_z[i][j][k])
                     - context.mass_flux_z[i][j][k - 1]) / dz;
                rho_next[i][j][k] = static_cast<float>(
                    static_cast<double>(rho[i][j][k]) - dt_s * div);
            }
    rho = std::move(rho_next);
    context.complete();
    REQUIRE(context.validate(nr, nth, nz, dt_s));

    ScalarBoundaryTransportReport scalar_report;
    advect_scalar_conservative_3d_cartesian(
        scalar, scalar, dt_s, /*positive_definite=*/true, &context,
        &scalar_report);

    long double inventory_after = 0.0L;
    double maximum_constant_error = 0.0;
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                inventory_after += static_cast<long double>(rho[i][j][k])
                                 * scalar[i][j][k];
                maximum_constant_error = std::max(
                    maximum_constant_error,
                    std::abs(static_cast<double>(scalar[i][j][k])
                             - constant_q));
            }
    const double relative_inventory_error = static_cast<double>(
        std::abs(inventory_after - inventory_before)
        / std::abs(inventory_before));
    REQUIRE(maximum_constant_error < 2.0e-9);
    REQUIRE(relative_inventory_error < 2.0e-7);
    REQUIRE(scalar_report.complete);
    REQUIRE(scalar_report.outward_boundary_inventory == 0.0);
    REQUIRE(scalar_report.flux_implied_state_change == 0.0);
    REQUIRE(
        scalar_report.committed_state_change
        == Approx(static_cast<double>(
            (inventory_after - inventory_before) * dr * dtheta * dz)));
    REQUIRE(
        scalar_report.closure_residual
        == scalar_report.committed_state_change);
}

TEST_CASE("C-grid open top shares one dry-mass flux across continuity, ledger, and scalar transport",
          "[numerics][advection][cartesian][cgrid][open-top][SCI-042]")
{
    const int nr = 6, nth = 5, nz = 4;
    const double dt_s = 0.25;
    setup_cartesian_grid(nr, nth, nz,
                         /*dx=*/100.0, /*dy=*/80.0, /*dz=*/50.0, dt_s);
    global_stagger_type = StaggerType::CGrid;
    global_lateral_bc_west = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_east = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_south = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_north = tmv::init::config::LateralBC::RigidSlip;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Open;

    constexpr float rho0 = 1.2f;
    constexpr float top_w = 0.4f;
    constexpr float q0 = 0.01f;
    rho.resize(nr, nth, nz, rho0);
    u.resize(nr, nth, nz, 0.0f);
    v.resize(nr, nth, nz, 0.0f);
    w.resize(nr, nth, nz, 0.0f);
    qv.resize(nr, nth, nz, 0.0f);
    qc.resize(nr, nth, nz, 0.0f);
    qr.resize(nr, nth, nz, 0.0f);
    qi.resize(nr, nth, nz, 0.0f);
    qs.resize(nr, nth, nz, 0.0f);
    qg.resize(nr, nth, nz, 0.0f);
    qh.resize(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            w[i][j][nz - 1] = top_w;

    CartesianCGridTransportContext context;
    context.begin(nr, nth, nz, dt_s);
    const tmv::CarrierKineticBoundaryInputs carrier_inputs{{
        &qv, &qc, &qr, &qi, &qs, &qg, &qh,
    }};
    tmv::CarrierKineticBoundaryInputs partial_inputs;
    partial_inputs.water[0] = &qv;
    CartesianCGridTransportContext partial_context;
    partial_context.begin(nr, nth, nz, dt_s);
    REQUIRE_THROWS(partial_context.accumulate(
        rho, u, v, w, dt_s, nullptr, &partial_inputs));
    context.accumulate(
        rho, u, v, w, dt_s, nullptr, &carrier_inputs);
    REQUIRE(context.mass_flux_z[2][2][nz - 1]
            == Approx(static_cast<double>(rho0) * top_w));
    REQUIRE(context.carrier_kinetic_transport_complete());

    // Apply the exact retained continuity flux to the physical density cells.
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
        {
            const double mass_div =
                static_cast<double>(context.mass_flux_z[i][j][nz - 1]) / dz;
            rho[i][j][nz - 1] = static_cast<float>(
                static_cast<double>(rho[i][j][nz - 1]) - dt_s * mass_div);
        }
    context.complete();

    const tmv::DynamicsBoundaryMassReport report = context.boundary_mass_report(
        rho, dr, dtheta, dz);
    const double expected_outward = dt_s
        * static_cast<double>((nr - 2) * (nth - 2))
        * dr * dtheta * static_cast<double>(rho0) * top_w;
    REQUIRE(report.outward_boundary_mass_kg
            == Approx(expected_outward).epsilon(2.0e-7));
    const double expected_carrier_outward =
        expected_outward * 0.25
        * static_cast<double>(top_w) * static_cast<double>(top_w);
    REQUIRE(context.outward_carrier_kinetic_transport_j()
            == Approx(expected_carrier_outward).epsilon(2.0e-7));
    REQUIRE(report.flux_implied_state_change_kg
            == Approx(-report.outward_boundary_mass_kg));
    REQUIRE(report.committed_state_change_kg
            == Approx(-expected_outward).epsilon(3.0e-5));
    REQUIRE(std::abs(report.closure_residual_kg)
            < 3.0e-5 * expected_outward);

    Field3D scalar(nr, nth, nz, q0);
    ScalarBoundaryTransportReport scalar_report;
    advect_scalar_conservative_3d_cartesian(
        scalar, scalar, dt_s, /*positive_definite=*/true, &context,
        &scalar_report);
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
                REQUIRE(scalar[i][j][k] == Approx(q0).margin(2.0e-8));
    const double expected_scalar_outward =
        expected_outward * static_cast<double>(q0);
    REQUIRE(scalar_report.complete);
    REQUIRE(scalar_report.outward_boundary_inventory
            == Approx(expected_scalar_outward).epsilon(2.0e-7));
    REQUIRE(scalar_report.flux_implied_state_change
            == Approx(-expected_scalar_outward).epsilon(2.0e-7));
    REQUIRE(scalar_report.committed_state_change
            == Approx(-expected_scalar_outward).epsilon(3.0e-5));
    REQUIRE(std::abs(scalar_report.closure_residual)
            < 3.0e-5 * expected_scalar_outward);

    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("conservative C-grid reconstruction selector executes distinct TVD and WENO5 operators",
          "[numerics][advection][cartesian][cgrid][selection][config-truth]")
{
    constexpr int nr = 12;
    constexpr int nth = 4;
    constexpr int nz = 4;
    setup_cartesian_grid(nr, nth, nz, 1.0, 1.0, 1.0, 0.2);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(nr, nth, nz, 1.0f);
    set_uniform_velocity(1.0, 0.0, 0.0);

    Field3D source(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                source[i][j][k] = static_cast<float>(
                    0.25 + 0.5 * std::exp(-0.35 * (i - 5) * (i - 5)));

    Field3D tvd;
    Field3D weno5;
    advect_scalar_conservative_3d_cartesian(
        source, tvd, 0.2, false, nullptr, nullptr,
        ScalarMoistEnergyKind::None, nullptr,
        tmv::CartesianFaceReconstruction::MonotonicCentered);
    advect_scalar_conservative_3d_cartesian(
        source, weno5, 0.2, false, nullptr, nullptr,
        ScalarMoistEnergyKind::None, nullptr,
        tmv::CartesianFaceReconstruction::WENO5);

    REQUIRE(std::memcmp(tvd.data(), weno5.data(),
                        tvd.size() * sizeof(float)) != 0);
    const auto extrema = std::minmax_element(source.data(),
                                             source.data() + source.size());
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                REQUIRE(tvd[i][j][k] >= *extrema.first);
                REQUIRE(tvd[i][j][k] <= *extrema.second);
    }
}

TEST_CASE("mapped collocated public advection selection is conservative deterministic and atomic",
          "[numerics][advection][cartesian][agrid][terrain][selection][SCI-430]")
{
    constexpr int nr = 13;
    constexpr int nth = 10;
    constexpr int nz = 8;
    constexpr double dx_m = 100.0;
    constexpr double dy_m = 80.0;
    constexpr double dzeta_m = 50.0;
    constexpr double dt_s = 0.2;
    setup_cartesian_grid(nr, nth, nz, dx_m, dy_m, dzeta_m, dt_s);
    global_stagger_type = StaggerType::Collocated;
    global_lateral_bc_west = tmv::init::config::LateralBC::Rigid;
    global_lateral_bc_east = tmv::init::config::LateralBC::Rigid;
    global_lateral_bc_south = tmv::init::config::LateralBC::Rigid;
    global_lateral_bc_north = tmv::init::config::LateralBC::Rigid;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;

    const TerrainMetrics3D metrics = make_sloping_transport_metrics(
        nr, nth, nz, dx_m, dy_m, dzeta_m);
    rho.resize(nr, nth, nz, 1.1f);
    u.resize(nr, nth, nz, 2.0f);
    v.resize(nr, nth, nz, -1.0f);
    w.resize(nr, nth, nz, 0.25f);
    CartesianCGridTransportContext context;
    prepare_sloping_transport_context(metrics, dt_s, context);
    REQUIRE(context.terrain_metrics() == &metrics);
    REQUIRE(context.mass_flux_x[3][3][3] != 0.0f);

    Field3D source(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                source[i][j][k] = static_cast<float>(
                    0.3 + 0.35 * std::exp(
                        -0.13 * (i - 6) * (i - 6)
                        -0.18 * (j - 4) * (j - 4))
                    + 0.015 * std::sin(0.8 * k + 0.2 * i));

    const auto run_public = [&](const char* scheme_id, int threads)
    {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        Field3D result = source;
        AdvectionRuntimeState runtime;
        runtime.config.scheme_id = scheme_id;
        advect_scalar_3d(
            result, dt_s, 0.0, false, &context, &runtime);
        return result;
    };

#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#endif
    const Field3D none = run_public("none", 1);
    REQUIRE(std::memcmp(none.data(), source.data(),
                        source.size() * sizeof(float)) == 0);

    const Field3D tvd_one = run_public("tvd", 1);
    const Field3D tvd_four = run_public("tvd", 4);
    const Field3D weno_one = run_public("weno5", 1);
    const Field3D weno_four = run_public("weno5", 4);
    REQUIRE(std::memcmp(tvd_one.data(), tvd_four.data(),
                        tvd_one.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(weno_one.data(), weno_four.data(),
                        weno_one.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(tvd_one.data(), weno_one.data(),
                        tvd_one.size() * sizeof(float)) != 0);

    const auto inventory = [&](const Field3D& density,
                               const Field3D& scalar)
    {
        long double total = 0.0L;
        for (int i = 1; i < nr - 1; ++i)
            for (int j = 1; j < nth - 1; ++j)
                for (int k = 1; k < nz - 1; ++k)
                    total += static_cast<long double>(metrics.J(i, j, k)) *
                             density[i][j][k] * scalar[i][j][k];
        return total;
    };
    const long double beginning_inventory = inventory(
        context.density_begin, source);

    const std::array<std::pair<CartesianFaceReconstruction,
                               const Field3D*>, 2> selections = {{
        {CartesianFaceReconstruction::MonotonicCentered, &tvd_one},
        {CartesianFaceReconstruction::WENO5, &weno_one},
    }};
    for (const auto& [reconstruction, public_result] : selections)
    {
        Field3D direct;
        ScalarBoundaryTransportReport report;
        advect_scalar_conservative_3d_cartesian(
            source, direct, dt_s, false, &context, &report,
            ScalarMoistEnergyKind::None, nullptr, reconstruction);
        REQUIRE(std::memcmp(direct.data(), public_result->data(),
                            direct.size() * sizeof(float)) == 0);
        const long double ending_inventory = inventory(rho, direct);
        const double relative_error = static_cast<double>(
            std::abs(ending_inventory - beginning_inventory) /
            std::abs(beginning_inventory));
        REQUIRE(relative_error < 2.0e-7);
        REQUIRE(report.complete);
        REQUIRE(report.outward_boundary_inventory == 0.0);
        REQUIRE(report.flux_implied_state_change == 0.0);
    }

    for (const CartesianFaceReconstruction reconstruction : {
             CartesianFaceReconstruction::MonotonicCentered,
             CartesianFaceReconstruction::WENO5})
    {
        Field3D first = source;
        Field3D wrong_shape(nr - 1, nth, nz, 0.5f);
        const Field3D before = first;
        std::array<CartesianScalarTransport, 2> batch{};
        batch[0].field = &first;
        batch[1].field = &wrong_shape;
        REQUIRE_THROWS_AS(
            advect_scalars_conservative_3d_cartesian(
                batch.data(), batch.size(), dt_s, &context,
                reconstruction),
            std::invalid_argument);
        REQUIRE(std::memcmp(first.data(), before.data(),
                            first.size() * sizeof(float)) == 0);
    }
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#endif
}

TEST_CASE("Collocated Cartesian periodic diagonal transport closes 100-step inventory",
          "[numerics][advection][cartesian][agrid][AG-003][periodic][long]")
{
    const int nr = 14, nth = 13, nz = 5;
    const double dt_s = 0.25;
    setup_cartesian_grid(nr, nth, nz,
                         /*dx=*/100.0, /*dy=*/100.0, /*dz=*/80.0, dt_s);
    global_stagger_type = StaggerType::Collocated;
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;

    const auto close_periodic = [=](Field3D& field) {
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                field[0][j][k] = field[nr - 2][j][k];
                field[nr - 1][j][k] = field[1][j][k];
            }
        for (int i = 0; i < nr; ++i)
            for (int k = 0; k < nz; ++k)
            {
                field[i][0][k] = field[i][nth - 2][k];
                field[i][nth - 1][k] = field[i][1][k];
            }
    };
    const auto inventory = [=](const Field3D& density,
                               const Field3D& scalar) {
        long double total = 0.0L;
        for (int i = 1; i < nr - 1; ++i)
            for (int j = 1; j < nth - 1; ++j)
                for (int k = 1; k < nz - 1; ++k)
                    total += static_cast<long double>(density[i][j][k])
                           * scalar[i][j][k];
        return total;
    };
    Field3D pressure(nr, nth, nz, 90000.0f);
    const std::vector<double> base_pressure(
        static_cast<std::size_t>(nz), 90000.0);
    const tmv::cartesian_mass_flux::CollocatedPressureCoupling
        pressure_coupling{
            &pressure, &base_pressure, dt_s, dr, dtheta, dz};

    for (const double sign : {1.0, -1.0})
    {
        INFO("diagonal velocity sign=" << sign);
        rho.resize(nr, nth, nz, 1.1f);
        u.resize(nr, nth, nz, static_cast<float>(sign * 7.0));
        v.resize(nr, nth, nz, static_cast<float>(sign * 5.0));
        w.resize(nr, nth, nz, 0.0f);
        Field3D scalar(nr, nth, nz, 0.0f);
        for (int i = 1; i < nr - 1; ++i)
            for (int j = 1; j < nth - 1; ++j)
                for (int k = 1; k < nz - 1; ++k)
                {
                    const double dx_cell = static_cast<double>(i - 6);
                    const double dy_cell = static_cast<double>(j - 6);
                    scalar[i][j][k] = static_cast<float>(
                        1.0e-4 + 0.01 * std::exp(
                            -0.16 * (dx_cell * dx_cell + dy_cell * dy_cell)));
                }
        close_periodic(scalar);
        const long double before = inventory(rho, scalar);

        for (int step = 0; step < 100; ++step)
        {
            CartesianCGridTransportContext context;
            context.begin(nr, nth, nz, dt_s);
            context.accumulate(
                rho, u, v, w, dt_s, &pressure_coupling);
            context.complete();
            REQUIRE(context.mass_flux_x[0][6][2]
                    == context.mass_flux_x[nr - 2][6][2]);
            REQUIRE(context.mass_flux_y[6][0][2]
                    == context.mass_flux_y[6][nth - 2][2]);
            advect_scalar_conservative_3d_cartesian(
                scalar, scalar, dt_s, /*positive_definite=*/true, &context);
            close_periodic(scalar);
        }

        const long double after = inventory(rho, scalar);
        double minimum = std::numeric_limits<double>::infinity();
        for (int i = 1; i < nr - 1; ++i)
            for (int j = 1; j < nth - 1; ++j)
                for (int k = 1; k < nz - 1; ++k)
                    minimum = std::min(
                        minimum, static_cast<double>(scalar[i][j][k]));
        const double relative_error = static_cast<double>(
            std::abs(after - before) / std::abs(before));
        REQUIRE(minimum >= 0.0);
        REQUIRE(relative_error < 2.0e-6);
    }

    // Force the outgoing-flux limiter active at a sparse periodic donor. The
    // west/east and south/north copies must use the physical donor's limiter,
    // not an uninitialized ghost-cell limiter, or periodic inventory leaks.
    rho.resize(nr, nth, nz, 1.0f);
    u.resize(nr, nth, nz, 300.0f);
    v.resize(nr, nth, nz, 300.0f);
    w.resize(nr, nth, nz, 0.0f);
    Field3D sparse(nr, nth, nz, 0.0f);
    sparse[nr - 2][nth - 2][2] = 1.0f;
    close_periodic(sparse);
    const long double sparse_before = inventory(rho, sparse);
    CartesianCGridTransportContext limiter_context;
    limiter_context.begin(nr, nth, nz, dt_s);
    limiter_context.accumulate(
        rho, u, v, w, dt_s, &pressure_coupling);
    limiter_context.complete();
    ScalarBoundaryTransportReport limiter_report;
    advect_scalar_conservative_3d_cartesian(
        sparse, sparse, dt_s, /*positive_definite=*/true, &limiter_context,
        &limiter_report);
    close_periodic(sparse);
    const long double sparse_after = inventory(rho, sparse);
    double sparse_minimum = std::numeric_limits<double>::infinity();
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
                sparse_minimum = std::min(
                    sparse_minimum, static_cast<double>(sparse[i][j][k]));
    REQUIRE(sparse_minimum >= 0.0);
    REQUIRE(static_cast<double>(
        std::abs(sparse_after - sparse_before) / sparse_before) < 2.0e-6);
    REQUIRE(limiter_report.complete);
    REQUIRE(limiter_report.outward_boundary_inventory == 0.0);
    REQUIRE(limiter_report.flux_implied_state_change == 0.0);
    REQUIRE(std::abs(limiter_report.closure_residual)
            < 2.0e-6 * static_cast<double>(sparse_before)
                * dr * dtheta * dz);
}

TEST_CASE("Cartesian scalar boundary report requires retained transport history",
          "[numerics][advection][cartesian][boundary-ledger][fail-closed]")
{
    setup_cartesian_grid(/*nr=*/6, /*nth=*/5, /*nz=*/4,
                         /*dx=*/100.0, /*dy=*/80.0, /*dz=*/50.0,
                         /*dt=*/0.25);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 1.0f);
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    Field3D scalar(NR, NTH, NZ, 0.01f);
    Field3D output(NR, NTH, NZ, -1.0f);
    const Field3D output_before = output;
    ScalarBoundaryTransportReport report;

    REQUIRE_THROWS_WITH(
        advect_scalar_conservative_3d_cartesian(
            scalar, output, 0.25, true, nullptr, &report),
        Catch::Matchers::Contains("requires a retained mass-flux context"));
    REQUIRE_FALSE(report.complete);
    for (std::size_t index = 0; index < output.size(); ++index)
        REQUIRE(output.data()[index] == output_before.data()[index]);

    ScalarMoistEnergyTransportReportV1 energy_report;
    REQUIRE_THROWS_WITH(
        advect_scalar_conservative_3d_cartesian(
            scalar, output, 0.25, true, nullptr, nullptr,
            ScalarMoistEnergyKind::WaterVapor, &energy_report),
        Catch::Matchers::Contains("requires a retained mass-flux context"));
    REQUIRE_FALSE(energy_report.complete);
    REQUIRE_THROWS_WITH(
        advect_scalar_conservative_3d_cartesian(
            scalar, output, 0.25, true, nullptr, nullptr,
            ScalarMoistEnergyKind::WaterVapor, nullptr),
        Catch::Matchers::Contains("exactly one non-null report"));
    for (std::size_t index = 0; index < output.size(); ++index)
        REQUIRE(output.data()[index] == output_before.data()[index]);
}

TEST_CASE("CPU C-grid scalar batch reuses retained mass flux and in-place scratch",
          "[numerics][advection][cartesian][cgrid][cpu008][batch]")
{
    const int nr = 14, nth = 12, nz = 8;
    const double dt_s = 0.4;
    setup_cartesian_grid(nr, nth, nz,
                         /*dx=*/120.0, /*dy=*/150.0, /*dz=*/90.0, dt_s);
    global_stagger_type = StaggerType::CGrid;

    rho.resize(nr, nth, nz, 0.0f);
    u.resize(nr, nth, nz, 0.0f);
    v.resize(nr, nth, nz, 0.0f);
    w.resize(nr, nth, nz, 0.0f);
    Field3D theta_batch(nr, nth, nz, 0.0f);
    Field3D moisture_batch(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                rho[i][j][k] = static_cast<float>(
                    1.05 + 0.002 * i - 0.001 * j + 0.0005 * k);
                u[i][j][k] = static_cast<float>(4.0 + 0.04 * j);
                v[i][j][k] = static_cast<float>(-2.0 + 0.03 * i);
                w[i][j][k] = static_cast<float>(0.15 * std::sin(0.4 * k));
                theta_batch[i][j][k] = static_cast<float>(
                    300.0 + 0.4 * std::sin(0.3 * i)
                    * std::cos(0.2 * j) * std::exp(-0.1 * k));
                moisture_batch[i][j][k] = static_cast<float>(
                    0.002 + 0.001 * std::exp(
                        -0.08 * ((i - 7) * (i - 7) + (j - 6) * (j - 6))));
            }

    CartesianCGridTransportContext context;
    context.begin(nr, nth, nz, dt_s);
    context.accumulate(rho, u, v, w, dt_s);
    context.complete();

    const Field3D theta_initial = theta_batch;
    const Field3D moisture_initial = moisture_batch;

    Field3D theta_reference;
    advect_scalar_conservative_3d_cartesian(
        theta_batch, theta_reference, dt_s, false, &context);
    Field3D moisture_reference;
    advect_scalar_conservative_3d_cartesian(
        moisture_batch, moisture_reference, dt_s, true, &context);

    CartesianScalarTransport batch[] = {
        {&theta_batch, 0.0, false},
        {&moisture_batch, 0.0, true},
    };
    advect_scalars_conservative_3d_cartesian(
        batch, std::size(batch), dt_s, &context);

    for (std::size_t idx = 0; idx < theta_batch.size(); ++idx)
    {
        REQUIRE(theta_batch.data()[idx] == theta_reference.data()[idx]);
        REQUIRE(moisture_batch.data()[idx] == moisture_reference.data()[idx]);
    }

    // The first-class CPU backend consumes the same high-level host views,
    // so dispatching through AdvectionComputeKernels is bit-identical to the
    // direct CPU-008 batch and does not rebuild context.
    theta_batch = theta_initial;
    moisture_batch = moisture_initial;
    Field3D* host_fields[] = {&theta_batch, &moisture_batch};
    const float* q_in[] = {theta_batch.data(), moisture_batch.data()};
    float* q_out[] = {theta_batch.data(), moisture_batch.data()};
    const float dt_kappas[] = {0.0f, 0.0f};
    const int positive[] = {0, 1};
    const double kappas[] = {0.0, 0.0};
    ScalarBoundaryTransportReport cpu_backend_report_storage[2];
    ScalarBoundaryTransportReport* cpu_backend_reports[] = {
        &cpu_backend_report_storage[0], &cpu_backend_report_storage[1],
    };

    ConservativeCartesianCgridTransportDispatch dispatch;
    dispatch.u = u.data();
    dispatch.v = v.data();
    dispatch.w = w.data();
    dispatch.rho_data = rho.data();
    dispatch.q_in = q_in;
    dispatch.q_out = q_out;
    dispatch.dt_kappas = dt_kappas;
    dispatch.positive_definite = positive;
    dispatch.n_scalars = 2;
    dispatch.nx = nr;
    dispatch.ny = nth;
    dispatch.nz = nz;
    dispatch.dt_step = static_cast<float>(dt_s);
    dispatch.use_supplied_mass_flux = true;
    dispatch.density_begin = context.density_begin.data();
    dispatch.mass_flux_x = context.mass_flux_x.data();
    dispatch.mass_flux_y = context.mass_flux_y.data();
    dispatch.mass_flux_z = context.mass_flux_z.data();
    dispatch.host_fields = host_fields;
    dispatch.host_kappas = kappas;
    dispatch.host_context = &context;
    dispatch.dt_step_host = dt_s;
    dispatch.host_boundary_reports = cpu_backend_reports;

    auto cpu_backend = create_cpu_compute_backend();
    std::string backend_error;
    REQUIRE(cpu_backend->initialize(backend_error));
    REQUIRE(cpu_backend->advection_kernels() != nullptr);
    REQUIRE(cpu_backend->advection_kernels()
        ->dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
    for (std::size_t idx = 0; idx < theta_batch.size(); ++idx)
    {
        REQUIRE(theta_batch.data()[idx] == theta_reference.data()[idx]);
        REQUIRE(moisture_batch.data()[idx] == moisture_reference.data()[idx]);
    }
    REQUIRE(cpu_backend_report_storage[0].complete);
    REQUIRE(cpu_backend_report_storage[1].complete);
    REQUIRE(std::isfinite(
        cpu_backend_report_storage[0].closure_residual));
    REQUIRE(std::isfinite(
        cpu_backend_report_storage[1].closure_residual));
    cpu_backend->shutdown();

    const Field3D unchanged = theta_batch;
    CartesianScalarTransport invalid[] = {
        {&theta_batch, 0.0, false},
        {nullptr, 0.0, true},
    };
    REQUIRE_THROWS(advect_scalars_conservative_3d_cartesian(
        invalid, std::size(invalid), dt_s, &context));
    for (std::size_t idx = 0; idx < theta_batch.size(); ++idx)
    {
        REQUIRE(theta_batch.data()[idx] == unchanged.data()[idx]);
    }
    REQUIRE_THROWS(advect_scalars_conservative_3d_cartesian(
        batch, std::size(batch), dt_s, nullptr));
    CartesianScalarTransport chained_filter[] = {
        {&theta_batch, 0.01, false},
    };
    REQUIRE_THROWS_AS(advect_scalars_conservative_3d_cartesian(
        chained_filter, std::size(chained_filter), dt_s, &context),
        std::invalid_argument);

    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("C-grid theta transport retains exact interface-strip fluxes",
          "[numerics][advection][cartesian][cgrid][interface-flux]")
{
    const int nr = 10, nth = 9, nz = 4;
    const double dt_s = 0.25;
    setup_cartesian_grid(nr, nth, nz, 100.0, 80.0, 50.0, dt_s);
    global_stagger_type = StaggerType::CGrid;

    rho.resize(nr, nth, nz, 1.0f);
    u.resize(nr, nth, nz, 2.0f);
    v.resize(nr, nth, nz, -1.0f);
    w.resize(nr, nth, nz, 0.0f);
    Field3D theta_field(nr, nth, nz, 300.0f);

    CartesianCGridTransportContext context;
    context.begin(nr, nth, nz, dt_s);
    context.accumulate(rho, u, v, w, dt_s);
    context.complete();

    CartesianInterfaceFluxRegister register_one;
    register_one.configure(nr, nth, nz, {3, 7, 2, 6, 0, nz});
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    advect_scalar_conservative_3d_cartesian(
        theta_field, theta_field, dt_s, false, &context, nullptr,
        ScalarMoistEnergyKind::None, nullptr,
        CartesianFaceReconstruction::WENO5, &register_one);

    const auto& captured = register_one.theta();
    REQUIRE(captured.contribution_count == 1);
    REQUIRE(captured.accumulated_dt_s == dt_s);
    REQUIRE(captured.west.size() == 4U * static_cast<std::size_t>(nz));
    REQUIRE(captured.south.size() == 4U * static_cast<std::size_t>(nz));
    std::size_t bin = 0;
    for (int j = 2; j < 6; ++j)
        for (int k = 0; k < nz; ++k, ++bin)
        {
            const double scale = dt_s * dtheta * dz * 300.0;
            REQUIRE(captured.west[bin] ==
                    Approx(scale * context.mass_flux_x[2][j][k]));
            REQUIRE(captured.east[bin] ==
                    Approx(scale * context.mass_flux_x[6][j][k]));
        }
    bin = 0;
    for (int i = 3; i < 7; ++i)
        for (int k = 0; k < nz; ++k, ++bin)
        {
            const double scale = dt_s * dr * dz * 300.0;
            REQUIRE(captured.south[bin] ==
                    Approx(scale * context.mass_flux_y[i][1][k]));
            REQUIRE(captured.north[bin] ==
                    Approx(scale * context.mass_flux_y[i][5][k]));
        }

    Field3D theta_four(nr, nth, nz, 300.0f);
    CartesianInterfaceFluxRegister register_four;
    register_four.configure(nr, nth, nz, {3, 7, 2, 6, 0, nz});
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    advect_scalar_conservative_3d_cartesian(
        theta_four, theta_four, dt_s, false, &context, nullptr,
        ScalarMoistEnergyKind::None, nullptr,
        CartesianFaceReconstruction::WENO5, &register_four);
    for (std::size_t index = 0; index < theta_field.size(); ++index)
        REQUIRE(theta_four.data()[index] == theta_field.data()[index]);
    REQUIRE(register_four.theta().west == captured.west);
    REQUIRE(register_four.theta().east == captured.east);
    REQUIRE(register_four.theta().south == captured.south);
    REQUIRE(register_four.theta().north == captured.north);

    Field3D theta_fine(nr, nth, nz, 300.0f);
    CartesianInterfaceFluxRegister fine_register;
    fine_register.configure(nr, nth, nz, {3, 7, 2, 6, 0, nz}, 2, 2);
    advect_scalar_conservative_3d_cartesian(
        theta_fine, theta_fine, dt_s, false, &context, nullptr,
        ScalarMoistEnergyKind::None, nullptr,
        CartesianFaceReconstruction::WENO5, &fine_register);
    REQUIRE(fine_register.theta().west.size() ==
            2U * static_cast<std::size_t>(nz));
    for (std::size_t coarse = 0; coarse < fine_register.theta().west.size();
         ++coarse)
    {
        const std::size_t tangential = coarse / static_cast<std::size_t>(nz);
        const std::size_t k = coarse % static_cast<std::size_t>(nz);
        const std::size_t first =
            (2U * tangential) * static_cast<std::size_t>(nz) + k;
        const std::size_t second = first + static_cast<std::size_t>(nz);
        REQUIRE(fine_register.theta().west[coarse] ==
                captured.west[first] + captured.west[second]);
        REQUIRE(fine_register.theta().east[coarse] ==
                captured.east[first] + captured.east[second]);
        REQUIRE(fine_register.theta().south[coarse] ==
                captured.south[first] + captured.south[second]);
        REQUIRE(fine_register.theta().north[coarse] ==
                captured.north[first] + captured.north[second]);
    }

    const Field3D before_failure = theta_field;
    CartesianInterfaceFluxRegister wrong_shape;
    wrong_shape.configure(nr + 1, nth, nz, {3, 7, 2, 6, 0, nz});
    REQUIRE_THROWS_WITH(
        advect_scalar_conservative_3d_cartesian(
            theta_field, theta_field, dt_s, false, &context, nullptr,
            ScalarMoistEnergyKind::None, nullptr,
            CartesianFaceReconstruction::WENO5, &wrong_shape),
        Catch::Matchers::Contains("not configured for the active level"));
    REQUIRE(wrong_shape.theta().contribution_count == 0);
    for (std::size_t index = 0; index < theta_field.size(); ++index)
        REQUIRE(theta_field.data()[index] == before_failure.data()[index]);

    Field3D late_theta(nr, nth, nz, 301.0f);
    Field3D late_qv(nr, nth, nz, 0.01f);
    late_qv[4][4][1] = std::numeric_limits<float>::quiet_NaN();
    ScalarBoundaryTransportReport late_qv_report;
    CartesianScalarTransport late_batch[] = {
        {&late_theta, 0.0, false, nullptr,
         ScalarMoistEnergyKind::None, nullptr, &register_one},
        {&late_qv, 0.0, true, &late_qv_report},
    };
    const auto committed_before_late_failure = register_one.theta();
    REQUIRE_THROWS(advect_scalars_conservative_3d_cartesian(
        late_batch, std::size(late_batch), dt_s, &context));
    REQUIRE(register_one.theta().west ==
            committed_before_late_failure.west);
    REQUIRE(register_one.theta().east ==
            committed_before_late_failure.east);
    REQUIRE(register_one.theta().south ==
            committed_before_late_failure.south);
    REQUIRE(register_one.theta().north ==
            committed_before_late_failure.north);
    REQUIRE(register_one.theta().accumulated_dt_s ==
            committed_before_late_failure.accumulated_dt_s);
    REQUIRE(register_one.theta().contribution_count ==
            committed_before_late_failure.contribution_count);

    global_stagger_type = StaggerType::Collocated;
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

TEST_CASE("Milbrandt transports all six number moments with retained mass flux",
          "[numerics][advection][cartesian][cgrid][milbrandt][moments]")
{
    const int nr = 14, nth = 12, nz = 8;
    const double dt_s = 0.35;
    setup_cartesian_grid(
        nr, nth, nz, /*dx=*/120.0, /*dy=*/150.0,
        /*dz=*/90.0, dt_s);
    global_stagger_type = StaggerType::CGrid;

    rho.resize(nr, nth, nz, 0.0f);
    u.resize(nr, nth, nz, 0.0f);
    v.resize(nr, nth, nz, 0.0f);
    w.resize(nr, nth, nz, 0.0f);
    Field3D qc_state(nr, nth, nz, 0.0f);
    Field3D qr_state(nr, nth, nz, 0.0f);
    Field3D qi_state(nr, nth, nz, 0.0f);
    Field3D qs_state(nr, nth, nz, 0.0f);
    Field3D qg_state(nr, nth, nz, 0.0f);
    Field3D qh_state(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                rho[i][j][k] = static_cast<float>(
                    1.05 + 0.002 * i - 0.001 * j + 0.0005 * k);
                u[i][j][k] = static_cast<float>(4.0 + 0.04 * j);
                v[i][j][k] = static_cast<float>(-2.0 + 0.03 * i);
                w[i][j][k] = static_cast<float>(
                    0.15 * std::sin(0.4 * k));
                const float pattern = static_cast<float>(
                    0.6 + 0.3 * std::sin(0.31 * i)
                    * std::cos(0.27 * j)
                    * std::exp(-0.08 * k));
                qc_state[i][j][k] = 1.2e-3f * pattern;
                qr_state[i][j][k] = 8.0e-4f * pattern;
                qi_state[i][j][k] = 2.0e-4f * pattern;
                qs_state[i][j][k] = 3.0e-4f * pattern;
                qg_state[i][j][k] = 1.5e-4f * pattern;
                qh_state[i][j][k] = 8.0e-5f * pattern;
            }

    CartesianCGridTransportContext context;
    context.begin(nr, nth, nz, dt_s);
    context.accumulate(rho, u, v, w, dt_s);
    context.complete();

    tmv::MilbrandtScheme scheme;
    scheme.prepare_prognostic_state(
        rho, qc_state, qr_state, qi_state,
        qs_state, qg_state, qh_state);
    const auto before = scheme.prognostic_scalars_for_transport();
    REQUIRE(before.size() == 6);
    std::array<Field3D, 6> reference;
    for (std::size_t moment = 0; moment < before.size(); ++moment)
    {
        advect_scalar_conservative_3d_cartesian(
            *before[moment].field, reference[moment],
            dt_s, true, &context);
    }

    AdvectionRuntimeState advection_runtime;
    advection_runtime.config.scheme_id = "weno5";
    advect_microphysics_prognostics_3d(
        scheme, dt_s, &context, &advection_runtime);
    const auto after = scheme.prognostic_scalars_for_transport();
    REQUIRE(after.size() == reference.size());
    for (std::size_t moment = 0; moment < after.size(); ++moment)
    {
        INFO("moment = " << after[moment].name);
        REQUIRE(after[moment].field->size()
                == reference[moment].size());
        for (std::size_t index = 0;
             index < reference[moment].size(); ++index)
        {
            REQUIRE(after[moment].field->data()[index]
                    == reference[moment].data()[index]);
            REQUIRE(after[moment].field->data()[index] >= 0.0f);
        }
    }

    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("Tracer advection driver: signal gate and C-grid routing (Phase 4.3)",
          "[numerics][advection][cartesian][cgrid][tracer]")
{
    // Free-stream invariant: conservative transport of an
    // exactly-zero field are exactly zero, so a zero tracer must come out
    // unchanged. A seeded tracer must advect identically to the production
    // kernel through whichever backend (the Phase 4.3 GPU one-scalar
    // batch or the CPU fallback) is active.
    const int nr = 20, nth = 20, nz = 12;
    const double dx_m = 400.0, dz_m = 250.0, dt_s = 1.5;
    const double kappa = 0.0;

    setup_cartesian_grid(nr, nth, nz, dx_m, dx_m, dz_m, dt_s);
    global_stagger_type = StaggerType::CGrid;

    u.resize(nr, nth, nz, 0.0f);
    v.resize(nr, nth, nz, 0.0f);
    w.resize(nr, nth, nz, 0.0f);
    rho.resize(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                u[i][j][k]   = 12.0f;
                v[i][j][k]   = -6.0f;
                w[i][j][k]   = 0.5f;
                rho[i][j][k] = static_cast<float>(
                    1.1 * std::exp(-(static_cast<double>(k) * dz_m) / 8000.0));
            }

    SECTION("zero tracer stays exactly zero")
    {
        AdvectionRuntimeState advection_runtime;
        advection_runtime.config.scheme_id = "weno5";
        tracer.resize(nr, nth, nz, 0.0f);
        advect_tracer_3d(dt_s, kappa, nullptr, &advection_runtime);
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                    REQUIRE(tracer[i][j][k] == 0.0f);
    }

    SECTION("seeded tracer matches the production kernel pair")
    {
        AdvectionRuntimeState advection_runtime;
        advection_runtime.config.scheme_id = "weno5";
        tracer.resize(nr, nth, nz, 0.0f);
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const double di = (i - 10.0) / 2.5;
                    const double dj = (j - 10.0) / 2.5;
                    const double dk = (k - 6.0) / 2.5;
                    tracer[i][j][k] = static_cast<float>(
                        std::exp(-0.5 * (di * di + dj * dj + dk * dk)));
                }
        const Field3D pre = tracer;

        advect_tracer_3d(dt_s, kappa, nullptr, &advection_runtime);

        Field3D ref_adv;
        advect_scalar_conservative_3d_cartesian(pre, ref_adv, dt_s);
        const Field3D& ref_final = ref_adv;

        double max_abs = 0.0;
        double max_motion = 0.0;
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    max_abs = std::max(max_abs,
                        std::abs(static_cast<double>(tracer[i][j][k])
                               - static_cast<double>(ref_final[i][j][k])));
                    max_motion = std::max(max_motion,
                        std::abs(static_cast<double>(tracer[i][j][k])
                               - static_cast<double>(pre[i][j][k])));
                }
        INFO("tracer vs production reference max_abs=" << max_abs
             << " motion=" << max_motion);
        REQUIRE(max_motion > 1.0e-3);   // the kernel actually ran
        REQUIRE(max_abs < 2.0e-5);      // GPU parity band; 0 on CPU fallback
    }

    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("passive tracer closes the exact t=3501.75 negative-subnormal residue",
          "[numerics][advection][cartesian][cgrid][tracer][roundoff][regression]")
{
    constexpr int nr = 6;
    constexpr int nth = 5;
    constexpr int nz = 4;
    constexpr double dx_m = 100.0;
    constexpr double dy_m = 80.0;
    constexpr double dz_m = 50.0;
    setup_cartesian_grid(nr, nth, nz, dx_m, dy_m, dz_m, 0.75);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(nr, nth, nz, 1.25f);
    tracer.resize(nr, nth, nz, 0.0f);
    global_grid_geometry.z_faces = {0.0, 40.0, 100.0, 180.0, 280.0};
    global_grid_geometry.z = {20.0, 70.0, 140.0, 230.0};

    constexpr std::uint32_t failure_bits = 0x8000000bU;
    float failure_value = 0.0f;
    std::memcpy(&failure_value, &failure_bits, sizeof(failure_bits));
    REQUIRE(failure_value
            == -11.0f * std::numeric_limits<float>::denorm_min());
    tracer[2][2][1] = failure_value;
    constexpr std::uint32_t second_failure_bits = 0x80000005U;
    float second_failure_value = 0.0f;
    std::memcpy(&second_failure_value, &second_failure_bits,
                sizeof(second_failure_bits));
    tracer[3][2][2] = second_failure_value;
    rho[3][2][2] = 0.8f;
    const float positive_subnormal =
        7.0f * std::numeric_limits<float>::denorm_min();
    tracer[2][2][2] = positive_subnormal;
    tracer[0][0][0] = -0.0f;

    AdvectionRuntimeState runtime;
    runtime.tracer_advected_in_batch = true;
    advect_tracer_3d(0.75, 0.0, nullptr, &runtime);

    std::uint32_t repaired_bits = 1U;
    const float repaired_value = tracer[2][2][1];
    std::memcpy(&repaired_bits, &repaired_value, sizeof(repaired_bits));
    REQUIRE(repaired_bits == 0U);
    std::uint32_t second_repaired_bits = 1U;
    const float second_repaired_value = tracer[3][2][2];
    std::memcpy(&second_repaired_bits, &second_repaired_value,
                sizeof(second_repaired_bits));
    REQUIRE(second_repaired_bits == 0U);
    std::uint32_t closure_bits = 1U;
    const float closure_value = tracer[0][0][0];
    std::memcpy(&closure_bits, &closure_value, sizeof(closure_bits));
    REQUIRE(closure_bits == 0U);
    REQUIRE(tracer[2][2][2] == positive_subnormal);
    REQUIRE_FALSE(runtime.tracer_advected_in_batch);

    const auto& report = runtime.tracer_roundoff_report;
    REQUIRE(report.complete);
    REQUIRE(report.negative_subnormal_count == 2);
    REQUIRE(report.negative_zero_count == 1);
    REQUIRE(report.closure_canonicalization_count == 1);
    std::uint32_t minimum_bits = 0U;
    std::memcpy(&minimum_bits, &report.most_negative_repaired_value,
                sizeof(minimum_bits));
    REQUIRE(minimum_bits == failure_bits);
    const double expected_correction =
        std::ldexp(11.0, -149) * 1.25 * dx_m * dy_m * 60.0
        + std::ldexp(5.0, -149) * static_cast<double>(0.8f)
            * dx_m * dy_m * 80.0;
    REQUIRE(report.physical_inventory_correction_kg
            == expected_correction);
    REQUIRE(report.thermodynamic_energy_correction_j == 0.0);
    REQUIRE(runtime.cumulative_tracer_roundoff_report.complete);
    REQUIRE(runtime.cumulative_tracer_roundoff_report
                .physical_inventory_correction_kg == expected_correction);
    REQUIRE(runtime.tracer_roundoff_report_step_count == 1);
    REQUIRE(runtime.tracer_roundoff_cumulative_history_known);

    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("active donor ratios stay strict when FP64 values round to FP32 one",
          "[numerics][advection][cartesian][positive-definite][roundoff]")
{
    constexpr double captured_active_ratio = 0.99999998652141264;
    REQUIRE(captured_active_ratio < 1.0);
    REQUIRE(static_cast<float>(captured_active_ratio) == 1.0f);

    const float stored = tmv::cartesian_advection_detail::
        conservative_pd_limiter_factor(captured_active_ratio);
    constexpr float roundoff_factor =
        1.0f - 32.0f * std::numeric_limits<float>::epsilon();
    REQUIRE(stored == std::nextafter(roundoff_factor, 0.0f));
    REQUIRE(stored < 1.0f);
    REQUIRE(tmv::cartesian_advection_detail::
                conservative_pd_limiter_factor(1.0) == 1.0f);
}

TEST_CASE("passive tracer roundoff transaction is thread exact",
          "[numerics][advection][cartesian][cgrid][tracer][roundoff][deterministic]")
{
    setup_cartesian_grid(8, 7, 5, 90.0, 70.0, 40.0, 0.75);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 1.1f);
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#endif

    const auto run = [](int threads)
    {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        tracer.resize(NR, NTH, NZ, 0.0f);
        tracer[2][2][1] =
            -11.0f * std::numeric_limits<float>::denorm_min();
        tracer[4][3][2] =
            -5.0f * std::numeric_limits<float>::denorm_min();
        tracer[0][1][1] =
            -3.0f * std::numeric_limits<float>::denorm_min();
        AdvectionRuntimeState runtime;
        runtime.tracer_advected_in_batch = true;
        advect_tracer_3d(0.75, 0.0, nullptr, &runtime);
        return std::make_pair(tracer, runtime.tracer_roundoff_report);
    };

    const auto serial = run(1);
    const auto parallel = run(4);
    const auto campaign_threads = run(12);
    const auto require_same = [&](const auto& candidate)
    {
        REQUIRE(serial.first.size() == candidate.first.size());
        REQUIRE(std::memcmp(serial.first.data(), candidate.first.data(),
                            serial.first.size() * sizeof(float)) == 0);
        REQUIRE(serial.second.physical_inventory_correction_kg
                == candidate.second.physical_inventory_correction_kg);
        REQUIRE(serial.second.negative_subnormal_count
                == candidate.second.negative_subnormal_count);
        REQUIRE(serial.second.closure_canonicalization_count
                == candidate.second.closure_canonicalization_count);
        REQUIRE(serial.second.most_negative_repaired_value
                == candidate.second.most_negative_repaired_value);
    };
    require_same(parallel);
    require_same(campaign_threads);
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#endif
    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("passive tracer publishes a complete zero-repair step",
          "[numerics][advection][cartesian][cgrid][tracer][roundoff][ledger]")
{
    setup_cartesian_grid(6, 5, 4, 100.0, 80.0, 50.0, 0.75);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 1.0f);
    tracer.resize(NR, NTH, NZ, 0.25f);
    tracer[2][2][1] =
        5.0f * std::numeric_limits<float>::denorm_min();
    const Field3D before = tracer;
    AdvectionRuntimeState runtime;
    runtime.tracer_advected_in_batch = true;
    advect_tracer_3d(0.75, 0.0, nullptr, &runtime);
    REQUIRE(std::memcmp(before.data(), tracer.data(),
                        tracer.size() * sizeof(float)) == 0);
    REQUIRE(runtime.tracer_roundoff_report.complete);
    REQUIRE(runtime.tracer_roundoff_report
                .physical_inventory_correction_kg == 0.0);
    REQUIRE(runtime.tracer_roundoff_report.negative_subnormal_count == 0);
    REQUIRE(runtime.tracer_roundoff_report.negative_zero_count == 0);
    REQUIRE(runtime.cumulative_tracer_roundoff_report.complete);
    REQUIRE(runtime.tracer_roundoff_report_step_count == 1);
    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("passive tracer material violations fail before repair publication",
          "[numerics][advection][cartesian][cgrid][tracer][roundoff][fail-closed]")
{
    setup_cartesian_grid(6, 5, 4, 100.0, 80.0, 50.0, 0.75);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 1.0f);
    const std::array<float, 3> violations = {
        -std::numeric_limits<float>::min(),
        std::numeric_limits<float>::infinity(),
        std::nextafter(passive_tracer_max,
                       std::numeric_limits<float>::infinity()),
    };
    for (float violation : violations)
    {
        tracer.resize(NR, NTH, NZ, 0.0f);
        tracer[2][2][1] =
            -11.0f * std::numeric_limits<float>::denorm_min();
        tracer[3][2][1] = violation;
        const Field3D before = tracer;
        AdvectionRuntimeState runtime;
        runtime.tracer_advected_in_batch = true;
        runtime.tracer_roundoff_report = {
            1.0, 0.0,
            -std::numeric_limits<float>::denorm_min(),
            1, 0, 0, true};
        runtime.cumulative_tracer_roundoff_report =
            runtime.tracer_roundoff_report;
        runtime.tracer_roundoff_report_step_count = 1;
        const auto prior = runtime.tracer_roundoff_report;
        REQUIRE_THROWS_WITH(
            advect_tracer_3d(0.75, 0.0, nullptr, &runtime),
            Catch::Matchers::Contains("material contract violation"));
        REQUIRE(std::memcmp(before.data(), tracer.data(),
                            tracer.size() * sizeof(float)) == 0);
        REQUIRE(runtime.tracer_advected_in_batch);
        REQUIRE(runtime.tracer_roundoff_report
                    .physical_inventory_correction_kg
                == prior.physical_inventory_correction_kg);
        REQUIRE(runtime.tracer_roundoff_report
                    .negative_subnormal_count
                == prior.negative_subnormal_count);
        REQUIRE(runtime.tracer_roundoff_report.complete == prior.complete);
        REQUIRE(runtime.tracer_roundoff_report_step_count == 1);
    }

    tracer.resize(NR, NTH, NZ, 0.0f);
    tracer[2][2][1] = std::nextafter(
        passive_tracer_max, std::numeric_limits<float>::infinity());
    const Field3D standalone_before = tracer;
    AdvectionRuntimeState standalone;
    standalone.tracer_roundoff_report = {
        2.0, 0.0, -std::numeric_limits<float>::denorm_min(),
        1, 0, 0, true};
    standalone.cumulative_tracer_roundoff_report =
        standalone.tracer_roundoff_report;
    standalone.tracer_roundoff_report_step_count = 1;
    REQUIRE_THROWS_WITH(
        advect_tracer_3d(0.75, 0.0, nullptr, &standalone),
        Catch::Matchers::Contains("requires finite, in-range passive tracer"));
    REQUIRE(std::memcmp(standalone_before.data(), tracer.data(),
                        tracer.size() * sizeof(float)) == 0);
    REQUIRE(standalone.tracer_roundoff_report
                .physical_inventory_correction_kg == 2.0);
    REQUIRE(standalone.tracer_roundoff_report_step_count == 1);
    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("passive tracer rejects malformed caller ledger transactionally",
          "[numerics][advection][cartesian][cgrid][tracer][roundoff][fail-closed]")
{
    setup_cartesian_grid(6, 5, 4, 100.0, 80.0, 50.0, 0.75);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 1.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    tracer[2][2][1] =
        -11.0f * std::numeric_limits<float>::denorm_min();
    const Field3D before = tracer;

    AdvectionRuntimeState runtime;
    runtime.tracer_advected_in_batch = true;
    runtime.tracer_roundoff_report_step_count = 1;
    runtime.cumulative_tracer_roundoff_report = {
        1.0, 0.0, 0.0f, 0, 0, 0, true};
    const auto prior = runtime.cumulative_tracer_roundoff_report;
    REQUIRE_THROWS_WITH(
        advect_tracer_3d(0.75, 0.0, nullptr, &runtime),
        Catch::Matchers::Contains("malformed caller-owned ledger"));
    REQUIRE(std::memcmp(before.data(), tracer.data(),
                        tracer.size() * sizeof(float)) == 0);
    REQUIRE(runtime.tracer_advected_in_batch);
    REQUIRE(runtime.tracer_roundoff_report_step_count == 1);
    REQUIRE(runtime.cumulative_tracer_roundoff_report
                .physical_inventory_correction_kg
            == prior.physical_inventory_correction_kg);
    REQUIRE(runtime.cumulative_tracer_roundoff_report
                .negative_subnormal_count == 0);
    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("passive tracer preserves unknown cumulative history after legacy resume",
          "[numerics][advection][cartesian][cgrid][tracer][roundoff][restart]")
{
    setup_cartesian_grid(6, 5, 4, 100.0, 80.0, 50.0, 0.75);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 1.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    tracer[2][2][1] =
        -11.0f * std::numeric_limits<float>::denorm_min();
    AdvectionRuntimeState runtime;
    runtime.tracer_roundoff_cumulative_history_known = false;
    runtime.tracer_advected_in_batch = true;
    advect_tracer_3d(0.75, 0.0, nullptr, &runtime);
    REQUIRE(runtime.tracer_roundoff_report.complete);
    REQUIRE_FALSE(runtime.cumulative_tracer_roundoff_report.complete);
    REQUIRE_FALSE(runtime.tracer_roundoff_cumulative_history_known);
    REQUIRE(runtime.tracer_roundoff_report_step_count == 1);
    REQUIRE(runtime.cumulative_tracer_roundoff_report
                .negative_subnormal_count == 1);
    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("Thermodynamic advection consumes caller-owned hydrometeor activity",
          "[numerics][advection][cartesian][cpu022][activity]")
{
    setup_cartesian_grid(
        /*nr=*/12, /*nth=*/10, /*nz=*/8,
        /*dx=*/100.0, /*dy=*/120.0, /*dz=*/80.0, /*dt=*/0.5);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 1.0f);
    theta.resize(NR, NTH, NZ, 300.0f);
    qv.resize(NR, NTH, NZ, 0.01f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    set_uniform_velocity(0.0, 0.0, 0.0);
    qc[4][3][2] = 5.0e-13f;
    qr[5][4][3] = 1.0e-4f;

    AdvectionRuntimeState runtime;
    advect_thermodynamics_3d(
        /*dt=*/0.5, /*kappa_theta=*/0.0, /*kappa_moisture=*/0.0,
        /*context=*/nullptr, &runtime);
    REQUIRE(runtime.fallback_scans == 1);
    REQUIRE(runtime.metadata_consumptions == 0);
    REQUIRE(runtime.hydrometeor_active[0]);
    REQUIRE(runtime.hydrometeor_active[1]);
    REQUIRE_FALSE(runtime.activity_current);

    runtime.publish_hydrometeor_activity(
        {true, true, false, false, false, false}, NR, NTH, NZ);
    advect_thermodynamics_3d(
        /*dt=*/0.5, /*kappa_theta=*/0.0, /*kappa_moisture=*/0.0,
        /*context=*/nullptr, &runtime);
    REQUIRE(runtime.fallback_scans == 1);
    REQUIRE(runtime.metadata_consumptions == 1);
    REQUIRE_FALSE(runtime.activity_current);

    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("CPU thermodynamic batch publishes every scalar boundary ledger",
          "[numerics][advection][cartesian][boundary-ledger][batch]")
{
    setup_cartesian_grid(
        /*nr=*/8, /*nth=*/7, /*nz=*/6,
        /*dx=*/100.0, /*dy=*/120.0, /*dz=*/80.0, /*dt=*/0.5);
    global_stagger_type = StaggerType::CGrid;
    global_lateral_bc_west = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_east = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_south = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_north = tmv::init::config::LateralBC::RigidSlip;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
    rho.resize(NR, NTH, NZ, 1.0f);
    p.resize(NR, NTH, NZ, 90000.0f);
    theta.resize(NR, NTH, NZ, 300.0f);
    qv.resize(NR, NTH, NZ, 0.01f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    set_uniform_velocity(0.0, 0.0, 0.0);
    qc[3][3][2] = 5.0e-13f;

    CartesianCGridTransportContext context;
    context.begin(NR, NTH, NZ, 0.5);
    context.accumulate(rho, u, v, w, 0.5);
    context.complete();

    AdvectionRuntimeState runtime;
    runtime.thermodynamic_boundary_reports_requested = true;
    runtime.thermodynamic_moist_energy_reports_requested = true;
    advect_thermodynamics_3d(
        0.5, 0.0, 0.0, &context, &runtime);

    REQUIRE(runtime.thermodynamic_boundary_reports_complete);
    REQUIRE(runtime.thermodynamic_moist_energy_reports_complete);
    for (std::size_t index = 0;
         index < runtime.thermodynamic_boundary_reports.size(); ++index)
    {
        INFO("report index=" << index);
        const auto& report =
            runtime.thermodynamic_boundary_reports[index];
        REQUIRE(report.complete);
        REQUIRE(report.outward_boundary_inventory == 0.0);
        REQUIRE(report.flux_implied_state_change == 0.0);
        REQUIRE(report.committed_state_change == 0.0);
        REQUIRE(report.closure_residual == 0.0);
        const auto& energy =
            runtime.thermodynamic_moist_energy_reports[index];
        REQUIRE(energy.complete);
        REQUIRE(energy.committed_state_change_j == 0.0);
        REQUIRE(energy.requested_state_change_j == 0.0);
        REQUIRE(energy.outward_boundary_energy_j == 0.0);
        REQUIRE(energy.closure_residual_j == 0.0);
    }
    REQUIRE(runtime.hydrometeor_active[0]);
    REQUIRE(qc[3][3][2] == 5.0e-13f);

    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("C-grid limited vapor commit closes moist-energy basis v2 at operator time",
          "[numerics][advection][cartesian][energy-ledger][SCI-148]")
{
    const bool large = GENERATE(false, true);
    const int nr = large ? 44 : 9, nth = large ? 43 : 8, nz = large ? 40 : 6;
    const double dt_s = 0.2;
    setup_cartesian_grid(
        nr, nth, nz, /*dx=*/100.0, /*dy=*/120.0, /*dz=*/80.0, dt_s);
    global_stagger_type = StaggerType::CGrid;
    global_grid_geometry.initialize(
        nr, nth, nz, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    global_lateral_bc_west = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_east = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_south = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_north = tmv::init::config::LateralBC::RigidSlip;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;

    rho.resize(nr, nth, nz, 0.0f);
    p.resize(nr, nth, nz, 0.0f);
    theta.resize(nr, nth, nz, 0.0f);
    qv.resize(nr, nth, nz, 0.0f);
    qc.resize(nr, nth, nz, 0.0f);
    qr.resize(nr, nth, nz, 0.0f);
    qi.resize(nr, nth, nz, 0.0f);
    qs.resize(nr, nth, nz, 0.0f);
    qg.resize(nr, nth, nz, 0.0f);
    qh.resize(nr, nth, nz, 0.0f);
    u.resize(nr, nth, nz, 0.0f);
    v.resize(nr, nth, nz, 0.0f);
    w.resize(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                rho[i][j][k] = static_cast<float>(
                    1.08 + 0.004 * i - 0.002 * j + 0.001 * k);
                p[i][j][k] = static_cast<float>(
                    92000.0 - 650.0 * k + 17.0 * i);
                theta[i][j][k] = static_cast<float>(
                    296.0 + 0.3 * k + 0.04 * j);
                qv[i][j][k] = static_cast<float>(
                    0.004 + 0.00015 * i + 0.00008 * j
                    + 0.00004 * k);
                if (i >= 1 && i < nr - 2)
                    u[i][j][k] = static_cast<float>(
                        3.0 + 0.12 * j - 0.05 * k);
                if (j >= 1 && j < nth - 2)
                    v[i][j][k] = static_cast<float>(
                        -1.5 + 0.08 * i + 0.03 * k);
                if (k < nz - 1)
                    w[i][j][k] = static_cast<float>(
                        0.06 * std::sin(0.4 * (i + j + k)));
            }

    CartesianCGridTransportContext context;
    context.begin(nr, nth, nz, dt_s);
    context.accumulate(rho, u, v, w, dt_s);
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double mx =
                    static_cast<double>(context.mass_flux_x[i][j][k])
                    - context.mass_flux_x[i - 1][j][k];
                const double my =
                    static_cast<double>(context.mass_flux_y[i][j][k])
                    - context.mass_flux_y[i][j - 1][k];
                const double mz_top = k <= nz - 2
                    ? static_cast<double>(context.mass_flux_z[i][j][k])
                    : 0.0;
                const double mz_bottom = k >= 1
                    ? static_cast<double>(
                          context.mass_flux_z[i][j][k - 1])
                    : 0.0;
                const double divergence =
                    mx / dr + my / dtheta + (mz_top - mz_bottom) / dz;
                rho[i][j][k] = static_cast<float>(
                    static_cast<double>(rho[i][j][k])
                    - dt_s * divergence);
            }
    context.complete();

    const Field3D qv_before = qv;
    const Field3D rho_end = rho;
    const auto vapor_reservoirs =
        [&](const Field3D& vapor)
    {
        std::array<long double, 4> sums{};
        constexpr double cv_v =
            physical_constants::specific_heat_water_vapor_cp_jkgk
            - physical_constants::gas_constant_water_vapor_jkgk;
        constexpr double vapor_reference =
            physical_constants::latent_heat_vaporization_jkg
            - physical_constants::gas_constant_water_vapor_jkgk
                * physical_constants::freezing_temperature_k;
        constexpr double kappa =
            physical_constants::gas_constant_dry_air_jkgk
            / physical_constants::specific_heat_cp_jkgk;
        const long double volume =
            static_cast<long double>(dr) * dtheta * dz;
        for (int i = 1; i < nr - 1; ++i)
            for (int j = 1; j < nth - 1; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const double temperature =
                        static_cast<double>(theta[i][j][k])
                        * std::pow(
                            static_cast<double>(p[i][j][k])
                                / physical_constants::reference_pressure_pa,
                            kappa);
                    const double u_w = static_cast<double>(u[i - 1][j][k]);
                    const double u_e = static_cast<double>(u[i][j][k]);
                    const double v_s = static_cast<double>(v[i][j - 1][k]);
                    const double v_n = static_cast<double>(v[i][j][k]);
                    const double w_b = k == 0
                        ? 0.0 : static_cast<double>(w[i][j][k - 1]);
                    const double w_t = static_cast<double>(w[i][j][k]);
                    const double kinetic_specific = 0.25
                        * (u_w * u_w + u_e * u_e
                           + v_s * v_s + v_n * v_n
                           + w_b * w_b + w_t * w_t);
                    // Difference the stored species first: subtracting whole
                    // domain inventories loses small changes on binary64 long double.
                    const long double species_mass = volume
                        * static_cast<long double>(rho_end[i][j][k])
                        * (static_cast<long double>(vapor[i][j][k])
                           - static_cast<long double>(qv_before[i][j][k]));
                    sums[0] += species_mass * cv_v
                        * (temperature
                           - physical_constants::freezing_temperature_k);
                    sums[1] += species_mass * vapor_reference;
                    sums[2] += species_mass * kinetic_specific;
                    sums[3] += species_mass
                        * physical_constants::gravity_ms2
                        * global_grid_geometry.z[static_cast<std::size_t>(k)];
                }
        return sums;
    };

    ScalarBoundaryTransportReport boundary_report;
    ScalarMoistEnergyTransportReportV1 energy_report;
#ifdef _OPENMP
    const int original_thread_count = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    advect_scalar_conservative_3d_cartesian(
        qv, qv, dt_s, /*positive_definite=*/true, &context,
        &boundary_report, ScalarMoistEnergyKind::WaterVapor, &energy_report);
    const auto after = vapor_reservoirs(qv);

    REQUIRE(energy_report.complete);
    REQUIRE(
        ScalarMoistEnergyTransportReportV1::report_schema_version == 1);
    REQUIRE(
        ScalarMoistEnergyTransportReportV1::energy_basis_version == 2);
    REQUIRE(energy_report.dry_internal_change_j == 0.0);
    REQUIRE(energy_report.subgrid_tke_change_j == 0.0);
    REQUIRE(energy_report.water_sensible_change_j
            == Approx(static_cast<double>(after[0]))
                   .epsilon(1.0e-11));
    REQUIRE(energy_report.phase_reference_change_j
            == Approx(static_cast<double>(after[1]))
                   .epsilon(1.0e-11));
    REQUIRE(energy_report.carrier_kinetic_change_j
            == Approx(static_cast<double>(after[2]))
                   .epsilon(1.0e-11));
    REQUIRE(energy_report.gravitational_change_j
            == Approx(static_cast<double>(after[3]))
                   .epsilon(1.0e-11));
    REQUIRE(energy_report.outward_boundary_energy_j == 0.0);
    REQUIRE(energy_report.interior_conversion_j
            == energy_report.limited_face_state_change_j);
    REQUIRE(energy_report.requested_state_change_j
            == energy_report.density_history_remap_j
                + energy_report.limited_face_state_change_j);
    REQUIRE(energy_report.committed_state_change_j
            == energy_report.water_sensible_change_j
                + energy_report.phase_reference_change_j
                + energy_report.carrier_kinetic_change_j
                + energy_report.gravitational_change_j);
    REQUIRE(energy_report.storage_rounding_j
            == energy_report.committed_state_change_j
                - energy_report.requested_state_change_j);
    REQUIRE(energy_report.closure_residual_j == 0.0);
    REQUIRE(boundary_report.complete);

    const Field3D one_thread_result = qv;
    const ScalarMoistEnergyTransportReportV1 one_thread_report =
        energy_report;
    qv = qv_before;
    rho = rho_end;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    ScalarMoistEnergyTransportReportV1 many_thread_report;
    advect_scalar_conservative_3d_cartesian(
        qv, qv, dt_s, /*positive_definite=*/true, &context,
        nullptr, ScalarMoistEnergyKind::WaterVapor, &many_thread_report);
    REQUIRE(std::equal(
        qv.data(), qv.data() + qv.size(), one_thread_result.data()));
    REQUIRE(many_thread_report.water_sensible_change_j
            == one_thread_report.water_sensible_change_j);
    REQUIRE(many_thread_report.phase_reference_change_j
            == one_thread_report.phase_reference_change_j);
    REQUIRE(many_thread_report.carrier_kinetic_change_j
            == one_thread_report.carrier_kinetic_change_j);
    REQUIRE(many_thread_report.gravitational_change_j
            == one_thread_report.gravitational_change_j);
    REQUIRE(many_thread_report.density_history_remap_j
            == one_thread_report.density_history_remap_j);
    REQUIRE(many_thread_report.limited_face_state_change_j
            == one_thread_report.limited_face_state_change_j);
    REQUIRE(many_thread_report.storage_rounding_j
            == one_thread_report.storage_rounding_j);
#ifdef _OPENMP
    omp_set_num_threads(original_thread_count);
#endif
}

TEST_CASE("scalar energy preflight rejects invalid late cells before mutating either destination layout",
          "[numerics][advection][cartesian][energy-ledger][validation]")
{
    const bool large = GENERATE(false, true);
    const int nr = large ? 44 : 9, nth = large ? 43 : 8, nz = large ? 40 : 6;
    setup_cartesian_grid(nr, nth, nz, 100.0, 120.0, 80.0, 0.2);
    for (const auto staggering : {StaggerType::CGrid, StaggerType::Collocated})
    {
        global_stagger_type = staggering;
        global_grid_geometry.initialize(nr, nth, nz, dr, dz, dtheta,
                                        CoordinateSystem::Cartesian, staggering);
        rho.resize(nr, nth, nz, 1.0f);
        p.resize(nr, nth, nz, 90000.0f);
        theta.resize(nr, nth, nz, 300.0f);
        qv.resize(nr, nth, nz, 0.005f);
        for (auto* field : {&qc, &qr, &qi, &qs, &qg, &qh})
            field->resize(nr, nth, nz, 0.0f);
        set_uniform_velocity(0.0, 0.0, 0.0);
        CartesianCGridTransportContext context;
        context.begin(nr, nth, nz, 0.2);
        const std::vector<double> base_pressure(static_cast<std::size_t>(nz), 90000.0);
        const tmv::cartesian_mass_flux::CollocatedPressureCoupling pressure_coupling{
            &p, &base_pressure, 0.2, dr, dtheta, dz};
        context.accumulate(rho, u, v, w, 0.2,
                           staggering == StaggerType::Collocated ? &pressure_coupling : nullptr);
        context.complete();
        const int k = staggering == StaggerType::CGrid ? nz - 1 : nz - 2;
        for (const int threads : {1, 4})
        {
#ifdef _OPENMP
            const int previous_threads = omp_get_max_threads();
            omp_set_num_threads(threads);
#endif
            for (const auto kind : {ScalarMoistEnergyKind::PotentialTemperature,
                                    ScalarMoistEnergyKind::WaterVapor,
                                    ScalarMoistEnergyKind::LiquidWater,
                                    ScalarMoistEnergyKind::IceWater})
                for (const bool in_place : {false, true})
                    for (const float invalid : {-1.0f, std::numeric_limits<float>::quiet_NaN(),
                                                 std::numeric_limits<float>::infinity()})
                    {
                        CAPTURE(large, threads, in_place, invalid);
                        p[nr - 2][nth - 2][k] = invalid;
                        Field3D source(nr, nth, nz, 0.01f);
                        Field3D destination(nr, nth, nz, -17.0f);
                        const Field3D saved_source = source, saved_destination = destination;
                        ScalarMoistEnergyTransportReportV1 report;
                        report.complete = true;
                        REQUIRE_THROWS_WITH(
                            advect_scalar_conservative_3d_cartesian(
                                source, in_place ? source : destination, 0.2, true, &context,
                                nullptr, kind, &report),
                            Catch::Matchers::Contains("non-finite operator-time coefficient"));
                        REQUIRE_FALSE(report.complete);
                        REQUIRE(std::memcmp(source.data(), saved_source.data(), source.size() * sizeof(float)) == 0);
                        REQUIRE(std::memcmp(destination.data(), saved_destination.data(), destination.size() * sizeof(float)) == 0);
                        p[nr - 2][nth - 2][k] = 90000.0f;
                    }
#ifdef _OPENMP
            omp_set_num_threads(previous_threads);
#endif
        }
    }
    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("flat C-grid device scalar attribution preserves FP32 commit currency",
          "[numerics][advection][cartesian][device][energy-ledger][SCI-159]")
{
    const int nr = 8, nth = 7, nz = 5;
    constexpr double dt_s = 0.25;
    setup_cartesian_grid(
        nr, nth, nz, /*dx=*/90.0, /*dy=*/110.0, /*dz=*/70.0, dt_s);
    global_stagger_type = StaggerType::CGrid;
    global_grid_geometry.initialize(
        nr, nth, nz, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::CGrid);

    Field3D density_begin(nr, nth, nz, 1.1f);
    Field3D density_end = density_begin;
    Field3D mass_flux_x(nr, nth, nz, 0.0f);
    Field3D mass_flux_y(nr, nth, nz, 0.0f);
    Field3D mass_flux_z(nr, nth, nz, 0.0f);
    Field3D scalar_begin(nr, nth, nz, 0.0f);
    Field3D scalar_end(nr, nth, nz, 0.0f);

    rho = density_end;
    p.resize(nr, nth, nz, 90000.0f);
    theta.resize(nr, nth, nz, 300.0f);
    qv.resize(nr, nth, nz, 0.0f);
    qc.resize(nr, nth, nz, 0.0f);
    qr.resize(nr, nth, nz, 0.0f);
    qi.resize(nr, nth, nz, 0.0f);
    qs.resize(nr, nth, nz, 0.0f);
    qg.resize(nr, nth, nz, 0.0f);
    qh.resize(nr, nth, nz, 0.0f);
    u.resize(nr, nth, nz, 2.0f);
    v.resize(nr, nth, nz, -1.0f);
    w.resize(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const float vapor = static_cast<float>(
                    0.004 + 0.0001 * i + 0.00003 * j + 0.00002 * k);
                scalar_begin[i][j][k] = vapor;
                scalar_end[i][j][k] = vapor;
                qv[i][j][k] = vapor;
            }

    // With zero face flux the only requested term is the device's own FP32
    // rho*q product roundoff. A one-ULP committed perturbation must remain
    // visible in storage rounding rather than being hidden in a boundary
    // term.
    scalar_end[3][3][2] =
        std::nextafter(scalar_end[3][3][2], 1.0f);

    FlatCGridDeviceScalarAttributionWorkspace workspace;
    ScalarBoundaryTransportReport boundary;
    ScalarMoistEnergyTransportReportV1 energy;
#ifdef _OPENMP
    const int original_thread_count = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    attribute_flat_cgrid_device_scalar_commit(
        scalar_begin.data(), scalar_end.data(),
        density_begin.data(), density_end.data(),
        mass_flux_x.data(), mass_flux_y.data(), mass_flux_z.data(),
        nr, nth, nz,
        static_cast<float>(1.0 / dr),
        static_cast<float>(1.0 / dtheta),
        static_cast<float>(1.0 / dz),
        static_cast<float>(dt_s),
        /*positive_definite=*/true,
        &boundary, ScalarMoistEnergyKind::WaterVapor, &energy,
        workspace);

    REQUIRE(boundary.complete);
    REQUIRE(boundary.outward_boundary_inventory == 0.0);
    REQUIRE(boundary.flux_implied_state_change == 0.0);
    REQUIRE(boundary.committed_state_change > 0.0);
    REQUIRE(boundary.closure_residual
            == boundary.committed_state_change);
    REQUIRE(energy.complete);
    REQUIRE(energy.density_history_remap_j == 0.0);
    REQUIRE(energy.limited_face_state_change_j != 0.0);
    REQUIRE(energy.requested_state_change_j
            == energy.limited_face_state_change_j);
    REQUIRE(energy.outward_boundary_energy_j == 0.0);
    REQUIRE(energy.interior_conversion_j
            == energy.limited_face_state_change_j);
    REQUIRE(energy.committed_state_change_j > 0.0);
    REQUIRE(energy.storage_rounding_j
            == energy.committed_state_change_j
                - energy.requested_state_change_j);
    const double closure_scale = std::max({
        1.0,
        std::abs(energy.committed_state_change_j),
        std::abs(energy.requested_state_change_j),
        std::abs(energy.storage_rounding_j),
    });
    REQUIRE(std::abs(energy.closure_residual_j)
            <= 8.0 * std::numeric_limits<double>::epsilon()
                * closure_scale);

    const float* flux_x_storage = workspace.scalar_flux_x.data();
    const float* flux_y_storage = workspace.scalar_flux_y.data();
    const float* flux_z_storage = workspace.scalar_flux_z.data();
    const float* limiter_storage = workspace.donor_limiter.data();
    const ScalarBoundaryTransportReport one_thread_boundary = boundary;
    const ScalarMoistEnergyTransportReportV1 one_thread_energy = energy;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    attribute_flat_cgrid_device_scalar_commit(
        scalar_begin.data(), scalar_end.data(),
        density_begin.data(), density_end.data(),
        mass_flux_x.data(), mass_flux_y.data(), mass_flux_z.data(),
        nr, nth, nz,
        static_cast<float>(1.0 / dr),
        static_cast<float>(1.0 / dtheta),
        static_cast<float>(1.0 / dz),
        static_cast<float>(dt_s),
        /*positive_definite=*/true,
        &boundary, ScalarMoistEnergyKind::WaterVapor, &energy,
        workspace);
    REQUIRE(workspace.scalar_flux_x.data() == flux_x_storage);
    REQUIRE(workspace.scalar_flux_y.data() == flux_y_storage);
    REQUIRE(workspace.scalar_flux_z.data() == flux_z_storage);
    REQUIRE(workspace.donor_limiter.data() == limiter_storage);
    REQUIRE(boundary.committed_state_change
            == one_thread_boundary.committed_state_change);
    REQUIRE(boundary.closure_residual
            == one_thread_boundary.closure_residual);
    REQUIRE(energy.water_sensible_change_j
            == one_thread_energy.water_sensible_change_j);
    REQUIRE(energy.phase_reference_change_j
            == one_thread_energy.phase_reference_change_j);
    REQUIRE(energy.carrier_kinetic_change_j
            == one_thread_energy.carrier_kinetic_change_j);
    REQUIRE(energy.gravitational_change_j
            == one_thread_energy.gravitational_change_j);
    REQUIRE(energy.storage_rounding_j
            == one_thread_energy.storage_rounding_j);

    constexpr float uniform_scalar = 0.01f;
    constexpr float top_mass_flux = 0.2f;
    std::fill(
        scalar_begin.data(), scalar_begin.data() + scalar_begin.size(),
        uniform_scalar);
    scalar_end = scalar_begin;
    std::fill(
        density_begin.data(),
        density_begin.data() + density_begin.size(), 1.1f);
    density_end = density_begin;
    std::fill(
        mass_flux_x.data(), mass_flux_x.data() + mass_flux_x.size(), 0.0f);
    std::fill(
        mass_flux_y.data(), mass_flux_y.data() + mass_flux_y.size(), 0.0f);
    std::fill(
        mass_flux_z.data(), mass_flux_z.data() + mass_flux_z.size(), 0.0f);
    const float inv_dz = static_cast<float>(1.0 / dz);
    const float dt_device = static_cast<float>(dt_s);
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
        {
            const int k = nz - 1;
            mass_flux_z[i][j][k] = top_mass_flux;
            const float density_tendency =
                -(top_mass_flux * inv_dz);
            density_end[i][j][k] =
                density_begin[i][j][k]
                + density_tendency * dt_device;
            const float scalar_flux = top_mass_flux * uniform_scalar;
            const float outgoing = scalar_flux * inv_dz;
            const float requested =
                density_begin[i][j][k] * uniform_scalar
                - dt_device * outgoing;
            scalar_end[i][j][k] =
                requested / density_end[i][j][k];
        }
    rho = density_end;

    ScalarBoundaryTransportReport open_top_boundary;
    attribute_flat_cgrid_device_scalar_commit(
        scalar_begin.data(), scalar_end.data(),
        density_begin.data(), density_end.data(),
        mass_flux_x.data(), mass_flux_y.data(), mass_flux_z.data(),
        nr, nth, nz,
        static_cast<float>(1.0 / dr),
        static_cast<float>(1.0 / dtheta),
        inv_dz, dt_device,
        /*positive_definite=*/true,
        &open_top_boundary, ScalarMoistEnergyKind::None, nullptr,
        workspace);
    const double expected_top_outward =
        static_cast<double>(dt_device)
        * static_cast<double>((nr - 2) * (nth - 2))
        * dr * dtheta
        * static_cast<double>(top_mass_flux * uniform_scalar);
    REQUIRE(open_top_boundary.complete);
    REQUIRE(open_top_boundary.outward_boundary_inventory
            == Approx(expected_top_outward).epsilon(2.0e-7));
    REQUIRE(open_top_boundary.flux_implied_state_change
            == -open_top_boundary.outward_boundary_inventory);
    REQUIRE(open_top_boundary.committed_state_change
            == Approx(-expected_top_outward).epsilon(2.0e-4));
    REQUIRE(std::abs(open_top_boundary.closure_residual)
            <= 2.0e-4 * expected_top_outward);

    REQUIRE_THROWS_WITH(
        attribute_flat_cgrid_device_scalar_commit(
            scalar_begin.data(), scalar_end.data(),
            density_begin.data(), density_end.data(),
            mass_flux_x.data(), mass_flux_y.data(), mass_flux_z.data(),
            nr, nth, nz,
            static_cast<float>(1.0 / dr),
            static_cast<float>(1.0 / dtheta),
            static_cast<float>(1.0 / dz),
            static_cast<float>(dt_s),
            /*positive_definite=*/true,
            nullptr, ScalarMoistEnergyKind::WaterVapor, nullptr,
            workspace),
        Catch::Matchers::Contains("partial energy descriptor"));
#ifdef _OPENMP
    omp_set_num_threads(original_thread_count);
#endif
}

TEST_CASE("Thermodynamic batch rejects corrupt inactive species before mutation",
          "[numerics][advection][cartesian][integrity][transactional]")
{
    setup_cartesian_grid(
        /*nr=*/8, /*nth=*/7, /*nz=*/6,
        /*dx=*/100.0, /*dy=*/120.0, /*dz=*/80.0, /*dt=*/0.5);
    global_stagger_type = StaggerType::CGrid;
    rho.resize(NR, NTH, NZ, 1.0f);
    theta.resize(NR, NTH, NZ, 300.0f);
    qv.resize(NR, NTH, NZ, 0.01f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    set_uniform_velocity(0.0, 0.0, 0.0);

    const Field3D theta_before = theta;
    qg[0][0][0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        advect_thermodynamics_3d(
            0.5, 0.0, 0.0, nullptr, nullptr),
        std::invalid_argument);
    REQUIRE(std::equal(theta.data(), theta.data() + theta.size(),
                       theta_before.data()));

    qg[0][0][0] = 0.0f;
    rho[0][0][0] = 0.5f * density_min_kgm3;
    REQUIRE_THROWS_AS(
        advect_thermodynamics_3d(
            0.5, 0.0, 0.0, nullptr, nullptr),
        std::invalid_argument);

    global_stagger_type = StaggerType::Collocated;
}

TEST_CASE("advection runtime owns dispatch and reusable storage",
          "[numerics][advection][state][modularity]")
{
    REQUIRE(&advection_scheme == &global_advection_runtime_state.scheme);
    REQUIRE(&global_advection_config == &global_advection_runtime_state.config);

    AdvectionRuntimeState original;
    original.config.scheme_id = "tvd";
    original.scheme = create_advection_scheme("tvd");
    original.scheme->initialize(original.config);
    original.scalar_scratch_a.resize(2, 1, 1, 3.0f);
    original.vertical_tendencies.dqdt_adv.resize(2, 1, 1, -1.0f);
    original.vertical_tendencies.fused_dst =
        original.scalar_scratch_a.data();
    original.tracer_advected_in_batch = true;

    AdvectionRuntimeState copy(original);
    REQUIRE(copy.scheme != nullptr);
    REQUIRE(copy.scheme.get() != original.scheme.get());
    REQUIRE(copy.scheme->name() == "tvd");
    REQUIRE(copy.scalar_scratch_a.data() != original.scalar_scratch_a.data());
    REQUIRE(copy.vertical_tendencies.dqdt_adv.data() !=
            original.vertical_tendencies.dqdt_adv.data());
    REQUIRE(copy.vertical_tendencies.fused_dst == nullptr);
    REQUIRE(copy.tracer_advected_in_batch);
}

TEST_CASE("interleaved advection runtimes keep tracer handshakes independent",
          "[numerics][advection][state][modularity]")
{
    setup_cartesian_grid(
        /*nr=*/4, /*nth=*/4, /*nz=*/4,
        /*dx=*/100.0, /*dy=*/100.0, /*dz=*/100.0, /*dt=*/0.5);
    rho.resize(NR, NTH, NZ, 1.0f);
    tracer.resize(NR, NTH, NZ, 1.0f);

    AdvectionRuntimeState first;
    AdvectionRuntimeState second;
    first.tracer_advected_in_batch = true;
    second.tracer_advected_in_batch = true;

    advect_tracer_3d(0.5, 0.0, nullptr, &first);

    REQUIRE_FALSE(first.tracer_advected_in_batch);
    REQUIRE(second.tracer_advected_in_batch);
}

TEST_CASE("Cartesian advection kernels reject invalid metrics shapes and coefficients",
          "[numerics][advection][cartesian][config-truth]")
{
    setup_cartesian_grid(4, 4, 4, 100.0, 120.0, 80.0, 0.5);
    set_uniform_velocity(1.0, 1.0, 0.0);
    rho.resize(NR, NTH, NZ, 1.0f);
    Field3D source(NR, NTH, NZ, 2.0f);
    Field3D destination(NR, NTH, NZ, 9.0f);

    dr = 0.0;
    REQUIRE_THROWS_AS(
        advect_scalar_1d_x_kernel_cartesian(source, destination, 0.5),
        std::invalid_argument);
    REQUIRE(destination[1][1][1] == Approx(9.0f));
    dr = 100.0;

    REQUIRE_THROWS_AS(
        apply_diffusion_kernel_cartesian(source, destination, 0.5, -1.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        apply_diffusion_kernel_cartesian(source, destination, 0.5, 1.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        apply_diffusion_kernel_cartesian(
            source, destination, std::numeric_limits<double>::quiet_NaN(), 0.0),
        std::invalid_argument);

    Field3D wrong_shape(3, 4, 4, 2.0f);
    REQUIRE_THROWS_AS(
        advect_scalar_conservative_3d_cartesian(
            wrong_shape, destination, 0.5),
        std::invalid_argument);
}

TEST_CASE("RK mass RHS is timestep independent and conservative in compressible periodic flow",
          "[numerics][advection][rk][conservation]")
{
    for (const auto staggering : {StaggerType::CGrid, StaggerType::Collocated})
    {
        setup_cartesian_grid(12, 8, 6, 100.0, 100.0, 100.0, 0.1);
        global_stagger_type = staggering;
        global_grid_metrics.terrain_metrics_active = false;
        global_lateral_bc_west = global_lateral_bc_east =
            global_lateral_bc_south = global_lateral_bc_north =
                tmv::init::config::LateralBC::Periodic;
        global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
        global_top_bc = tmv::init::config::TopBC::Rigid;
        rho.resize(NR, NTH, NZ, 1.0f);
        u.resize(NR, NTH, NZ, 0.0f);
        v.resize(NR, NTH, NZ, 0.0f);
        w.resize(NR, NTH, NZ, 0.0f);
        Field3D scalar(NR, NTH, NZ, 0.0f);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    const int periodic_i = (i + NR - 3) % (NR - 2);
                    const double phase = 2.0 * kPi * periodic_i / (NR - 2);
                    rho[i][j][k] = 1.0f + 0.15f * std::cos(phase);
                    u[i][j][k] = 3.0f + std::sin(phase);
                    scalar[i][j][k] = 0.01f + 0.002f * std::sin(phase + 0.3);
                }
        Field3D first;
        for (const double step : {0.1, 0.5})
        {
            tmv::CartesianCGridTransportContext fluxes;
            fluxes.capture_instantaneous(rho, u, v, w, step);
            Field3D tendency;
            tmv::compute_conservative_scalar_mass_tendency_cartesian(
                scalar, tendency, step, true, fluxes,
                tmv::CartesianFaceReconstruction::MonotonicCentered);
            if (first.empty()) first = tendency;
            else REQUIRE(std::memcmp(first.data(), tendency.data(),
                        first.size() * sizeof(float)) == 0);
            const auto domain = physical_scalar_domain(NR, NTH, NZ, staggering);
            double net = 0.0;
            double magnitude = 0.0;
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        net += tendency[i][j][k];
                        magnitude += std::abs(tendency[i][j][k]);
                    }
            REQUIRE(magnitude > 0.0);
            REQUIRE(std::abs(net) < 1.0e-6 * magnitude);
        }
    }
}

TEST_CASE("Strict native WENO restores fields and ledgers after partial device failure",
          "[numerics][advection][native-weno][transaction]")
{
    struct FailingBackend : ComputeBackend, AdvectionComputeKernels
    {
        int         mode  = 0;
        int         calls = 0;
        std::string name() const override
        {
            return "injected-native-weno";
        }
        bool initialize(std::string&) override
        {
            return true;
        }
        void shutdown() override
        {
        }
        AdvectionComputeKernels* advection_kernels() override
        {
            return this;
        }
        const AdvectionComputeKernels* advection_kernels() const override
        {
            return this;
        }
        bool supports(AdvectionKernelOp op) const override
        {
            return op == AdvectionKernelOp::ConservativeCartesianCgridTransport;
        }
        bool supports_supplied_cartesian_cgrid_mass_flux() const override
        {
            return true;
        }
        bool dispatch_advect_scalars_conservative_cartesian_cgrid(
            const ConservativeCartesianCgridTransportDispatch& dispatch) override
        {
            ++calls;
            const int index                                           = (2 * dispatch.ny + 2) * dispatch.nz + 2;
            dispatch.q_out[0][index]                                  = 301;
            dispatch.host_boundary_reports[0]->committed_state_change = 123;
            if (mode == 0)
                throw std::runtime_error("injected late failure");
            if (mode == 1)
                return false;
            dispatch.q_out[1][index] = mode == 2 ? std::numeric_limits<float>::quiet_NaN() : -1;
            return true;
        }
    };
    setup_cartesian_grid(8, 7, 6, 100, 120, 80, .5);
    const auto prior_stagger = global_stagger_type;
    global_stagger_type      = StaggerType::CGrid;
    auto view                = global_simulation_state();
    struct Restore
    {
        RuntimeConfigurationState* config;
        bool                       admission;
        bool                       native_dry_tvd;
        bool                       repairs     = primary_state_repairs_enabled;
        bool                       fail_closed = primary_state_repairs_fail_closed;
        StaggerType                stagger;
        ~Restore()
        {
            config->admission.native_weno5 = admission;
            config->admission.native_dry_tvd = native_dry_tvd;
            set_primary_state_repair_policy(repairs, fail_closed);
            global_stagger_type = stagger;
        }
    } restore{view.configuration,
              view.configuration->admission.native_weno5,
              view.configuration->admission.native_dry_tvd,
              primary_state_repairs_enabled,
              primary_state_repairs_fail_closed,
              prior_stagger};
    view.configuration->admission.native_weno5 = true;
    set_primary_state_repair_policy(false, true);
    rho.resize(NR, NTH, NZ, 1);
    p.resize(NR, NTH, NZ, 90000);
    theta.resize(NR, NTH, NZ, 300);
    qv.resize(NR, NTH, NZ, .01f);
    for (auto* field : {&qc, &qr, &qi, &qs, &qg, &qh, &tracer})
        field->resize(NR, NTH, NZ, 0);
    set_uniform_velocity(0, 0, 0);
    CartesianCGridTransportContext context;
    context.begin(NR, NTH, NZ, .5);
    context.accumulate(rho, u, v, w, .5);
    context.complete();
    ComputeBackendRuntimeState compute;
    compute.config.allow_fallback = false;
    compute.active_kind           = ComputeBackendKind::Metal;
    auto  backend                 = std::make_unique<FailingBackend>();
    auto* injected                = backend.get();
    compute.backend               = std::move(backend);
    view.compute_runtime          = &compute;
    AdvectionRuntimeState runtime;
    runtime.config.scheme_id = "weno5";
    runtime.reset_for_shape(NR, NTH, NZ);
    runtime.thermodynamic_boundary_reports_requested                 = true;
    runtime.thermodynamic_boundary_reports_complete                  = true;
    runtime.thermodynamic_boundary_reports[0].committed_state_change = 17;
    const Field3D original_theta = theta, original_vapor = qv;
    for (int mode = 0; mode < 4; ++mode)
    {
        injected->mode = mode;
        CHECK_THROWS(advect_thermodynamics_3d(view, .5, 0, 0, &context, &runtime));
        CHECK(injected->calls == mode + 1);
        CHECK(std::equal(theta.data(), theta.data() + theta.size(), original_theta.data()));
        CHECK(std::equal(qv.data(), qv.data() + qv.size(), original_vapor.data()));
        CHECK(runtime.thermodynamic_boundary_reports_complete);
        CHECK(runtime.thermodynamic_boundary_reports[0].committed_state_change == 17);
    }
    view.configuration->admission.native_weno5 = false;
    view.configuration->admission.native_dry_tvd = true;
    runtime.config.scheme_id = "tvd";
    injected->calls = 0;
    for (int mode = 0; mode < 4; ++mode)
    {
        injected->mode = mode;
        CHECK_THROWS(advect_thermodynamics_3d(view, .5, 0, 0, &context, &runtime));
        CHECK(injected->calls == mode + 1);
        CHECK(std::equal(theta.data(), theta.data() + theta.size(), original_theta.data()));
        CHECK(std::equal(qv.data(), qv.data() + qv.size(), original_vapor.data()));
        CHECK(runtime.thermodynamic_boundary_reports_complete);
        CHECK(runtime.thermodynamic_boundary_reports[0].committed_state_change == 17);
    }
}

TEST_CASE("Native private-moment WENO validates every result before publishing and preserves units",
          "[numerics][advection][native-weno][private-moments]")
{
    struct Moments : MilbrandtScheme
    {
        Field3D                               mixing{8, 7, 6, 3}, concentration{8, 7, 6, 20};
        tmv::MicrophysicsPrognosticScalarList prognostic_scalars_for_transport() override
        {
            return {{"mixing", &mixing, true, tmv::MicrophysicsScalarStorage::DryAirMixingRatio},
                    {"concentration", &concentration, true, tmv::MicrophysicsScalarStorage::VolumeConcentration}};
        }
    } moments;
    struct Backend : ComputeBackend, AdvectionComputeKernels
    {
        int                        mode = 0, calls = 0;
        const SimulationStateView* expected = nullptr;
        std::string                name() const override
        {
            return "private-moment-test";
        }
        bool initialize(std::string&) override
        {
            return true;
        }
        void shutdown() override
        {
        }
        AdvectionComputeKernels* advection_kernels() override
        {
            return this;
        }
        const AdvectionComputeKernels* advection_kernels() const override
        {
            return this;
        }
        bool supports(AdvectionKernelOp op) const override
        {
            return op == AdvectionKernelOp::ConservativeCartesianCgridTransport;
        }
        bool supports_supplied_cartesian_cgrid_mass_flux() const override
        {
            return true;
        }
        bool dispatch_advect_scalars_conservative_cartesian_cgrid(
            const ConservativeCartesianCgridTransportDispatch& d) override
        {
            ++calls;
            CHECK(d.weno);
            CHECK(d.execution_state == expected);
            CHECK(d.n_scalars == 2);
            CHECK(d.use_supplied_mass_flux);
            CHECK(d.q_in[0][0] == 3);
            CHECK(d.q_in[1][0] == 20);
            for (int field = 0; field < d.n_scalars; ++field)
                for (int n = 0; n < d.nx * d.ny * d.nz; ++n)
                    d.q_out[field][n] = .5f * d.q_in[field][n];
            if (mode == 0)
                throw std::runtime_error("late private-moment failure");
            if (mode == 1)
                return false;
            if (mode == 2)
                d.q_out[1][0] = std::numeric_limits<float>::quiet_NaN();
            if (mode == 3)
                d.q_out[1][0] = -1;
            if (mode == 4)
                d.q_out[1][0] = std::numeric_limits<float>::max();
            return true;
        }
    };
    SimulationStorageState storage{{8, 7, 6, 100, 100, .2, 100}};
    storage.grid.geometry.initialize(8, 7, 6, 100, 100, 100, CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.configuration.coordinate_system      = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type           = StaggerType::CGrid;
    storage.configuration.admission.native_weno5 = true;
    storage.fields.rho.resize(8, 7, 6, 1);
    for (auto* field : {&storage.fields.u, &storage.fields.v, &storage.fields.w})
        field->resize(8, 7, 6, 0);
    auto                           view = make_simulation_state_view(storage);
    CartesianCGridTransportContext context;
    context.begin(view, 8, 7, 6, .2);
    context.accumulate(view, storage.fields.rho, storage.fields.u, storage.fields.v, storage.fields.w, .2);
    context.complete(view);
    storage.fields.rho.fill(2);
    ComputeBackendRuntimeState compute;
    compute.config.allow_fallback = false;
    compute.active_kind           = ComputeBackendKind::Metal;
    auto  backend                 = std::make_unique<Backend>();
    auto* injected                = backend.get();
    injected->expected            = &view;
    compute.backend               = std::move(backend);
    view.compute_runtime          = &compute;
    AdvectionRuntimeState runtime;
    runtime.config.scheme_id = "weno5";
    storage.fields.rho.fill(0);
    CHECK_THROWS(advect_microphysics_prognostics_3d(view, moments, .2, &context, &runtime));
    CHECK(injected->calls == 0);
    storage.fields.rho.fill(2);
    storage.fields.u.resize(1, 1, 1, 0);
    CHECK_THROWS(advect_microphysics_prognostics_3d(view, moments, .2, &context, &runtime));
    CHECK(injected->calls == 0);
    storage.fields.u.resize(8, 7, 6, 0);
    for (int mode = 0; mode < 5; ++mode)
    {
        injected->mode = mode;
        CHECK_THROWS(advect_microphysics_prognostics_3d(view, moments, .2, &context, &runtime));
        CHECK(injected->calls == mode + 1);
        CHECK(std::all_of(
            moments.mixing.data(), moments.mixing.data() + moments.mixing.size(), [](float x) { return x == 3; }));
        CHECK(std::all_of(moments.concentration.data(),
                          moments.concentration.data() + moments.concentration.size(),
                          [](float x) { return x == 20; }));
    }
    injected->mode = 5;
    REQUIRE_NOTHROW(advect_microphysics_prognostics_3d(view, moments, .2, &context, &runtime));
    CHECK(std::all_of(
        moments.mixing.data(), moments.mixing.data() + moments.mixing.size(), [](float x) { return x == 1.5f; }));
    CHECK(std::all_of(moments.concentration.data(),
                      moments.concentration.data() + moments.concentration.size(),
                      [](float x) { return x == 20; }));
}

TEST_CASE("CPU conservative dispatch honors an independent execution view", "[numerics][advection][scoped-dispatch]")
{
    SimulationStorageState storage{{8, 7, 6, 100, 100, .2, 100}};
    storage.grid.geometry.initialize(8, 7, 6, 100, 100, 100, CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type      = StaggerType::CGrid;
    storage.fields.rho.resize(8, 7, 6, 1.18f);
    for (auto* field : {&storage.fields.u, &storage.fields.v, &storage.fields.w})
        field->resize(8, 7, 6, 0);
    auto                           view = make_simulation_state_view(storage);
    CartesianCGridTransportContext context;
    context.begin(view, 8, 7, 6, .2);
    context.accumulate(view, storage.fields.rho, storage.fields.u, storage.fields.v, storage.fields.w, .2);
    context.complete(view);
    Field3D                                     scalar(8, 7, 6, 300.03f);
    const Field3D                               before        = scalar;
    Field3D*                                    fields[]      = {&scalar};
    const float*                                input[]       = {scalar.data()};
    float*                                      output[]      = {scalar.data()};
    int                                         positive[]    = {0};
    double                                      diffusivity[] = {0};
    ConservativeCartesianCgridTransportDispatch dispatch{};
    dispatch.execution_state   = &view;
    dispatch.n_scalars         = 1;
    dispatch.host_fields       = fields;
    dispatch.host_kappas       = diffusivity;
    dispatch.host_context      = &context;
    dispatch.dt_step_host      = .2;
    dispatch.q_in              = input;
    dispatch.q_out             = output;
    dispatch.positive_definite = positive;
    auto        backend        = create_cpu_compute_backend();
    std::string error;
    REQUIRE(backend->initialize(error));
    REQUIRE(backend->advection_kernels()->dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
    CHECK(std::equal(scalar.data(), scalar.data() + scalar.size(), before.data()));
    backend->shutdown();
}

TEST_CASE("Device scalar flux attribution preserves sub-ULP reconstructed increments",
          "[numerics][advection][native-weno][flux-precision]")
{
    constexpr int          nx = 8, ny = 7, nz = 6;
    SimulationStorageState storage{{nx, ny, nz, 100, 100, .2, 100}};
    storage.grid.geometry.initialize(nx, ny, nz, 100, 100, 100, CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type      = StaggerType::CGrid;
    storage.fields.rho.resize(nx, ny, nz, 1);
    auto        view = make_simulation_state_view(storage);
    const float ulp  = std::nextafter(300.0f, 301.0f) - 300.0f;
    for (int axis = 0; axis < 3; ++axis)
        for (float flux : {7.3f, -7.3f})
        {
            Field3D scalar(nx, ny, nz), mfx(nx, ny, nz, 0), mfy(nx, ny, nz, 0), mfz(nx, ny, nz, 0);
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        const int position = axis == 0 ? i : axis == 1 ? j : k;
                        scalar(i, j, k)    = 300.0f + (position - 2) * ulp;
                    }
            Field3D* mass_flux[] = {&mfx, &mfy, &mfz};
            mass_flux[axis]->fill(flux);
            tmv::FlatCGridDeviceScalarAttributionWorkspace workspace;
            tmv::ScalarBoundaryTransportReport             report;
            tmv::attribute_flat_cgrid_device_scalar_commit(view,
                                                           scalar.data(),
                                                           scalar.data(),
                                                           storage.fields.rho.data(),
                                                           storage.fields.rho.data(),
                                                           mfx.data(),
                                                           mfy.data(),
                                                           mfz.data(),
                                                           nx,
                                                           ny,
                                                           nz,
                                                           .01f,
                                                           .01f,
                                                           .01f,
                                                           .2f,
                                                           false,
                                                           &report,
                                                           tmv::ScalarMoistEnergyKind::None,
                                                           nullptr,
                                                           workspace);
            const double   face          = 300.0 + .5 * static_cast<double>(ulp);
            const float    expected      = static_cast<float>(static_cast<double>(flux) * face);
            const Field3D* scalar_flux[] = {
                &workspace.scalar_flux_x, &workspace.scalar_flux_y, &workspace.scalar_flux_z};
            REQUIRE(report.complete);
            CHECK((*scalar_flux[axis])(2, 2, 2) == expected);
            CHECK(flux * static_cast<float>(face) != expected);
        }
}

TEST_CASE("Positive scalar interface registers capture the committed donor-limited flux in a batch",
          "[numerics][advection][interface][moist]")
{
    setup_cartesian_grid(6, 6, 4, 1, 1, 1, .9);
    global_stagger_type    = StaggerType::CGrid;
    global_lateral_bc_west = global_lateral_bc_east = global_lateral_bc_south = global_lateral_bc_north =
        tmv::init::config::LateralBC::ZeroGradient;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc    = tmv::init::config::TopBC::Rigid;
    rho.resize(NR, NTH, NZ, 1);
    u.resize(NR, NTH, NZ, .25f);
    v.resize(NR, NTH, NZ, .25f);
    w.resize(NR, NTH, NZ, .25f);
    Field3D first(NR, NTH, NZ, 0), second(NR, NTH, NZ, 0);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                first(i, j, k)  = float(std::max(0, i + j + k - 4));
                second(i, j, k) = 2 * first(i, j, k);
                if (k == NZ - 1)
                    w(i, j, k) = 0;
            }
    CartesianCGridTransportContext context;
    context.begin(NR, NTH, NZ, .9);
    context.capture_density_begin(rho);
    context.mass_flux_x.fill(.25f);
    context.mass_flux_y.fill(.25f);
    context.mass_flux_z.fill(.25f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            context.mass_flux_z(i, j, NZ - 1) = 0;
            rho(i, j, 0)                      = .775f;
            rho(i, j, NZ - 1)                 = 1.225f;
        }
    context.record_external_accumulation(.9, 1);
    context.complete();
    tmv::CartesianInterfaceFluxRegister a, b;
    a.configure(NR, NTH, NZ, {2, 3, 2, 3, 1, 2});
    b.configure(NR, NTH, NZ, {2, 3, 2, 3, 1, 2});
    tmv::CartesianScalarTransport scalars[2];
    scalars[0].field                   = &first;
    scalars[0].positive_definite       = true;
    scalars[0].interface_flux_register = &a;
    scalars[1].field                   = &second;
    scalars[1].positive_definite       = true;
    scalars[1].interface_flux_register = &b;
    advect_scalars_conservative_3d_cartesian(
        scalars, 2, .9, &context, tmv::CartesianFaceReconstruction::MonotonicCentered);
    CHECK(a.scalar().contribution_count == 1);
    CHECK(b.scalar().contribution_count == 1);
    CHECK(a.scalar().accumulated_dt_s == .9);
    CHECK(b.scalar().accumulated_dt_s == .9);
    REQUIRE(a.scalar().east.size() == 1);
    CHECK(a.scalar().east[0] < .3375);
    CHECK(a.scalar().east[0] == Approx(1.0 / 3).margin(2e-6));
    CHECK(a.scalar().north[0] == a.scalar().east[0]);
    CHECK(b.scalar().east[0] == Approx(2 * a.scalar().east[0]).margin(1e-12));
    CHECK(first(2, 2, 1) >= 0);
    CHECK(first(2, 2, 1) < 1e-5);
    const auto old                     = a.scalar();
    scalars[1].interface_flux_register = &a;
    REQUIRE_THROWS(advect_scalars_conservative_3d_cartesian(scalars, 2, .9, &context));
    CHECK(a.scalar().contribution_count == old.contribution_count);
    CHECK(a.scalar().east == old.east);
}

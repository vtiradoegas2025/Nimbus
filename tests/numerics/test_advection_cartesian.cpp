/**
 * @file test_advection_cartesian.cpp
 * @brief Verification gates for Phase A.5 (Cartesian scalar advection).
 *
 * Phase A.5 of the Coordinate Backend Plan (`docs/CoordinateBackend_Plan.md`).
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

#include "core/coordinate_system.hpp"
#include "core/field3d.hpp"
#include "core/runtime_config.hpp"
#include "core/simulation.hpp"
#include "numerics/advection/advection.hpp"
#include "numerics/advection/advection_base.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace
{

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Grid setup helpers
// ---------------------------------------------------------------------------

/**
 * @brief Configures the global grid + coord-system globals for a Cartesian
 *        advection test. The same routine is reused by both gates.
 *
 * `dr` is reused as both the x and y cell spacing — Phase A square cells.
 * `dtheta` is set to a non-trivial value on purpose: the Cartesian path
 * must NOT depend on `dtheta` (any read of `dtheta` from the Cartesian
 * helpers would be a Bug-7 regression). If a future change accidentally
 * leaks `dtheta` into the Cartesian advection path, both tests will fail
 * loudly because the centroid will drift.
 */
void setup_cartesian_grid(int nr, int nth, int nz, double dx_m, double dz_m, double dt_s)
{
    NR = nr;
    NTH = nth;
    NZ = nz;
    dr = dx_m;
    dz = dz_m;
    dt = dt_s;
    dtheta = 2.0 * kPi / std::max(nth, 1);  // intentionally non-zero — must not affect Cartesian path
    global_coordinate_system = CoordinateSystem::Cartesian;
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
 * Cell coordinates: x_i = i * dr, y_j = j * dr, z_k = k * dz.
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
            const double y = static_cast<double>(j) * dr;
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
            const double y = static_cast<double>(j) * dr;
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
            const double y = static_cast<double>(j) * dr;
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
          "[numerics][advection][cartesian]")
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

    setup_cartesian_grid(nr, nth, nz, dx_m, dz_m, dt_s);

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
          "[numerics][advection][cartesian]")
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

    setup_cartesian_grid(nr, nth, nz, dx_m, dz_m, dt_s);

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
// Gate 3: Cartesian path is invariant under non-zero `dtheta`
// ---------------------------------------------------------------------------

TEST_CASE("Cartesian advection: dispatch ignores dtheta (Bug-7 IC analog)",
          "[numerics][advection][cartesian]")
{
    // The Cartesian kernels must NOT read `dtheta`. If they do, the
    // result will depend on the cylindrical grid's azimuthal spacing —
    // which is the exact false-coupling that broke the supercell case
    // (Bug 7) in the cylindrical dynamics. This test runs the same
    // scenario twice with two very different `dtheta` values and
    // confirms the resulting fields are bit-equal.
    const int nr = 100;
    const int nth = 16;
    const int nz = 5;
    const double dx_m = 100.0;
    const double dz_m = 100.0;
    const double dt_s = 1.0;
    const int n_steps = 20;

    setup_cartesian_grid(nr, nth, nz, dx_m, dz_m, dt_s);
    set_uniform_velocity(20.0, 10.0, 0.0);
    disable_numerics_vertical_advection();

    Field3D q_a;
    place_gaussian_bump_2d(q_a, 3000.0, 800.0, 800.0, 1.0);

    // Snapshot the IC for the second run before any advection mutates it.
    Field3D q_b;
    q_b.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                q_b[i][j][k] = q_a[i][j][k];

    // Run A with the standard dtheta from setup_cartesian_grid.
    const double dtheta_a = dtheta;
    for (int step = 0; step < n_steps; ++step)
    {
        advect_scalar_3d(q_a, dt_s, /*kappa=*/0.0);
    }

    // Run B with a deliberately weird dtheta value. If the Cartesian
    // helpers leak `dtheta` into their flux computation, q_b will end up
    // different from q_a.
    dtheta = 0.123456;
    for (int step = 0; step < n_steps; ++step)
    {
        advect_scalar_3d(q_b, dt_s, /*kappa=*/0.0);
    }

    INFO("dtheta_a = " << dtheta_a << ", dtheta_b = " << 0.123456);

    double max_diff = 0.0;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double diff = std::abs(static_cast<double>(q_a[i][j][k]) -
                                              static_cast<double>(q_b[i][j][k]));
                if (diff > max_diff)
                {
                    max_diff = diff;
                }
            }
        }
    }

    // Bit-for-bit equality is the goal. Use a tiny epsilon to absorb any
    // future FMA-vs-non-FMA reorderings.
    INFO("max |q_a − q_b| = " << max_diff);
    REQUIRE(max_diff < 1.0e-12);
}

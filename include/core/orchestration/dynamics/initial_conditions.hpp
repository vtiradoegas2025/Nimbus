/**
 * @file initial_conditions.hpp
 * @brief Initial-condition entry points dispatched by coordinate system.
 *
 * Phase A.4 of the Coordinate Backend Plan (`docs/CoordinateBackend_Plan.md`).
 *
 * The runtime owns one cylindrical and one Cartesian initial-condition
 * implementation. The cylindrical body is inline in
 * `src/core/equations.cpp::initialize` (its body predates this file). The
 * Cartesian implementations are declared here and defined in
 * `src/core/initial_conditions_cartesian.cpp` for two reasons:
 *
 *   1. Tests can link the Cartesian IC code without dragging in all of
 *      `equations.cpp` (which transitively requires advection, microphysics,
 *      radiation, etc.). The test_harness already provides the global state
 *      symbols the helpers read and write.
 *
 *   2. The Cartesian implementations do not need any of the cylindrical
 *      `cos θ`/`sin θ` projection machinery, the radial-axis bubble
 *      ring, or the multi-layer hydrostatic builder. They stay small,
 *      self-contained, and easy to audit.
 *
 * Wind field naming: both coordinate backends use the same globals:
 *   u  — horizontal-1 (radial in cylindrical, zonal in Cartesian)
 *   v  — horizontal-2 (azimuthal in cylindrical, meridional in Cartesian)
 *   w  — vertical (unchanged)
 */

#pragma once

/**
 * @brief Fills the wind field on a Cartesian grid with the configured
 *        hodograph profile, no `cos θ`/`sin θ` projection.
 *
 * For every cell (i, j, k):
 *   u[i][j][k]       = u_x(z[k])
 *   v[i][j][k] = u_y(z[k])
 *   w[i][j][k]       = 0
 *
 * where `(u_x(z), u_y(z))` come from `compute_wind_profile(global_wind_profile, z)`.
 * The wind is constant in (x, y) at every level — that is the entire point
 * of Phase A: a uniform Cartesian wind stays uniform when the dynamics
 * looks at it on a Cartesian grid, which is the property the cylindrical
 * antisymmetric BC violates and the Bug 7 ledger documents in detail.
 *
 * Preconditions:
 *   - `u`, `v`, `w` have all been resized to (NR, NTH, NZ).
 *   - `dz > 0` and NZ >= 1.
 *   - `global_wind_profile` is set.
 */
void apply_cartesian_wind_initialization();

/**
 * @brief Fills the wind field on an Arakawa C-grid cylindrical mesh with the
 *        configured Cartesian hodograph projected onto the staggered (r, theta)
 *        face positions. Phase C.3 of the Coordinate Backend Plan.
 *
 * Field placement on this scheme (matches docs/CoordinateBackend_Plan.md
 * "Field Storage Convention"):
 *   u (radial)    r-face         u[i][j][k]   at (r_face[i], theta[j],         z[k])
 *   v (azimuthal) theta-face     v[i][j][k]   at (r[i],      theta_{j+1/2},    z[k])
 *   w (vertical)  z-face         w[i][j][k]   at (r[i],      theta[j],         z_face[k])
 *
 * For every cell (i, j, k):
 *   u[i][j][k] =   u_x(z[k]) * cos(theta[j])         + u_y(z[k]) * sin(theta[j])
 *   v[i][j][k] = - u_x(z[k]) * sin(theta_{j+1/2})    + u_y(z[k]) * cos(theta_{j+1/2})
 *   w[i][j][k] = 0
 *
 * The trig argument for u uses theta[j] (the cell-center azimuth, since the
 * r-face shares its cell's theta), while v uses theta_{j+1/2} = (j + 0.5) *
 * dtheta -- the half-cell-shifted azimuth where the theta-face lives. This
 * shift is the only structural difference vs the collocated cylindrical wind
 * init; it is what the C-grid divergence operator expects so that the
 * discrete flux-form divergence picks up only the geometric O(dtheta^2)
 * cylindrical-from-Cartesian projection error rather than an additional
 * placement-induced error.
 *
 * The continuous divergence of a uniform Cartesian wind is exactly zero;
 * the discrete C-grid divergence is
 *   (1 - sin(dtheta/2) / (dtheta/2)) * (u_x cos(theta_j) + u_y sin(theta_j)) / r_i
 *   ~ (dtheta^2 / 24) * |u_h| / r_i
 * which converges to zero quadratically with grid refinement. This is the
 * inherent "cylindrical-from-Cartesian" placement error that motivated the
 * Cartesian backend in Phase A; on the cylindrical C-grid we minimize it
 * (and avoid the Bug 7 amplification at the axis) by using face-centered
 * placement.
 *
 * Preconditions:
 *   - `u`, `v`, `w` have all been resized to (NR, NTH, NZ).
 *   - `dz > 0`, `dtheta > 0`, NZ >= 1, NTH >= 1.
 *   - `global_grid_geometry` has been initialized for cylindrical C-grid
 *     (so `geo.z[k]` and the cell-center sin/cos lookups are populated).
 *   - `global_wind_profile` is set; `compute_wind_profile` returns the
 *     Cartesian (u_x, u_y) at altitude z.
 */
void apply_cylindrical_cgrid_wind_initialization();

/**
 * @brief Adds the configured trigger-bubble Δθ patch on a Cartesian grid.
 *
 * The bubble is a 3D Gaussian centered at
 *   (global_bubble_center_x_m, global_bubble_center_y_m, global_bubble_center_z_m)
 * with the standard cell-coordinate mapping
 *   x[i] = i * dr,  y[j] = j * dr,  z[k] = k * dz
 * (Phase A reuses `dr` for both `dx` and `dy` — square horizontal cells).
 *
 * The Δθ falloff is the same Gaussian the cylindrical path uses:
 *   factor(dist) = exp(−(dist / (radius / 3))²)   for dist <= radius
 *   factor       = 0                              otherwise
 *
 * The function only writes to `theta`. It is *additive*: the bubble Δθ is
 * added to the existing theta field, so call this after the base-state
 * theta has been initialized.
 *
 * Preconditions:
 *   - `theta` has been resized to (NR, NTH, NZ) and contains the base
 *     state to which the bubble Δθ should be added.
 *   - `dr > 0` and `dz > 0`.
 *   - `global_bubble_radius_m > 0` (the helper enforces a 100 m floor).
 */
void apply_cartesian_bubble_initialization();

/**
 * @brief Adds the configured trigger-bubble Δθ patch on a cylindrical grid.
 *
 * 2D Gaussian ring in the (r, z) plane, uniform around all θ. Body lives in
 * `src/core/orchestration/dynamics/equations.cpp`; this declaration exists
 * so the trigger module can call into it.
 *
 * Same Gaussian falloff as the Cartesian variant:
 *   factor(dist) = exp(−(dist / (radius / 3))²)   for dist <= radius
 *   factor       = 0                              otherwise
 *
 * Preconditions match apply_cartesian_bubble_initialization.
 */
void apply_cylindrical_bubble_initialization();

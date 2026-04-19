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
 * Field aliasing note (Phase A): while the Cartesian dynamics scheme is
 * active, the cylindrical-named globals carry Cartesian wind components:
 *   u       -> u_x   (zonal)
 *   v_theta -> u_y   (meridional)
 *   w       -> u_z   (vertical, unchanged)
 * The 267-site `v_theta -> v` rename is deferred to Phase B (see Bug 7 in
 * `docs/Journey.md` for the field-aliasing rationale).
 */

#pragma once

/**
 * @brief Fills the wind field on a Cartesian grid with the configured
 *        hodograph profile, no `cos θ`/`sin θ` projection.
 *
 * For every cell (i, j, k):
 *   u[i][j][k]       = u_x(z[k])
 *   v_theta[i][j][k] = u_y(z[k])
 *   w[i][j][k]       = 0
 *
 * where `(u_x(z), u_y(z))` come from `compute_wind_profile(global_wind_profile, z)`.
 * The wind is constant in (x, y) at every level — that is the entire point
 * of Phase A: a uniform Cartesian wind stays uniform when the dynamics
 * looks at it on a Cartesian grid, which is the property the cylindrical
 * antisymmetric BC violates and the Bug 7 ledger documents in detail.
 *
 * Preconditions:
 *   - `u`, `v_theta`, `w` have all been resized to (NR, NTH, NZ).
 *   - `dz > 0` and NZ >= 1.
 *   - `global_wind_profile` is set.
 */
void apply_cartesian_wind_initialization();

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

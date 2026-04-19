/**
 * @file boundary_conditions.hpp
 * @brief Boundary-condition entry points dispatched by coordinate system.
 *
 * The runtime owns one cylindrical and one Cartesian boundary-condition
 * implementation; `apply_boundary_conditions()` in `src/core/dynamics.cpp`
 * picks one based on the active `global_coordinate_system`.
 *
 * Each coordinate system has its own implementation file:
 *   - `src/core/boundary_conditions_cylindrical.cpp`
 *   - `src/core/boundary_conditions_cartesian.cpp`
 *
 * Tests can link either BC file independently (the test_harness provides
 * the global state symbols the functions read/write).
 */

#pragma once

/**
 * @brief Applies Cartesian (open lateral, rigid lid + rigid surface) boundary
 *        conditions to the global prognostic state.
 *
 * Operates on the global Field3D state declared in `core/simulation.hpp`:
 *
 *   - Lateral x faces (i = 0, i = NR-1): zero-gradient (open) on
 *     u, v_theta (carrying u_y), w, rho, p, theta, qv, qc, qr, qi, qs, qg, qh.
 *     There is no axis reflection — Cartesian has no singular axis.
 *
 *   - Lateral y faces (j = 0, j = NTH-1): zero-gradient (open) on the same
 *     fields. There is no periodic theta wraparound — Cartesian has no
 *     azimuthal periodicity.
 *
 *   - Vertical faces (k = 0, k = NZ-1): rigid lid + rigid surface
 *     (w = 0 at top and bottom), zero-gradient on u and v_theta (free-slip),
 *     hydrostatic extrapolation for pressure (matching the discrete
 *     `dp/dz = -rho*g` relation the centered dynamics stencil expects),
 *     zero-gradient on rho, theta, and the moisture variables.
 *
 * The function does NOT call `enforce_primary_state_bounds`; the dispatcher
 * in `apply_boundary_conditions()` runs that defensive clamp after the
 * coordinate-specific BCs so both backends share the same final guard.
 *
 * Field aliasing note (Phase A): while the Cartesian dynamics scheme is
 * active, the cylindrical-named globals carry Cartesian wind components:
 *   u       -> u_x   (zonal)
 *   v_theta -> u_y   (meridional)
 *   w       -> u_z   (vertical, unchanged)
 * The 267-site `v_theta -> v` rename is deferred to Phase B.
 *
 * Preconditions:
 *   - All referenced global Field3D state has been resized to (NR, NTH, NZ).
 *   - NR, NTH, NZ are >= 2 (so the boundary cells have an interior neighbor).
 *   - dz > 0.
 */
void apply_cartesian_boundary_conditions();

/**
 * @brief Applies cylindrical boundary conditions to the global prognostic state.
 *
 * Axis-reflection at i=0 (antisymmetric u_r, symmetric w/rho/p/scalars),
 * zero-gradient at i=NR-1, vertical rigid lid + rigid surface with
 * hydrostatic pressure extrapolation, zero-gradient scalars on all faces.
 *
 * The function does NOT call `enforce_primary_state_bounds`; the dispatcher
 * in `apply_boundary_conditions()` runs that defensive clamp after the
 * coordinate-specific BCs so both backends share the same final guard.
 *
 * Preconditions:
 *   - All referenced global Field3D state has been resized to (NR, NTH, NZ).
 *   - NR, NTH, NZ are >= 2.
 *   - dz > 0.
 */
void apply_cylindrical_boundary_conditions();

/**
 * @brief Lightweight BCs for acoustic substeps (u, v, w, rho, p only).
 *
 * Skips theta and all moisture fields. Uses the same convention as the
 * full BCs (coordinate-specific lateral, rigid-lid vertical, hydrostatic
 * pressure extrapolation) but at lower cost for the N-per-step acoustic loop.
 */
void apply_acoustic_boundary_conditions();

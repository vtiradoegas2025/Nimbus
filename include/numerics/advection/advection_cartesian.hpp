/**
 * @file advection_cartesian.hpp
 * @brief Cartesian-grid scalar advection helpers (Phase A.5).
 *
 * 
 *
 * The cylindrical horizontal advection kernels live as `static` helpers in
 * `src/advection/advection.cpp`:
 *
 *   - `advect_scalar_1d_r_kernel`     — first-order upwind in r
 *   - `advect_scalar_1d_theta_kernel` — first-order upwind in θ with the
 *                                       `(v / r) * dq/dθ` factor and the
 *                                       periodic `(j ± 1) % NTH` wraparound
 *   - `apply_diffusion_kernel`        — Laplacian with the `1/(r²·dθ²)`
 *                                       term in the y direction
 *
 * These kernels embed two cylindrical-specific assumptions that the
 * Cartesian backend must reject:
 *
 *   1. The `1/r` factor in the azimuthal flux (a Cartesian uniform flow
 *      `u_y` would not become a function of i if it weren't for this).
 *   2. The periodic `j ↔ j + NTH` wraparound. On a Cartesian grid the
 *      j = 0 and j = NTH − 1 faces are physical boundaries, not the same
 *      cell on the other side of the axis.
 *
 * This header declares three free functions implemented in
 * `src/advection/advection_cartesian.cpp` that compute the same scalar
 * advection step the cylindrical kernels do — first-order upwind +
 * Forward-Euler — but with straight Cartesian indexing in (x, y, z).
 *
 * The dispatcher in `src/advection/advection.cpp::advect_scalar_3d`
 * branches on `global_coordinate_system` and calls these helpers in the
 * same Strang-like split sequence the cylindrical path uses
 * (`x/2 → y/2 → z → y/2 → x/2 → diffusion`). The vertical TVD scheme is
 * already coordinate-agnostic and is reused verbatim.
 *
 * Wind field naming: both coordinate backends use the same globals:
 *   u  — horizontal-1 (radial in cylindrical, zonal in Cartesian)
 *   v  — horizontal-2 (azimuthal in cylindrical, meridional in Cartesian)
 *   w  — vertical (unchanged)
 *
 * Grid spacing note (Phase A): `dr` is reused as both the x cell spacing
 * (`dx`) and the y cell spacing (`dy`) — square horizontal cells. The
 * `dtheta` global is *not* read by the Cartesian helpers; if it ever is,
 * that's a Cartesian-vs-cylindrical bleed and a bug.
 */

#pragma once

#include "core/field3d.hpp"

/**
 * @brief First-order upwind scalar advection in the x direction.
 *
 * Reads the global `u` field as the x velocity. Uses `dr` as the x cell
 * spacing. Steps the source field by `dt` with Forward Euler:
 *
 *   q^{n+1}[i][j][k] = q^n[i][j][k] − dt · u[i][j][k] · (dq/dx)_upwind
 *
 * where the upwind difference is selected per cell on the sign of `u`:
 *
 *   u > 0  →  dq/dx = (q[i  ] − q[i−1]) / dx
 *   u < 0  →  dq/dx = (q[i+1] − q[i  ]) / dx
 *   u = 0  →  dq/dx = 0
 *
 * Boundary handling (Cartesian, zero-gradient):
 *   - x faces (i = 0 and i = NR − 1): `dst` is seeded from `src`, no
 *     interior update is written, so the boundary value carries through
 *     unchanged. This is the discrete equivalent of `∂q/∂x = 0` at the
 *     wall.
 *   - y faces (j = 0 and j = NTH − 1): the i-loop walks all interior
 *     cells regardless of j, so j-boundary cells are written by this
 *     kernel. The y kernel that runs after will leave them in the
 *     zero-gradient state.
 *   - z faces (k = 0 and k = NZ − 1): the k-loop runs from 1 to NZ − 2,
 *     so the floor and lid are not modified.
 *
 * Preconditions:
 *   - `src.size_r() == NR`, `src.size_th() == NTH`, `src.size_z() == NZ`
 *   - `dst` will be resized to match `src` if it isn't already
 *   - `dr > 0`
 *   - `u` is sized (NR, NTH, NZ)
 *
 * @param src  Input scalar field (read-only).
 * @param dst  Output scalar field (overwritten).
 * @param dt   Forward-Euler step length in seconds.
 */
void advect_scalar_1d_x_kernel_cartesian(const Field3D& src, Field3D& dst, double dt);

/**
 * @brief First-order upwind scalar advection in the y direction.
 *
 * Reads the global `v` field as the y velocity (Phase-A field
 * aliasing — `v` carries `u_y` while the Cartesian backend is
 * active). Uses `dr` as the y cell spacing. Steps the source field by
 * `dt` with Forward Euler:
 *
 *   q^{n+1}[i][j][k] = q^n[i][j][k] − dt · v[i][j][k] · (dq/dy)_upwind
 *
 * **No `1/r` factor.** **No periodic `(j ± 1) % NTH` wraparound.** The
 * j = 0 and j = NTH − 1 faces are physical boundaries.
 *
 * Boundary handling:
 *   - y faces (j = 0 and j = NTH − 1): not written; carry the seeded
 *     `src` value (zero-gradient).
 *   - x faces (i = 0 and i = NR − 1): not written.
 *   - z faces (k = 0 and k = NZ − 1): not written.
 *
 * Preconditions:
 *   - `src.size_r() == NR`, `src.size_th() == NTH`, `src.size_z() == NZ`
 *   - `dst` will be resized to match `src` if it isn't already
 *   - `dr > 0`
 *   - `v` is sized (NR, NTH, NZ)
 *
 * @param src  Input scalar field (read-only).
 * @param dst  Output scalar field (overwritten).
 * @param dt   Forward-Euler step length in seconds.
 */
void advect_scalar_1d_y_kernel_cartesian(const Field3D& src, Field3D& dst, double dt);

/**
 * @brief Cartesian Laplacian-with-Forward-Euler diffusion kernel.
 *
 * Applies one explicit diffusion step:
 *
 *   q^{n+1}[i][j][k] = q^n[i][j][k] + dt · κ · ∇²q
 *
 * where the discrete Laplacian is the standard 7-point Cartesian stencil:
 *
 *   ∇²q = (q[i+1] − 2 q + q[i−1]) / dx²
 *       + (q[j+1] − 2 q + q[j−1]) / dy²
 *       + (q[k+1] − 2 q + q[k−1]) / dz²
 *
 * No `1/(r² · dθ²)` factor. No periodic wraparound on j.
 *
 * If `kappa <= 0` the function copies `src` into `dst` and returns
 * (matching the cylindrical helper's short-circuit).
 *
 * Boundary handling: the same six-face zero-gradient convention as the
 * advection helpers — `dst` is seeded from `src`, the interior cells
 * (1..NR−2, 1..NTH−2, 1..NZ−2) are overwritten, and the six face planes
 * carry the source values through unchanged.
 *
 * Preconditions:
 *   - `src.size_r() == NR`, `src.size_th() == NTH`, `src.size_z() == NZ`
 *   - `dst` will be resized to match `src` if it isn't already
 *   - `dr > 0`, `dz > 0`
 *
 * @param src    Input scalar field (read-only).
 * @param dst    Output scalar field (overwritten).
 * @param dt     Step length in seconds.
 * @param kappa  Diffusivity in m²/s. Non-positive disables diffusion.
 */
void apply_diffusion_kernel_cartesian(const Field3D& src, Field3D& dst, double dt, double kappa);

/**
 * @file advection_cylindrical_cgrid.hpp
 * @brief Cylindrical Arakawa C-grid scalar advection helpers (Phase C.7).
 *
 * On the cylindrical C-grid the radial velocity u lives at r-face[i]
 * (between cell centers i and i+1), the azimuthal velocity v lives at
 * theta-face (between cell centers j and j+1), and the vertical velocity
 * w lives at z-face[k] (between cell centers k and k+1). Scalars (theta,
 * qv, qc, qr, qi, qs, qg, qh, tracer) live at cell centers.
 *
 * Because the face velocities are exactly at the locations needed for an
 * upwind flux calculation, no spatial interpolation of velocity is
 * required by the scalar advection step. This is the structural
 * advantage of the C-grid for conservative scalar transport.
 *
 * Each kernel below implements TVD-MUSCL flux-form advection (MC limiter)
 * in one direction with Forward-Euler integration:
 *
 *     q^{n+1}[i][j][k] = q^n[i][j][k] - dt * div_F
 *
 * where div_F is the finite-volume flux divergence in that direction:
 *
 *     div_F_r[i]     = (r_face[i] * F_right - r_face[i-1] * F_left ) / (r[i] * dr)
 *     div_F_theta[j] = (F_north - F_south) / (r[i] * dtheta)
 *     div_F_z[k]     = (F_top - F_bot) / dz
 *
 * The flux at every shared face uses the SAME upwind value, so the
 * pairwise cancellation that gives mass conservation is exact in
 * floating point: at face r_face[i] cell i contributes -r_face[i]*F[i]
 * and cell i+1 contributes +r_face[i]*F[i] when the per-cell update is
 * unwound through the cell volume. Total interior scalar mass is
 * therefore preserved to floating-point precision when the boundary
 * face fluxes vanish (axis u=0, outer wall u=0, surface w=0, lid w=0).
 *
 * Loop ranges (matching the SupercellCGridScheme cell-center compute
 * domain so the BC scheme owns the same boundary layers for advection
 * and dynamics):
 *
 *     i = 1..NR-2  (axis cell i=0 left to BC; outer wall i=NR-1 left to BC)
 *     j = 0..NTH-1 (periodic in theta; the BC scheme handles ghost copies)
 *     k = 1..NZ-2  (surface k=0 and lid k=NZ-1 left to BC)
 *
 * GPU dispatch (Phase C.9): each kernel attempts the matching C-grid
 * compute shader (advect_radial_cgrid.comp etc.) and falls back to the
 * CPU body when the pipeline is unavailable. The collocated cylindrical
 * shaders (advect_radial.comp etc.) read u, v, w as if they lived at
 * cell centers and remain unsafe for staggered fields; the C-grid path
 * never invokes them.
 *
 * Field naming: u (radial), v (azimuthal), w (vertical) -- the same
 * globals used by the collocated cylindrical and Cartesian backends.
 */

#pragma once

#include "core/field3d.hpp"

/**
 * @brief TVD-MUSCL flux-form scalar advection in the radial direction
 *        on the cylindrical Arakawa C-grid (MC limiter).
 *
 * Reads the global @c u field (radial velocity at r-face[i]) and steps
 * @p src forward by @p dt with Forward Euler:
 *
 *     q^{n+1}[i][j][k] = q^n[i][j][k] - dt * div_F_r[i]
 *
 * Phase C.9: attempts dispatch_radial_advection_cgrid_backend first
 * and falls back to the CPU body when the GPU pipeline is unavailable.
 *
 * @param src Input scalar field (read-only).
 * @param dst Output scalar field (overwritten; resized to (NR, NTH, NZ) if needed).
 * @param dt  Forward-Euler step length in seconds.
 */
void advect_scalar_1d_r_kernel_cylindrical_cgrid(const Field3D& src, Field3D& dst, double dt);

/**
 * @brief TVD-MUSCL flux-form scalar advection in the azimuthal direction
 *        on the cylindrical Arakawa C-grid (MC limiter).
 *
 * Reads the global @c v field (azimuthal velocity at theta-face) and
 * steps @p src forward by @p dt with Forward Euler:
 *
 *     q^{n+1}[i][j][k] = q^n[i][j][k] - dt * div_F_theta[j]
 *
 * Periodic in j: j_prev = (j - 1 + NTH) % NTH, j_next = (j + 1) % NTH.
 *
 * Phase C.9: attempts dispatch_azimuthal_advection_cgrid_backend first
 * and falls back to the CPU body when the GPU pipeline is unavailable.
 *
 * @param src Input scalar field (read-only).
 * @param dst Output scalar field (overwritten; resized to (NR, NTH, NZ) if needed).
 * @param dt  Forward-Euler step length in seconds.
 */
void advect_scalar_1d_theta_kernel_cylindrical_cgrid(const Field3D& src, Field3D& dst, double dt);

/**
 * @brief TVD-MUSCL flux-form scalar advection in the vertical direction
 *        on the cylindrical Arakawa C-grid (MC limiter).
 *
 * Reads the global @c w field (vertical velocity at z-face[k]) and
 * steps @p src forward by @p dt with Forward Euler:
 *
 *     q^{n+1}[i][j][k] = q^n[i][j][k] - dt * div_F_z[k]
 *
 * Phase C.9: attempts dispatch_vertical_advection_cgrid_backend first
 * and falls back to the CPU body when the GPU pipeline is unavailable.
 *
 * @param src Input scalar field (read-only).
 * @param dst Output scalar field (overwritten; resized to (NR, NTH, NZ) if needed).
 * @param dt  Forward-Euler step length in seconds.
 */
void advect_scalar_1d_z_kernel_cylindrical_cgrid(const Field3D& src, Field3D& dst, double dt);

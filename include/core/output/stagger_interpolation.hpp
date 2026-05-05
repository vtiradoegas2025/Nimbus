/**
 * @file stagger_interpolation.hpp
 * @brief Face-to-center velocity interpolation for output paths (Phase C.8).
 *
 * On the cylindrical Arakawa C-grid (Phase C of the Coordinate Backend
 * Plan), velocity components are stored at cell faces:
 *
 *   u[i][j][k] at r_face[i]  -- right face of cell i
 *   v[i][j][k] at theta_face -- "north" face of cell j (theta_{j+1/2})
 *   w[i][j][k] at z_face[k]  -- top face of cell k
 *
 * Output consumers (npy writer, shm channel for the live viewer, derived
 * field computations) expect velocities at cell centers so that all
 * fields share the same coordinate convention. The helpers in this
 * module produce a cell-center Field3D from a face-staggered Field3D
 * by arithmetic averaging across the two adjacent faces.
 *
 * Conventions (matching `StaggeredCylindricalDerivatives::interp_from_*_face`
 * already used by the dynamics diagnostics in
 * `SupercellCGridScheme::compute_vorticity_diagnostics`):
 *
 *   u_center[i][j][k] = 0.5 * (u[i-1][j][k] + u[i][j][k]),  for i >= 1
 *   u_center[0][j][k] = 0.5 * u[0][j][k]                    (axis cell:
 *                       only the right face exists; the left face is
 *                       the axis r=0 with the implicit u=0 BC)
 *
 *   v_center[i][j][k] = 0.5 * (v[i][j_prev][k] + v[i][j][k])
 *                       where j_prev = (j - 1 + NTH) % NTH (periodic)
 *
 *   w_center[i][j][k] = 0.5 * (w[i][j][k-1] + w[i][j][k]),  for k >= 1
 *   w_center[i][j][0] = 0.5 * w[i][j][0]                    (surface ghost
 *                       below z=0 is the rigid surface w=0)
 *
 * All other fields (rho, p, theta, q*, vorticity, reflectivity, tracer)
 * already live at cell centers on both collocated and C-grid
 * configurations -- they require no interpolation.
 *
 * The helpers are no-ops on collocated inputs: collocated u/v/w already
 * live at cell centers, and the arithmetic-average helpers would
 * smear them. The runtime-side dispatcher in
 * `headless_runtime.cpp` therefore only invokes these helpers when
 * `global_stagger_type == StaggerType::CGrid`. The helpers do not
 * inspect the global stagger flag themselves so unit tests can exercise
 * them in isolation against a synthetic C-grid placement.
 */

#pragma once

#include "core/field3d.hpp"


/**
 * @brief Interpolates a radial velocity Field3D from r-faces to cell centers.
 *
 *   u_center[i][j][k] = 0.5 * (u_face[i-1][j][k] + u_face[i][j][k]),  i >= 1
 *   u_center[0][j][k] = 0.5 * u_face[0][j][k]
 *
 * @param u_face   Source field with u stored at r-face[i].
 * @param u_center Destination cell-center field. Resized to (NR, NTH, NZ)
 *                 if it does not already match.
 */
void interpolate_u_face_to_center(const Field3D& u_face, Field3D& u_center);


/**
 * @brief Interpolates an azimuthal velocity Field3D from theta-faces to
 *        cell centers (periodic in j).
 *
 *   v_center[i][j][k] = 0.5 * (v_face[i][j_prev][k] + v_face[i][j][k])
 *
 * @param v_face   Source field with v stored at theta-face (theta_{j+1/2}).
 * @param v_center Destination cell-center field.
 */
void interpolate_v_face_to_center(const Field3D& v_face, Field3D& v_center);


/**
 * @brief Interpolates a vertical velocity Field3D from z-faces to cell centers.
 *
 *   w_center[i][j][k] = 0.5 * (w_face[i][j][k-1] + w_face[i][j][k]),  k >= 1
 *   w_center[i][j][0] = 0.5 * w_face[i][j][0]
 *
 * @param w_face   Source field with w stored at z-face[k].
 * @param w_center Destination cell-center field.
 */
void interpolate_w_face_to_center(const Field3D& w_face, Field3D& w_center);

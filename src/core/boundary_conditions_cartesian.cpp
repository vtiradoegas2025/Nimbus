/**
 * @file boundary_conditions_cartesian.cpp
 * @brief Cartesian boundary-condition implementation (Phase A.3).
 *
 * Open lateral (zero-gradient) boundaries on the four x/y faces and
 * rigid-lid + rigid-surface boundaries on the two vertical faces, with
 * hydrostatic pressure extrapolation. See the contract in
 * `include/core/boundary_conditions.hpp` for the per-field semantics.
 *
 * The cylindrical implementation lives inline in
 * `src/core/dynamics.cpp::apply_boundary_conditions`. The dispatcher in
 * that same function calls into this file when
 * `global_coordinate_system == CoordinateSystem::Cartesian`.
 */

#include "core/boundary_conditions.hpp"
#include "core/simulation.hpp"
#include "physics/dynamics_base.hpp"

#include <algorithm>
#include <cmath>

namespace
{

/**
 * @brief Zero-gradient copy of every prognostic field on a single (j, k) row
 *        of the i = 0 / i = NR-1 lateral faces.
 *
 * Pulled into a helper because we run the same operation twice (once at the
 * x-min face, once at the x-max face) and the field list is long. Inlined by
 * the compiler.
 */
inline void copy_lateral_x_face(int j, int k, int dst_i, int src_i)
{
    u[dst_i][j][k]       = u[src_i][j][k];
    v_theta[dst_i][j][k] = v_theta[src_i][j][k];
    w[dst_i][j][k]       = w[src_i][j][k];
    rho[dst_i][j][k]     = rho[src_i][j][k];
    p[dst_i][j][k]       = p[src_i][j][k];
    theta[dst_i][j][k]   = theta[src_i][j][k];
    qv[dst_i][j][k]      = qv[src_i][j][k];
    qc[dst_i][j][k]      = qc[src_i][j][k];
    qr[dst_i][j][k]      = qr[src_i][j][k];
    qi[dst_i][j][k]      = qi[src_i][j][k];
    qs[dst_i][j][k]      = qs[src_i][j][k];
    qg[dst_i][j][k]      = qg[src_i][j][k];
    qh[dst_i][j][k]      = qh[src_i][j][k];
}

/**
 * @brief Zero-gradient copy of every prognostic field on a single (i, k) row
 *        of the j = 0 / j = NTH-1 lateral faces.
 */
inline void copy_lateral_y_face(int i, int k, int dst_j, int src_j)
{
    u[i][dst_j][k]       = u[i][src_j][k];
    v_theta[i][dst_j][k] = v_theta[i][src_j][k];
    w[i][dst_j][k]       = w[i][src_j][k];
    rho[i][dst_j][k]     = rho[i][src_j][k];
    p[i][dst_j][k]       = p[i][src_j][k];
    theta[i][dst_j][k]   = theta[i][src_j][k];
    qv[i][dst_j][k]      = qv[i][src_j][k];
    qc[i][dst_j][k]      = qc[i][src_j][k];
    qr[i][dst_j][k]      = qr[i][src_j][k];
    qi[i][dst_j][k]      = qi[i][src_j][k];
    qs[i][dst_j][k]      = qs[i][src_j][k];
    qg[i][dst_j][k]      = qg[i][src_j][k];
    qh[i][dst_j][k]      = qh[i][src_j][k];
}

}  // namespace

void apply_cartesian_boundary_conditions()
{
    // ---------------------------------------------------------------
    // Lateral x faces (i = 0 and i = NR-1): open / zero-gradient.
    //
    // Cartesian has no singular axis at i = 0, so there is no antisymmetric
    // ghost cell convention here. Every field copies its first interior
    // neighbor outward — the same convention used by all six Cartesian
    // boundaries except w on the rigid lid/surface.
    // ---------------------------------------------------------------
    for (int j = 0; j < NTH; ++j)
    {
        for (int k = 0; k < NZ; ++k)
        {
            copy_lateral_x_face(j, k, /*dst_i=*/0,      /*src_i=*/1);
            copy_lateral_x_face(j, k, /*dst_i=*/NR - 1, /*src_i=*/NR - 2);
        }
    }

    // ---------------------------------------------------------------
    // Lateral y faces (j = 0 and j = NTH-1): open / zero-gradient.
    //
    // Cartesian has no periodic theta wraparound. Each face copies its
    // interior neighbor outward.
    // ---------------------------------------------------------------
    for (int i = 0; i < NR; ++i)
    {
        for (int k = 0; k < NZ; ++k)
        {
            copy_lateral_y_face(i, k, /*dst_j=*/0,       /*src_j=*/1);
            copy_lateral_y_face(i, k, /*dst_j=*/NTH - 1, /*src_j=*/NTH - 2);
        }
    }

    // ---------------------------------------------------------------
    // Vertical faces (k = 0 and k = NZ-1).
    //
    // - w (vertical velocity): Dirichlet 0 at both faces (rigid surface
    //   below, rigid lid above). The model is non-hydrostatic compressible
    //   so this also lets vertically propagating sound waves bounce off
    //   the lid; the diffusion sponge layer further down handles damping.
    //
    // - u, v_theta (carrying u_x, u_y): zero-gradient (free slip). Stress
    //   from the surface is reintroduced by the boundary-layer scheme.
    //
    // - rho: zero-gradient. The base-state density profile is smooth and
    //   `rho[k=1]` is a good proxy for `rho[k=0]`.
    //
    // - p: hydrostatic extrapolation
    //     p[NZ-1] = p[NZ-2] - rho_top * g * dz
    //     p[0]    = p[1]    + rho_bot * g * dz
    //   This is NOT a zero-gradient BC. Using zero-gradient causes the
    //   centered dp/dz stencil at the next-to-boundary cell to halve, the
    //   -dp/dz/rho - g residual becomes -g/2 ≈ -4.9 m/s^2, and the model
    //   accelerates downward at the top edge until it slams into the
    //   velocity clamps. Same logic in reverse at the surface. The
    //   cylindrical BC has the same fix; we replicate it verbatim here.
    //
    // - theta and the moisture variables: zero-gradient. They do not
    //   appear directly in the discrete hydrostatic relation.
    // ---------------------------------------------------------------
    const double g_local  = static_cast<double>(g);
    const double dz_local = (std::isfinite(dz) && dz > 0.0) ? dz : 1.0;

    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            w[i][j][0]      = 0.0f;
            w[i][j][NZ - 1] = 0.0f;

            u[i][j][0]      = u[i][j][1];
            u[i][j][NZ - 1] = u[i][j][NZ - 2];

            v_theta[i][j][0]      = v_theta[i][j][1];
            v_theta[i][j][NZ - 1] = v_theta[i][j][NZ - 2];

            rho[i][j][0]      = rho[i][j][1];
            rho[i][j][NZ - 1] = rho[i][j][NZ - 2];

            // Hydrostatic extrapolation. rho is clamped to a small positive
            // floor so the multiplication is well defined even if the
            // adjacent interior cell is degenerate.
            const double rho_top = std::max(static_cast<double>(rho[i][j][NZ - 2]), 1.0e-3);
            const double rho_bot = std::max(static_cast<double>(rho[i][j][1]),      1.0e-3);
            p[i][j][NZ - 1] = static_cast<float>(
                static_cast<double>(p[i][j][NZ - 2]) - rho_top * g_local * dz_local);
            p[i][j][0] = static_cast<float>(
                static_cast<double>(p[i][j][1]) + rho_bot * g_local * dz_local);

            theta[i][j][0]      = theta[i][j][1];
            theta[i][j][NZ - 1] = theta[i][j][NZ - 2];

            qv[i][j][0] = qv[i][j][1]; qv[i][j][NZ - 1] = qv[i][j][NZ - 2];
            qc[i][j][0] = qc[i][j][1]; qc[i][j][NZ - 1] = qc[i][j][NZ - 2];
            qr[i][j][0] = qr[i][j][1]; qr[i][j][NZ - 1] = qr[i][j][NZ - 2];
            qi[i][j][0] = qi[i][j][1]; qi[i][j][NZ - 1] = qi[i][j][NZ - 2];
            qs[i][j][0] = qs[i][j][1]; qs[i][j][NZ - 1] = qs[i][j][NZ - 2];
            qg[i][j][0] = qg[i][j][1]; qg[i][j][NZ - 1] = qg[i][j][NZ - 2];
            qh[i][j][0] = qh[i][j][1]; qh[i][j][NZ - 1] = qh[i][j][NZ - 2];
        }
    }
}

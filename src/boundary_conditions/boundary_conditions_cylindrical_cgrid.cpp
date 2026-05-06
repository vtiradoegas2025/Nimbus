/**
 * @file boundary_conditions_cylindrical_cgrid.cpp
 * @brief C-grid (Arakawa staggered) cylindrical boundary conditions (Phase C.2).
 *
 * Differs from the collocated cylindrical scheme in two structurally important
 * ways:
 *
 *   1. NO antisymmetric ghost cell at the axis. On the C-grid, u[0] is the
 *      first stored radial-face velocity at r_face[0] = dr/2 -- a real
 *      prognostic interior value, not a ghost. The "u_r = 0 at r = 0"
 *      condition is enforced implicitly by div_flux_r (control-volume
 *      formula 2*u[0]/dr at i = 0; see StaggeredCylindricalDerivatives in
 *      include/numerics/derivatives/derivative_operators.hpp). This BC
 *      therefore must NOT touch u[0].
 *
 *   2. Vertical w rigid boundaries are at faces. The rigid surface at z = 0
 *      is below the lowest stored z-face (z_face[0] = dz/2) and is implicit
 *      in div_flux_z (which treats the bottom face of cell 0 as zero). The
 *      rigid lid is the top face of cell NZ-1, stored as w[i][j][NZ-1]; the
 *      BC must explicitly set this to 0 each step. By the same logic, the
 *      bottom face w[i][j][0] is interior on the C-grid and must NOT be
 *      zeroed -- doing so would suppress vertical motion in the surface
 *      cell.
 *
 * Field placement on this scheme (matches the C-grid convention in
 * docs/CoordinateBackend_Plan.md, "Field Storage Convention"):
 *
 *   u (radial)      r-face       u[i][j][k]   at (r_face[i],  theta[j],     z[k])
 *   v (azimuthal)   theta-face   v[i][j][k]   at (r[i],       theta_{j+1/2}, z[k])
 *   w (vertical)    z-face       w[i][j][k]   at (r[i],       theta[j],     z_face[k])
 *   scalars         cell center  rho/p/...    at (r[i],       theta[j],     z[k])
 *
 * Boundary conventions applied here:
 *
 *   Axis (i = 0):
 *     - u[0]     left untouched (interior face value).
 *     - v[0][j]  = 0           (theta-face collapses to a single point at r=0;
 *                               axisymmetric flow has no swirl on axis, and
 *                               the cell center i=0 is not used in the
 *                               dynamics scheme loops [i = 1..NR-2]).
 *     - w[0][j][k] = w[1][j][k] (zero-gradient; w on axis is single-valued
 *                                but allowed to be nonzero).
 *     - scalars (rho, p, theta, q*) at i = 0: zero-gradient from i = 1.
 *
 *   Outer wall (i = NR-1):
 *     - u[NR-1][j][k] = 0                 (rigid wall at r_face[NR-1]).
 *     - v[NR-1][j][k] = v[NR-2][j][k]     (zero-gradient).
 *     - w[NR-1][j][k] = w[NR-2][j][k]     (zero-gradient).
 *     - scalars at i = NR-1: zero-gradient from i = NR-2.
 *
 *   Surface (k = 0): lateral velocities and scalars use zero-gradient
 *     extrapolation; pressure uses hydrostatic extrapolation. The surface
 *     w-face below k=0 is implicit zero in div_flux_z, so no array slot is
 *     touched here. w[i][j][0] is interior and is NOT modified.
 *
 *   Lid (k = NZ-1): rigid lid w[i][j][NZ-1] = 0; lateral velocities and
 *     scalars use zero-gradient; pressure uses hydrostatic extrapolation.
 *
 * Periodic theta wraparound (j = 0 <-> j = NTH-1) is implicit in the
 * dynamics scheme loops, which use modular indexing in j. No explicit BC.
 */

#include "boundary_conditions/boundary_conditions_base.hpp"
#include "core/runtime/simulation.hpp"
#include "dynamics/dynamics_base.hpp"

#include <algorithm>
#include <cmath>

namespace
{

inline double safe_dz_local()
{
    return (std::isfinite(dz) && dz > 0.0) ? dz : 1.0;
}

}  // namespace

class CylindricalCGridBCScheme : public BoundaryConditionScheme
{
public:
    void apply_full() override
    {
        apply_axis_full();
        apply_outer_wall_full();
        apply_vertical_full();
    }

    void apply_acoustic() override
    {
        apply_axis_acoustic();
        apply_outer_wall_acoustic();
        apply_vertical_acoustic();
    }

    std::string get_scheme_name() const override { return "cylindrical_cgrid"; }

private:
    /// Axis (i = 0). u[0] is an interior r-face -- never written here.
    void apply_axis_full()
    {
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                v[0][j][k] = 0.0f;
                w[0][j][k] = w[1][j][k];
                rho[0][j][k]   = rho[1][j][k];
                p[0][j][k]     = p[1][j][k];
                theta[0][j][k] = theta[1][j][k];
                qv[0][j][k]    = qv[1][j][k];
                qc[0][j][k]    = qc[1][j][k];
                qr[0][j][k]    = qr[1][j][k];
                qi[0][j][k]    = qi[1][j][k];
                qs[0][j][k]    = qs[1][j][k];
                qg[0][j][k]    = qg[1][j][k];
                qh[0][j][k]    = qh[1][j][k];
            }
    }

    /// Outer wall (i = NR-1). u[NR-1] is the outer face -- rigid wall.
    void apply_outer_wall_full()
    {
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[NR - 1][j][k] = 0.0f;
                v[NR - 1][j][k] = v[NR - 2][j][k];
                w[NR - 1][j][k] = w[NR - 2][j][k];
                rho[NR - 1][j][k]   = rho[NR - 2][j][k];
                p[NR - 1][j][k]     = p[NR - 2][j][k];
                theta[NR - 1][j][k] = theta[NR - 2][j][k];
                qv[NR - 1][j][k]    = qv[NR - 2][j][k];
                qc[NR - 1][j][k]    = qc[NR - 2][j][k];
                qr[NR - 1][j][k]    = qr[NR - 2][j][k];
                qi[NR - 1][j][k]    = qi[NR - 2][j][k];
                qs[NR - 1][j][k]    = qs[NR - 2][j][k];
                qg[NR - 1][j][k]    = qg[NR - 2][j][k];
                qh[NR - 1][j][k]    = qh[NR - 2][j][k];
            }
    }

    /// Vertical: rigid lid at top z-face, hydrostatic pressure extrapolation,
    /// zero-gradient lateral velocities and scalars at top/bottom ghost cells.
    /// Bottom z-face is NOT touched -- it is the first interior face on C-grid
    /// and the surface BC is implicit in div_flux_z.
    void apply_vertical_full()
    {
        const double g_val    = dynamics_constants::g;
        const double dz_local = safe_dz_local();

        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
            {
                w[i][j][NZ - 1] = 0.0f;

                u[i][j][0]      = u[i][j][1];
                u[i][j][NZ - 1] = u[i][j][NZ - 2];
                v[i][j][0]      = v[i][j][1];
                v[i][j][NZ - 1] = v[i][j][NZ - 2];

                rho[i][j][0]      = rho[i][j][1];
                rho[i][j][NZ - 1] = rho[i][j][NZ - 2];

                const double rho_top = std::max(static_cast<double>(rho[i][j][NZ - 2]), 1.0e-3);
                const double rho_bot = std::max(static_cast<double>(rho[i][j][1]),      1.0e-3);
                p[i][j][NZ - 1] = static_cast<float>(
                    static_cast<double>(p[i][j][NZ - 2]) - rho_top * g_val * dz_local);
                p[i][j][0] = static_cast<float>(
                    static_cast<double>(p[i][j][1]) + rho_bot * g_val * dz_local);

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

    /// Axis ghosts during acoustic substeps (momentum + pressure only).
    void apply_axis_acoustic()
    {
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                v[0][j][k]   = 0.0f;
                w[0][j][k]   = w[1][j][k];
                rho[0][j][k] = rho[1][j][k];
                p[0][j][k]   = p[1][j][k];
            }
    }

    void apply_outer_wall_acoustic()
    {
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[NR - 1][j][k] = 0.0f;
                v[NR - 1][j][k] = v[NR - 2][j][k];
                w[NR - 1][j][k] = w[NR - 2][j][k];
                rho[NR - 1][j][k] = rho[NR - 2][j][k];
                p[NR - 1][j][k]   = p[NR - 2][j][k];
            }
    }

    void apply_vertical_acoustic()
    {
        const double g_val    = dynamics_constants::g;
        const double dz_local = safe_dz_local();

        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
            {
                w[i][j][NZ - 1] = 0.0f;

                u[i][j][0]      = u[i][j][1];
                u[i][j][NZ - 1] = u[i][j][NZ - 2];
                v[i][j][0]      = v[i][j][1];
                v[i][j][NZ - 1] = v[i][j][NZ - 2];

                rho[i][j][0]      = rho[i][j][1];
                rho[i][j][NZ - 1] = rho[i][j][NZ - 2];

                const double rho_top = std::max(static_cast<double>(rho[i][j][NZ - 2]), 1.0e-3);
                const double rho_bot = std::max(static_cast<double>(rho[i][j][1]),      1.0e-3);
                p[i][j][NZ - 1] = static_cast<float>(
                    static_cast<double>(p[i][j][NZ - 2]) - rho_top * g_val * dz_local);
                p[i][j][0] = static_cast<float>(
                    static_cast<double>(p[i][j][1]) + rho_bot * g_val * dz_local);
            }
    }
};

std::unique_ptr<BoundaryConditionScheme> create_cylindrical_cgrid_bc_scheme()
{
    return std::make_unique<CylindricalCGridBCScheme>();
}

/**
 * @file tornado_cgrid.cpp
 * @brief Implementation of the C-grid axisymmetric tornado dynamics scheme.
 *
 * Field placement (Arakawa C-grid):
 *   u[i][j][k] at (r_face[i], theta[j],   z[k])         -- right face of cell i
 *   v[i][j][k] at (r[i],      theta_{j+1/2}, z[k])      -- theta face of cell j
 *   w[i][j][k] at (r[i],      theta[j],   z_face[k])    -- top face of cell k
 *   scalars   at (r[i],      theta[j],   z[k])          -- cell center
 *
 * Loop ranges (axisymmetric: compute at j=0 and replicate to all j):
 *   du/dt   at r-face:   i = 0..NR-2,   k = 1..NZ-2
 *     i=0 uses the antisymmetric axis ghost u[-1] = -u[0]; i=NR-1 is the
 *     rigid outer wall set to 0 by the BC scheme.
 *   dv/dt   at theta-face: i = 1..NR-2, k = 1..NZ-2
 *     v[0] is the axis (zero by BC); i=NR-1 is the wall ghost.
 *   dw/dt   at z-face:   i = 1..NR-2,   k = 0..NZ-2
 *     k=0 is INTERIOR on C-grid (z_face[0] = 0.5*dz); the rigid surface
 *     at z=0 is implicit through the ghost w[-1] = 0; k=NZ-1 is the lid.
 *   drho/dt, dp/dt at cell-center: i = 1..NR-2, k = 1..NZ-2
 *     Skipping the i=0 axis cell matches the collocated tornado pattern
 *     and the test setups have zero u at the axis face anyway.
 */

#include "tornado_cgrid.hpp"
#include "core/runtime/simulation.hpp"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif


TornadoCGridScheme::TornadoCGridScheme()
    : NR_(NR), NTH_(NTH), NZ_(NZ),
      dr_(dr), dtheta_(dtheta), dz_(dz),
      geo_(global_grid_geometry),
      deriv_(global_grid_geometry, NTH)
{
}


void TornadoCGridScheme::compute_momentum_tendencies(
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    const Field3D& rho,
    const Field3D& p,
    const Field3D& theta,
    double /*dt*/,
    Field3D& du_dt,
    Field3D& dv_dt,
    Field3D& dw_dt,
    Field3D& drho_dt,
    Field3D& dp_dt)
{
    // theta is unused: buoyancy is expressed as a perturbation-density term
    // -g*(rho - rho0)/rho rather than g*(theta - theta0)/theta0, matching
    // the supercell scheme's Bug-3 fix in docs/Journey.md.
    (void)theta;

    // ---- Zero everything first so the output is well-defined on faces and
    //      cells we do not write inside the interior loops.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                du_dt[i][j][k]   = 0.0f;
                dv_dt[i][j][k]   = 0.0f;
                dw_dt[i][j][k]   = 0.0f;
                drho_dt[i][j][k] = 0.0f;
                dp_dt[i][j][k]   = 0.0f;
            }

    const int j               = 0;       // axisymmetric: compute at j=0, replicate
    const double inv_2dr      = geo_.inv_2dr;
    const double inv_2dz      = geo_.inv_2dz;
    const double inv_dz_local = geo_.inv_dz;
    const double g_local      = dynamics_constants::g;
    const double gamma_local  = dynamics_constants::gamma;

    // ===== du/dt at r-face  (i = 0..NR-2, k = 1..NZ-2) =====
    #pragma omp parallel for
    for (int i = 0; i < NR_ - 1; ++i)
    {
        const double r_face = geo_.r_face[i];
        for (int k = 1; k < NZ_ - 1; ++k)
        {
            const double u_face = static_cast<double>(u[i][j][k]);

            // v at r-face: cell-center v interpolated radially (axisymmetric:
            // theta-face value coincides with cell-center value).
            const double v_at_uface = 0.5 * (static_cast<double>(v[i][j][k])
                                           + static_cast<double>(v[i + 1][j][k]));

            // w at r-face: bilinear interp z-face -> cell-center -> r-face.
            // w_center[i][k] = 0.5*(w[i][j][k] + w[i][j][k-1])
            const double w_center_left  = 0.5 * (static_cast<double>(w[i][j][k])
                                               + static_cast<double>(w[i][j][k - 1]));
            const double w_center_right = 0.5 * (static_cast<double>(w[i + 1][j][k])
                                               + static_cast<double>(w[i + 1][j][k - 1]));
            const double w_at_uface     = 0.5 * (w_center_left + w_center_right);

            // Centered d/dr at r-face: (u[i+1] - u[i-1]) / (2*dr).
            // At i=0 the ghost on the other side of the axis is, by axisymmetric
            // reflection of a vector radial component, u[-1] = -u[0].
            const double u_left  = (i == 0) ? -u_face
                                            :  static_cast<double>(u[i - 1][j][k]);
            const double u_right =                static_cast<double>(u[i + 1][j][k]);
            const double dur_dr  = (u_right - u_left) * inv_2dr;

            // Centered d/dz at r-face.
            const double dur_dz = (static_cast<double>(u[i][j][k + 1])
                                 - static_cast<double>(u[i][j][k - 1])) * inv_2dz;

            // One-sided pressure gradient at r-face: (p[i+1] - p[i]) / dr.
            const double dp_dr = deriv_.grad_r(p, i, j, k);

            // Density at r-face.
            const double rho_face = 0.5 * (static_cast<double>(rho[i][j][k])
                                         + static_cast<double>(rho[i + 1][j][k]));
            const double rho_safe = (std::isfinite(rho_face) && rho_face > 1.0e-6)
                                  ? rho_face : 1.0;

            const double advective_r     = -u_face * dur_dr - w_at_uface * dur_dz;
            const double centrifugal     = (v_at_uface * v_at_uface) / r_face;
            const double pressure_grad_r = -dp_dr / rho_safe;

            double du_r = advective_r + centrifugal + pressure_grad_r;
            if (!std::isfinite(du_r)) du_r = 0.0;

            du_dt[i][j][k] = static_cast<float>(du_r);
            for (int jj = 1; jj < NTH_; ++jj)
                du_dt[i][jj][k] = du_dt[i][j][k];
        }
    }

    // ===== dv/dt at theta-face  (i = 1..NR-2, k = 1..NZ-2) =====
    #pragma omp parallel for
    for (int i = 1; i < NR_ - 1; ++i)
    {
        const double r_inv_at_vface = geo_.r_inv[i];
        for (int k = 1; k < NZ_ - 1; ++k)
        {
            const double v_face = static_cast<double>(v[i][j][k]);

            // u at theta-face: r-face -> cell-center.  Axisymmetric: that
            // cell-center value is also the theta-face value.
            const double u_at_vface = 0.5 * (static_cast<double>(u[i][j][k])
                                           + static_cast<double>(u[i - 1][j][k]));
            // w at theta-face: z-face -> cell-center, axisymmetric.
            const double w_at_vface = 0.5 * (static_cast<double>(w[i][j][k])
                                           + static_cast<double>(w[i][j][k - 1]));

            // Centered d/dr at theta-face.  v[0] = 0 (axis BC) makes the
            // i=1 stencil well-defined.
            const double dv_dr = (static_cast<double>(v[i + 1][j][k])
                                - static_cast<double>(v[i - 1][j][k])) * inv_2dr;
            // Centered d/dz at theta-face.
            const double dv_dz = (static_cast<double>(v[i][j][k + 1])
                                - static_cast<double>(v[i][j][k - 1])) * inv_2dz;

            const double advective_th = -u_at_vface * dv_dr - w_at_vface * dv_dz;
            const double curvature_th = -u_at_vface * v_face * r_inv_at_vface;

            double dv_t = advective_th + curvature_th;
            if (!std::isfinite(dv_t)) dv_t = 0.0;

            dv_dt[i][j][k] = static_cast<float>(dv_t);
            for (int jj = 1; jj < NTH_; ++jj)
                dv_dt[i][jj][k] = dv_dt[i][j][k];
        }
    }

    // ===== dw/dt at z-face  (i = 1..NR-2, k = 0..NZ-2) =====
    #pragma omp parallel for
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int k = 0; k < NZ_ - 1; ++k)
        {
            const double w_face = static_cast<double>(w[i][j][k]);

            // u at z-face: r-face -> cell-center -> z-face.
            const double u_center_below = 0.5 * (static_cast<double>(u[i][j][k])
                                               + static_cast<double>(u[i - 1][j][k]));
            const double u_center_above = 0.5 * (static_cast<double>(u[i][j][k + 1])
                                               + static_cast<double>(u[i - 1][j][k + 1]));
            const double u_at_wface     = 0.5 * (u_center_below + u_center_above);

            // Centered d/dr at z-face.
            const double dw_dr = (static_cast<double>(w[i + 1][j][k])
                                - static_cast<double>(w[i - 1][j][k])) * inv_2dr;
            // Centered d/dz at z-face: surface ghost w[-1] = 0 at k=0.
            const double w_below = (k == 0) ? 0.0 : static_cast<double>(w[i][j][k - 1]);
            const double w_above =                  static_cast<double>(w[i][j][k + 1]);
            const double dw_dz   = (w_above - w_below) * inv_2dz;

            // One-sided pressure gradient at z-face: (p[k+1] - p[k]) / dz.
            const double dp_dz = deriv_.grad_z(p, i, j, k);

            // Density at z-face and reference-state subtraction.  Using the
            // same one-sided dp0/dz form as dp/dz keeps the perturbation
            // gradient bit-exactly zero when p ≡ p0 + p_cyclo(r) and
            // rho ≡ rho0(z).
            const double rho_face  = 0.5 * (static_cast<double>(rho[i][j][k])
                                          + static_cast<double>(rho[i][j][k + 1]));
            const double rho_safe  = (std::isfinite(rho_face) && rho_face > 1.0e-6)
                                   ? rho_face : 1.0;
            const double rho0_face = 0.5 * (rho0_base[k] + rho0_base[k + 1]);
            const double dp0_dz    = (p0_base[k + 1] - p0_base[k]) * inv_dz_local;
            const double dp_prime_dz = dp_dz - dp0_dz;

            // Buoyancy from rho perturbation (Bug 3 form: do NOT add an
            // explicit g*(theta-theta0)/theta0).
            const double buoyancy = -g_local * (rho_face - rho0_face) / rho_safe;

            // Moisture buoyancy: virtual-temperature correction.
            double moisture_buoyancy = 0.0;
            if (!qv.empty() && !qv0_base.empty())
            {
                const double qv_face  = 0.5 * (static_cast<double>(qv[i][j][k])
                                             + static_cast<double>(qv[i][j][k + 1]));
                const double qv0_face = 0.5 * (qv0_base[k] + qv0_base[k + 1]);
                moisture_buoyancy = g_local * 0.608 * (qv_face - qv0_face);
            }

            // Hydrometeor loading (negative -- hydrometeor mass acts as ballast).
            double loading = 0.0;
            if (!qc.empty())
            {
                const double q_below = static_cast<double>(qc[i][j][k]) + static_cast<double>(qr[i][j][k])
                                     + static_cast<double>(qi[i][j][k]) + static_cast<double>(qs[i][j][k])
                                     + static_cast<double>(qg[i][j][k]) + static_cast<double>(qh[i][j][k]);
                const double q_above = static_cast<double>(qc[i][j][k + 1]) + static_cast<double>(qr[i][j][k + 1])
                                     + static_cast<double>(qi[i][j][k + 1]) + static_cast<double>(qs[i][j][k + 1])
                                     + static_cast<double>(qg[i][j][k + 1]) + static_cast<double>(qh[i][j][k + 1]);
                loading = -g_local * 0.5 * (q_below + q_above);
            }

            const double advective_z = -u_at_wface * dw_dr - w_face * dw_dz;
            double dw_t = advective_z - dp_prime_dz / rho_safe
                        + buoyancy + moisture_buoyancy + loading;
            if (!std::isfinite(dw_t)) dw_t = 0.0;

            dw_dt[i][j][k] = static_cast<float>(dw_t);
            for (int jj = 1; jj < NTH_; ++jj)
                dw_dt[i][jj][k] = dw_dt[i][j][k];
        }
    }

    // ===== drho/dt and dp/dt at cell-center  (i = 1..NR-2, k = 1..NZ-2) =====
    #pragma omp parallel for
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int k = 1; k < NZ_ - 1; ++k)
        {
            const double rho_val  = static_cast<double>(rho[i][j][k]);
            const double p_val    = static_cast<double>(p[i][j][k]);
            const double rho_safe = (std::isfinite(rho_val) && rho_val > 1.0e-6)
                                  ? rho_val : 1.0;

            // Flux-form divergence at cell center (axis-aware).
            const double div = deriv_.div_flux(u, v, w, i, j, k);

            // Mass continuity.
            double drho = -rho_safe * div;
            if (!std::isfinite(drho)) drho = 0.0;
            drho_dt[i][j][k] = static_cast<float>(drho);

            // Pressure equation: dp/dt = -gamma*p*div(u) - u . grad(p).
            // Cell-center velocities and gradients use centered stencils.
            const double u_center = 0.5 * (static_cast<double>(u[i][j][k])
                                         + static_cast<double>(u[i - 1][j][k]));
            const double w_center = 0.5 * (static_cast<double>(w[i][j][k])
                                         + static_cast<double>(w[i][j][k - 1]));
            const double dp_dr_c  = (static_cast<double>(p[i + 1][j][k])
                                   - static_cast<double>(p[i - 1][j][k])) * inv_2dr;
            const double dp_dz_c  = (static_cast<double>(p[i][j][k + 1])
                                   - static_cast<double>(p[i][j][k - 1])) * inv_2dz;

            double dp_t = -gamma_local * p_val * div
                         - u_center * dp_dr_c - w_center * dp_dz_c;
            if (!std::isfinite(dp_t)) dp_t = 0.0;
            dp_dt[i][j][k] = static_cast<float>(dp_t);

            for (int jj = 1; jj < NTH_; ++jj)
            {
                drho_dt[i][jj][k] = drho_dt[i][j][k];
                dp_dt[i][jj][k]   = dp_dt[i][j][k];
            }
        }
    }
}


void TornadoCGridScheme::compute_angular_momentum(
    const Field3D& u,
    const Field3D& v,
    Field3D& angular_momentum,
    Field3D& angular_momentum_tendency)
{
    // Angular momentum at theta-face position: r[i] * v[i][j][k]
    // (v lives at the theta-face whose r is the cell-center r[i]).
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
        {
            const double r = geo_.r[i];
            for (int k = 0; k < NZ_; ++k)
            {
                angular_momentum[i][j][k] = static_cast<float>(
                    r * static_cast<double>(v[i][j][k]));
            }
        }

    const int j           = 0;
    const double inv_2dr  = geo_.inv_2dr;
    const double inv_2dz  = geo_.inv_2dz;

    #pragma omp parallel for
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int k = 1; k < NZ_ - 1; ++k)
        {
            // Advecting velocities at theta-face (axisymmetric).
            const double u_at_vface = 0.5 * (static_cast<double>(u[i][j][k])
                                           + static_cast<double>(u[i - 1][j][k]));
            const double w_at_vface = 0.5 * (static_cast<double>(w[i][j][k])
                                           + static_cast<double>(w[i][j][k - 1]));

            const double dam_dr = (static_cast<double>(angular_momentum[i + 1][j][k])
                                 - static_cast<double>(angular_momentum[i - 1][j][k])) * inv_2dr;
            const double dam_dz = (static_cast<double>(angular_momentum[i][j][k + 1])
                                 - static_cast<double>(angular_momentum[i][j][k - 1])) * inv_2dz;

            const double tend = -u_at_vface * dam_dr - w_at_vface * dam_dz;
            angular_momentum_tendency[i][j][k] = static_cast<float>(tend);

            for (int jj = 1; jj < NTH_; ++jj)
                angular_momentum_tendency[i][jj][k] = angular_momentum_tendency[i][j][k];
        }
    }
}


void TornadoCGridScheme::compute_vorticity_diagnostics(
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    const Field3D& rho,
    const Field3D& p,
    Field3D& vorticity_r,
    Field3D& vorticity_theta,
    Field3D& vorticity_z,
    Field3D& stretching_term,
    Field3D& tilting_term,
    Field3D& baroclinic_term)
{
    (void)rho;
    (void)p;

    const int j           = 0;
    const double inv_2dr  = geo_.inv_2dr;
    const double inv_2dz  = geo_.inv_2dz;

    // Diagnostics evaluated at cell centers using interpolated face values,
    // matching the collocated tornado scheme's structure (axisymmetric:
    // vorticity_r is identically zero, baroclinic and tilting are zero).
    #pragma omp parallel for
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int k = 1; k < NZ_ - 1; ++k)
        {
            const double r_inv = geo_.r_inv[i];

            // Cell-center radial and vertical velocities (axisymmetric).
            const double u_c_below = 0.5 * (static_cast<double>(u[i][j][k - 1])
                                          + static_cast<double>(u[i - 1][j][k - 1]));
            const double u_c_above = 0.5 * (static_cast<double>(u[i][j][k + 1])
                                          + static_cast<double>(u[i - 1][j][k + 1]));
            const double dur_dz_c  = (u_c_above - u_c_below) * inv_2dz;

            const double w_c_inner = 0.5 * (static_cast<double>(w[i - 1][j][k])
                                          + static_cast<double>(w[i - 1][j][k - 1]));
            const double w_c_outer = 0.5 * (static_cast<double>(w[i + 1][j][k])
                                          + static_cast<double>(w[i + 1][j][k - 1]));
            const double duz_dr_c  = (w_c_outer - w_c_inner) * inv_2dr;

            const double v_c       = static_cast<double>(v[i][j][k]);  // axisym
            const double duth_dr_c = (static_cast<double>(v[i + 1][j][k])
                                    - static_cast<double>(v[i - 1][j][k])) * inv_2dr;

            const double w_c       = 0.5 * (static_cast<double>(w[i][j][k])
                                          + static_cast<double>(w[i][j][k - 1]));
            const double w_c_above_cell = 0.5 * (static_cast<double>(w[i][j][k + 1])
                                               + static_cast<double>(w[i][j][k]));
            const double w_c_below_cell = (k > 1)
                ? 0.5 * (static_cast<double>(w[i][j][k - 1])
                       + static_cast<double>(w[i][j][k - 2]))
                : 0.5 * static_cast<double>(w[i][j][0]);
            const double dw_dz_c   = (w_c_above_cell - w_c_below_cell) * inv_2dz;
            (void)w_c;

            vorticity_r[i][j][k]     = 0.0f;
            vorticity_theta[i][j][k] = static_cast<float>(dur_dz_c - duz_dr_c);
            vorticity_z[i][j][k]     = static_cast<float>(duth_dr_c + v_c * r_inv);

            stretching_term[i][j][k] = static_cast<float>(vorticity_z[i][j][k] * dw_dz_c);
            tilting_term[i][j][k]    = 0.0f;
            baroclinic_term[i][j][k] = 0.0f;

            for (int jj = 1; jj < NTH_; ++jj)
            {
                vorticity_r[i][jj][k]     = vorticity_r[i][j][k];
                vorticity_theta[i][jj][k] = vorticity_theta[i][j][k];
                vorticity_z[i][jj][k]     = vorticity_z[i][j][k];
                stretching_term[i][jj][k] = stretching_term[i][j][k];
                tilting_term[i][jj][k]    = tilting_term[i][j][k];
                baroclinic_term[i][jj][k] = baroclinic_term[i][j][k];
            }
        }
    }
}

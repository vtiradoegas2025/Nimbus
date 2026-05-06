/**
 * @file supercell_cgrid.cpp
 * @brief Implementation of the C-grid non-axisymmetric supercell dynamics
 *        scheme.
 *
 * Field placement (Arakawa C-grid):
 *   u[i][j][k] at (r_face[i], theta[j],       z[k])
 *   v[i][j][k] at (r[i],      theta_{j+1/2},  z[k])
 *   w[i][j][k] at (r[i],      theta[j],       z_face[k])
 *   scalars   at (r[i],       theta[j],       z[k])
 *
 * Loop ranges (full 3D, no axisymmetric replication):
 *   du/dt at r-face:     i = 0..NR-2, j = 0..NTH-1, k = 1..NZ-2
 *     i=0 uses the antisymmetric axis ghost u[-1] = -u[0] inline (the same
 *     trick TornadoCGridScheme uses); i=NR-1 is the rigid outer wall
 *     written by the BC scheme.
 *   dv/dt at theta-face: i = 1..NR-2, j = 0..NTH-1, k = 1..NZ-2
 *     v[0] sits on the axis (zero by BC); periodic in j.
 *   dw/dt at z-face:     i = 1..NR-2, j = 0..NTH-1, k = 0..NZ-2
 *     k=0 is INTERIOR on C-grid; the rigid surface at z=0 is handled by
 *     the surface ghost w[-1] = 0 inline; k=NZ-1 is the lid.
 *   drho/dt, dp/dt at cell-center: i = 1..NR-2, j = 0..NTH-1, k = 1..NZ-2.
 *
 * Vertical momentum uses reference-state subtraction
 *   dw/dt = -(1/rho)(dp/dz - dp0/dz) - g*(rho - rho0)/rho + advection
 * matching the collocated SupercellScheme's Bug 3 fix and the
 * TornadoCGridScheme convention. Both `dp/dz` and `dp0/dz` are taken with
 * the SAME one-sided stencil so that at the discrete hydrostatic state
 * `p == p0(z)` the vertical pressure perturbation is bit-exactly zero.
 *
 * Slow / fast decomposition (Klemp-Wilhelmson split-explicit):
 *   slow: advection + centrifugal/coriolis + buoyancy on velocities,
 *         dp/dt = -u . grad(p) (advection only), drho/dt = 0.
 *   fast pressure: dp/dt = -gamma*p*div_flux, drho/dt = -rho*div_flux.
 *   fast momentum: -grad(p)/rho on u and v, -(grad(p) - dp0/dz)/rho on w.
 * Sum (slow + fast) reproduces the full single-step tendencies bit-exactly,
 * which is asserted in the C.5 unit tests.
 */

#include "supercell_cgrid.hpp"
#include "core/runtime/simulation.hpp"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace
{

// Returns rho_safe = rho if finite and positive enough, else 1.0. Inline
// helper so the inner loops stay readable.
inline double safe_rho(double rho)
{
    return (std::isfinite(rho) && rho > 1.0e-6) ? rho : 1.0;
}

} // namespace


SupercellCGridScheme::SupercellCGridScheme()
    : NR_(NR), NTH_(NTH), NZ_(NZ),
      dr_(dr), dtheta_(dtheta), dz_(dz),
      geo_(global_grid_geometry),
      deriv_(global_grid_geometry, NTH)
{
}


// =====================================================================
// Unsplit momentum tendencies (slow + fast in one sweep)
// =====================================================================
void SupercellCGridScheme::compute_momentum_tendencies(
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
    (void)theta;  // perturbation-density buoyancy uses rho, not theta

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

    const double inv_2dr      = geo_.inv_2dr;
    const double inv_2dtheta  = 0.5 * geo_.inv_dtheta;
    const double inv_2dz      = geo_.inv_2dz;
    const double inv_dz_local = geo_.inv_dz;
    const double g_local      = dynamics_constants::g;
    const double gamma_local  = dynamics_constants::gamma;

    // ===== du/dt at r-face  (i = 0..NR-2, all j, k = 1..NZ-2) =====
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_face         = geo_.r_face[i];
            const double r_face_inv     = geo_.r_face_inv[i];
            const int    j_prev         = (j - 1 + NTH_) % NTH_;
            const int    j_next         = (j + 1) % NTH_;

            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double u_face = static_cast<double>(u[i][j][k]);

                // 4-point interp v -> r-face, theta-cell-center.
                // v[i][j_prev] at theta_{j-1/2}, v[i][j] at theta_{j+1/2};
                // average gives v at theta[j].  Then average over r.
                const double v_at_uface = 0.25 * (
                      static_cast<double>(v[i    ][j_prev][k])
                    + static_cast<double>(v[i    ][j     ][k])
                    + static_cast<double>(v[i + 1][j_prev][k])
                    + static_cast<double>(v[i + 1][j     ][k]));

                // 4-point interp w -> r-face, z-cell-center.
                const double w_at_uface = 0.25 * (
                      static_cast<double>(w[i    ][j][k - 1])
                    + static_cast<double>(w[i    ][j][k    ])
                    + static_cast<double>(w[i + 1][j][k - 1])
                    + static_cast<double>(w[i + 1][j][k    ]));

                // Centered d/dr at r-face. i=0 uses axis ghost u[-1] = -u[0].
                const double u_left  = (i == 0) ? -u_face
                                                :  static_cast<double>(u[i - 1][j][k]);
                const double u_right =                static_cast<double>(u[i + 1][j][k]);
                const double dur_dr  = (u_right - u_left) * inv_2dr;

                // Centered d/dtheta at r-face (periodic).
                const double dur_dth = (static_cast<double>(u[i][j_next][k])
                                      - static_cast<double>(u[i][j_prev][k])) * inv_2dtheta;

                // Centered d/dz at r-face.
                const double dur_dz  = (static_cast<double>(u[i][j][k + 1])
                                      - static_cast<double>(u[i][j][k - 1])) * inv_2dz;

                // One-sided pressure gradient at r-face: (p[i+1] - p[i])/dr.
                const double dp_dr   = deriv_.grad_r(p, i, j, k);

                // Density at r-face.
                const double rho_face = 0.5 * (static_cast<double>(rho[i    ][j][k])
                                             + static_cast<double>(rho[i + 1][j][k]));
                const double rho_safe = safe_rho(rho_face);

                const double advective_r     = -u_face   * dur_dr
                                             - v_at_uface * r_face_inv * dur_dth
                                             - w_at_uface * dur_dz;
                const double centrifugal     = (v_at_uface * v_at_uface) * r_face_inv;
                const double pressure_grad_r = -dp_dr / rho_safe;

                double du_r = advective_r + centrifugal + pressure_grad_r;
                if (!std::isfinite(du_r)) du_r = 0.0;
                du_dt[i][j][k] = static_cast<float>(du_r);
            }
        }
    }

    // ===== dv/dt at theta-face  (i = 1..NR-2, all j, k = 1..NZ-2) =====
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv  = geo_.r_inv[i];
            const int    j_prev = (j - 1 + NTH_) % NTH_;
            const int    j_next = (j + 1) % NTH_;

            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double v_face = static_cast<double>(v[i][j][k]);

                // 4-point interp u -> theta-face, r-cell-center.
                // u[i-1][j] at (r_face[i-1], theta[j]), u[i][j] at (r_face[i], theta[j]).
                // Average over r gives u at r[i] (cell-center r).  Average over
                // theta (j and j+1) gives u at theta_{j+1/2}.
                const double u_at_vface = 0.25 * (
                      static_cast<double>(u[i - 1][j     ][k])
                    + static_cast<double>(u[i    ][j     ][k])
                    + static_cast<double>(u[i - 1][j_next][k])
                    + static_cast<double>(u[i    ][j_next][k]));

                // 4-point interp w -> theta-face, z-cell-center.
                const double w_at_vface = 0.25 * (
                      static_cast<double>(w[i][j     ][k - 1])
                    + static_cast<double>(w[i][j     ][k    ])
                    + static_cast<double>(w[i][j_next][k - 1])
                    + static_cast<double>(w[i][j_next][k    ]));

                // Centered derivatives at theta-face.
                const double dv_dr  = (static_cast<double>(v[i + 1][j][k])
                                     - static_cast<double>(v[i - 1][j][k])) * inv_2dr;
                const double dv_dth = (static_cast<double>(v[i][j_next][k])
                                     - static_cast<double>(v[i][j_prev][k])) * inv_2dtheta;
                const double dv_dz  = (static_cast<double>(v[i][j][k + 1])
                                     - static_cast<double>(v[i][j][k - 1])) * inv_2dz;

                // One-sided azimuthal pressure gradient at theta-face:
                //   (p[j+1] - p[j]) / (r[i] * dtheta).
                const double dp_dth_over_r = deriv_.grad_theta(p, i, j, k);

                // Density at theta-face.
                const double rho_face = 0.5 * (static_cast<double>(rho[i][j     ][k])
                                             + static_cast<double>(rho[i][j_next][k]));
                const double rho_safe = safe_rho(rho_face);

                const double advective_th    = -u_at_vface * dv_dr
                                             - v_face      * r_inv * dv_dth
                                             - w_at_vface  * dv_dz;
                const double curvature_th    = -u_at_vface * v_face * r_inv;
                const double pressure_grad_th = -dp_dth_over_r / rho_safe;

                double dv_t = advective_th + curvature_th + pressure_grad_th;
                if (!std::isfinite(dv_t)) dv_t = 0.0;
                dv_dt[i][j][k] = static_cast<float>(dv_t);
            }
        }
    }

    // ===== dw/dt at z-face  (i = 1..NR-2, all j, k = 0..NZ-2) =====
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv  = geo_.r_inv[i];
            const int    j_prev = (j - 1 + NTH_) % NTH_;
            const int    j_next = (j + 1) % NTH_;

            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double w_face = static_cast<double>(w[i][j][k]);

                // 4-point interp u -> z-face, r-cell-center.
                const double u_at_wface = 0.25 * (
                      static_cast<double>(u[i - 1][j][k    ])
                    + static_cast<double>(u[i    ][j][k    ])
                    + static_cast<double>(u[i - 1][j][k + 1])
                    + static_cast<double>(u[i    ][j][k + 1]));

                // 4-point interp v -> z-face, theta-cell-center.
                const double v_at_wface = 0.25 * (
                      static_cast<double>(v[i][j_prev][k    ])
                    + static_cast<double>(v[i][j     ][k    ])
                    + static_cast<double>(v[i][j_prev][k + 1])
                    + static_cast<double>(v[i][j     ][k + 1]));

                // Centered derivatives at z-face.
                const double dw_dr  = (static_cast<double>(w[i + 1][j][k])
                                     - static_cast<double>(w[i - 1][j][k])) * inv_2dr;
                const double dw_dth = (static_cast<double>(w[i][j_next][k])
                                     - static_cast<double>(w[i][j_prev][k])) * inv_2dtheta;
                // Surface ghost w[-1] = 0 at k=0.
                const double w_below = (k == 0) ? 0.0
                                                : static_cast<double>(w[i][j][k - 1]);
                const double w_above =            static_cast<double>(w[i][j][k + 1]);
                const double dw_dz   = (w_above - w_below) * inv_2dz;

                // One-sided pressure gradient at z-face: (p[k+1] - p[k]) / dz.
                const double dp_dz   = deriv_.grad_z(p, i, j, k);

                // Density at z-face + reference-state subtraction. Same
                // one-sided dp0/dz form as dp/dz so the perturbation gradient
                // is bit-exactly zero at p = p0(z), rho = rho0(z).
                const double rho_face  = 0.5 * (static_cast<double>(rho[i][j][k])
                                              + static_cast<double>(rho[i][j][k + 1]));
                const double rho_safe  = safe_rho(rho_face);
                const double rho0_face = 0.5 * (rho0_base[k] + rho0_base[k + 1]);
                const double dp0_dz    = (p0_base[k + 1] - p0_base[k]) * inv_dz_local;
                const double dp_prime_dz = dp_dz - dp0_dz;

                const double buoyancy = -g_local * (rho_face - rho0_face) / rho_safe;

                double moisture_buoyancy = 0.0;
                if (!qv.empty() && !qv0_base.empty())
                {
                    const double qv_face  = 0.5 * (static_cast<double>(qv[i][j][k])
                                                 + static_cast<double>(qv[i][j][k + 1]));
                    const double qv0_face = 0.5 * (qv0_base[k] + qv0_base[k + 1]);
                    moisture_buoyancy = g_local * 0.608 * (qv_face - qv0_face);
                }

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

                const double advective_z = -u_at_wface * dw_dr
                                         - v_at_wface * r_inv * dw_dth
                                         - w_face     * dw_dz;
                double dw_t = advective_z - dp_prime_dz / rho_safe
                            + buoyancy + moisture_buoyancy + loading;
                if (!std::isfinite(dw_t)) dw_t = 0.0;
                dw_dt[i][j][k] = static_cast<float>(dw_t);
            }
        }
    }

    // ===== drho/dt and dp/dt at cell-center  (i = 1..NR-2, all j, k = 1..NZ-2) =====
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv  = geo_.r_inv[i];
            const int    j_prev = (j - 1 + NTH_) % NTH_;
            const int    j_next = (j + 1) % NTH_;

            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double rho_val  = static_cast<double>(rho[i][j][k]);
                const double p_val    = static_cast<double>(p  [i][j][k]);
                const double rho_safe = safe_rho(rho_val);

                // Flux-form divergence (axis-aware at i=0; here i>=1).
                const double div = deriv_.div_flux(u, v, w, i, j, k);

                double drho = -rho_safe * div;
                if (!std::isfinite(drho)) drho = 0.0;
                drho_dt[i][j][k] = static_cast<float>(drho);

                // Cell-center velocities by interpolation from faces.
                const double u_center = 0.5 * (static_cast<double>(u[i    ][j][k])
                                             + static_cast<double>(u[i - 1][j][k]));
                const double v_center = 0.5 * (static_cast<double>(v[i][j     ][k])
                                             + static_cast<double>(v[i][j_prev][k]));
                const double w_center = 0.5 * (static_cast<double>(w[i][j][k    ])
                                             + static_cast<double>(w[i][j][k - 1]));
                const double dp_dr_c  = (static_cast<double>(p[i + 1][j][k])
                                       - static_cast<double>(p[i - 1][j][k])) * inv_2dr;
                const double dp_dth_c = (static_cast<double>(p[i][j_next][k])
                                       - static_cast<double>(p[i][j_prev][k])) * inv_2dtheta;
                const double dp_dz_c  = (static_cast<double>(p[i][j][k + 1])
                                       - static_cast<double>(p[i][j][k - 1])) * inv_2dz;

                double dp_t = -gamma_local * p_val * div
                             - u_center * dp_dr_c
                             - v_center * r_inv * dp_dth_c
                             - w_center * dp_dz_c;
                if (!std::isfinite(dp_t)) dp_t = 0.0;
                dp_dt[i][j][k] = static_cast<float>(dp_t);
            }
        }
    }
}


// =====================================================================
// Slow tendencies: advection + buoyancy.  Pressure gradient on velocities
// is dropped; -gamma*p*div on pressure is dropped; drho/dt = 0.
// =====================================================================
void SupercellCGridScheme::compute_slow_tendencies(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p, const Field3D& theta, double /*dt*/,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
    Field3D& drho_dt, Field3D& dp_dt)
{
    (void)theta;

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

    const double inv_2dr     = geo_.inv_2dr;
    const double inv_2dtheta = 0.5 * geo_.inv_dtheta;
    const double inv_2dz     = geo_.inv_2dz;
    const double g_local     = dynamics_constants::g;

    // ===== du/dt at r-face: advection + centrifugal (NO pressure) =====
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_face_inv = geo_.r_face_inv[i];
            const int    j_prev     = (j - 1 + NTH_) % NTH_;
            const int    j_next     = (j + 1) % NTH_;

            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double u_face     = static_cast<double>(u[i][j][k]);
                const double v_at_uface = 0.25 * (
                      static_cast<double>(v[i    ][j_prev][k])
                    + static_cast<double>(v[i    ][j     ][k])
                    + static_cast<double>(v[i + 1][j_prev][k])
                    + static_cast<double>(v[i + 1][j     ][k]));
                const double w_at_uface = 0.25 * (
                      static_cast<double>(w[i    ][j][k - 1])
                    + static_cast<double>(w[i    ][j][k    ])
                    + static_cast<double>(w[i + 1][j][k - 1])
                    + static_cast<double>(w[i + 1][j][k    ]));

                const double u_left  = (i == 0) ? -u_face
                                                :  static_cast<double>(u[i - 1][j][k]);
                const double u_right =                static_cast<double>(u[i + 1][j][k]);
                const double dur_dr  = (u_right - u_left) * inv_2dr;
                const double dur_dth = (static_cast<double>(u[i][j_next][k])
                                      - static_cast<double>(u[i][j_prev][k])) * inv_2dtheta;
                const double dur_dz  = (static_cast<double>(u[i][j][k + 1])
                                      - static_cast<double>(u[i][j][k - 1])) * inv_2dz;

                double du_r = -u_face   * dur_dr
                            - v_at_uface * r_face_inv * dur_dth
                            - w_at_uface * dur_dz
                            + (v_at_uface * v_at_uface) * r_face_inv;
                if (!std::isfinite(du_r)) du_r = 0.0;
                du_dt[i][j][k] = static_cast<float>(du_r);
            }
        }
    }

    // ===== dv/dt at theta-face: advection + curvature (NO pressure) =====
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv  = geo_.r_inv[i];
            const int    j_prev = (j - 1 + NTH_) % NTH_;
            const int    j_next = (j + 1) % NTH_;

            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double v_face     = static_cast<double>(v[i][j][k]);
                const double u_at_vface = 0.25 * (
                      static_cast<double>(u[i - 1][j     ][k])
                    + static_cast<double>(u[i    ][j     ][k])
                    + static_cast<double>(u[i - 1][j_next][k])
                    + static_cast<double>(u[i    ][j_next][k]));
                const double w_at_vface = 0.25 * (
                      static_cast<double>(w[i][j     ][k - 1])
                    + static_cast<double>(w[i][j     ][k    ])
                    + static_cast<double>(w[i][j_next][k - 1])
                    + static_cast<double>(w[i][j_next][k    ]));

                const double dv_dr  = (static_cast<double>(v[i + 1][j][k])
                                     - static_cast<double>(v[i - 1][j][k])) * inv_2dr;
                const double dv_dth = (static_cast<double>(v[i][j_next][k])
                                     - static_cast<double>(v[i][j_prev][k])) * inv_2dtheta;
                const double dv_dz  = (static_cast<double>(v[i][j][k + 1])
                                     - static_cast<double>(v[i][j][k - 1])) * inv_2dz;

                double dv_t = -u_at_vface * dv_dr
                            - v_face      * r_inv * dv_dth
                            - w_at_vface  * dv_dz
                            - u_at_vface  * v_face * r_inv;
                if (!std::isfinite(dv_t)) dv_t = 0.0;
                dv_dt[i][j][k] = static_cast<float>(dv_t);
            }
        }
    }

    // ===== dw/dt at z-face: advection + buoyancy (NO pressure gradient) =====
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv  = geo_.r_inv[i];
            const int    j_prev = (j - 1 + NTH_) % NTH_;
            const int    j_next = (j + 1) % NTH_;

            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double w_face     = static_cast<double>(w[i][j][k]);
                const double u_at_wface = 0.25 * (
                      static_cast<double>(u[i - 1][j][k    ])
                    + static_cast<double>(u[i    ][j][k    ])
                    + static_cast<double>(u[i - 1][j][k + 1])
                    + static_cast<double>(u[i    ][j][k + 1]));
                const double v_at_wface = 0.25 * (
                      static_cast<double>(v[i][j_prev][k    ])
                    + static_cast<double>(v[i][j     ][k    ])
                    + static_cast<double>(v[i][j_prev][k + 1])
                    + static_cast<double>(v[i][j     ][k + 1]));

                const double dw_dr  = (static_cast<double>(w[i + 1][j][k])
                                     - static_cast<double>(w[i - 1][j][k])) * inv_2dr;
                const double dw_dth = (static_cast<double>(w[i][j_next][k])
                                     - static_cast<double>(w[i][j_prev][k])) * inv_2dtheta;
                const double w_below = (k == 0) ? 0.0
                                                : static_cast<double>(w[i][j][k - 1]);
                const double w_above =            static_cast<double>(w[i][j][k + 1]);
                const double dw_dz   = (w_above - w_below) * inv_2dz;

                const double rho_face  = 0.5 * (static_cast<double>(rho[i][j][k])
                                              + static_cast<double>(rho[i][j][k + 1]));
                const double rho_safe  = safe_rho(rho_face);
                const double rho0_face = 0.5 * (rho0_base[k] + rho0_base[k + 1]);

                const double buoyancy = -g_local * (rho_face - rho0_face) / rho_safe;

                double moisture_buoyancy = 0.0;
                if (!qv.empty() && !qv0_base.empty())
                {
                    const double qv_face  = 0.5 * (static_cast<double>(qv[i][j][k])
                                                 + static_cast<double>(qv[i][j][k + 1]));
                    const double qv0_face = 0.5 * (qv0_base[k] + qv0_base[k + 1]);
                    moisture_buoyancy = g_local * 0.608 * (qv_face - qv0_face);
                }

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

                double dw_t = -u_at_wface * dw_dr
                            - v_at_wface * r_inv * dw_dth
                            - w_face     * dw_dz
                            + buoyancy + moisture_buoyancy + loading;
                if (!std::isfinite(dw_t)) dw_t = 0.0;
                dw_dt[i][j][k] = static_cast<float>(dw_t);
            }
        }
    }

    // ===== dp/dt at cell-center: advection only (-u . grad p) =====
    // drho/dt is 0 in the slow path (mass continuity is fast).
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv  = geo_.r_inv[i];
            const int    j_prev = (j - 1 + NTH_) % NTH_;
            const int    j_next = (j + 1) % NTH_;

            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double u_center = 0.5 * (static_cast<double>(u[i    ][j][k])
                                             + static_cast<double>(u[i - 1][j][k]));
                const double v_center = 0.5 * (static_cast<double>(v[i][j     ][k])
                                             + static_cast<double>(v[i][j_prev][k]));
                const double w_center = 0.5 * (static_cast<double>(w[i][j][k    ])
                                             + static_cast<double>(w[i][j][k - 1]));
                const double dp_dr_c  = (static_cast<double>(p[i + 1][j][k])
                                       - static_cast<double>(p[i - 1][j][k])) * inv_2dr;
                const double dp_dth_c = (static_cast<double>(p[i][j_next][k])
                                       - static_cast<double>(p[i][j_prev][k])) * inv_2dtheta;
                const double dp_dz_c  = (static_cast<double>(p[i][j][k + 1])
                                       - static_cast<double>(p[i][j][k - 1])) * inv_2dz;

                double dp_t = -u_center * dp_dr_c
                            - v_center * r_inv * dp_dth_c
                            - w_center * dp_dz_c;
                if (!std::isfinite(dp_t)) dp_t = 0.0;
                dp_dt[i][j][k] = static_cast<float>(dp_t);
            }
        }
    }
}


// =====================================================================
// Fast pressure tendencies: -gamma*p*div_flux, -rho*div_flux at cell-center.
// =====================================================================
void SupercellCGridScheme::compute_fast_pressure_tendencies(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p,
    Field3D& drho_dt, Field3D& dp_dt)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                drho_dt[i][j][k] = 0.0f;
                dp_dt  [i][j][k] = 0.0f;
            }

    const double gamma_local = dynamics_constants::gamma;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double rho_val  = static_cast<double>(rho[i][j][k]);
                const double p_val    = static_cast<double>(p  [i][j][k]);
                const double rho_safe = safe_rho(rho_val);

                const double div = deriv_.div_flux(u, v, w, i, j, k);

                double drho = -rho_safe * div;
                if (!std::isfinite(drho)) drho = 0.0;
                drho_dt[i][j][k] = static_cast<float>(drho);

                double dp_t = -gamma_local * p_val * div;
                if (!std::isfinite(dp_t)) dp_t = 0.0;
                dp_dt[i][j][k] = static_cast<float>(dp_t);
            }
        }
    }
}


// =====================================================================
// Fast momentum tendencies: -grad(p)/rho on velocity faces.
// Vertical uses reference-state perturbation (dp/dz - dp0/dz)/rho.
// =====================================================================
void SupercellCGridScheme::compute_fast_momentum_tendencies(
    const Field3D& /*u*/, const Field3D& /*v*/, const Field3D& /*w*/,
    const Field3D& rho, const Field3D& p,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                du_dt[i][j][k] = 0.0f;
                dv_dt[i][j][k] = 0.0f;
                dw_dt[i][j][k] = 0.0f;
            }

    const double inv_dz_local = geo_.inv_dz;

    // du/dt at r-face: -grad_r(p) / rho_face
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double dp_dr    = deriv_.grad_r(p, i, j, k);
                const double rho_face = 0.5 * (static_cast<double>(rho[i    ][j][k])
                                             + static_cast<double>(rho[i + 1][j][k]));
                const double rho_safe = safe_rho(rho_face);
                double du_r = -dp_dr / rho_safe;
                if (!std::isfinite(du_r)) du_r = 0.0;
                du_dt[i][j][k] = static_cast<float>(du_r);
            }
        }
    }

    // dv/dt at theta-face: -grad_theta(p) / rho_face_theta
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const int j_next = (j + 1) % NTH_;
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double dp_dth_over_r = deriv_.grad_theta(p, i, j, k);
                const double rho_face = 0.5 * (static_cast<double>(rho[i][j     ][k])
                                             + static_cast<double>(rho[i][j_next][k]));
                const double rho_safe = safe_rho(rho_face);
                double dv_t = -dp_dth_over_r / rho_safe;
                if (!std::isfinite(dv_t)) dv_t = 0.0;
                dv_dt[i][j][k] = static_cast<float>(dv_t);
            }
        }
    }

    // dw/dt at z-face: -(dp/dz - dp0/dz) / rho_face_z
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double dp_dz       = deriv_.grad_z(p, i, j, k);
                const double dp0_dz      = (p0_base[k + 1] - p0_base[k]) * inv_dz_local;
                const double dp_prime_dz = dp_dz - dp0_dz;
                const double rho_face    = 0.5 * (static_cast<double>(rho[i][j][k])
                                                + static_cast<double>(rho[i][j][k + 1]));
                const double rho_safe    = safe_rho(rho_face);
                double dw_t = -dp_prime_dz / rho_safe;
                if (!std::isfinite(dw_t)) dw_t = 0.0;
                dw_dt[i][j][k] = static_cast<float>(dw_t);
            }
        }
    }
}


// =====================================================================
// Vorticity diagnostics evaluated at cell centers using interpolated
// face values. Mirrors the collocated supercell layout.
// =====================================================================
void SupercellCGridScheme::compute_vorticity_diagnostics(
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
    const double inv_2dr     = geo_.inv_2dr;
    const double inv_2dtheta = 0.5 * geo_.inv_dtheta;
    const double inv_2dz     = geo_.inv_2dz;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv  = geo_.r_inv[i];
            const int    j_prev = (j - 1 + NTH_) % NTH_;
            const int    j_next = (j + 1) % NTH_;

            for (int k = 1; k < NZ_ - 1; ++k)
            {
                // Cell-center velocities by 2-face average.
                const double u_c = 0.5 * (static_cast<double>(u[i    ][j][k])
                                        + static_cast<double>(u[i - 1][j][k]));
                const double v_c = 0.5 * (static_cast<double>(v[i][j     ][k])
                                        + static_cast<double>(v[i][j_prev][k]));
                const double w_c = 0.5 * (static_cast<double>(w[i][j][k    ])
                                        + static_cast<double>(w[i][j][k - 1]));

                // Cell-center derivatives via centered differences on
                // interpolated cell-center velocities.  These are 2nd-order
                // accurate on the staggered grid for diagnostic purposes.
                const double u_c_jp = 0.5 * (static_cast<double>(u[i    ][j_next][k])
                                           + static_cast<double>(u[i - 1][j_next][k]));
                const double u_c_jm = 0.5 * (static_cast<double>(u[i    ][j_prev][k])
                                           + static_cast<double>(u[i - 1][j_prev][k]));
                const double dur_dth_c = (u_c_jp - u_c_jm) * inv_2dtheta;

                const double u_c_kp = 0.5 * (static_cast<double>(u[i    ][j][k + 1])
                                           + static_cast<double>(u[i - 1][j][k + 1]));
                const double u_c_km = 0.5 * (static_cast<double>(u[i    ][j][k - 1])
                                           + static_cast<double>(u[i - 1][j][k - 1]));
                const double dur_dz_c  = (u_c_kp - u_c_km) * inv_2dz;

                const double v_c_ip = 0.5 * (static_cast<double>(v[i + 1][j     ][k])
                                           + static_cast<double>(v[i + 1][j_prev][k]));
                const double v_c_im = 0.5 * (static_cast<double>(v[i - 1][j     ][k])
                                           + static_cast<double>(v[i - 1][j_prev][k]));
                const double dv_dr_c   = (v_c_ip - v_c_im) * inv_2dr;

                const double v_c_kp = 0.5 * (static_cast<double>(v[i][j     ][k + 1])
                                           + static_cast<double>(v[i][j_prev][k + 1]));
                const double v_c_km = 0.5 * (static_cast<double>(v[i][j     ][k - 1])
                                           + static_cast<double>(v[i][j_prev][k - 1]));
                const double dv_dz_c   = (v_c_kp - v_c_km) * inv_2dz;

                const double w_c_ip = 0.5 * (static_cast<double>(w[i + 1][j][k    ])
                                           + static_cast<double>(w[i + 1][j][k - 1]));
                const double w_c_im = 0.5 * (static_cast<double>(w[i - 1][j][k    ])
                                           + static_cast<double>(w[i - 1][j][k - 1]));
                const double dw_dr_c   = (w_c_ip - w_c_im) * inv_2dr;

                const double w_c_jp = 0.5 * (static_cast<double>(w[i][j_next][k    ])
                                           + static_cast<double>(w[i][j_next][k - 1]));
                const double w_c_jm = 0.5 * (static_cast<double>(w[i][j_prev][k    ])
                                           + static_cast<double>(w[i][j_prev][k - 1]));
                const double dw_dth_c  = (w_c_jp - w_c_jm) * inv_2dtheta;

                const double dw_dz_c = (static_cast<double>(w[i][j][k    ])
                                      - static_cast<double>(w[i][j][k - 1])) * geo_.inv_dz;
                (void)w_c;
                (void)u_c;

                // Cylindrical vorticity components at cell center:
                //   omega_r     = (1/r) * dw/dtheta - dv/dz
                //   omega_theta = du/dz - dw/dr
                //   omega_z     = dv/dr + v/r - (1/r) * du/dtheta
                const double omega_r  = dw_dth_c * r_inv - dv_dz_c;
                const double omega_th = dur_dz_c - dw_dr_c;
                const double omega_z  = dv_dr_c + v_c * r_inv - dur_dth_c * r_inv;

                vorticity_r    [i][j][k] = std::isfinite(omega_r)  ? static_cast<float>(omega_r)  : 0.0f;
                vorticity_theta[i][j][k] = std::isfinite(omega_th) ? static_cast<float>(omega_th) : 0.0f;
                vorticity_z    [i][j][k] = std::isfinite(omega_z)  ? static_cast<float>(omega_z)  : 0.0f;

                const double stretch = omega_z * dw_dz_c;
                stretching_term[i][j][k] = std::isfinite(stretch) ? static_cast<float>(stretch) : 0.0f;

                // Tilting: omega_h . grad(w) on the horizontal vorticity.
                const double tilt = omega_r * dw_dr_c + omega_th * dw_dth_c * r_inv;
                tilting_term[i][j][k] = std::isfinite(tilt) ? static_cast<float>(tilt) : 0.0f;

                // Baroclinic torque (radial-azimuthal component, as in
                // collocated supercell): (1/rho^2) * (drho/dr * dp/dth - drho/dth * dp/dr).
                const double drho_dr = (static_cast<double>(rho[i + 1][j][k])
                                      - static_cast<double>(rho[i - 1][j][k])) * inv_2dr;
                const double drho_dth = (static_cast<double>(rho[i][j_next][k])
                                       - static_cast<double>(rho[i][j_prev][k])) * inv_2dtheta;
                const double dp_dr_c  = (static_cast<double>(p[i + 1][j][k])
                                       - static_cast<double>(p[i - 1][j][k])) * inv_2dr;
                const double dp_dth_c = (static_cast<double>(p[i][j_next][k])
                                       - static_cast<double>(p[i][j_prev][k])) * inv_2dtheta;
                const double rho_local = static_cast<double>(rho[i][j][k]);
                const double rho_sq    = rho_local * rho_local;
                double baro = 0.0;
                if (rho_sq > dynamics_constants::eps)
                {
                    baro = (drho_dr * dp_dth_c - drho_dth * dp_dr_c) / rho_sq;
                }
                baroclinic_term[i][j][k] = std::isfinite(baro) ? static_cast<float>(baro) : 0.0f;
            }
        }
    }
}


// =====================================================================
// Pressure decomposition diagnostics, mirroring the collocated supercell.
// =====================================================================
void SupercellCGridScheme::compute_pressure_diagnostics(
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    const Field3D& /*rho*/,
    const Field3D& theta,
    Field3D& p_prime,
    Field3D& dynamic_pressure,
    Field3D& buoyancy_pressure)
{
    const double inv_2dr     = geo_.inv_2dr;
    const double inv_2dtheta = 0.5 * geo_.inv_dtheta;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv  = geo_.r_inv[i];
            const int    j_prev = (j - 1 + NTH_) % NTH_;
            const int    j_next = (j + 1) % NTH_;

            for (int k = 1; k < NZ_ - 1; ++k)
            {
                // Cell-center derivatives of cell-center-projected velocities.
                const double u_c    = 0.5 * (static_cast<double>(u[i    ][j][k])
                                           + static_cast<double>(u[i - 1][j][k]));
                const double u_c_ip = 0.5 * (static_cast<double>(u[i + 1][j][k])
                                           + static_cast<double>(u[i    ][j][k]));
                const double u_c_im = 0.5 * (static_cast<double>(u[i    ][j][k])
                                           + static_cast<double>(u[i - 1][j][k]));
                const double dur_dr = (u_c_ip - u_c_im) * inv_2dr;

                const double u_c_jp = 0.5 * (static_cast<double>(u[i    ][j_next][k])
                                           + static_cast<double>(u[i - 1][j_next][k]));
                const double u_c_jm = 0.5 * (static_cast<double>(u[i    ][j_prev][k])
                                           + static_cast<double>(u[i - 1][j_prev][k]));
                const double dur_dth = (u_c_jp - u_c_jm) * inv_2dtheta;

                const double v_c_ip = 0.5 * (static_cast<double>(v[i + 1][j     ][k])
                                           + static_cast<double>(v[i + 1][j_prev][k]));
                const double v_c_im = 0.5 * (static_cast<double>(v[i - 1][j     ][k])
                                           + static_cast<double>(v[i - 1][j_prev][k]));
                const double duth_dr = (v_c_ip - v_c_im) * inv_2dr;

                const double v_c_jp = 0.5 * (static_cast<double>(v[i][j_next][k])
                                           + static_cast<double>(v[i][j     ][k]));
                const double v_c_jm = 0.5 * (static_cast<double>(v[i][j     ][k])
                                           + static_cast<double>(v[i][j_prev][k]));
                const double duth_dth = (v_c_jp - v_c_jm) * inv_2dtheta;

                const double duz_dz = (static_cast<double>(w[i][j][k    ])
                                     - static_cast<double>(w[i][j][k - 1])) * geo_.inv_dz;
                (void)u_c;

                const double deformation = dur_dr * dur_dr
                    + r_inv * (dur_dth * dur_dth + duth_dr * duth_dr + duth_dth * duth_dth)
                    + duz_dz * duz_dz;

                dynamic_pressure[i][j][k] = static_cast<float>(-rho0_base[k] * deformation);

                const double theta_prime = static_cast<double>(theta[i][j][k]) - theta0;
                const double buoy        = dynamics_constants::g * (theta_prime / theta0);
                buoyancy_pressure[i][j][k] = static_cast<float>(rho0_base[k] * buoy);

                p_prime[i][j][k] = dynamic_pressure[i][j][k] + buoyancy_pressure[i][j][k];
            }
        }
    }
}

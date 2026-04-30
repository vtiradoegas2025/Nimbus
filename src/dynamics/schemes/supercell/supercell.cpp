/**
 * @file supercell.cpp
 * @brief Implementation for the dynamics module.
 *
 * Provides executable logic for the dynamics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/dynamics subsystem.
 */

#include "supercell.hpp"
#include "core/simulation.hpp"
#include "compute/compute_kernel_template.hpp"
#include "util/grid_metric_utils.hpp"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Initializes the supercell scheme.
 */
SupercellScheme::SupercellScheme()
    : NR_(NR), NTH_(NTH), NZ_(NZ),
      dr_(dr), dtheta_(dtheta), dz_(dz),
      geo_(global_grid_geometry),
      deriv_(std::make_unique<CylindricalDerivatives>(global_grid_metrics, dtheta, NTH, NZ))
{
}

/**
 * @brief Computes the momentum tendencies for the supercell scheme.
 */

void SupercellScheme::compute_momentum_tendencies(
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    const Field3D& rho,
    const Field3D& p,
    const Field3D& theta,
    double dt,
    Field3D& du_dt,
    Field3D& dv_dt,
    Field3D& dw_dt,
    Field3D& drho_dt,
    Field3D& dp_dt)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            for (int k = 0; k < NZ_; ++k)
            {
                du_dt[i][j][k] = 0.0f;
                dv_dt[i][j][k] = 0.0f;
                dw_dt[i][j][k] = 0.0f;
                drho_dt[i][j][k] = 0.0f;
                dp_dt[i][j][k] = 0.0f;
            }
        }
    }

    // Try GPU dispatch for interior points (only when terrain metrics are NOT active)
    if (!grid_metric::has_terrain_metrics(global_grid_metrics))
    {
        // Precompute total hydrometeor loading for GPU dispatch.
        const size_t n_total = static_cast<size_t>(NR_) * NTH_ * NZ_;
        std::vector<float> loading_buf(n_total, 0.0f);
        if (!qc.empty())
        {
            for (size_t idx = 0; idx < n_total; ++idx)
            {
                loading_buf[idx] = qc.data()[idx] + qr.data()[idx] +
                                   qi.data()[idx] + qs.data()[idx] +
                                   qg.data()[idx] + qh.data()[idx];
            }
        }

        if (dispatch_supercell_tendencies_backend(
                u.data(), v.data(), w.data(),
                rho.data(), p.data(), theta.data(),
                loading_buf.data(),
                du_dt.data(), dv_dt.data(), dw_dt.data(),
                drho_dt.data(), dp_dt.data(),
                NR_, NTH_, NZ_,
                static_cast<float>(dr_), static_cast<float>(dtheta_), static_cast<float>(dz_),
                static_cast<float>(dynamics_constants::g),
                static_cast<float>(dynamics_constants::gamma),
                static_cast<float>(dynamics_constants::theta0)))
        {
            return;
        }
    }

    // CPU fallback
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv = geo_.r_inv[i];
            int j_prev = (j - 1 + NTH_) % NTH_;
            int j_next = (j + 1) % NTH_;

            for (int k = 1; k < NZ_ - 1; ++k)
            {
                double ur = u[i][j][k];
                double uth = v[i][j][k];
                double uz = w[i][j][k];
                double rho_val = rho[i][j][k];
                double p_val = p[i][j][k];
                double theta_val = theta[i][j][k];
                const double rho_safe = (std::isfinite(rho_val) && rho_val > 1.0e-6) ? rho_val : 1.0;

                double dur_dr = deriv_->di(u, i, j, k);
                double dur_dth = deriv_->dj(u, i, j, k);
                double dur_dz = deriv_->dk(u, i, j, k);

                double duth_dr = deriv_->di(v, i, j, k);
                double duth_dth = deriv_->dj(v, i, j, k);
                double duth_dz = deriv_->dk(v, i, j, k);

                double duz_dr = deriv_->di(w, i, j, k);
                double duz_dth = deriv_->dj(w, i, j, k);
                double duz_dz = deriv_->dk(w, i, j, k);

                double dp_dr = deriv_->di(p, i, j, k);
                double dp_dth = deriv_->dj(p, i, j, k);
                double dp_dz = deriv_->dk(p, i, j, k);

                double advective_r = -ur * dur_dr - uth * r_inv * dur_dth - uz * dur_dz;
                double centrifugal = uth * uth * r_inv;
                double pressure_grad_r = -dp_dr / rho_safe;

                double du_r = advective_r + centrifugal + pressure_grad_r;
                if (!std::isfinite(du_r))
                {
                    du_r = 0.0;
                }
                du_dt[i][j][k] = static_cast<float>(du_r);

                double advective_th = -ur * duth_dr - uth * r_inv * duth_dth - uz * duth_dz;
                double coriolis_th = -ur * uth * r_inv;
                double pressure_grad_th = -dp_dth * r_inv / rho_safe;

                double du_theta = advective_th + coriolis_th + pressure_grad_th;
                if (!std::isfinite(du_theta))
                {
                    du_theta = 0.0;
                }
                dv_dt[i][j][k] = static_cast<float>(du_theta);

                double advective_z = -ur * duz_dr - uth * r_inv * duz_dth - uz * duz_dz;

                // Vertical momentum with reference-state subtraction:
                //   dw/dt = -(1/ρ)(∂p/∂z - ∂p₀/∂z) - g(ρ - ρ₀)/ρ + advection
                //
                // This eliminates the discrete hydrostatic imbalance that the
                // naive form dw/dt = -(1/ρ)∂p/∂z - g produces on a collocated
                // grid. At initialization p ≡ p₀ and ρ ≡ ρ₀, so both
                // perturbation terms are exactly zero.
                //
                // Do NOT add an explicit g·(θ-θ₀)/θ₀ term — see
                // docs/Journey.md Phase 2 "Bug 3: Double-counted buoyancy".
                (void)theta_val;
                const double dp0_dz = (p0_base[k + 1] - p0_base[k - 1]) / (2.0 * dz_);
                const double dp_prime_dz = dp_dz - dp0_dz;
                const double rho0_k = rho0_base[k];
                const double buoyancy = -dynamics_constants::g * (rho_val - rho0_k) / rho_safe;
                double moisture_buoyancy = 0.0;
                if (!qv.empty() && !qv0_base.empty())
                {
                    const double qv_val = static_cast<double>(qv[i][j][k]);
                    const double qv0_k = qv0_base[k];
                    moisture_buoyancy = dynamics_constants::g * 0.608 * (qv_val - qv0_k);
                }
                double loading = 0.0;
                if (!qc.empty())
                {
                    loading = -dynamics_constants::g *
                        (static_cast<double>(qc[i][j][k]) + static_cast<double>(qr[i][j][k]) +
                         static_cast<double>(qi[i][j][k]) + static_cast<double>(qs[i][j][k]) +
                         static_cast<double>(qg[i][j][k]) + static_cast<double>(qh[i][j][k]));
                }
                double du_z = advective_z - dp_prime_dz / rho_safe + buoyancy + moisture_buoyancy + loading;
                if (!std::isfinite(du_z))
                {
                    du_z = 0.0;
                }
                dw_dt[i][j][k] = static_cast<float>(du_z);

                double divergence = dur_dr + duz_dz + (ur + duth_dth) * r_inv;
                double drho = -rho_safe * divergence;
                if (!std::isfinite(drho))
                {
                    drho = 0.0;
                }
                drho_dt[i][j][k] = static_cast<float>(drho);

                double gamma_term = dynamics_constants::gamma * p_val * divergence;
                double advection_p = -ur * dp_dr - uth * r_inv * dp_dth - uz * dp_dz;
                double dp_t = -gamma_term + advection_p;
                if (!std::isfinite(dp_t))
                {
                    dp_t = 0.0;
                }
                dp_dt[i][j][k] = static_cast<float>(dp_t);
            }
        }
    }
}

// =========================================================================
// Split-explicit acoustic substep methods (cylindrical coordinates)
// =========================================================================

void SupercellScheme::compute_slow_tendencies(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p, const Field3D& theta,
    double /*dt*/,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
    Field3D& drho_dt, Field3D& dp_dt)
{
    (void)theta;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                du_dt[i][j][k] = 0.0f; dv_dt[i][j][k] = 0.0f;
                dw_dt[i][j][k] = 0.0f; drho_dt[i][j][k] = 0.0f;
                dp_dt[i][j][k] = 0.0f;
            }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv = geo_.r_inv[i];
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double ur = u[i][j][k];
                const double uth = v[i][j][k];
                const double uz = w[i][j][k];
                const double rho_val = rho[i][j][k];
                const double rho_safe = (std::isfinite(rho_val) && rho_val > 1.0e-6) ? rho_val : 1.0;

                // Radial momentum: advection + centrifugal (no pressure gradient)
                const double dur_dr = deriv_->di(u, i, j, k);
                const double dur_dth = deriv_->dj(u, i, j, k);
                const double dur_dz = deriv_->dk(u, i, j, k);
                double du_r_val = -ur * dur_dr - uth * r_inv * dur_dth - uz * dur_dz + uth * uth * r_inv;
                if (!std::isfinite(du_r_val)) du_r_val = 0.0;
                du_dt[i][j][k] = static_cast<float>(du_r_val);

                // Azimuthal momentum: advection + coriolis-like (no pressure gradient)
                const double duth_dr = deriv_->di(v, i, j, k);
                const double duth_dth = deriv_->dj(v, i, j, k);
                const double duth_dz = deriv_->dk(v, i, j, k);
                double du_th_val = -ur * duth_dr - uth * r_inv * duth_dth - uz * duth_dz - ur * uth * r_inv;
                if (!std::isfinite(du_th_val)) du_th_val = 0.0;
                dv_dt[i][j][k] = static_cast<float>(du_th_val);

                // Vertical momentum: advection + buoyancy + loading (no pressure gradient)
                const double duz_dr = deriv_->di(w, i, j, k);
                const double duz_dth = deriv_->dj(w, i, j, k);
                const double duz_dz = deriv_->dk(w, i, j, k);
                const double advective_z = -ur * duz_dr - uth * r_inv * duz_dth - uz * duz_dz;
                const double rho0_k = rho0_base[k];
                const double buoyancy = -dynamics_constants::g * (rho_val - rho0_k) / rho_safe;
                double moisture_buoyancy = 0.0;
                if (!qv.empty() && !qv0_base.empty())
                {
                    const double qv_val = static_cast<double>(qv[i][j][k]);
                    const double qv0_k = qv0_base[k];
                    moisture_buoyancy = dynamics_constants::g * 0.608 * (qv_val - qv0_k);
                }
                double loading = 0.0;
                if (!qc.empty())
                {
                    loading = -dynamics_constants::g *
                        (static_cast<double>(qc[i][j][k]) + static_cast<double>(qr[i][j][k]) +
                         static_cast<double>(qi[i][j][k]) + static_cast<double>(qs[i][j][k]) +
                         static_cast<double>(qg[i][j][k]) + static_cast<double>(qh[i][j][k]));
                }
                double du_z_val = advective_z + buoyancy + moisture_buoyancy + loading;
                if (!std::isfinite(du_z_val)) du_z_val = 0.0;
                dw_dt[i][j][k] = static_cast<float>(du_z_val);

                // Density: no slow tendency
                drho_dt[i][j][k] = 0.0f;

                // Pressure: advection only
                const double dp_dr = deriv_->di(p, i, j, k);
                const double dp_dth = deriv_->dj(p, i, j, k);
                const double dp_dz_val = deriv_->dk(p, i, j, k);
                double dp_val = -ur * dp_dr - uth * r_inv * dp_dth - uz * dp_dz_val;
                if (!std::isfinite(dp_val)) dp_val = 0.0;
                dp_dt[i][j][k] = static_cast<float>(dp_val);
            }
        }
    }
}

void SupercellScheme::compute_fast_pressure_tendencies(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p,
    Field3D& drho_dt, Field3D& dp_dt)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            { drho_dt[i][j][k] = 0.0f; dp_dt[i][j][k] = 0.0f; }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv = geo_.r_inv[i];
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double rho_val = rho[i][j][k];
                const double p_val = p[i][j][k];
                const double rho_safe = (std::isfinite(rho_val) && rho_val > 1.0e-6) ? rho_val : 1.0;
                const double ur = u[i][j][k];

                // Cylindrical divergence: du/dr + u/r + (1/r)du_th/dth + dw/dz
                const double dur_dr = deriv_->di(u, i, j, k);
                const double duth_dth = deriv_->dj(v, i, j, k);
                const double duz_dz = deriv_->dk(w, i, j, k);
                const double divergence = dur_dr + (ur + duth_dth) * r_inv + duz_dz;

                double drho_val = -rho_safe * divergence;
                if (!std::isfinite(drho_val)) drho_val = 0.0;
                drho_dt[i][j][k] = static_cast<float>(drho_val);

                double dp_val = -dynamics_constants::gamma * p_val * divergence;
                if (!std::isfinite(dp_val)) dp_val = 0.0;
                dp_dt[i][j][k] = static_cast<float>(dp_val);
            }
        }
    }
}

void SupercellScheme::compute_fast_momentum_tendencies(
    const Field3D& /*u*/, const Field3D& /*v*/, const Field3D& /*w*/,
    const Field3D& rho, const Field3D& p,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            { du_dt[i][j][k] = 0.0f; dv_dt[i][j][k] = 0.0f; dw_dt[i][j][k] = 0.0f; }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv = geo_.r_inv[i];
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double rho_val = rho[i][j][k];
                const double rho_safe = (std::isfinite(rho_val) && rho_val > 1.0e-6) ? rho_val : 1.0;

                const double dp_dr = deriv_->di(p, i, j, k);
                const double dp_dth = deriv_->dj(p, i, j, k);
                const double dp_dz_val = deriv_->dk(p, i, j, k);

                // Radial pressure gradient
                double du_r_val = -dp_dr / rho_safe;
                if (!std::isfinite(du_r_val)) du_r_val = 0.0;
                du_dt[i][j][k] = static_cast<float>(du_r_val);

                // Azimuthal pressure gradient
                double du_th_val = -dp_dth * r_inv / rho_safe;
                if (!std::isfinite(du_th_val)) du_th_val = 0.0;
                dv_dt[i][j][k] = static_cast<float>(du_th_val);

                // Vertical: perturbation pressure gradient (reference-state subtraction)
                const double dp0_dz = (p0_base[k + 1] - p0_base[k - 1]) / (2.0 * dz_);
                const double dp_prime_dz = dp_dz_val - dp0_dz;
                double du_z_val = -dp_prime_dz / rho_safe;
                if (!std::isfinite(du_z_val)) du_z_val = 0.0;
                dw_dt[i][j][k] = static_cast<float>(du_z_val);
            }
        }
    }
}

/**
 * @brief Computes the vorticity diagnostics for the supercell scheme.
 */
void SupercellScheme::compute_vorticity_diagnostics(
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
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r_inv = geo_.r_inv[i];
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                double duth_dz = deriv_->dk(v, i, j, k);
                double duz_dth = deriv_->dj(w, i, j, k);
                vorticity_r[i][j][k] = static_cast<float>(duz_dth * r_inv - duth_dz);
                if (!std::isfinite(vorticity_r[i][j][k]))
                {
                    vorticity_r[i][j][k] = 0.0f;
                }

                const double duz_dr = deriv_->di(w, i, j, k);
                const double dur_dz = deriv_->dk(u, i, j, k);
                vorticity_theta[i][j][k] = static_cast<float>(dur_dz - duz_dr);
                if (!std::isfinite(vorticity_theta[i][j][k]))
                {
                    vorticity_theta[i][j][k] = 0.0f;
                }

                const double duth_dr = deriv_->di(v, i, j, k);
                const double dur_dth = deriv_->dj(u, i, j, k);
                const double v_local = static_cast<double>(v[i][j][k]);
                const double zeta = duth_dr + (v_local - dur_dth) * r_inv;
                vorticity_z[i][j][k] = static_cast<float>(std::isfinite(zeta) ? zeta : 0.0);
                
                double dw_dz = deriv_->dk(w, i, j, k);
                stretching_term[i][j][k] = static_cast<float>(zeta * dw_dz);
                if (!std::isfinite(stretching_term[i][j][k]))
                {
                    stretching_term[i][j][k] = 0.0f;
                }

                double dw_dr = deriv_->di(w, i, j, k);
                double dv_dz = deriv_->dk(v, i, j, k);
                double dw_dth = deriv_->dj(w, i, j, k);
                double du_dz = deriv_->dk(u, i, j, k);
                tilting_term[i][j][k] = dw_dr * dv_dz - dw_dth * r_inv * du_dz;
                if (!std::isfinite(tilting_term[i][j][k]))
                {
                    tilting_term[i][j][k] = 0.0f;
                }

                double drho_dr = deriv_->di(rho, i, j, k);
                double drho_dth = deriv_->dj(rho, i, j, k);
                double drho_dz = deriv_->dk(rho, i, j, k);

                double dp_dr = deriv_->di(p, i, j, k);
                double dp_dth = deriv_->dj(p, i, j, k);
                double dp_dz = deriv_->dk(p, i, j, k);

                double rho_sq = rho[i][j][k] * rho[i][j][k];

                if (rho_sq > dynamics_constants::eps) 
                {
                    baroclinic_term[i][j][k] = (1.0 / rho_sq) *
                        (drho_dr * dp_dth - drho_dth * dp_dr);
                } 
                else 
                {
                    baroclinic_term[i][j][k] = 0.0;
                }
                if (!std::isfinite(baroclinic_term[i][j][k]))
                {
                    baroclinic_term[i][j][k] = 0.0f;
                }
            }
        }
    }
}

/**
 * @brief Computes the pressure diagnostics for the supercell scheme.
 */
void SupercellScheme::compute_pressure_diagnostics(
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    const Field3D& rho,
    const Field3D& theta,
    Field3D& p_prime,
    Field3D& dynamic_pressure,
    Field3D& buoyancy_pressure)
{

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                double dur_dr = deriv_->di(u, i, j, k);
                double dur_dth = deriv_->dj(u, i, j, k);
                double duth_dr = deriv_->di(v, i, j, k);
                double duth_dth = deriv_->dj(v, i, j, k);
                double duz_dz = deriv_->dk(w, i, j, k);

                const double r_inv = geo_.r_inv[i];
                double deformation = dur_dr * dur_dr + r_inv *
                    (dur_dth * dur_dth + duth_dr * duth_dr + duth_dth * duth_dth) + duz_dz * duz_dz;

                dynamic_pressure[i][j][k] = -rho0_base[k] * deformation;

                double theta_prime = theta[i][j][k] - theta0;
                double buoyancy = dynamics_constants::g * (theta_prime / theta0);
                buoyancy_pressure[i][j][k] = rho0_base[k] * buoyancy;

                p_prime[i][j][k] = dynamic_pressure[i][j][k] + buoyancy_pressure[i][j][k];
            }
        }
    }
}


// Derivative operators moved to DerivativeOperators (Phase B.2).
// CylindricalDerivatives is constructed in SupercellScheme::SupercellScheme().

/**
 * @brief Computes the vorticity in the radial direction.
 */
double SupercellScheme::compute_vorticity_r(double dtheta_u_z, double dz_u_theta) const 
{
    return dtheta_u_z - dz_u_theta;
}

/**
 * @brief Computes the vorticity in the azimuthal direction.
 */
double SupercellScheme::compute_vorticity_theta(double dz_u_r, double dr_u_z) const 
{
    return dz_u_r - dr_u_z;
}

/**
 * @brief Computes the vorticity in the vertical direction.
 */
double SupercellScheme::compute_vorticity_z(double dr_u_theta, double dtheta_u_r, double r) const
{
    const double r_inv_local = (r > 0.0) ? 1.0 / r : 0.0;
    return r_inv_local * (dr_u_theta - dtheta_u_r) + dr_u_theta * r_inv_local;
}

/**
 * @brief Computes the buoyancy.
 */
double SupercellScheme::compute_buoyancy(double theta_prime, double rho, double rho0) const 
{
    return dynamics_constants::g * (theta_prime / theta0) * (rho0 / rho);
}

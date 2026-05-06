/**
 * @file tornado.cpp
 * @brief Implementation for the dynamics module.
 *
 * Provides executable logic for the dynamics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/dynamics subsystem.
 */

#include "tornado.hpp"
#include "core/runtime/simulation.hpp"
#include "compute/compute_kernel_template.hpp"
#include "util/grid_metric_utils.hpp"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif


TornadoScheme::TornadoScheme()
    : NR_(NR), NTH_(NTH), NZ_(NZ), dr_(dr), dtheta_(dtheta), dz_(dz),
      geo_(global_grid_geometry),
      deriv_(std::make_unique<CylindricalDerivatives>(global_grid_metrics, dtheta, NTH, NZ))
{
}

/**
 * @brief Computes the momentum tendencies for the tornado scheme.
 */
void TornadoScheme::compute_momentum_tendencies(const Field3D& u,
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

        if (dispatch_tornado_tendencies_backend(
                u.data(), v.data(), w.data(),
                rho.data(), p.data(), theta.data(),
                loading_buf.data(),
                du_dt.data(), dv_dt.data(), dw_dt.data(),
                drho_dt.data(), dp_dt.data(),
                NR_, NTH_, NZ_,
                static_cast<float>(dr_), static_cast<float>(dz_),
                static_cast<float>(dynamics_constants::g),
                static_cast<float>(dynamics_constants::theta0),
                static_cast<float>(dynamics_constants::eps),
                0.01f))  // Vortex damping friction coefficient
        {
            return;
        }
    }

    // CPU fallback — axisymmetric: compute at j=0, replicate to all j
    const int j = 0;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int k = 1; k < NZ_ - 1; ++k)
        {
            const double r_inv = geo_.r_inv[i];
            double ur = u[i][j][k];
            double uth = v[i][j][k];
            double uz = w[i][j][k];
            double rho_val = rho[i][j][k];
            double p_val = p[i][j][k];
            if (!std::isfinite(rho_val) || rho_val <= 1.0e-6)
            {
                rho_val = 1.0;
            }

            double dur_dr = deriv_->di(u, i, j, k);
            double dur_dz = deriv_->dk(u, i, j, k);

            double duth_dr = deriv_->di(v, i, j, k);
            double duth_dz = deriv_->dk(v, i, j, k);

            double duz_dr = deriv_->di(w, i, j, k);
            double duz_dz = deriv_->dk(w, i, j, k);

            double dp_dr = deriv_->di(p, i, j, k);
            double dp_dz = deriv_->dk(p, i, j, k);

            double advective_r = -ur * dur_dr - uz * dur_dz;
            double centrifugal = uth * uth * r_inv;
            double pressure_grad_r = -dp_dr / rho_val;

            double du_r = advective_r + centrifugal + pressure_grad_r;
            if (!std::isfinite(du_r))
            {
                du_r = 0.0;
            }
            du_dt[i][j][k] = static_cast<float>(du_r);

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                du_dt[i][jj][k] = du_dt[i][j][k];
            }

            double advective_th = -ur * duth_dr - uz * duth_dz;
            double coriolis_th = -ur * uth * r_inv;

            double fv = 0.0;

            if (i > 0 && i < NR_-1) 
            {
                double v_here = uth;
                double v_inner = (i > 0) ? v[i-1][j][k] : 0.0;
                double v_outer = (i < NR_-1) ? v[i+1][j][k] : 0.0;

                if (v_here > v_inner && v_here > v_outer) 
                {
                    fv = -0.01 * v_here;
                }
            }

            double du_th = advective_th + coriolis_th + fv;
            if (!std::isfinite(du_th))
            {
                du_th = 0.0;
            }
            dv_dt[i][j][k] = static_cast<float>(du_th);

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                dv_dt[i][jj][k] = dv_dt[i][j][k];
            }

            double advective_z = -ur * duz_dr - uz * duz_dz;
            double pressure_grad_z = -dp_dz / rho_val;

            // Fully compressible vertical momentum equation:
            //   dw/dt = -(1/ρ) ∂p/∂z - g - g*q_total + advection
            // Buoyancy is implicit in (-∂p/∂z/ρ - g). See the matching comment
            // in src/dynamics/schemes/supercell/supercell.cpp for the full
            // explanation and docs/Journey.md Phase 2 "Bug 3: Double-counted
            // buoyancy". Do NOT reintroduce an explicit g·(θ-θ₀)/θ₀ term here.
            //
            // Virtual temperature buoyancy: moist air is lighter (Rv/Rd - 1 = 0.608).
            double moisture_buoyancy = 0.0;
            if (!qv.empty() && !qv0_base.empty())
            {
                const double qv_val = static_cast<double>(qv[i][j][k]);
                const double qv0_k = qv0_base[k];
                moisture_buoyancy = dynamics_constants::g * 0.608 * (qv_val - qv0_k);
            }

            // Precipitation loading (Klemp & Wilhelmson 1978): hydrometeor
            // mass acts as ballast in the vertical momentum equation.
            double loading = 0.0;
            if (!qc.empty())
            {
                loading = dynamics_constants::g *
                    (static_cast<double>(qc[i][j][k]) + static_cast<double>(qr[i][j][k]) +
                     static_cast<double>(qi[i][j][k]) + static_cast<double>(qs[i][j][k]) +
                     static_cast<double>(qg[i][j][k]) + static_cast<double>(qh[i][j][k]));
            }
            double du_z = advective_z + pressure_grad_z - dynamics_constants::g + moisture_buoyancy - loading;
            if (!std::isfinite(du_z))
            {
                du_z = 0.0;
            }
            dw_dt[i][j][k] = static_cast<float>(du_z);

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                dw_dt[i][jj][k] = dw_dt[i][j][k];
            }

            double drho_dt_val = -rho_val * (dur_dr + ur * r_inv + duz_dz);
            if (!std::isfinite(drho_dt_val))
            {
                drho_dt_val = 0.0;
            }
            drho_dt[i][j][k] = drho_dt_val;

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                drho_dt[i][jj][k] = drho_dt_val;
            }

            double cyclostrophic_dp_dr = rho_val * uth * uth * r_inv;
            double dp_t = -ur * dp_dr - uz * dp_dz + cyclostrophic_dp_dr;
            if (!std::isfinite(dp_t))
            {
                dp_t = 0.0;
            }
            dp_dt[i][j][k] = static_cast<float>(dp_t);

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                dp_dt[i][jj][k] = dp_dt[i][j][k];
            }
        }
    }
}

/**
 * @brief Computes the angular momentum for the tornado scheme.
 */
void TornadoScheme::compute_angular_momentum(
    const Field3D& u,
    const Field3D& v,
    Field3D& angular_momentum,
    Field3D& angular_momentum_tendency)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r = geo_.r[i];
            for (int k = 0; k < NZ_; ++k)
            {
                double v_val = v[i][j][k];
                angular_momentum[i][j][k] = r * v_val;

                double ur = u[i][j][k];
                double uz = w[i][j][k];

                double dm_dr = 0.0, dm_dz = 0.0;

                if (i > 0 && i < NR_-1) dm_dr = deriv_->di(angular_momentum, i, j, k);
                if (k > 0 && k < NZ_-1) dm_dz = deriv_->dk(angular_momentum, i, j, k);

                angular_momentum_tendency[i][j][k] = -ur * dm_dr - uz * dm_dz;
            }
        }
    }
}

/**
 * @brief Computes the vorticity diagnostics for the tornado scheme.
 */
void TornadoScheme::compute_vorticity_diagnostics(
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
    const int j = 0;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int k = 1; k < NZ_ - 1; ++k)
        {
            const double r_inv = geo_.r_inv[i];
            double dur_dz = deriv_->dk(u, i, j, k);
            double duz_dr = deriv_->di(w, i, j, k);

            vorticity_r[i][j][k] = 0.0;
            vorticity_theta[i][j][k] = dur_dz - duz_dr;
            const double duth_dr = deriv_->di(v, i, j, k);
            vorticity_z[i][j][k] = duth_dr + v[i][j][k] * r_inv;

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                vorticity_r[i][jj][k] = vorticity_r[i][j][k];
                vorticity_theta[i][jj][k] = vorticity_theta[i][j][k];
                vorticity_z[i][jj][k] = vorticity_z[i][j][k];
            }

            double zeta = vorticity_z[i][j][k];
            double dw_dz = deriv_->dk(w, i, j, k);

            stretching_term[i][j][k] = zeta * dw_dz;

            tilting_term[i][j][k] = 0.0;

            baroclinic_term[i][j][k] = 0.0f;

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                stretching_term[i][jj][k] = stretching_term[i][j][k];
                tilting_term[i][jj][k] = tilting_term[i][j][k];
                baroclinic_term[i][jj][k] = baroclinic_term[i][j][k];
            }
        }
    }
}




// Derivative operators moved to DerivativeOperators (Phase B.2).
// CylindricalDerivatives is constructed in TornadoScheme::TornadoScheme().

/**
 * @brief Computes the radial mass flux.
 */
double TornadoScheme::compute_radial_mass_flux(const Field3D& u,
                                               const Field3D& rho,
                                               int i, int k) const 
{
    int j = 0;
    const double r = geo_.r[i];
    return rho[i][j][k] * r * u[i][j][k];
}

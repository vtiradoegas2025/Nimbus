/**
 * @file tornado.cpp
 * @brief Implementation for the dynamics module.
 *
 * Provides executable logic for the dynamics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/dynamics subsystem.
 */

#include "tornado.hpp"
#include "core/simulation.hpp"
#include "numerics/compute_kernel_template.hpp"
#include "util/grid_metric_utils.hpp"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif


TornadoScheme::TornadoScheme()
    : NR_(NR), NTH_(NTH), NZ_(NZ), dr_(dr), dtheta_(dtheta), dz_(dz)
{
}

/**
 * @brief Computes the momentum tendencies for the tornado scheme.
 */
void TornadoScheme::compute_momentum_tendencies(const Field3D& u_r,
    const Field3D& u_theta,
    const Field3D& u_z,
    const Field3D& rho,
    const Field3D& p,
    const Field3D& theta,
    double dt,
    Field3D& du_r_dt,
    Field3D& du_theta_dt,
    Field3D& du_z_dt,
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
                du_r_dt[i][j][k] = 0.0f;
                du_theta_dt[i][j][k] = 0.0f;
                du_z_dt[i][j][k] = 0.0f;
                drho_dt[i][j][k] = 0.0f;
                dp_dt[i][j][k] = 0.0f;
            }
        }
    }

    // Try GPU dispatch for interior points (only when terrain metrics are NOT active)
    if (!grid_metric::has_terrain_metrics(global_grid_metrics) &&
        dispatch_tornado_tendencies_backend(
            u_r.data(), u_theta.data(), u_z.data(),
            rho.data(), p.data(), theta.data(),
            du_r_dt.data(), du_theta_dt.data(), du_z_dt.data(),
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

    // CPU fallback — axisymmetric: compute at j=0, replicate to all j
    const int j = 0;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int k = 1; k < NZ_ - 1; ++k)
        {
            double r = i * dr_ + dynamics_constants::eps;
            double ur = u_r[i][j][k];
            double uth = u_theta[i][j][k];
            double uz = u_z[i][j][k];
            double rho_val = rho[i][j][k];
            double p_val = p[i][j][k];
            if (!std::isfinite(rho_val) || rho_val <= 1.0e-6)
            {
                rho_val = 1.0;
            }

            double dur_dr = compute_dr(u_r, i, j, k);
            double dur_dz = compute_dz(u_r, i, j, k);

            double duth_dr = compute_dr(u_theta, i, j, k);
            double duth_dz = compute_dz(u_theta, i, j, k);

            double duz_dr = compute_dr(u_z, i, j, k);
            double duz_dz = compute_dz(u_z, i, j, k);

            double dp_dr = compute_dr(p, i, j, k);
            double dp_dz = compute_dz(p, i, j, k);

            double advective_r = -ur * dur_dr - uz * dur_dz;
            double centrifugal = uth * uth / r;
            double pressure_grad_r = -dp_dr / rho_val;

            double du_r = advective_r + centrifugal + pressure_grad_r;
            if (!std::isfinite(du_r))
            {
                du_r = 0.0;
            }
            du_r_dt[i][j][k] = static_cast<float>(du_r);

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                du_r_dt[i][jj][k] = du_r_dt[i][j][k];
            }

            double advective_th = -ur * duth_dr - uz * duth_dz;
            double coriolis_th = -ur * uth / r;

            double fv = 0.0;

            if (i > 0 && i < NR_-1) 
            {
                double v_here = uth;
                double v_inner = (i > 0) ? u_theta[i-1][j][k] : 0.0;
                double v_outer = (i < NR_-1) ? u_theta[i+1][j][k] : 0.0;

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
            du_theta_dt[i][j][k] = static_cast<float>(du_th);

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                du_theta_dt[i][jj][k] = du_theta_dt[i][j][k];
            }

            double advective_z = -ur * duz_dr - uz * duz_dz;
            double pressure_grad_z = -dp_dz / rho_val;

            // Fully compressible vertical momentum equation:
            //   dw/dt = -(1/ρ) ∂p/∂z - g + advection
            // Buoyancy is implicit in (-∂p/∂z/ρ - g). See the matching comment
            // in src/dynamics/schemes/supercell/supercell.cpp for the full
            // explanation and docs/Journey.md Phase 2 "Bug 3: Double-counted
            // buoyancy". Do NOT reintroduce an explicit g·(θ-θ₀)/θ₀ term here.
            double du_z = advective_z + pressure_grad_z - dynamics_constants::g;
            if (!std::isfinite(du_z))
            {
                du_z = 0.0;
            }
            du_z_dt[i][j][k] = static_cast<float>(du_z);

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                du_z_dt[i][jj][k] = du_z_dt[i][j][k];
            }

            double drho_dt_val = -rho_val * (dur_dr + ur / r + duz_dz);
            if (!std::isfinite(drho_dt_val))
            {
                drho_dt_val = 0.0;
            }
            drho_dt[i][j][k] = drho_dt_val;

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                drho_dt[i][jj][k] = drho_dt_val;
            }

            double cyclostrophic_dp_dr = rho_val * uth * uth / r;
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
    const Field3D& u_r,
    const Field3D& u_theta,
    Field3D& angular_momentum,
    Field3D& angular_momentum_tendency)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            const double r = i * dr_ + dynamics_constants::eps;
            for (int k = 0; k < NZ_; ++k)
            {
                double v_theta = u_theta[i][j][k];
                angular_momentum[i][j][k] = r * v_theta;

                double ur = u_r[i][j][k];
                double uz = w[i][j][k];

                double dm_dr = 0.0, dm_dz = 0.0;

                if (i > 0 && i < NR_-1) dm_dr = compute_dr(angular_momentum, i, j, k);
                if (k > 0 && k < NZ_-1) dm_dz = compute_dz(angular_momentum, i, j, k);

                angular_momentum_tendency[i][j][k] = -ur * dm_dr - uz * dm_dz;
            }
        }
    }
}

/**
 * @brief Computes the vorticity diagnostics for the tornado scheme.
 */
void TornadoScheme::compute_vorticity_diagnostics(
    const Field3D& u_r,
    const Field3D& u_theta,
    const Field3D& u_z,
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
            double r = i * dr_ + dynamics_constants::eps;
            double dur_dz = compute_dz(u_r, i, j, k);
            double duz_dr = compute_dr(u_z, i, j, k);

            vorticity_r[i][j][k] = 0.0;
            vorticity_theta[i][j][k] = dur_dz - duz_dr;
            const double duth_dr = compute_dr(u_theta, i, j, k);
            vorticity_z[i][j][k] = duth_dr + (u_theta[i][j][k] / r);

            for (int jj = 1; jj < NTH_; ++jj) 
            {
                vorticity_r[i][jj][k] = vorticity_r[i][j][k];
                vorticity_theta[i][jj][k] = vorticity_theta[i][j][k];
                vorticity_z[i][jj][k] = vorticity_z[i][j][k];
            }

            double zeta = vorticity_z[i][j][k];
            double dw_dz = compute_dz(u_z, i, j, k);

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




/**
 * @brief Computes the derivative in the radial direction.
 */
double TornadoScheme::compute_dr(const Field3D& field, int i, int j, int k) const 
{
    const double dx_local = std::max(grid_metric::local_dx(global_grid_metrics, i, j, k), 1.0e-6);
    return (field[i + 1][j][k] - field[i - 1][j][k]) / (2.0 * dx_local);
}

/**
 * @brief Computes the derivative in the vertical direction.
 */
double TornadoScheme::compute_dz(const Field3D& field, int i, int j, int k) const 
{
    const double denom = std::max(grid_metric::centered_dz_span(global_grid_metrics, i, j, k, NZ_), 1.0e-6);
    return (field[i][j][k + 1] - field[i][j][k - 1]) / denom;
}

/**
 * @brief Computes the radial mass flux.
 */
double TornadoScheme::compute_radial_mass_flux(const Field3D& u_r,
                                               const Field3D& rho,
                                               int i, int k) const 
{
    int j = 0;
    double r = i * dr_ + dynamics_constants::eps;
    return rho[i][j][k] * r * u_r[i][j][k];
}

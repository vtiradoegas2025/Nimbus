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
#include "numerics/compute_kernel_template.hpp"
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
      dr_(dr), dtheta_(dtheta), dz_(dz) 
{
}

/**
 * @brief Computes the momentum tendencies for the supercell scheme.
 */

void SupercellScheme::compute_momentum_tendencies(
    const Field3D& u_r,
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
        dispatch_supercell_tendencies_backend(
            u_r.data(), u_theta.data(), u_z.data(),
            rho.data(), p.data(), theta.data(),
            du_r_dt.data(), du_theta_dt.data(), du_z_dt.data(),
            drho_dt.data(), dp_dt.data(),
            NR_, NTH_, NZ_,
            static_cast<float>(dr_), static_cast<float>(dtheta_), static_cast<float>(dz_),
            static_cast<float>(dynamics_constants::g),
            static_cast<float>(dynamics_constants::gamma),
            static_cast<float>(dynamics_constants::theta0)))
    {
        return;
    }

    // CPU fallback
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            double r = i * dr_ + dynamics_constants::eps;
            int j_prev = (j - 1 + NTH_) % NTH_;
            int j_next = (j + 1) % NTH_;

            for (int k = 1; k < NZ_ - 1; ++k) 
            {
                double ur = u_r[i][j][k];
                double uth = u_theta[i][j][k];
                double uz = u_z[i][j][k];
                double rho_val = rho[i][j][k];
                double p_val = p[i][j][k];
                double theta_val = theta[i][j][k];
                const double rho_safe = (std::isfinite(rho_val) && rho_val > 1.0e-6) ? rho_val : 1.0;

                double dur_dr = compute_dr(u_r, i, j, k);
                double dur_dth = compute_dtheta(u_r, i, j, k);
                double dur_dz = compute_dz(u_r, i, j, k);

                double duth_dr = compute_dr(u_theta, i, j, k);
                double duth_dth = compute_dtheta(u_theta, i, j, k);
                double duth_dz = compute_dz(u_theta, i, j, k);

                double duz_dr = compute_dr(u_z, i, j, k);
                double duz_dth = compute_dtheta(u_z, i, j, k);
                double duz_dz = compute_dz(u_z, i, j, k);

                double dp_dr = compute_dr(p, i, j, k);
                double dp_dth = compute_dtheta(p, i, j, k);
                double dp_dz = compute_dz(p, i, j, k);

                double advective_r = -ur * dur_dr - (uth / r) * dur_dth - uz * dur_dz;
                double centrifugal = uth * uth / r;
                double pressure_grad_r = -dp_dr / rho_safe;

                double du_r = advective_r + centrifugal + pressure_grad_r;
                if (!std::isfinite(du_r))
                {
                    du_r = 0.0;
                }
                du_r_dt[i][j][k] = static_cast<float>(du_r);

                double advective_th = -ur * duth_dr - (uth / r) * duth_dth - uz * duth_dz;
                double coriolis_th = -ur * uth / r;
                double pressure_grad_th = -dp_dth / (rho_safe * r);

                double du_theta = advective_th + coriolis_th + pressure_grad_th;
                if (!std::isfinite(du_theta))
                {
                    du_theta = 0.0;
                }
                du_theta_dt[i][j][k] = static_cast<float>(du_theta);

                double advective_z = -ur * duz_dr - (uth / r) * duz_dth - uz * duz_dz;
                double pressure_grad_z = -dp_dz / rho_safe;

                double theta_prime = theta_val - theta0;
                double buoyancy = dynamics_constants::g * (theta_prime / theta0);

                double du_z = advective_z + pressure_grad_z - dynamics_constants::g + buoyancy;
                if (!std::isfinite(du_z))
                {
                    du_z = 0.0;
                }
                du_z_dt[i][j][k] = static_cast<float>(du_z);

                double divergence = dur_dr + duz_dz + ur / r + duth_dth / r;
                double drho = -rho_safe * divergence;
                if (!std::isfinite(drho))
                {
                    drho = 0.0;
                }
                drho_dt[i][j][k] = static_cast<float>(drho);

                double gamma_term = dynamics_constants::gamma * p_val * divergence;
                double advection_p = -ur * dp_dr - (uth / r) * dp_dth - uz * dp_dz;
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

/**
 * @brief Computes the vorticity diagnostics for the supercell scheme.
 */
void SupercellScheme::compute_vorticity_diagnostics(
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
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            double r = i * dr_ + dynamics_constants::eps;
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                double duth_dz = compute_dz(u_theta, i, j, k);
                double duz_dth = compute_dtheta(u_z, i, j, k);
                vorticity_r[i][j][k] = static_cast<float>((duz_dth / r) - duth_dz);
                if (!std::isfinite(vorticity_r[i][j][k]))
                {
                    vorticity_r[i][j][k] = 0.0f;
                }

                const double duz_dr = compute_dr(u_z, i, j, k);
                const double dur_dz = compute_dz(u_r, i, j, k);
                vorticity_theta[i][j][k] = static_cast<float>(dur_dz - duz_dr);
                if (!std::isfinite(vorticity_theta[i][j][k]))
                {
                    vorticity_theta[i][j][k] = 0.0f;
                }

                const double duth_dr = compute_dr(u_theta, i, j, k);
                const double dur_dth = compute_dtheta(u_r, i, j, k);
                const double v_theta_local = static_cast<double>(u_theta[i][j][k]);
                const double zeta = duth_dr + (v_theta_local - dur_dth) / r;
                vorticity_z[i][j][k] = static_cast<float>(std::isfinite(zeta) ? zeta : 0.0);
                
                double dw_dz = compute_dz(u_z, i, j, k);
                stretching_term[i][j][k] = static_cast<float>(zeta * dw_dz);
                if (!std::isfinite(stretching_term[i][j][k]))
                {
                    stretching_term[i][j][k] = 0.0f;
                }

                double dw_dr = compute_dr(u_z, i, j, k);
                double dv_dz = compute_dz(u_theta, i, j, k);
                double dw_dth = compute_dtheta(u_z, i, j, k);
                double du_dz = compute_dz(u_r, i, j, k);
                tilting_term[i][j][k] = dw_dr * dv_dz - (dw_dth / r) * du_dz;
                if (!std::isfinite(tilting_term[i][j][k]))
                {
                    tilting_term[i][j][k] = 0.0f;
                }

                double drho_dr = compute_dr(rho, i, j, k);
                double drho_dth = compute_dtheta(rho, i, j, k);
                double drho_dz = compute_dz(rho, i, j, k);

                double dp_dr = compute_dr(p, i, j, k);
                double dp_dth = compute_dtheta(p, i, j, k);
                double dp_dz = compute_dz(p, i, j, k);

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
    const Field3D& u_r,
    const Field3D& u_theta,
    const Field3D& u_z,
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
                double dur_dr = compute_dr(u_r, i, j, k);
                double dur_dth = compute_dtheta(u_r, i, j, k);
                double duth_dr = compute_dr(u_theta, i, j, k);
                double duth_dth = compute_dtheta(u_theta, i, j, k);
                double duz_dz = compute_dz(u_z, i, j, k);

                double deformation = dur_dr * dur_dr + (1.0/(i*dr_+dynamics_constants::eps)) *
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


/**
 * @brief Computes the derivative in the radial direction.
 */
double SupercellScheme::compute_dr(const Field3D& field, int i, int j, int k) const 
{
    const double dx_local = std::max(grid_metric::local_dx(global_grid_metrics, i, j, k), 1.0e-6);
    return (field[i + 1][j][k] - field[i - 1][j][k]) / (2.0 * dx_local);
}

/**
 * @brief Computes the derivative in the azimuthal direction.
 */
double SupercellScheme::compute_dtheta(const Field3D& field, int i, int j, int k) const 
{
    int j_prev = (j - 1 + NTH_) % NTH_;
    int j_next = (j + 1) % NTH_;
    return (field[i][j_next][k] - field[i][j_prev][k]) / (2 * dtheta_);
}

/**
 * @brief Computes the derivative in the vertical direction.
 */
double SupercellScheme::compute_dz(const Field3D& field, int i, int j, int k) const 
{
    const double denom = std::max(grid_metric::centered_dz_span(global_grid_metrics, i, j, k, NZ_), 1.0e-6);
    return (field[i][j][k + 1] - field[i][j][k - 1]) / denom;
}

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
    return (1.0 / r) * (dr_u_theta - dtheta_u_r) + dr_u_theta / r;
}

/**
 * @brief Computes the buoyancy.
 */
double SupercellScheme::compute_buoyancy(double theta_prime, double rho, double rho0) const 
{
    return dynamics_constants::g * (theta_prime / theta0) * (rho0 / rho);
}

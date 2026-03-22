/**
 * @file weno5.cpp
 * @brief Implementation for the numerics module.
 *
 * Provides executable logic for the numerics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/numerics subsystem.
 */

#include "weno5.hpp"
#include "util/grid_metric_utils.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

namespace
{
std::vector<double> build_vertical_spacing_column(const GridMetrics& grid, int i, int j, int nz)
{
    std::vector<double> dz_col(static_cast<std::size_t>(std::max(0, nz)), 1.0);
    for (int k = 0; k < nz; ++k)
    {
        dz_col[static_cast<std::size_t>(k)] = grid_metric::local_dz(grid, i, j, k, nz);
    }
    return dz_col;
}
}

/**
 * @brief Implements the WENO5 advection scheme.
 */
WENO5Scheme::WENO5Scheme() : weno_epsilon_(numerics_constants::weno_epsilon) {}

/**
 * @brief Initializes the WENO5 advection scheme.
 */
void WENO5Scheme::initialize() {initialize(AdvectionConfig{});}

/**
 * @brief Initializes the WENO5 advection scheme.
 */
void WENO5Scheme::initialize(const AdvectionConfig& cfg) 
{
    config_ = cfg;
    weno_epsilon_ = numerics_constants::weno_epsilon;

    std::cout << "Initialized WENO5 advection scheme" << std::endl;

    if (cfg.positivity) 
    {
        std::cout << "  Positivity preservation enabled" << std::endl;
    }
}

/**
 * @brief Computes the flux divergence.
 */
void WENO5Scheme::compute_flux_divergence(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt
) 
{
    if (!state.q || !state.grid || !state.w) 
    {
        throw std::runtime_error("Invalid state for WENO5 advection");
    }

    const int NR = state.q->size_r();
    const int NTH = state.q->size_th();
    const int NZ = state.q->size_z();
    const double limiter_dt = (std::isfinite(cfg.positivity_dt) && cfg.positivity_dt > 0.0)
        ? cfg.positivity_dt
        : 1.0;

    tendencies.dqdt_adv.resize(NR, NTH, NZ, 0.0f);

    double max_cfl = 0.0;


    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            std::vector<double> q_col(NZ);
            std::vector<double> w_col(NZ);
            for (int k = 0; k < NZ; ++k) {
                q_col[k] = static_cast<double>((*state.q)[i][j][k]);
                w_col[k] = static_cast<double>((*state.w)[i][j][k]);
            }
            const std::vector<double> dz_col = build_vertical_spacing_column(*state.grid, i, j, NZ);

            std::vector<double> dqdt_col(NZ, 0.0);
            weno5_advect_1d(q_col, w_col, dz_col, limiter_dt, dqdt_col);

            for (int k = 0; k < NZ; ++k)
            {
                tendencies.dqdt_adv[i][j][k] = static_cast<float>(dqdt_col[k]);
            }

            for (int k = 0; k < NZ; ++k) 
            {
                const double dz_k = dz_col[static_cast<std::size_t>(k)];
                double cfl = std::abs(w_col[k]) / std::max(dz_k, 1.0e-6);
                max_cfl = std::max(max_cfl, cfl);
            }
        }
    }

    if (diag_opt) 
    {
        diag_opt->max_cfl_z = max_cfl;
        diag_opt->suggested_dt = (max_cfl > 1.0e-12)
            ? (cfg.cfl_target / max_cfl)
            : std::numeric_limits<double>::infinity();
    }
}

/**
 * @brief Suggests the time step.
 */
double WENO5Scheme::suggest_dt(const AdvectionConfig& cfg,const AdvectionStateView& state) 
{
    if (!state.grid || !state.q) return 1.0;

    const int NR = state.q->size_r();
    const int NTH = state.q->size_th();
    const int NZ = state.q->size_z();

    double max_cfl = 0.0;
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                if (state.u) 
                {
                    const double dx_local = grid_metric::local_dx(*state.grid, i, j, k);
                    double cfl_x = std::abs(static_cast<double>((*state.u)[i][j][k])) / std::max(dx_local, 1.0e-6);
                    max_cfl = std::max(max_cfl, cfl_x);
                }

                if (state.v) 
                {
                    const double dy_local = grid_metric::local_dy(*state.grid, i, j, k);
                    double cfl_y = std::abs(static_cast<double>((*state.v)[i][j][k])) / std::max(dy_local, 1.0e-6);
                    max_cfl = std::max(max_cfl, cfl_y);
                }

                if (state.w) 
                {
                    const double dz_k = grid_metric::local_dz(*state.grid, i, j, k, NZ);
                    double cfl_z = std::abs(static_cast<double>((*state.w)[i][j][k])) / std::max(dz_k, 1.0e-6);
                    max_cfl = std::max(max_cfl, cfl_z);
                }
            }
        }
    }

    return (max_cfl > 1.0e-12) ? (cfg.cfl_target / max_cfl) : 1.0;
}


/**
 * @brief Performs the WENO5 reconstruction for the left state.
 */
double WENO5Scheme::weno5_reconstruct_left(const std::vector<double>& q, int i) 
{
    if (i < 2 || i >= static_cast<int>(q.size()) - 3) 
    {
        return q[i];
    }

    double q0 = q_tilde_0(q, i);
    double q1 = q_tilde_1(q, i);
    double q2 = q_tilde_2(q, i);

    double beta0 = smoothness_indicator(q, i-2, i+1);
    double beta1 = smoothness_indicator(q, i-1, i+2);
    double beta2 = smoothness_indicator(q, i, i+3);

    double alpha0 = d0 / ((weno_epsilon_ + beta0) * (weno_epsilon_ + beta0));
    double alpha1 = d1 / ((weno_epsilon_ + beta1) * (weno_epsilon_ + beta1));
    double alpha2 = d2 / ((weno_epsilon_ + beta2) * (weno_epsilon_ + beta2));

    double sum_alpha = alpha0 + alpha1 + alpha2;
    if (!std::isfinite(sum_alpha) || sum_alpha <= 0.0)
    {
        return q[i];
    }
    double omega0 = alpha0 / sum_alpha;
    double omega1 = alpha1 / sum_alpha;
    double omega2 = alpha2 / sum_alpha;

    return omega0 * q0 + omega1 * q1 + omega2 * q2;
}

/**
 * @brief Performs the WENO5 reconstruction for the right state.
 */
double WENO5Scheme::weno5_reconstruct_right(const std::vector<double>& q, int i) 
{
    if (i < 1 || i >= static_cast<int>(q.size()) - 4) {
        return q[i+1];
    }

    double q0 = q_tilde_0(q, i+1);
    double q1 = q_tilde_1(q, i+1);
    double q2 = q_tilde_2(q, i+1);

    double beta0 = smoothness_indicator(q, i-1, i+2);
    double beta1 = smoothness_indicator(q, i, i+3);
    double beta2 = smoothness_indicator(q, i+1, i+4);

    double alpha0 = d0 / ((weno_epsilon_ + beta0) * (weno_epsilon_ + beta0));
    double alpha1 = d1 / ((weno_epsilon_ + beta1) * (weno_epsilon_ + beta1));
    double alpha2 = d2 / ((weno_epsilon_ + beta2) * (weno_epsilon_ + beta2));

    double sum_alpha = alpha0 + alpha1 + alpha2;
    if (!std::isfinite(sum_alpha) || sum_alpha <= 0.0)
    {
        return q[i+1];
    }
    double omega0 = alpha0 / sum_alpha;
    double omega1 = alpha1 / sum_alpha;
    double omega2 = alpha2 / sum_alpha;

    return omega0 * q0 + omega1 * q1 + omega2 * q2;
}

/**
 * @brief Computes the smoothness indicator.
 */
double WENO5Scheme::smoothness_indicator(const std::vector<double>& q, int start, int end) 
{
    if (start < 0 || end >= static_cast<int>(q.size()) || end - start != 3) 
    {
        return 1e6;
    }

    double beta = 0.0;
    if (!std::isfinite(q[start]) || !std::isfinite(q[start+1]) || !std::isfinite(q[start+2]))
    {
        return 1e6;
    }
    beta += 13.0/12.0 * std::pow(q[start] - 2*q[start+1] + q[start+2], 2);
    beta += 1.0/4.0 * std::pow(q[start] - 4*q[start+1] + 3*q[start+2], 2);

    return beta;
}

/**
 * @brief Computes the q_tilde_0.
 */
double WENO5Scheme::q_tilde_0(const std::vector<double>& q, int i) 
{
    return (1.0/3.0)*q[i-2] - (7.0/6.0)*q[i-1] + (11.0/6.0)*q[i];
}

/**
 * @brief Computes the q_tilde_1.
 */
double WENO5Scheme::q_tilde_1(const std::vector<double>& q, int i) 
{
    return -(1.0/6.0)*q[i-1] + (5.0/6.0)*q[i] + (1.0/3.0)*q[i+1];
}

/**
 * @brief Computes the q_tilde_2.
 */
double WENO5Scheme::q_tilde_2(const std::vector<double>& q, int i) 
{
    return (1.0/3.0)*q[i] + (5.0/6.0)*q[i+1] - (1.0/6.0)*q[i+2];
}

/**
 * @brief Computes the Rusanov flux.
 */
double WENO5Scheme::rusanov_flux(double q_left, double q_right, double velocity) 
{
    double alpha = std::abs(velocity);
    return 0.5 * velocity * (q_left + q_right) - 0.5 * alpha * (q_right - q_left);
}

/**
 * @brief Advects the 1D field.
 */
void WENO5Scheme::weno5_advect_1d(
    const std::vector<double>& q,
    const std::vector<double>& velocity,
    const std::vector<double>& dx,
    double dt,
    std::vector<double>& dqdt
) 
{
    const int n = q.size();
    dqdt.assign(n, 0.0);
    if (n == 0)
    {
        return;
    }
    auto dx_at = [&](int idx) -> double
    {
        if (idx >= 0 && idx < static_cast<int>(dx.size()))
        {
            return std::max(std::abs(dx[static_cast<std::size_t>(idx)]), 1.0e-6);
        }
        return 1.0;
    };

    std::vector<double> flux(n+1, 0.0);

    for (int i = 0; i < n-1; ++i) 
    {
        double q_left = weno5_reconstruct_left(q, i);
        double q_right = weno5_reconstruct_right(q, i);

        flux[i+1] = rusanov_flux(q_left, q_right, velocity[i]);
    }

    flux[0] = 0.0;
    flux[n] = 0.0;

    for (int i = 0; i < n; ++i) 
    {
        double dflux_dx = (flux[i+1] - flux[i]) / dx_at(i);
        dqdt[i] = -dflux_dx;
    }

    if (config_.positivity) 
    {
        apply_positivity_preservation(q, dqdt, dt);
    }
}

/**
 * @brief Applies the positivity preservation.
 */
void WENO5Scheme::apply_positivity_preservation(
    const std::vector<double>& q,
    std::vector<double>& dqdt,
    double dt,
    double min_value) 
{
    const std::size_t count = std::min(q.size(), dqdt.size());
    const double dt_safe = std::max(std::abs(dt), 1.0e-12);

    for (std::size_t idx = 0; idx < count; ++idx)
    {
        double tendency = dqdt[idx];
        if (!std::isfinite(tendency))
        {
            tendency = 0.0;
        }
        const double floor_tendency = (min_value - q[idx]) / dt_safe;
        if (tendency < floor_tendency)
        {
            tendency = floor_tendency;
        }
        dqdt[idx] = tendency;
    }
}

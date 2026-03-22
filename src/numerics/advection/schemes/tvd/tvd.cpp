/**
 * @file tvd.cpp
 * @brief Implementation for the numerics module.
 *
 * Provides executable logic for the numerics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/numerics subsystem.
 */

#include "tvd.hpp"
#include "util/grid_metric_utils.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
std::string canonicalize_limiter_id(std::string value)
{
    std::string canonical;
    canonical.reserve(value.size());
    for (unsigned char c : value)
    {
        if (c == '_' || c == '-' || std::isspace(c))
        {
            continue;
        }
        canonical.push_back(static_cast<char>(std::tolower(c)));
    }
    return canonical;
}

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
 * @brief Initializes the TVD scheme with a default limiter function.
 */
TVDScheme::TVDScheme() : limiter_function_(nullptr) 
{
}

/**
 * @brief Initializes the TVD scheme with a default configuration.
 */
void TVDScheme::initialize() 
{
    initialize(AdvectionConfig{});
}

/**
 * @brief Initializes the TVD scheme with a configuration.
 */
void TVDScheme::initialize(const AdvectionConfig& cfg) 
{
    config_ = cfg;
    const std::string limiter_id = canonicalize_limiter_id(cfg.limiter_id);
    std::string selected_limiter = "mc";

    if (limiter_id == "minmod") 
    {
        limiter_function_ = &minmod_limiter;
        selected_limiter = "minmod";
    } 

    else if (limiter_id == "vanleer") 
    {
        limiter_function_ = &vanleer_limiter;
        selected_limiter = "vanleer";
    } 
    
    else if (limiter_id == "superbee") 
    {
        limiter_function_ = &superbee_limiter;
        selected_limiter = "superbee";
    } 

    else if (limiter_id == "mc") 
    {
        limiter_function_ = &mc_limiter;
        selected_limiter = "mc";
    } 
    
    else if (limiter_id == "universal") 
    {
        limiter_function_ = &universal_limiter;
        selected_limiter = "universal";
    } 
    else 
    {
        std::cerr << "Warning: Unknown limiter '" << cfg.limiter_id << "', using MC limiter" << std::endl;
        limiter_function_ = &mc_limiter;
        selected_limiter = "mc";
    }

    config_.limiter_id = selected_limiter;
    std::cout << "Initialized TVD advection scheme with " << selected_limiter << " limiter" << std::endl;
}


/**
 * @brief Computes the minmod limiter.
 */
double TVDScheme::minmod_limiter(double r) 
{
    return std::max(0.0, std::min(1.0, r));
}


/**
 * @brief Computes the vanleer limiter.
 */
double TVDScheme::vanleer_limiter(double r) 
{
    return (std::abs(r) + r) / (1.0 + std::abs(r));
}


/**
 * @brief Computes the superbee limiter.
 */
double TVDScheme::superbee_limiter(double r) 
{
    return std::max({0.0, std::min(1.0, 2.0 * r), std::min(2.0, r)});
}


/**
 * @brief Computes the mc limiter.
 */
double TVDScheme::mc_limiter(double r) 
{
    return std::max(0.0, std::min({(1.0 + r) / 2.0, 2.0, 2.0 * r}));
}


/**
 * @brief Computes the universal limiter.
 */
double TVDScheme::universal_limiter(double r) 
{
    double phi = 0.0;

    if (r >= 0.0 && r <= 1.0) 
    {
        phi = std::min(2.0 * r, (1.0 + r) / 2.0);
    } 

    else if (r > 1.0) 
    {
        phi = std::min(r, 2.0);
    }
    return phi;
}


/**
 * @brief Computes the flux divergence.
 */
void TVDScheme::compute_flux_divergence(const AdvectionConfig& cfg, const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt
) 
{
    if (!state.q || !state.grid || !state.w) {throw std::runtime_error("Invalid state for TVD advection");}

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
            for (int k = 0; k < NZ; ++k) 
            {
                q_col[k] = static_cast<double>((*state.q)[i][j][k]);
                w_col[k] = static_cast<double>((*state.w)[i][j][k]);
            }
            const std::vector<double> dz_col = build_vertical_spacing_column(*state.grid, i, j, NZ);

            std::vector<double> dqdt_col(NZ, 0.0);
            advect_1d(q_col, w_col, dz_col, limiter_dt, dqdt_col);

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
double TVDScheme::suggest_dt(const AdvectionConfig& cfg, const AdvectionStateView& state) 
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
 * @brief Performs the MUSCL reconstruction.
 */
void TVDScheme::muscl_reconstruct(const std::vector<double>& q, std::vector<double>& q_left,
    std::vector<double>& q_right, double (*limiter)(double)) 
{
    const int n = q.size();
    if (n == 0)
    {
        q_left.clear();
        q_right.clear();
        return;
    }

    q_left.resize(n);
    q_right.resize(n);

    for (int i = 1; i < n - 1; ++i) 
    {
        double denominator = q[i+1] - q[i] + numerics_constants::epsilon;
        double r = (q[i] - q[i-1]) / denominator;

        double phi = limiter(r);

        double delta_q = phi * (q[i+1] - q[i]);

        q_left[i] = q[i] - 0.5 * delta_q;
        q_right[i] = q[i] + 0.5 * delta_q;
    }

    q_left[0] = q[0];
    q_right[0] = q[0];
    q_left[n-1] = q[n-1];
    q_right[n-1] = q[n-1];
}


/**
 * @brief Computes the numerical flux.
 */
double TVDScheme::numerical_flux(double q_left, double q_right, double velocity) 
{
    if (velocity >= 0.0) 
    {
        return velocity * q_left;
    } 
    else 
    {
        return velocity * q_right;
    }
}

/**
 * @brief Advects the 1D field.
 */
void TVDScheme::advect_1d(const std::vector<double>& q,const std::vector<double>& velocity, const std::vector<double>& dx,
     double dt,std::vector<double>& dqdt) 
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

    std::vector<double> q_left, q_right;
    muscl_reconstruct(q, q_left, q_right, limiter_function_);

    for (int i = 0; i < n - 1; ++i) 
    {
        double flux_right = numerical_flux(q_right[i], q_left[i+1], velocity[i]);
        double flux_left = (i > 0) ? numerical_flux(q_right[i-1], q_left[i], velocity[i-1]) : 0.0;

        double dflux_dx = (flux_right - flux_left) / dx_at(i);

        dqdt[i] -= dflux_dx;
    }

    if (config_.positivity) 
    {
        apply_positivity_limiter(q, dqdt, dt);
    }
}


/**
 * @brief Applies the positivity limiter.
 */
void TVDScheme::apply_positivity_limiter(const std::vector<double>& q, std::vector<double>& dqdt, double dt, double min_value) 
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

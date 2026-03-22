/**
 * @file tke.cpp
 * @brief Implementation for the turbulence module.
 *
 * Provides executable logic for the turbulence runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/turbulence subsystem.
 */

#include "tke.hpp"
#include "core/field_pool.hpp"
#include "util/grid_metric_utils.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <cmath>

namespace
{
inline bool field_matches_state(const Field3D& field, const TurbulenceStateView& state)
{
    return field.size_r() == state.NR &&
           field.size_th() == state.NTH &&
           field.size_z() == state.NZ;
}
}



/**
 * @brief Initializes the TKE turbulence scheme.
 */
TKEScheme::TKEScheme()
    : Ce1_(1.0), Ce2_(1.33), c_k_(0.1), c_eps_(0.19),
      Pr_t_(0.7), Sc_t_(0.7), c_l_(0.15), l_max_(500.0) {
}

/**
 * @brief Gets the required fields.
 */
int TKEScheme::required_fields() const 
{
    return static_cast<int>(TurbulenceRequirements::BASIC) |
           static_cast<int>(TurbulenceRequirements::TKE);
}

/**
 * @brief Initializes the TKE turbulence scheme.
 */ 
void TKEScheme::initialize(const TurbulenceConfig& cfg) 
{
    Ce1_ = 1.0;
    Ce2_ = 1.33;
    c_k_ = 0.1;
    c_eps_ = 0.19;
    Pr_t_ = cfg.Pr_t;
    Sc_t_ = cfg.Sc_t;
    c_l_ = 0.15;
    l_max_ = 500.0;

    tmv::log_info("Initialized TKE Turbulence (1.5-order closure):");
    tmv::log_info("  Prognostic TKE: enabled");
    tmv::log_info("  Ce1 = ", Ce1_, ", Ce2 = ", Ce2_);
    tmv::log_info("  Max mixing length = ", l_max_, " m");
}

/**
 * @brief Computes the TKE turbulence scheme.
 */
void TKEScheme::compute(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state,
    TurbulenceTendencies& tend,
    TurbulenceDiagnostics* diag_opt
) 
{
    const bool external_tke_ready = state.tke && field_matches_state(*state.tke, state);
    if (external_tke_ready)
    {
        tke_ = *state.tke;
    }
    else if (!field_matches_state(tke_, state))
    {
        tke_.resize(state.NR, state.NTH, state.NZ, 0.1f);
    }

    tend.dudt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    tend.dvdt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    tend.dwdt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    tend.dthetadt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    tend.dqvdt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
    tend.dtkedt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);

    compute_eddy_coefficients_from_tke(cfg, grid, state);

    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu_t_, tend.dudt_sgs, tend.dvdt_sgs, tend.dwdt_sgs
    );

    eddy_viscosity::compute_scalar_diffusion_tendencies(
        cfg, state, grid, K_theta_, *state.theta, tend.dthetadt_sgs
    );

    if (state.qv) 
    {
        eddy_viscosity::compute_scalar_diffusion_tendencies(
            cfg, state, grid, K_q_, *state.qv, tend.dqvdt_sgs
        );
    }

    update_tke_prognostic(cfg, grid, state, tend.dtkedt_sgs);

    if (diag_opt) 
    {
        diag_opt->nu_t = nu_t_;
        diag_opt->K_theta = K_theta_;
        diag_opt->K_q = K_q_;
        diag_opt->K_tke = K_tke_;

        auto& pool = FieldPool::instance();
        auto sp_guard = pool.scoped_acquire(state.NR, state.NTH, state.NZ);
        auto bp_guard = pool.scoped_acquire(state.NR, state.NTH, state.NZ);
        Field3D& shear_prod = sp_guard.field;
        Field3D& buoy_prod = bp_guard.field;
        diag_opt->dissipation.resize(state.NR, state.NTH, state.NZ, 0.0f);

        compute_tke_production(state, grid, nu_t_, shear_prod, buoy_prod);

        diag_opt->shear_prod = shear_prod;
        diag_opt->buoy_prod = buoy_prod;

        for (int i = 0; i < state.NR; ++i) 
        {

            for (int j = 0; j < state.NTH; ++j)
            {

                for (int k = 0; k < state.NZ; ++k) 
                {
                    double e = static_cast<double>(tke_[i][j][k]);
                    double l = compute_mixing_length(cfg, grid, e,
                        eddy_viscosity::compute_brunt_vaisala_frequency(state, grid, i, j, k), i, j, k);
                    double eps = c_eps_ * std::pow(e, 1.5) / l;
                    diag_opt->dissipation[i][j][k] = static_cast<float>(eps);
                }
            }
        }
    }
}


/**
 * @brief Computes the mixing length.
 */
double TKEScheme::compute_mixing_length(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    double e,
    double N,
    int i, int j, int k
) 
{
    double Delta = eddy_viscosity::compute_filter_width(cfg, grid, i, j, k);

    const double e_safe = std::max(e, 0.0);
    double l_stability = (N > 1e-6) ? c_l_ * std::sqrt(e_safe) / N : Delta;

    double l = std::min(Delta, l_stability);

    return std::max(1.0e-3, std::min(l, l_max_));
}

/**
 * @brief Computes the eddy coefficients from the TKE.
 */
void TKEScheme::compute_eddy_coefficients_from_tke(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state
) {
    nu_t_.resize(state.NR, state.NTH, state.NZ, 0.0f);
    K_theta_.resize(state.NR, state.NTH, state.NZ, 0.0f);
    K_q_.resize(state.NR, state.NTH, state.NZ, 0.0f);
    K_tke_.resize(state.NR, state.NTH, state.NZ, 0.0f);

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < state.NR; ++i)
    {
        for (int j = 0; j < state.NTH; ++j)
        {
            for (int k = 0; k < state.NZ; ++k)
            {
                double e = static_cast<double>(tke_[i][j][k]);
                if (!std::isfinite(e))
                {
                    e = 1.0e-3;
                }
                e = std::max(e, 0.0);
                double N = eddy_viscosity::compute_brunt_vaisala_frequency(state, grid, i, j, k);
                if (!std::isfinite(N) || N < 0.0)
                {
                    N = 0.0;
                }

                double l = compute_mixing_length(cfg, grid, e, N, i, j, k);

                double nu_t = c_k_ * l * std::sqrt(std::max(e, 1e-6));
                if (!std::isfinite(nu_t) || nu_t < 0.0)
                {
                    nu_t = 0.0;
                }

                nu_t = std::min(nu_t, static_cast<double>(cfg.nu_t_max));

                nu_t_[i][j][k] = static_cast<float>(nu_t);

                double K_theta, K_q, K_tke;
                eddy_viscosity::compute_eddy_diffusivities(nu_t, Pr_t_, Sc_t_, K_theta, K_q, K_tke);
                if (!std::isfinite(K_theta) || K_theta < 0.0) K_theta = 0.0;
                if (!std::isfinite(K_q) || K_q < 0.0) K_q = 0.0;
                if (!std::isfinite(K_tke) || K_tke < 0.0) K_tke = 0.0;

                K_theta = std::min(K_theta, static_cast<double>(cfg.K_max));
                K_q = std::min(K_q, static_cast<double>(cfg.K_max));
                K_tke = std::min(K_tke, static_cast<double>(cfg.K_max));

                K_theta_[i][j][k] = static_cast<float>(K_theta);
                K_q_[i][j][k] = static_cast<float>(K_q);
                K_tke_[i][j][k] = static_cast<float>(K_tke);
            }
        }
    }
}

/**
 * @brief Updates the TKE prognostic equation.
 */
void TKEScheme::update_tke_prognostic(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state,
    Field3D& dtke_dt
) 
{
    dtke_dt.resize(state.NR, state.NTH, state.NZ, 0.0f);

    auto& pool = FieldPool::instance();
    auto sp_guard = pool.scoped_acquire(state.NR, state.NTH, state.NZ);
    auto bp_guard = pool.scoped_acquire(state.NR, state.NTH, state.NZ);
    Field3D& shear_prod = sp_guard.field;
    Field3D& buoy_prod = bp_guard.field;

    compute_tke_production(state, grid, nu_t_, shear_prod, buoy_prod);

    for (int i = 0; i < state.NR; ++i) 
    {
        for (int j = 0; j < state.NTH; ++j) 
        {
            for (int k = 0; k < state.NZ; ++k) 
            {
                double e = static_cast<double>(tke_[i][j][k]);
                if (!std::isfinite(e))
                {
                    e = 1.0e-3;
                }
                e = std::max(e, 0.0);
                double N = eddy_viscosity::compute_brunt_vaisala_frequency(state, grid, i, j, k);
                if (!std::isfinite(N) || N < 0.0)
                {
                    N = 0.0;
                }
                double l = compute_mixing_length(cfg, grid, e, N, i, j, k);
                l = std::max(l, 1.0e-3);

                double P_s = static_cast<double>(shear_prod[i][j][k]);

                double P_b = static_cast<double>(buoy_prod[i][j][k]);
                if (!std::isfinite(P_s)) P_s = 0.0;
                if (!std::isfinite(P_b)) P_b = 0.0;

                double eps = Ce2_ * std::pow(std::max(e, 1e-6), 1.5) / l;
                if (!std::isfinite(eps) || eps < 0.0) eps = 0.0;

                double diffusion = 0.0;
                if (k > 0 && k < state.NZ - 1)
                {
                    const double dz_local = std::max(grid_metric::local_dz(grid, i, j, k, state.NZ), 1.0);
                    double e_km1 = static_cast<double>(tke_[i][j][k - 1]);
                    double e_k = static_cast<double>(tke_[i][j][k]);
                    double e_kp1 = static_cast<double>(tke_[i][j][k + 1]);
                    if (!std::isfinite(e_km1)) e_km1 = 0.0;
                    if (!std::isfinite(e_k)) e_k = 0.0;
                    if (!std::isfinite(e_kp1)) e_kp1 = 0.0;
                    const double d2e_dz2 = (e_kp1 - 2.0 * e_k + e_km1) / (dz_local * dz_local);
                    diffusion = static_cast<double>(K_tke_[i][j][k]) * d2e_dz2;
                }
                if (!std::isfinite(diffusion)) diffusion = 0.0;

                double dtke = P_s + P_b - eps + diffusion;
                if (!std::isfinite(dtke))
                {
                    dtke = 0.0;
                }

                dtke_dt[i][j][k] = static_cast<float>(dtke);
            }
        }
    }
}

/**
 * @brief Computes the TKE production.
 */
void TKEScheme::compute_tke_production(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& nu_t,
    Field3D& shear_prod,
    Field3D& buoy_prod
) 
{
    for (int i = 0; i < state.NR; ++i) 
    {
        for (int j = 0; j < state.NTH; ++j) 
        {
            for (int k = 0; k < state.NZ; ++k) 
            {
                eddy_viscosity::StrainRate S = eddy_viscosity::compute_strain_rate_3d(
                    state, grid, i, j, k);
                double nu_local = static_cast<double>(nu_t[i][j][k]);
                if (!std::isfinite(nu_local) || nu_local < 0.0)
                {
                    nu_local = 0.0;
                }
                const double strain_mag = std::isfinite(S.magnitude) ? S.magnitude : 0.0;
                double P_s = 2.0 * nu_local * strain_mag * strain_mag;
                if (!std::isfinite(P_s))
                {
                    P_s = 0.0;
                }

                shear_prod[i][j][k] = static_cast<float>(P_s);


                if (k < state.NZ - 1) 
                {
                    double theta_k = static_cast<double>((*state.theta)[i][j][k]);
                    double theta_kp1 = static_cast<double>((*state.theta)[i][j][k+1]);
                    if (!std::isfinite(theta_k) || !std::isfinite(theta_kp1) || theta_k <= 1.0e-6)
                    {
                        buoy_prod[i][j][k] = 0.0f;
                        continue;
                    }
                    const double dz_local = std::max(grid_metric::local_dz(grid, i, j, k, state.NZ), 1.0);
                    double dtheta_dz = (theta_kp1 - theta_k) / dz_local;
                    const double Pr_safe = std::max(std::abs(Pr_t_), 1.0e-6);
                    double P_b = -nu_local / Pr_safe *
                                 (turbulence_constants::g / theta_k) * dtheta_dz;
                    if (!std::isfinite(P_b))
                    {
                        P_b = 0.0;
                    }
                    buoy_prod[i][j][k] = static_cast<float>(P_b);
                }
            }
        }
    }
}

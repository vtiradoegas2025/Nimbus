/**
 * @file radiation.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "core/simulation.hpp"
#include "physics/boundary_layer_base.hpp"
#include "physics/radiation_base.hpp"
#include "radiation/factory.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

namespace
{
constexpr double k_default_radiation_dt = 300.0;
constexpr double k_default_tau_lw_ref = 6.0;
constexpr double k_default_tau_sw_ref = 0.22;
constexpr double k_default_n_lw = 4.0;
constexpr double k_default_n_sw = 2.0;

double compute_idealized_mu0(double simulation_time_s)
{
    constexpr double pi = 3.14159265358979323846;
    constexpr double latitude_deg = 35.0;
    constexpr double declination_deg = 15.0;
    constexpr double seconds_per_day = 86400.0;

    const double latitude = latitude_deg * pi / 180.0;
    const double declination = declination_deg * pi / 180.0;
    const double t_local = std::fmod(std::max(0.0, simulation_time_s), seconds_per_day);
    const double day_fraction = t_local / seconds_per_day;
    const double hour_angle = 2.0 * pi * (day_fraction - 0.5);

    const double mu0 = std::sin(latitude) * std::sin(declination) +
                       std::cos(latitude) * std::cos(declination) * std::cos(hour_angle);
    return std::max(0.0, mu0);
}

}


std::unique_ptr<RadiationSchemeBase> radiation_scheme = nullptr;
RadiationConfig global_radiation_config;
double last_radiation_time = std::numeric_limits<double>::lowest();


/**
 * @brief Initializes the radiation scheme.
 */
void initialize_radiation(const std::string& scheme_name, const RadiationConfig& cfg) 
{
    try 
    {
        global_radiation_config = cfg;
        const std::string requested_scheme = scheme_name;
        std::string resolved_scheme = canonicalize_radiation_scheme_id(scheme_name);
        if (resolved_scheme.empty())
        {
            resolved_scheme = "simple_grey";
        }
        if (!is_radiation_scheme_implemented(resolved_scheme))
        {
            tmv::log_warn("[RADIATION CONFIG] Requested scheme '", requested_scheme,
                         "' maps to '", resolved_scheme,
                         "', which is known but not implemented in this build. ",
                         "Using default 'simple_grey'.");
            resolved_scheme = "simple_grey";
        }
        global_radiation_config.scheme_id = resolved_scheme;

        if (!std::isfinite(global_radiation_config.dt_radiation) || global_radiation_config.dt_radiation <= 0.0)
        {
            tmv::log_warn("[RADIATION CONFIG] Invalid dt_radiation=", global_radiation_config.dt_radiation, "; using default ", k_default_radiation_dt, " s");
            global_radiation_config.dt_radiation = k_default_radiation_dt;
        }
        if (!std::isfinite(global_radiation_config.tau_lw_ref) || global_radiation_config.tau_lw_ref < 0.0)
        {
            tmv::log_warn("[RADIATION CONFIG] Invalid tau_lw_ref=", global_radiation_config.tau_lw_ref, "; using default ", k_default_tau_lw_ref);
            global_radiation_config.tau_lw_ref = k_default_tau_lw_ref;
        }
        if (!std::isfinite(global_radiation_config.tau_sw_ref) || global_radiation_config.tau_sw_ref < 0.0)
        {
            tmv::log_warn("[RADIATION CONFIG] Invalid tau_sw_ref=", global_radiation_config.tau_sw_ref, "; using default ", k_default_tau_sw_ref);
            global_radiation_config.tau_sw_ref = k_default_tau_sw_ref;
        }
        if (!std::isfinite(global_radiation_config.n_lw) || global_radiation_config.n_lw <= 0.0)
        {
            tmv::log_warn("[RADIATION CONFIG] Invalid n_lw=", global_radiation_config.n_lw, "; using default ", k_default_n_lw);
            global_radiation_config.n_lw = k_default_n_lw;
        }
        if (!std::isfinite(global_radiation_config.n_sw) || global_radiation_config.n_sw <= 0.0)
        {
            tmv::log_warn("[RADIATION CONFIG] Invalid n_sw=", global_radiation_config.n_sw, "; using default ", k_default_n_sw);
            global_radiation_config.n_sw = k_default_n_sw;
        }

        radiation_scheme = create_radiation_scheme(global_radiation_config.scheme_id);
        radiation_scheme->initialize(global_radiation_config);
        tmv::log_info("Initialized radiation scheme: ", global_radiation_config.scheme_id);

        dtheta_dt_rad.resize(NR, NTH, NZ, 0.0f);
        last_radiation_time = -global_radiation_config.dt_radiation;

    } 
    catch (const std::exception& e) 
    {
        tmv::log_error("Error initializing radiation: ", e.what());
        throw;
    }
}

/**
 * @brief Steps the radiation forward in time.
 */
void step_radiation(double current_time) 
{
    if (!radiation_scheme) return;
    if (NZ <= 0) return;
    if (current_time - last_radiation_time < global_radiation_config.dt_radiation)
    {
        return;
    }
    last_radiation_time = current_time;

    const double mu0_idealized = compute_idealized_mu0(current_time);
    const double kappa = R_d / cp;

    std::vector<double> p_col(NZ);
    std::vector<double> T_col(NZ);
    std::vector<double> rho_col(NZ);
    std::vector<double> dz_col(NZ, dz);
    std::vector<double> qv_col;
    const bool has_qv = !qv.empty();

    if (has_qv)
    {
        qv_col.resize(NZ);
    }

    const double surface_albedo = std::clamp(global_surface_config.albedo, 0.0, 1.0);
    const double surface_emissivity = std::clamp(global_surface_config.emissivity, 0.0, 1.0);
    const bool use_config_tsfc = std::isfinite(global_surface_config.Tsfc) && global_surface_config.Tsfc >= 150.0 &&
        global_surface_config.Tsfc <= 400.0;

    int sanitized_nonfinite = 0;
    int tendency_shape_mismatch_columns = 0;

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            RadiationColumnStateView col;

            for (int k = 0; k < NZ; ++k) 
            {
                p_col[k] = static_cast<double>(p[i][j][k]);
            }
            col.p = &p_col;

            for (int k = 0; k < NZ; ++k) 
            {
                double theta_val = theta[i][j][k];
                double p_val = p[i][j][k];
                
                if (theta_val < 200.0 || theta_val > 500.0 || p_val < 10000.0 || p_val > 120000.0) 
                {
                    double z = k * dz;
                    T_col[k] = theta0 - 0.0065 * z;

                    if (i == 0 && j == 0 && k < 3)
                    {
                        tmv::log_warn("[RADIATION WARNING] Invalid theta/p at i=", i, ",j=", j, ",k=", k,
                                      ": theta=", theta_val, "K, p=", p_val, "Pa, using default T=", T_col[k], "K");
                    }
                } 
                else 
                {
                    T_col[k] = theta_val * pow(p_val / p0, kappa);
                }
            }
            col.T = &T_col;

            for (int k = 0; k < NZ; ++k) 
            {
                rho_col[k] = static_cast<double>(rho[i][j][k]);
            }
            col.rho = &rho_col;
            col.dz = &dz_col;

            col.mu0 = mu0_idealized;
            col.S0 = S0_solar;

            col.Tsfc = use_config_tsfc ? global_surface_config.Tsfc : T_col[0];
            col.albedo_sw = surface_albedo;
            col.emissivity_lw = surface_emissivity;

            if (has_qv) 
            {
                for (int k = 0; k < NZ; ++k) 
                {
                    qv_col[k] = static_cast<double>(qv[i][j][k]);
                }
                col.qv = &qv_col;
            }
            else
            {
                col.qv = nullptr;
            }

            RadiationColumnTendencies tend;
            radiation_scheme->compute_column(global_radiation_config, col, tend);
            
            if (tend.dTdt_rad.size() != static_cast<size_t>(NZ))
            {
                ++tendency_shape_mismatch_columns;
            }

            for (int k = 0; k < NZ; ++k) 
            {
                const size_t k_idx = static_cast<size_t>(k);
                double dTdt_val = 0.0;
                if (k_idx < tend.dTdt_rad.size())
                {
                    dTdt_val = tend.dTdt_rad[k_idx];
                }
                else
                {
                    ++sanitized_nonfinite;
                }

                if (!std::isfinite(dTdt_val))
                {
                    dTdt_val = 0.0;
                    ++sanitized_nonfinite;
                }

                double p_safe = static_cast<double>(p[i][j][k]);
                if (!std::isfinite(p_safe) || p_safe <= 0.0)
                {
                    p_safe = p0;
                    ++sanitized_nonfinite;
                }
                const double theta_scale = std::pow(p0 / p_safe, kappa);
                double dtheta_dt_val = dTdt_val * theta_scale;
                if (!std::isfinite(dtheta_dt_val))
                {
                    dtheta_dt_val = 0.0;
                    ++sanitized_nonfinite;
                }
                dtheta_dt_rad[i][j][k] = static_cast<float>(dtheta_dt_val);
            }
        }
    }

    if (sanitized_nonfinite > 0)
    {
        tmv::log_warn("[RADIATION GUARD] sanitized non-finite tendencies: ", sanitized_nonfinite);
    }
    if (tendency_shape_mismatch_columns > 0)
    {
        tmv::log_warn("[RADIATION GUARD] columns with malformed tendency sizes: ", tendency_shape_mismatch_columns);
    }
}

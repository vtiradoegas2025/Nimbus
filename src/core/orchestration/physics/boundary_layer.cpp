/**
 * @file boundary_layer.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "core/simulation.hpp"
#include "boundary_layer/factory.hpp"
#include "util/log.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>


std::unique_ptr<BoundaryLayerSchemeBase> boundary_layer_scheme = nullptr;
BoundaryLayerConfig global_boundary_layer_config;
SurfaceConfig global_surface_config;
double last_boundary_layer_time = std::numeric_limits<double>::lowest();
bool g_boundary_layer_updated_this_step = false;

namespace
{
constexpr double k_default_pbl_dt = 60.0;
constexpr double k_default_pbl_max_height = 2000.0;
constexpr double k_default_min_ustar = 1.0e-4;
}


/**
 * @brief Initializes the boundary layer scheme.
 */
void initialize_boundary_layer(const std::string& scheme_name,const BoundaryLayerConfig& cfg, const SurfaceConfig& sfc) 
{
    try 
    {
        global_boundary_layer_config = cfg;
        global_boundary_layer_config.scheme_id = scheme_name;
        global_surface_config = sfc;

        if (!std::isfinite(global_boundary_layer_config.dt_pbl) ||
            global_boundary_layer_config.dt_pbl <= 0.0)
        {
            tmv::log_warn("[PBL CONFIG] Invalid dt_pbl=", global_boundary_layer_config.dt_pbl,
                         "; using default ", k_default_pbl_dt, " s");
            global_boundary_layer_config.dt_pbl = k_default_pbl_dt;
        }
        if (!std::isfinite(global_boundary_layer_config.pbl_max_height) ||
            global_boundary_layer_config.pbl_max_height <= 0.0)
        {
            tmv::log_warn("[PBL CONFIG] Invalid pbl_max_height=", global_boundary_layer_config.pbl_max_height,
                         "; using default ", k_default_pbl_max_height, " m");
            global_boundary_layer_config.pbl_max_height = k_default_pbl_max_height;
        }
        if (!std::isfinite(global_boundary_layer_config.min_ustar) ||
            global_boundary_layer_config.min_ustar < 0.0)
        {
            tmv::log_warn("[PBL CONFIG] Invalid min_ustar=", global_boundary_layer_config.min_ustar,
                         "; using default ", k_default_min_ustar, " m/s");
            global_boundary_layer_config.min_ustar = k_default_min_ustar;
        }
        boundary_layer_scheme = create_boundary_layer_scheme(scheme_name);
        boundary_layer_scheme->initialize(global_boundary_layer_config);

        dtheta_dt_pbl.resize(NR, NTH, NZ, 0.0f);
        dqv_dt_pbl.resize(NR, NTH, NZ, 0.0f);
        du_dt_pbl.resize(NR, NTH, NZ, 0.0f);
        dv_dt_pbl.resize(NR, NTH, NZ, 0.0f);
        dtke_dt_pbl.resize(NR, NTH, NZ, 0.0f);
        last_boundary_layer_time = -global_boundary_layer_config.dt_pbl;
        g_boundary_layer_updated_this_step = false;

        tmv::log_info("Initialized boundary layer scheme: ", scheme_name);
        tmv::log_info("  PBL cadence: ", global_boundary_layer_config.dt_pbl, " s");
        tmv::log_info("  Surface fluxes: ", (global_boundary_layer_config.apply_surface_fluxes ? "enabled" : "disabled"));

    } 
    catch (const std::exception& e) 
    {
        tmv::log_error("Error initializing boundary layer: ", e.what());
        throw;
    }
}

/**
 * @brief Steps the boundary layer forward in time.
 */
void step_boundary_layer(double current_time) 
{
    g_boundary_layer_updated_this_step = false;
    if (!boundary_layer_scheme) return;

    if (current_time - last_boundary_layer_time < global_boundary_layer_config.dt_pbl) 
    {
        return;
    }

    last_boundary_layer_time = current_time;
    int sanitized_nonfinite = 0;
    int tendency_shape_mismatch_columns = 0;
    std::vector<double> p_col(NZ);
    std::vector<double> theta_col(NZ);
    std::vector<double> qv_col(NZ);
    std::vector<double> u_col(NZ);
    std::vector<double> v_col(NZ);
    std::vector<double> rho_col(NZ);
    std::vector<double> z_int_col(NZ + 1);
    std::vector<double> tke_col;
    if (!tke.empty())
    {
        tke_col.resize(NZ);
    }

    z_int_col[0] = 0.0;
    for (int k = 1; k <= NZ; ++k)
    {
        z_int_col[k] = z_int_col[k - 1] + dz;
    }

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            BoundaryLayerColumnStateView col;

            for (int k = 0; k < NZ; ++k) 
            {
                p_col[k] = static_cast<double>(p[i][j][k]);
            }
            col.p = &p_col;

            for (int k = 0; k < NZ; ++k) 
            {
                theta_col[k] = static_cast<double>(theta[i][j][k]);
            }
            col.theta = &theta_col;

            for (int k = 0; k < NZ; ++k) 
            {
                qv_col[k] = static_cast<double>(qv[i][j][k]);
            }
            col.qv = &qv_col;

            for (int k = 0; k < NZ; ++k) 
            {
                u_col[k] = static_cast<double>(u[i][j][k]);
                v_col[k] = static_cast<double>(v[i][j][k]);
            }
            col.u = &u_col;
            col.v = &v_col;

            for (int k = 0; k < NZ; ++k) 
            {
                rho_col[k] = static_cast<double>(rho[i][j][k]);
            }
            col.rho = &rho_col;

            col.z_int = &z_int_col;

            col.u_sfc = static_cast<double>(u[i][j][0]);
            col.v_sfc = static_cast<double>(v[i][j][0]);
            col.theta_sfc = static_cast<double>(theta[i][j][0]);
            col.qv_sfc = static_cast<double>(qv[i][j][0]);
            col.z_sfc = dz * 0.5;


            if (!tke.empty()) 
            {
                for (int k = 0; k < NZ; ++k) 
                {
                    tke_col[k] = static_cast<double>(tke[i][j][k]);
                }
                col.tke = &tke_col;
            }
            else
            {
                col.tke = nullptr;
            }

            BoundaryLayerTendencies tend;
            boundary_layer_scheme->compute_column(global_boundary_layer_config, global_surface_config, col, tend);

            const bool tendency_shapes_ok =
                tend.dthetadt_pbl.size() == static_cast<size_t>(NZ) &&
                tend.dqvdt_pbl.size() == static_cast<size_t>(NZ) &&
                tend.dudt_pbl.size() == static_cast<size_t>(NZ) &&
                tend.dvdt_pbl.size() == static_cast<size_t>(NZ) &&
                tend.dtkedt_pbl.size() == static_cast<size_t>(NZ);
            if (!tendency_shapes_ok)
            {
                ++tendency_shape_mismatch_columns;
            }

            for (int k = 0; k < NZ; ++k) 
            {
                const size_t k_idx = static_cast<size_t>(k);
                auto tendency_or_zero = [&](const std::vector<double>& values) -> float
                {
                    if (k_idx >= values.size())
                    {
                        ++sanitized_nonfinite;
                        return 0.0f;
                    }
                    return static_cast<float>(values[k_idx]);
                };

                float dtheta_val = tendency_or_zero(tend.dthetadt_pbl);
                float dqv_val = tendency_or_zero(tend.dqvdt_pbl);
                float du_val = tendency_or_zero(tend.dudt_pbl);
                float dv_val = tendency_or_zero(tend.dvdt_pbl);
                float dtke_val = tendency_or_zero(tend.dtkedt_pbl);

                if (!std::isfinite(dtheta_val)) { dtheta_val = 0.0f; ++sanitized_nonfinite; }
                if (!std::isfinite(dqv_val)) { dqv_val = 0.0f; ++sanitized_nonfinite; }
                if (!std::isfinite(du_val)) { du_val = 0.0f; ++sanitized_nonfinite; }
                if (!std::isfinite(dv_val)) { dv_val = 0.0f; ++sanitized_nonfinite; }
                if (!std::isfinite(dtke_val)) { dtke_val = 0.0f; ++sanitized_nonfinite; }

                dtheta_dt_pbl[i][j][k] = dtheta_val;
                dqv_dt_pbl[i][j][k] = dqv_val;
                du_dt_pbl[i][j][k] = du_val;
                dv_dt_pbl[i][j][k] = dv_val;
                dtke_dt_pbl[i][j][k] = dtke_val;
            }
        }
    }

    if (sanitized_nonfinite > 0)
    {
        tmv::log_warn("[PBL GUARD] sanitized non-finite tendencies: ", sanitized_nonfinite);
    }
    if (tendency_shape_mismatch_columns > 0)
    {
        tmv::log_warn("[PBL GUARD] columns with malformed tendency sizes: ",
                     tendency_shape_mismatch_columns);
    }
    g_boundary_layer_updated_this_step = true;
}

/**
 * @brief Reports whether boundary-layer tendencies were updated this step.
 * @return True if `step_boundary_layer` performed an update.
 */
bool boundary_layer_updated_this_step()
{
    return g_boundary_layer_updated_this_step;
}

/**
 * @file turbulence.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "core/simulation.hpp"
#include "turbulence/factory.hpp"
#include "turbulence/base/eddy_viscosity.hpp"
#include "util/simd_utils.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>

std::unique_ptr<TurbulenceSchemeBase> turbulence_scheme = nullptr;
TurbulenceConfig global_turbulence_config;

TurbulenceTendencies turbulence_tendencies;

namespace
{
double last_turbulence_time = std::numeric_limits<double>::lowest();
GridMetrics fallback_turbulence_grid;
int fallback_grid_nr = -1;
int fallback_grid_nth = -1;
int fallback_grid_nz = -1;
double fallback_grid_dr = std::numeric_limits<double>::quiet_NaN();
double fallback_grid_dtheta = std::numeric_limits<double>::quiet_NaN();
double fallback_grid_dz = std::numeric_limits<double>::quiet_NaN();

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string normalize_scheme_name(std::string value)
{
    value = lower_copy(value);
    if (value == "smag" || value == "smagorinsky_lilly" || value == "smagorinsky-lilly")
    {
        return "smagorinsky";
    }
    if (value == "1.5" || value == "1.5order" || value == "1.5-order")
    {
        return "tke";
    }
    return value;
}

bool valid_turbulence_mode(const std::string& mode)
{
    return mode == "3d" || mode == "horizontal_only" || mode == "vertical_only";
}

bool valid_filter_width(const std::string& value)
{
    return value == "dx" || value == "cubic_root" || value == "user";
}

bool valid_stability_correction(const std::string& value)
{
    return value == "none" || value == "ri" || value == "tke_based";
}

inline bool field_matches_domain(const Field3D& f)
{
    return f.size_r() == NR && f.size_th() == NTH && f.size_z() == NZ;
}

void ensure_tendency_shape(TurbulenceTendencies& tendencies)
{
    if (!field_matches_domain(tendencies.dudt_sgs)) tendencies.dudt_sgs.resize(NR, NTH, NZ, 0.0f);
    if (!field_matches_domain(tendencies.dvdt_sgs)) tendencies.dvdt_sgs.resize(NR, NTH, NZ, 0.0f);
    if (!field_matches_domain(tendencies.dwdt_sgs)) tendencies.dwdt_sgs.resize(NR, NTH, NZ, 0.0f);
    if (!field_matches_domain(tendencies.dthetadt_sgs)) tendencies.dthetadt_sgs.resize(NR, NTH, NZ, 0.0f);
    if (!field_matches_domain(tendencies.dqvdt_sgs)) tendencies.dqvdt_sgs.resize(NR, NTH, NZ, 0.0f);
    if (!field_matches_domain(tendencies.dtkedt_sgs)) tendencies.dtkedt_sgs.resize(NR, NTH, NZ, 0.0f);
}

void clear_tendencies(TurbulenceTendencies& tendencies)
{
    tendencies.dudt_sgs.fill(0.0f);
    tendencies.dvdt_sgs.fill(0.0f);
    tendencies.dwdt_sgs.fill(0.0f);
    tendencies.dthetadt_sgs.fill(0.0f);
    tendencies.dqvdt_sgs.fill(0.0f);
    tendencies.dtkedt_sgs.fill(0.0f);
}

const GridMetrics& get_runtime_turbulence_grid()
{
    const bool global_grid_ready =
        std::isfinite(global_grid_metrics.dx) &&
        std::isfinite(global_grid_metrics.dy) &&
        global_grid_metrics.dx > 0.0 &&
        global_grid_metrics.dy > 0.0 &&
        static_cast<int>(global_grid_metrics.dz.size()) == NZ;

    if (global_grid_ready)
    {
        return global_grid_metrics;
    }

    const bool reinitialize =
        fallback_grid_nr != NR ||
        fallback_grid_nth != NTH ||
        fallback_grid_nz != NZ ||
        fallback_grid_dr != dr ||
        fallback_grid_dtheta != dtheta ||
        fallback_grid_dz != dz;

    if (reinitialize)
    {
        fallback_grid_nr = NR;
        fallback_grid_nth = NTH;
        fallback_grid_nz = NZ;
        fallback_grid_dr = dr;
        fallback_grid_dtheta = dtheta;
        fallback_grid_dz = dz;

        fallback_turbulence_grid.dx = dr;
        fallback_turbulence_grid.dy = std::max(dr, dtheta * dr);
        fallback_turbulence_grid.dz.assign(static_cast<std::size_t>(NZ), dz);
        fallback_turbulence_grid.z_int.resize(static_cast<std::size_t>(NZ + 1));
        fallback_turbulence_grid.z_int[0] = 0.0;
        for (int k = 1; k <= NZ; ++k)
        {
            fallback_turbulence_grid.z_int[static_cast<std::size_t>(k)] =
                fallback_turbulence_grid.z_int[static_cast<std::size_t>(k - 1)] + dz;
        }
        fallback_turbulence_grid.terrain_metrics_active = false;
        fallback_turbulence_grid.terrain_metrics = nullptr;
        fallback_turbulence_grid.terrain_topography = nullptr;
    }

    return fallback_turbulence_grid;
}

int sanitize_nonfinite_tendency(Field3D& field)
{
    float* const data = field.data();
    const int count = static_cast<int>(field.size());
    return simd_utils::sanitize_nonfinite(data, 0.0f, data, count);
}
}

/**
 * @brief Initializes the turbulence scheme and runtime configuration.
 * @param scheme_name Requested turbulence scheme identifier.
 * @param cfg Runtime turbulence configuration.
 */
void initialize_turbulence(const std::string& scheme_name,
                          const TurbulenceConfig& cfg) {
    try {
        global_turbulence_config = cfg;
        std::string requested_scheme = scheme_name.empty() ? cfg.scheme_id : scheme_name;
        requested_scheme = normalize_scheme_name(requested_scheme);
        if (requested_scheme.empty())
        {
            requested_scheme = "smagorinsky";
        }
        global_turbulence_config.scheme_id = requested_scheme;

        if (!std::isfinite(global_turbulence_config.dt_sgs) ||
            global_turbulence_config.dt_sgs <= 0.0)
        {
            tmv::log_warn("[TURBULENCE CONFIG] Invalid dt_sgs=", global_turbulence_config.dt_sgs, "; using 1.0 s");
            global_turbulence_config.dt_sgs = 1.0;
        }
        if (!std::isfinite(global_turbulence_config.Cs) || global_turbulence_config.Cs < 0.0)
        {
            tmv::log_warn("[TURBULENCE CONFIG] Invalid Cs=", global_turbulence_config.Cs, "; using default ", turbulence_constants::C_s_default);
            global_turbulence_config.Cs = turbulence_constants::C_s_default;
        }
        if (!std::isfinite(global_turbulence_config.Pr_t) || global_turbulence_config.Pr_t <= 0.0)
        {
            tmv::log_warn("[TURBULENCE CONFIG] Invalid Pr_t=", global_turbulence_config.Pr_t, "; using default ", turbulence_constants::Pr_t_default);
            global_turbulence_config.Pr_t = turbulence_constants::Pr_t_default;
        }
        if (!std::isfinite(global_turbulence_config.Sc_t) || global_turbulence_config.Sc_t <= 0.0)
        {
            tmv::log_warn("[TURBULENCE CONFIG] Invalid Sc_t=", global_turbulence_config.Sc_t, "; using default ", turbulence_constants::Sc_t_default);
            global_turbulence_config.Sc_t = turbulence_constants::Sc_t_default;
        }
        if (!std::isfinite(global_turbulence_config.nu_t_max) || global_turbulence_config.nu_t_max < 0.0)
        {
            tmv::log_warn("[TURBULENCE CONFIG] Invalid nu_t_max=", global_turbulence_config.nu_t_max, "; using 1000.0");
            global_turbulence_config.nu_t_max = 1000.0;
        }
        if (!std::isfinite(global_turbulence_config.K_max) || global_turbulence_config.K_max < 0.0)
        {
            tmv::log_warn("[TURBULENCE CONFIG] Invalid K_max=", global_turbulence_config.K_max, "; using 1000.0");
            global_turbulence_config.K_max = 1000.0;
        }

        global_turbulence_config.mode = lower_copy(global_turbulence_config.mode);
        if (!valid_turbulence_mode(global_turbulence_config.mode))
        {
            tmv::log_warn("[TURBULENCE CONFIG] Invalid mode='", global_turbulence_config.mode, "'; using '3d'");
            global_turbulence_config.mode = "3d";
        }
        global_turbulence_config.filter_width = lower_copy(global_turbulence_config.filter_width);
        if (!valid_filter_width(global_turbulence_config.filter_width))
        {
            tmv::log_warn("[TURBULENCE CONFIG] Invalid filter_width='", global_turbulence_config.filter_width, "'; using 'cubic_root'");
            global_turbulence_config.filter_width = "cubic_root";
        }
        global_turbulence_config.stability_correction =
            lower_copy(global_turbulence_config.stability_correction);
        if (!valid_stability_correction(global_turbulence_config.stability_correction))
        {
            tmv::log_warn("[TURBULENCE CONFIG] Invalid stability_correction='", global_turbulence_config.stability_correction, "'; using 'none'");
            global_turbulence_config.stability_correction = "none";
        }

        turbulence_scheme = create_turbulence_scheme(global_turbulence_config.scheme_id);
        turbulence_scheme->initialize(global_turbulence_config);

        tmv::log_info("Initialized turbulence scheme: ", global_turbulence_config.scheme_id);
        tmv::log_info("  SGS cadence: ", global_turbulence_config.dt_sgs, " s");
        tmv::log_info("  Cs = ", global_turbulence_config.Cs);
        last_turbulence_time = -global_turbulence_config.dt_sgs;
        ensure_tendency_shape(turbulence_tendencies);

    } catch (const std::exception& e) {
        tmv::log_error("Error initializing turbulence: ", e.what());
        throw;
    }
}

/**
 * @brief Computes turbulence tendencies at configured cadence.
 * @param current_time Current simulation time in seconds.
 * @param tendencies Output tendency container.
 */
void step_turbulence(double current_time, TurbulenceTendencies& tendencies) 
{
    if (!turbulence_scheme) 
    {
        ensure_tendency_shape(tendencies);
        clear_tendencies(tendencies);
        return;
    }

    if (current_time - last_turbulence_time < global_turbulence_config.dt_sgs) 
    {
        return;
    }

    last_turbulence_time = current_time;

    TurbulenceStateView state;
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.rho = &rho;
    state.theta = &theta;
    state.qv = &qv;
    state.tke = &tke;
    state.NR = NR;
    state.NTH = NTH;
    state.NZ = NZ;

    const GridMetrics& grid = get_runtime_turbulence_grid();
    ensure_tendency_shape(tendencies);
    clear_tendencies(tendencies);

    turbulence_scheme->compute(global_turbulence_config, grid, state, tendencies);

    const int sanitized_nonfinite =
        sanitize_nonfinite_tendency(tendencies.dudt_sgs) +
        sanitize_nonfinite_tendency(tendencies.dvdt_sgs) +
        sanitize_nonfinite_tendency(tendencies.dwdt_sgs) +
        sanitize_nonfinite_tendency(tendencies.dthetadt_sgs) +
        sanitize_nonfinite_tendency(tendencies.dqvdt_sgs) +
        sanitize_nonfinite_tendency(tendencies.dtkedt_sgs);

    if (sanitized_nonfinite > 0)
    {
        tmv::log_warn("[TURBULENCE GUARD] sanitized non-finite tendencies: ", sanitized_nonfinite);
    }
}

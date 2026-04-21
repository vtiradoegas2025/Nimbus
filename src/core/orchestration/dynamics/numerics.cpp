/**
 * @file numerics.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "core/simulation.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/advection/factory.hpp"
#include "numerics/diffusion/factory.hpp"
#include "numerics/time_stepping/factory.hpp"
#include "terrain/terrain_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "turbulence/turbulence_base.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <iostream>
#include <limits>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

double sanitize_positive(double value, double fallback)
{
    if (!std::isfinite(value) || value <= 0.0)
    {
        return fallback;
    }
    return value;
}

double compute_max_flow_speed()
{
    if (u.size_r() != NR || u.size_th() != NTH || u.size_z() != NZ ||
        v.size_r() != NR || v.size_th() != NTH || v.size_z() != NZ ||
        w.size_r() != NR || w.size_th() != NTH || w.size_z() != NZ)
    {
        return 0.0;
    }

    double max_speed = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:max_speed)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double speed =
                    std::max({std::abs(static_cast<double>(u[i][j][k])),
                              std::abs(static_cast<double>(v[i][j][k])),
                              std::abs(static_cast<double>(w[i][j][k]))});
                max_speed = std::max(max_speed, speed);
            }
        }
    }
    return max_speed;
}

bool terrain_metrics_ready_for_grid()
{
    return global_terrain_metrics.z.size_r() == NR &&
           global_terrain_metrics.z.size_th() == NTH &&
           global_terrain_metrics.z.size_z() == NZ &&
           global_terrain_metrics.J.size_r() == NR &&
           global_terrain_metrics.J.size_th() == NTH &&
           global_terrain_metrics.J.size_z() == NZ;
}
}

std::unique_ptr<AdvectionSchemeBase> advection_scheme;
AdvectionConfig global_advection_config;
std::unique_ptr<DiffusionSchemeBase> diffusion_scheme;
DiffusionConfig global_diffusion_config;
std::unique_ptr<TimeSteppingSchemeBase> time_stepping_scheme;
TimeSteppingConfig global_time_stepping_config;

GridMetrics global_grid_metrics;

/**
 * @brief Refreshes runtime grid metrics from current terrain and spacing.
 */
void refresh_grid_metrics_from_terrain()
{
    const double dx_safe = sanitize_positive(dr, 1.0);
    const double dtheta_safe = (std::isfinite(dtheta) && dtheta > 0.0) ? dtheta : 0.0;
    const double dy_safe = std::max(1.0e-6, (dtheta_safe > 0.0) ? (dtheta_safe * dx_safe) : dx_safe);
    const double dz_safe = sanitize_positive(dz, 1.0);

    global_grid_metrics.dx = dx_safe;
    global_grid_metrics.dy = dy_safe;
    global_grid_metrics.terrain_metrics_active = false;
    global_grid_metrics.terrain_metrics = nullptr;
    global_grid_metrics.terrain_topography = nullptr;

    if (NZ <= 0)
    {
        global_grid_metrics.dz.clear();
        global_grid_metrics.z_int.clear();
        return;
    }

    global_grid_metrics.dz.assign(NZ, dz_safe);
    global_grid_metrics.z_int.assign(NZ + 1, 0.0);

    // For flat terrain (scheme "none"), the metrics arrays are identity
    // (J=1.0, z=k*dz) and don't need to be activated. Keeping them
    // inactive enables GPU compute dispatch for dynamics kernels, which
    // assume uniform grid spacing. Only activate for real terrain schemes
    // (bell, schar, etc.) where metric corrections are needed.
    const bool flat_terrain = (global_terrain_config.scheme_id == "none");

    if (!terrain_metrics_ready_for_grid() || flat_terrain)
    {
        for (int k = 1; k <= NZ; ++k)
        {
            global_grid_metrics.z_int[k] = global_grid_metrics.z_int[k - 1] + dz_safe;
        }
        if (flat_terrain)
        {
            // Still set z_int from the identity metrics for consistency,
            // but do NOT activate terrain_metrics_active.
            global_grid_metrics.terrain_metrics = &global_terrain_metrics;
            global_grid_metrics.terrain_topography = &global_topography;
        }
        return;
    }

    global_grid_metrics.terrain_metrics_active = true;
    global_grid_metrics.terrain_metrics = &global_terrain_metrics;
    global_grid_metrics.terrain_topography = &global_topography;

    for (int k = 0; k < NZ; ++k)
    {
        double min_dz = std::numeric_limits<double>::infinity();

        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                double dz_local = dz_safe;
                if (NZ > 1)
                {
                    if (k == 0)
                    {
                        dz_local = std::abs(global_terrain_metrics.z(i, j, 1) - global_terrain_metrics.z(i, j, 0));
                    }
                    else if (k == NZ - 1)
                    {
                        dz_local = std::abs(global_terrain_metrics.z(i, j, NZ - 1) - global_terrain_metrics.z(i, j, NZ - 2));
                    }
                    else
                    {
                        const double dz_up = std::abs(global_terrain_metrics.z(i, j, k) - global_terrain_metrics.z(i, j, k - 1));
                        const double dz_down = std::abs(global_terrain_metrics.z(i, j, k + 1) - global_terrain_metrics.z(i, j, k));
                        dz_local = 0.5 * (dz_up + dz_down);
                    }
                }

                dz_local = sanitize_positive(dz_local, dz_safe);
                min_dz = std::min(min_dz, dz_local);
            }
        }

        global_grid_metrics.dz[k] = std::isfinite(min_dz) ? sanitize_positive(min_dz, dz_safe) : dz_safe;
    }

    global_grid_metrics.z_int[0] = 0.0;
    for (int k = 1; k <= NZ; ++k)
    {
        const double dz_layer = sanitize_positive(global_grid_metrics.dz[k - 1], dz_safe);
        global_grid_metrics.z_int[k] = global_grid_metrics.z_int[k - 1] + dz_layer;
    }
}

/**
 * @brief Initializes advection, diffusion, and time-stepping subsystems.
 */
void initialize_numerics()
{
    refresh_grid_metrics_from_terrain();

    if (global_advection_config.scheme_id.empty())
    {
        global_advection_config.scheme_id = "tvd";
    }
    global_advection_config.scheme_id = lower_copy(global_advection_config.scheme_id);
    advection_scheme = create_advection_scheme(global_advection_config.scheme_id);
    advection_scheme->initialize(global_advection_config);

    if (global_diffusion_config.scheme_id.empty())
    {
        global_diffusion_config.scheme_id = "explicit";
    }
    global_diffusion_config.scheme_id = lower_copy(global_diffusion_config.scheme_id);
    diffusion_scheme = create_diffusion_scheme(global_diffusion_config.scheme_id);
    diffusion_scheme->initialize(global_diffusion_config);

    if (global_time_stepping_config.scheme_id.empty())
    {
        global_time_stepping_config.scheme_id = "rk3";
    }
    global_time_stepping_config.scheme_id = lower_copy(global_time_stepping_config.scheme_id);
    time_stepping_scheme = create_time_stepping_scheme(global_time_stepping_config.scheme_id);
    time_stepping_scheme->initialize(global_time_stepping_config, nullptr);

    tmv::log_info("Initialized numerics framework:");
    tmv::log_info("  Advection: ", advection_scheme->name());
    tmv::log_info("  Diffusion: ", diffusion_scheme->name());
    tmv::log_info("  Time stepping: ", time_stepping_scheme->name());
}

/**
 * @brief Chooses a stable runtime timestep from CFL and diffusion limits.
 * @return Recommended timestep in seconds.
 */
double choose_runtime_timestep()
{
    const double dt_current = sanitize_positive(dt, 1.0e-3);
    const double dt_min_cfg = sanitize_positive(global_time_stepping_config.dt_min, 1.0e-6);
    const double dt_max_cfg = sanitize_positive(global_time_stepping_config.dt_max, std::max(dt_current, 1.0));
    const double dt_min = std::min(dt_min_cfg, dt_max_cfg);
    const double dt_max = std::max(dt_min_cfg, dt_max_cfg);

    double dt_limited = dt_current;
    if (dt_limited < dt_min)
    {
        dt_limited = dt_min;
    }
    if (dt_limited > dt_max)
    {
        dt_limited = dt_max;
    }

    const double max_speed = compute_max_flow_speed();
    const double dx_safe = sanitize_positive(global_grid_metrics.dx, sanitize_positive(dr, 1.0));
    const double dy_safe = sanitize_positive(global_grid_metrics.dy, dx_safe);
    double min_dz_safe = sanitize_positive(dz, 1.0);
    if (!global_grid_metrics.dz.empty())
    {
        for (double dz_level : global_grid_metrics.dz)
        {
            min_dz_safe = std::min(min_dz_safe, sanitize_positive(dz_level, min_dz_safe));
        }
    }
    const double min_spacing = std::max(1.0e-6, std::min({dx_safe, dy_safe, min_dz_safe}));

    double scheme_cfl = 1.0;
    if (time_stepping_scheme)
    {
        TimeSteppingState ts_state{};
        ts_state.dt = dt_current;
        ts_state.time = simulation_time;
        const double suggested = time_stepping_scheme->suggest_dt(global_time_stepping_config, ts_state);
        if (std::isfinite(suggested) && suggested > 0.0)
        {
            scheme_cfl = suggested;
        }
    }
    const double cfl_safety = sanitize_positive(global_time_stepping_config.cfl_safety, numerics_constants::cfl_target);

    double dt_advection_cap = std::numeric_limits<double>::infinity();
    if (max_speed > 1.0e-9)
    {
        dt_advection_cap = cfl_safety * scheme_cfl * min_spacing / max_speed;
    }

    double dt_diffusion_cap = std::numeric_limits<double>::infinity();
    if (diffusion_scheme)
    {
        const std::string diffusion_id = lower_copy(diffusion_scheme->name());
        if (diffusion_id == "explicit")
        {
            const double K_h = std::max(0.0, global_diffusion_config.K_h);
            const double K_v = std::max(0.0, global_diffusion_config.K_v);
            if (K_h > 0.0 || K_v > 0.0)
            {
                const double inv_dx2 = 1.0 / std::max(dx_safe * dx_safe, 1.0e-12);
                const double inv_dy2 = 1.0 / std::max(dy_safe * dy_safe, 1.0e-12);
                const double inv_dz2 = 1.0 / std::max(min_dz_safe * min_dz_safe, 1.0e-12);
                const double denom = 2.0 * (K_h * (inv_dx2 + inv_dy2) + K_v * inv_dz2);
                if (denom > 0.0)
                {
                    dt_diffusion_cap = 0.9 / denom;
                }
            }
        }
    }

    // Acoustic CFL diagnostic.
    // Estimate sound speed from base-state surface values and warn if dt
    // exceeds the acoustic stability limit without split-explicit enabled.
    double dt_acoustic_cap = std::numeric_limits<double>::infinity();
    if (!p0_base.empty() && !rho0_base.empty() && rho0_base[0] > 0.0)
    {
        const double c_s = std::sqrt(gamma * p0_base[0] / rho0_base[0]);
        if (c_s > 0.0)
        {
            dt_acoustic_cap = cfl_safety * min_dz_safe / c_s;
        }
    }

    const double dt_physical_cap = std::min(dt_advection_cap, dt_diffusion_cap);
    if (std::isfinite(dt_physical_cap) && dt_physical_cap > 0.0)
    {
        if (global_time_stepping_config.adaptive_dt)
        {
            if (dt_physical_cap < dt_min)
            {
                dt_limited = dt_physical_cap;
            }
            else
            {
                dt_limited = std::min(dt_max, std::max(dt_min, dt_physical_cap));
            }
        }
        else
        {
            dt_limited = std::min(dt_limited, dt_physical_cap);
            if (dt_limited < dt_min && dt_physical_cap < dt_min)
            {
                dt_limited = dt_physical_cap;
            }
        }
    }

    if (!std::isfinite(dt_limited) || dt_limited <= 0.0)
    {
        return dt_current;
    }

    if (std::isfinite(dt_advection_cap) || std::isfinite(dt_diffusion_cap))
    {
        tmv::log_debug("[NUMERICS] timestep caps: current=", dt_current,
                       " limited=", dt_limited,
                       " advection_cap=", dt_advection_cap,
                       " diffusion_cap=", dt_diffusion_cap,
                       " acoustic_cap=", dt_acoustic_cap);
    }

    if (!global_time_stepping_config.split_acoustic
        && std::isfinite(dt_acoustic_cap) && dt_limited > dt_acoustic_cap)
    {
        tmv::log_warn("[NUMERICS] dt=", dt_limited,
                      " exceeds acoustic CFL limit=", dt_acoustic_cap,
                      " (c_s*dt/dz > 1). Enable split_acoustic=true or reduce dt.");
    }

    return dt_limited;
}

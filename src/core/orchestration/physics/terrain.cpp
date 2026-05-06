/**
 * @file terrain.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "core/runtime/simulation.hpp"
#include "terrain/factory.hpp"
#include "terrain/base/topography.hpp"
#include "util/string_utils.hpp"
#include "util/log.hpp"
#include <iostream>
#include <cmath>


namespace
{
std::string normalize_terrain_scheme_id(const std::string& value)
{
    const std::string normalized = tmv::strutil::trim_and_lower(value);
    if (normalized.empty() || normalized == "none" || normalized == "flat")
    {
        return "none";
    }
    if (normalized == "bell")
    {
        return "bell";
    }
    if (normalized == "schar" || normalized == "schaer")
    {
        return "schar";
    }
    return "";
}

std::string normalize_terrain_coord_id(const std::string& value)
{
    const std::string normalized = tmv::strutil::trim_and_lower(value);
    if (normalized.empty() || normalized == "btf" || normalized == "terrain_following" ||
        normalized == "terrain-following" || normalized == "sigma")
    {
        return "btf";
    }
    if (normalized == "smoothed" || normalized == "smooth")
    {
        return "smoothed";
    }
    if (normalized == "cartesian" || normalized == "flat" || normalized == "none")
    {
        return "cartesian";
    }
    return "";
}

double sanitize_positive(double value, double fallback, const char* label)
{
    if (!std::isfinite(value) || value <= terrain_constants::epsilon)
    {
        tmv::log_warn("Invalid ", label, "=", value, ". Using ", fallback, ".");
        return fallback;
    }
    return value;
}

double sanitize_non_negative(double value,
                             double fallback,
                             const char* label)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        tmv::log_warn("Invalid ", label, "=", value, ". Using ", fallback, ".");
        return fallback;
    }
    return value;
}

void sanitize_terrain_config(TerrainConfig& cfg)
{
    const std::string scheme = normalize_terrain_scheme_id(cfg.scheme_id);
    if (scheme.empty())
    {
        tmv::log_warn("Unknown terrain.scheme '", cfg.scheme_id, "'. Falling back to 'none'.");
        cfg.scheme_id = "none";
    }
    else
    {
        cfg.scheme_id = scheme;
    }

    const std::string coord = normalize_terrain_coord_id(cfg.coord_id);
    if (coord.empty())
    {
        tmv::log_warn("Unknown terrain.coord_id '", cfg.coord_id, "'. Falling back to 'btf'.");
        cfg.coord_id = "btf";
    }
    else
    {
        cfg.coord_id = coord;
    }

    cfg.ztop = sanitize_positive(cfg.ztop, 20000.0, "terrain.ztop");
    cfg.smoothing_decay = sanitize_non_negative(cfg.smoothing_decay, 0.1, "terrain.smoothing_decay");
    cfg.bell.h0 = sanitize_non_negative(cfg.bell.h0, 1000.0, "terrain.bell.h0");
    cfg.bell.a = sanitize_positive(std::abs(cfg.bell.a), 5000.0, "terrain.bell.a");
    cfg.schar.H = sanitize_non_negative(cfg.schar.H, 1000.0, "terrain.schar.H");
    cfg.schar.a = sanitize_positive(std::abs(cfg.schar.a), 5000.0, "terrain.schar.a");
    cfg.schar.ell = sanitize_positive(std::abs(cfg.schar.ell), 4000.0, "terrain.schar.ell");
}

double compute_max_terrain_height(const Topography2D& topo)
{
    double max_height = 0.0;
    for (const auto& row : topo.h)
    {
        for (double h : row)
        {
            if (std::isfinite(h))
            {
                max_height = std::max(max_height, h);
            }
        }
    }
    return max_height;
}

void enforce_safe_metric_top(TerrainConfig& cfg, const Topography2D& topo)
{
    if (cfg.coord_id == "cartesian")
    {
        return;
    }

    const double max_height = compute_max_terrain_height(topo);
    if (cfg.ztop > max_height + terrain_constants::epsilon)
    {
        return;
    }

    const double adjusted = std::max(max_height + 1.0, max_height * 1.05);
    tmv::log_warn("terrain.ztop (", cfg.ztop,
                  ") must exceed max terrain height (", max_height,
                  ") for terrain-following metrics. Using ztop=", adjusted, ".");
    cfg.ztop = adjusted;
}
}

std::unique_ptr<TerrainSchemeBase> terrain_scheme;
TerrainConfig global_terrain_config;
Topography2D global_topography;
TerrainMetrics3D global_terrain_metrics;

/**
 * @brief Initializes the terrain scheme.
 */
void initialize_terrain(const std::string& scheme_name, const TerrainConfig& cfg) 
{
    try 
    {
        global_terrain_config = cfg;
        if (!scheme_name.empty())
        {
            global_terrain_config.scheme_id = scheme_name;
        }
        if (global_terrain_config.scheme_id.empty())
        {
            global_terrain_config.scheme_id = "none";
        }
        sanitize_terrain_config(global_terrain_config);

        terrain_scheme = create_terrain_scheme(global_terrain_config.scheme_id);
        terrain_scheme->initialize(global_terrain_config);

        build_terrain_fields();

        tmv::log_info("Initialized terrain scheme: ", global_terrain_config.scheme_id);

    } 
    catch (const std::exception& e) 
    {
        tmv::log_error("Error initializing terrain: ", e.what());
        throw;
    }
}

/**
 * @brief Builds the terrain fields.
 */
void build_terrain_fields() 
{
    if (!terrain_scheme)
    {
        if (global_terrain_config.scheme_id.empty())
        {
            global_terrain_config.scheme_id = "none";
        }
        sanitize_terrain_config(global_terrain_config);
        terrain_scheme = create_terrain_scheme(global_terrain_config.scheme_id);
        terrain_scheme->initialize(global_terrain_config);
    }

    topography::initialize_topography(global_topography, NR, NTH);
    topography::initialize_metrics(global_terrain_metrics, NR, NTH, NZ);

    terrain_scheme->build_topography(global_terrain_config, global_topography);
    enforce_safe_metric_top(global_terrain_config, global_topography);

    TerrainDiagnostics diag;
    if (global_terrain_config.compute_metrics)
    {
        terrain_scheme->build_metrics(global_terrain_config, global_topography, global_terrain_metrics, &diag);
    }
    else
    {
        auto zeta_levels = topography::build_zeta_levels(NZ, global_terrain_config.ztop);
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    global_terrain_metrics.zeta(i, j, k) = zeta_levels[k];
                    global_terrain_metrics.z(i, j, k) = zeta_levels[k];
                    global_terrain_metrics.J(i, j, k) = 1.0;
                    global_terrain_metrics.mx(i, j, k) = 0.0;
                    global_terrain_metrics.my(i, j, k) = 0.0;
                }
            }
        }
        topography::check_coordinate_validity(global_terrain_metrics, diag);
    }

    const double max_height = compute_max_terrain_height(global_topography);

    tmv::log_info("Terrain fields built:",
                  " scheme=", global_terrain_config.scheme_id,
                  ", max_height=", max_height, " m",
                  ", min_J=", diag.min_jacobian,
                  ", max_J=", diag.max_jacobian);
    if (diag.coordinate_folding)
    {
        tmv::log_warn("Terrain warning: coordinate folding detected in terrain metrics");
    }

    refresh_grid_metrics_from_terrain();
}

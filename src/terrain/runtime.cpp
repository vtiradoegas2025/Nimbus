#include "compute/backend_runtime.hpp"
#include "core/runtime/grid_state.hpp"
#include "numerics/numerics_base.hpp"
#include "terrain/runtime.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "terrain/factory.hpp"
#include "terrain/base/topography.hpp"
#include "util/string_utils.hpp"
#include "util/log.hpp"
#include <iostream>
#include <cmath>
#include <limits>


namespace
{
std::string normalize_terrain_scheme_id(const std::string& value)
{
    const std::string normalized = tmv::strutil::trim_and_lower(value);
    if (normalized.empty())
    {
        return "none";
    }
    const auto* descriptor =
        tmv::find_scheme_descriptor(tmv::get_terrain_scheme_descriptors(), normalized);
    return descriptor ? descriptor->id : std::string{};
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

std::string normalize_horizontal_placement_id(const std::string& value)
{
    const std::string normalized = tmv::strutil::trim_and_lower(value);
    if (normalized.empty() || normalized == "grid_attached")
        return "grid_attached";
    if (normalized == "earth_fixed")
        return "earth_fixed";
    return "";
}

double require_positive(double value, const char* label)
{
    if (!std::isfinite(value) || value <= terrain_constants::epsilon)
    {
        throw std::invalid_argument(std::string(label) + " must be positive and finite");
    }
    return value;
}

double require_non_negative(double value, const char* label)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::invalid_argument(std::string(label) + " must be non-negative and finite");
    }
    return value;
}

void sanitize_terrain_config(TerrainConfig& cfg)
{
    const std::string scheme = normalize_terrain_scheme_id(cfg.scheme_id);
    if (scheme.empty())
    {
        throw std::invalid_argument("Unknown terrain.scheme '" + cfg.scheme_id + "'");
    }
    else
    {
        cfg.scheme_id = scheme;
    }

    const std::string coord = normalize_terrain_coord_id(cfg.coord_id);
    if (coord.empty())
    {
        throw std::invalid_argument("Unknown terrain.coord_id '" + cfg.coord_id + "'");
    }
    else
    {
        cfg.coord_id = coord;
    }

    const std::string placement =
        normalize_horizontal_placement_id(cfg.horizontal_placement_id);
    if (placement.empty())
        throw std::invalid_argument(
            "Unknown terrain.horizontal_placement '"
            + cfg.horizontal_placement_id + "'");
    cfg.horizontal_placement_id = placement;

    cfg.ztop = require_positive(cfg.ztop, "terrain.ztop");
    cfg.smoothing_decay = require_non_negative(cfg.smoothing_decay, "terrain.smoothing_decay");
    cfg.bell.h0 = require_non_negative(cfg.bell.h0, "terrain.bell.h0");
    cfg.bell.a = require_positive(cfg.bell.a, "terrain.bell.a");
    cfg.schar.H = require_non_negative(cfg.schar.H, "terrain.schar.H");
    cfg.schar.a = require_positive(cfg.schar.a, "terrain.schar.a");
    cfg.schar.ell = require_positive(cfg.schar.ell, "terrain.schar.ell");
    if (!std::isfinite(cfg.feature_center_earth_x_m)
        || !std::isfinite(cfg.feature_center_earth_y_m))
        throw std::invalid_argument(
            "terrain Earth feature center must be finite");
    if (cfg.horizontal_placement_id == "grid_attached"
        && (cfg.feature_center_earth_x_m != 0.0
            || cfg.feature_center_earth_y_m != 0.0))
        throw std::invalid_argument(
            "grid-attached terrain cannot carry ignored Earth feature-center coordinates");
    if (cfg.horizontal_placement_id == "earth_fixed")
    {
        if (cfg.scheme_id != "bell" && cfg.scheme_id != "schar")
            throw std::invalid_argument(
                "Earth-fixed terrain supports only bell or schar analytic schemes");
        if (!cfg.compute_derivatives || !cfg.compute_metrics)
            throw std::invalid_argument(
                "Earth-fixed terrain requires complete derivative and metric fields");
        if (cfg.coord_id != "btf" && cfg.coord_id != "smoothed")
            throw std::invalid_argument(
                "Earth-fixed terrain supports only btf or smoothed metrics");
    }
    if (cfg.scheme_id != "none" && !cfg.compute_metrics)
    {
        throw std::invalid_argument(
            "active terrain requires terrain.compute_metrics=true");
    }
    if (cfg.scheme_id != "none" && cfg.coord_id == "cartesian")
    {
        throw std::invalid_argument(
            "active terrain requires btf or smoothed coordinates; cartesian coordinates would ignore the topography");
    }
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

    throw std::invalid_argument(
        "terrain.ztop (" + std::to_string(cfg.ztop) +
        ") must exceed max terrain height (" + std::to_string(max_height) +
        ") for terrain-following metrics");
}
}

namespace tmv
{

std::unique_ptr<TerrainSchemeBase>& terrain_scheme =
    global_terrain_runtime_state.scheme;
TerrainConfig& global_terrain_config = global_terrain_runtime_state.config;
Topography2D& global_topography = global_terrain_runtime_state.topography;
TerrainMetrics3D& global_terrain_metrics = global_terrain_runtime_state.metrics;

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

void initialize_earth_fixed_terrain(
    const std::string& scheme_name,
    const TerrainConfig& cfg,
    const EarthFixedTerrainBuildRequest& request)
{
    TerrainRuntimeState candidate;
    candidate.config = cfg;
    if (!scheme_name.empty())
        candidate.config.scheme_id = scheme_name;
    if (candidate.config.scheme_id.empty())
        candidate.config.scheme_id = "none";
    sanitize_terrain_config(candidate.config);
    if (candidate.config.horizontal_placement_id != "earth_fixed")
        throw std::invalid_argument(
            "Earth-frame terrain initialization requires horizontal_placement=earth_fixed");
    const double expected_top = static_cast<double>(request.nz)
        * require_positive(::dz, "grid.dz");
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(expected_top),
                    std::abs(candidate.config.ztop)});
    if (std::abs(candidate.config.ztop - expected_top) > tolerance)
        throw std::invalid_argument("Earth-fixed terrain requires terrain.ztop == grid.nz * grid.dz");

    candidate.scheme = create_terrain_scheme(candidate.config.scheme_id);
    candidate.scheme->initialize(candidate.config);
    auto* backend = mutable_compute_backend();
    auto* kernels = backend ? backend->terrain_kernels() : nullptr;
    // Point-sampled nested meshes share canonical double-precision geometry;
    // continuously moving cell-average meshes use the native geometry owner.
    const bool native = request.sampling == EarthFixedTerrainSampling::CellAverageBilinear &&
                        active_compute_backend_kind() != ComputeBackendKind::Cpu;
    if (native && (!kernels || compute_backend_fallback_active()))
        throw std::runtime_error("native Earth-fixed initialization requires its geometry owner without fallback");
    auto                     prepared    = native ? prepare_earth_fixed_terrain_commit(candidate, request, *kernels)
                                                  : prepare_earth_fixed_terrain_commit(candidate, request);
    const TerrainDiagnostics diagnostics = prepared.geometry.diagnostics;
    if (!commit_earth_fixed_terrain(prepared, candidate))
        throw std::runtime_error("Earth-fixed terrain initialization lost its detached commit identity");

    global_terrain_runtime_state = std::move(candidate);
    tmv::log_info("Terrain fields built: scheme=",
                  global_terrain_config.scheme_id,
                  ", placement=earth_fixed, generation=",
                  global_terrain_runtime_state.earth_fixed_placement.generation,
                  ", geometry_owner=",
                  native ? "native_cell_average" :
                      (request.sampling == EarthFixedTerrainSampling::Point ? "canonical_host_point" : "host_cell_average"),
                  ", max_height=",
                  diagnostics.max_height,
                  " m",
                  ", min_J=",
                  diagnostics.min_jacobian,
                  ", max_J=",
                  diagnostics.max_jacobian);
    tmv::log_info("Initialized terrain scheme: ", global_terrain_config.scheme_id);
    ::refresh_grid_metrics_from_terrain();
}

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

    if (global_terrain_config.horizontal_placement_id == "earth_fixed")
        throw std::invalid_argument(
            "Earth-fixed terrain requires an explicit Earth-frame geometry commit");

    if (global_terrain_config.scheme_id != "none")
    {
        const double expected_top = static_cast<double>(NZ)
            * require_positive(::dz, "grid.dz");
        const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(expected_top),
                        std::abs(global_terrain_config.ztop)});
        if (std::abs(global_terrain_config.ztop - expected_top) > tolerance)
        {
            throw std::invalid_argument(
                "active terrain requires terrain.ztop == grid.nz * grid.dz so the computational vertical coordinate matches dynamics");
        }
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

    ::refresh_grid_metrics_from_terrain();
}

}  // namespace tmv

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/infra/grid_coordinates.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/output/durable_file.hpp"
#include "core/output/field_brick_hierarchy.hpp"
#include "core/output/field_registry.hpp"
#include "diagnostics/field_contract.hpp"
#include "util/string_utils.hpp"

/**
 * @file step_manifest.hpp
 * @brief Step-level output manifest JSON generated from the shared field registry.
 */

struct OutputStepManifestGrid
{
    int         nr                = 0;
    int         nth               = 0;
    int         nz                = 0;
    std::string coordinate_system = "cartesian";
    std::string staggering        = "collocated";
    double      dr_m              = 0.0;
    double      dy_m              = 0.0;
    // Legacy compatibility alias. Cartesian manifests store dy metres here.
    double dtheta_rad = 0.0;
    double dz_m       = 0.0;
    std::string domain_motion_mode = "stationary";
    std::string horizontal_frame_contract =
        "constant_translation_earth_wind_grid_transport_v1";
    double grid_origin_t0_earth_x_m = 0.0;
    double grid_origin_t0_earth_y_m = 0.0;
    double grid_origin_earth_x_m = 0.0;
    double grid_origin_earth_y_m = 0.0;
    double frame_velocity_u_ms = 0.0;
    double frame_velocity_v_ms = 0.0;
    bool   terrain_following = false;
    double terrain_min_z_m   = std::numeric_limits<double>::quiet_NaN();
    double terrain_max_z_m   = std::numeric_limits<double>::quiet_NaN();
    std::string terrain_geometry_sha256;
    std::string terrain_geometry_publication_generation;
    std::string terrain_sampling;
    std::string coordinate_remap_method;
};

struct OutputStepManifestSoundings
{
    bool        enabled = false;
    std::string scheme;
    std::string file_path;
    std::string interpolation_method;
};

struct OutputStepManifestCompression
{
    std::string field_id;
    std::string codec = "zfp";
    std::string mode;
    double      mode_parameter = 0.0;
    std::string mode_parameter_units;
    std::string error_contract;
    std::string frame_role;
    std::vector<int> parent_step_indices;
    double      sparse_threshold = 0.0;
    bool        float16_prefilter = false;
    std::string decoded_dtype = "float32";
    int         container_version = 3;
    std::string spatial_layout = "independently_decodable_bricks";
    std::array<int, 3> brick_shape{64, 64, 32};
    int contract_version = 0;
};

inline bool is_reversible_zfp_radar_field_id(std::string_view field_id)
{
    return field_id == "reflectivity_dbz" || field_id == "radial_velocity" ||
        field_id == "zdr" || field_id == "kdp" || field_id == "rhohv";
}

inline OutputStepManifestCompression make_reversible_zfp_manifest_compression(
    const std::string& field_id)
{
    if (!is_reversible_zfp_radar_field_id(field_id))
        throw std::invalid_argument("reversible ZFP manifest requires a selected radar field");
    OutputStepManifestCompression compression;
    compression.field_id = field_id;
    compression.contract_version = 2;
    compression.mode = "reversible";
    compression.mode_parameter = 0.0;
    compression.mode_parameter_units = "none";
    compression.error_contract = "bitwise_float32_reconstruction";
    compression.frame_role = "keyframe";
    compression.container_version = 4;
    return compression;
}

struct OutputStepManifestRadar
{
    bool enabled = false;
    std::vector<std::string> products;
    std::string reflectivity_operator;
    std::string velocity_operator;
    std::string zdr_operator;
    std::string polarimetric_operator;
    double radar_x_m = 0.0;
    double radar_y_m = 0.0;
    double radar_z_m = 0.0;
    bool apply_scatterer_correction = false;
};

struct OutputStepManifestContext
{
    int                         step_index        = 0;
    double                      simulation_time_s = 0.0;
    std::string                 step_dir;
    std::string                 publication_generation;
    std::string                 storage_format = "npy_3d";
    std::string                 data_product_kind = "history";
    std::string                 zfp_field_execution = "serial";
    int                         zfp_field_workers = 1;
    int                         zfp_field_scratch_mib = 64;
    std::string                 domain_recycling_transaction_id;
    int                         domain_recycling_tile_ordinal = -1;
    // Ordinary frames contain the model's closure/halo storage. A retiring
    // swath tile has already been cropped to physical cells.
    bool                        storage_includes_model_halos = true;
    OutputStepManifestGrid      grid;
    OutputStepManifestSoundings soundings;
    OutputStepManifestRadar     radar;
    // Exact per-artifact lossy transform contract. Empty for lossless output.
    std::vector<OutputStepManifestCompression> compression;
    // Exact file stems published for this step. Unlike the broad field catalog
    // below, this closes the discovery race by letting readers require an exact
    // match with the directory inventory they observed.
    std::vector<std::string> published_fields;
    bool field_brick_hierarchies = false;
};

inline constexpr int         kNimbusOutputManifestVersion     = 7;
inline constexpr const char* kNimbusRunOutputContractFilename = "nimbus_output.json";

inline std::string field_brick_hierarchy_sha256_token(std::string_view field_id)
{
    return "nimbus-pending-field-brick-sha256:" + std::string(field_id);
}

inline std::string make_output_publication_generation()
{
    static std::atomic<std::uint64_t> serial{0};
    std::random_device                entropy_source;
    const std::uint64_t               entropy =
        (static_cast<std::uint64_t>(entropy_source()) << 32U) ^ static_cast<std::uint64_t>(entropy_source());
    const std::uint64_t clock =
        static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^
        serial.fetch_add(1, std::memory_order_relaxed);

    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << entropy << std::setw(16) << clock;
    return out.str();
}

/**
 * Publish a small JSON contract without ever exposing a partially written final
 * file.  The temporary file lives beside the destination, so rename is atomic
 * on the filesystem that owns the run directory.
 */
inline bool
publish_output_json_atomically(const std::filesystem::path& path, const std::string& json, std::string& error)
{
    error.clear();
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::error_code             ec;
    std::filesystem::remove(temporary, ec);
    ec.clear();

    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            error = "failed to open temporary manifest '" + temporary.string() + "'";
            return false;
        }
        out.write(json.data(), static_cast<std::streamsize>(json.size()));
        out.flush();
        if (!out.good())
        {
            error = "failed to flush temporary manifest '" + temporary.string() + "'";
            out.close();
            std::filesystem::remove(temporary, ec);
            return false;
        }
    }

    // Flush before the rename: otherwise the rename can reach the disk while
    // the JSON body has not, publishing an empty or torn contract as complete.
    if (!nimbus::output::sync_file_to_disk(temporary, error))
    {
        std::filesystem::remove(temporary, ec);
        return false;
    }

    if (!nimbus::output::atomic_replace_file(temporary, path, error))
    {
        std::filesystem::remove(temporary, ec);
        return false;
    }

    // The rename is only durable once the directory entry itself is flushed.
    const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    return nimbus::output::sync_directory_to_disk(parent, error);
}

inline std::string build_output_run_contract_json(const std::string& publication_generation,
                                                  const char*        publication_state,
                                                  bool               includes_nested_child_history = false,
                                                  bool               includes_qualification_diagnostics = false,
                                                  bool               includes_ordinary_history = true)
{
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"nimbus.output\",\n";
    out << "  \"manifest_version\": " << kNimbusOutputManifestVersion << ",\n";
    out << "  \"publication_generation\": \"" << tmv::strutil::json_escape(publication_generation) << "\",\n";
    out << "  \"publication_state\": \"" << tmv::strutil::json_escape(publication_state) << "\",\n";
    out << "  \"frame_completion\": \"atomic_step_manifest\",\n";
    out << "  \"history_index\": {\"schema\": \"nimbus.history.index\", \"version\": 1, \"checksum\": \"sha256\"},\n";
    out << "  \"data_products\": [";
    bool first = true;
    const auto append_product = [&](const char* product)
    {
        if (!first)
            out << ", ";
        out << "\"" << product << "\"";
        first = false;
    };
    if (includes_ordinary_history)
        append_product("history");
    if (includes_nested_child_history && includes_ordinary_history)
        append_product("nested_child_history");
    if (includes_qualification_diagnostics)
    {
        if (includes_nested_child_history)
            append_product("nested_child_qualification_diagnostics");
        else
            append_product("qualification_diagnostics");
    }
    out << "]\n";
    out << "}\n";
    return out.str();
}

inline std::string build_output_step_manifest_json(const OutputStepManifestContext& context)
{
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    StaggerType staggering;
    if (context.grid.staggering == "c_grid")
        staggering = StaggerType::CGrid;
    else if (context.grid.staggering == "collocated")
        staggering = StaggerType::Collocated;
    else
        throw std::invalid_argument(
            "output manifest requires canonical c_grid or collocated staggering");
    const PhysicalCellDomain domain = context.storage_includes_model_halos
        ? physical_scalar_domain(
              context.grid.nr, context.grid.nth, context.grid.nz, staggering)
        : PhysicalCellDomain{
              0, context.grid.nr, 0, context.grid.nth, 0, context.grid.nz};
    const int i_max = domain.i_end > domain.i_begin
        ? domain.i_end - 1 : domain.i_begin;
    const int j_max = domain.j_end > domain.j_begin
        ? domain.j_end - 1 : domain.j_begin;
    const int k_max = domain.k_end > domain.k_begin
        ? domain.k_end - 1 : domain.k_begin;
    const int i_halo_upper = std::max(0, context.grid.nr - domain.i_end);
    const int j_halo_upper = std::max(0, context.grid.nth - domain.j_end);
    const int k_halo_upper = std::max(0, context.grid.nz - domain.k_end);
    if (context.grid.terrain_following
        && (!std::isfinite(context.grid.terrain_min_z_m)
            || !std::isfinite(context.grid.terrain_max_z_m)
            || !(context.grid.terrain_max_z_m > context.grid.terrain_min_z_m)
            || (context.storage_format != "netcdf"
                && (context.grid.terrain_geometry_sha256.size() != 64
                    || context.grid.terrain_geometry_publication_generation
                           != context.publication_generation))))
    {
        throw std::invalid_argument(
            "terrain output manifest requires finite increasing physical vertical bounds");
    }
    const double frame_values[] = {
        context.grid.grid_origin_t0_earth_x_m,
        context.grid.grid_origin_t0_earth_y_m,
        context.grid.grid_origin_earth_x_m,
        context.grid.grid_origin_earth_y_m,
        context.grid.frame_velocity_u_ms,
        context.grid.frame_velocity_v_ms,
    };
    if (context.grid.domain_motion_mode.empty()
        || context.grid.horizontal_frame_contract.empty()
        || std::any_of(
            std::begin(frame_values), std::end(frame_values),
            [](double value) { return !std::isfinite(value); }))
    {
        throw std::invalid_argument(
            "output manifest requires complete finite horizontal-frame provenance");
    }
    const bool nested_child_history =
        context.data_product_kind == "nested_child_history";
    const bool qualification_diagnostics =
        context.data_product_kind == "qualification_diagnostics";
    const bool nested_child_qualification_diagnostics =
        context.data_product_kind ==
        "nested_child_qualification_diagnostics";
    const bool retiring_tile =
        context.data_product_kind == "retiring_swath_tile";
    const bool mixed_reversible_zfp = std::any_of(
        context.compression.begin(), context.compression.end(),
        [](const OutputStepManifestCompression& item) { return item.contract_version == 2; });
    if (mixed_reversible_zfp)
    {
        if (context.storage_format != "zfp")
            throw std::invalid_argument("mixed reversible ZFP requires zfp storage");
        if (context.compression.size() != context.published_fields.size())
            throw std::invalid_argument("mixed reversible ZFP requires one compressed artifact per field");
        std::vector<std::string> published_radar;
        for (const auto& field : context.published_fields)
            if (is_reversible_zfp_radar_field_id(field))
                published_radar.push_back(field);
        std::sort(published_radar.begin(), published_radar.end());
        auto selected_radar = context.radar.products;
        std::sort(selected_radar.begin(), selected_radar.end());
        if (!context.radar.enabled || published_radar.empty() ||
            selected_radar != published_radar ||
            std::adjacent_find(selected_radar.begin(), selected_radar.end()) !=
                selected_radar.end())
            throw std::invalid_argument("mixed reversible ZFP radar provenance is incomplete");
        std::size_t reversible_fields = 0;
        std::size_t accuracy_fields = 0;
        for (const auto& field : context.published_fields)
        {
            const auto item = std::find_if(
                context.compression.begin(), context.compression.end(),
                [&](const OutputStepManifestCompression& candidate)
                { return candidate.field_id == field; });
            if (item == context.compression.end() ||
                std::count_if(context.compression.begin(), context.compression.end(),
                    [&](const OutputStepManifestCompression& candidate)
                    { return candidate.field_id == field; }) != 1)
                throw std::invalid_argument("mixed reversible ZFP field metadata is incomplete");
            if (is_reversible_zfp_radar_field_id(field))
            {
                if (item->contract_version != 2)
                    throw std::invalid_argument("mixed reversible ZFP radar field lacks v4 contract");
                ++reversible_fields;
            }
            else
            {
                if (item->contract_version != 0 || item->mode != "accuracy" ||
                    !std::isfinite(item->mode_parameter) || item->mode_parameter <= 0.0 ||
                    item->mode_parameter_units != "absolute_field_units" ||
                    item->error_contract != "zfp_absolute_accuracy_before_prefilters" ||
                    item->container_version != 3 || item->sparse_threshold != 0.0 ||
                    item->float16_prefilter)
                    throw std::invalid_argument("mixed reversible ZFP non-radar field lacks v3 accuracy contract");
                ++accuracy_fields;
            }
        }
        if (reversible_fields == 0 || accuracy_fields == 0)
            throw std::invalid_argument("mixed reversible ZFP requires radar and accuracy fields");
    }
    for (const auto& item : context.compression)
    {
        if (item.contract_version == 2 &&
            (item.codec != "zfp" || item.mode != "reversible" ||
             item.mode_parameter != 0.0 || std::signbit(item.mode_parameter) ||
             item.mode_parameter_units != "none" ||
             item.error_contract != "bitwise_float32_reconstruction" ||
             item.frame_role != "keyframe" || !item.parent_step_indices.empty() ||
             item.sparse_threshold != 0.0 || item.float16_prefilter ||
             item.decoded_dtype != "float32" || item.container_version != 4 ||
             item.spatial_layout != "independently_decodable_bricks" ||
             item.brick_shape != std::array<int, 3>{64, 64, 32}))
            throw std::invalid_argument("output manifest has invalid reversible ZFP field contract");
        if (item.contract_version != 0 && item.contract_version != 2)
            throw std::invalid_argument("output manifest has unsupported ZFP field contract version");
        if (item.contract_version != 2 && item.container_version == 4)
            throw std::invalid_argument("output manifest v4 field lacks reversible contract");
    }
    if (!context.compression.empty() &&
        ((context.zfp_field_execution != "serial" &&
          context.zfp_field_execution != "field_parallel") ||
         context.zfp_field_workers < 1 || context.zfp_field_workers > 16 ||
         context.zfp_field_scratch_mib < 1 ||
         context.zfp_field_scratch_mib > 4096))
    {
        throw std::invalid_argument(
            "output manifest has invalid ZFP field execution provenance");
    }
    if ((context.data_product_kind != "history" && !nested_child_history &&
         !qualification_diagnostics &&
         !nested_child_qualification_diagnostics && !retiring_tile)
        || (retiring_tile
            && (context.domain_recycling_transaction_id.empty()
                || context.domain_recycling_tile_ordinal < 0)))
    {
        throw std::invalid_argument(
            "output manifest has invalid data-product semantics");
    }

    out << "{\n";
    out << "  \"schema\": \"nimbus.step\",\n";
    out << "  \"manifest_version\": " << kNimbusOutputManifestVersion << ",\n";
    out << "  \"publication_state\": \"complete\",\n";
    out << "  \"publication_generation\": \"" << tmv::strutil::json_escape(context.publication_generation) << "\",\n";
    out << "  \"step_index\": " << context.step_index << ",\n";
    out << "  \"simulation_time_s\": " << context.simulation_time_s << ",\n";
    out << "  \"step_dir\": \"" << tmv::strutil::json_escape(context.step_dir) << "\",\n";
    out << "  \"data_product\": {\n";
    out << "    \"kind\": \""
        << tmv::strutil::json_escape(context.data_product_kind) << "\",\n";
    out << "    \"restart_complete\": false,\n";
    out << "    \"storage_format\": \"" << tmv::strutil::json_escape(context.storage_format) << "\"";
    if (!context.compression.empty())
    {
        out << ",\n";
        if (mixed_reversible_zfp)
            out << "    \"compression_contract\": \"nimbus.zfp.per_field.v2\",\n";
        else
            out << "    \"lossy_compression_contract\": \"nimbus.zfp.per_field.v1\",\n";
        out << "    \"field_execution_policy\": \""
            << tmv::strutil::json_escape(context.zfp_field_execution)
            << "\",\n";
        out << "    \"field_workers\": " << context.zfp_field_workers
            << ",\n";
        out << "    \"field_scratch_budget_mib\": "
            << context.zfp_field_scratch_mib << "\n";
    }
    else
    {
        out << "\n";
    }
    out << "  },\n";
    if (retiring_tile)
    {
        out << "  \"domain_recycling\": {\"transaction_id\": \""
            << tmv::strutil::json_escape(
                   context.domain_recycling_transaction_id)
            << "\", \"tile_ordinal\": "
            << context.domain_recycling_tile_ordinal
            << ", \"publication_phase\": \"captured_precommit\"},\n";
    }
    out << "  \"grid\": {\n";
    out << "    \"coordinate_system\": \"" << tmv::strutil::json_escape(context.grid.coordinate_system) << "\",\n";
    out << "    \"axis_order\": [\"x\", \"y\", \"z\"],\n";
    out << "    \"dimensions\": {\"x\": " << context.grid.nr << ", \"y\": " << context.grid.nth
        << ", \"z\": " << context.grid.nz << "},\n";
    out << "    \"spacing_m\": {\"x\": " << context.grid.dr_m << ", \"y\": " << context.grid.dy_m
        << ", \"z\": " << context.grid.dz_m << "},\n";
    out << "    \"native_model_staggering\": \"" << tmv::strutil::json_escape(context.grid.staggering) << "\",\n";
    out << "    \"publication_staggering\": \"cell_centered\",\n";
    out << "    \"horizontal_frame\": {\"contract\": \""
        << tmv::strutil::json_escape(context.grid.horizontal_frame_contract)
        << "\", \"motion_mode\": \""
        << tmv::strutil::json_escape(context.grid.domain_motion_mode)
        << "\", \"grid_origin_t0_earth_m\": ["
        << context.grid.grid_origin_t0_earth_x_m << ", "
        << context.grid.grid_origin_t0_earth_y_m
        << "], \"grid_origin_earth_m\": ["
        << context.grid.grid_origin_earth_x_m << ", "
        << context.grid.grid_origin_earth_y_m
        << "], \"frame_velocity_earth_ms\": ["
        << context.grid.frame_velocity_u_ms << ", "
        << context.grid.frame_velocity_v_ms
        << "], \"published_horizontal_wind\": \"earth_relative\", "
           "\"horizontal_transport_velocity\": \"grid_relative\"},\n";
    if (context.grid.terrain_following)
    {
        out << "    \"vertical_geometry\": {\"kind\": \"terrain_following\", "
            << "\"center_coordinate\": \"physical_z_center_m\", "
            << "\"interface_coordinate\": \"physical_z_interface_m\", "
            << "\"dtype\": \"float64\", \"layout\": \"x_y_z\", "
            << "\"storage\": \""
            << (context.storage_format == "netcdf" ? "netcdf_variable" : "versioned_npy_geometry")
            << "\"";
        if (context.storage_format != "netcdf")
        {
            const std::string relative = "../geometry/" +
                context.grid.terrain_geometry_publication_generation + "/" +
                context.grid.terrain_geometry_sha256;
            out << ", \"contract\": \"nimbus.terrain-geometry.v1\", "
                << "\"content_sha256\": \"" << context.grid.terrain_geometry_sha256 << "\", "
                << "\"publication_generation\": \""
                << tmv::strutil::json_escape(context.grid.terrain_geometry_publication_generation)
                << "\", \"geometry_manifest\": \"" << relative << "/manifest.json\", "
                << "\"center_path\": \"" << relative << "/physical_z_center_m.npy\", "
                << "\"interface_path\": \"" << relative << "/physical_z_interface_m.npy\"";
        }
        if (!context.grid.terrain_sampling.empty())
            out << ", \"sampling\": \"" << tmv::strutil::json_escape(context.grid.terrain_sampling) << "\"";
        if (!context.grid.coordinate_remap_method.empty())
            out << ", \"coordinate_remap_method\": \""
                << tmv::strutil::json_escape(context.grid.coordinate_remap_method) << "\", \"coordinate_remap_order\": 1";
        out << "},\n";
        out << "    \"horizontal_scalar_origin_m\": ["
            << context.grid.grid_origin_earth_x_m
                 + tmv::grid_coordinates::scalar_center(0, context.grid.dr_m)
            << ", "
            << context.grid.grid_origin_earth_y_m
                 + tmv::grid_coordinates::scalar_center(0, context.grid.dy_m)
            << "],\n";
        out << "    \"grid_horizontal_scalar_origin_m\": ["
            << tmv::grid_coordinates::scalar_center(0, context.grid.dr_m)
            << ", "
            << tmv::grid_coordinates::scalar_center(0, context.grid.dy_m)
            << "],\n";
    }
    else
    {
        out << "    \"scalar_origin_m\": ["
            << context.grid.grid_origin_earth_x_m
                 + tmv::grid_coordinates::scalar_center(0, context.grid.dr_m)
            << ", "
            << context.grid.grid_origin_earth_y_m
                 + tmv::grid_coordinates::scalar_center(0, context.grid.dy_m)
            << ", "
            << tmv::grid_coordinates::scalar_center(0, context.grid.dz_m) << "],\n";
        out << "    \"grid_scalar_origin_m\": ["
            << tmv::grid_coordinates::scalar_center(0, context.grid.dr_m)
            << ", "
            << tmv::grid_coordinates::scalar_center(0, context.grid.dy_m)
            << ", "
            << tmv::grid_coordinates::scalar_center(0, context.grid.dz_m)
            << "],\n";
    }
    out << "    \"physical_bounds_m\": {\"lower\": ["
        << context.grid.grid_origin_earth_x_m
             + static_cast<double>(domain.i_begin) * context.grid.dr_m << ", "
        << context.grid.grid_origin_earth_y_m
             + static_cast<double>(domain.j_begin) * context.grid.dy_m << ", "
        << (context.grid.terrain_following
                ? context.grid.terrain_min_z_m
                : static_cast<double>(domain.k_begin) * context.grid.dz_m)
        << "], \"upper\": ["
        << context.grid.grid_origin_earth_x_m
             + static_cast<double>(domain.i_end) * context.grid.dr_m << ", "
        << context.grid.grid_origin_earth_y_m
             + static_cast<double>(domain.j_end) * context.grid.dy_m << ", "
        << (context.grid.terrain_following
                ? context.grid.terrain_max_z_m
                : static_cast<double>(domain.k_end) * context.grid.dz_m)
        << "]},\n";
    out << "    \"valid_index_range\": {\"lower\": [" << domain.i_begin << ", " << domain.j_begin << ", " << domain.k_begin << "], \"upper\": [" << i_max
        << ", " << j_max << ", " << k_max << "]},\n";
    out << "    \"halo_width\": {\"lower\": [" << domain.i_begin << ", " << domain.j_begin << ", " << domain.k_begin << "], \"upper\": [" << i_halo_upper << ", "
        << j_halo_upper << ", " << k_halo_upper << "]}\n";
    out << "  },\n";
    if (context.storage_format == "npy_2d_slices")
    {
        out << "  \"slice_index\": {\n";
        out << "    \"axis\": \"y\",\n";
        out << "    \"min\": " << domain.j_begin << ",\n";
        out << "    \"max\": " << j_max << ",\n";
        out << "    \"count\": " << domain.ny() << ",\n";
        out << "    \"storage_file_prefix\": \"th{y_index}_\"\n";
        out << "  },\n";
    }
    out << "  \"field_identity\": {\n";
    out << "    \"canonical_schema\": \"nimbus.fields\",\n";
    out << "    \"canonical_version\": 1,\n";
    out << "    \"cm1_alias_schema\": \"cm1.history\",\n";
    out << "    \"cm1_alias_version\": 1\n";
    out << "  },\n";
    out << "  \"soundings\": {\n";
    out << "    \"enabled\": " << (context.soundings.enabled ? "true" : "false") << ",\n";
    out << "    \"scheme\": \"" << tmv::strutil::json_escape(context.soundings.scheme) << "\",\n";
    out << "    \"file_path\": \"" << tmv::strutil::json_escape(context.soundings.file_path) << "\",\n";
    out << "    \"interpolation_method\": \"" << tmv::strutil::json_escape(context.soundings.interpolation_method)
        << "\"\n";
    out << "  },\n";
    if (context.radar.enabled)
    {
        out << "  \"radar_provenance\": {\n";
        out << "    \"products\": [";
        for (std::size_t index = 0; index < context.radar.products.size(); ++index)
        {
            if (index != 0) out << ", ";
            out << "\"" << tmv::strutil::json_escape(context.radar.products[index]) << "\"";
        }
        out << "],\n";
        out << "    \"reflectivity_operator\": \""
            << tmv::strutil::json_escape(context.radar.reflectivity_operator) << "\",\n";
        out << "    \"velocity_operator\": \""
            << tmv::strutil::json_escape(context.radar.velocity_operator) << "\",\n";
        out << "    \"zdr_operator\": \""
            << tmv::strutil::json_escape(context.radar.zdr_operator) << "\",\n";
        out << "    \"polarimetric_operator\": \""
            << tmv::strutil::json_escape(context.radar.polarimetric_operator) << "\",\n";
        out << "    \"instrument_location_earth_m\": ["
            << context.radar.radar_x_m << ", " << context.radar.radar_y_m
            << ", " << context.radar.radar_z_m << "],\n";
        out << "    \"scatterer_correction\": "
            << (context.radar.apply_scatterer_correction ? "true" : "false") << "\n";
        out << "  },\n";
    }
    out << "  \"published_fields\": [";
    for (std::size_t index = 0; index < context.published_fields.size(); ++index)
    {
        if (index > 0)
        {
            out << ", ";
        }
        out << "\"" << tmv::strutil::json_escape(context.published_fields[index]) << "\"";
    }
    out << "],\n";
    out << "  \"fields\": [\n";

    bool first_field = true;

    auto write_field_entry = [&](const char* field_id, const char* suffix, const char* category, bool derived)
    {
        const auto compression = std::find_if(
            context.compression.begin(),
            context.compression.end(),
            [&](const OutputStepManifestCompression& item) { return item.field_id == field_id; });
        if (!first_field)
        {
            out << ",\n";
        }
        first_field = false;

        out << "    {\n";
        out << "      \"field_id\": \"" << tmv::strutil::json_escape(field_id) << "\",\n";
        out << "      \"category\": \"" << tmv::strutil::json_escape(category) << "\",\n";
        out << "      \"derived\": " << (derived ? "true" : "false") << ",\n";
        out << "      \"file_suffix\": \"" << tmv::strutil::json_escape(suffix) << "\",\n";
        out << "      \"file_pattern\": \"";
        if (context.storage_format == "npy_2d_slices")
        {
            out << "th{y_index}_" << tmv::strutil::json_escape(suffix) << ".npy";
        }
        else if (context.storage_format == "zfp" && compression != context.compression.end())
        {
            out << tmv::strutil::json_escape(suffix) << ".zfp3d";
        }
        else if (context.storage_format == "csv")
        {
            out << tmv::strutil::json_escape(suffix) << ".csv";
        }
        else if (context.storage_format == "netcdf")
        {
            out << "history.nc";
        }
        else
        {
            out << tmv::strutil::json_escape(suffix) << ".npy";
        }
        out << "\"";

        if (context.storage_format == "zfp" &&
            compression == context.compression.end() &&
            std::find(context.published_fields.begin(),
                      context.published_fields.end(), field_id) !=
                context.published_fields.end())
        {
            out << ",\n";
            out << "      \"storage_format\": \"npy_2d\"";
        }

        if (context.storage_format == "netcdf")
        {
            out << ",\n";
            out << "      \"storage_layout\": \"shared_file\",\n";
            out << "      \"variable_name\": \"" << tmv::strutil::json_escape(field_id) << "\"";
        }

        if (const tmv::FieldContract* contract = tmv::find_field_contract(field_id))
        {
            out << ",\n";
            out << "      \"units\": \"" << tmv::strutil::json_escape(contract->units) << "\",\n";
            out << "      \"description\": \"" << tmv::strutil::json_escape(contract->description) << "\",\n";
            out << "      \"grid_location\": \"" << tmv::to_string(contract->grid_location) << "\",\n";
            out << "      \"state_semantic\": \"" << tmv::to_string(contract->state_semantic) << "\",\n";
            if (contract->allows_missing_values)
                out << "      \"missing_value\": \"NaN\",\n";
            out << "      \"cm1_aliases\": [";
            for (std::size_t alias_index = 0; alias_index < contract->cm1_aliases.size(); ++alias_index)
            {
                if (alias_index > 0)
                {
                    out << ", ";
                }
                out << "\"" << tmv::strutil::json_escape(contract->cm1_aliases[alias_index]) << "\"";
            }
            out << "]";

            if (contract->default_bounds.has_min || contract->default_bounds.has_max)
            {
                out << ",\n";
                out << "      \"bounds\": {";
                bool first_bound = true;
                if (contract->default_bounds.has_min)
                {
                    out << "\"min\": " << contract->default_bounds.min_value;
                    first_bound = false;
                }
                if (contract->default_bounds.has_max)
                {
                    if (!first_bound)
                    {
                        out << ", ";
                    }
                    out << "\"max\": " << contract->default_bounds.max_value;
                }
                out << "}";
            }
        }

        if (compression != context.compression.end())
        {
            out << ",\n";
            out << "      \"compression\": {\n";
            if (compression->contract_version != 0)
                out << "        \"contract_version\": "
                    << compression->contract_version << ",\n";
            out << "        \"codec\": \"" << tmv::strutil::json_escape(compression->codec) << "\",\n";
            out << "        \"mode\": \"" << tmv::strutil::json_escape(compression->mode) << "\",\n";
            out << "        \"mode_parameter\": " << compression->mode_parameter << ",\n";
            out << "        \"mode_parameter_units\": \""
                << tmv::strutil::json_escape(compression->mode_parameter_units) << "\",\n";
            out << "        \"error_contract\": \""
                << tmv::strutil::json_escape(compression->error_contract) << "\",\n";
            out << "        \"frame_role\": \""
                << tmv::strutil::json_escape(compression->frame_role) << "\",\n";
            out << "        \"parent_step_indices\": [";
            for (std::size_t parent = 0; parent < compression->parent_step_indices.size(); ++parent)
            {
                if (parent > 0) out << ", ";
                out << compression->parent_step_indices[parent];
            }
            out << "],\n";
            out << "        \"sparse_threshold\": " << compression->sparse_threshold << ",\n";
            out << "        \"float16_prefilter\": "
                << (compression->float16_prefilter ? "true" : "false") << ",\n";
            out << "        \"decoded_dtype\": \""
                << tmv::strutil::json_escape(compression->decoded_dtype) << "\",\n";
            out << "        \"container_version\": "
                << compression->container_version << ",\n";
            out << "        \"spatial_layout\": \""
                << tmv::strutil::json_escape(compression->spatial_layout)
                << "\",\n";
            out << "        \"brick_shape\": ["
                << compression->brick_shape[0] << ", "
                << compression->brick_shape[1] << ", "
                << compression->brick_shape[2] << "]\n";
            out << "      }";
        }

        const bool field_published = std::find(
            context.published_fields.begin(), context.published_fields.end(), field_id)
            != context.published_fields.end();
        const tmv::FieldContract* hierarchy_contract = tmv::find_field_contract(field_id);
        if (context.field_brick_hierarchies && field_published && hierarchy_contract != nullptr &&
            hierarchy_contract->dimensionality == tmv::FieldDimensionality::Volume3D)
        {
            const bool zfp_payload = context.storage_format == "zfp" &&
                compression != context.compression.end();
            const auto zfp_padding = zfp_payload
                ? nimbus::output::zfp_decoded_extrema_padding(
                    compression->mode, compression->mode_parameter,
                    compression->sparse_threshold, compression->float16_prefilter)
                : std::optional<double>{};
            const bool zfp_bounds_safe = zfp_padding.has_value();
            const bool culling_safe = !zfp_payload || zfp_bounds_safe;
            const double extrema_padding = zfp_padding.value_or(0.0);
            out << ",\n";
            out << "      \"brick_hierarchy\": {\n";
            out << "        \"schema\": \"nimbus.field-brick-hierarchy\",\n";
            out << "        \"schema_version\": "
                << (context.grid.terrain_following ? 3 : 2) << ",\n";
            out << "        \"path\": \""
                << tmv::strutil::json_escape(field_id) << ".bricks.json\",\n";
            out << "        \"content_sha256\": \""
                << field_brick_hierarchy_sha256_token(field_id) << "\",\n";
            out << "        \"native_brick_shape\": [64, 64, 32],\n";
            out << "        \"extrema_semantics\": \""
                << (culling_safe
                        ? (extrema_padding > 0.0 ? "decoded_value_bounds" : "stored_values")
                        : "pre_storage_source") << "\",\n";
            out << "        \"extrema_absolute_padding\": "
                << extrema_padding << ",\n";
            out << "        \"culling_safe\": "
                << (culling_safe ? "true" : "false") << "\n";
            out << "      }";
        }

        out << "\n";
        out << "    }";
    };

    for (const auto& field : output_field_registry())
    {
        write_field_entry(
            field.field_id, field.suffix, output_field_category_name(field.kind), output_field_is_derived(field));
    }

    out << "\n";
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

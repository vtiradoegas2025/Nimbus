#include "core/output/domain_recycling_swath.hpp"

#include "core/infra/domain_recycling_provenance.hpp"
#include "core/output/durable_file.hpp"
#include "core/output/field_registry.hpp"
#include "core/output/output_writer.hpp"
#include "core/output/step_manifest.hpp"
#include "util/string_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nimbus::output
{
namespace
{

bool plans_equal(const tmv::domain::WholeCellShiftPlan& lhs,
                 const tmv::domain::WholeCellShiftPlan& rhs) noexcept
{
    return lhs.nr == rhs.nr && lhs.nth == rhs.nth && lhs.nz == rhs.nz
        && lhs.shift_i_cells == rhs.shift_i_cells
        && lhs.shift_j_cells == rhs.shift_j_cells
        && lhs.physical_i_begin == rhs.physical_i_begin
        && lhs.physical_i_end == rhs.physical_i_end
        && lhs.physical_j_begin == rhs.physical_j_begin
        && lhs.physical_j_end == rhs.physical_j_end
        && lhs.destination_i_begin == rhs.destination_i_begin
        && lhs.destination_i_end == rhs.destination_i_end
        && lhs.destination_j_begin == rhs.destination_j_begin
        && lhs.destination_j_end == rhs.destination_j_end
        && lhs.source_i_begin == rhs.source_i_begin
        && lhs.source_i_end == rhs.source_i_end
        && lhs.source_j_begin == rhs.source_j_begin
        && lhs.source_j_end == rhs.source_j_end
        && lhs.retained_cells == rhs.retained_cells
        && lhs.leading_cells == rhs.leading_cells
        && lhs.trailing_cells == rhs.trailing_cells;
}

bool safe_transaction_id(const std::string& value) noexcept
{
    return !value.empty() && value.size() <= 128
        && std::all_of(value.begin(), value.end(), [](unsigned char c) {
               return (c >= 'a' && c <= 'z')
                   || (c >= 'A' && c <= 'Z')
                   || (c >= '0' && c <= '9') || c == '-' || c == '_';
           });
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs,
                            const char* label)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
        throw std::overflow_error(std::string(label) + " size overflow");
    return lhs * rhs;
}

std::uint64_t fnv1a64(std::string_view text) noexcept
{
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char byte : text)
    {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

struct ArtifactIntegrity
{
    std::string relative_path;
    std::uint64_t bytes = 0;
    std::uint64_t fnv1a64 = 0;
};

ArtifactIntegrity hash_artifact(
    const std::filesystem::path& root,
    const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error(
            "cannot hash retiring swath artifact: " + path.string());
    ArtifactIntegrity result;
    result.relative_path =
        std::filesystem::relative(path, root).generic_string();
    result.fnv1a64 = UINT64_C(14695981039346656037);
    std::array<char, 64 * 1024> buffer{};
    while (input)
    {
        input.read(buffer.data(),
                   static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        for (std::streamsize index = 0; index < count; ++index)
        {
            result.fnv1a64 ^=
                static_cast<unsigned char>(buffer[static_cast<std::size_t>(index)]);
            result.fnv1a64 *= UINT64_C(1099511628211);
        }
        result.bytes += static_cast<std::uint64_t>(count);
    }
    if (!input.eof())
        throw std::runtime_error(
            "failed hashing retiring swath artifact: " + path.string());
    return result;
}

std::vector<ArtifactIntegrity> inventory_artifacts(
    const std::filesystem::path& root)
{
    std::vector<ArtifactIntegrity> artifacts;
    std::error_code ec;
    for (std::filesystem::recursive_directory_iterator iterator(root, ec), end;
         iterator != end; iterator.increment(ec))
    {
        if (ec)
            throw std::runtime_error(
                "failed listing retiring swath artifacts: " + ec.message());
        if (iterator->is_regular_file())
            artifacts.push_back(hash_artifact(root, iterator->path()));
    }
    if (ec)
        throw std::runtime_error(
            "failed listing retiring swath artifacts: " + ec.message());
    std::sort(artifacts.begin(), artifacts.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.relative_path < rhs.relative_path;
              });
    return artifacts;
}

void add_tile(std::vector<RetiringSwathTile>& tiles,
              int i_begin, int i_end, int j_begin, int j_end)
{
    if (i_begin >= i_end || j_begin >= j_end)
        return;
    tiles.push_back({tiles.size(), i_begin, i_end, j_begin, j_end});
}

std::string tile_name(std::size_t ordinal)
{
    std::ostringstream out;
    out << "tile_" << std::setfill('0') << std::setw(2) << ordinal;
    return out.str();
}

OutputStepManifestCompression compression_for(
    const OutputConfig& config, const std::string& field_id)
{
    OutputStepManifestCompression compression;
    compression.field_id = field_id;
    compression.mode = config.zfp_mode;
    compression.frame_role = "keyframe";
    compression.sparse_threshold =
        is_sparse_eligible(field_id) ? config.zfp_sparse_threshold : 0.0;
    compression.float16_prefilter =
        config.zfp_float16_prefilter && is_float16_eligible(field_id);
    if (config.zfp_mode == "accuracy")
    {
        compression.mode_parameter =
            effective_zfp_accuracy_tolerance(config, field_id);
        compression.mode_parameter_units = "absolute_field_units";
        compression.error_contract =
            "zfp_absolute_accuracy_before_prefilters";
    }
    else if (config.zfp_mode == "precision")
    {
        compression.mode_parameter = config.zfp_precision_bits;
        compression.mode_parameter_units = "significant_binary_bits";
        compression.error_contract = "visualization_only_no_absolute_bound";
    }
    else
    {
        compression.mode_parameter = config.zfp_rate_bps;
        compression.mode_parameter_units = "bits_per_value";
        compression.error_contract = "visualization_only_no_absolute_bound";
    }
    return compression;
}

std::string build_tile_manifest(
    const RetiringSwathPublication& publication,
    const RetiringSwathTile& tile,
    const ExportSnapshot& snapshot)
{
    OutputStepManifestContext context;
    context.step_index = static_cast<int>(tile.ordinal);
    context.simulation_time_s = publication.simulation_time_s;
    context.step_dir = snapshot.step_dir.filename().string();
    context.publication_generation = publication.publication_generation;
    context.field_brick_hierarchies = true;
    context.storage_format = output_format_name(publication.output.format);
    context.data_product_kind = "retiring_swath_tile";
    context.domain_recycling_transaction_id = publication.transaction_id;
    context.domain_recycling_tile_ordinal = static_cast<int>(tile.ordinal);
    context.storage_includes_model_halos = false;
    context.grid.nr = tile.nx();
    context.grid.nth = tile.ny();
    context.grid.nz = publication.plan.nz;
    context.grid.coordinate_system = "cartesian";
    context.grid.staggering = publication.native_staggering;
    context.grid.dr_m = publication.dx_m;
    context.grid.dy_m = publication.dy_m;
    context.grid.dtheta_rad = publication.dy_m;
    context.grid.dz_m = publication.dz_m;
    context.grid.domain_motion_mode = publication.domain_motion_mode;
    context.grid.horizontal_frame_contract =
        snapshot.grid.horizontal_frame_contract;
    context.grid.grid_origin_t0_earth_x_m =
        snapshot.grid.grid_origin_t0_earth_x_m;
    context.grid.grid_origin_t0_earth_y_m =
        snapshot.grid.grid_origin_t0_earth_y_m;
    context.grid.grid_origin_earth_x_m =
        snapshot.grid.grid_origin_earth_x_m;
    context.grid.grid_origin_earth_y_m =
        snapshot.grid.grid_origin_earth_y_m;
    context.grid.frame_velocity_u_ms = snapshot.grid.frame_velocity_u_ms;
    context.grid.frame_velocity_v_ms = snapshot.grid.frame_velocity_v_ms;
    for (const auto& field : snapshot.fields)
    {
        context.published_fields.push_back(field.name);
        if (publication.output.format == OutputFormat::zfp && field.is_3d)
            context.compression.push_back(
                compression_for(publication.output, field.name));
    }
    std::sort(context.published_fields.begin(),
              context.published_fields.end());
    return build_output_step_manifest_json(context);
}

std::string build_event_manifest(
    const RetiringSwathPublication& publication,
    const std::vector<RetiringSwathTile>& tiles,
    const tmv::domain::HorizontalPoint& origin,
    const std::vector<ArtifactIntegrity>& artifacts)
{
    const auto& plan = publication.plan;
    std::ostringstream out;
    out << "{\n"
        << "  \"schema\": \"nimbus.domain_recycling_swath\",\n"
        << "  \"schema_version\": 1,\n"
        << "  \"publication_phase\": \"captured_precommit\",\n"
        << "  \"transaction_id\": \""
        << tmv::strutil::json_escape(publication.transaction_id) << "\",\n"
        << "  \"publication_generation\": \""
        << tmv::strutil::json_escape(publication.publication_generation)
        << "\",\n"
        << "  \"transaction_sequence\": "
        << publication.transaction_sequence << ",\n"
        << "  \"previous_committed_chain_fnv1a64\": \""
        << std::hex << std::setfill('0') << std::setw(16)
        << publication.previous_committed_chain_fnv1a64 << std::dec
        << "\",\n"
        << "  \"transaction_chain_algorithm\": \""
        << tmv::domain::domain_recycling_transaction_provenance_algorithm_id()
        << "\",\n"
        << "  \"algorithm\": \"" << retiring_swath_algorithm_id()
        << "\",\n"
        << "  \"simulation_time_s\": " << std::setprecision(17)
        << publication.simulation_time_s << ",\n"
        << "  \"storage_format\": \""
        << output_format_name(publication.output.format) << "\",\n"
        << "  \"pre_shift_grid_origin_earth_m\": [" << origin.x_m
        << ", " << origin.y_m << "],\n"
        << "  \"spacing_m\": [" << publication.dx_m << ", "
        << publication.dy_m << ", " << publication.dz_m << "],\n"
        << "  \"shift_cells\": [" << plan.shift_i_cells << ", "
        << plan.shift_j_cells << "],\n"
        << "  \"physical_index_bounds\": {\"i\": ["
        << plan.physical_i_begin << ", " << plan.physical_i_end
        << "], \"j\": [" << plan.physical_j_begin << ", "
        << plan.physical_j_end << "]},\n"
        << "  \"retained_source_index_bounds\": {\"i\": ["
        << plan.source_i_begin << ", " << plan.source_i_end
        << "], \"j\": [" << plan.source_j_begin << ", "
        << plan.source_j_end << "]},\n"
        << "  \"tiles\": [";
    for (std::size_t index = 0; index < tiles.size(); ++index)
    {
        const auto& tile = tiles[index];
        if (index != 0)
            out << ",";
        out << "\n    {\"ordinal\": " << tile.ordinal
            << ", \"directory\": \"" << tile_name(tile.ordinal)
            << "\", \"source_index_bounds\": {\"i\": ["
            << tile.i_begin << ", " << tile.i_end << "], \"j\": ["
            << tile.j_begin << ", " << tile.j_end
            << "]}, \"earth_bounds_m\": {\"lower\": ["
            << origin.x_m + tile.i_begin * publication.dx_m << ", "
            << origin.y_m + tile.j_begin * publication.dy_m
            << "], \"upper\": ["
            << origin.x_m + tile.i_end * publication.dx_m << ", "
            << origin.y_m + tile.j_end * publication.dy_m << "]}}";
    }
    out << "\n  ],\n  \"artifacts\": [";
    for (std::size_t index = 0; index < artifacts.size(); ++index)
    {
        const auto& artifact = artifacts[index];
        if (index != 0)
            out << ",";
        out << "\n    {\"path\": \""
            << tmv::strutil::json_escape(artifact.relative_path)
            << "\", \"bytes\": " << artifact.bytes
            << ", \"fnv1a64\": \"" << std::hex << std::setfill('0')
            << std::setw(16) << artifact.fnv1a64 << std::dec << "\"}";
    }
    out << "\n  ]\n}\n";
    return out.str();
}

} // namespace

std::vector<RetiringSwathTile> plan_retiring_scalar_swath_tiles(
    const tmv::domain::WholeCellShiftPlan& plan)
{
    const auto canonical =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            plan.nr, plan.nth, plan.nz, plan.shift_i_cells,
            plan.shift_j_cells, plan.physical_i_begin,
            plan.physical_i_end, plan.physical_j_begin,
            plan.physical_j_end);
    if (!plans_equal(canonical, plan))
        throw std::invalid_argument(
            "retiring swath requires a canonical whole-cell plan");

    std::vector<RetiringSwathTile> tiles;
    tiles.reserve(4);
    add_tile(tiles, plan.physical_i_begin, plan.source_i_begin,
             plan.physical_j_begin, plan.physical_j_end);
    add_tile(tiles, plan.source_i_end, plan.physical_i_end,
             plan.physical_j_begin, plan.physical_j_end);
    add_tile(tiles, plan.source_i_begin, plan.source_i_end,
             plan.physical_j_begin, plan.source_j_begin);
    add_tile(tiles, plan.source_i_begin, plan.source_i_end,
             plan.source_j_end, plan.physical_j_end);
    if (tiles.empty())
        throw std::logic_error("non-zero shift produced no retiring swath");
    return tiles;
}

bool publish_retiring_swath_precommit(
    const RetiringSwathPublication& publication,
    RetiringSwathPublicationResult& result,
    std::string& error)
{
    result = {};
    error.clear();
    std::filesystem::path staging;
    bool renamed = false;
    try
    {
        if (publication.swath_root.empty()
            || !safe_transaction_id(publication.transaction_id)
            || !safe_transaction_id(publication.publication_generation)
            || publication.transaction_sequence == 0)
            throw std::invalid_argument(
                "retiring swath requires a root, safe transaction/generation, and positive sequence");
        if (!std::isfinite(publication.simulation_time_s)
            || !std::isfinite(publication.dx_m) || publication.dx_m <= 0.0
            || !std::isfinite(publication.dy_m) || publication.dy_m <= 0.0
            || !std::isfinite(publication.dz_m) || publication.dz_m <= 0.0)
            throw std::invalid_argument(
                "retiring swath requires finite time and positive spacing");
        publication.pre_shift_frame.validate();
        if (publication.domain_motion_mode.empty()
            || (publication.native_staggering != "c_grid"
                && publication.native_staggering != "collocated"))
            throw std::invalid_argument(
                "retiring swath has invalid frame or staggering metadata");
        if (publication.output.format != OutputFormat::npy_3d
            && publication.output.format != OutputFormat::netcdf
            && publication.output.format != OutputFormat::zfp)
            throw std::invalid_argument(
                "retiring swath requires native 3-D, netCDF, or ZFP output");

        const auto tiles = plan_retiring_scalar_swath_tiles(publication.plan);
        const std::size_t retired_columns =
            publication.plan.trailing_cells
            / static_cast<std::size_t>(publication.plan.nz);
        if (retired_columns == 0
            || checked_product(retired_columns,
                               static_cast<std::size_t>(publication.plan.nz),
                               "retiring volume")
                   != publication.plan.trailing_cells)
            throw std::invalid_argument(
                "retiring swath plan has inconsistent trailing counts");
        std::size_t tiled_columns = 0;
        for (const auto& tile : tiles)
        {
            if (tile.columns()
                > std::numeric_limits<std::size_t>::max() - tiled_columns)
                throw std::overflow_error(
                    "retiring swath tile-column report overflow");
            tiled_columns += tile.columns();
        }
        if (tiled_columns != retired_columns)
            throw std::logic_error(
                "retiring swath tiles do not exactly cover retirement");

        std::set<std::string> names;
        std::set<std::string> supplied;
        if (publication.fields.empty())
            throw std::invalid_argument("retiring swath field set is empty");
        for (const auto& field : publication.fields)
        {
            if (field.field_id.empty() || field.packed_values == nullptr
                || !names.insert(field.field_id).second
                || find_output_field(field.field_id) == nullptr)
                throw std::invalid_argument(
                    "retiring swath contains an unknown, duplicate, or unbound field");
            supplied.insert(field.field_id);
            const std::size_t expected = field.is_3d
                ? publication.plan.trailing_cells : retired_columns;
            if (field.packed_values->size() != expected)
                throw std::invalid_argument(
                    "retiring swath field payload size does not match the plan");
            if (std::any_of(field.packed_values->begin(),
                            field.packed_values->end(),
                            [](float value) { return !std::isfinite(value); }))
                throw std::invalid_argument(
                    "retiring swath field contains a non-finite value");
        }
        const std::set<std::string> configured(
            publication.output.resolved_fields.begin(),
            publication.output.resolved_fields.end());
        if (configured.empty() || configured != supplied)
            throw std::invalid_argument(
                "retiring swath fields must exactly match resolved output fields");
        if (publication.output.format == OutputFormat::zfp)
        {
            const auto zfp_errors =
                validate_qualified_zfp_history(publication.output);
            if (!zfp_errors.empty())
                throw std::invalid_argument(
                    "retiring swath ZFP contract is not qualified: "
                    + zfp_errors.front());
            if (std::any_of(publication.fields.begin(),
                            publication.fields.end(),
                            [](const RetiringSwathField& field) {
                                return !field.is_3d;
                            }))
                throw std::invalid_argument(
                    "qualified retiring ZFP swaths require 3-D fields");
        }

        std::error_code ec;
        std::filesystem::create_directories(publication.swath_root, ec);
        if (ec)
            throw std::runtime_error(
                "failed to create swath root: " + ec.message());
        const auto final_directory = publication.swath_root
            / ("transaction_" + publication.transaction_id);
        staging = final_directory.string() + ".tmp";
        if (std::filesystem::exists(final_directory))
            throw std::invalid_argument(
                "retiring swath transaction already exists");
        std::filesystem::remove_all(staging, ec);
        ec.clear();
        std::filesystem::create_directories(staging, ec);
        if (ec)
            throw std::runtime_error(
                "failed to create swath staging directory: " + ec.message());

        OutputConfig writer_config = publication.output;
        writer_config.async_io = false;
        // Spatially disjoint tiles have no valid temporal delta parent.
        writer_config.zfp_keyframe_interval = 0;
        writer_config.zfp_predictive_delta = false;
        AsyncOutputWriter writer(writer_config);
        const auto origin = publication.pre_shift_frame.earth_origin_at(
            publication.simulation_time_s);

        for (const auto& tile : tiles)
        {
            ExportSnapshot snapshot;
            snapshot.export_index = static_cast<int>(tile.ordinal);
            snapshot.simulation_time_s = publication.simulation_time_s;
            snapshot.step_dir = staging / tile_name(tile.ordinal);
            snapshot.publication_generation = publication.publication_generation;
            snapshot.data_product_kind = "retiring_swath_tile";
            snapshot.domain_recycling_transaction_id =
                publication.transaction_id;
            snapshot.domain_recycling_tile_ordinal =
                static_cast<int>(tile.ordinal);
            snapshot.grid.nx = tile.nx();
            snapshot.grid.ny = tile.ny();
            snapshot.grid.nz = publication.plan.nz;
            snapshot.grid.dx_m = publication.dx_m;
            snapshot.grid.dy_m = publication.dy_m;
            snapshot.grid.dz_m = publication.dz_m;
            snapshot.grid.coordinate_system = "cartesian";
            snapshot.grid.native_staggering =
                publication.native_staggering;
            snapshot.grid.domain_motion_mode =
                publication.domain_motion_mode;
            snapshot.grid.grid_origin_t0_earth_x_m =
                publication.pre_shift_frame.earth_origin_at_t0_m.x_m
                + tile.i_begin * publication.dx_m;
            snapshot.grid.grid_origin_t0_earth_y_m =
                publication.pre_shift_frame.earth_origin_at_t0_m.y_m
                + tile.j_begin * publication.dy_m;
            snapshot.grid.grid_origin_earth_x_m = origin.x_m
                + tile.i_begin * publication.dx_m;
            snapshot.grid.grid_origin_earth_y_m = origin.y_m
                + tile.j_begin * publication.dy_m;
            snapshot.grid.frame_velocity_u_ms =
                publication.pre_shift_frame.velocity_ms.u_ms;
            snapshot.grid.frame_velocity_v_ms =
                publication.pre_shift_frame.velocity_ms.v_ms;

            snapshot.fields.reserve(publication.fields.size());
            for (const auto& source : publication.fields)
            {
                const std::size_t levels = source.is_3d
                    ? static_cast<std::size_t>(publication.plan.nz) : 1U;
                FieldSnapshotEntry entry;
                entry.name = source.field_id;
                entry.dim0 = tile.nx();
                entry.dim1 = tile.ny();
                entry.dim2 = source.is_3d ? publication.plan.nz : 0;
                entry.is_3d = source.is_3d;
                entry.data.resize(checked_product(tile.columns(), levels,
                                                  "retiring tile field"));
                std::size_t packed_column = 0;
                for (int i = publication.plan.physical_i_begin;
                     i < publication.plan.physical_i_end; ++i)
                    for (int j = publication.plan.physical_j_begin;
                         j < publication.plan.physical_j_end; ++j)
                    {
                        if (i >= publication.plan.source_i_begin
                            && i < publication.plan.source_i_end
                            && j >= publication.plan.source_j_begin
                            && j < publication.plan.source_j_end)
                            continue;
                        if (i >= tile.i_begin && i < tile.i_end
                            && j >= tile.j_begin && j < tile.j_end)
                        {
                            const std::size_t local_column =
                                static_cast<std::size_t>(i - tile.i_begin)
                                    * static_cast<std::size_t>(tile.ny())
                                + static_cast<std::size_t>(j - tile.j_begin);
                            std::copy_n(
                                source.packed_values->data()
                                    + packed_column * levels,
                                levels,
                                entry.data.data() + local_column * levels);
                        }
                        ++packed_column;
                    }
                if (packed_column != retired_columns)
                    throw std::logic_error(
                        "retiring swath packing traversal disagrees with plan");
                snapshot.fields.push_back(std::move(entry));
            }
            snapshot.manifest_json =
                build_tile_manifest(publication, tile, snapshot);
            if (!writer.submit(std::move(snapshot)))
                throw std::runtime_error(writer.error_message());
        }

        const auto artifacts = inventory_artifacts(staging);
        if (artifacts.empty())
            throw std::logic_error(
                "retiring swath publication produced no artifacts");
        const std::string event_manifest =
            build_event_manifest(publication, tiles, origin, artifacts);
        std::string publish_error;
        if (!publish_output_json_atomically(
                staging / "event.json", event_manifest, publish_error))
            throw std::runtime_error(publish_error);
        if (!sync_directory_to_disk(staging, publish_error))
            throw std::runtime_error(publish_error);
        std::filesystem::rename(staging, final_directory, ec);
        if (ec)
            throw std::runtime_error(
                "failed to publish swath transaction: " + ec.message());
        renamed = true;
        if (!sync_directory_to_disk(publication.swath_root, publish_error))
            throw std::runtime_error(publish_error);

        std::size_t bytes = 0;
        for (const auto& entry :
             std::filesystem::recursive_directory_iterator(final_directory))
            if (entry.is_regular_file())
            {
                const std::uintmax_t size = entry.file_size();
                if (size > std::numeric_limits<std::size_t>::max() - bytes)
                    throw std::overflow_error(
                        "retiring swath byte report overflow");
                bytes += static_cast<std::size_t>(size);
            }
        result.event_directory = final_directory;
        result.tiles_published = tiles.size();
        result.fields_published = checked_product(
            tiles.size(), publication.fields.size(),
            "retiring swath field report");
        const std::size_t volume_values = checked_product(
            publication.plan.trailing_cells,
            std::count_if(publication.fields.begin(), publication.fields.end(),
                          [](const RetiringSwathField& field) {
                              return field.is_3d;
                          }),
            "retiring swath value report");
        const std::size_t surface_values = checked_product(
            retired_columns,
            std::count_if(publication.fields.begin(), publication.fields.end(),
                          [](const RetiringSwathField& field) {
                              return !field.is_3d;
                          }),
            "retiring swath surface report");
        if (surface_values
            > std::numeric_limits<std::size_t>::max() - volume_values)
            throw std::overflow_error(
                "retiring swath total value report overflow");
        result.values_published = volume_values + surface_values;
        result.bytes_published = bytes;
        result.event_manifest_bytes =
            static_cast<std::uint64_t>(event_manifest.size());
        result.event_manifest_fnv1a64 = fnv1a64(event_manifest);
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        if (!renamed && !staging.empty())
        {
            std::error_code cleanup_error;
            std::filesystem::remove_all(staging, cleanup_error);
        }
        return false;
    }
}

const char* retiring_swath_algorithm_id() noexcept
{
    return "canonical_rectangular_earth_swath_capture_v1";
}

} // namespace nimbus::output

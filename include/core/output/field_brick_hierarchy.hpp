#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/field/field_snapshot.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "util/string_utils.hpp"

namespace nimbus::output
{

inline constexpr std::array<int, 3> k_field_brick_shape{64, 64, 32};

inline int field_brick_workers(unsigned requested, std::size_t tasks)
{
#ifdef _OPENMP
    if (!omp_in_parallel())
        return static_cast<int>(std::max<std::size_t>(1,
            std::min({std::size_t(requested), std::size_t(omp_get_max_threads()), tasks})));
#endif
    return 1;
}

inline std::optional<double> zfp_decoded_extrema_padding(
    const std::string& mode,
    double mode_parameter,
    double sparse_threshold,
    bool float16_prefilter)
{
    if (mode != "accuracy" || !std::isfinite(mode_parameter) || mode_parameter < 0.0 ||
        !std::isfinite(sparse_threshold) || sparse_threshold < 0.0 || float16_prefilter)
        return std::nullopt;
    return mode_parameter + sparse_threshold;
}

inline void brick_hash_u64(nimbus::crypto::Sha256& hash, std::uint64_t value)
{
    std::uint8_t bytes[8];
    for (unsigned index = 0; index < 8; ++index)
        bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
    hash.update(bytes, sizeof(bytes));
}

inline void brick_hash_u32(nimbus::crypto::Sha256& hash, std::uint32_t value)
{
    std::uint8_t bytes[4];
    for (unsigned index = 0; index < 4; ++index)
        bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
    hash.update(bytes, sizeof(bytes));
}

struct FieldBrickSummary
{
    std::size_t finite_count = 0;
    std::size_t nonfinite_count = 0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    std::string content_sha256;
};

struct FieldBrickLodPayloadIdentity
{
    std::string path;
    std::string content_sha256;
    struct Geometry
    {
        std::array<std::string, 4> paths;
        std::array<std::string, 4> content_sha256;
    };
    std::optional<Geometry> geometry;
};

struct FieldBrickLodGeometry
{
    std::vector<double> x_m;
    std::vector<double> y_m;
    std::vector<double> z_m;
    std::vector<double> volume_m3;
};

using FieldBrickLodPublisher = std::function<FieldBrickLodPayloadIdentity(
    int,
    const std::array<int, 3>&,
    const std::vector<float>&,
    const FieldBrickLodGeometry*)>;

struct FieldBrickRecord
{
    std::array<int, 3> index{};
    std::array<int, 3> origin{};
    std::array<int, 3> extent{};
    std::array<double, 6> physical_bounds{};
    FieldBrickSummary summary;
};

struct FieldBrickLevelRecord
{
    int level = 0;
    std::array<int, 3> stride{1, 1, 1};
    std::array<int, 3> shape{};
    std::optional<FieldBrickLodPayloadIdentity> payload;
    std::vector<FieldBrickRecord> bricks;
};

inline FieldBrickSummary summarize_field_brick_data(
    const std::string& field_identity,
    const float* data,
    int dim1,
    int dim2,
    const std::array<int, 3>& origin,
    const std::array<int, 3>& extent)
{
    nimbus::crypto::Sha256 hash;
    static constexpr char contract[] = "nimbus.field-brick.values.v1";
    hash.update(contract, sizeof(contract) - 1);
    brick_hash_u64(hash, field_identity.size());
    hash.update(field_identity.data(), field_identity.size());
    for (int value : origin) brick_hash_u64(hash, static_cast<std::uint64_t>(value));
    for (int value : extent) brick_hash_u64(hash, static_cast<std::uint64_t>(value));
    FieldBrickSummary summary;
    std::array<std::uint8_t, 4096> hash_buffer{};
    std::size_t hash_buffer_size = 0;
    const auto flush_hash = [&]()
    {
        if (hash_buffer_size == 0) return;
        hash.update(hash_buffer.data(), hash_buffer_size);
        hash_buffer_size = 0;
    };
    for (int i = 0; i < extent[0]; ++i)
        for (int j = 0; j < extent[1]; ++j)
            for (int k = 0; k < extent[2]; ++k)
            {
                const std::size_t index =
                    (static_cast<std::size_t>(origin[0] + i) * dim1 +
                     static_cast<std::size_t>(origin[1] + j)) * dim2 +
                    static_cast<std::size_t>(origin[2] + k);
                std::uint32_t bits = 0;
                const float value = data[index];
                std::memcpy(&bits, &value, sizeof(bits));
                if (hash_buffer_size + 4 > hash_buffer.size()) flush_hash();
                for (unsigned byte = 0; byte < 4; ++byte)
                    hash_buffer[hash_buffer_size++] =
                        static_cast<std::uint8_t>(bits >> (8U * byte));
                if (!std::isfinite(value))
                {
                    ++summary.nonfinite_count;
                    continue;
                }
                ++summary.finite_count;
                summary.minimum = std::min(summary.minimum, value);
                summary.maximum = std::max(summary.maximum, value);
            }
    flush_hash();
    summary.content_sha256 = hash.hex_digest();
    return summary;
}

inline FieldBrickSummary summarize_field_brick(
    const FieldSnapshotEntry& entry,
    const std::array<int, 3>& origin,
    const std::array<int, 3>& extent)
{
    return summarize_field_brick_data(
        entry.name, entry.values().data(), entry.dim1, entry.dim2, origin, extent);
}

inline std::array<double, 6> field_brick_physical_bounds(
    const ExportGridMetadata& grid,
    const std::array<int, 3>& origin,
    const std::array<int, 3>& extent)
{
    std::array<double, 6> bounds{
        grid.grid_origin_earth_x_m + origin[0] * grid.dx_m,
        grid.grid_origin_earth_x_m + (origin[0] + extent[0]) * grid.dx_m,
        grid.grid_origin_earth_y_m + origin[1] * grid.dy_m,
        grid.grid_origin_earth_y_m + (origin[1] + extent[1]) * grid.dy_m,
        origin[2] * grid.dz_m,
        (origin[2] + extent[2]) * grid.dz_m,
    };
    if (!grid.terrain_following) return bounds;

    bounds[4] = grid.terrain_min_z_m;
    bounds[5] = grid.terrain_max_z_m;
    const std::size_t expected = static_cast<std::size_t>(grid.nx) * grid.ny * (grid.nz + 1);
    if (grid.physical_z_interface_m.size() != expected) return bounds;
    bounds[4] = std::numeric_limits<double>::infinity();
    bounds[5] = -std::numeric_limits<double>::infinity();
    for (int i = origin[0]; i < origin[0] + extent[0]; ++i)
        for (int j = origin[1]; j < origin[1] + extent[1]; ++j)
            for (int k : {origin[2], origin[2] + extent[2]})
            {
                const std::size_t index =
                    (static_cast<std::size_t>(i) * grid.ny + j) * (grid.nz + 1) + k;
                const double value = grid.physical_z_interface_m[index];
                bounds[4] = std::min(bounds[4], value);
                bounds[5] = std::max(bounds[5], value);
            }
    if (!std::isfinite(bounds[4]) || !std::isfinite(bounds[5]) || !(bounds[5] > bounds[4]))
        throw std::invalid_argument("field brick terrain bounds are invalid");
    return bounds;
}

struct FieldBrickReducedLevel
{
    std::vector<float> values;
    std::vector<double> support_weights;
    FieldBrickLodGeometry geometry;
};

inline FieldBrickReducedLevel reduce_field_brick_level(
    const float* input,
    const double* input_support_weights,
    const FieldBrickLodGeometry* input_geometry,
    const std::array<int, 3>& input_shape,
    const std::array<int, 3>& input_offset,
    const std::array<int, 3>& source_shape,
    const ExportGridMetadata* initial_terrain_grid = nullptr,
    unsigned worker_limit = 1)
{
    const std::array<int, 3> output_shape{
        (source_shape[0] + 1) / 2,
        (source_shape[1] + 1) / 2,
        (source_shape[2] + 1) / 2,
    };
    const std::size_t output_count = static_cast<std::size_t>(output_shape[0]) *
        static_cast<std::size_t>(output_shape[1]) * static_cast<std::size_t>(output_shape[2]);
    FieldBrickReducedLevel output;
    output.values.resize(output_count);
    output.support_weights.resize(output_count);
    const bool terrain_geometry = initial_terrain_grid != nullptr ||
        (input_geometry != nullptr && !input_geometry->volume_m3.empty());
    if (terrain_geometry)
    {
        output.geometry.x_m.resize(output_count);
        output.geometry.y_m.resize(output_count);
        output.geometry.z_m.resize(output_count);
        output.geometry.volume_m3.resize(output_count);
    }
    [[maybe_unused]] const int workers = field_brick_workers(worker_limit, output_count >= 65536 ? output_shape[0] : 1);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(workers) if(workers > 1)
#endif
    for (int i = 0; i < output_shape[0]; ++i)
        for (int j = 0; j < output_shape[1]; ++j)
            for (int k = 0; k < output_shape[2]; ++k)
            {
                double sum = 0.0;
                double weight_sum = 0.0;
                double x_sum = 0.0;
                double y_sum = 0.0;
                double z_sum = 0.0;
                bool scalar_invalid = false;
                bool geometry_invalid = false;
                for (int di = 0; di < 2 && 2 * i + di < source_shape[0]; ++di)
                    for (int dj = 0; dj < 2 && 2 * j + dj < source_shape[1]; ++dj)
                        for (int dk = 0; dk < 2 && 2 * k + dk < source_shape[2]; ++dk)
                        {
                            const int si = input_offset[0] + 2 * i + di;
                            const int sj = input_offset[1] + 2 * j + dj;
                            const int sk = input_offset[2] + 2 * k + dk;
                            const std::size_t source_index =
                                (static_cast<std::size_t>(si) * input_shape[1] + sj) *
                                input_shape[2] + sk;
                            const float value = input[source_index];
                            double support = input_support_weights
                                ? input_support_weights[source_index]
                                : 1.0;
                            double x = 0.0;
                            double y = 0.0;
                            double z = 0.0;
                            if (initial_terrain_grid)
                            {
                                const std::size_t center_index = source_index;
                                const std::size_t interface_index =
                                    (static_cast<std::size_t>(si) * initial_terrain_grid->ny + sj) *
                                    (initial_terrain_grid->nz + 1) + sk;
                                const double lower = initial_terrain_grid->physical_z_interface_m[interface_index];
                                const double upper = initial_terrain_grid->physical_z_interface_m[interface_index + 1];
                                support = initial_terrain_grid->dx_m * initial_terrain_grid->dy_m *
                                    (upper - lower);
                                x = initial_terrain_grid->grid_origin_earth_x_m +
                                    (static_cast<double>(si) + 0.5) * initial_terrain_grid->dx_m;
                                y = initial_terrain_grid->grid_origin_earth_y_m +
                                    (static_cast<double>(sj) + 0.5) * initial_terrain_grid->dy_m;
                                z = initial_terrain_grid->physical_z_center_m[center_index];
                            }
                            else if (terrain_geometry)
                            {
                                x = input_geometry->x_m[source_index];
                                y = input_geometry->y_m[source_index];
                                z = input_geometry->z_m[source_index];
                            }
                            if (!std::isfinite(support) || !(support > 0.0) ||
                                (terrain_geometry && (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))))
                            {
                                scalar_invalid = true;
                                geometry_invalid = true;
                                continue;
                            }
                            weight_sum += support;
                            if (terrain_geometry)
                            {
                                x_sum += x * support;
                                y_sum += y * support;
                                z_sum += z * support;
                            }
                            if (!std::isfinite(value)) scalar_invalid = true;
                            else sum += static_cast<double>(value) * support;
                        }
                const std::size_t output_index =
                    (static_cast<std::size_t>(i) * output_shape[1] + j) * output_shape[2] + k;
                output.values[output_index] = scalar_invalid || !(weight_sum > 0.0)
                    ? std::numeric_limits<float>::quiet_NaN()
                    : static_cast<float>(sum / weight_sum);
                output.support_weights[output_index] = weight_sum;
                if (terrain_geometry)
                {
                    const double missing = std::numeric_limits<double>::quiet_NaN();
                    output.geometry.x_m[output_index] = geometry_invalid || !(weight_sum > 0.0)
                        ? missing : x_sum / weight_sum;
                    output.geometry.y_m[output_index] = geometry_invalid || !(weight_sum > 0.0)
                        ? missing : y_sum / weight_sum;
                    output.geometry.z_m[output_index] = geometry_invalid || !(weight_sum > 0.0)
                        ? missing : z_sum / weight_sum;
                    output.geometry.volume_m3[output_index] = geometry_invalid ? 0.0 : weight_sum;
                }
            }
    return output;
}

inline FieldBrickLevelRecord summarize_field_brick_level(
    const std::string& field_name,
    const float* data,
    const std::array<int, 3>& data_shape,
    const std::array<int, 3>& data_offset,
    const ExportGridMetadata& grid,
    const std::array<int, 3>& source_offset,
    const std::array<int, 3>& source_shape,
    int level,
    const std::array<int, 3>& stride,
    const std::array<int, 3>& level_shape,
    unsigned worker_limit = 1)
{
    FieldBrickLevelRecord record;
    record.level = level;
    record.stride = stride;
    record.shape = level_shape;
    const std::array<int, 3> counts{
        (level_shape[0] + k_field_brick_shape[0] - 1) / k_field_brick_shape[0],
        (level_shape[1] + k_field_brick_shape[1] - 1) / k_field_brick_shape[1],
        (level_shape[2] + k_field_brick_shape[2] - 1) / k_field_brick_shape[2],
    };
    const std::size_t brick_count = static_cast<std::size_t>(counts[0]) * counts[1] * counts[2];
    record.bricks.resize(brick_count);
    std::vector<std::exception_ptr> failures(brick_count);
    [[maybe_unused]] const int workers = field_brick_workers(worker_limit, brick_count >= 8 ? brick_count : 1);
    const std::string identity = level == 0
        ? field_name
        : field_name + ".lod" + std::to_string(level);
#ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(static) num_threads(workers) if(workers > 1)
#endif
    for (int bi = 0; bi < counts[0]; ++bi)
        for (int bj = 0; bj < counts[1]; ++bj)
            for (int bk = 0; bk < counts[2]; ++bk)
            {
                const std::size_t index = (static_cast<std::size_t>(bi) * counts[1] + bj) * counts[2] + bk;
                try
                {
                    FieldBrickRecord brick;
                    brick.index = {bi, bj, bk};
                    brick.origin = {
                        bi * k_field_brick_shape[0],
                        bj * k_field_brick_shape[1],
                        bk * k_field_brick_shape[2],
                    };
                    brick.extent = {
                        std::min(k_field_brick_shape[0], level_shape[0] - brick.origin[0]),
                        std::min(k_field_brick_shape[1], level_shape[1] - brick.origin[1]),
                        std::min(k_field_brick_shape[2], level_shape[2] - brick.origin[2]),
                    };
                    const std::array<int, 3> summary_origin{
                        data_offset[0] + brick.origin[0],
                        data_offset[1] + brick.origin[1],
                        data_offset[2] + brick.origin[2],
                    };
                    brick.summary = summarize_field_brick_data(
                        identity, data, data_shape[1], data_shape[2], summary_origin, brick.extent);
                    const std::array<int, 3> physical_origin{
                        source_offset[0] + brick.origin[0] * stride[0],
                        source_offset[1] + brick.origin[1] * stride[1],
                        source_offset[2] + brick.origin[2] * stride[2],
                    };
                    const std::array<int, 3> physical_extent{
                        std::min(source_shape[0] - brick.origin[0] * stride[0], brick.extent[0] * stride[0]),
                        std::min(source_shape[1] - brick.origin[1] * stride[1], brick.extent[1] * stride[1]),
                        std::min(source_shape[2] - brick.origin[2] * stride[2], brick.extent[2] * stride[2]),
                    };
                    brick.physical_bounds = field_brick_physical_bounds(
                        grid, physical_origin, physical_extent);
                    record.bricks[index] = std::move(brick);
                }
                catch (...)
                {
                    failures[index] = std::current_exception();
                }
            }
    for (const auto& failure : failures)
        if (failure) std::rethrow_exception(failure);
    return record;
}

inline std::string build_field_brick_hierarchy_json(
    const FieldSnapshotEntry& entry,
    const ExportGridMetadata& grid,
    const std::string& publication_generation,
    bool culling_safe,
    bool storage_includes_model_halos = true,
    double extrema_absolute_padding = 0.0,
    const FieldBrickLodPublisher& publish_lod = {},
    unsigned worker_limit = 1)
{
    if (!entry.is_3d || entry.dim0 <= 0 || entry.dim1 <= 0 || entry.dim2 <= 0 ||
        entry.values().size() != static_cast<std::size_t>(entry.dim0) * entry.dim1 * entry.dim2)
        throw std::invalid_argument("field brick hierarchy requires one complete 3-D field");
    if (grid.nx != entry.dim0 || grid.ny != entry.dim1 || grid.nz != entry.dim2 ||
        !(grid.dx_m > 0.0) || !(grid.dy_m > 0.0) || !(grid.dz_m > 0.0) ||
        publication_generation.empty() || !std::isfinite(extrema_absolute_padding) ||
        extrema_absolute_padding < 0.0 || (!culling_safe && extrema_absolute_padding != 0.0))
        throw std::invalid_argument("field brick hierarchy requires complete grid and generation metadata");
    if (grid.terrain_following)
    {
        const std::size_t centers = static_cast<std::size_t>(grid.nx) * grid.ny * grid.nz;
        const std::size_t interfaces = static_cast<std::size_t>(grid.nx) * grid.ny * (grid.nz + 1);
        if (grid.physical_z_center_m.size() != centers ||
            grid.physical_z_interface_m.size() != interfaces)
            throw std::invalid_argument(
                "terrain field brick hierarchy requires complete physical geometry");
    }

    StaggerType staggering;
    if (grid.native_staggering == "c_grid")
        staggering = StaggerType::CGrid;
    else if (grid.native_staggering == "collocated")
        staggering = StaggerType::Collocated;
    else
        throw std::invalid_argument("field brick hierarchy requires canonical staggering");
    const PhysicalCellDomain domain = storage_includes_model_halos
        ? physical_scalar_domain(entry.dim0, entry.dim1, entry.dim2, staggering)
        : PhysicalCellDomain{0, entry.dim0, 0, entry.dim1, 0, entry.dim2};
    const std::array<int, 3> source_offset{domain.i_begin, domain.j_begin, domain.k_begin};
    const std::array<int, 3> shape{domain.nx(), domain.ny(), domain.nz()};
    if (std::any_of(shape.begin(), shape.end(), [](int count) { return count <= 0; }))
        throw std::invalid_argument("field brick hierarchy physical domain is empty");

    std::vector<FieldBrickLevelRecord> levels;
    levels.push_back(summarize_field_brick_level(
        entry.name,
        entry.values().data(),
        {entry.dim0, entry.dim1, entry.dim2},
        source_offset,
        grid,
        source_offset,
        shape,
        0,
        {1, 1, 1},
        shape, worker_limit));
    if (publish_lod)
    {
        std::array<int, 3> stride{1, 1, 1};
        std::array<int, 3> current_shape = shape;
        const float* current_data = entry.values().data();
        std::array<int, 3> current_data_shape{entry.dim0, entry.dim1, entry.dim2};
        std::array<int, 3> current_offset = source_offset;
        FieldBrickReducedLevel current_owned;
        while (current_shape[0] > k_field_brick_shape[0] ||
               current_shape[1] > k_field_brick_shape[1] ||
               current_shape[2] > k_field_brick_shape[2])
        {
            auto next = reduce_field_brick_level(
                current_data,
                current_owned.support_weights.empty() ? nullptr : current_owned.support_weights.data(),
                current_owned.geometry.volume_m3.empty() ? nullptr : &current_owned.geometry,
                current_data_shape,
                current_offset,
                current_shape,
                current_owned.support_weights.empty() && grid.terrain_following ? &grid : nullptr, worker_limit);
            const std::array<int, 3> next_shape{
                (current_shape[0] + 1) / 2,
                (current_shape[1] + 1) / 2,
                (current_shape[2] + 1) / 2,
            };
            for (int axis = 0; axis < 3; ++axis) stride[axis] *= 2;
            const int level_number = static_cast<int>(levels.size());
            auto level = summarize_field_brick_level(
                entry.name,
                next.values.data(),
                next_shape,
                {0, 0, 0},
                grid,
                source_offset,
                shape,
                level_number,
                stride,
                next_shape, worker_limit);
            level.payload = publish_lod(
                level_number,
                next_shape,
                next.values,
                next.geometry.volume_m3.empty() ? nullptr : &next.geometry);
            if (level.payload->path.empty() || level.payload->content_sha256.size() != 64)
                throw std::runtime_error("field brick LOD publisher returned an invalid identity");
            levels.push_back(std::move(level));
            current_owned = std::move(next);
            current_data = current_owned.values.data();
            current_data_shape = next_shape;
            current_shape = next_shape;
            current_offset = {0, 0, 0};
        }
    }

    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n  \"schema\": \"nimbus.field-brick-hierarchy\",\n"
        << "  \"schemaVersion\": " << (publish_lod ? (grid.terrain_following ? 3 : 2) : 1)
        << ",\n  \"field\": \""
        << tmv::strutil::json_escape(entry.name) << "\",\n"
        << "  \"publicationGeneration\": \""
        << tmv::strutil::json_escape(publication_generation) << "\",\n"
        << "  \"sourceShape\": [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "],\n"
        << "  \"sourceIndexOffset\": [" << source_offset[0] << ", "
        << source_offset[1] << ", " << source_offset[2] << "],\n"
        << "  \"sourceDtype\": \"float32\",\n  \"sampleLocation\": \"cell_center\",\n"
        << "  \"extremaSemantics\": \""
        << (culling_safe
                ? (extrema_absolute_padding > 0.0 ? "decoded_value_bounds" : "stored_values")
                : "pre_storage_source") << "\",\n"
        << "  \"extremaAbsolutePadding\": " << extrema_absolute_padding << ",\n"
        << "  \"cullingSafe\": " << (culling_safe ? "true" : "false") << ",\n"
        << "  \"levels\": [";
    for (std::size_t level_index = 0; level_index < levels.size(); ++level_index)
    {
        const auto& level = levels[level_index];
        if (level_index > 0) out << ',';
        out << "\n  {\n    \"level\": " << level.level << ",\n"
            << "    \"stride\": [" << level.stride[0] << ", " << level.stride[1]
            << ", " << level.stride[2] << "],\n"
            << "    \"shape\": [" << level.shape[0] << ", " << level.shape[1]
            << ", " << level.shape[2] << "],\n"
            << "    \"brickShape\": [" << k_field_brick_shape[0] << ", "
            << k_field_brick_shape[1] << ", " << k_field_brick_shape[2] << "],\n"
            << "    \"source\": {\"kind\": \""
            << (level.level == 0 ? "primary_field" : "npy_lod") << "\"";
        if (level.payload)
        {
            out << ", \"path\": \"" << tmv::strutil::json_escape(level.payload->path)
                << "\", \"contentSha256\": \"" << level.payload->content_sha256
                << "\", \"aggregation\": \"recursive_cell_mean_2x2x2\"";
            if (level.payload->geometry)
            {
                static constexpr std::array<const char*, 4> names{
                    "xM", "yM", "zM", "volumeM3"};
                out << ", \"geometry\": {\"kind\": \"physical_volume_centroids\"";
                for (std::size_t component = 0; component < names.size(); ++component)
                    out << ", \"" << names[component] << "\": {\"path\": \""
                        << tmv::strutil::json_escape(level.payload->geometry->paths[component])
                        << "\", \"contentSha256\": \""
                        << level.payload->geometry->content_sha256[component] << "\"}";
                out << '}';
            }
        }
        out << "},\n    \"bricks\": [\n";
        bool first = true;
        for (const auto& brick : level.bricks)
        {
                if (!first) out << ",\n";
                first = false;
                out << "      {\"id\": \"l" << level.level << ':'
                    << brick.index[0] << ':' << brick.index[1] << ':' << brick.index[2]
                    << "\", \"index\": [" << brick.index[0] << ", " << brick.index[1]
                    << ", " << brick.index[2]
                    << "], \"origin\": [" << brick.origin[0] << ", " << brick.origin[1]
                    << ", " << brick.origin[2]
                    << "], \"extent\": [" << brick.extent[0] << ", " << brick.extent[1]
                    << ", " << brick.extent[2]
                    << "], \"scalarRange\": {\"minimum\": ";
                if (brick.summary.finite_count == 0) out << "null";
                else out << static_cast<double>(brick.summary.minimum) - extrema_absolute_padding;
                out << ", \"maximum\": ";
                if (brick.summary.finite_count == 0) out << "null";
                else out << static_cast<double>(brick.summary.maximum) + extrema_absolute_padding;
                out << ", \"finiteCount\": " << brick.summary.finite_count
                    << ", \"nonFiniteCount\": " << brick.summary.nonfinite_count << "}, "
                    << "\"physicalBoundsM\": {\"lower\": [" << brick.physical_bounds[0]
                    << ", " << brick.physical_bounds[2] << ", " << brick.physical_bounds[4]
                    << "], \"upper\": [" << brick.physical_bounds[1] << ", "
                    << brick.physical_bounds[3] << ", " << brick.physical_bounds[5] << "]}, "
                    << "\"contentSha256\": \""
                    << brick.summary.content_sha256 << "\"}";
        }
        out << "\n    ]\n  }";
    }
    out << "\n  ]\n}\n";
    return out.str();
}

} // namespace nimbus::output

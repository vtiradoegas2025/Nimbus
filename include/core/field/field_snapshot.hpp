#pragma once

/**
 * @file field_snapshot.hpp
 * @brief Snapshot of field data for async handoff to background writer.
 *
 * An ExportSnapshot captures all field data needed for one export step.
 * Background snapshots own their fields. Synchronous publication may borrow
 * immutable core fields until submit() returns; the background writer rejects
 * borrowed buffers. Derived fields retain ownership of computation buffers.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/infra/sha256.hpp"

class FieldValuesView
{
public:
    FieldValuesView(const float* values, std::size_t count) : values_(values), count_(count) {}
    const float* data() const noexcept { return values_; }
    std::size_t size() const noexcept { return count_; }
    const float* begin() const noexcept { return values_; }
    const float* end() const noexcept { return count_ == 0 ? values_ : values_ + count_; }
    const float& operator[](std::size_t index) const noexcept { return values_[index]; }

private:
    const float* values_;
    std::size_t count_;
};

struct FieldSnapshotEntry
{
    FieldSnapshotEntry() = default;
    FieldSnapshotEntry(std::string field_name, std::vector<float> field_data,
                       int first, int second = 0, int third = 0, bool volume = true)
        : name(std::move(field_name)), data(std::move(field_data)),
          dim0(first), dim1(second), dim2(third), is_3d(volume) {}

    std::string        name;
    std::vector<float> data;      ///< Contiguous buffer (NR*NTH*NZ for 3D, NR*NZ for 2D)
    int                dim0  = 0; ///< First dimension
    int                dim1  = 0; ///< Second dimension (0 if 2D)
    int                dim2  = 0; ///< Third dimension (0 if 2D)
    bool               is_3d = true;

    FieldValuesView values() const noexcept
    {
        return borrowed_ ? FieldValuesView(borrowed_data_, borrowed_count_)
                         : FieldValuesView(data.data(), data.size());
    }

    bool is_borrowed() const noexcept { return borrowed_; }

    // The caller keeps this storage alive and unchanged through synchronous
    // publication, including visualization sidecars and manifest commit.
    void borrow_for_synchronous_write(const float* values, std::size_t count)
    {
        if (!data.empty() || (count != 0 && values == nullptr))
            throw std::invalid_argument("invalid borrowed output field storage");
        borrowed_data_ = values;
        borrowed_count_ = count;
        borrowed_ = true;
    }

private:
    const float* borrowed_data_ = nullptr;
    std::size_t borrowed_count_ = 0;
    bool borrowed_ = false;
};

/**
 * @brief Cartesian grid metadata required by self-describing history formats.
 *
 * NPY payloads obtain this information from the adjacent manifest.  Container
 * formats such as netCDF must also carry it inside the file so that a copied
 * history product remains scientifically interpretable.
 */
struct ExportGridMetadata
{
    int         nx                = 0;
    int         ny                = 0;
    int         nz                = 0;
    double      dx_m              = 0.0;
    double      dy_m              = 0.0;
    double      dz_m              = 0.0;
    std::string coordinate_system = "cartesian";
    std::string native_staggering = "collocated";
    std::string domain_motion_mode = "stationary";
    std::string horizontal_frame_contract =
        "constant_translation_earth_wind_grid_transport_v1";
    // Earth coordinates of the grid's lower horizontal face at t0 and at
    // this publication time. Prognostic u/v remain Earth-relative.
    double grid_origin_t0_earth_x_m = 0.0;
    double grid_origin_t0_earth_y_m = 0.0;
    double grid_origin_earth_x_m = 0.0;
    double grid_origin_earth_y_m = 0.0;
    double frame_velocity_u_ms = 0.0;
    double frame_velocity_v_ms = 0.0;
    bool        terrain_following = false;
    // Row-major [x][y][z] physical heights. Interface storage has nz+1.
    // These are lossless geometry metadata and are never ZFP-compressed.
    std::vector<double> physical_z_center_m;
    std::vector<double> physical_z_interface_m;
    // NPY/ZFP geometry is an immutable series-level artifact, but every
    // field-brick hierarchy snapshot retains the coordinates it needs to
    // publish physical bounds and terrain-aware LOD geometry.
    std::string terrain_geometry_sha256;
    std::string terrain_geometry_publication_generation;
    double terrain_min_z_m = 0.0;
    double terrain_max_z_m = 0.0;
};

inline std::string terrain_geometry_content_sha256(
    int nx, int ny, int nz,
    const std::vector<double>& centers,
    const std::vector<double>& interfaces)
{
    nimbus::crypto::Sha256 hash;
    static constexpr char contract[] = "nimbus.terrain-geometry.v1";
    hash.update(contract, sizeof(contract) - 1);
    const auto update_u64_le = [&](std::uint64_t value)
    {
        std::uint8_t bytes[8];
        for (unsigned index = 0; index < 8; ++index)
            bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
        hash.update(bytes, sizeof(bytes));
    };
    update_u64_le(static_cast<std::uint64_t>(nx));
    update_u64_le(static_cast<std::uint64_t>(ny));
    update_u64_le(static_cast<std::uint64_t>(nz));
    const auto update_values = [&](const std::vector<double>& values)
    {
        for (double value : values)
        {
            std::uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(bits));
            update_u64_le(bits);
        }
    };
    update_values(centers);
    update_values(interfaces);
    return hash.hex_digest();
}

inline std::string terrain_geometry_content_sha256(const ExportGridMetadata& grid)
{
    return terrain_geometry_content_sha256(
        grid.nx, grid.ny, grid.nz,
        grid.physical_z_center_m, grid.physical_z_interface_m);
}

/**
 * @brief Complete snapshot of one export timestep.
 */
struct ExportSnapshot
{
    int                   export_index      = 0;
    double                simulation_time_s = 0.0;
    std::filesystem::path step_dir;
    std::string           publication_generation;

    /// Publication semantics. Ordinary cadence output keeps the defaults;
    /// Recycling retiring tiles use a distinct non-restart history product.
    std::string data_product_kind = "history";
    std::string domain_recycling_transaction_id;
    int         domain_recycling_tile_ordinal = -1;

    /// Scalar-cell grid represented by every field in this history frame.
    ExportGridMetadata grid;

    /// All field entries to write
    std::vector<FieldSnapshotEntry> fields;

    /// Pre-serialized manifest JSON
    std::string manifest_json;

    /// Pre-serialized validation report JSON
    std::string           validation_json;
    std::filesystem::path validation_path;

    /// Approximate total bytes of field data
    std::size_t total_bytes() const
    {
        std::size_t sum = 0;
        for (const auto& f : fields)
        {
            sum += f.values().size() * sizeof(float);
        }
        sum += grid.physical_z_center_m.size() * sizeof(double);
        sum += grid.physical_z_interface_m.size() * sizeof(double);
        return sum;
    }

    std::size_t owned_bytes() const
    {
        std::size_t sum = 0;
        for (const auto& f : fields)
            sum += f.data.size() * sizeof(float);
        sum += grid.physical_z_center_m.size() * sizeof(double);
        sum += grid.physical_z_interface_m.size() * sizeof(double);
        return sum;
    }
};

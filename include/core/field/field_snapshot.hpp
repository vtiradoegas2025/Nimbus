#pragma once

/**
 * @file field_snapshot.hpp
 * @brief Snapshot of field data for async handoff to background writer.
 *
 * An ExportSnapshot captures all field data needed for one export step.
 * Core fields are deep-copied from the global Field3D objects (which the
 * simulation thread will mutate immediately). Derived fields are moved
 * from computation buffers. The snapshot is self-contained and can be
 * serialized to disk on a background thread without any shared state.
 */

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief One field's worth of export data.
 */
struct FieldSnapshotEntry
{
    std::string name;
    std::vector<float> data;   ///< Contiguous buffer (NR*NTH*NZ for 3D, NR*NZ for 2D)
    int dim0 = 0;              ///< First dimension
    int dim1 = 0;              ///< Second dimension (0 if 2D)
    int dim2 = 0;              ///< Third dimension (0 if 2D)
    bool is_3d = true;
};

/**
 * @brief Complete snapshot of one export timestep.
 */
struct ExportSnapshot
{
    int export_index = 0;
    double simulation_time_s = 0.0;
    std::filesystem::path step_dir;

    /// All field entries to write
    std::vector<FieldSnapshotEntry> fields;

    /// Pre-serialized manifest JSON
    std::string manifest_json;

    /// Pre-serialized validation report JSON
    std::string validation_json;
    std::filesystem::path validation_path;

    /// Approximate total bytes of field data
    std::size_t total_bytes() const
    {
        std::size_t sum = 0;
        for (const auto& f : fields)
        {
            sum += f.data.size() * sizeof(float);
        }
        return sum;
    }
};

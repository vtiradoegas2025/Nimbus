#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/field/field_snapshot.hpp"
#include "core/output/npy_writer.hpp"
#include "core/output/output_config.hpp"

/**
 * @file derived_volume_writer.hpp
 * @brief Shared output dispatch for derived 3D diagnostic volumes.
 */

inline void write_or_snapshot_derived_volume_fields(
    std::unordered_map<std::string, std::vector<float>>& derived_3d_bufs,
    ExportSnapshot& snapshot,
    bool use_async,
    OutputFormat format,
    int nr,
    int nth,
    int nz,
    const std::string& step_path)
{
    for (auto& [name, buf] : derived_3d_bufs)
    {
        if (use_async)
        {
            FieldSnapshotEntry entry;
            entry.name = name;
            entry.dim0 = nr;
            entry.dim1 = nth;
            entry.dim2 = nz;
            entry.is_3d = true;
            entry.data = std::move(buf);
            snapshot.fields.push_back(std::move(entry));
        }
        else if (format == OutputFormat::csv)
        {
            csv::write_3d(buf.data(), nr, nth, nz, step_path + "/" + name + ".csv");
        }
        else
        {
            npy::write_3d(buf.data(), nr, nth, nz, step_path + "/" + name + ".npy");
        }
    }
}

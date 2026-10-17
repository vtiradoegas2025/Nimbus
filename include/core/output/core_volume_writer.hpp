#pragma once

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/field/field3d.hpp"
#include "core/field/field_snapshot.hpp"
#include "core/output/npy_writer.hpp"
#include "core/output/output_config.hpp"

/**
 * @file core_volume_writer.hpp
 * @brief Shared output dispatch for core 3D fields.
 */

template <typename FieldPtr>
struct CoreVolumeFieldBinding
{
    const char* name;
    FieldPtr field;
};

template <typename FieldPtr>
inline void write_or_snapshot_core_volume_fields(
    const std::vector<CoreVolumeFieldBinding<FieldPtr>>& bindings,
    const std::unordered_set<std::string>& active_fields,
    ExportSnapshot& snapshot,
    bool use_async,
    OutputFormat format,
    int nr,
    int nth,
    int nz,
    const std::string& step_path,
    bool borrow_synchronous_fields = false)
{
    for (const auto& binding : bindings)
    {
        if (!active_fields.count(binding.name)) continue;

        if (use_async)
        {
            FieldSnapshotEntry entry;
            entry.name = binding.name;
            entry.dim0 = nr;
            entry.dim1 = nth;
            entry.dim2 = nz;
            entry.is_3d = true;
            if (borrow_synchronous_fields)
                entry.borrow_for_synchronous_write(binding.field->data(), binding.field->size());
            else
                entry.data.assign(binding.field->data(),
                                  binding.field->data() + binding.field->size());
            snapshot.fields.push_back(std::move(entry));
        }
        else if (format == OutputFormat::csv)
        {
            csv::write_field3d(*binding.field, step_path + "/" + binding.name + ".csv");
        }
        else
        {
            npy::write_field3d(*binding.field, step_path + "/" + binding.name + ".npy");
        }
    }
}

template <typename FieldPtr>
inline void write_core_field_slices(
    const std::vector<CoreVolumeFieldBinding<FieldPtr>>& bindings,
    const std::unordered_set<std::string>& active_fields,
    int theta_index,
    const std::string& step_path)
{
    const std::string base_path = step_path + "/th" + std::to_string(theta_index);
    for (const auto& binding : bindings)
    {
        if (active_fields.count(binding.name))
        {
            npy::write_field_slice(*binding.field,
                                   theta_index,
                                   base_path + "_" + binding.name + ".npy");
        }
    }
}

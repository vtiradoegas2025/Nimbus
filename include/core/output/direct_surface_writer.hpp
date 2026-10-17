#pragma once

#include <cstddef>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/field/field_snapshot.hpp"
#include "core/output/field_registry.hpp"
#include "core/output/npy_writer.hpp"

/**
 * @file direct_surface_writer.hpp
 * @brief Shared output dispatch for direct 2D surface diagnostic fields.
 */

template <typename SurfaceBufferResolver>
inline bool write_or_snapshot_direct_surface_fields(
    const std::unordered_set<std::string>& active_fields,
    SurfaceBufferResolver&& surface_buffer_for_field,
    std::string& error_message,
    ExportSnapshot& snapshot,
    bool use_async,
    int nr,
    int nth,
    const std::string& step_path,
    bool borrow_synchronous_fields = false)
{
    const std::size_t expected_size =
        static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth);

    for (const auto& field : output_field_registry())
    {
        if (!output_field_is_direct_surface_2d(field)) continue;
        if (!active_fields.count(field.field_id)) continue;

        const std::vector<float>* data =
            surface_buffer_for_field(field.field_id);
        if (data == nullptr)
        {
            error_message =
                "missing direct-surface output buffer for field='" +
                std::string(field.field_id) + "'";
            return false;
        }
        if (data->size() != expected_size)
        {
            error_message =
                "direct-surface output buffer has the wrong size for field='" +
                std::string(field.field_id) + "'";
            return false;
        }

        if (use_async)
        {
            FieldSnapshotEntry entry;
            entry.name = field.field_id;
            entry.dim0 = nr;
            entry.dim1 = nth;
            entry.dim2 = 0;
            entry.is_3d = false;
            if (borrow_synchronous_fields)
                entry.borrow_for_synchronous_write(data->data(), data->size());
            else
                entry.data = *data;
            snapshot.fields.push_back(std::move(entry));
        }
        else
        {
            if (!npy::write_2d(data->data(), nr, nth,
                               step_path + "/" + field.field_id + ".npy"))
            {
                error_message =
                    "failed to write direct-surface output field='" +
                    std::string(field.field_id) + "'";
                return false;
            }
        }
    }
    return true;
}

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/field/field_snapshot.hpp"
#include "core/output/field_registry.hpp"
#include "core/output/npy_writer.hpp"

/**
 * @file tranche_writer.hpp
 * @brief Shared output dispatch for surface, column, and radar tranche products.
 */

struct RegisteredDiagnosticProduct
{
    const char* field_id;
    std::function<bool(int, std::vector<float>&)> compute_slice;
};

template <typename SliceValidator>
inline bool write_or_snapshot_column_tranche_fields(
    const std::vector<RegisteredDiagnosticProduct>& products,
    const std::unordered_set<std::string>& active_fields,
    SliceValidator&& validate_slice,
    std::string& error_message,
    ExportSnapshot& snapshot,
    bool use_async,
    int nr,
    int nth,
    int frame_export_index,
    const std::string& step_path)
{
    const std::size_t column_map_size =
        static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth);
    struct SelectedProduct
    {
        const RegisteredDiagnosticProduct* product;
        std::vector<float> map;
    };
    std::vector<SelectedProduct> selected;
    for (const auto& product : products)
    {
        if (!active_fields.count(product.field_id)) continue;
        const auto* field = find_output_field(product.field_id);
        if (!field || !output_field_is_tranche_product(*field))
        {
            error_message = "missing tranche output registry metadata for field='" + std::string(product.field_id) + "'";
            return false;
        }
        if (field->kind == OutputFieldKind::ColumnTranche || field->kind == OutputFieldKind::SurfaceTranche)
            selected.push_back({&product, std::vector<float>(column_map_size, std::numeric_limits<float>::quiet_NaN())});
    }
    std::vector<float> column_scratch;
    for (int th = 0; th < nth; ++th)
    {
        for (auto& entry : selected)
        {
            const auto& product = *entry.product;
            if (!product.compute_slice(th, column_scratch))
            {
                error_message = "failed to compute column diagnostic field='" + std::string(product.field_id)
                    + "' step=" + std::to_string(frame_export_index) + " theta=" + std::to_string(th) + ": " + error_message;
                return false;
            }
            if (column_scratch.size() < static_cast<std::size_t>(nr))
            {
                error_message = "column diagnostic field='" + std::string(product.field_id) + "' produced too few samples";
                return false;
            }
            for (int i = 0; i < nr; ++i)
                entry.map[static_cast<std::size_t>(i) * nth + th] = column_scratch[static_cast<std::size_t>(i)];
        }
    }
    for (auto& selected_product : selected)
    {
        const auto& product = *selected_product.product;
        auto& column_map = selected_product.map;
        if (!validate_slice(product.field_id, column_map, 0)) return false;
        if (use_async)
        {
            FieldSnapshotEntry entry;
            entry.name = product.field_id;
            entry.dim0 = nr;
            entry.dim1 = nth;
            entry.dim2 = 0;
            entry.is_3d = false;
            entry.data = std::move(column_map);
            snapshot.fields.push_back(std::move(entry));
        }
        else if (!npy::write_2d(column_map.data(), nr, nth, step_path + "/" + product.field_id + ".npy"))
        {
            error_message = "failed to write planar tranche output field='" + std::string(product.field_id)
                + "' step=" + std::to_string(frame_export_index);
            return false;
        }
    }
    return true;
}

template <typename SliceValidator>
inline bool write_tranche_slice_fields(
    const std::vector<RegisteredDiagnosticProduct>& products,
    const std::unordered_set<std::string>& active_fields,
    SliceValidator&& validate_slice,
    std::string& error_message,
    bool use_3d,
    int theta_index,
    int frame_export_index,
    int nr,
    int nz,
    const std::string& step_path,
    std::vector<float>& tranche_slice_buffer)
{
    for (const auto& product : products)
    {
        if (!active_fields.count(product.field_id))
        {
            continue;
        }
        const OutputFieldDescriptor* field = find_output_field(product.field_id);
        if (field == nullptr || !output_field_is_tranche_product(*field))
        {
            error_message =
                "missing tranche output registry metadata for field='" +
                std::string(product.field_id) + "'";
            return false;
        }
        // In 3D mode planar column and surface products have already been
        // assembled across every theta slice. Recomputing them here produces
        // no output and can dominate diagnostic export cost.
        if (use_3d && (field->kind == OutputFieldKind::ColumnTranche ||
                       field->kind == OutputFieldKind::SurfaceTranche))
        {
            continue;
        }
        if (!product.compute_slice(theta_index, tranche_slice_buffer))
        {
            error_message =
                "failed to compute tranche diagnostic field='" +
                std::string(product.field_id) +
                "' step=" + std::to_string(frame_export_index) +
                " theta=" + std::to_string(theta_index);
            return false;
        }
        if (!use_3d &&
            (field->kind == OutputFieldKind::ColumnTranche ||
             field->kind == OutputFieldKind::SurfaceTranche) &&
            tranche_slice_buffer.size() == static_cast<std::size_t>(nr))
        {
            tranche_slice_buffer.resize(
                static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz));
            for (int k = nz - 1; k >= 0; --k)
                for (int i = 0; i < nr; ++i)
                    tranche_slice_buffer[
                        static_cast<std::size_t>(k) *
                            static_cast<std::size_t>(nr) +
                        static_cast<std::size_t>(i)] =
                        tranche_slice_buffer[static_cast<std::size_t>(i)];
        }
        if (!validate_slice(product.field_id, tranche_slice_buffer, theta_index))
        {
            return false;
        }
        if (!use_3d)
        {
            const std::string base_path = step_path + "/th" + std::to_string(theta_index);
            if (!npy::write_2d(tranche_slice_buffer,
                               nz,
                               nr,
                               base_path + "_" + field->suffix + ".npy"))
            {
                error_message =
                    "failed to write tranche slice field='" +
                    std::string(product.field_id) + "' step=" +
                    std::to_string(frame_export_index) + " theta=" +
                    std::to_string(theta_index);
                return false;
            }
        }
    }
    return true;
}

#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/output/field_registry.hpp"
#include "core/output/npy_writer.hpp"

/**
 * @file derived_slice_scratch.hpp
 * @brief Derived-volume slice buffers and registry bindings for output writers.
 */

struct DerivedSliceBinding
{
    const char* field_id;
    std::vector<float>* values;
};

struct DerivedSliceScratch
{
    explicit DerivedSliceScratch(std::size_t slice_size)
        : slice_size_(slice_size), discard_slice_(slice_size, 0.0f)
    {
        for (const auto& field : output_field_registry())
        {
            if (output_field_is_derived_volume(field))
            {
                requested_fields_.insert(field.field_id);
            }
        }
        allocate_dependency_closure();
    }

    DerivedSliceScratch(
        std::size_t slice_size,
        const std::unordered_set<std::string>& active_fields)
        : slice_size_(slice_size), discard_slice_(slice_size, 0.0f)
    {
        for (const auto& field : output_field_registry())
        {
            if (output_field_is_derived_volume(field)
                && active_fields.count(field.field_id))
            {
                requested_fields_.insert(field.field_id);
            }
        }
        allocate_dependency_closure();
    }

    std::vector<float>& writable_buffer(std::string_view field_id)
    {
        const auto it = buffers_.find(std::string(field_id));
        return it == buffers_.end() ? discard_slice_ : it->second;
    }

    std::vector<float>* buffer(std::string_view field_id)
    {
        const auto it = buffers_.find(std::string(field_id));
        return it == buffers_.end() ? nullptr : &it->second;
    }

    bool requested(std::string_view field_id) const
    {
        return requested_fields_.count(std::string(field_id)) != 0;
    }

    bool has_buffer(std::string_view field_id) const
    {
        return buffers_.count(std::string(field_id)) != 0;
    }

    bool requests_only_simple_pointwise_products() const
    {
        if (requested_fields_.empty()) return false;
        static const std::unordered_set<std::string> simple_products = {
            "temperature",
            "theta_prime",
            "total_condensate",
            "cloud_condensate",
            "cloud_liquid_condensate",
            "cloud_frozen_condensate",
            "reflectivity_dbz",
            "vorticity_magnitude",
            "buoyancy",
        };
        for (const std::string& field_id : requested_fields_)
        {
            if (!simple_products.count(field_id)) return false;
        }
        return true;
    }

    std::size_t allocated_array_count() const noexcept
    {
        return buffers_.size() + (discard_slice_.empty() ? 0U : 1U);
    }

    std::size_t requested_field_count() const noexcept
    {
        return requested_fields_.size();
    }

private:
    void allocate(std::string_view field_id)
    {
        buffers_.try_emplace(std::string(field_id), slice_size_, 0.0f);
    }

    void allocate_dependency_closure()
    {
        for (const std::string& field_id : requested_fields_)
        {
            allocate(field_id);
        }
        const auto requested = [&](std::string_view field_id) {
            return requested_fields_.count(std::string(field_id)) != 0;
        };
        if (requested("theta_e") || requested("theta_w")) allocate("temperature");
        if (requested("theta_v")) allocate("total_condensate");
        if (requested("forward_trajectories") || requested("backward_trajectories"))
            allocate("trajectory_paths");
        if (requested("parcel_buoyancy_trajectory")) allocate("buoyancy");
        if (requested("vorticity_trajectory")) allocate("vorticity_magnitude");
        if (requested("rhi_slice")) allocate("reflectivity_dbz");
        if (requested("hodograph_aligned_cross_section"))
            allocate("storm_relative_winds");
    }

    std::size_t slice_size_ = 0;
    std::unordered_set<std::string> requested_fields_;
    std::unordered_map<std::string, std::vector<float>> buffers_;
    std::vector<float> discard_slice_;
};

inline int derived_slice_scratch_array_count(
    const std::unordered_set<std::string>& active_fields)
{
    DerivedSliceScratch scratch(1, active_fields);
    return scratch.requested_field_count() == 0
        ? 0
        : static_cast<int>(scratch.allocated_array_count());
}

inline std::vector<float>* derived_slice_buffer(DerivedSliceScratch& scratch, std::string_view field_id)
{
    return scratch.buffer(field_id);
}

inline bool build_derived_slice_bindings(DerivedSliceScratch& scratch,
                                         std::vector<DerivedSliceBinding>& bindings,
                                         std::string& error_message)
{
    bindings.clear();
    for (const auto& field : output_field_registry())
    {
        if (!output_field_is_derived_volume(field)) continue;

        if (!scratch.requested(field.field_id)) continue;
        std::vector<float>* values = derived_slice_buffer(scratch, field.field_id);
        if (values == nullptr)
        {
            error_message =
                "missing derived-slice scratch buffer for field='" +
                std::string(field.field_id) + "'";
            return false;
        }
        bindings.push_back({field.field_id, values});
    }
    return true;
}

inline void prepare_derived_volume_output_buffers(
    const std::unordered_set<std::string>& active_fields,
    bool use_3d,
    std::size_t volume_size,
    std::unordered_map<std::string, std::vector<float>>& buffers)
{
    buffers.clear();
    if (!use_3d)
    {
        return;
    }

    for (const auto& field : output_field_registry())
    {
        if (output_field_is_derived_volume(field) &&
            active_fields.count(field.field_id))
        {
            buffers[field.field_id].resize(volume_size, 0.0f);
        }
    }
}

inline void scatter_derived_slice_to_volume(const std::vector<float>& slice,
                                            std::vector<float>& volume,
                                            int theta_index,
                                            int nr,
                                            int nth,
                                            int nz)
{
    // slice is [k * nr + i], volume is [i * nth * nz + j * nz + k].
    for (int k = 0; k < nz; ++k)
    {
        for (int i = 0; i < nr; ++i)
        {
            volume[static_cast<std::size_t>(i) * static_cast<std::size_t>(nth) *
                   static_cast<std::size_t>(nz) +
                   static_cast<std::size_t>(theta_index) * static_cast<std::size_t>(nz) +
                   static_cast<std::size_t>(k)] =
                slice[static_cast<std::size_t>(k) * static_cast<std::size_t>(nr) +
                      static_cast<std::size_t>(i)];
        }
    }
}

template <typename SliceValidator>
inline bool write_or_scatter_derived_slice_fields(
    const DerivedSliceBinding* bindings,
    std::size_t binding_count,
    const std::unordered_set<std::string>& active_fields,
    std::unordered_map<std::string, std::vector<float>>& derived_3d_bufs,
    SliceValidator&& validate_slice,
    bool use_3d,
    int theta_index,
    int nr,
    int nth,
    int nz,
    const std::string& step_path)
{
    for (std::size_t b = 0; b < binding_count; ++b)
    {
        const DerivedSliceBinding& binding = bindings[b];
        if (!validate_slice(binding.field_id, *binding.values))
        {
            return false;
        }
    }

    if (use_3d)
    {
        for (std::size_t b = 0; b < binding_count; ++b)
        {
            const DerivedSliceBinding& binding = bindings[b];
            auto it = derived_3d_bufs.find(binding.field_id);
            if (it != derived_3d_bufs.end())
            {
                scatter_derived_slice_to_volume(
                    *binding.values,
                    it->second,
                    theta_index,
                    nr,
                    nth,
                    nz);
            }
        }
        return true;
    }

    const std::string base_path = step_path + "/th" + std::to_string(theta_index);
    for (std::size_t b = 0; b < binding_count; ++b)
    {
        const DerivedSliceBinding& binding = bindings[b];
        if (active_fields.count(binding.field_id))
        {
            npy::write_2d(*binding.values,
                          nz,
                          nr,
                          base_path + "_" + binding.field_id + ".npy");
        }
    }
    return true;
}

inline bool active_output_fields_need_derived_slices(
    const std::unordered_set<std::string>& active_fields)
{
    for (const auto& field : output_field_registry())
    {
        if ((output_field_is_derived_volume(field) ||
             output_field_is_tranche_product(field)) &&
            active_fields.count(field.field_id))
        {
            return true;
        }
    }
    return false;
}

inline bool active_output_fields_need_derived_volume_slices(
    const std::unordered_set<std::string>& active_fields)
{
    for (const auto& field : output_field_registry())
    {
        if (output_field_is_derived_volume(field)
            && active_fields.count(field.field_id))
        {
            return true;
        }
    }
    return false;
}

inline bool active_output_fields_need_tranche_products(
    const std::unordered_set<std::string>& active_fields)
{
    for (const auto& field : output_field_registry())
    {
        if (output_field_is_tranche_product(field)
            && active_fields.count(field.field_id))
        {
            return true;
        }
    }
    return false;
}

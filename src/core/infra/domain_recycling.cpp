#include "core/infra/domain_recycling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace tmv::domain
{
namespace
{

std::size_t checked_product(std::size_t a, std::size_t b, const char* label)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error(std::string(label) + " size overflow");
    return a * b;
}

std::size_t checked_sum(std::size_t a, std::size_t b, const char* label)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error(std::string(label) + " size overflow");
    return a + b;
}

bool plans_equal(const WholeCellShiftPlan& a, const WholeCellShiftPlan& b) noexcept
{
    return a.nr == b.nr && a.nth == b.nth && a.nz == b.nz &&
           a.shift_i_cells == b.shift_i_cells &&
           a.shift_j_cells == b.shift_j_cells &&
           a.physical_i_begin == b.physical_i_begin &&
           a.physical_i_end == b.physical_i_end &&
           a.physical_j_begin == b.physical_j_begin &&
           a.physical_j_end == b.physical_j_end &&
           a.destination_i_begin == b.destination_i_begin &&
           a.destination_i_end == b.destination_i_end &&
           a.destination_j_begin == b.destination_j_begin &&
           a.destination_j_end == b.destination_j_end &&
           a.source_i_begin == b.source_i_begin &&
           a.source_i_end == b.source_i_end &&
           a.source_j_begin == b.source_j_begin &&
           a.source_j_end == b.source_j_end &&
           a.retained_cells == b.retained_cells &&
           a.leading_cells == b.leading_cells &&
           a.trailing_cells == b.trailing_cells;
}

void validate_binding(const WholeCellShiftPlan& owner_plan,
                      const HomogeneousFieldShiftBinding& binding,
                      std::size_t index,
                      const std::vector<HomogeneousFieldShiftBinding>& bindings)
{
    const std::string name = binding.field_name.empty()
                                 ? "field binding " + std::to_string(index)
                                 : std::string(binding.field_name);
    if (binding.field == nullptr)
        throw std::invalid_argument(name + " has no field owner");
    if ((binding.leading_profile == nullptr)
        == (binding.leading_field == nullptr))
        throw std::invalid_argument(
            name + " must own exactly one leading profile or field");
    if (binding.field->size_r() != owner_plan.nr || binding.field->size_th() != owner_plan.nth ||
        binding.field->size_z() != owner_plan.nz)
        throw std::invalid_argument(name + " shape does not match the whole-cell shift plan");
    if (binding.leading_profile != nullptr)
    {
        if (binding.leading_profile->size()
            != static_cast<std::size_t>(owner_plan.nz))
            throw std::invalid_argument(
                name + " leading profile length does not match nz");
        for (float value : *binding.leading_profile)
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    name + " leading profile contains a non-finite value");
    }
    else
    {
        if (binding.leading_field->size_r() != owner_plan.nr
            || binding.leading_field->size_th() != owner_plan.nth
            || binding.leading_field->size_z() != owner_plan.nz)
            throw std::invalid_argument(
                name + " leading field shape does not match the owner plan");
        for (std::size_t value = 0; value < binding.leading_field->size(); ++value)
            if (!std::isfinite(binding.leading_field->data()[value]))
                throw std::invalid_argument(
                    name + " leading field contains a non-finite value");
    }
    for (std::size_t previous = 0; previous < index; ++previous)
        if (bindings[previous].field == binding.field)
            throw std::invalid_argument(name + " aliases an earlier field binding");
}

void validate_surface_binding(
    const WholeCellShiftPlan& plan,
    const HomogeneousSurfaceShiftBinding& binding,
    std::size_t index,
    const std::vector<HomogeneousSurfaceShiftBinding>& bindings)
{
    const std::string name = binding.field_name.empty()
                                 ? "surface binding " + std::to_string(index)
                                 : std::string(binding.field_name);
    if (binding.field == nullptr)
        throw std::invalid_argument(name + " has no field owner");
    const std::size_t expected = checked_product(
        static_cast<std::size_t>(plan.nr), static_cast<std::size_t>(plan.nth), "surface field");
    if (binding.field->size() != expected)
        throw std::invalid_argument(name + " shape does not match nr*nth");
    if (binding.leading_field != nullptr)
    {
        if (binding.leading_field->size() != expected)
            throw std::invalid_argument(
                name + " leading surface shape does not match nr*nth");
        for (float value : *binding.leading_field)
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    name + " leading surface contains a non-finite value");
    }
    else if (!std::isfinite(binding.leading_value))
        throw std::invalid_argument(name + " leading value is non-finite");
    for (std::size_t previous = 0; previous < index; ++previous)
        if (bindings[previous].field == binding.field)
            throw std::invalid_argument(name + " aliases an earlier surface binding");
}

void validate_scalar_binding(
    const HomogeneousScalarCommitBinding& binding,
    std::size_t index,
    const std::vector<HomogeneousScalarCommitBinding>& bindings)
{
    const std::string name = binding.scalar_name.empty()
        ? "scalar binding " + std::to_string(index)
        : std::string(binding.scalar_name);
    if (binding.target == nullptr)
        throw std::invalid_argument(name + " has no scalar owner");
    if (!std::isfinite(binding.committed_value))
        throw std::invalid_argument(name + " committed value is non-finite");
    for (std::size_t previous = 0; previous < index; ++previous)
        if (bindings[previous].target == binding.target)
            throw std::invalid_argument(name + " aliases an earlier scalar binding");
}

void validate_record_binding(
    const WholeCellShiftPlan& plan,
    const HomogeneousSurfaceRecordShiftBinding& binding,
    std::size_t index,
    const std::vector<HomogeneousSurfaceRecordShiftBinding>& bindings)
{
    const std::string name = binding.record_name.empty()
        ? "surface record binding " + std::to_string(index)
        : std::string(binding.record_name);
    if (binding.records == nullptr
        || ((binding.leading_record == nullptr)
            == (binding.leading_records == nullptr)))
        throw std::invalid_argument(
            name + " requires record storage and exactly one leading record owner");
    const std::size_t expected = checked_product(
        static_cast<std::size_t>(plan.nr),
        static_cast<std::size_t>(plan.nth), "surface record count");
    if (binding.record_count != expected || binding.record_size == 0)
        throw std::invalid_argument(name + " shape does not match nr*nth records");
    (void)checked_product(
        binding.record_count, binding.record_size, "surface record bytes");
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(
        binding.records);
    const std::uintptr_t leading = reinterpret_cast<std::uintptr_t>(
        binding.leading_records != nullptr
            ? binding.leading_records : binding.leading_record);
    const std::size_t bytes = binding.record_count * binding.record_size;
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
        throw std::overflow_error(name + " address range overflowed");
    if (leading >= begin && leading < begin + bytes)
        throw std::invalid_argument(name + " leading record aliases shifted storage");
    for (std::size_t previous = 0; previous < index; ++previous)
        if (bindings[previous].records == binding.records)
            throw std::invalid_argument(name + " aliases an earlier record binding");
}

void validate_flag_binding(
    const HomogeneousBooleanCommitBinding& binding,
    std::size_t index,
    const std::vector<HomogeneousBooleanCommitBinding>& bindings)
{
    const std::string name = binding.flag_name.empty()
        ? "flag binding " + std::to_string(index)
        : std::string(binding.flag_name);
    if (binding.target == nullptr)
        throw std::invalid_argument(name + " has no flag owner");
    for (std::size_t previous = 0; previous < index; ++previous)
        if (bindings[previous].target == binding.target)
            throw std::invalid_argument(name + " aliases an earlier flag binding");
}

void validate_runtime_invalidation_binding(
    const HomogeneousRuntimeInvalidationBinding& binding,
    std::size_t index,
    const std::vector<HomogeneousRuntimeInvalidationBinding>& bindings)
{
    const std::string name = binding.invalidation_name.empty()
        ? "runtime invalidation binding " + std::to_string(index)
        : std::string(binding.invalidation_name);
    if (binding.runtime == nullptr || binding.invalidate == nullptr)
        throw std::invalid_argument(
            name + " has no runtime owner or invalidation function");
    for (std::size_t previous = 0; previous < index; ++previous)
        if (bindings[previous].runtime == binding.runtime)
            throw std::invalid_argument(
                name + " aliases an earlier runtime invalidation binding");
}

void validate_runtime_commit_binding(
    const HomogeneousRuntimeCommitBinding& binding,
    std::size_t index,
    const std::vector<HomogeneousRuntimeCommitBinding>& bindings)
{
    const std::string name = binding.commit_name.empty()
        ? "runtime commit binding " + std::to_string(index)
        : std::string(binding.commit_name);
    if (binding.prepared_state == nullptr || binding.commit == nullptr)
        throw std::invalid_argument(
            name + " has no prepared state or commit function");
    for (std::size_t previous = 0; previous < index; ++previous)
        if (bindings[previous].prepared_state == binding.prepared_state)
            throw std::invalid_argument(
                name + " aliases an earlier runtime commit binding");
}

void validate_trailing_binding(
    const WholeCellShiftPlan& owner_plan,
    const TrailingFieldExportBinding& binding,
    std::size_t index,
    const std::vector<TrailingFieldExportBinding>& bindings)
{
    const std::string name = binding.field_name.empty()
                                 ? "trailing binding " + std::to_string(index)
                                 : std::string(binding.field_name);
    if (binding.source == nullptr)
        throw std::invalid_argument(name + " has no source field");
    if (binding.packed_values == nullptr)
        throw std::invalid_argument(name + " has no export storage");
    if (binding.source->size_r() != owner_plan.nr || binding.source->size_th() != owner_plan.nth ||
        binding.source->size_z() != owner_plan.nz)
        throw std::invalid_argument(name + " source shape does not match the whole-cell shift plan");
    if (binding.packed_values->size() != owner_plan.trailing_cells)
        throw std::invalid_argument(name + " export storage size does not match the trailing volume");
    for (std::size_t previous = 0; previous < index; ++previous)
    {
        if (bindings[previous].source == binding.source)
            throw std::invalid_argument(name + " aliases an earlier trailing source");
        if (bindings[previous].packed_values == binding.packed_values)
            throw std::invalid_argument(name + " aliases earlier trailing export storage");
    }
}

void validate_trailing_surface_binding(
    const WholeCellShiftPlan& plan,
    const TrailingSurfaceExportBinding& binding,
    std::size_t index,
    const std::vector<TrailingSurfaceExportBinding>& bindings)
{
    const std::string name = binding.field_name.empty()
        ? "trailing surface binding " + std::to_string(index)
        : std::string(binding.field_name);
    if (binding.source == nullptr || binding.packed_values == nullptr)
        throw std::invalid_argument(
            name + " has no source or export storage");
    const std::size_t surface_size = checked_product(
        static_cast<std::size_t>(plan.nr),
        static_cast<std::size_t>(plan.nth), "surface field");
    if (binding.source->size() != surface_size)
        throw std::invalid_argument(
            name + " source shape does not match nr*nth");
    if (plan.nz <= 0 || plan.trailing_cells % static_cast<std::size_t>(plan.nz) != 0)
        throw std::invalid_argument(
            name + " plan has an invalid trailing surface count");
    const std::size_t trailing_surface_values =
        plan.trailing_cells / static_cast<std::size_t>(plan.nz);
    if (binding.packed_values->size() != trailing_surface_values)
        throw std::invalid_argument(
            name + " export storage size does not match retiring columns");
    for (std::size_t previous = 0; previous < index; ++previous)
    {
        if (bindings[previous].source == binding.source)
            throw std::invalid_argument(
                name + " aliases an earlier trailing surface source");
        if (bindings[previous].packed_values == binding.packed_values)
            throw std::invalid_argument(
                name + " aliases earlier trailing surface export storage");
    }
}

void commit_field_shift(Field3D& field, const std::vector<float>& profile, const WholeCellShiftPlan& plan) noexcept
{
    const std::size_t nth = static_cast<std::size_t>(plan.nth);
    const std::size_t nz = static_cast<std::size_t>(plan.nz);
    const std::size_t i_slab_values = nth * nz;
    float* const data = field.data();

    if (plan.shift_i_cells > 0)
    {
        const std::size_t source_offset = static_cast<std::size_t>(plan.shift_i_cells) * i_slab_values;
        const std::size_t values = static_cast<std::size_t>(plan.nr - plan.shift_i_cells) * i_slab_values;
        std::memmove(data, data + source_offset, values * sizeof(float));
    }
    else if (plan.shift_i_cells < 0)
    {
        const std::size_t destination_offset = static_cast<std::size_t>(-plan.shift_i_cells) * i_slab_values;
        const std::size_t values = static_cast<std::size_t>(plan.nr + plan.shift_i_cells) * i_slab_values;
        std::memmove(data + destination_offset, data, values * sizeof(float));
    }

    if (plan.shift_j_cells != 0)
    {
        const std::size_t row_values = static_cast<std::size_t>(plan.nth - std::abs(plan.shift_j_cells)) * nz;
#ifdef _OPENMP
#pragma omp parallel for if (plan.nr > 8 && field.size() > (1U << 20)) schedule(static)
#endif
        for (int i = 0; i < plan.nr; ++i)
        {
            float* const slab = data + static_cast<std::size_t>(i) * i_slab_values;
            if (plan.shift_j_cells > 0)
            {
                const std::size_t source_offset = static_cast<std::size_t>(plan.shift_j_cells) * nz;
                std::memmove(slab, slab + source_offset, row_values * sizeof(float));
            }
            else
            {
                const std::size_t destination_offset = static_cast<std::size_t>(-plan.shift_j_cells) * nz;
                std::memmove(slab + destination_offset, slab, row_values * sizeof(float));
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (field.size() > (1U << 20)) schedule(static)
#endif
    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
        {
            if (i >= plan.destination_i_begin && i < plan.destination_i_end &&
                j >= plan.destination_j_begin && j < plan.destination_j_end)
                continue;
            float* const column = field.row_ptr(i, j);
            std::memcpy(column, profile.data(), nz * sizeof(float));
        }
}

void commit_field_shift(
    Field3D& field, const Field3D& leading,
    const WholeCellShiftPlan& plan) noexcept
{
    const std::size_t nth = static_cast<std::size_t>(plan.nth);
    const std::size_t nz = static_cast<std::size_t>(plan.nz);
    const std::size_t i_slab_values = nth * nz;
    float* const data = field.data();

    if (plan.shift_i_cells > 0)
    {
        const std::size_t source_offset =
            static_cast<std::size_t>(plan.shift_i_cells) * i_slab_values;
        const std::size_t values =
            static_cast<std::size_t>(plan.nr - plan.shift_i_cells)
            * i_slab_values;
        std::memmove(data, data + source_offset, values * sizeof(float));
    }
    else if (plan.shift_i_cells < 0)
    {
        const std::size_t destination_offset =
            static_cast<std::size_t>(-plan.shift_i_cells) * i_slab_values;
        const std::size_t values =
            static_cast<std::size_t>(plan.nr + plan.shift_i_cells)
            * i_slab_values;
        std::memmove(data + destination_offset, data, values * sizeof(float));
    }

    if (plan.shift_j_cells != 0)
    {
        const std::size_t row_values =
            static_cast<std::size_t>(plan.nth - std::abs(plan.shift_j_cells))
            * nz;
        for (int i = 0; i < plan.nr; ++i)
        {
            float* const slab =
                data + static_cast<std::size_t>(i) * i_slab_values;
            if (plan.shift_j_cells > 0)
                std::memmove(
                    slab,
                    slab + static_cast<std::size_t>(plan.shift_j_cells) * nz,
                    row_values * sizeof(float));
            else
                std::memmove(
                    slab + static_cast<std::size_t>(-plan.shift_j_cells) * nz,
                    slab, row_values * sizeof(float));
        }
    }

    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
        {
            if (i >= plan.destination_i_begin && i < plan.destination_i_end
                && j >= plan.destination_j_begin
                && j < plan.destination_j_end)
                continue;
            std::memcpy(
                field.row_ptr(i, j), leading.row_ptr(i, j),
                nz * sizeof(float));
        }
}

void commit_surface_shift(std::vector<float>& field,
                          float leading_value,
                          const WholeCellShiftPlan& plan) noexcept
{
    const std::size_t nth = static_cast<std::size_t>(plan.nth);
    float* const data = field.data();

    if (plan.shift_i_cells > 0)
    {
        const std::size_t source_offset = static_cast<std::size_t>(plan.shift_i_cells) * nth;
        const std::size_t values = static_cast<std::size_t>(plan.nr - plan.shift_i_cells) * nth;
        std::memmove(data, data + source_offset, values * sizeof(float));
    }
    else if (plan.shift_i_cells < 0)
    {
        const std::size_t destination_offset = static_cast<std::size_t>(-plan.shift_i_cells) * nth;
        const std::size_t values = static_cast<std::size_t>(plan.nr + plan.shift_i_cells) * nth;
        std::memmove(data + destination_offset, data, values * sizeof(float));
    }

    if (plan.shift_j_cells != 0)
    {
        const std::size_t row_values = static_cast<std::size_t>(plan.nth - std::abs(plan.shift_j_cells));
#ifdef _OPENMP
#pragma omp parallel for if (field.size() > (1U << 20)) schedule(static)
#endif
        for (int i = 0; i < plan.nr; ++i)
        {
            float* const row = data + static_cast<std::size_t>(i) * nth;
            if (plan.shift_j_cells > 0)
                std::memmove(row, row + plan.shift_j_cells, row_values * sizeof(float));
            else
                std::memmove(row - plan.shift_j_cells, row, row_values * sizeof(float));
        }
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (field.size() > (1U << 20)) schedule(static)
#endif
    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
        {
            if (i >= plan.destination_i_begin && i < plan.destination_i_end &&
                j >= plan.destination_j_begin && j < plan.destination_j_end)
                continue;
            data[static_cast<std::size_t>(i) * nth + static_cast<std::size_t>(j)] = leading_value;
        }
}

void commit_surface_record_shift(
    const HomogeneousSurfaceRecordShiftBinding& binding,
    const WholeCellShiftPlan& plan) noexcept
{
    const std::size_t nth = static_cast<std::size_t>(plan.nth);
    const std::size_t stride = binding.record_size;
    auto* const data = static_cast<unsigned char*>(binding.records);
    if (plan.shift_i_cells > 0)
    {
        const std::size_t source =
            static_cast<std::size_t>(plan.shift_i_cells) * nth * stride;
        const std::size_t bytes =
            static_cast<std::size_t>(plan.nr - plan.shift_i_cells)
            * nth * stride;
        std::memmove(data, data + source, bytes);
    }
    else if (plan.shift_i_cells < 0)
    {
        const std::size_t destination =
            static_cast<std::size_t>(-plan.shift_i_cells) * nth * stride;
        const std::size_t bytes =
            static_cast<std::size_t>(plan.nr + plan.shift_i_cells)
            * nth * stride;
        std::memmove(data + destination, data, bytes);
    }
    if (plan.shift_j_cells != 0)
    {
        const std::size_t row_records =
            static_cast<std::size_t>(plan.nth - std::abs(plan.shift_j_cells));
        for (int i = 0; i < plan.nr; ++i)
        {
            auto* const row = data + static_cast<std::size_t>(i) * nth * stride;
            if (plan.shift_j_cells > 0)
                std::memmove(
                    row, row + static_cast<std::size_t>(plan.shift_j_cells) * stride,
                    row_records * stride);
            else
                std::memmove(
                    row + static_cast<std::size_t>(-plan.shift_j_cells) * stride,
                    row, row_records * stride);
        }
    }
    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
        {
            if (i >= plan.destination_i_begin && i < plan.destination_i_end
                && j >= plan.destination_j_begin
                && j < plan.destination_j_end)
                continue;
            const std::size_t destination =
                static_cast<std::size_t>(i) * nth
                + static_cast<std::size_t>(j);
            const void* source = binding.leading_records != nullptr
                ? static_cast<const unsigned char*>(binding.leading_records)
                    + destination * stride
                : binding.leading_record;
            std::memcpy(
                data + (static_cast<std::size_t>(i) * nth
                        + static_cast<std::size_t>(j)) * stride,
                source, stride);
        }
}

void commit_trailing_export(const Field3D& source,
                            std::vector<float>& packed,
                            const WholeCellShiftPlan& plan) noexcept
{
    const std::size_t nz = static_cast<std::size_t>(plan.nz);
    std::size_t offset = 0;
    for (int i = plan.physical_i_begin; i < plan.physical_i_end; ++i)
        for (int j = plan.physical_j_begin; j < plan.physical_j_end; ++j)
        {
            if (i >= plan.source_i_begin && i < plan.source_i_end &&
                j >= plan.source_j_begin && j < plan.source_j_end)
                continue;
            std::memcpy(packed.data() + offset, source.row_ptr(i, j), nz * sizeof(float));
            offset += nz;
        }
}

void commit_trailing_surface_export(
    const std::vector<float>& source,
    std::vector<float>& packed,
    const WholeCellShiftPlan& plan) noexcept
{
    const std::size_t nth = static_cast<std::size_t>(plan.nth);
    std::size_t offset = 0;
    for (int i = plan.physical_i_begin; i < plan.physical_i_end; ++i)
        for (int j = plan.physical_j_begin; j < plan.physical_j_end; ++j)
        {
            if (i >= plan.source_i_begin && i < plan.source_i_end
                && j >= plan.source_j_begin && j < plan.source_j_end)
                continue;
            packed[offset++] = source[
                static_cast<std::size_t>(i) * nth
                + static_cast<std::size_t>(j)];
        }
}

} // namespace

WholeCellShiftPlan plan_whole_cell_shift_for_physical_domain(
    int nr, int nth, int nz, int shift_i_cells, int shift_j_cells,
    int physical_i_begin, int physical_i_end,
    int physical_j_begin, int physical_j_end)
{
    if (nr <= 0 || nth <= 0 || nz <= 0)
        throw std::invalid_argument("whole-cell shift requires positive field dimensions");
    if (shift_i_cells == 0 && shift_j_cells == 0)
        throw std::invalid_argument("whole-cell shift requires a non-zero displacement");
    const std::int64_t abs_i = shift_i_cells < 0 ? -static_cast<std::int64_t>(shift_i_cells) : shift_i_cells;
    const std::int64_t abs_j = shift_j_cells < 0 ? -static_cast<std::int64_t>(shift_j_cells) : shift_j_cells;
    if (physical_i_begin < 0 || physical_i_end > nr ||
        physical_j_begin < 0 || physical_j_end > nth ||
        physical_i_begin >= physical_i_end || physical_j_begin >= physical_j_end)
        throw std::invalid_argument("whole-cell shift physical domain is invalid");
    if (abs_i >= physical_i_end - physical_i_begin ||
        abs_j >= physical_j_end - physical_j_begin)
        throw std::invalid_argument(
            "whole-cell shift must retain at least one physical cell on each horizontal axis");

    WholeCellShiftPlan plan;
    plan.nr = nr;
    plan.nth = nth;
    plan.nz = nz;
    plan.shift_i_cells = shift_i_cells;
    plan.shift_j_cells = shift_j_cells;
    plan.physical_i_begin = physical_i_begin;
    plan.physical_i_end = physical_i_end;
    plan.physical_j_begin = physical_j_begin;
    plan.physical_j_end = physical_j_end;
    plan.destination_i_begin = std::max(physical_i_begin, physical_i_begin - shift_i_cells);
    plan.destination_i_end = std::min(physical_i_end, physical_i_end - shift_i_cells);
    plan.destination_j_begin = std::max(physical_j_begin, physical_j_begin - shift_j_cells);
    plan.destination_j_end = std::min(physical_j_end, physical_j_end - shift_j_cells);
    plan.source_i_begin = plan.destination_i_begin + shift_i_cells;
    plan.source_i_end = plan.destination_i_end + shift_i_cells;
    plan.source_j_begin = plan.destination_j_begin + shift_j_cells;
    plan.source_j_end = plan.destination_j_end + shift_j_cells;

    const std::size_t retained_i = static_cast<std::size_t>(plan.destination_i_end - plan.destination_i_begin);
    const std::size_t retained_j = static_cast<std::size_t>(plan.destination_j_end - plan.destination_j_begin);
    plan.retained_cells = checked_product(checked_product(retained_i, retained_j, "retained overlap"),
                                          static_cast<std::size_t>(nz),
                                          "retained overlap");
    const std::size_t total = checked_product(
        checked_product(static_cast<std::size_t>(nr), static_cast<std::size_t>(nth), "field"),
        static_cast<std::size_t>(nz),
        "field");
    plan.leading_cells = total - plan.retained_cells;
    const std::size_t physical_total = checked_product(
        checked_product(static_cast<std::size_t>(physical_i_end - physical_i_begin),
                        static_cast<std::size_t>(physical_j_end - physical_j_begin),
                        "physical field"),
        static_cast<std::size_t>(nz), "physical field");
    plan.trailing_cells = physical_total - plan.retained_cells;
    return plan;
}

WholeCellShiftPlan plan_whole_cell_shift(
    int nr, int nth, int nz, int shift_i_cells, int shift_j_cells)
{
    return plan_whole_cell_shift_for_physical_domain(
        nr, nth, nz, shift_i_cells, shift_j_cells, 0, nr, 0, nth);
}

WholeCellShiftPlan plan_whole_cell_shift_for_horizontal_location(
    const WholeCellShiftPlan& scalar_plan,
    HorizontalFieldLocation location)
{
    const WholeCellShiftPlan canonical = plan_whole_cell_shift_for_physical_domain(
        scalar_plan.nr, scalar_plan.nth, scalar_plan.nz,
        scalar_plan.shift_i_cells, scalar_plan.shift_j_cells,
        scalar_plan.physical_i_begin, scalar_plan.physical_i_end,
        scalar_plan.physical_j_begin, scalar_plan.physical_j_end);
    if (!plans_equal(canonical, scalar_plan))
        throw std::invalid_argument("scalar whole-cell shift plan is internally inconsistent");

    int i_begin = scalar_plan.physical_i_begin;
    int i_end = scalar_plan.physical_i_end;
    int j_begin = scalar_plan.physical_j_begin;
    int j_end = scalar_plan.physical_j_end;
    switch (location)
    {
    case HorizontalFieldLocation::ScalarCenter:
        return scalar_plan;
    case HorizontalFieldLocation::XFace:
        --i_begin;
        break;
    case HorizontalFieldLocation::YFace:
        --j_begin;
        break;
    default:
        throw std::invalid_argument("whole-cell shift field location is invalid");
    }
    return plan_whole_cell_shift_for_physical_domain(
        scalar_plan.nr, scalar_plan.nth, scalar_plan.nz,
        scalar_plan.shift_i_cells, scalar_plan.shift_j_cells,
        i_begin, i_end, j_begin, j_end);
}

DomainFrameShiftReport commit_whole_cell_shift_to_domain_frame(
    HorizontalDomainFrame& frame,
    const WholeCellShiftPlan& plan,
    double dx_m,
    double dy_m)
{
    const WholeCellShiftPlan canonical =
        plan_whole_cell_shift_for_physical_domain(
            plan.nr, plan.nth, plan.nz,
            plan.shift_i_cells, plan.shift_j_cells,
            plan.physical_i_begin, plan.physical_i_end,
            plan.physical_j_begin, plan.physical_j_end);
    if (!plans_equal(canonical, plan))
        throw std::invalid_argument(
            "domain-frame shift requires a canonical whole-cell plan");
    if (!std::isfinite(dx_m) || dx_m <= 0.0 ||
        !std::isfinite(dy_m) || dy_m <= 0.0)
        throw std::invalid_argument(
            "domain-frame shift requires finite positive horizontal spacing");

    frame.validate();
    DomainFrameShiftReport report;
    report.shift_i_cells = plan.shift_i_cells;
    report.shift_j_cells = plan.shift_j_cells;
    report.displacement_m = {
        static_cast<double>(plan.shift_i_cells) * dx_m,
        static_cast<double>(plan.shift_j_cells) * dy_m,
    };
    report.previous_earth_origin_at_t0_m = frame.earth_origin_at_t0_m;
    report.committed_earth_origin_at_t0_m = {
        frame.earth_origin_at_t0_m.x_m + report.displacement_m.x_m,
        frame.earth_origin_at_t0_m.y_m + report.displacement_m.y_m,
    };
    const double values[] = {
        report.displacement_m.x_m,
        report.displacement_m.y_m,
        report.committed_earth_origin_at_t0_m.x_m,
        report.committed_earth_origin_at_t0_m.y_m,
    };
    if (std::any_of(std::begin(values), std::end(values),
                    [](double value) { return !std::isfinite(value); }))
        throw std::overflow_error(
            "domain-frame whole-cell displacement is not finite");

    HorizontalDomainFrame committed = frame;
    committed.earth_origin_at_t0_m =
        report.committed_earth_origin_at_t0_m;
    committed.validate();
    frame = committed;
    return report;
}

const char* domain_frame_shift_algorithm_id() noexcept
{
    return "integer_whole_cell_earth_origin_commit_v1";
}

HomogeneousFieldShiftReport execute_homogeneous_field_shift(
    const WholeCellShiftPlan&                         plan,
    const std::vector<HomogeneousFieldShiftBinding>& bindings)
{
    return execute_homogeneous_window_shift(plan, bindings, {}).volumes;
}

static HomogeneousWindowShiftReport evaluate_homogeneous_window_shift(
    const WholeCellShiftPlan&                           plan,
    const std::vector<HomogeneousFieldShiftBinding>&   volume_bindings,
    const std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings,
    const std::vector<HomogeneousScalarCommitBinding>& scalar_bindings,
    const std::vector<HomogeneousSurfaceRecordShiftBinding>& record_bindings,
    const std::vector<HomogeneousBooleanCommitBinding>& flag_bindings,
    const std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings,
    const std::vector<HomogeneousRuntimeCommitBinding>&
        runtime_commit_bindings,
    bool commit_volumes,
    bool commit_nonvolumes)
{
    const WholeCellShiftPlan canonical = plan_whole_cell_shift_for_physical_domain(
        plan.nr, plan.nth, plan.nz, plan.shift_i_cells, plan.shift_j_cells,
        plan.physical_i_begin, plan.physical_i_end,
        plan.physical_j_begin, plan.physical_j_end);
    if (!plans_equal(canonical, plan))
        throw std::invalid_argument("whole-cell shift plan is internally inconsistent");
    if (volume_bindings.empty() && surface_bindings.empty()
        && scalar_bindings.empty() && record_bindings.empty()
        && flag_bindings.empty() && runtime_invalidation_bindings.empty()
        && runtime_commit_bindings.empty())
        throw std::invalid_argument(
            "homogeneous window shift requires at least one binding");
    std::vector<WholeCellShiftPlan> volume_plans;
    volume_plans.reserve(volume_bindings.size());
    for (std::size_t index = 0; index < volume_bindings.size(); ++index)
    {
        volume_plans.push_back(plan_whole_cell_shift_for_horizontal_location(
            plan, volume_bindings[index].horizontal_location));
        validate_binding(volume_plans.back(), volume_bindings[index], index,
                         volume_bindings);
    }
    for (std::size_t index = 0; index < surface_bindings.size(); ++index)
        validate_surface_binding(plan, surface_bindings[index], index, surface_bindings);
    for (std::size_t index = 0; index < scalar_bindings.size(); ++index)
        validate_scalar_binding(scalar_bindings[index], index, scalar_bindings);
    for (std::size_t index = 0; index < record_bindings.size(); ++index)
        validate_record_binding(
            plan, record_bindings[index], index, record_bindings);
    for (std::size_t index = 0; index < flag_bindings.size(); ++index)
        validate_flag_binding(flag_bindings[index], index, flag_bindings);
    for (std::size_t index = 0;
         index < runtime_invalidation_bindings.size(); ++index)
        validate_runtime_invalidation_binding(
            runtime_invalidation_bindings[index], index,
            runtime_invalidation_bindings);
    for (std::size_t index = 0;
         index < runtime_commit_bindings.size(); ++index)
        validate_runtime_commit_binding(
            runtime_commit_bindings[index], index,
            runtime_commit_bindings);

    HomogeneousWindowShiftReport report;
    report.volumes.fields_shifted = volume_bindings.size();
    for (const auto& owner_plan : volume_plans)
    {
        report.volumes.retained_values = checked_sum(
            report.volumes.retained_values, owner_plan.retained_cells,
            "retained report");
        report.volumes.initialized_values = checked_sum(
            report.volumes.initialized_values, owner_plan.leading_cells,
            "initialized report");
    }
    report.volumes.retained_bytes =
        checked_product(report.volumes.retained_values, sizeof(float), "retained report bytes");
    report.volumes.initialized_bytes =
        checked_product(report.volumes.initialized_values, sizeof(float), "initialized report bytes");

    const std::size_t retained_surface_cells = checked_product(
        static_cast<std::size_t>(canonical.destination_i_end - canonical.destination_i_begin),
        static_cast<std::size_t>(canonical.destination_j_end - canonical.destination_j_begin),
        "retained surface overlap");
    const std::size_t total_surface_cells = checked_product(
        static_cast<std::size_t>(plan.nr), static_cast<std::size_t>(plan.nth), "surface field");
    const std::size_t leading_surface_cells = total_surface_cells - retained_surface_cells;
    report.surfaces_shifted = surface_bindings.size();
    report.retained_surface_values =
        checked_product(retained_surface_cells, surface_bindings.size(), "retained surface report");
    report.initialized_surface_values =
        checked_product(leading_surface_cells, surface_bindings.size(), "initialized surface report");
    report.retained_surface_bytes =
        checked_product(report.retained_surface_values, sizeof(float), "retained surface report bytes");
    report.initialized_surface_bytes =
        checked_product(report.initialized_surface_values, sizeof(float), "initialized surface report bytes");
    report.scalars_committed = scalar_bindings.size();
    report.record_surfaces_shifted = record_bindings.size();
    report.retained_record_values = checked_product(
        retained_surface_cells, record_bindings.size(),
        "retained surface record report");
    report.initialized_record_values = checked_product(
        leading_surface_cells, record_bindings.size(),
        "initialized surface record report");
    for (const auto& binding : record_bindings)
    {
        report.retained_record_bytes = checked_sum(
            report.retained_record_bytes,
            checked_product(retained_surface_cells, binding.record_size,
                            "retained surface record bytes"),
            "retained surface record byte report");
        report.initialized_record_bytes = checked_sum(
            report.initialized_record_bytes,
            checked_product(leading_surface_cells, binding.record_size,
                            "initialized surface record bytes"),
            "initialized surface record byte report");
    }
    report.flags_committed = flag_bindings.size();
    report.runtime_invalidations_committed =
        runtime_invalidation_bindings.size();
    report.runtime_commits_committed = runtime_commit_bindings.size();

    // Everything capable of throwing, including report arithmetic, belongs to
    // preflight. After the first write the transaction is allocation-free and
    // consists only of noexcept moves/fills over already validated storage.
    if (commit_volumes)
        for (std::size_t index = 0; index < volume_bindings.size(); ++index)
        {
            const auto& binding = volume_bindings[index];
            if (binding.leading_profile != nullptr)
                commit_field_shift(
                    *binding.field, *binding.leading_profile,
                    volume_plans[index]);
            else
                commit_field_shift(
                    *binding.field, *binding.leading_field,
                    volume_plans[index]);
        }
    if (commit_nonvolumes)
        commit_preflighted_homogeneous_window_nonvolume(
            plan, surface_bindings, scalar_bindings, record_bindings,
            flag_bindings, runtime_invalidation_bindings,
            runtime_commit_bindings);

    return report;
}

HomogeneousWindowShiftReport preflight_homogeneous_window_shift(
    const WholeCellShiftPlan& plan,
    const std::vector<HomogeneousFieldShiftBinding>& volume_bindings,
    const std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings,
    const std::vector<HomogeneousScalarCommitBinding>& scalar_bindings,
    const std::vector<HomogeneousSurfaceRecordShiftBinding>& record_bindings,
    const std::vector<HomogeneousBooleanCommitBinding>& flag_bindings,
    const std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings,
    const std::vector<HomogeneousRuntimeCommitBinding>& runtime_commit_bindings)
{
    return evaluate_homogeneous_window_shift(
        plan, volume_bindings, surface_bindings, scalar_bindings,
        record_bindings, flag_bindings, runtime_invalidation_bindings,
        runtime_commit_bindings, false, false);
}

void commit_preflighted_homogeneous_window_nonvolume(
    const WholeCellShiftPlan& plan,
    const std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings,
    const std::vector<HomogeneousScalarCommitBinding>& scalar_bindings,
    const std::vector<HomogeneousSurfaceRecordShiftBinding>& record_bindings,
    const std::vector<HomogeneousBooleanCommitBinding>& flag_bindings,
    const std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings,
    const std::vector<HomogeneousRuntimeCommitBinding>&
        runtime_commit_bindings) noexcept
{
    for (const auto& binding : surface_bindings)
    {
        commit_surface_shift(*binding.field, binding.leading_value, plan);
        if (binding.leading_field != nullptr)
        {
            const std::size_t nth = static_cast<std::size_t>(plan.nth);
            for (int i = 0; i < plan.nr; ++i)
                for (int j = 0; j < plan.nth; ++j)
                {
                    if (i >= plan.destination_i_begin
                        && i < plan.destination_i_end
                        && j >= plan.destination_j_begin
                        && j < plan.destination_j_end)
                        continue;
                    const std::size_t index =
                        static_cast<std::size_t>(i) * nth
                        + static_cast<std::size_t>(j);
                    (*binding.field)[index] =
                        (*binding.leading_field)[index];
                }
        }
    }
    for (const auto& binding : scalar_bindings)
        *binding.target = binding.committed_value;
    for (const auto& binding : record_bindings)
        commit_surface_record_shift(binding, plan);
    for (const auto& binding : flag_bindings)
        *binding.target = binding.committed_value;
    for (const auto& binding : runtime_invalidation_bindings)
        binding.invalidate(binding.runtime);
    for (const auto& binding : runtime_commit_bindings)
        binding.commit(binding.prepared_state);
}

HomogeneousWindowShiftReport execute_homogeneous_window_shift(
    const WholeCellShiftPlan& plan,
    const std::vector<HomogeneousFieldShiftBinding>& volume_bindings,
    const std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings,
    const std::vector<HomogeneousScalarCommitBinding>& scalar_bindings,
    const std::vector<HomogeneousSurfaceRecordShiftBinding>& record_bindings,
    const std::vector<HomogeneousBooleanCommitBinding>& flag_bindings,
    const std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings,
    const std::vector<HomogeneousRuntimeCommitBinding>& runtime_commit_bindings)
{
    return evaluate_homogeneous_window_shift(
        plan, volume_bindings, surface_bindings, scalar_bindings,
        record_bindings, flag_bindings, runtime_invalidation_bindings,
        runtime_commit_bindings, true, true);
}

TrailingFieldExportReport export_trailing_fields(
    const WholeCellShiftPlan&                       plan,
    const std::vector<TrailingFieldExportBinding>& bindings)
{
    const WholeCellShiftPlan canonical = plan_whole_cell_shift_for_physical_domain(
        plan.nr, plan.nth, plan.nz, plan.shift_i_cells, plan.shift_j_cells,
        plan.physical_i_begin, plan.physical_i_end,
        plan.physical_j_begin, plan.physical_j_end);
    if (!plans_equal(canonical, plan))
        throw std::invalid_argument("whole-cell shift plan is internally inconsistent");
    if (bindings.empty())
        throw std::invalid_argument("trailing export requires at least one field binding");
    std::vector<WholeCellShiftPlan> owner_plans;
    owner_plans.reserve(bindings.size());
    for (std::size_t index = 0; index < bindings.size(); ++index)
    {
        owner_plans.push_back(plan_whole_cell_shift_for_horizontal_location(
            plan, bindings[index].horizontal_location));
        validate_trailing_binding(owner_plans.back(), bindings[index], index,
                                  bindings);
    }

    TrailingFieldExportReport report;
    report.fields_exported = bindings.size();
    for (const auto& owner_plan : owner_plans)
        report.values_exported = checked_sum(
            report.values_exported, owner_plan.trailing_cells,
            "trailing export report");
    report.bytes_exported = checked_product(report.values_exported, sizeof(float), "trailing export bytes");

    for (std::size_t index = 0; index < bindings.size(); ++index)
        commit_trailing_export(*bindings[index].source,
                               *bindings[index].packed_values,
                               owner_plans[index]);
    return report;
}

TrailingFieldExportReport export_trailing_surfaces(
    const WholeCellShiftPlan& plan,
    const std::vector<TrailingSurfaceExportBinding>& bindings)
{
    const WholeCellShiftPlan canonical =
        plan_whole_cell_shift_for_physical_domain(
            plan.nr, plan.nth, plan.nz,
            plan.shift_i_cells, plan.shift_j_cells,
            plan.physical_i_begin, plan.physical_i_end,
            plan.physical_j_begin, plan.physical_j_end);
    if (!plans_equal(canonical, plan))
        throw std::invalid_argument(
            "whole-cell shift plan is internally inconsistent");
    if (bindings.empty())
        throw std::invalid_argument(
            "trailing surface export requires at least one binding");
    for (std::size_t index = 0; index < bindings.size(); ++index)
        validate_trailing_surface_binding(
            plan, bindings[index], index, bindings);

    const std::size_t values_per_surface =
        plan.trailing_cells / static_cast<std::size_t>(plan.nz);
    TrailingFieldExportReport report;
    report.fields_exported = bindings.size();
    report.values_exported = checked_product(
        values_per_surface, bindings.size(),
        "trailing surface export report");
    report.bytes_exported = checked_product(
        report.values_exported, sizeof(float),
        "trailing surface export bytes");
    for (const auto& binding : bindings)
        commit_trailing_surface_export(
            *binding.source, *binding.packed_values, plan);
    return report;
}

const char* homogeneous_field_shift_algorithm_id() noexcept
{
    return "stagger_aware_in_place_homogeneous_whole_cell_shift_v2";
}

const char* terrain_aware_field_shift_algorithm_id() noexcept
{
    return "stagger_aware_destination_indexed_terrain_whole_cell_shift_v1";
}

} // namespace tmv::domain

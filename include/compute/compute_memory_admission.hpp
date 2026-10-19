#pragma once

#include "compute/backend_types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

/**
 * @file compute_memory_admission.hpp
 * @brief Backend-resolved accelerator working-set admission.
 */

struct ComputeMemoryPreflight
{
    int field_equivalents = 0;
    std::size_t resident_level_count = 0;
    std::uint64_t field_bytes = 0;
    std::uint64_t host_residency_bytes = 0;
    std::uint64_t device_residency_bytes = 0;
    std::uint64_t device_budget_bytes = 0;
    bool accelerator = false;
    bool unified_host_device_memory = false;
    bool device_heap_known = false;
    bool device_budget_is_live = false;
    bool exceeds_storage_buffer_range = false;
    bool exceeds_device_budget = false;
    // Structural input errors and saturated arithmetic are programming or
    // configuration failures, not ordinary budget-policy warnings.
    bool invalid_input = false;
    bool arithmetic_saturated = false;
};

inline std::uint64_t saturated_product(
    std::uint64_t a, std::uint64_t b) noexcept
{
    return b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b
        ? std::numeric_limits<std::uint64_t>::max()
        : a * b;
}

inline std::uint64_t saturated_sum(
    std::uint64_t a, std::uint64_t b) noexcept
{
    return a > std::numeric_limits<std::uint64_t>::max() - b
        ? std::numeric_limits<std::uint64_t>::max()
        : a + b;
}

struct ComputeMemoryLevelDimensions
{
    int nx = 0;
    int ny = 0;
    int nz = 0;
};

/**
 * The current accelerator dynamics plan uses a conservative 51-field-equivalent
 * capacity envelope for persistent names, simultaneous dispatch transients,
 * legacy single-field buffers, alignment, and small profile rounding. This is
 * not a universal named-field cardinality: specialized transactions such as
 * domain recycling have their own wider binding limit and must retain their
 * separate admission boundary.
 */
inline constexpr int kComputeMaximumRetainedFieldEquivalents = 51;

/**
 * Admit every grid level whose fields may be resident on the accelerator at
 * the same time. A storage-buffer limit applies to the largest individual
 * field, while the heap limit applies to the saturated sum of all retained
 * level working sets.
 *
 * The current backend contract exposes one conservative capacity envelope,
 * not separate persistent and reusable-transient counts. Count the complete
 * envelope for each simultaneously resident level until that ownership split
 * becomes an explicit backend property; this may overestimate shared scratch
 * but cannot under-admit the device heap.
 */
inline ComputeMemoryPreflight assess_compute_memory_hierarchy(
    const ComputeMemoryProperties& properties,
    const ComputeMemoryLevelDimensions* levels,
    std::size_t level_count,
    int field_equivalents = kComputeMaximumRetainedFieldEquivalents,
    int device_budget_percent = 80) noexcept
{
    ComputeMemoryPreflight result;
    result.accelerator = properties.accelerator;
    result.unified_host_device_memory =
        properties.unified_host_device_memory;
    if (!properties.accelerator)
        return result;

    result.field_equivalents = std::max(0, field_equivalents);
    result.resident_level_count = level_count;
    result.device_budget_is_live = properties.device_budget_is_live;
    result.device_heap_known = properties.device_local_heap_bytes > 0 || properties.device_budget_is_live;
    if (result.device_heap_known)
    {
        const std::uint64_t percent = static_cast<std::uint64_t>(
            std::clamp(device_budget_percent, 1, 100));
        // Compute floor(heap * percent / 100) without overflowing the
        // intermediate product for very large reported heaps.
        std::uint64_t capacity = properties.device_local_heap_bytes;
        if (properties.device_budget_is_live)
            capacity = capacity > 0
                ? std::min(capacity, properties.device_available_budget_bytes)
                : properties.device_available_budget_bytes;
        result.device_budget_bytes =
            (capacity / 100) * percent + ((capacity % 100) * percent) / 100;
    }

    if (level_count > 0 && levels == nullptr)
    {
        result.invalid_input = true;
        return result;
    }
    for (std::size_t level = 0; level < level_count; ++level)
    {
        if (levels[level].nx <= 0 || levels[level].ny <= 0 ||
            levels[level].nz <= 0)
        {
            result.invalid_input = true;
            return result;
        }
    }

    const auto multiply = [&result](std::uint64_t a, std::uint64_t b) {
        if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b)
        {
            result.arithmetic_saturated = true;
        }
        return saturated_product(a, b);
    };
    const auto add = [&result](std::uint64_t a, std::uint64_t b) {
        if (a > std::numeric_limits<std::uint64_t>::max() - b)
        {
            result.arithmetic_saturated = true;
        }
        return saturated_sum(a, b);
    };

    for (std::size_t level = 0; level < result.resident_level_count; ++level)
    {
        const std::uint64_t sx = static_cast<std::uint64_t>(levels[level].nx);
        const std::uint64_t sy = static_cast<std::uint64_t>(levels[level].ny);
        const std::uint64_t sz = static_cast<std::uint64_t>(levels[level].nz);
        const std::uint64_t cells =
            multiply(multiply(sx, sy), sz);
        const std::uint64_t level_field_bytes =
            multiply(cells, sizeof(float));
        const std::uint64_t level_residency_bytes = multiply(
            level_field_bytes,
            static_cast<std::uint64_t>(result.field_equivalents));
        result.field_bytes = std::max(result.field_bytes, level_field_bytes);
        result.host_residency_bytes = add(
            result.host_residency_bytes, level_residency_bytes);
    }
    // Unified allocations consume this host-visible physical memory once.
    // Discrete allocations consume the same amount in staging and again in
    // the device-local heap; the two budgets are intentionally separate.
    result.device_residency_bytes = result.host_residency_bytes;

    result.exceeds_storage_buffer_range =
        properties.max_storage_buffer_bytes > 0 &&
        result.field_bytes > properties.max_storage_buffer_bytes;
    if (result.device_heap_known)
    {
        result.exceeds_device_budget =
            result.device_residency_bytes > result.device_budget_bytes;
    }
    return result;
}

inline ComputeMemoryPreflight assess_compute_memory(
    const ComputeMemoryProperties& properties,
    int nx, int ny, int nz,
    int field_equivalents = kComputeMaximumRetainedFieldEquivalents,
    int device_budget_percent = 80) noexcept
{
    const ComputeMemoryLevelDimensions root{nx, ny, nz};
    return assess_compute_memory_hierarchy(
        properties, &root, 1, field_equivalents, device_budget_percent);
}

inline bool compute_memory_admission_should_reject(
    const ComputeMemoryPreflight& preflight, bool reject_policy) noexcept
{
    if (!preflight.accelerator)
        return false;
    // A warning-only budget policy cannot make malformed dimensions or an
    // unrepresentable working set safe to allocate.
    if (preflight.invalid_input || preflight.arithmetic_saturated)
        return true;
    return reject_policy &&
        (preflight.exceeds_storage_buffer_range ||
         (preflight.device_heap_known && preflight.exceeds_device_budget));
}

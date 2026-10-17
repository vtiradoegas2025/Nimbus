/**
 * @file vulkan_residency_policy.hpp
 * @brief Pure validity policy for named Vulkan field residency.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace tmv::vulkan_residency
{

/** @brief Host/device identity recorded after a completed producer fence. */
struct ResidentBinding
{
    const void* host_identity = nullptr;
    std::size_t bytes = 0;
    std::uint64_t step_generation = 0;
    bool device_valid = false;
};

/**
 * @brief Whether a consumer may bind a resident buffer without re-uploading.
 *
 * Pointer identity alone is deliberately insufficient: the producing fence
 * must have committed the binding in the current model-step generation, for
 * the exact byte extent, and residency must be enabled explicitly.
 */
[[nodiscard]] inline bool reusable(
    const ResidentBinding& binding,
    std::uint64_t current_step_generation,
    const void* host_identity,
    std::size_t bytes,
    bool residency_enabled)
{
    return residency_enabled && current_step_generation != 0 &&
           binding.device_valid &&
           binding.step_generation == current_step_generation &&
           binding.host_identity == host_identity &&
           binding.bytes == bytes;
}

/**
 * @brief Whether a write-only output may safely skip host initialization.
 *
 * The dispatch helper cannot prove shader control flow, but it can enforce the
 * necessary domain condition: every field element must have an invocation.
 * Call sites remain responsible for marking only shaders that write every
 * output on every in-range path.
 */
[[nodiscard]] inline bool covers_full_field(
    std::uint64_t total_invocations,
    std::uint64_t total_cells)
{
    return total_cells != 0 && total_invocations == total_cells;
}

} // namespace tmv::vulkan_residency

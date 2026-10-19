/**
 * @file vulkan_dispatch_policy.hpp
 * @brief Checked linear-domain sizing and Vulkan 2D dispatch planning.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <vulkan/vulkan.h>

namespace tmv::vulkan_dispatch
{

struct LinearPlan
{
    uint32_t groups_x = 0;
    uint32_t groups_y = 0;
    uint32_t groups_z = 1;
};

inline bool checked_extent(int x,
                           int y,
                           int z,
                           std::size_t element_size,
                           std::size_t& elements,
                           VkDeviceSize& bytes,
                           std::string& error)
{
    if (x <= 0 || y <= 0 || z <= 0 || element_size == 0)
    {
        error = "dispatch dimensions and element size must be positive";
        return false;
    }
    const std::size_t sx = static_cast<std::size_t>(x);
    const std::size_t sy = static_cast<std::size_t>(y);
    const std::size_t sz = static_cast<std::size_t>(z);
    if (sx > std::numeric_limits<std::size_t>::max() / sy ||
        sx * sy > std::numeric_limits<std::size_t>::max() / sz)
    {
        error = "dispatch dimensions overflow size_t";
        return false;
    }
    elements = sx * sy * sz;
    if (elements > std::numeric_limits<std::size_t>::max() / element_size ||
        elements * element_size > std::numeric_limits<VkDeviceSize>::max())
    {
        error = "dispatch buffer byte size overflows host or Vulkan size";
        return false;
    }
    bytes = static_cast<VkDeviceSize>(elements * element_size);
    return true;
}

inline bool plan_linear(std::uint64_t invocations,
                        uint32_t local_size_x,
                        const VkPhysicalDeviceLimits& limits,
                        LinearPlan& plan,
                        std::string& error)
{
    plan = {};
    if (invocations == 0 || local_size_x == 0)
    {
        error = "linear dispatch requires non-zero invocations and local size";
        return false;
    }
    if (local_size_x > limits.maxComputeWorkGroupInvocations ||
        local_size_x > limits.maxComputeWorkGroupSize[0])
    {
        error = "shader local_size_x exceeds device workgroup limits";
        return false;
    }
    const std::uint64_t groups = 1u + (invocations - 1u) / local_size_x;
    const std::uint64_t max_x = limits.maxComputeWorkGroupCount[0];
    const std::uint64_t max_y = limits.maxComputeWorkGroupCount[1];
    if (max_x == 0 || max_y == 0)
    {
        error = "device reports a zero linear dispatch group-count limit";
        return false;
    }
    const std::uint64_t groups_x = groups < max_x ? groups : max_x;
    const std::uint64_t groups_y = (groups + groups_x - 1u) / groups_x;
    if (groups_y > max_y)
    {
        error = "linear dispatch exceeds combined X/Y workgroup-count limits";
        return false;
    }
    plan.groups_x = static_cast<uint32_t>(groups_x);
    plan.groups_y = static_cast<uint32_t>(groups_y);
    plan.groups_z = 1;
    return true;
}

}  // namespace tmv::vulkan_dispatch

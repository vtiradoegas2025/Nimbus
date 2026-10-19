/**
 * @file swapchain_policy.hpp
 * @brief Pure selection policy for portable swapchain creation fields.
 */

#pragma once

#include <array>
#include <vector>

#include <vulkan/vulkan.h>

namespace vkcpp::swapchain
{

/** @brief Pick the first conventional composite-alpha mode advertised by a surface. */
[[nodiscard]] constexpr VkCompositeAlphaFlagBitsKHR choose_composite_alpha(
    VkCompositeAlphaFlagsKHR supported)
{
    constexpr std::array<VkCompositeAlphaFlagBitsKHR, 4> preferred{
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    };
    for (const VkCompositeAlphaFlagBitsKHR candidate : preferred)
    {
        if ((supported & candidate) != 0) { return candidate; }
    }
    return static_cast<VkCompositeAlphaFlagBitsKHR>(0);
}

/** @brief Use the requested present mode when advertised, otherwise portable FIFO. */
[[nodiscard]] inline VkPresentModeKHR choose_present_mode(
    const std::vector<VkPresentModeKHR>& supported, VkPresentModeKHR requested)
{
    for (const VkPresentModeKHR mode : supported)
    {
        if (mode == requested) { return mode; }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

}  // namespace vkcpp::swapchain

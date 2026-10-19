/**
 * @file capture_format.hpp
 * @brief Pure swapchain-capture format and pixel conversion policy.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan.h>

namespace vkcpp::capture
{

enum class PixelOrder
{
    unsupported,
    rgba,
    bgra,
};

/** @brief Resolve the byte order for four-byte swapchain formats we can save. */
[[nodiscard]] constexpr PixelOrder pixel_order(VkFormat format)
{
    switch (format)
    {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
            return PixelOrder::rgba;
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
            return PixelOrder::bgra;
        default:
            return PixelOrder::unsupported;
    }
}

/** @brief Whether a surface and selected format support direct image readback. */
[[nodiscard]] constexpr bool direct_readback_supported(VkImageUsageFlags usage, VkFormat format)
{
    return (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0 &&
           pixel_order(format) != PixelOrder::unsupported;
}

/** @brief Convert one packed four-byte swapchain pixel to RGB. */
inline void to_rgb(PixelOrder order, const std::uint8_t* source, std::uint8_t* destination)
{
    if (order == PixelOrder::bgra)
    {
        destination[0] = source[2];
        destination[1] = source[1];
        destination[2] = source[0];
        return;
    }

    destination[0] = source[0];
    destination[1] = source[1];
    destination[2] = source[2];
}

}  // namespace vkcpp::capture

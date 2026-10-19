#pragma once

/**
 * @file vulkan_device_context.hpp
 * @brief Long-lived Vulkan loader/device/compute-queue ownership record.
 */

#include <cstdint>
#include <limits>

#include "compute/vulkan_memory_plan.hpp"

#if defined(__has_include)
#if __has_include(<vulkan/vulkan.h>)
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#define NIMBUS_VULKAN_DEVICE_CONTEXT_AVAILABLE 1
#endif
#endif

#if defined(NIMBUS_VULKAN_DEVICE_CONTEXT_AVAILABLE)
namespace tmv_vulkan
{

struct DeviceContext
{
    void* loader_handle = nullptr;
    std::uint32_t loader_api_version = VK_API_VERSION_1_0;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    std::uint32_t compute_queue_family_index =
        std::numeric_limits<std::uint32_t>::max();

    int selected_device_index = -1;
    VkPhysicalDeviceProperties selected_device_properties{};
    VkPhysicalDeviceFeatures selected_device_features{};
    VkQueueFamilyProperties selected_compute_queue_family{};
    bool selected_supports_portability_subset = false;
    VulkanMemoryPlan memory_plan{};
    // Compatibility projection for the existing allocation/dispatch paths.
    // The explicit plan is authoritative and only DirectShared sets this.
    bool has_unified_memory = false;

    bool available() const noexcept { return device != VK_NULL_HANDLE; }
    const char* api_name() const noexcept { return "vulkan"; }
    const char* device_name() const noexcept
    {
        return available() ? selected_device_properties.deviceName : "";
    }
};

} // namespace tmv_vulkan
#endif

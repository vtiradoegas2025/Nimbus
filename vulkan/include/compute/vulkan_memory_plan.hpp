#pragma once

/**
 * @file vulkan_memory_plan.hpp
 * @brief Pure Vulkan memory-topology classification for backend admission.
 *
 * The classifier deliberately consumes only a synthetic memory-property table
 * and compatible buffer-memory bits plus the physical device type. This keeps
 * resizable BAR on discrete GPUs distinct from integrated shared memory.
 */

#include <cstdint>
#include <limits>
#include <string_view>

#if defined(__has_include)
#if __has_include(<vulkan/vulkan.h>)
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#define NIMBUS_VULKAN_MEMORY_PLAN_AVAILABLE 1
#endif
#endif

#if defined(NIMBUS_VULKAN_MEMORY_PLAN_AVAILABLE)
namespace tmv_vulkan
{

constexpr std::uint32_t kInvalidVulkanMemoryIndex =
    std::numeric_limits<std::uint32_t>::max();

enum class VulkanMemoryTopology
{
    Unavailable,
    DirectShared,
    StagedDiscrete,
    HostVisibleDeviceLocalAperture,
};

enum class VulkanMemoryPolicy
{
    Auto,
    Direct,
    Staged,
};

struct VulkanMemoryPlan
{
    VulkanMemoryTopology topology = VulkanMemoryTopology::Unavailable;
    VulkanMemoryPolicy policy = VulkanMemoryPolicy::Auto;
    std::uint32_t working_memory_type = kInvalidVulkanMemoryIndex;
    std::uint32_t staging_memory_type = kInvalidVulkanMemoryIndex;
    std::uint32_t direct_memory_type = kInvalidVulkanMemoryIndex;
    std::uint32_t primary_device_heap = kInvalidVulkanMemoryIndex;
    std::uint32_t direct_heap = kInvalidVulkanMemoryIndex;
    VkDeviceSize primary_device_heap_bytes = 0;
    VkDeviceSize direct_heap_bytes = 0;
    VkMemoryPropertyFlags working_memory_properties = 0;
    VkMemoryPropertyFlags staging_memory_properties = 0;
    VkMemoryPropertyFlags direct_memory_properties = 0;

    bool available() const noexcept
    {
        return topology != VulkanMemoryTopology::Unavailable;
    }

    bool uses_direct_shared_memory() const noexcept
    {
        return topology == VulkanMemoryTopology::DirectShared &&
            policy != VulkanMemoryPolicy::Staged;
    }
};

struct VulkanNonCoherentRange
{
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
};

/** Expand a mapped subrange to the device's non-coherent atom boundaries. */
inline bool align_vulkan_noncoherent_range(
    VkDeviceSize offset, VkDeviceSize size, VkDeviceSize allocation_size,
    VkDeviceSize atom_size, VulkanNonCoherentRange& out) noexcept
{
    out = {};
    if (size == 0 || atom_size == 0 || offset >= allocation_size ||
        size > allocation_size - offset)
    {
        return false;
    }
    const VkDeviceSize aligned_offset = offset - (offset % atom_size);
    const VkDeviceSize end = offset + size;
    VkDeviceSize aligned_end = allocation_size;
    if (end < allocation_size)
    {
        const VkDeviceSize remainder = end % atom_size;
        const VkDeviceSize padding = remainder == 0 ? 0 : atom_size - remainder;
        if (padding <= allocation_size - end)
        {
            aligned_end = end + padding;
        }
    }
    if (aligned_end <= aligned_offset) return false;
    out.offset = aligned_offset;
    out.size = aligned_end - aligned_offset;
    return true;
}

inline const char* vulkan_memory_policy_name(
    VulkanMemoryPolicy policy) noexcept
{
    switch (policy)
    {
        case VulkanMemoryPolicy::Direct: return "direct";
        case VulkanMemoryPolicy::Staged: return "staged";
        case VulkanMemoryPolicy::Auto:
        default: return "auto";
    }
}

inline bool parse_vulkan_memory_policy(
    std::string_view value, VulkanMemoryPolicy& out) noexcept
{
    if (value.empty() || value == "auto")
    {
        out = VulkanMemoryPolicy::Auto;
        return true;
    }
    if (value == "direct")
    {
        out = VulkanMemoryPolicy::Direct;
        return true;
    }
    if (value == "staged")
    {
        out = VulkanMemoryPolicy::Staged;
        return true;
    }
    return false;
}

inline bool apply_vulkan_memory_policy(
    VulkanMemoryPlan& plan, VulkanMemoryPolicy policy) noexcept
{
    if (!plan.available()) return false;
    if (policy == VulkanMemoryPolicy::Direct &&
        plan.topology != VulkanMemoryTopology::DirectShared)
    {
        return false;
    }
    if (policy == VulkanMemoryPolicy::Staged &&
        (plan.working_memory_type == kInvalidVulkanMemoryIndex ||
         plan.staging_memory_type == kInvalidVulkanMemoryIndex))
    {
        return false;
    }
    plan.policy = policy;
    return true;
}

inline const char* vulkan_memory_topology_name(
    VulkanMemoryTopology topology) noexcept
{
    switch (topology)
    {
        case VulkanMemoryTopology::DirectShared:
            return "direct_shared";
        case VulkanMemoryTopology::StagedDiscrete:
            return "staged_discrete";
        case VulkanMemoryTopology::HostVisibleDeviceLocalAperture:
            return "host_visible_device_local_aperture";
        case VulkanMemoryTopology::Unavailable:
        default:
            return "unavailable";
    }
}

inline bool vulkan_memory_flags_contain(
    VkMemoryPropertyFlags actual, VkMemoryPropertyFlags required) noexcept
{
    return (actual & required) == required;
}

/** Select a compatible type while honoring a topology-ranked preference. */
inline std::uint32_t select_vulkan_memory_type(
    const VkPhysicalDeviceMemoryProperties& memory,
    std::uint32_t compatible_memory_types,
    VkMemoryPropertyFlags required,
    std::uint32_t preferred = kInvalidVulkanMemoryIndex) noexcept
{
    const auto acceptable = [&](std::uint32_t type) {
        return type < memory.memoryTypeCount &&
            (compatible_memory_types & (std::uint32_t{1} << type)) != 0 &&
            vulkan_memory_flags_contain(
                memory.memoryTypes[type].propertyFlags, required);
    };
    if (preferred != kInvalidVulkanMemoryIndex && acceptable(preferred))
    {
        return preferred;
    }
    for (std::uint32_t type = 0; type < memory.memoryTypeCount; ++type)
    {
        if (acceptable(type)) return type;
    }
    return kInvalidVulkanMemoryIndex;
}

/**
 * Classify the memory plan for a representative storage/transfer buffer.
 *
 * CPU-visible VRAM remains an aperture even when resizable BAR exposes the
 * entire discrete heap. CPU readback uses cached host staging in that case.
 */
inline VulkanMemoryPlan classify_vulkan_memory_plan(
    const VkPhysicalDeviceMemoryProperties& memory,
    std::uint32_t compatible_memory_types,
    VkPhysicalDeviceType device_type = VK_PHYSICAL_DEVICE_TYPE_OTHER) noexcept
{
    VulkanMemoryPlan plan;

    for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap)
    {
        const auto& properties = memory.memoryHeaps[heap];
        if ((properties.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 &&
            properties.size >= plan.primary_device_heap_bytes)
        {
            plan.primary_device_heap = heap;
            plan.primary_device_heap_bytes = properties.size;
        }
    }

    constexpr VkMemoryPropertyFlags kDeviceFlags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    constexpr VkMemoryPropertyFlags kHostFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    constexpr VkMemoryPropertyFlags kDirectFlags =
        kDeviceFlags | kHostFlags;

    int staging_score = std::numeric_limits<int>::min();
    int working_score = std::numeric_limits<int>::min();
    int direct_score = std::numeric_limits<int>::min();
    for (std::uint32_t type = 0; type < memory.memoryTypeCount; ++type)
    {
        if ((compatible_memory_types & (std::uint32_t{1} << type)) == 0)
        {
            continue;
        }
        const auto& properties = memory.memoryTypes[type];
        if (properties.heapIndex >= memory.memoryHeapCount)
        {
            continue;
        }
        const auto flags = properties.propertyFlags;
        const auto heap_bytes = memory.memoryHeaps[properties.heapIndex].size;

        const int device_score = (properties.heapIndex == plan.primary_device_heap ? 4 : 0) +
            ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0 ? 2 : 0);
        if (vulkan_memory_flags_contain(flags, kDeviceFlags) && device_score > working_score)
        {
            working_score = device_score;
            plan.working_memory_type = type;
            plan.working_memory_properties = flags;
        }

        if (vulkan_memory_flags_contain(flags, kHostFlags))
        {
            // Readback benefits from cached system memory.  Prefer a pure host
            // heap to consuming a limited device-local aperture for staging.
            int score = 0;
            if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) score += 2;
            if ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) score += 8;
            if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) score += 4;
            if (score > staging_score)
            {
                staging_score = score;
                plan.staging_memory_type = type;
                plan.staging_memory_properties = flags;
            }
        }

        const int host_score = ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0 ? 4 : 0) +
            ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0 ? 2 : 0);
        if (vulkan_memory_flags_contain(flags, kDirectFlags) &&
            (heap_bytes > plan.direct_heap_bytes || (heap_bytes == plan.direct_heap_bytes && host_score > direct_score)))
        {
            direct_score = host_score;
            plan.direct_memory_type = type;
            plan.direct_heap = properties.heapIndex;
            plan.direct_heap_bytes = heap_bytes;
            plan.direct_memory_properties = flags;
        }
    }

    if (device_type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        plan.direct_memory_type != kInvalidVulkanMemoryIndex &&
        plan.direct_heap_bytes >= plan.primary_device_heap_bytes &&
        plan.primary_device_heap_bytes != 0)
    {
        plan.topology = VulkanMemoryTopology::DirectShared;
        plan.working_memory_type = plan.direct_memory_type;
        plan.working_memory_properties = plan.direct_memory_properties;
        return plan;
    }

    if (plan.working_memory_type != kInvalidVulkanMemoryIndex &&
        plan.staging_memory_type != kInvalidVulkanMemoryIndex)
    {
        plan.topology =
            plan.direct_memory_type != kInvalidVulkanMemoryIndex
                ? VulkanMemoryTopology::HostVisibleDeviceLocalAperture
                : VulkanMemoryTopology::StagedDiscrete;
    }
    return plan;
}

} // namespace tmv_vulkan
#endif

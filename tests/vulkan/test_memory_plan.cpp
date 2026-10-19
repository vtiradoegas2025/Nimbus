#include "catch2/catch.hpp"
#include "compute/vulkan_memory_plan.hpp"

#if defined(NIMBUS_VULKAN_MEMORY_PLAN_AVAILABLE)
TEST_CASE("Vulkan memory topology accepts a primary shared heap",
          "[vulkan][backend][memory]")
{
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 1;
    memory.memoryHeaps[0].size = VkDeviceSize{32} << 30;
    memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryTypeCount = 1;
    memory.memoryTypes[0].heapIndex = 0;
    memory.memoryTypes[0].propertyFlags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    const auto plan = tmv_vulkan::classify_vulkan_memory_plan(memory, 0b1);
    REQUIRE(plan.topology == tmv_vulkan::VulkanMemoryTopology::DirectShared);
    REQUIRE(plan.uses_direct_shared_memory());
    REQUIRE(plan.working_memory_type == 0);
    REQUIRE(plan.staging_memory_type == 0);
    REQUIRE(plan.primary_device_heap_bytes == (VkDeviceSize{32} << 30));
}

TEST_CASE("Vulkan memory topology selects conventional staged discrete heaps",
          "[vulkan][backend][memory]")
{
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 2;
    memory.memoryHeaps[0].size = VkDeviceSize{16} << 30;
    memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryHeaps[1].size = VkDeviceSize{64} << 30;
    memory.memoryTypeCount = 2;
    memory.memoryTypes[0].heapIndex = 0;
    memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    memory.memoryTypes[1].heapIndex = 1;
    memory.memoryTypes[1].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    const auto plan = tmv_vulkan::classify_vulkan_memory_plan(memory, 0b11);
    REQUIRE(plan.topology == tmv_vulkan::VulkanMemoryTopology::StagedDiscrete);
    REQUIRE_FALSE(plan.uses_direct_shared_memory());
    REQUIRE(plan.working_memory_type == 0);
    REQUIRE(plan.staging_memory_type == 1);
}

TEST_CASE("Vulkan memory topology does not promote a small BAR aperture",
          "[vulkan][backend][memory]")
{
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 3;
    memory.memoryHeaps[0].size = VkDeviceSize{16} << 30;
    memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryHeaps[1].size = VkDeviceSize{256} << 20;
    memory.memoryHeaps[1].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryHeaps[2].size = VkDeviceSize{64} << 30;
    memory.memoryTypeCount = 3;
    memory.memoryTypes[0].heapIndex = 0;
    memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    memory.memoryTypes[1].heapIndex = 1;
    memory.memoryTypes[1].propertyFlags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    memory.memoryTypes[2].heapIndex = 2;
    memory.memoryTypes[2].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    const auto plan = tmv_vulkan::classify_vulkan_memory_plan(memory, 0b111);
    REQUIRE(
        plan.topology ==
        tmv_vulkan::VulkanMemoryTopology::HostVisibleDeviceLocalAperture);
    REQUIRE_FALSE(plan.uses_direct_shared_memory());
    REQUIRE(plan.working_memory_type == 0);
    REQUIRE(plan.staging_memory_type == 2);
    REQUIRE(plan.direct_memory_type == 1);
    REQUIRE(plan.direct_heap_bytes == (VkDeviceSize{256} << 20));
}

TEST_CASE("Vulkan memory topology respects buffer compatibility bits",
          "[vulkan][backend][memory]")
{
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 2;
    memory.memoryHeaps[0].size = VkDeviceSize{8} << 30;
    memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryHeaps[1].size = VkDeviceSize{32} << 30;
    memory.memoryTypeCount = 3;
    memory.memoryTypes[0].heapIndex = 0;
    memory.memoryTypes[0].propertyFlags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    memory.memoryTypes[1].heapIndex = 0;
    memory.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    memory.memoryTypes[2].heapIndex = 1;
    memory.memoryTypes[2].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // The otherwise-direct type 0 is incompatible with this buffer.
    const auto plan = tmv_vulkan::classify_vulkan_memory_plan(memory, 0b110);
    REQUIRE(plan.topology == tmv_vulkan::VulkanMemoryTopology::StagedDiscrete);
    REQUIRE(plan.working_memory_type == 1);
    REQUIRE(plan.staging_memory_type == 2);
}

TEST_CASE("Resizable BAR retains device-local working storage and cached host staging",
          "[vulkan][backend][memory]")
{
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 2;
    memory.memoryHeaps[0] = {VkDeviceSize{32} << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    memory.memoryHeaps[1] = {VkDeviceSize{128} << 30, 0};
    memory.memoryTypeCount = 3;
    memory.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    memory.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 1};
    memory.memoryTypes[2] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0};
    auto plan = tmv_vulkan::classify_vulkan_memory_plan(memory, 0b111, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    REQUIRE(plan.topology == tmv_vulkan::VulkanMemoryTopology::HostVisibleDeviceLocalAperture);
    REQUIRE_FALSE(plan.uses_direct_shared_memory());
    REQUIRE(plan.working_memory_type == 0);
    REQUIRE(plan.staging_memory_type == 1);
    REQUIRE(plan.direct_memory_type == 2);
    REQUIRE_FALSE(tmv_vulkan::apply_vulkan_memory_policy(plan, tmv_vulkan::VulkanMemoryPolicy::Direct));
    REQUIRE(tmv_vulkan::apply_vulkan_memory_policy(plan, tmv_vulkan::VulkanMemoryPolicy::Staged));
    REQUIRE(plan.staging_memory_type == 1);
    const auto incompatible = tmv_vulkan::classify_vulkan_memory_plan(memory, 0b101, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    REQUIRE(incompatible.staging_memory_type == 2);
    REQUIRE_FALSE(incompatible.uses_direct_shared_memory());
}

TEST_CASE("Integrated shared heaps prefer cached compatible mappings",
          "[vulkan][backend][memory]")
{
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 1;
    memory.memoryHeaps[0] = {VkDeviceSize{32} << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    memory.memoryTypeCount = 2;
    constexpr auto shared = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    memory.memoryTypes[0] = {shared | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0};
    memory.memoryTypes[1] = {shared | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0};
    const auto plan = tmv_vulkan::classify_vulkan_memory_plan(memory, 0b11, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    REQUIRE(plan.uses_direct_shared_memory());
    REQUIRE(plan.direct_memory_type == 0);
    REQUIRE(plan.staging_memory_type == 0);
    const auto incompatible = tmv_vulkan::classify_vulkan_memory_plan(memory, 0b10, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    REQUIRE(incompatible.direct_memory_type == 1);
}

TEST_CASE("Vulkan memory selection honors the ranked plan and compatibility",
          "[vulkan][backend][memory]")
{
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 2;
    memory.memoryHeaps[0].size = VkDeviceSize{256} << 20;
    memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryHeaps[1].size = VkDeviceSize{16} << 30;
    memory.memoryHeaps[1].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryTypeCount = 3;
    memory.memoryTypes[0].heapIndex = 0;
    memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    memory.memoryTypes[1].heapIndex = 1;
    memory.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    memory.memoryTypes[2].heapIndex = 1;
    memory.memoryTypes[2].propertyFlags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // Type zero is the first legal device-local match, but the topology plan
    // ranked type one on the primary heap.
    REQUIRE(
        tmv_vulkan::select_vulkan_memory_type(
            memory, 0b111, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1) == 1);
    // A preferred type that this buffer cannot use is never forced.
    REQUIRE(
        tmv_vulkan::select_vulkan_memory_type(
            memory, 0b101, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1) == 0);
    REQUIRE(
        tmv_vulkan::select_vulkan_memory_type(
            memory, 0b011, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 2) ==
        tmv_vulkan::kInvalidVulkanMemoryIndex);
}

TEST_CASE("Vulkan memory topology accepts non-coherent host storage",
          "[vulkan][backend][memory]")
{
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 1;
    memory.memoryHeaps[0].size = VkDeviceSize{8} << 30;
    memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryTypeCount = 2;
    memory.memoryTypes[0].heapIndex = 0;
    memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    memory.memoryTypes[1].heapIndex = 0;
    memory.memoryTypes[1].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

    const auto plan = tmv_vulkan::classify_vulkan_memory_plan(memory, 0b11);
    REQUIRE(plan.topology == tmv_vulkan::VulkanMemoryTopology::StagedDiscrete);
    REQUIRE(plan.available());
    REQUIRE_FALSE(plan.uses_direct_shared_memory());
    REQUIRE(
        (plan.staging_memory_properties &
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0);
}

TEST_CASE("Vulkan memory topology fails closed without host-visible storage",
          "[vulkan][backend][memory]")
{
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 1;
    memory.memoryHeaps[0].size = VkDeviceSize{8} << 30;
    memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryTypeCount = 1;
    memory.memoryTypes[0].heapIndex = 0;
    memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    const auto plan = tmv_vulkan::classify_vulkan_memory_plan(memory, 0b1);
    REQUIRE(plan.topology == tmv_vulkan::VulkanMemoryTopology::Unavailable);
    REQUIRE_FALSE(plan.available());
}

TEST_CASE("Vulkan memory policy forces staged and rejects unavailable direct",
          "[vulkan][backend][memory]")
{
    tmv_vulkan::VulkanMemoryPolicy parsed{};
    REQUIRE(tmv_vulkan::parse_vulkan_memory_policy("", parsed));
    REQUIRE(parsed == tmv_vulkan::VulkanMemoryPolicy::Auto);
    REQUIRE(tmv_vulkan::parse_vulkan_memory_policy("auto", parsed));
    REQUIRE(parsed == tmv_vulkan::VulkanMemoryPolicy::Auto);
    REQUIRE(tmv_vulkan::parse_vulkan_memory_policy("direct", parsed));
    REQUIRE(parsed == tmv_vulkan::VulkanMemoryPolicy::Direct);
    REQUIRE(tmv_vulkan::parse_vulkan_memory_policy("staged", parsed));
    REQUIRE(parsed == tmv_vulkan::VulkanMemoryPolicy::Staged);
    REQUIRE_FALSE(tmv_vulkan::parse_vulkan_memory_policy("shared", parsed));

    tmv_vulkan::VulkanMemoryPlan shared;
    shared.topology = tmv_vulkan::VulkanMemoryTopology::DirectShared;
    shared.working_memory_type = 0;
    shared.staging_memory_type = 0;
    shared.direct_memory_type = 0;

    REQUIRE(tmv_vulkan::apply_vulkan_memory_policy(
        shared, tmv_vulkan::VulkanMemoryPolicy::Staged));
    REQUIRE_FALSE(shared.uses_direct_shared_memory());
    REQUIRE(shared.policy == tmv_vulkan::VulkanMemoryPolicy::Staged);
    REQUIRE(tmv_vulkan::apply_vulkan_memory_policy(
        shared, tmv_vulkan::VulkanMemoryPolicy::Direct));
    REQUIRE(shared.uses_direct_shared_memory());

    tmv_vulkan::VulkanMemoryPlan aperture;
    aperture.topology =
        tmv_vulkan::VulkanMemoryTopology::HostVisibleDeviceLocalAperture;
    aperture.working_memory_type = 0;
    aperture.staging_memory_type = 1;
    aperture.direct_memory_type = 2;
    REQUIRE_FALSE(tmv_vulkan::apply_vulkan_memory_policy(
        aperture, tmv_vulkan::VulkanMemoryPolicy::Direct));
    REQUIRE(tmv_vulkan::apply_vulkan_memory_policy(
        aperture, tmv_vulkan::VulkanMemoryPolicy::Staged));
}

TEST_CASE("Vulkan non-coherent ranges align and fail closed at boundaries",
          "[vulkan][backend][memory]")
{
    tmv_vulkan::VulkanNonCoherentRange range;
    REQUIRE(tmv_vulkan::align_vulkan_noncoherent_range(
        130, 100, 1024, 256, range));
    REQUIRE(range.offset == 0);
    REQUIRE(range.size == 256);

    REQUIRE(tmv_vulkan::align_vulkan_noncoherent_range(
        900, 124, 1024, 256, range));
    REQUIRE(range.offset == 768);
    REQUIRE(range.size == 256);

    REQUIRE(tmv_vulkan::align_vulkan_noncoherent_range(
        1, 1, 1024, 2048, range));
    REQUIRE(range.offset == 0);
    REQUIRE(range.size == 1024);

    REQUIRE_FALSE(tmv_vulkan::align_vulkan_noncoherent_range(
        900, 125, 1024, 256, range));
    REQUIRE_FALSE(tmv_vulkan::align_vulkan_noncoherent_range(
        0, 1, 1024, 0, range));
    REQUIRE_FALSE(tmv_vulkan::align_vulkan_noncoherent_range(
        1024, 1, 1024, 256, range));
}

#endif
